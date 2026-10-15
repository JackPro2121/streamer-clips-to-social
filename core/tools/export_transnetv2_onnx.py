#!/usr/bin/env python3
"""
Export TransNetV2 to ONNX and prove the export is faithful.

Why: the render job must not install torch. The ML tier already proves the
``transnetv2-pytorch`` model works on CI (model_smoke), so this exports the raw
network once, verifies it against the PyTorch outputs, and the resulting ONNX is
hosted on the immutable release and fetched by ``core/tools/fetch_models.py``.

Export spec (what ``core/shot_detection.TransNetV2Detector`` consumes):
    input  ``frames``      uint8  [1, 100, 27, 48, 3]
    output ``single_frame`` float [1, 100, 1]   (sigmoid applied)
    output ``many_hot``     float [1, 100, 1]   (sigmoid applied)

Parity checks, both required:
1. Window level: PyTorch wrapper vs onnxruntime on random uint8 windows,
   max absolute difference must be < 1e-4.
2. Video level: the full padding/window/splice pipeline run through the ONNX
   session must reproduce ``TransNetV2.predict_frames`` on real frames from a
   video, and the resulting scene boundaries must match.

    python core/tools/export_transnetv2_onnx.py --video clip.mp4 --output transnetv2.onnx
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

WINDOW = 100
STEP = 50
HALF = 25
INPUT_SIZE = (27, 48, 3)


class ExportWrapper:
    """Kept as a plain function; torch is only imported inside main()."""


def build_wrapper(model):
    import torch

    class Wrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, frames):
            single, heads = self.inner.forward(frames)
            return torch.sigmoid(single), torch.sigmoid(heads["many_hot"])

    return Wrapper(model)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pad_frames(frames):
    """The reference padding: 25 copies of the first frame, then 25+step-r frames."""
    import numpy as np

    if len(frames) == 0:
        raise ValueError("no frames")
    remainder = len(frames) % STEP
    end_pad = HALF + STEP - (remainder if remainder else STEP)
    head = np.repeat(frames[:1], HALF, axis=0)
    tail = np.repeat(frames[-1:], end_pad, axis=0)
    return np.concatenate([head, frames, tail], axis=0)


def onnx_predict_frames(session, frames) -> Tuple[Any, Any]:
    """Torch-free replication of TransNetV2.predict_frames for parity testing."""
    import numpy as np

    padded = pad_frames(np.asarray(frames, dtype=np.uint8))
    singles: List[Any] = []
    many: List[Any] = []
    input_name = session.get_inputs()[0].name
    for start in range(0, len(padded) - WINDOW + 1, STEP):
        batch = padded[start:start + WINDOW][None, ...]
        single, many_hot = session.run(None, {input_name: batch})
        singles.append(single[0, HALF:HALF + STEP, 0])
        many.append(many_hot[0, HALF:HALF + STEP, 0])
    single = np.concatenate(singles, axis=0)[: len(frames)]
    many_hot = np.concatenate(many, axis=0)[: len(frames)]
    return single, many_hot


def predictions_to_scenes(predictions, threshold: float = 0.5):
    import numpy as np

    bits = (np.asarray(predictions) > threshold).astype(np.uint8)
    scenes: List[List[int]] = []
    previous, start = 0, 0
    last = 0
    for index, bit in enumerate(bits):
        if previous == 1 and bit == 0:
            start = index
        if previous == 0 and bit == 1 and index != 0:
            scenes.append([start, index])
        previous = bit
        last = index
    if previous == 0:
        scenes.append([start, last])
    if not scenes:
        return [[0, len(bits) - 1]]
    return scenes


def decode_video_frames(video: Path, max_seconds: float) -> Any:
    """Decode at 25fps and resize to 27x48 RGB uint8, the way the model sees video."""
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(str(video))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(max(2, fps * max_seconds))
        frames: List[Any] = []
        while len(frames) < total:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            resized = cv2.resize(frame, (INPUT_SIZE[1], INPUT_SIZE[0]), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"could not decode frames from {video}")
    return np.stack(frames).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("transnetv2.onnx"))
    parser.add_argument("--manifest", type=Path, default=Path("transnetv2_onnx_manifest.json"))
    parser.add_argument("--video", type=Path, default=None, help="real clip for video-level parity")
    parser.add_argument("--max-seconds", type=float, default=15.0)
    args = parser.parse_args()

    import numpy as np
    import onnxruntime
    import torch
    from transnetv2_pytorch import TransNetV2

    model = TransNetV2(device="cpu")
    model.eval()
    wrapper = build_wrapper(model)
    wrapper.eval()

    dummy = torch.zeros(1, WINDOW, *INPUT_SIZE, dtype=torch.uint8)
    torch.onnx.export(
        wrapper,
        dummy,
        str(args.output),
        input_names=["frames"],
        output_names=["single_frame", "many_hot"],
        opset_version=17,
    )
    print(f"[+] exported {args.output} ({args.output.stat().st_size:,} bytes)")

    # torch's exporter can spill the weights into an external side file. The
    # release must carry exactly one self-contained artifact, so consolidate.
    side_file = args.output.with_name(args.output.name + ".data")
    if side_file.exists():
        import onnx

        proto = onnx.load(str(args.output))
        onnx.save_model(proto, str(args.output), save_as_external_data=False)
        side_file.unlink()
        print(f"[+] consolidated external data into {args.output} ({args.output.stat().st_size:,} bytes)")
    if args.output.with_name(args.output.name + ".data").exists():
        raise SystemExit("external data file still present after consolidation")
    if args.output.stat().st_size < 1_000_000:
        raise SystemExit(
            f"export is suspiciously small ({args.output.stat().st_size} bytes); "
            "weights likely missing"
        )

    session = onnxruntime.InferenceSession(
        str(args.output), providers=["CPUExecutionProvider"]
    )
    input_meta = session.get_inputs()[0]
    print(f"[*] input {input_meta.name} {input_meta.shape} {input_meta.type}")

    # --- parity 1: window level ------------------------------------------------
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(2):
        batch = rng.integers(0, 256, size=(1, WINDOW, *INPUT_SIZE), dtype=np.uint8)
        with torch.no_grad():
            reference_single, reference_many = wrapper(torch.from_numpy(batch))
        onnx_single, onnx_many = session.run(None, {input_meta.name: batch})
        worst = max(
            worst,
            float(np.abs(reference_single.numpy() - onnx_single).max()),
            float(np.abs(reference_many.numpy() - onnx_many).max()),
        )
    print(f"[*] window parity max abs diff: {worst:.2e}")
    if worst > 1e-4:
        raise SystemExit(f"ONNX export is not faithful: max abs diff {worst}")

    manifest: Dict[str, Any] = {
        "file": args.output.name,
        "bytes": args.output.stat().st_size,
        "sha256": sha256_of(args.output),
        "opset": 17,
        "input": {"name": input_meta.name, "shape": input_meta.shape, "type": input_meta.type},
        "window_parity_max_abs_diff": worst,
    }

    # --- parity 2: video level -------------------------------------------------
    if args.video is not None:
        frames = decode_video_frames(args.video, args.max_seconds)
        with torch.no_grad():
            reference_single, _reference_many = model.predict_frames(
                torch.from_numpy(frames), quiet=True
            )
        reference_single = reference_single.numpy()
        onnx_single, _onnx_many = onnx_predict_frames(session, frames)
        video_worst = float(np.abs(reference_single - onnx_single).max())
        reference_scenes = predictions_to_scenes(reference_single)
        onnx_scenes = predictions_to_scenes(onnx_single)
        print(f"[*] video parity max abs diff: {video_worst:.2e}")
        print(f"[*] torch scenes: {reference_scenes}")
        print(f"[*] onnx  scenes: {onnx_scenes}")
        if video_worst > 1e-4:
            raise SystemExit(f"video-level ONNX drift too large: {video_worst}")
        if reference_scenes != onnx_scenes:
            raise SystemExit("scene lists disagree between torch and ONNX")
        manifest["video_parity"] = {
            "video": args.video.name,
            "frames": int(len(frames)),
            "max_abs_diff": video_worst,
            "scenes": onnx_scenes,
        }

    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
