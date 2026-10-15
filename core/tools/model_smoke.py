#!/usr/bin/env python3
"""
Smoke-test the deep-verification model tier on a real clip.

Proves, on CI, that the pretrained models actually install, load, and produce
numbers -- not merely that pip resolved:

* TransNetV2 (weights bundled in its wheel) -> shot boundaries
* SyncNet + S3FD (official Oxford VGG checkpoints) -> per-face AV offset and
  confidence, which is both the lip-sync measurement and the active-speaker
  signal

    python core/tools/model_smoke.py --video clip.mp4 --max-seconds 15

Exit code 0 only when every stage that ran reported usable numbers.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

# ``parents[1]`` is the clips-core root in both layouts: ``core/`` when the
# engine is mounted as a submodule, the repo root when it is checked out alone.
_ENGINE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = _ENGINE_ROOT / "models"


def _trim(video: Path, seconds: float) -> Path:
    handle = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    handle.close()
    target = Path(handle.name)
    command = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(video), "-t", f"{seconds:.2f}",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
        "-c:a", "aac", str(target),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    if result.returncode != 0 or not target.exists() or target.stat().st_size == 0:
        raise RuntimeError(f"trim failed: {result.stderr[:300]}")
    return target


def run_transnetv2(video: Path) -> Dict[str, Any]:
    from transnetv2_pytorch import TransNetV2

    model = TransNetV2(device="cpu")
    scenes = model.detect_scenes(str(video), threshold=0.5) or []
    return {
        "ok": len(scenes) >= 1,
        "scene_count": len(scenes),
        "first_scenes": [
            {k: v for k, v in scene.items() if k in ("start_time", "end_time", "start_frame", "end_frame")}
            for scene in scenes[:5]
        ],
    }


def run_syncnet(video: Path, models_dir: Path) -> Dict[str, Any]:
    from syncnet_python.syncnet_pipeline import PipelineConfig, SyncNetPipeline

    config = PipelineConfig(
        s3fd_weights=str(models_dir / "sfd_face.pth"),
        syncnet_weights=str(models_dir / "syncnet_v2.model"),
        frame_rate=25,
        batch_size=10,
        min_track=10,
    )
    pipeline = SyncNetPipeline(config, device="cpu")
    offsets, confidences, distances, max_confidence, min_distance, _faces, has_face = (
        pipeline.inference(str(video))
    )
    return {
        "ok": bool(has_face) and bool(offsets),
        "has_face": bool(has_face),
        "track_count": len(offsets),
        "offsets": offsets[:8],
        "confidences": [round(float(value), 4) for value in confidences[:8]],
        "distances": [round(float(value), 4) for value in distances[:8]],
        "max_confidence": round(float(max_confidence), 4),
        "min_distance": round(float(min_distance), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--max-seconds", type=float, default=15.0)
    parser.add_argument("--skip", nargs="*", default=[], help="stage names to skip")
    args = parser.parse_args()

    if not args.video.exists():
        print(f"video not found: {args.video}", file=sys.stderr)
        return 2

    working = _trim(args.video, args.max_seconds)
    report: Dict[str, Any] = {"video": str(args.video), "seconds": args.max_seconds}
    failures: List[str] = []

    for name, runner in (
        ("transnetv2", lambda: run_transnetv2(working)),
        ("syncnet", lambda: run_syncnet(working, args.models_dir)),
    ):
        if name in args.skip:
            report[name] = {"ok": None, "skipped": True}
            continue
        try:
            report[name] = runner()
        except Exception as error:  # noqa: BLE001 - the smoke test's job is to report
            report[name] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        if report[name].get("ok") is False:
            failures.append(name)

    print(json.dumps(report, indent=2))
    working.unlink(missing_ok=True)

    if failures:
        print(f"\nSMOKE FAILED: {', '.join(failures)}", file=sys.stderr)
        return 1
    print("\nSMOKE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
