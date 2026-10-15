"""
Shot boundary detection: TransNetV2 (primary) with PySceneDetect (fallback).

`scene_classifier.detect_clip_shots` relies on PySceneDetect's AdaptiveDetector,
which is content-adaptive and therefore *misses hard cuts on low-motion
content* -- precisely the talking-head podcast case this pipeline exists for. A
cut between two similar shots produces almost no pixel delta, so an adaptive
content threshold never fires.

TransNetV2 is a small convolutional network trained specifically for shot
boundaries, so it does not have that blind spot. It is also a genuinely cheap
model (~few MB, 48x27 input), unlike a video LLM.

**The weights are not downloaded.** There is no verified canonical URL for a
TransNetV2 ONNX export, and this project does not invent URLs. Place the file at
`core/models/transnetv2.onnx` (exported from the soCzech/TransNetV2 reference
implementation) and the ONNX path engages automatically. Until then every call
falls through to PySceneDetect and behaviour is byte-identical to before.

The post-processing -- clamping to the requested window, merging sub-minimum
shots -- lives here and is shared by both detectors, so the fallback cannot
drift away from the primary over time.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.config import (
    SHOT_DETECTOR,
    TRANSNET_MODEL_PATH,
    TRANSNET_MIN_CUT_CONFIDENCE,
)

ShotList = List[Tuple[float, float]]


@dataclass
class ShotBoundaryResult:
    shots: ShotList
    source: str
    transition_count: int = 0
    mean_confidence: float = 0.0
    notes: List[str] = field(default_factory=list)


def normalise_shots(
    raw: Sequence[Tuple[float, float]],
    start_sec: float,
    end_sec: float,
    min_shot_duration: float = 0.4,
) -> ShotList:
    """
    Clamp boundaries to the window and merge shots shorter than the minimum.

    Shared by every detector so the fallback cannot diverge from the primary.
    A shot shorter than `min_shot_duration` is folded into its predecessor
    rather than dropped, because dropping it would punch a hole in the timeline
    and leave the renderer with a gap.
    """
    duration = max(0.1, end_sec - start_sec)
    single = [(round(start_sec, 2), round(end_sec, 2))]

    kept: ShotList = []
    for shot_start, shot_end in raw:
        clamped_start = max(start_sec, float(shot_start))
        clamped_end = min(end_sec, float(shot_end))
        if clamped_end - clamped_start > 0.1:
            kept.append((round(clamped_start, 2), round(clamped_end, 2)))
    if not kept:
        return single

    if kept[0][0] > start_sec:
        kept[0] = (round(start_sec, 2), kept[0][1])
    if kept[-1][1] < end_sec:
        kept[-1] = (kept[-1][0], round(end_sec, 2))

    merged: ShotList = []
    for shot_start, shot_end in kept:
        if not merged:
            merged.append((shot_start, shot_end))
        elif (shot_end - shot_start) < min_shot_duration:
            previous_start, _ = merged[-1]
            merged[-1] = (previous_start, shot_end)
        else:
            merged.append((shot_start, shot_end))

    # Merging can still leave a short trailing shot, and clamping guarantees the
    # total span, so the result is a strict partition of the window.
    if not merged:
        return single
    if len(merged) > 1 and (merged[-1][1] - merged[-1][0]) < min_shot_duration:
        head_start, _ = merged[-2]
        merged = merged[:-2] + [(head_start, merged[-1][1])]
    covered = sum(shot_end - shot_start for shot_start, shot_end in merged)
    if covered <= 0 or abs(covered - duration) > 0.5:
        return single
    return merged


def _detect_with_pyscenedetect(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    min_shot_duration: float,
) -> Tuple[ShotList, str]:
    """PySceneDetect AdaptiveDetector. The pre-existing behaviour, unchanged."""
    default_shot = [(round(start_sec, 2), round(end_sec, 2))]
    try:
        from scenedetect import (
            AdaptiveDetector,
            FrameTimecode,
            SceneManager,
            open_video,
        )
        video = open_video(str(video_path))
        fps = video.frame_rate
        if not fps or fps <= 0:
            return default_shot, "scenedetect"
        start_tc = FrameTimecode(timecode=start_sec, fps=fps)
        dur_tc = FrameTimecode(timecode=end_sec - start_sec, fps=fps)
        video.seek(start_tc)
        manager = SceneManager()
        manager.add_detector(AdaptiveDetector(
            adaptive_threshold=2.5,
            min_scene_len=max(2, int(fps * min_shot_duration)),
        ))
        manager.detect_scenes(video, frame_skip=2, duration=dur_tc)
        scenes = manager.get_scene_list()
        if not scenes:
            return default_shot, "scenedetect"
        raw = [
            (max(start_sec, float(scene[0].seconds)), min(end_sec, float(scene[1].seconds)))
            for scene in scenes
        ]
        return normalise_shots(raw, start_sec, end_sec, min_shot_duration), "scenedetect"
    except Exception as error:
        print(f"[-] Scene detection fallback (error: {error}). Using monolithic segment.")
        return [(round(start_sec, 2), round(end_sec, 2))], "single"


class TransNetV2Detector:
    """
    TransNetV2 over ONNX Runtime, with an OpenCV DNN fallback for runtimes
    without onnxruntime installed.

    The ONNX graph is the project's own export of the proven
    ``transnetv2-pytorch`` model (core/tools/export_transnetv2_onnx.py), fetched from
    the immutable golden-masters release by core/tools/fetch_models.py:

        input  ``frames``       uint8 [1, 100, 27, 48, 3]
        output ``single_frame`` float [1, 100, 1]  (sigmoid applied)

    The windowing replicates the reference implementation exactly -- pad 25
    frames at the head and ``25 + step - (n % step)`` at the tail, slide a
    100-frame window by 50, keep each window's middle 50 predictions -- because
    that is what the export's parity proof measured against. Averaging
    overlapping windows instead (the previous runner) does not match the model's
    training-time context and was never exercised with real weights.
    """

    INPUT_WIDTH = 48
    INPUT_HEIGHT = 27
    WINDOW = 100
    STEP = 50
    HALF = 25
    EXPECTED_INPUT_SHAPE = [1, 100, 27, 48, 3]
    # Decode cap: 3000 frames is ~100s at 30fps, comfortably above the 30-55s
    # clip contract; longer windows stride uniformly and report the rate used.
    MAX_FRAMES = 3000

    def __init__(self, session: Any, fps: float) -> None:
        self._session = session
        self._fps = max(1.0, float(fps))

    @classmethod
    def try_create(cls, fps: float) -> Optional["TransNetV2Detector"]:
        model_path = Path(TRANSNET_MODEL_PATH)
        if not model_path.exists():
            return None
        try:
            import onnxruntime  # type: ignore
            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = 2
            session = onnxruntime.InferenceSession(
                str(model_path), sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            inputs = session.get_inputs()
            shape = [int(dim) if isinstance(dim, int) else dim for dim in inputs[0].shape]
            if shape != cls.EXPECTED_INPUT_SHAPE or "uint8" not in inputs[0].type:
                print(
                    f"[-] TransNetV2 model has an unexpected signature "
                    f"({inputs[0].name} {inputs[0].shape} {inputs[0].type}); "
                    f"expected frames {cls.EXPECTED_INPUT_SHAPE} tensor(uint8). "
                    f"Using PySceneDetect."
                )
                return None
            return cls(session, fps)
        except Exception as error:
            print(f"[-] TransNetV2 via onnxruntime unavailable ({error}); trying OpenCV DNN.")
        try:
            import cv2
            net = cv2.dnn.readNetFromONNX(str(model_path))
            return cls(net, fps)
        except Exception as error:
            print(f"[-] TransNetV2 model unusable ({error}); using PySceneDetect.")
            return None

    def _infer(self, batch: Any) -> Any:
        """Single-frame probabilities [1, T, 1] for one window."""
        try:
            output_names = [output.name for output in self._session.get_outputs()]
            if "single_frame" in output_names:
                return self._session.run(
                    ["single_frame"],
                    {self._session.get_inputs()[0].name: batch},
                )[0]
            return self._session.run(
                None, {self._session.get_inputs()[0].name: batch}
            )[0]
        except AttributeError:
            # OpenCV DNN: no get_inputs(), and run() takes positional args.
            self._session.setInput(batch)
            return self._session.forward()

    @staticmethod
    def _prepare(frame: Any) -> Any:
        import cv2
        import numpy as np

        resized = cv2.resize(
            frame, (TransNetV2Detector.INPUT_WIDTH, TransNetV2Detector.INPUT_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
        # The exported graph normalises internally; feed uint8 RGB exactly as
        # core/tools/export_transnetv2_onnx.py decoded during the parity proof.
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.uint8)

    def _transition_strengths(self, frames: Sequence[Any]) -> List[float]:
        """
        Per-frame probability that a shot boundary sits on that frame.

        Reference windowing (see class docstring): head pad 25, tail pad
        ``25 + step - (n % step)``, window 100, stride 50, keep the middle 50
        of each window's predictions, then truncate to the frame count.
        """
        import numpy as np

        total = len(frames)
        if total < 2:
            return [0.0] * total
        window, step, half = self.WINDOW, self.STEP, self.HALF
        remainder = total % step
        end_pad = half + step - (remainder if remainder else step)
        padded: List[Any] = (
            [frames[0]] * half + list(frames) + [frames[-1]] * end_pad
        )

        strengths: List[float] = []
        for start in range(0, len(padded) - window + 1, step):
            chunk = padded[start:start + window]
            batch = np.stack([self._prepare(frame) for frame in chunk])[None, ...]
            probabilities = np.asarray(self._infer(batch), dtype=np.float64)
            if probabilities.ndim == 3:
                probabilities = probabilities[0]
            values = probabilities[:, 0] if probabilities.ndim == 2 else probabilities
            strengths.extend(float(value) for value in values[half:half + step])
        return strengths[:total]

    def shots(
        self,
        frames: Sequence[Any],
        start_sec: float,
        end_sec: float,
        min_shot_duration: float,
    ) -> Tuple[ShotList, int, float]:
        strengths = self._transition_strengths(frames)
        threshold = max(0.0, min(1.0, TRANSNET_MIN_CUT_CONFIDENCE))
        boundaries: List[float] = []
        confidences: List[float] = []
        for index, strength in enumerate(strengths):
            # Frame 0 can never be a cut: the window starts there.
            if index == 0 or strength < threshold:
                continue
            boundaries.append(start_sec + (index - 1) / self._fps)
            confidences.append(strength)

        raw: List[Tuple[float, float]] = []
        cursor = start_sec
        for boundary in boundaries:
            if boundary - cursor < min_shot_duration:
                continue
            raw.append((cursor, boundary))
            cursor = boundary
        raw.append((cursor, end_sec))
        mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        return (
            normalise_shots(raw, start_sec, end_sec, min_shot_duration),
            len(raw) - 1,
            mean_confidence,
        )


_detector_lock = threading.Lock()
_reported: Dict[str, bool] = {}


def detect_shot_boundaries(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    min_shot_duration: float = 0.4,
    sample_fps: float = 4.0,
) -> ShotBoundaryResult:
    """
    Resolve shot boundaries, preferring TransNetV2 and falling back cleanly.

    The ONNX path is only attempted when explicitly requested or when the
    weights are present, so a machine without them behaves exactly as it did
    before this module existed.
    """
    duration = max(0.1, end_sec - start_sec)
    requested = (SHOT_DETECTOR or "auto").strip().lower()

    if requested in ("auto", "transnet"):
        result = _try_transnet(
            video_path, start_sec, end_sec, min_shot_duration, sample_fps
        )
        if result is not None:
            return result
        if requested == "transnet" and not _reported.get("transnet_missing"):
            print("[-] SHOT_DETECTOR=transnet but no usable model; using PySceneDetect.")
            _reported["transnet_missing"] = True

    shots, source = _detect_with_pyscenedetect(
        video_path, start_sec, end_sec, min_shot_duration
    )
    return ShotBoundaryResult(
        shots=shots,
        source=source,
        transition_count=max(0, len(shots) - 1),
        notes=[f"duration={duration:.2f}s"],
    )


def _try_transnet(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    min_shot_duration: float,
    sample_fps: float,
) -> Optional[ShotBoundaryResult]:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            return None
        detector = TransNetV2Detector.try_create(fps)
        if detector is None:
            return None

        start_frame = max(0, int(start_sec * fps))
        end_frame = max(start_frame + 2, int(end_sec * fps))
        total = max(2, end_frame - start_frame)
        # TransNetV2's reference windowing expects consecutive frames, so
        # `sample_fps` is ignored here: decode every frame up to the cap, and
        # only stride when a window is longer than the cap. The previous runner
        # advanced the loop counter without skipping reads, so it fed the first
        # N consecutive frames and then labelled them as 4fps samples -- the
        # boundary timestamps would have been wrong had weights ever been
        # present.
        stride = 1
        if total > TransNetV2Detector.MAX_FRAMES:
            stride = total // TransNetV2Detector.MAX_FRAMES + 1
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames: List[Any] = []
        consumed = 0
        while consumed < total and len(frames) < TransNetV2Detector.MAX_FRAMES:
            success, frame = capture.read()
            if not success or frame is None:
                break
            if consumed % stride == 0:
                frames.append(frame)
            consumed += 1
    finally:
        capture.release()

    if len(frames) < 2:
        return None

    effective_fps = fps / stride
    if not _reported.get("transnet_ok"):
        print(f"[+] TransNetV2 shot detection active ({len(frames)} frames at {effective_fps:.2f}fps).")
        _reported["transnet_ok"] = True

    detector._fps = effective_fps
    shots, cuts, confidence = detector.shots(
        frames, start_sec, end_sec, min_shot_duration
    )
    return ShotBoundaryResult(
        shots=shots,
        source="transnetv2",
        transition_count=cuts,
        mean_confidence=round(confidence, 4),
        notes=[
            f"frames={len(frames)}",
            f"effective_fps={effective_fps:.2f}",
            f"stride={stride}",
            f"mean_confidence={confidence:.3f}",
        ],
    )


def reset_shot_detector_cache() -> None:
    with _detector_lock:
        _reported.clear()
