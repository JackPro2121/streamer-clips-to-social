"""Face boxes on the **rendered** output, for caption-placement checking.

Why
---
Captions render across the subject's face. Measured on
``clip_1_THE_REAL_FEAR_OF_GOING_BROKE.mp4`` at t=1.2s, 14.0s and 31.0s, the
caption baseline sits at y ~ 490 of 1920 -- 25.5% of frame height, directly over
the eyebrows and eyes -- while ``SAFE_ZONE_TOP`` is 240px (12.5%).

``editorial_qa`` does not catch this because it compares ``shot.caption_rect``
against ``shot.protected_regions``, and both are written by the same code
minutes apart. It never opens a frame.

The collision-avoidance relocation in ``composition_planner`` moves captions
*up* when it detects text at the bottom, and that relocation is what put the
caption on the face. ``upper_center`` resolves to roughly 25% of frame height,
which clears the platform UI band and lands squarely on the eyes. So the check
that would have caught this needs to know where the faces are **in the output**,
after the crop, not where they were in the source.

Method
------
Reuses the tracked YuNet ONNX model that ``face_tracker`` already depends on
(``core/models/face_detection_yunet_2023mar.onnx``, 232 KB, committed to the repo),
so this adds no dependency. Detection runs on decoded frames at the analysis
resolution, and boxes are returned in that same resolution, which is the
coordinate space :mod:`core.verification.captions` expects.

Degrades to "no faces" rather than raising. A missing model or a decode failure
produces an empty list and a reason, and the caller decides whether that blocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import probe

# YuNet's own input. The model is scale-agnostic but its accuracy is tuned near
# this size, and it keeps detection cost flat regardless of clip resolution.
DETECT_WIDTH = 640
DETECT_HEIGHT = 360

# Confidence and NMS thresholds. These match the values face_tracker uses
# (face_tracker.py:166-168) so the two do not disagree about who counts as a
# face, which would make a caption "on a face" here and "not on a face" there.
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.3

# Frames sampled across the clip. A face is present or absent for seconds at a
# time, so a modest sample is enough and keeps the check cheap.
DEFAULT_SAMPLES = 24


@dataclass
class FaceBox:
    """One detected face, in analysis-resolution pixels."""

    x: float
    y: float
    width: float
    height: float
    score: float

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    # The region a caption must avoid is NOT the whole face box. In a tight
    # portrait crop -- which is exactly what portrait_face produces -- the face
    # fills most of the frame, so "fraction of caption pixels inside the face box"
    # saturates at 100% and carries no information. Measured: with the full box
    # plus padding, five of six golden masters reported 100%.
    #
    # What actually makes a caption unreadable is covering the **upper** face:
    # brow and eyes. So the avoidance region is the upper band of the face,
    # anchored from the top of the box, with modest side padding. A caption
    # sitting on the forehead is still wrong, and this catches it, but the number
    # now means something.
    def upper_face(
        self, fraction: float = 0.55, sides: float = 0.25
    ) -> Tuple[float, float, float, float]:
        """Return ``(x, y, w, h)`` for the brow-and-eyes region of this face."""
        pad_x = self.width * sides
        width = self.width + 2 * pad_x
        height = self.height * fraction
        return (max(0.0, self.x - pad_x), max(0.0, self.y), width, height)

    def padded(self, top: float = 0.55, bottom: float = 0.35,
               sides: float = 0.30) -> Tuple[float, float, float, float]:
        """Whole face plus margin. Informational only; see :meth:`upper_face`."""
        pad_x = self.width * sides
        pad_y = self.height * top
        return (
            max(0.0, self.x - pad_x),
            max(0.0, self.y - pad_y),
            self.width + 2 * pad_x,
            self.height + pad_y + self.height * bottom,
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "width": round(self.width, 1),
            "height": round(self.height, 1),
            "score": round(self.score, 3),
        }


@dataclass
class FaceReport:
    """Faces found across a clip, and where."""

    ok: bool = True
    reason: str = ""
    detector: str = ""
    samples: int = 0
    frames_with_faces: int = 0
    boxes: List[FaceBox] = field(default_factory=list)
    largest: Optional[Dict[str, Any]] = None
    notes: List[str] = field(default_factory=list)

    @property
    def any_faces(self) -> bool:
        return bool(self.boxes)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "detector": self.detector,
            "samples": self.samples,
            "frames_with_faces": self.frames_with_faces,
            "box_count": len(self.boxes),
            "largest": self.largest,
            "notes": self.notes,
        }


def _load_yunet() -> Tuple[Optional[Any], str]:
    """Load the tracked YuNet model. Returns ``(detector, reason)``.

    Delegates to :func:`core.face_tracker.get_face_detector`, which owns the
    model path *and* the download fallback, and falls back through MediaPipe to
    Haar. Re-implementing the load here was a real bug: the first version of this
    module opened the model path directly and reported "not present" on CI,
    because the file tracked in git is named ``face_detection_yunen_2023mar.onnx``
    (note the transposed letters) while the loader looks for
    ``face_detection_yunet.onnx`` and fetches it at runtime. The caption check
    therefore silently did nothing in CI while reporting nothing wrong.

    That is precisely the failure this whole package exists to prevent -- a gate
    that cannot run and does not say so -- so the fix is twofold: reuse the loader,
    and make an unrunnable check report a warning rather than passing quietly.
    """
    try:
        import cv2
    except ImportError:
        return None, "OpenCV is not installed"

    if not hasattr(cv2, "FaceDetectorYN"):
        return None, "this OpenCV build has no FaceDetectorYN (OpenCV >= 4.5.4 required)"

    try:
        from core.face_tracker import get_face_detector
    except Exception as error:
        return None, f"face_tracker unavailable: {error}"

    try:
        detector = get_face_detector()
    except Exception as error:
        return None, f"no face detector could be constructed: {error}"
    if detector is None:
        return None, "face_tracker returned no detector"
    if not hasattr(detector, "detect"):
        # face_tracker falls back through MediaPipe and Haar, which expose
        # process()/detectMultiScale() instead. detect_in_frame below only knows
        # the YuNet API, so those backends silently produced zero faces and the
        # caption gate reported "no faces" instead of "wrong detector".
        return None, (
            f"face detector backend {type(detector).__name__} has no detect(); "
            f"only the OpenCV YuNet detector can run the caption check"
        )
    return detector, ""


def detect_in_frame(frame: Any, detector: Any) -> List[FaceBox]:
    """Detect faces in one BGR frame.

    ``face_tracker.get_face_detector`` constructs YuNet at a fixed 320x320 input,
    and MediaPipe's legacy detector is fixed at 640x640, so the frame is resized
    to whatever the detector was built for. ``detect_in_frame`` stays pure enough
    to test with a stub detector.
    """
    import cv2

    height, width = frame.shape[:2]
    target = getattr(detector, "_input_size", None)
    if not isinstance(target, (tuple, list)) or len(target) != 2:
        # YuNet, the only detector that matters here. face_tracker creates it at
        # (320, 320); YuNet accepts the frame size it is given.
        target = (320, 320)
    target = (int(target[0]), int(target[1]))
    if (width, height) != target:
        resized = cv2.resize(frame, target)
    else:
        resized = frame
    try:
        # face_tracker.detect_faces_in_frame sets the input size on this shared,
        # cached detector to the full frame before every detect(). Reusing it here
        # without resetting the size means detect() runs with a 320x320 image
        # while the detector still expects e.g. 1920x1080, and YuNet returns
        # nothing -- which is exactly how the CI caption gate reported "no faces
        # detected" on every clip while the same code found faces locally.
        if hasattr(detector, "setInputSize"):
            detector.setInputSize((resized.shape[1], resized.shape[0]))
        _, faces = detector.detect(resized)
    except Exception:
        return []
    if faces is None:
        return []

    boxes: List[FaceBox] = []
    for face in faces:
        try:
            x, y, box_width, box_height = [float(value) for value in face[:4]]
            score = float(face[14]) if len(face) > 14 else 1.0
        except (TypeError, ValueError, IndexError):
            continue
        # Rescale back into the caller's coordinate space.
        if (width, height) != target:
            scale_x = width / float(target[0])
            scale_y = height / float(target[1])
            x *= scale_x
            box_width *= scale_x
            y *= scale_y
            box_height *= scale_y
        boxes.append(FaceBox(x=x, y=y, width=box_width, height=box_height, score=score))
    return boxes


def analyse(
    path: Path,
    samples: int = DEFAULT_SAMPLES,
    max_keep: int = 12,
) -> FaceReport:
    """Sample the clip and collect the faces present in it."""
    try:
        import cv2  # noqa: F401
    except ImportError:  # pragma: no cover
        return FaceReport(ok=False, reason="OpenCV is not installed")

    detector, reason = _load_yunet()
    if detector is None:
        return FaceReport(ok=False, reason=reason, detector="yunet")

    frames, decode_reason = probe.decode_colour_frames(
        path, DETECT_WIDTH, DETECT_HEIGHT, max_frames=samples
    )
    if not frames:
        return FaceReport(
            ok=False, reason=decode_reason or "no frames decoded", detector="yunet"
        )

    report = FaceReport(detector="yunet", samples=len(frames))
    collected: List[FaceBox] = []
    for frame in frames:
        found = detect_in_frame(frame, detector)
        if found:
            report.frames_with_faces += 1
            collected.extend(found)

    if not collected:
        report.reason = "no faces detected in any sampled frame"
        return report

    # Keep the largest faces; small background faces are not what a caption
    # collides with in a way a viewer would notice.
    collected.sort(key=lambda box: box.width * box.height, reverse=True)
    report.boxes = collected[:max_keep]
    report.largest = report.boxes[0].as_dict()
    report.reason = (
        f"{report.frames_with_faces}/{report.samples} sampled frames contained at "
        f"least one face; largest {report.largest['width']:.0f}x"
        f"{report.largest['height']:.0f} at ({report.largest['x']:.0f},"
        f"{report.largest['y']:.0f})"
    )
    return report


def analyse_timeline(
    path: Path,
    samples: int = DEFAULT_SAMPLES,
) -> List[Tuple[float, List[FaceBox]]]:
    """Per-sample faces with their time position, for time-ranged captions.

    ``analyse`` merges every sampled frame into one report, which is why the
    caption repair could only ever choose one static band. A
    ``multi_shot_dynamic`` clip switches speakers at different heights, so each
    time window needs to avoid the faces that are actually on screen in that
    window. Returns ``(time_fraction, boxes)`` per decoded frame, evenly spaced
    across the clip; fractions are exact because ``decode_colour_frames``
    samples uniformly.
    """
    try:
        import cv2  # noqa: F401
    except ImportError:  # pragma: no cover
        return []

    detector, _reason = _load_yunet()
    if detector is None:
        return []

    frames, _decode_reason = probe.decode_colour_frames(
        path, DETECT_WIDTH, DETECT_HEIGHT, max_frames=samples
    )
    if not frames:
        return []

    denominator = max(1, len(frames) - 1)
    return [
        (index / denominator, detect_in_frame(frame, detector) or [])
        for index, frame in enumerate(frames)
    ]


def avoid_rects(
    report: FaceReport,
    use_largest_only: bool = True,
    upper_only: bool = True,
) -> List[Any]:
    """Convert detected faces into caption-avoidance rectangles.

    ``upper_only`` (the default) returns the brow-and-eyes band rather than the
    whole face, because the whole face saturates in a tight crop and reports a
    meaningless 100%. See :meth:`FaceBox.upper_face`.

    Returns :class:`core.verification.captions.Rect` objects in the same
    coordinate space the face report used.
    """
    from .captions import Rect

    boxes = report.boxes[:1] if use_largest_only and report.boxes else report.boxes
    rects: List[Any] = []
    for box in boxes:
        if upper_only:
            x, y, width, height = box.upper_face()
        else:
            x, y, width, height = box.padded()
        rects.append(Rect(x=int(x), y=int(y), width=int(width), height=int(height)))
    return rects
