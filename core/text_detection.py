"""
Text-region detection, in tiers.

`caption_source_collision` is a *blocking* QA code, but until now nothing ever
populated real `protected_regions`. `source_index` used a Sobel edge heuristic
that fires on slides and on any high-contrast graphic, and misses real text on
a busy background. The check was structurally blind: it could not fail, so it
could not protect anything.

Tiers, best first:

1. **OCR** (RapidOCR / PP-OCR ONNX). Reads real glyphs and reports per-glyph
   confidence. The dependency cannot be installed or verified on the machine
   this was built on, so it is optional and degrades cleanly to the next tier.
2. **Edge** (the original Sobel heuristic). Always available. Crude, but it is
   already in production, it is deterministic, and it needs no tuning.

**A zero-dependency MSER tier was implemented, measured, and removed.** Recorded
here because the negative result is worth not repeating:

- MSER returns *zero* regions on unblurred rendered text -- antialiased glyph
  edges are not stable enough to be extremal at multiple thresholds. A Gaussian
  sigma of 3 fixes that, restoring ~100 regions/frame.
- Those ~100 regions cover only ~20-26 per text line where the line has ~35
  characters. MSER **silently misses glyphs**, so the surviving fragments sit
  34-132 px apart. Merging them into lines needs a gap threshold wider than an
  entire text block, which merges unrelated regions. Measured: a rendered line
  of "Most people misunderstand this completely" came back as 2-3 fragments at
  any threshold, and one of the three lines was lost entirely.
- On a soft lower-third over a busy background it returned **0 regions** -- the
  exact case the edge heuristic already handles.
- Loosening the stability parameters (`max_variation` 0.6, `min_diversity` 0.05)
  does yield more regions, ~1072/frame, which floods the protected-region list
  and makes the collision check fire on everything. Worse than detecting nothing.

Also note the OpenCV Python bindings are inconsistent here: the method is
`detectRegions` (camelCase) while the constructor takes `min_area` /
`max_variation` (snake_case), and neither appears in the type stubs, so a wrong
guess fails only at runtime rather than at type-check time.

Every tier returns normalised boxes with a `source` label so a report can say
which one produced a region.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from core.config import (
    OCR_CONFIDENCE_MIN,
    OCR_MAX_REGIONS_PER_FRAME,
)


@dataclass
class TextBox:
    """A normalised (0..1) text region."""

    x: float
    y: float
    width: float
    height: float
    confidence: float
    source: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "confidence": self.confidence,
            "source": self.source,
        }

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    def iou(self, other: "TextBox") -> float:
        overlap_x = min(self.right, other.right) - max(self.x, other.x)
        overlap_y = min(self.bottom, other.bottom) - max(self.y, other.y)
        if overlap_x <= 0 or overlap_y <= 0:
            return 0.0
        intersection = overlap_x * overlap_y
        union = self.width * self.height + other.width * other.height - intersection
        return intersection / union if union > 0 else 0.0


def _normalise(
    x: float, y: float, width: float, height: float,
    frame_width: int, frame_height: int,
    confidence: float, source: str,
) -> Optional[TextBox]:
    if frame_width <= 0 or frame_height <= 0 or width <= 1 or height <= 1:
        return None
    box = TextBox(
        x=round(max(0.0, min(1.0, x / frame_width)), 4),
        y=round(max(0.0, min(1.0, y / frame_height)), 4),
        width=round(max(0.0, min(1.0, x / frame_width + width / frame_width)), 4)
        - round(max(0.0, min(1.0, x / frame_width)), 4),
        height=round(max(0.0, min(1.0, y / frame_height + height / frame_height)), 4)
        - round(max(0.0, min(1.0, y / frame_height)), 4),
        confidence=round(float(confidence), 4),
        source=source,
    )
    if box.width <= 0.001 or box.height <= 0.001:
        return None
    return box


def merge_text_boxes(
    boxes: Sequence[TextBox],
    iou_threshold: float = 0.3,
) -> List[TextBox]:
    """
    Collapse boxes that describe the same text across frames.

    Necessary because a detector running per frame produces boxes that jitter by
    a few pixels each time. The index previously de-duplicated on exact
    coordinates, which worked for the edge heuristic's stable boxes but would
    have produced a hundred near-identical `protected_regions` for real OCR --
    which in turn would make the caption collision check fire on everything.

    Greedy clustering, highest-confidence box first so the winner is the most
    trustworthy, and the kept box is the mean of its cluster.
    """
    if not boxes:
        return []
    ordered = sorted(boxes, key=lambda box: -box.confidence)
    clusters: List[List[TextBox]] = []
    for box in ordered:
        for cluster in clusters:
            if any(box.iou(member) >= iou_threshold for member in cluster):
                cluster.append(box)
                break
        else:
            clusters.append([box])

    merged: List[TextBox] = []
    for cluster in clusters:
        count = float(len(cluster))
        merged.append(TextBox(
            x=round(sum(box.x for box in cluster) / count, 4),
            y=round(sum(box.y for box in cluster) / count, 4),
            width=round(sum(box.width for box in cluster) / count, 4),
            height=round(sum(box.height for box in cluster) / count, 4),
            confidence=round(max(box.confidence for box in cluster), 4),
            source=cluster[0].source,
        ))
    merged.sort(key=lambda box: (box.y, box.x))
    return merged


class EdgeTextDetector:
    """
    The original Sobel-edge heuristic, kept as the always-available floor.

    Fires on any high-contrast horizontal structure, so it over-reports on
    graphics and under-reports on soft text over a busy image. Confidence is
    deliberately low (0.45) so a real OCR tier always outranks it.
    """

    name = "edge"

    def detect(self, frame: np.ndarray) -> List[TextBox]:
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gradient = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        binary = cv2.convertScaleAbs(gradient)
        _, binary = cv2.threshold(binary, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(
            connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        boxes: List[TextBox] = []
        for contour in contours:
            x, y, box_width, box_height = cv2.boundingRect(contour)
            aspect = box_width / max(1, box_height)
            if aspect < 2.5:
                continue
            if box_width < width * 0.1 or box_height > height * 0.15:
                continue
            box = _normalise(x, y, box_width, box_height, width, height, 0.45, self.name)
            if box is not None:
                boxes.append(box)
        return boxes


class OcrTextDetector:
    """
    RapidOCR / PP-OCR ONNX adapter.

    The dependency cannot be installed or verified on the machine this was built
    on, so this class is constructed lazily and every failure path -- missing
    package, missing weights, a runtime error inside the engine -- resolves to
    "no regions" rather than raising. The caller then falls back to the next
    tier, so an unverified dependency can degrade the output but never break the
    pipeline.
    """

    name = "ocr"

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    @classmethod
    def try_create(cls) -> Optional["OcrTextDetector"]:
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore
        except Exception:
            return None
        try:
            engine = RapidOCR()
        except Exception as error:
            print(f"[-] OCR engine failed to initialise, falling back: {error}")
            return None
        return cls(engine)

    def detect(self, frame: np.ndarray) -> List[TextBox]:
        height, width = frame.shape[:2]
        try:
            result = self._engine(frame)
        except Exception:
            return []
        if not result:
            return []
        # RapidOCR returns (results, elapse) where each result is
        # (quadrilateral, text, confidence) -- but the exact shape has changed
        # between releases, so every layer is validated rather than trusted.
        payload = result[0] if isinstance(result, tuple) else result
        if not isinstance(payload, (list, tuple)):
            return []
        boxes: List[TextBox] = []
        for entry in payload:
            # Confidence is the THIRD element, not the second. Reading index 1
            # picks up the recognised text instead, float() then fails, and the
            # box is dropped by the handler below -- so a perfectly good
            # response yields zero regions with no error logged anywhere.
            if not isinstance(entry, (list, tuple)) or len(entry) < 3:
                continue
            polygon, confidence = entry[0], entry[2]
            try:
                confidence_value = float(confidence)
            except (TypeError, ValueError):
                continue
            if confidence_value < OCR_CONFIDENCE_MIN:
                continue
            rect = _quad_to_rect(polygon)
            if rect is None:
                continue
            x, y, box_width, box_height = rect
            box = _normalise(
                x, y, box_width, box_height, width, height,
                confidence_value, self.name,
            )
            if box is not None:
                boxes.append(box)
        return boxes[:OCR_MAX_REGIONS_PER_FRAME]


def _quad_to_rect(polygon: Any) -> Optional[Tuple[int, int, int, int]]:
    """Convert an OCR quadrilateral to an axis-aligned (x, y, w, h)."""
    try:
        points = [(float(point[0]), float(point[1])) for point in polygon]
    except (TypeError, ValueError, IndexError):
        return None
    if len(points) < 3:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    left, top = int(min(xs)), int(min(ys))
    return left, top, int(max(xs)) - left, int(max(ys)) - top


class _ChainedDetector:
    """Runs tiers in order and returns the first tier that finds anything."""

    def __init__(self, tiers: Sequence[Any]) -> None:
        self._tiers = list(tiers)

    @property
    def name(self) -> str:
        return "+".join(tier.name for tier in self._tiers) or "none"

    def detect(self, frame: np.ndarray) -> List[TextBox]:
        for tier in self._tiers:
            try:
                boxes = tier.detect(frame)
            except Exception as error:
                print(f"[-] Text tier {tier.name} failed, trying next: {error}")
                continue
            if boxes:
                return boxes
        return []


_detector: Optional[Any] = None
_detector_lock = threading.Lock()
_reported: Dict[str, bool] = {}


def get_text_detector(force: Optional[str] = None) -> Any:
    """
    Resolve the text detector once and cache it.

    `auto` walks the tiers and stops at the first that is actually available, so
    installing the OCR dependency later needs no code change -- only
    `pip install rapidocr-onnxruntime`.
    """
    global _detector
    from core.config import TEXT_DETECTOR

    requested = (force or TEXT_DETECTOR or "auto").strip().lower()
    with _detector_lock:
        if _detector is not None and force is None:
            return _detector

        tiers: List[Any] = []
        if requested == "edge":
            tiers.append(EdgeTextDetector())
        elif requested == "ocr":
            # Explicitly pinned to OCR: still fall back, but say so loudly,
            # because "I asked for OCR and silently got the heuristic" is
            # exactly the kind of quiet degradation that hides a broken setup.
            ocr = OcrTextDetector.try_create()
            if ocr is not None:
                tiers.append(ocr)
            else:
                print("[-] TEXT_DETECTOR=ocr but RapidOCR is unavailable; using the edge heuristic.")
            tiers.append(EdgeTextDetector())
        else:
            ocr = OcrTextDetector.try_create()
            if ocr is not None:
                if not _reported.get("ocr_ok"):
                    print("[+] OCR text detection active (RapidOCR ONNX).")
                    _reported["ocr_ok"] = True
                tiers.append(ocr)
            elif not _reported.get("ocr_absent"):
                print("[*] RapidOCR not installed; text detection uses the edge heuristic.")
                _reported["ocr_absent"] = True
            tiers.append(EdgeTextDetector())

        resolved = _ChainedDetector(tiers) if len(tiers) > 1 else tiers[0]
        if force is None:
            _detector = resolved
        return resolved


def detect_text_regions(frame: np.ndarray, force: Optional[str] = None) -> List[TextBox]:
    """Detect text regions in one frame, merged and bounded."""
    if frame is None or frame.size == 0:
        return []
    boxes = get_text_detector(force).detect(frame)
    return merge_text_boxes(boxes)[:OCR_MAX_REGIONS_PER_FRAME]


def reset_text_detector_cache() -> None:
    """Drop the cached detector. For tests and for re-reading config changes."""
    global _detector
    with _detector_lock:
        _detector = None
        _reported.clear()
