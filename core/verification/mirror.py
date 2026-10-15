"""Selfie-flip (mirrored) source detection.

A published clip in the corpus is horizontally flipped: the "SHURE SM7B"
microphone label reads right-to-left, and the subject and background are
consistently mirrored. There is no flip filter anywhere in the renderer, so the
flip is inherited from the source, and nothing in the pipeline noticed. Editorial
QA passed the clip.

**A single frame cannot tell you whether it is mirrored.** There is no universal
"correct" orientation for arbitrary footage, so any absolute heuristic --
bilateral symmetry, face-shape priors, lighting direction -- will fire on
legitimate content. The one thing that *does* carry the information is text,
because text has a canonical reading direction.

So this detector is **self-normalising**: it runs OCR on the frame and on the
horizontally flipped frame, and compares.

* Correctly oriented text is recognised in the original and not in the mirror.
* Mirrored text is the reverse.

Whichever orientation OCR reads better is the correct one. There is no absolute
threshold to tune and no content that can produce a false positive, because the
comparison is against the same content in both orientations.

Honest limits, established by measurement rather than assumption
----------------------------------------------------------------
CI run 1, without caption exclusion, on the six golden masters: the detector
returned a **false negative** on a clip that is demonstrably mirrored. It found 11
OCR boxes at 0.98 confidence in the original orientation against 10 at 0.88 in the
mirrored one, and concluded "correct".

The cause is structural. Ten of those eleven boxes are the pipeline's own
burned-in captions, which libass renders after the frame is assembled and which
are therefore correctly oriented no matter what the source was. One genuinely
mirrored element cannot outvote ten correct ones.

CI run 2, with the caption band excluded: **every clip now reports "no readable
text in either orientation"**, including the mirrored one. Excluding the captions
left nothing, because the captions were the only text RapidOCR could read. The
"SHURE SM7B" microphone label is small, curved and low-contrast, and RapidOCR does
not return a box for it.

So the conclusion is not "the detector is broken". It is this:

    **For podcast footage, the burned-in captions are the only reliably
    machine-readable text, and they carry no information about source
    orientation. Pixel-level mirror detection on rendered output is therefore not
    viable for this content type.**

That is a fundamental limit, not a tuning problem. There is no way to detect
mirroring from pixels alone unless the content is orientation-dependent, and text
is the only such content; if the text cannot be read, the question is
unanswerable.

The viable place to check is therefore **the source**, before the render:

* the renderer contains no flip filter (verified: no ``hflip``/``vflip``/
  ``transpose`` anywhere in the pipeline), so ``mirrored(output) ==
  mirrored(source)`` exactly;
* the source has the channel's own text, not our captions, so there is nothing to
  exclude;
* it is one check per download rather than per render.

:func:`analyse` therefore takes an optional ``source_path``: pass it when the
source segment is available and the measurement is exact, and omit it for
historical artifacts, where the honest answer is "not determinable".

Requirements:

* **Needs OCR.** Without RapidOCR the answer is "not measurable". It is not
  guessed from symmetry or face-shape priors, because a guessed answer here is
  worse than none.
* **Reuses** :mod:`core.text_detection` rather than calling RapidOCR directly.
  That module already contains the fix for HANDOFF section 4 bug 12, where the
  OCR response was parsed as ``(box, text, confidence)`` but ``entry[1]`` -- the
  recognised *text* -- was read as the confidence. ``float("MOST PEOPLE")``
  raised, the handler swallowed it, and the detector silently produced zero
  regions while appearing to work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import probe

# Sample count. Enough to average out per-frame OCR noise, few enough that the
# check stays cheap; OCR is the expensive part, not the decoding.
DEFAULT_SAMPLES = 6

# Decode size for OCR. RapidOCR is tuned for roughly this scale and going larger
# only costs time.
OCR_WIDTH = 960
OCR_HEIGHT = 1280

# A frame counts as "readable" only above this mean confidence. Below it the
# frame is not evidence either way.
MIN_READABLE_CONFIDENCE = 0.55

# The mirrored orientation must beat the original by this ratio, on the
# aggregate, for the clip to be called mirrored. A ratio rather than an absolute
# gap, because confidence scales differ between engines and content.
MIRROR_DOMINANCE_RATIO = 1.15

# A clip needs at least this many readable frames before it is judged.
MIN_READABLE_FRAMES = 2


@dataclass
class MirrorReport:
    """Mirror verdict for one clip."""

    ok: bool = True
    measurable: bool = False
    reason: str = ""
    mirrored: bool = False
    samples: int = 0
    readable_frames: int = 0
    original_confidence: float = 0.0
    mirrored_confidence: float = 0.0
    dominance_ratio: float = 1.0
    boxes_original: int = 0
    boxes_mirrored: int = 0
    suppressed_boxes: int = 0
    detector: str = ""
    notes: List[str] = field(default_factory=list)

    @property
    def severity(self) -> str:
        if self.mirrored:
            return "error"
        if not self.measurable:
            return "info"
        return "ok"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "measurable": self.measurable,
            "severity": self.severity,
            "reason": self.reason,
            "mirrored": self.mirrored,
            "samples": self.samples,
            "readable_frames": self.readable_frames,
            "original_confidence": round(self.original_confidence, 4),
            "mirrored_confidence": round(self.mirrored_confidence, 4),
            "dominance_ratio": round(self.dominance_ratio, 4),
            "boxes_original": self.boxes_original,
            "boxes_mirrored": self.boxes_mirrored,
            "suppressed_boxes": self.suppressed_boxes,
            "detector": self.detector,
            "notes": self.notes,
        }


def _mean_confidence(boxes: List[Any]) -> float:
    if not boxes:
        return 0.0
    return sum(float(box.confidence) for box in boxes) / len(boxes)


def classify(
    original_boxes: List[Any],
    mirrored_boxes: List[Any],
    original_confidence: float,
    mirrored_confidence: float,
    samples: int,
) -> MirrorReport:
    """Pure decision, so the threshold behaviour is testable without OCR.

    ``original_boxes`` / ``mirrored_boxes`` are the detected regions in each
    orientation; they are used for reporting and as a tiebreak, not for the
    primary signal.
    """
    report = MirrorReport(
        samples=samples,
        readable_frames=0,
        original_confidence=original_confidence,
        mirrored_confidence=mirrored_confidence,
        boxes_original=len(original_boxes),
        boxes_mirrored=len(mirrored_boxes),
    )
    best_original = max(original_confidence, 0.0)
    best_mirrored = max(mirrored_confidence, 0.0)

    if best_original < MIN_READABLE_CONFIDENCE and best_mirrored < MIN_READABLE_CONFIDENCE:
        report.reason = (
            f"no readable text in either orientation "
            f"(original {best_original:.2f}, mirrored {best_mirrored:.2f}); "
            f"mirror state is not determinable from this clip"
        )
        return report

    report.readable_frames = 1
    report.measurable = True
    report.dominance_ratio = (
        best_mirrored / best_original if best_original > 1e-6 else float("inf")
    )
    report.mirrored = report.dominance_ratio >= MIRROR_DOMINANCE_RATIO
    if report.mirrored:
        report.reason = (
            f"OCR reads the horizontally flipped frame {report.dominance_ratio:.2f}x "
            f"better ({best_mirrored:.2f} vs {best_original:.2f}), which indicates "
            f"selfie-flipped source footage"
        )
    else:
        report.reason = (
            f"OCR reads the original frame better "
            f"({best_original:.2f} vs {best_mirrored:.2f}); orientation is correct"
        )
    return report


def analyse(
    path: Path,
    samples: int = DEFAULT_SAMPLES,
    exclude_rects: Optional[List[Any]] = None,
    source_path: Optional[Path] = None,
) -> MirrorReport:
    """Detect selfie-flipped footage by comparing OCR across orientations.

    Pass ``source_path`` when the un-rendered segment is available. That is the
    exact measurement: the renderer never flips anything, so the output's
    orientation equals the source's, and the source carries the channel's own text
    rather than our captions.

    ``exclude_rects`` are regions, in the decode resolution used here, to leave
    out of the comparison. Pass the caption band from
    :func:`core.verification.captions.detect_caption_regions` when checking output
    directly. See the module docstring for why that exclusion is required, and for
    why output-only checking is not viable for this content type.
    """
    target = source_path if source_path is not None else path
    checking_source = source_path is not None

    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        return MirrorReport(ok=False, reason="numpy is not installed")

    try:
        from core.text_detection import detect_text_regions, get_text_detector
    except Exception as error:  # pragma: no cover - import guard
        return MirrorReport(ok=False, reason=f"text detection unavailable: {error}")

    try:
        detector = get_text_detector()
    except Exception as error:
        return MirrorReport(ok=False, reason=f"no text detector: {error}")

    frames, reason = probe.decode_colour_frames(
        target, OCR_WIDTH, OCR_HEIGHT, max_frames=samples * 8
    )
    if not frames:
        return MirrorReport(ok=False, reason=reason or "no frames decoded")

    detector_name = getattr(detector, "name", "unknown")
    if "ocr" not in str(detector_name).lower():
        return MirrorReport(
            ok=False,
            reason=(
                f"active text detector is '{detector_name}', not OCR. Mirror "
                f"detection needs OCR; set TEXT_DETECTOR=ocr to enable it."
            ),
            detector=str(detector_name),
        )

    # Evenly pick readable samples from what was decoded.
    if len(frames) <= samples:
        chosen = frames
    else:
        step = len(frames) / float(samples)
        chosen = [frames[int(index * step)] for index in range(samples)]

    mask: Optional[Any] = None
    if exclude_rects:
        from . import captions as captions_module

        # The band arrives in captions-module mask coordinates (270x480) and must
        # be rescaled to this module's decode resolution. Passing it unscaled
        # would exclude a 1/4-size region near the origin and silently exclude
        # nothing useful.
        scale_x = OCR_WIDTH / float(captions_module.MASK_WIDTH)
        scale_y = OCR_HEIGHT / float(captions_module.MASK_HEIGHT)
        mask = np.zeros(frames[0].shape[:2], dtype=bool)
        height, width = mask.shape
        for rect in exclude_rects:
            x0 = max(0, min(width, int(rect.x * scale_x)))
            y0 = max(0, min(height, int(rect.y * scale_y)))
            x1 = max(0, min(width, int(rect.right * scale_x)))
            y1 = max(0, min(height, int(rect.bottom * scale_y)))
            if x1 > x0 and y1 > y0:
                mask[y0:y1, x0:x1] = True
        if not mask.any():
            mask = None
            report_mask_note = "exclusion band was empty after rescaling; ignored"
        else:
            report_mask_note = None
    else:
        report_mask_note = None

    def _suppress(boxes: List[Any]) -> List[Any]:
        if mask is None:
            return boxes
        height, width = mask.shape
        kept: List[Any] = []
        for box in boxes:
            x0 = max(0, min(width, int(box.x)))
            y0 = max(0, min(height, int(box.y)))
            x1 = max(0, min(width, int(box.right)))
            y1 = max(0, min(height, int(box.bottom)))
            if x1 <= x0 or y1 <= y0:
                continue
            # A box is suppressed when most of it falls inside the excluded band.
            if mask[y0:y1, x0:x1].mean() > 0.5:
                continue
            kept.append(box)
        return kept

    original_scores: List[float] = []
    mirrored_scores: List[float] = []
    original_boxes: List[Any] = []
    mirrored_boxes: List[Any] = []
    suppressed = 0

    for frame in chosen:
        flipped = frame[:, ::-1, :]
        try:
            raw_forward = list(detect_text_regions(frame.copy(), force="ocr"))
            raw_reverse = list(detect_text_regions(flipped.copy(), force="ocr"))
        except Exception:
            # A detector that throws is not evidence in either direction.
            continue
        forward = _suppress(raw_forward)
        reverse = _suppress(raw_reverse)
        suppressed += (len(raw_forward) - len(forward)) + (len(raw_reverse) - len(reverse))
        original_boxes.extend(forward)
        mirrored_boxes.extend(reverse)
        original_scores.append(_mean_confidence(forward))
        mirrored_scores.append(_mean_confidence(reverse))

    if not original_scores:
        return MirrorReport(
            ok=False,
            reason="OCR produced no results on any sampled frame",
            detector=str(detector_name),
        )

    report = classify(
        original_boxes,
        mirrored_boxes,
        max(original_scores) if original_scores else 0.0,
        max(mirrored_scores) if mirrored_scores else 0.0,
        len(chosen),
    )
    report.detector = str(detector_name)
    report.suppressed_boxes = suppressed
    if checking_source:
        report.notes.append(
            "measured on the source segment: the renderer contains no flip "
            "filter, so the output inherits the source's orientation exactly"
        )
    if mask is not None:
        report.notes.append(
            f"excluded {suppressed} OCR box(es) covering the pipeline's own "
            f"burned-in captions before comparing orientations. Without that "
            f"exclusion the verdict is meaningless: libass renders captions after "
            f"the frame is assembled, so they are correctly oriented no matter "
            f"what the source was, and they outvote genuinely mirrored source text."
        )
    elif not checking_source:
        report.notes.append(
            "no source segment supplied, so the pipeline's own captions were "
            "included in the comparison. On a rendered clip that usually makes the "
            "verdict meaningless; check the source instead."
        )
    elif report_mask_note:
        report.notes.append(report_mask_note)
    if report.measurable and report.readable_frames < MIN_READABLE_FRAMES:
        report.notes.append(
            f"only {report.readable_frames} sample(s) carried readable text; "
            f"the verdict rests on a thin margin"
        )
    return report
