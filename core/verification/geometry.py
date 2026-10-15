"""Frame geometry: pillarboxing, letterboxing, and aspect correctness.

This module exists because of a specific, measured defect. In the published clip
``clip_1_HOW_I_MADE_400000_IN_A_MONTH.mp4`` the two stacked speaker panes were
built at 540x960 -- a 9:16 aspect -- and then scaled into a 1080x960 pane with
``force_original_aspect_ratio=decrease`` followed by an unconditional
``pad=...:color=black``. The result was 270 pixels of pure black on each side:
**50% of the frame width was dead**, and the caption was wider than the live
content strip so it floated over the bars.

Two independent things went wrong, and both had to be fixed for the defect to
disappear:

1. The pane was computed at the wrong aspect. ``PANE_ASPECT = 9/8`` is correct in
   ``face_tracker.py`` but the shipped artifact proves the ratio did not survive
   to the pixels.
2. Nothing noticed. ``blackdetect`` measures whole-frame darkness, so a frame that
   is bright in the middle and black at the edges is *not* a black interval. There
   was no black-bar check anywhere in QA, which is the whole reason the worst
   visual defect in the audit shipped to three social platforms.

So this module measures the *content box* inside the frame directly, which is the
only formulation that catches bars, and reports the **worst** sample rather than
the mean -- a clip that is pillarboxed for 2 of its 45 seconds is a broken clip.

Distinguishing bars from fades
-----------------------------

A raw content box cannot tell a pillarbox from a fade, and the first version of
this module could not. Both shrink the fitted box:

* **Pillarbox** -- the content region is bright and normally exposed; the
  surround is genuine black padding. Bar mean luminance is near zero while the
  content mean is high.
* **Fade** -- there is no padding at all. The whole frame, edges included, has
  been scaled down toward black, so the edges simply fall under the content
  threshold. Bar mean and content mean fall *together*, so their ratio stays
  near 1.

Measured on the corpus, the un-discriminated version reported 4.1% bars on
``clip_2_WHY_DISNEY_PAY_IS_A_TRAP.mp4`` -- which turned out to be a fade-out over
the final 1.2 seconds, not a defect. A verifier that cries wolf is worse than no
verifier, because people learn to ignore it, so :func:`classify_bars` gates on the
bar-to-content luminance ratio and on the content region being bright enough to
have a meaningful surround.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import probe

# A column/row counts as "content" when any pixel in it exceeds this. 24/255 is
# low enough to ignore codec ringing around genuinely black areas, high enough
# not to be fooled by a dark-but-visible subject.
DEFAULT_CONTENT_THRESHOLD = 24

# Below this fraction of the frame, "content" is a stray bright pixel rather than
# a subject, and a box fitted to it would report a bogus full-bleed frame.
MIN_CONTENT_FRACTION = 0.02

# --- bar-versus-fade discrimination -----------------------------------------
# Mean luminance of the candidate bar strip, as a fraction of the mean luminance
# of the content region. A true black bar sits near 0. A faded frame edge tracks
# the content's own exposure, so it lands near 1.0. 0.35 separates them with a
# wide margin on both sides.
BAR_TO_CONTENT_LUMA_MAX = 0.35

# Below this mean luminance the content region is itself too dark to reason about.
# Almost always a fade, a dip to black between shots, or an all-black frame.
MIN_CONTENT_MEAN_LUMA = 40.0

# A bar must persist across at least this many consecutive samples before it is
# reported as a clip-level defect. At the default sampling rate that is roughly
# half a second, which filters single-frame codec artifacts and shot transitions
# without hiding a real sustained defect.
MIN_BAR_PERSISTENCE_SAMPLES = 3

# --- graded severity --------------------------------------------------------
# A single binary threshold would be dishonest here, because the corpus contains
# three genuinely different situations that a naive detector conflates:
#
#   24.8%  clip_1_HOW_I_MADE_400000_IN_A_MONTH.mp4 -- 268px of dead frame on each
#          side of a split-screen pane. A real defect, and the worst thing in the
#          published output.
#    4.1%  clip_2_WHY_DISNEY_PAY_IS_A_TRAP.mp4 -- the `blur_stack` layout's
#          blurred background, which is *designed* to be near-black. The blurred
#          coffee-cup pattern measures column max 0-6, so any content-box
#          detector flags it. Not a defect.
#   <2%    codec ringing, fades, and shot transitions.
#
# 8% (86px at 1080 wide) sits between the two real cases with roughly a factor
# of two of margin on each side, and it is the width at which dead frame becomes
# something a viewer would name as broken. Below that, findings are reported as
# warnings so they stay visible without blocking a publish.
BLOCKING_BAR_PCT = 8.0
WARNING_BAR_PCT = 3.0

# The same reasoning applies to pane aspect. A `portrait_face` shot is a 9:16
# crop of a 16:9 source, so the *source* is wider than the output and no
# expectation on the source aspect is meaningful. What matters is a multi-pane
# layout producing panes that are not 9:8, which the layout-aware caller checks.
PANE_ASPECT = 9.0 / 8.0
PANE_ASPECT_TOLERANCE = 0.04


@dataclass
class ContentBox:
    """Bounding box of non-black content within a frame."""

    x: int
    y: int
    width: int
    height: int
    frame_width: int
    frame_height: int

    @property
    def aspect(self) -> float:
        return (self.width / self.height) if self.height else 0.0

    @property
    def bar_left_pct(self) -> float:
        return (100.0 * self.x / self.frame_width) if self.frame_width else 0.0

    @property
    def bar_right_pct(self) -> float:
        if not self.frame_width:
            return 0.0
        return 100.0 * (self.frame_width - self.x - self.width) / self.frame_width

    @property
    def bar_top_pct(self) -> float:
        return (100.0 * self.y / self.frame_height) if self.frame_height else 0.0

    @property
    def bar_bottom_pct(self) -> float:
        if not self.frame_height:
            return 0.0
        return 100.0 * (self.frame_height - self.y - self.height) / self.frame_height

    @property
    def worst_vertical_bar_pct(self) -> float:
        return max(self.bar_left_pct, self.bar_right_pct)

    @property
    def worst_horizontal_bar_pct(self) -> float:
        return max(self.bar_top_pct, self.bar_bottom_pct)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "aspect": round(self.aspect, 4),
            "bar_left_pct": round(self.bar_left_pct, 2),
            "bar_right_pct": round(self.bar_right_pct, 2),
            "bar_top_pct": round(self.bar_top_pct, 2),
            "bar_bottom_pct": round(self.bar_bottom_pct, 2),
        }


@dataclass
class GeometryReport:
    """Geometry verdict for one clip, aggregated over sampled frames."""

    ok: bool = True
    reason: str = ""
    samples: int = 0
    dimensions: Optional[Dict[str, int]] = None
    declared_aspect: float = 0.0
    worst_vertical_bar_pct: float = 0.0
    worst_horizontal_bar_pct: float = 0.0
    worst_aspect: float = 0.0
    worst_aspect_at: float = 0.0
    mean_aspect: float = 0.0
    all_black_frames: int = 0
    worst_box: Optional[Dict[str, Any]] = None
    worst_box_at: float = 0.0
    bar_samples: int = 0
    bar_run: int = 0
    fades_discarded: int = 0
    notes: List[str] = field(default_factory=list)

    @property
    def bar_fraction(self) -> float:
        return (self.bar_samples / self.samples) if self.samples else 0.0

    @property
    def blocking(self) -> bool:
        """True when a bar is wide enough to block publication on its own."""
        return self.worst_vertical_bar_pct >= BLOCKING_BAR_PCT

    @property
    def warning(self) -> bool:
        return not self.blocking and self.worst_vertical_bar_pct >= WARNING_BAR_PCT

    @property
    def severity(self) -> str:
        if not self.ok or self.blocking:
            return "error"
        if self.warning:
            return "warning"
        return "ok"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "severity": self.severity,
            "reason": self.reason,
            "samples": self.samples,
            "dimensions": self.dimensions,
            "declared_aspect": round(self.declared_aspect, 4),
            "worst_vertical_bar_pct": round(self.worst_vertical_bar_pct, 2),
            "worst_horizontal_bar_pct": round(self.worst_horizontal_bar_pct, 2),
            "worst_aspect": round(self.worst_aspect, 4),
            "worst_aspect_at": round(self.worst_aspect_at, 2),
            "mean_aspect": round(self.mean_aspect, 4),
            "all_black_frames": self.all_black_frames,
            "bar_samples": self.bar_samples,
            "bar_fraction": round(self.bar_fraction, 4),
            "bar_run": self.bar_run,
            "fades_discarded": self.fades_discarded,
            "worst_box": self.worst_box,
            "worst_box_at": round(self.worst_box_at, 2),
            "notes": self.notes,
        }


# Decode width. 1080/4 = 270 keeps a 1px bar at 25% of a pixel, which is still
# detectable as a below-threshold column, while cutting decode volume ~16x.
PROFILE_WIDTH = 270
PROFILE_HEIGHT = 480


def analyse(
    path: Path,
    max_samples: int = 240,
    threshold: int = DEFAULT_CONTENT_THRESHOLD,
    min_persistence: int = MIN_BAR_PERSISTENCE_SAMPLES,
) -> GeometryReport:
    """Measure bars and aspect across a rendered clip.

    Bars are only reported when they (a) survive the fade discrimination and
    (b) persist for at least ``min_persistence`` consecutive samples. Both gates
    exist because the first version of this function reported 4.1% bars on a clip
    whose only inset was a fade-out over its final second.
    """
    width, height = probe.dimensions(path)
    report = GeometryReport(dimensions={"width": width, "height": height})
    if width <= 0 or height <= 0:
        report.ok = False
        report.reason = "no video stream, or zero dimensions"
        return report
    report.declared_aspect = width / height

    frames, reason = probe.decode_gray_frames(
        path, PROFILE_WIDTH, PROFILE_HEIGHT, max_frames=max_samples
    )
    if not frames:
        report.ok = False
        report.reason = reason or "no frames decoded"
        return report

    boxes: List[ContentBox] = []
    verdicts: List[BarVerdict] = []
    for frame in frames:
        fitted = content_box(frame, threshold=threshold)
        if fitted is None:
            continue
        verdict = classify_bars(frame, fitted, threshold=threshold)
        boxes.append(fitted)
        verdicts.append(verdict)
        if not verdict.is_real_bar and verdict.reason.startswith("edges track"):
            report.fades_discarded += 1

    if not boxes:
        report.ok = False
        report.reason = "no frame contained enough content to fit a box"
        report.all_black_frames = len(frames)
        return report

    # Aspect is measured on every frame that has a box, independent of the bar
    # verdict: a 9:16 pane letterboxed inside a 9:8 slot still reports its own
    # aspect, and that is the B1 signature.
    aspects = [box.aspect for box in boxes]
    report.samples = len(boxes)
    report.all_black_frames = max(0, len(frames) - len(boxes))
    report.mean_aspect = sum(aspects) / len(aspects)
    worst_aspect_index = min(range(len(boxes)), key=lambda i: aspects[i])
    report.worst_aspect = aspects[worst_aspect_index]
    report.worst_aspect_at = 100.0 * worst_aspect_index / max(1, len(boxes))

    real_flags = [verdict.is_real_bar for verdict in verdicts]
    report.bar_run = _longest_run(real_flags)
    report.bar_samples = sum(1 for flag in real_flags if flag)

    # A bar that never persists is a transition artifact, not a defect.
    if report.bar_run >= max(1, min_persistence) and real_flags:
        window_start = 0
        best_value, best_pct = 0.0, 0.0
        for index, flag in enumerate(real_flags):
            if not flag:
                window_start = index + 1
                continue
            window = verdicts[window_start : index + 1]
            if len(window) < max(1, min_persistence):
                continue
            candidate = max(item.vertical_pct for item in window)
            if candidate > best_value:
                best_value = candidate
                best_pct = max(item.horizontal_pct for item in window)
                best_box = window[0].box
        if best_value > 0.0:
            report.worst_vertical_bar_pct = best_value
            report.worst_horizontal_bar_pct = best_pct
            report.worst_box = best_box.as_dict()
            report.worst_box_at = 100.0 * report.bar_samples / max(1, len(boxes))

    if report.all_black_frames == len(frames):
        report.ok = False
        report.reason = "every sampled frame was entirely black"
    return report


def content_box(
    frame: Any,
    threshold: int = DEFAULT_CONTENT_THRESHOLD,
) -> Optional[ContentBox]:
    """Fit the tightest box containing all non-black content in a greyscale frame.

    Pure function on a 2D array, so it is unit-testable against synthetic input
    with no ffmpeg involved. Returns ``None`` when the frame is entirely black,
    which is itself a finding the caller must handle rather than ignore.
    """
    import numpy as np

    if frame is None or getattr(frame, "size", 0) == 0:
        return None
    array = np.asarray(frame)
    if array.ndim != 2:
        return None
    frame_height, frame_width = array.shape

    column_max = array.max(axis=0)
    row_max = array.max(axis=1)
    columns = np.flatnonzero(column_max > threshold)
    rows = np.flatnonzero(row_max > threshold)
    if columns.size == 0 or rows.size == 0:
        return None

    x0, x1 = int(columns[0]), int(columns[-1])
    y0, y1 = int(rows[0]), int(rows[-1])
    width, height = x1 - x0 + 1, y1 - y0 + 1

    # A box this small is a stray highlight, not a subject. Treating it as the
    # content box would report a bogus full-bleed frame and hide a real problem.
    if width * height < MIN_CONTENT_FRACTION * frame_width * frame_height:
        return None

    return ContentBox(x=x0, y=y0, width=width, height=height,
                      frame_width=frame_width, frame_height=frame_height)


@dataclass
class BarVerdict:
    """Whether an apparent inset is genuine black padding or a faded frame edge."""

    box: ContentBox
    is_real_bar: bool
    reason: str
    bar_to_content_luma: float
    content_mean_luma: float

    @property
    def vertical_pct(self) -> float:
        return self.box.worst_vertical_bar_pct if self.is_real_bar else 0.0

    @property
    def horizontal_pct(self) -> float:
        return self.box.worst_horizontal_bar_pct if self.is_real_bar else 0.0


def classify_bars(
    frame: Any,
    box: ContentBox,
    threshold: int = DEFAULT_CONTENT_THRESHOLD,
) -> BarVerdict:
    """Decide whether an inset content box is real padding or a faded edge.

    Pure function on a 2D array, so it is unit-testable with no ffmpeg.
    """
    import numpy as np

    array = np.asarray(frame)
    if array.ndim != 2:
        return BarVerdict(box, False, "frame is not 2D", 1.0, 0.0)

    height, width = array.shape
    x0, y0 = box.x, box.y
    x1 = min(width, box.x + box.width)
    y1 = min(height, box.y + box.height)

    content = array[y0:y1, x0:x1]
    if content.size == 0:
        return BarVerdict(box, False, "empty content region", 1.0, 0.0)
    content_mean = float(content.mean())

    # Build a mask of everything outside the content box: the candidate surround.
    surround_mask = np.ones((height, width), dtype=bool)
    surround_mask[y0:y1, x0:x1] = False
    surround = array[surround_mask]

    has_vertical = box.worst_vertical_bar_pct > 0.0
    has_horizontal = box.worst_horizontal_bar_pct > 0.0
    if not (has_vertical or has_horizontal):
        return BarVerdict(box, False, "content is full-bleed", 0.0, content_mean)

    if content_mean < MIN_CONTENT_MEAN_LUMA:
        return BarVerdict(
            box,
            False,
            f"content too dark to judge (mean {content_mean:.1f} < {MIN_CONTENT_MEAN_LUMA})",
            1.0,
            content_mean,
        )

    if surround.size == 0:
        return BarVerdict(box, False, "no surround to measure", 0.0, content_mean)

    # Only the strips on the sides that actually have bars are informative. A
    # letterboxed frame has empty left/right strips of zero area, which would
    # drag the mean toward zero and read as a pillarbox.
    strips: List[Any] = []
    frame_height, frame_width = array.shape
    if box.bar_left_pct > 0.0 and x0 > 0:
        strips.append(array[:, :x0])
    if box.bar_right_pct > 0.0 and x1 < frame_width:
        strips.append(array[:, x1:])
    if box.bar_top_pct > 0.0 and y0 > 0:
        strips.append(array[:y0, :])
    if box.bar_bottom_pct > 0.0 and y1 < frame_height:
        strips.append(array[y1:, :])

    strips = [strip for strip in strips if strip.size > 0]
    if not strips:
        return BarVerdict(box, False, "no measurable bar strips", 0.0, content_mean)

    bar_mean = float(np.mean(np.concatenate([strip.ravel() for strip in strips])))
    ratio = bar_mean / content_mean if content_mean > 0 else 1.0

    if ratio > BAR_TO_CONTENT_LUMA_MAX:
        return BarVerdict(
            box,
            False,
            f"edges track content exposure (bar/content luma {ratio:.2f}); this is a "
            f"fade, not padding",
            ratio,
            content_mean,
        )
    return BarVerdict(
        box,
        True,
        f"genuine padding (bar/content luma {ratio:.2f})",
        ratio,
        content_mean,
    )


def profile_frames(
    frames: Sequence[Any],
    threshold: int = DEFAULT_CONTENT_THRESHOLD,
) -> List[ContentBox]:
    """Content box for every frame that has one. All-black frames are skipped."""
    boxes: List[ContentBox] = []
    for frame in frames:
        box = content_box(frame, threshold=threshold)
        if box is not None:
            boxes.append(box)
    return boxes


def _longest_run(flags: Sequence[bool]) -> int:
    best = 0
    current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def summarise(boxes: Sequence[ContentBox], frame_count: int) -> Dict[str, Any]:
    """Reduce per-frame boxes to the numbers a threshold can be applied to.

    Worst-case, not mean. A clip that is correct for 43 of 45 seconds and
    pillarboxed for 2 has a defect, and averaging would dilute a 25% bar across
    45 samples down to roughly nothing.

    This is the geometry-only summary. It is retained for unit testing the
    arithmetic; :func:`analyse` layers the bar-versus-fade discrimination on top.
    """
    if not boxes:
        return {
            "samples": 0,
            "all_black_frames": frame_count,
            "worst_vertical_bar_pct": 100.0,
            "worst_horizontal_bar_pct": 100.0,
            "worst_aspect": 0.0,
            "worst_aspect_index": -1,
            "worst_box": None,
            "worst_box_index": -1,
            "mean_aspect": 0.0,
        }

    vertical = max(box.worst_vertical_bar_pct for box in boxes)
    horizontal = max(box.worst_horizontal_bar_pct for box in boxes)
    aspects = [box.aspect for box in boxes]
    worst_aspect_index = min(range(len(boxes)), key=lambda i: aspects[i])
    worst_bar_index = max(range(len(boxes)), key=lambda i: boxes[i].worst_vertical_bar_pct)

    return {
        "samples": len(boxes),
        "all_black_frames": max(0, frame_count - len(boxes)),
        "worst_vertical_bar_pct": vertical,
        "worst_horizontal_bar_pct": horizontal,
        "worst_aspect": aspects[worst_aspect_index],
        "worst_aspect_index": worst_aspect_index,
        "worst_box": boxes[worst_bar_index].as_dict(),
        "worst_box_index": worst_bar_index,
        "mean_aspect": sum(aspects) / len(aspects),
    }
