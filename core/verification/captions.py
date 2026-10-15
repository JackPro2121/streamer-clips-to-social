"""Caption placement, measured against the pixels the pipeline actually burned in.

The defect
----------
Captions render across the subject's face. Measured on
``clip_1_THE_REAL_FEAR_OF_GOING_BROKE.mp4`` at t=1.2s, 14.0s and 31.0s, the
caption baseline sits at y ~ 490 of 1920 -- **25.5% of frame height, directly over
the eyebrows and eyes** -- while ``SAFE_ZONE_TOP`` is 240px (12.5%), so the text
is twice as deep into the frame as the safe zone allows. On the split-screen
segment the caption lands at y ~ 960, exactly on the pane divider.

Why the existing check missed it: ``editorial_qa`` compares ``shot.caption_rect``
against ``shot.protected_regions``, and **both are written by the same code**
minutes apart. It never samples a frame. It therefore confirms the plan, not the
video.

The fix in this module
----------------------
Render the real ASS through libass onto a black canvas and read back where the
glyphs actually land. That is the only formulation that can catch "the caption is
on the face", because it does not care what any plan claimed.

Reuse in the mirror detector
----------------------------
:func:`text_mask` is also what makes :mod:`core.verification.mirror` work.

The mirror detector's first CI run returned a **false negative** on the clip that
is demonstrably mirrored. The reason is structural and worth recording: OCR found
11 text boxes at 0.98 confidence in the original orientation and 10 at 0.88 in the
mirrored one, and concluded "correct". But 10 of those 11 boxes are the pipeline's
*own burned-in captions*, which libass renders after the frame is assembled and
which are therefore always correctly oriented no matter what the source was. One
genuinely mirrored element -- the "SHURE SM7B" microphone label -- cannot outvote
ten correct ones.

So mirror detection must exclude the pipeline's own text before comparing
orientations. That requires knowing where the pipeline drew, which is exactly what
this module computes. The two modules are coupled by design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import probe

# libass renders white glyphs with a black outline. Two thresholds separate the
# glyph fill from the outline, and both are useful:
#
#   FILL  - the glyph itself. Tight, so an overlap measurement is not inflated
#           by the outline padding around every character.
#   TOTAL - glyph plus outline. Wider, and the right region to *exclude* when
#           asking "did the pipeline draw anything here".
GLYPH_FILL_THRESHOLD = 128
GLYPH_TOTAL_THRESHOLD = 40

# A connected region smaller than this is a stray antialiasing artefact, not a
# word. At 1080x1920 a 2-character caption glyph cluster is well above this.
MIN_REGION_PIXELS = 24

# Decode size for the mask. Full resolution is unnecessary for "is there text
# here", and the mask is compared against face boxes which are themselves
# estimated, so mask precision beyond this buys nothing.
MASK_WIDTH = 270
MASK_HEIGHT = 480


@dataclass
class SafeZone:
    """Platform UI exclusion band, in absolute pixels of a 1080x1920 frame."""

    top: int = 240
    bottom: int = 380
    left: int = 120
    right: int = 120
    width: int = 1080
    height: int = 1920

    def as_dict(self) -> Dict[str, Any]:
        return {
            "top": self.top,
            "bottom": self.bottom,
            "left": self.left,
            "right": self.right,
            "width": self.width,
            "height": self.height,
        }

    def scaled(self, width: int, height: int) -> "SafeZone":
        return SafeZone(
            top=int(round(self.top * height / self.height)),
            bottom=int(round(self.bottom * height / self.height)),
            left=int(round(self.left * width / self.width)),
            right=int(round(self.right * width / self.width)),
            width=width,
            height=height,
        )


@dataclass
class Rect:
    """Axis-aligned box in absolute pixels."""

    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    def intersects(self, other: "Rect") -> bool:
        return not (
            self.right <= other.x
            or other.right <= self.x
            or self.bottom <= other.y
            or other.bottom <= self.y
        )

    def intersection_area(self, other: "Rect") -> int:
        width = min(self.right, other.right) - max(self.x, other.x)
        height = min(self.bottom, other.bottom) - max(self.y, other.y)
        return max(0, width) * max(0, height)

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    def as_dict(self) -> Dict[str, Any]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


@dataclass
class CaptionReport:
    """Caption placement verdict for one clip."""

    ok: bool = True
    reason: str = ""
    measured: bool = False
    ass_path: str = ""
    samples: int = 0
    # Fraction of caption pixels, worst sample, that fall on a face.
    on_face_pct: float = 0.0
    on_face_at: float = 0.0
    # Fraction of caption pixels outside the platform safe zone.
    outside_safe_zone_pct: float = 0.0
    # Fraction of caption pixels sitting on a horizontal divider band.
    on_divider_pct: float = 0.0
    # Widest caption row as a fraction of frame width. >1.0 means the text is
    # wider than the live content, which is how it ends up floating over pillarbox
    # bars.
    max_width_frac: float = 0.0
    caption_band: Optional[Dict[str, Any]] = None
    notes: List[str] = field(default_factory=list)

    @property
    def severity(self) -> str:
        if not self.measured:
            return "info"
        if not self.ok:
            return "error"
        return "ok"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "severity": self.severity,
            "reason": self.reason,
            "measured": self.measured,
            "ass_path": self.ass_path,
            "samples": self.samples,
            "on_face_pct": round(self.on_face_pct, 3),
            "on_face_at": round(self.on_face_at, 2),
            "outside_safe_zone_pct": round(self.outside_safe_zone_pct, 3),
            "on_divider_pct": round(self.on_divider_pct, 3),
            "max_width_frac": round(self.max_width_frac, 4),
            "caption_band": self.caption_band,
            "notes": self.notes,
        }


# --- pure geometry helpers, unit-testable without ffmpeg --------------------


def mask_pixels(mask: Any, threshold: int = GLYPH_FILL_THRESHOLD) -> int:
    import numpy as np

    array = np.asarray(mask)
    if array.size == 0:
        return 0
    return int((array > threshold).sum())


def fraction_inside(mask: Any, rects: Sequence[Rect], threshold: int = GLYPH_FILL_THRESHOLD) -> float:
    """Fraction of glyph pixels that fall inside any of ``rects``.

    Returns 0.0 when the mask is empty, so "no captions" is not reported as
    "captions perfectly placed".
    """
    import numpy as np

    array = np.asarray(mask)
    total = mask_pixels(array, threshold)
    if total == 0:
        return 0.0
    inside = np.zeros(array.shape, dtype=bool)
    height, width = array.shape
    for rect in rects:
        x0 = max(0, min(width, rect.x))
        y0 = max(0, min(height, rect.y))
        x1 = max(0, min(width, rect.right))
        y1 = max(0, min(height, rect.bottom))
        if x1 > x0 and y1 > y0:
            inside[y0:y1, x0:x1] = True
    hits = int((array[inside] > threshold).sum())
    return hits / float(total)


def fraction_outside_safe_zone(
    mask: Any, safe_zone: SafeZone, threshold: int = GLYPH_FILL_THRESHOLD
) -> float:
    """Fraction of glyph pixels inside the platform UI exclusion band.

    Named for what it counts: pixels that are *outside* the safe area.
    """
    import numpy as np

    array = np.asarray(mask)
    total = mask_pixels(array, threshold)
    if total == 0:
        return 0.0
    height, width = array.shape
    excluded = np.zeros(array.shape, dtype=bool)
    # Top band: platform search/header UI.
    excluded[0 : min(height, safe_zone.top), :] = True
    # Bottom band: handle, caption text, sound disc.
    bottom_start = max(0, height - safe_zone.bottom)
    excluded[bottom_start:, :] = True
    # Right column: like/comment/bookmark/share.
    right_start = max(0, width - safe_zone.right)
    excluded[:, right_start:] = True
    hits = int((array[excluded] > threshold).sum())
    return hits / float(total)


def mask_bounds(mask: Any, threshold: int = GLYPH_TOTAL_THRESHOLD) -> Optional[Rect]:
    """Tight bounding box of everything drawn into the mask."""
    import numpy as np

    array = np.asarray(mask)
    rows = np.flatnonzero((array > threshold).any(axis=1))
    cols = np.flatnonzero((array > threshold).any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    return Rect(
        x=int(cols[0]),
        y=int(rows[0]),
        width=int(cols[-1] - cols[0] + 1),
        height=int(rows[-1] - rows[0] + 1),
    )


def widest_row_frac(mask: Any, threshold: int = GLYPH_FILL_THRESHOLD) -> float:
    """Widest single row of glyphs, as a fraction of frame width.

    A value above 1.0 means the caption is wider than the frame's live content,
    which is how a caption ends up floating across pillarbox bars.
    """
    import numpy as np

    array = np.asarray(mask)
    if array.size == 0:
        return 0.0
    per_row = np.asarray((array > threshold).any(axis=1))
    if not bool(per_row.any()):
        return 0.0
    height, width = array.shape
    widest = 0
    for row in range(height):
        if not bool(per_row[row]):  # type: ignore[index]
            continue
        cols = np.flatnonzero(array[row] > threshold)
        if cols.size:
            widest = max(widest, int(cols[-1] - cols[0] + 1))
    return widest / float(width)


# --- ffmpeg-backed mask rendering -------------------------------------------


def text_mask(
    ass_path: Path,
    duration: float,
    width: int = MASK_WIDTH,
    height: int = MASK_HEIGHT,
    fonts_dir: Optional[Path] = None,
    max_frames: int = 120,
) -> Tuple[List[Any], str]:
    """Render an ASS file through libass onto black and read back the glyph mask.

    Returns ``(frames, reason)``. Empty frames with a reason means "not
    measurable", which callers must distinguish from "measured, nothing drawn".
    """
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        return [], "numpy is not installed"

    if not ass_path.exists():
        return [], f"ASS file not found: {ass_path}"

    total = max(0.1, float(duration))
    rate = min(30.0, max(1.0, max_frames / total))

    def _escape(path: Path) -> str:
        # ffmpeg filter arguments need ':' and '\' escaped on POSIX and Windows.
        text = str(path.resolve())
        return text.replace("\\", "/").replace(":", r"\:").replace("'", r"\'")

    subtitle_filter = f"subtitles='{_escape(ass_path)}'"
    if fonts_dir is not None and fonts_dir.exists():
        subtitle_filter += f":fontsdir='{_escape(fonts_dir)}'"

    command = [
        probe.ffmpeg_binary(),
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"color=c=black:s={width}x{height}:r={rate:.4f}:d={total:.3f}",
        "-vf",
        f"{subtitle_filter},format=gray",
        "-vsync",
        "0",
        "-frames:v",
        str(max_frames),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]
    code, data, stderr = probe._run_binary(command)
    frame_bytes = width * height
    if code != 0 or not data or frame_bytes <= 0:
        detail = stderr.strip()[:300] or f"ffmpeg exited {code}"
        return [], f"libass mask render failed: {detail}"

    count = len(data) // frame_bytes
    if count == 0:
        return [], "libass produced no frames"
    buffer = np.frombuffer(data[: count * frame_bytes], dtype=np.uint8)
    return list(buffer.reshape(count, height, width)), ""


# --- pixel-based caption detection (no ASS available) -----------------------
#
# The exact path above needs the ASS file, which is in scope during a run but is
# not retained for clips that were already published. The golden-master corpus is
# exactly that case, so this fallback measures the captions that are actually in
# the pixels.
#
# It keys on the colour signature every subtitle theme shares. Measured themes:
# hormozi #FFE600, neon_green #22FF33, luxury_gold #FFD700, cyber_cyan #00FFFF, and
# white body text. All of them are high-value and highly saturated, and libass
# draws a black outline around them, so a caption pixel is *bright and saturated*
# and its neighbour is *near black*.
#
# The outline requirement is what keeps this from firing on the source. A
# presenter wearing a red shirt is bright and saturated but is not surrounded by
# black, so it is excluded. That is a heuristic and it is reported as one: the
# caller gets ``confidence="heuristic"`` and is expected to treat the result as
# advisory rather than as a gate.

# The outline requirement is what keeps this from firing on the source, and the
# thresholds are tight for the same reason. A first attempt used
# ``value>=150, saturation>=90, dark<=70`` and the resulting "caption band" was
# the entire frame on all six golden masters: real footage has plenty of bright
# saturated pixels with a dark neighbour somewhere in a 3x3 window.
#
# The measured values below localise correctly. On
# ``clip_1_THE_REAL_FEAR_OF_GOING_BROKE.mp4`` they put the band at y 25.0-50.4% of
# frame height, which matches the 25.5% caption baseline measured by eye from a
# full-resolution frame. On ``clip_2_WHY_DISNEY_PAY_IS_A_TRAP.mp4`` they put it at
# 49.2-74.8%, matching that clip's mid-frame caption.
#
# They are this tight because the theme colours are extreme: #FFE600 has value
# 255 and saturation 230, so a wide margin buys nothing but false positives.
BRIGHT_VALUE_MIN = 210
BRIGHT_SATURATION_MIN = 160
OUTLINE_VALUE_MAX = 50


def saturated_bright_mask(colour: Any) -> Any:
    """Boolean mask of bright, saturated pixels in a BGR frame.

    :func:`saturated_bright_mask` is pure so it can be unit-tested on synthetic
    arrays with known answers.
    """
    import numpy as np

    array = np.asarray(colour)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("expected a BGR or RGB frame with at least 3 channels")
    blue = array[:, :, 0].astype(np.int16)
    green = array[:, :, 1].astype(np.int16)
    red = array[:, :, 2].astype(np.int16)
    peak = np.maximum(np.maximum(red, green), blue)
    trough = np.minimum(np.minimum(red, green), blue)
    saturation = peak - trough
    return (peak >= BRIGHT_VALUE_MIN) & (saturation >= BRIGHT_SATURATION_MIN)


def outlined_text_mask(colour: Any) -> Any:
    """Bright saturated pixels that have a near-black neighbour.

    The outline is the discriminator. Without it, any colourful element of the
    source frame reads as a caption.
    """
    import numpy as np

    bright = saturated_bright_mask(colour)
    if not bright.any():
        return bright
    grey = colour[:, :, :3].mean(axis=2)
    dark = grey <= OUTLINE_VALUE_MAX
    # A 3x3 dilate of the dark mask, minus the pixel itself, gives "has a dark
    # neighbour". cv2 is not required here; a numpy shift is enough and keeps the
    # module importable without OpenCV.
    near_dark = np.zeros_like(dark)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.roll(np.roll(dark, dy, axis=0), dx, axis=1)
            near_dark |= shifted
    return bright & near_dark


def detect_caption_regions(
    path: Path,
    width: int = MASK_WIDTH,
    height: int = MASK_HEIGHT,
    max_samples: int = 40,
) -> Tuple[List[Rect], str]:
    """Locate burned-in caption regions in a rendered clip, without the ASS.

    Returns ``(rects, reason)`` in mask coordinates. ``rects`` is empty when
    nothing confident was found, which the caller must distinguish from failure.

    Returns a single union rect: captions move between layout anchors over the
    course of a clip (upper_center at ~25%, split_divider at ~50%,
    lower_center at ~74%), so the useful region is the band the captions occupy
    anywhere, not any one frame's position.
    """
    import numpy as np

    frames, reason = probe.decode_colour_frames(
        path, width, height, max_frames=max_samples
    )
    if not frames:
        return [], reason or "no frames decoded"

    per_frame: List[Rect] = []
    for frame in frames:
        mask = outlined_text_mask(frame)
        if not mask.any():
            continue
        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        if rows.size == 0 or cols.size == 0:
            continue
        per_frame.append(
            Rect(
                x=int(cols[0]),
                y=int(rows[0]),
                width=int(cols[-1] - cols[0] + 1),
                height=int(rows[-1] - rows[0] + 1),
            )
        )
    if not per_frame:
        return [], "no bright outlined text found in any sampled frame"

    # Union across the clip: the band the captions occupy overall.
    x0 = min(r.x for r in per_frame)
    y0 = min(r.y for r in per_frame)
    x1 = max(r.right for r in per_frame)
    y1 = max(r.bottom for r in per_frame)
    union = Rect(x=x0, y=y0, width=x1 - x0, height=y1 - y0)

    # A union taller than the middle half of the frame means the signature is
    # picking up more than captions. Report it as unusable rather than handing a
    # meaningless exclusion region to the mirror detector, which would then
    # exclude everything and "pass" for the wrong reason.
    if union.height > height * 0.60:
        return [], (
            f"caption candidate band is {100 * union.height / height:.0f}% of frame "
            f"height, which is too broad to be captions; not usable as an exclusion "
            f"region"
        )
    return [union], ""


# The caption signal is a *persistent* one: our captions are on screen in
# nearly every frame, while a show's own graphics (logos, intro banners) appear
# in a few frames and vanish. A row of text must appear in at least this
# fraction of the text-bearing frames -- and at least half as often as the most
# persistent row -- to count as caption pixels.
#
# Evidence: run 37329847616's clip_3_STOP_MAKING_EXCUSES_NOW.mp4 measured 16%
# caption-on-face at t=0 with the caption safely at the bottom of every frame.
# All of the overlap came from the show's own burned-in graphics (the giant
# "THE DAVE RAMSEY SHOW" logo and the "STOP MAKING EXCUSES NOW" banner during
# the first ~4s) which ``outlined_text_mask`` cannot tell from captions. Rows
# carrying captions persist in >80% of frames; the intro graphics in ~15%.
CAPTION_ROW_MIN_FRACTION = 0.20
CAPTION_ROW_RELATIVE_FRACTION = 0.5


def _persistent_caption_rows(
    masks: Sequence[Any], height: int
) -> Optional[Any]:
    """Boolean row mask of the caption band, filtering transient source text.

    ``masks`` are the per-frame ``outlined_text_mask`` outputs. Returns ``None``
    when the filter cannot decide (no text at all, or nothing survives), so the
    caller measures everything rather than silently passing.
    """
    import numpy as np

    active = [mask for mask in masks if bool(np.any(mask))]
    if not active:
        return None
    hits = np.zeros(int(height), dtype=int)
    for mask in active:
        hits += np.asarray(mask).any(axis=1).astype(int)
    fractions = hits / float(len(active))
    threshold = max(
        CAPTION_ROW_MIN_FRACTION,
        CAPTION_ROW_RELATIVE_FRACTION * float(fractions.max()),
    )
    keep = fractions >= threshold
    if not bool(keep.any()):
        return None
    return keep


# Minimum outlined pixels in a 640x360 frame to count as a caption line.
# Prevents specular reflections, studio lights, and background filaments
# (3-15px) on frames without captions from reading as 100% on-face.
MIN_CAPTION_FRAME_PIXELS = 35


def measure_on_face(
    path: Path,
    avoid_rects: Sequence[Rect],
    face_space: Tuple[int, int] = (640, 360),
    max_samples: int = 40,
) -> Tuple[float, float, int]:
    """Fraction of burned-in caption pixels that fall on a face.

    Works from the rendered pixels, so it applies to clips published before the
    ASS file was retained.

    Returns ``(worst_on_face_pct, worst_at_pct, frames_measured)``. This is the
    measurement that the existing QA cannot make: ``editorial_qa`` compares a plan
    rect against plan regions, and the caption that lands on the face is produced
    by the *relocation* that those same regions triggered.

    Only *persistent* text rows count (see ``_persistent_caption_rows``): a
    programme's own on-screen graphics are transient and must not be measured
    as if they were our captions.
    """
    import numpy as np

    if not avoid_rects:
        return 0.0, 0.0, 0

    width, height = face_space
    frames, reason = probe.decode_colour_frames(
        path, width, height, max_frames=max_samples
    )
    if not frames:
        return 0.0, 0.0, 0

    masks = [outlined_text_mask(frame) for frame in frames]
    keep_rows = _persistent_caption_rows(masks, height)

    worst = 0.0
    worst_at = 0.0
    measured = 0
    for index, mask in enumerate(masks):
        if keep_rows is not None:
            mask = mask & keep_rows[:, None]
        total = int(mask.sum())
        if total < MIN_CAPTION_FRAME_PIXELS:
            continue
        measured += 1
        inside = np.zeros(mask.shape, dtype=bool)
        for rect in avoid_rects:
            x0 = max(0, min(width, rect.x))
            y0 = max(0, min(height, rect.y))
            x1 = max(0, min(width, rect.right))
            y1 = max(0, min(height, rect.bottom))
            if x1 > x0 and y1 > y0:
                inside[y0:y1, x0:x1] = True
        overlap = int((mask & inside).sum()) / float(total)
        if overlap > worst:
            worst = overlap
            worst_at = 100.0 * index / max(1, len(frames) - 1)
    return worst * 100.0, worst_at, measured


def analyse(
    ass_path: Path,
    duration: float,
    face_boxes: Optional[Sequence[Rect]] = None,
    dividers: Optional[Sequence[Rect]] = None,
    safe_zone: Optional[SafeZone] = None,
    fonts_dir: Optional[Path] = None,
    width: int = MASK_WIDTH,
    height: int = MASK_HEIGHT,
) -> CaptionReport:
    """Measure where the burned-in captions actually land.

    ``face_boxes`` are face rectangles **in mask coordinates**. Callers that have
    them in output coordinates must scale; passing unscaled 1080x1920 boxes into a
    270x480 mask would silently report zero overlap, which is the failure mode
    this whole module exists to prevent.
    """
    zone = (safe_zone or SafeZone()).scaled(width, height)
    report = CaptionReport(ass_path=str(ass_path), ok=True)

    frames, reason = text_mask(
        ass_path, duration, width=width, height=height, fonts_dir=fonts_dir
    )
    if not frames:
        report.reason = reason or "no mask frames"
        return report
    report.measured = True

    drawn = 0
    worst_face = 0.0
    worst_face_at = 0.0
    worst_outside = 0.0
    worst_divider = 0.0
    widest = 0.0
    band: Optional[Rect] = None

    for index, frame in enumerate(frames):
        if mask_pixels(frame) == 0:
            continue
        drawn += 1
        progress = 100.0 * index / max(1, len(frames) - 1)

        if face_boxes:
            overlap = fraction_inside(frame, face_boxes)
            if overlap > worst_face:
                worst_face = overlap
                worst_face_at = progress

        outside = fraction_outside_safe_zone(frame, zone)
        worst_outside = max(worst_outside, outside)

        if dividers:
            worst_divider = max(worst_divider, fraction_inside(frame, dividers))

        widest = max(widest, widest_row_frac(frame))
        current = mask_bounds(frame)
        if current is not None:
            # Union across the clip: the band the captions occupy overall.
            if band is None:
                band = current
            else:
                x0 = min(band.x, current.x)
                y0 = min(band.y, current.y)
                x1 = max(band.right, current.right)
                y1 = max(band.bottom, current.bottom)
                band = Rect(x=x0, y=y0, width=x1 - x0, height=y1 - y0)

    report.samples = drawn
    report.on_face_pct = worst_face * 100.0
    report.on_face_at = worst_face_at
    report.outside_safe_zone_pct = worst_outside * 100.0
    report.on_divider_pct = worst_divider * 100.0
    report.max_width_frac = widest
    report.caption_band = band.as_dict() if band is not None else None

    if drawn == 0:
        report.reason = (
            "libass rendered the ASS but no glyph pixels appeared; the caption "
            "file may be empty, mis-timed, or using a font that did not resolve"
        )
        return report

    problems: List[str] = []
    if worst_face > 0.02:
        problems.append(
            f"{worst_face * 100:.1f}% of caption pixels sit on a face "
            f"(worst at {worst_face_at:.0f}% through the clip)"
        )
    if widest > 1.0:
        problems.append(
            f"the widest caption row spans {widest:.2f}x the frame width, so it "
            f"crosses the live content edge"
        )
    if worst_divider > 0.30:
        problems.append(
            f"{worst_divider * 100:.0f}% of caption pixels sit on a pane divider"
        )

    if problems:
        report.ok = False
        report.reason = "; ".join(problems)
    else:
        report.reason = (
            f"captions clear of faces and within {widest:.2f}x frame width "
            f"across {drawn} sampled frames"
        )
    return report
