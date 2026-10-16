"""Boundary repair for rendered V2 clips.

Rendered clips inherit whatever the source segment ended with. The first live
V2 run shipped a clip whose final ~0.47s was the stream's own ending: a frozen
frame fading to black (verdict: ``freeze_unterminated`` 37.533→38.0, black
37.533→37.967, final frame 98.7 % near-black). The gate was right to block it —
a black flash at the end is visible to a viewer — but discovering that after a
full render, gate cycle and upload attempt is the expensive way.

``trim_dead_boundaries()`` measures near-black frames from both ends of the
rendered file and re-encodes without them, when what remains still satisfies the
duration contract. It never touches the middle of a clip, it is capped at
``MAX_BOUNDARY_TRIM_S`` per side, and it leaves the file byte-identical when
anything about the measurement is uncertain.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from core.config import AUDIO_BITRATE, IS_CI, VIDEO_CRF
from streamers.spec import (
    STREAMER_DURATION_QUANTIZATION_MARGIN_S,
    STREAMER_MIN_CLIP_DURATION_S,
)

# A frame whose mean luma (0-255) is below this is dead: the encoder's black
# frame at the end of a segment download, or the stream's own cut to black.
# gate._face_stability_check uses the same 12.0 to mean "too dark to measure".
DEAD_FRAME_MEAN_LUMA = 12.0

# A dead run longer than this is content (a genuinely dark scene that reaches a
# clip boundary), not a boundary artifact — never trim more than this per side.
MAX_BOUNDARY_TRIM_S = 3.0

# The scan is sequential and bounded. V2 clips are capped at 45s in selection,
# so anything past this is not a clip this pipeline produced and the repair
# declines to touch it.
MAX_SCAN_S = 120.0


@dataclass
class BoundaryTrim:
    """Outcome of one boundary-repair attempt (``changed`` is the headline)."""

    path: Path
    duration_before_s: float
    duration_after_s: float
    head_trim_s: float = 0.0
    tail_trim_s: float = 0.0
    skipped_reason: Optional[str] = None

    @property
    def changed(self) -> bool:
        return self.head_trim_s > 0.0 or self.tail_trim_s > 0.0


def _frame_means(cap, cv2, max_frames: int) -> list:
    """Mean luma of every frame, in order, up to ``max_frames``."""
    means = []
    while len(means) < max_frames:
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            break
        means.append(float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()))
    return means


def _dead_runs(means):
    """(head_frames, tail_frames) of consecutive dead frames."""
    head = 0
    while head < len(means) and means[head] < DEAD_FRAME_MEAN_LUMA:
        head += 1
    tail = 0
    while tail < len(means) - head and means[len(means) - 1 - tail] < DEAD_FRAME_MEAN_LUMA:
        tail += 1
    return head, tail


def trim_dead_boundaries(
    path: Path,
    *,
    min_duration_s: float = STREAMER_MIN_CLIP_DURATION_S,
    max_trim_s: float = MAX_BOUNDARY_TRIM_S,
    black_mean_luma: float = DEAD_FRAME_MEAN_LUMA,
) -> BoundaryTrim:
    """Remove near-black head/tail frames from a rendered clip, in place.

    Returns a ``BoundaryTrim`` describing what happened. On any uncertainty the
    original file is left exactly as it was and ``skipped_reason`` explains why.
    """
    path = Path(path)
    try:
        import cv2
    except Exception as exc:  # pragma: no cover - cv2 is present in CI
        return BoundaryTrim(path, 0.0, 0.0, skipped_reason=f"cv2 import failed: {exc}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return BoundaryTrim(path, 0.0, 0.0, skipped_reason=f"could not open {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0.0:
            return BoundaryTrim(path, 0.0, 0.0, skipped_reason="unreadable fps")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            return BoundaryTrim(path, 0.0, 0.0, skipped_reason="unreadable frame count")
        if total_frames / fps > MAX_SCAN_S:
            return BoundaryTrim(
                path,
                total_frames / fps,
                total_frames / fps,
                skipped_reason=f"clip longer than the {MAX_SCAN_S:.0f}s scan budget",
            )
        means = _frame_means(cap, cv2, max_frames=int(MAX_SCAN_S * fps) + 1)
    finally:
        cap.release()

    if not means:
        return BoundaryTrim(path, 0.0, 0.0, skipped_reason="no decodable frames")

    duration = len(means) / fps
    raw_head, raw_tail = _dead_runs(means)
    if raw_head + raw_tail >= len(means):
        return BoundaryTrim(
            path,
            duration,
            duration,
            skipped_reason="every scanned frame is dead — nothing to keep",
        )

    cap_frames = max(1, int(round(max_trim_s * fps)))
    head = min(raw_head, cap_frames)
    tail = min(raw_tail, cap_frames)
    if head == 0 and tail == 0:
        return BoundaryTrim(path, duration, duration)

    head_s = head / fps
    tail_s = tail / fps
    keep_s = duration - head_s - tail_s
    if keep_s < min_duration_s + STREAMER_DURATION_QUANTIZATION_MARGIN_S:
        return BoundaryTrim(
            path,
            duration,
            duration,
            skipped_reason=(
                f"trimming {head_s + tail_s:.2f}s would break the "
                f"{min_duration_s:.0f}s duration floor"
            ),
        )

    tmp = path.with_name(f"{path.stem}.trim{path.suffix or '.mp4'}")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{head_s:.3f}",
        "-i", str(path),
        "-t", f"{keep_s:.3f}",
        "-c:v", "libx264",
        "-crf", str(VIDEO_CRF),
        "-preset", "faster" if IS_CI else "medium",
        "-pix_fmt", "yuv420p",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-colorspace", "bt709",
        "-c:a", "aac",
        "-b:a", AUDIO_BITRATE,
        "-ar", "48000",
        "-movflags", "+faststart",
        str(tmp),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except Exception as exc:  # pragma: no cover - environment failure
        return BoundaryTrim(path, duration, duration, skipped_reason=f"ffmpeg failed to start: {exc}")

    if proc.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        return BoundaryTrim(
            path,
            duration,
            duration,
            skipped_reason=f"ffmpeg trim failed: {proc.stderr[-200:]}",
        )

    os.replace(tmp, path)
    return BoundaryTrim(
        path,
        duration,
        keep_s,
        head_trim_s=head_s,
        tail_trim_s=tail_s,
    )
