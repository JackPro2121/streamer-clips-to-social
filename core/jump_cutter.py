"""P1 Jump-Cut Engine: dead-air removal from word timestamps.

Firecrawl research (2026-10-05): removing dead air and filler pauses is the
industry-standard "feels edited" upgrade (TimeBolt, SavvyCut, Jump Cutter
ULTRA all sell exactly this), and 50-60% of short-form viewers drop off in the
first three seconds (KometMedia pacing guide), so pacing is retention. Clips
that keep every natural pause read as "raw excerpt", not "edited".

This module is the pure engine: it turns word timestamps into a keep-segment
list and a compacted timeline (old clip time -> new timeline). The renderer
consumes the plan; the caption layer remaps word times through it. No media
access, so every rule is unit-tested with constructed word lists.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence, Tuple, Union

# Gaps shorter than this are natural speech rhythm, not dead air.
MIN_DEAD_AIR_S = 0.65
# Breath kept on each side of a speech span so cuts do not clip onsets.
KEEP_PADDING_S = 0.08
# The pixel verdict's contract floor (30-55s). A cut plan that would push the
# clip below the floor is refused outright rather than shipping a short clip.
MIN_RESULT_DURATION_S = 30.0


@dataclass
class CutPlan:
    """Keep-segments in original clip time, plus the removed total."""

    keep_segments: List[Tuple[float, float]] = field(default_factory=list)
    original_duration_s: float = 0.0
    removed_s: float = 0.0
    applied: bool = False

    @property
    def duration_s(self) -> float:
        return sum(end - start for start, end in self.keep_segments)

    def remap(self, time_s: float) -> float:
        """Original clip time -> compacted timeline, snapping removed gaps.

        Times inside a removed gap snap to the following segment's start;
        times past the last segment clamp to the compacted end.
        """
        kept = 0.0
        for start, end in self.keep_segments:
            if time_s < start:
                return kept
            if time_s <= end:
                return kept + (time_s - start)
            kept += end - start
        return kept


def _speech_spans(
    words: Sequence[Tuple[float, float]], min_gap_s: float = MIN_DEAD_AIR_S
) -> List[Tuple[float, float]]:
    """Merge word (start, end) pairs into contiguous speech spans.

    A gap shorter than ``min_gap_s`` is speech rhythm and never cut, so it is
    also the merge threshold.
    """
    spans: List[Tuple[float, float]] = []
    for start, end in sorted(words):
        if end <= start:
            continue
        if spans and start - spans[-1][1] < min_gap_s:
            spans[-1] = (spans[-1][0], max(spans[-1][1], end))
        else:
            spans.append((float(start), float(end)))
    return spans


def build_cut_plan(
    words: Sequence[Tuple[float, float]],
    clip_duration_s: float,
    min_gap_s: float = MIN_DEAD_AIR_S,
    padding_s: float = KEEP_PADDING_S,
    min_duration_s: float = MIN_RESULT_DURATION_S,
) -> CutPlan:
    """Plan the dead-air cuts for one rendered clip's word timings.

    ``words`` are ``(start_s, end_s)`` pairs in clip-relative time. The plan is
    an identity (one keep-segment, ``applied=False``) whenever there is nothing
    worth cutting or a guard refuses: an empty word list, no gap at least
    ``min_gap_s``, or a result that would fall below ``min_duration_s``.
    """
    identity = CutPlan(
        keep_segments=[(0.0, float(clip_duration_s))],
        original_duration_s=float(clip_duration_s),
        removed_s=0.0,
        applied=False,
    )
    if not words or clip_duration_s <= 0.0:
        return identity

    spans = _speech_spans(words, min_gap_s)
    if not spans:
        return identity
    padded = [
        (max(0.0, start - padding_s), min(float(clip_duration_s), end + padding_s))
        for start, end in spans
    ]
    merged: List[Tuple[float, float]] = []
    for start, end in padded:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    # Keep the head/tail when the first/last span essentially starts/ends with
    # the clip: only a real dead-air run at the edge is worth removing.
    if merged[0][0] <= min_gap_s:
        merged[0] = (0.0, merged[0][1])
    if float(clip_duration_s) - merged[-1][1] <= min_gap_s:
        merged[-1] = (merged[-1][0], float(clip_duration_s))

    removed = float(clip_duration_s) - sum(end - start for start, end in merged)
    if removed <= 0.0:
        return identity
    if float(clip_duration_s) - removed < min_duration_s:
        return identity
    return CutPlan(
        keep_segments=merged,
        original_duration_s=float(clip_duration_s),
        removed_s=removed,
        applied=True,
    )


def build_concat_filtergraph(plan: CutPlan, source_offset_s: float = 0.0) -> str:
    """A ``filter_complex`` graph that compacts input 0 into ``[outv][outa]``.

    One trim/atrim per keep-segment, joined by ``concat``: this is the ffmpeg
    standard for jump cuts (research: TimeBolt/SavvyCut). Raises on an empty
    plan so a caller can never render a silent black clip by accident.

    ``source_offset_s`` shifts the keep-segments into the source file's own
    timeline: the render path's plan is clip-relative (0 = clip start) while
    the downloaded segment file begins at its own start, so the adoption
    passes the segment-relative clip window here.
    """
    if not plan.keep_segments:
        raise ValueError("cannot build a concat graph from an empty cut plan")
    video_parts: List[str] = []
    audio_parts: List[str] = []
    pairs: List[str] = []
    for index, (start, end) in enumerate(plan.keep_segments):
        file_start = start + float(source_offset_s)
        file_end = end + float(source_offset_s)
        video_parts.append(
            f"[0:v]trim=start={file_start:.3f}:end={file_end:.3f},setpts=PTS-STARTPTS[v{index}]"
        )
        audio_parts.append(
            f"[0:a]atrim=start={file_start:.3f}:end={file_end:.3f},asetpts=PTS-STARTPTS[a{index}]"
        )
        pairs.append(f"[v{index}][a{index}]")
    concat = (
        f"{''.join(pairs)}concat=n={len(plan.keep_segments)}:v=1:a=1[outv][outa]"
    )
    return ";".join(video_parts + audio_parts + [concat])



def _probe_duration(path):
    """Return media duration in seconds using ffprobe, or None on failure."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(path),
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except (OSError, TypeError, ValueError, subprocess.SubprocessError):
        pass
    return None


def _has_audio_stream(path):
    """Return True if the media file contains at least one audio stream."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "a",
                "-show_entries", "stream=codec_type",
                "-of", "json", str(path),
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout or "{}")
            return bool(data.get("streams"))
    except (OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError):
        pass
    return False


def compact_media(
    source_path: Union[str, Path],
    plan: CutPlan,
    destination_path: Union[str, Path],
    crf: int = 18,
    preset: str = "fast",
    source_offset_s: float = 0.0,
) -> bool:
    """Physically apply the cut plan to a media file.

    Returns False when the plan is not applied or the encode fails, so the
    caller keeps the original file rather than shipping a broken one. The
    renderer integration re-encodes anyway, so the intermediate uses a
    quality-preserving CRF rather than the final delivery settings.
    """
    if not plan.applied:
        return False
    source = Path(source_path)
    destination = Path(destination_path)
    graph = build_concat_filtergraph(plan, source_offset_s=source_offset_s)
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(source),
            "-filter_complex", graph,
            "-map", "[outv]", "-map", "[outa]",
            "-threads", "2",
            "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-movflags", "+faststart",
            str(destination),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=900,
    )
    ok = (
        result.returncode == 0
        and destination.exists()
        and destination.stat().st_size > 0
        and _has_audio_stream(destination)
    )
    if ok:
        actual = _probe_duration(destination)
        plan_duration = float(plan.duration_s)
        if actual is None or abs(actual - plan_duration) > 0.25:
            try:
                destination.unlink()
            except OSError:
                pass
            return False
    return ok


def prepare_compacted_clip(
    source_path: Union[str, Path],
    words: Sequence[Tuple[float, float]],
    clip_duration_s: float,
    work_dir: Union[str, Path],
    min_duration_s: float = MIN_RESULT_DURATION_S,
    source_offset_s: float = 0.0,
) -> "tuple[Path, List[Tuple[float, float]], float] | None":
    """Everything the render path needs to adopt a jump-cut in one call.

    Returns ``(compacted_path, remapped_words, new_duration_s)`` or ``None``
    when no cut is warranted (or the encode fails), in which case the caller
    keeps the original source, words and duration untouched. Word times are
    remapped with ``CutPlan.remap`` - the same function the end-to-end burst
    test proves against real media. The caller renders the compacted file from
    ``start_time=0`` since it begins at the clip's first kept segment.
    """
    plan = build_cut_plan(words, clip_duration_s, min_duration_s=min_duration_s)
    if not plan.applied:
        return None
    destination = (
        Path(work_dir)
        / f"{Path(source_path).stem}_compacted_{int(plan.removed_s * 1000)}ms.mp4"
    )
    if not compact_media(source_path, plan, destination, source_offset_s=source_offset_s):
        return None
    remapped = [(plan.remap(start), plan.remap(end)) for start, end in words]
    if not _has_audio_stream(destination):
        try:
            destination.unlink()
        except OSError:
            pass
        return None
    return destination, remapped, plan.duration_s
