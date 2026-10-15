"""Streamer High-Retention Editing Enhancer (V2).

Implements algorithm-tailored video editing dynamics designed for TikTok,
Reels, and YouTube Shorts:

1. Velocity Hook (First 0.35s):
   - Injects an instant 1.2x snap-zoom at t=0.0s with a synced Whoosh sound.
   - Breaks viewer scroll momentum in the first critical 1.5 seconds.

2. Pattern Interrupt Generator (The 4-Second Rule):
   - Short-form viewers drop off when visuals remain static for >5 seconds.
   - If energy peaks have a gap exceeding max_static_gap_s (default 6.0s),
     injects micro-zooms (0.6s) to re-engage viewer attention.

3. SFX Synchronizer:
   - Harmonizes and bounds SFX cues (whoosh, pop, ding) within clip duration.
"""

from typing import List, Tuple, Optional
from core.transcriber import TranscriptSegment
from streamers.spec import (
    STREAMER_PUNCH_ZOOM_MAX_SEGMENTS,
)


def build_streamer_retention_plan(
    clip_duration: float,
    peak_intervals: List[Tuple[float, float]],
    sfx_cues: List[Tuple[float, str]],
    segments: Optional[List[TranscriptSegment]] = None,
    inject_velocity_hook: bool = True,
    max_static_gap_s: float = 6.5,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, str]]]:
    """Generates an enhanced punch-zoom and SFX cue timeline tailored for viral retention.

    Args:
        clip_duration: Total length of the clip in seconds.
        peak_intervals: Audio energy intervals detected by analyze_audio_energy().
        sfx_cues: SFX cues already planned by viral detector or pacing.
        segments: Optional transcript segments for word-aligned pattern interrupts.
        inject_velocity_hook: Whether to force an instant snap-zoom at t=0.
        max_static_gap_s: Maximum seconds allowed between visual zoom interrupts.

    Returns:
        (enhanced_zoom_intervals, enhanced_sfx_cues)
    """
    enhanced_zooms: List[Tuple[float, float]] = []
    enhanced_sfx: List[Tuple[float, str]] = list(sfx_cues or [])

    # 1. First 0.35s Velocity Hook Snap-Zoom
    if inject_velocity_hook and clip_duration >= 2.0:
        enhanced_zooms.append((0.0, 0.35))
        # Ensure a whoosh SFX fires at t=0.0 if not already present near start
        has_start_whoosh = any(cue[0] <= 0.4 and cue[1] == "whoosh" for cue in enhanced_sfx)
        if not has_start_whoosh:
            enhanced_sfx.append((0.0, "whoosh"))

    # 2. Add existing decibel scream peaks
    for start, end in (peak_intervals or []):
        if 0.0 <= start < clip_duration:
            clamped_end = min(clip_duration, max(start + 0.4, end))
            # Avoid duplicate overlap with velocity hook
            if start < 0.4:
                continue
            enhanced_zooms.append((start, clamped_end))

    # Sort zooms chronologically
    enhanced_zooms.sort(key=lambda z: z[0])

    # 3. Pattern Interrupts (Prevent visual stagnation if gap > max_static_gap_s)
    filled_zooms: List[Tuple[float, float]] = []
    last_end = 0.0

    for z_start, z_end in enhanced_zooms:
        gap = z_start - last_end
        if gap > max_static_gap_s and (clip_duration - z_start) > 2.0:
            # Inject a 0.6s micro-zoom halfway through the lull
            mid_t = round(last_end + gap / 2.0, 2)
            filled_zooms.append((mid_t, round(mid_t + 0.6, 2)))
        filled_zooms.append((z_start, z_end))
        last_end = z_end

    # Check trailing lull before end of clip
    trailing_gap = clip_duration - last_end
    if trailing_gap > max_static_gap_s + 1.0 and clip_duration > 10.0:
        mid_t = round(last_end + trailing_gap / 2.0, 2)
        filled_zooms.append((mid_t, round(mid_t + 0.6, 2)))

    # Sort and clamp to max allowed zoom segments in spec
    filled_zooms.sort(key=lambda z: z[0])
    max_segs = max(3, STREAMER_PUNCH_ZOOM_MAX_SEGMENTS + 2)
    final_zooms = filled_zooms[:max_segs]

    # Clean and sort SFX cues within clip duration
    valid_sfx = [
        (round(cue_t, 2), s_type)
        for cue_t, s_type in enhanced_sfx
        if 0.0 <= cue_t < clip_duration
    ]
    valid_sfx.sort(key=lambda s: s[0])

    return final_zooms, valid_sfx
