"""ASD-Lite: which visible face is speaking, per time window.

Evidence (run 37347481817, clip_2 "THE LONELINESS EPIDEMIC..."): the clip
showed one face for 498/499 frames whose mouth-region motion correlated with
the voice at only +0.09, and a +-5s lag scan peaked at |0.17| with no
alignment -- so the picture was not time-shifted, the visible face simply was
not the speaker. The show is multi-speaker (8 faces detected, zero cuts) and
the composition picks crops on face geometry, which is speech-blind.

This module is the speech signal the composition lacks, CPU-only (YuNet faces,
mouth-region frame difference, audio RMS envelope). The decision logic is pure
(``scores_from_series`` / ``windowed_speakers``) so it is unit-testable with
synthetic series where the talker is known by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

WINDOW_S = 1.0
FRAME_WIDTH, FRAME_HEIGHT = 480, 270
# A face must clear both: some correlation with the voice, and real mouth
# movement (a still photo or a listener cannot win by accident).
SPEAKER_MIN_CORR = 0.15
SPEAKER_MIN_MOTION = 0.8


def pearson(a: Sequence[float], b: Sequence[float]) -> float:
    import numpy as np

    n = min(len(a), len(b))
    if n < 8:
        return 0.0
    first = np.asarray(a[:n], dtype=np.float64)
    second = np.asarray(b[:n], dtype=np.float64)
    first = first - first.mean()
    second = second - second.mean()
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.dot(first, second) / denominator) if denominator > 1e-12 else 0.0


def scores_from_series(
    face_motion: Sequence[Sequence[float]], envelope: Sequence[float]
) -> List[float]:
    """Correlation of each face's mouth motion with the voice envelope."""
    return [pearson(motion, envelope) for motion in face_motion]


def pick_speaker(face_motion: Sequence[Sequence[float]], envelope: Sequence[float]) -> int:
    """Index of the speaking face, or -1 when no face clears the thresholds."""
    import numpy as np

    best_index, best_score = -1, 0.0
    for index, motion in enumerate(face_motion):
        if len(motion) < 8:
            continue
        valid_motion = [m for m in motion if m > 0.0]
        motion_for_mean = valid_motion if valid_motion else motion
        if float(np.mean(motion_for_mean)) < SPEAKER_MIN_MOTION:
            continue
        score = pearson(motion, envelope)
        if score > best_score:
            best_index, best_score = index, score
    if best_score < SPEAKER_MIN_CORR:
        return -1
    return best_index


def windowed_speakers(
    face_motion: Sequence[Sequence[float]],
    envelope: Sequence[float],
    fps: float,
    window_s: float = WINDOW_S,
) -> List[Tuple[float, int, float]]:
    """Per-window ``(start_s, speaker_index, score)``; -1 when no talker.

    This is what the composition should consult: the speaker can change between
    windows, and a window with no confident talker (music, laughter, both
    silent) must not force a switch.
    """
    hop = max(1, int(round(fps * window_s)))
    max_motion_len = max((len(motion) for motion in face_motion), default=0)
    frame_count = min(max_motion_len, len(envelope)) if max_motion_len > 0 else 0
    results: List[Tuple[float, int, float]] = []
    for start in range(0, frame_count - hop + 1, hop):
        window_motion: List[List[float]] = []
        for motion in face_motion:
            m_slice = list(motion[start:start + hop]) if len(motion) >= start else []
            if len(m_slice) < hop:
                m_slice = m_slice + [0.0] * (hop - len(m_slice))
            window_motion.append(m_slice)
        window_envelope = list(envelope[start:start + hop])
        index = pick_speaker(window_motion, window_envelope)
        score = (
            pearson(window_motion[index], window_envelope) if index >= 0 else 0.0
        )
        results.append((start / fps, index, score))
    return results


def _extract_frames(clip: Path, fps: int) -> List[Any]:
    import numpy as np
    import subprocess

    result = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(clip),
            "-vf", f"fps={fps},scale={FRAME_WIDTH}:{FRAME_HEIGHT}",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
        ],
        capture_output=True,
    )
    raw = result.stdout
    size = FRAME_WIDTH * FRAME_HEIGHT * 3
    return [
        np.frombuffer(raw[i * size:(i + 1) * size], dtype=np.uint8).reshape(
            FRAME_HEIGHT, FRAME_WIDTH, 3
        )
        for i in range(len(raw) // size)
    ]


def _audio_envelope(clip: Path, fps: int) -> List[float]:
    import numpy as np

    from core.verification import av_sync

    samples = av_sync._decode_mono(clip, 16000)
    if samples is None or samples.size == 0:
        return []
    hop = max(1, int(16000 / fps))
    return [
        float(np.sqrt(np.mean(samples[i:i + hop] ** 2)))
        for i in range(0, max(1, samples.size - hop), hop)
    ]


def _mouth_patch(grey: Any, box: Any) -> Any:
    import cv2
    import numpy as np

    x0 = int(max(0, box.x))
    y0 = int(max(0, box.y + box.height * 0.55))
    x1 = int(min(grey.shape[1], box.x + box.width))
    y1 = int(min(grey.shape[0], box.y + box.height))
    patch = grey[y0:y1, x0:x1]
    if patch.size == 0:
        return np.zeros((10, 20), dtype=np.float32)
    return cv2.resize(patch, (20, 10)).astype(np.float32)


@dataclass
class Track:
    """One tracked face: mouth-motion series and its per-frame centre x."""

    motion: List[float] = field(default_factory=list)
    x: List[float] = field(default_factory=list)


def track_mouth_motion_with_positions(clip: Path, fps: int = 10) -> List[Track]:
    """Per-face tracks (motion + positions) strictly aligned to the clip timeline."""
    import numpy as np

    from core.verification import faces as faces_mod

    frames = _extract_frames(clip, fps)
    detector, _reason = faces_mod._load_yunet()
    if not frames or detector is None:
        return []
    total_frames = len(frames)
    tracks: List[dict] = []
    for frame_idx, frame in enumerate(frames):
        grey = frame.mean(axis=2)
        boxes = faces_mod.detect_in_frame(frame, detector) or []
        used: set[int] = set()
        for track in tracks:
            last = track["last_box"]
            best, best_dist = None, 1e9
            for box_index, box in enumerate(boxes):
                if box_index in used:
                    continue
                dist = abs((box.x + box.width / 2) - (last.x + last.width / 2)) + abs(
                    (box.y + box.height / 2) - (last.y + last.height / 2)
                )
                if dist < best_dist:
                    best, best_dist = box_index, dist
            if best is not None and best_dist < 0.25 * FRAME_WIDTH:
                used.add(best)
                box = boxes[best]
                patch = _mouth_patch(grey, box)
                # Compute difference across consecutive frames; 0.0 on re-entry
                if track["last_seen_frame"] == frame_idx - 1 and track["prev_patch"] is not None:
                    diff = float(np.mean(np.abs(patch - track["prev_patch"])))
                else:
                    diff = 0.0
                track["motion"][frame_idx] = diff
                track["x"][frame_idx] = float(box.x + box.width / 2)
                track["present"][frame_idx] = True
                track["prev_patch"] = patch
                track["last_box"] = box
                track["last_seen_frame"] = frame_idx
                track["total_seen"] += 1
        for box_index, box in enumerate(boxes):
            if box_index not in used and len(tracks) < 8:
                patch = _mouth_patch(grey, box)
                new_track = {
                    "motion": [0.0] * total_frames,
                    "x": [0.0] * total_frames,
                    "present": [False] * total_frames,
                    "last_box": box,
                    "prev_patch": patch,
                    "last_seen_frame": frame_idx,
                    "total_seen": 1,
                }
                new_track["x"][frame_idx] = float(box.x + box.width / 2)
                new_track["present"][frame_idx] = True
                tracks.append(new_track)

    # Filter out spurious 1-2 frame false positive detections
    valid_tracks = [t for t in tracks if t["total_seen"] >= 4]
    if not valid_tracks and tracks:
        valid_tracks = [max(tracks, key=lambda t: t["total_seen"])]

    return [Track(motion=track["motion"], x=track["x"]) for track in valid_tracks]


def track_mouth_motion(clip: Path, fps: int = 10) -> List[List[float]]:
    """Per-face-track mouth-motion series (frames-indexed, greedy nearest)."""
    return [track.motion for track in track_mouth_motion_with_positions(clip, fps)]


def shot_speaker_targets(
    tracks: Sequence[Track],
    envelope: Sequence[float],
    shots: Sequence[Tuple[float, float]],
    face_boxes_per_shot: Sequence[Sequence[Tuple[float, float, float, float]]],
    source_size: Tuple[int, int],
    fps: float,
    analysis_size: Tuple[int, int] = (FRAME_WIDTH, FRAME_HEIGHT),
    window_s: float = 2.0,
    min_score: float = SPEAKER_MIN_CORR,
) -> List[Optional[float]]:
    """New crop centre x (source pixels) per shot, or None to leave it alone.

    For each shot span the talker is chosen per window (``windowed_speakers``),
    votes are confidence-weighted across the span, and the winning track's mean
    position is mapped to the nearest shot face box. ``None`` is returned when
    no confident talker exists in the span or the track cannot be mapped; the
    caller then keeps the framing's geometric choice. This function is pure so
    the mapping is unit-tested with synthetic tracks where the talker and its
    position are known by construction.
    """
    if not tracks or not envelope:
        return [None] * len(shots)
    windows = windowed_speakers(
        [track.motion for track in tracks], envelope, fps, window_s
    )
    analysis_width = float(analysis_size[0])
    results: List[Optional[float]] = []
    for shot_index, (start_s, end_s) in enumerate(shots):
        votes = [0.0] * len(tracks)
        for window_start, track_index, score in windows:
            if track_index < 0 or score < min_score:
                continue
            if start_s <= window_start < end_s:
                votes[track_index] += score
        winner = max(range(len(votes)), key=lambda index: votes[index])
        if votes[winner] <= 0.0:
            results.append(None)
            continue
        positions = tracks[winner].x
        if not positions:
            results.append(None)
            continue
        start_f = max(0, int(round(start_s * fps)))
        end_f = min(len(positions), int(round(end_s * fps)))
        shot_xs = [pos for pos in positions[start_f:end_f] if pos > 0.0]
        if not shot_xs:
            shot_xs = [pos for pos in positions if pos > 0.0]
        if not shot_xs:
            results.append(None)
            continue
        mean_x_analysis = sum(shot_xs) / len(shot_xs)
        mean_x_source = mean_x_analysis * source_size[0] / analysis_width
        boxes = (
            face_boxes_per_shot[shot_index]
            if shot_index < len(face_boxes_per_shot)
            else []
        )
        if not boxes:
            # The planner's face boxes are often absent (run 37466588808 had
            # zero of them on every shot, which silently disabled this whole
            # fix). The talker track's own position IS a face centre - use it
            # directly rather than giving up.
            results.append(mean_x_source)
            continue
        best_centre = min(
            boxes,
            key=lambda box: abs((box[0] + box[2] / 2.0) - mean_x_source),
        )
        results.append(best_centre[0] + best_centre[2] / 2.0)
    return results


def analyse(clip: Path, fps: int = 10, window_s: float = WINDOW_S) -> List[Tuple[float, int, float]]:
    """Per-window speaker index for a (source or rendered) clip."""
    motion = track_mouth_motion(clip, fps=fps)
    if not motion:
        return []
    envelope = _audio_envelope(clip, fps)
    if not envelope:
        return []
    return windowed_speakers(motion, envelope, float(fps), window_s)
