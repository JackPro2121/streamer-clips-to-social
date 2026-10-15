"""Streamer Audio Energy & Scream Peak Detector.

High-performance audio waveform analyzer for detecting:
- Screams, desk slams, and high-energy streamer reactions (e.g. Kai Cenat, IShowSpeed)
- RMS decibel energy spikes above median baseline
- Timestamp windows for dynamic punch-zooms and viral moment candidate scoring
- Zero heavy dependencies (uses FFmpeg 8.x + NumPy PCM streaming)
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class AudioEnergyPeak:
    time_s: float
    duration_s: float
    peak_db: float
    delta_db: float
    is_scream: bool


def _unavailable(reason: str = "") -> Dict[str, Any]:
    """Result shape for "the waveform was never analysed".

    ``available=False`` exists so the caller can tell "no peaks found" apart from
    "analysis could not run". Previously both returned an empty ``peak_intervals``,
    so a failed extraction silently disabled punch-zoom with no signal anywhere.
    """
    if reason:
        print(f"[!] Audio analysis unavailable: {reason}")
    return {
        "available": False,
        "median_db": -60.0,
        "max_db": -60.0,
        "peaks": [],
        "peak_intervals": [],
    }


def analyze_audio_energy(
    media_path: Path,
    sample_rate: int = 16000,
    window_sec: float = 0.5,
    hop_sec: float = 0.25,
    scream_delta_db: float = 6.5
) -> Dict[str, Any]:
    """Analyzes audio waveform for instantaneous loudness spikes and scream moments.

    Returns dict with:
        available: bool                            # False when extraction failed
        median_db: float
        max_db: float
        peaks: List[AudioEnergyPeak]
        peak_intervals: List[Tuple[float, float]]  # (start, end) suitable for punch zooms
    """
    if not media_path.exists():
        return _unavailable(f"{media_path.name} does not exist")

    cmd = [
        "ffmpeg", "-y", "-i", str(media_path),
        "-f", "f32le", "-ac", "1", "-ar", str(sample_rate),
        "-vn", "-"
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        raw_audio, _ = proc.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        # Kill AND reap. An orphaned ffmpeg keeps decoding and holds the source
        # file open, which blocks its deletion (notably on Windows).
        proc.kill()
        try:
            proc.communicate(timeout=10)
        except Exception:  # pragma: no cover - best-effort reap
            pass
        return _unavailable(f"ffmpeg timed out after 60s on {media_path.name}")
    except Exception as exc:
        proc.kill()
        return _unavailable(f"ffmpeg failed on {media_path.name}: {exc}")

    if not raw_audio:
        return _unavailable(f"ffmpeg produced no PCM data for {media_path.name}")

    samples = np.frombuffer(raw_audio, dtype=np.float32)
    total_samples = len(samples)
    if total_samples == 0:
        return _unavailable(f"decoded zero samples for {media_path.name}")

    win_samples = int(sample_rate * window_sec)
    hop_samples = int(sample_rate * hop_sec)

    rms_values: List[float] = []
    times: List[float] = []

    for i in range(0, total_samples - win_samples + 1, hop_samples):
        chunk = samples[i:i + win_samples]
        rms = np.sqrt(np.mean(chunk ** 2) + 1e-12)
        db = float(20.0 * np.log10(rms))
        rms_values.append(db)
        times.append(i / sample_rate)

    if not rms_values:
        return _unavailable(f"clip shorter than one analysis window ({window_sec}s)")

    rms_arr = np.array(rms_values)
    median_db = float(np.median(rms_arr))
    max_db = float(np.max(rms_arr))

    peaks: List[AudioEnergyPeak] = []
    threshold = median_db + scream_delta_db

    # Find peaks above threshold
    for t, db in zip(times, rms_values):
        if db >= threshold:
            delta = db - median_db
            is_scream = delta >= (scream_delta_db + 3.0) or db >= -12.0
            peaks.append(AudioEnergyPeak(
                time_s=round(t, 2),
                duration_s=window_sec,
                peak_db=round(db, 1),
                delta_db=round(delta, 1),
                is_scream=is_scream
            ))

    # Cluster adjacent peaks into intervals
    intervals: List[Tuple[float, float]] = []
    if peaks:
        curr_start = peaks[0].time_s
        curr_end = curr_start + window_sec
        for p in peaks[1:]:
            if p.time_s <= curr_end + 0.5:
                curr_end = max(curr_end, p.time_s + window_sec)
            else:
                intervals.append((round(curr_start, 2), round(curr_end, 2)))
                curr_start = p.time_s
                curr_end = curr_start + window_sec
        intervals.append((round(curr_start, 2), round(curr_end, 2)))

    return {
        "available": True,
        "median_db": round(median_db, 1),
        "max_db": round(max_db, 1),
        "peaks": peaks,
        "peak_intervals": intervals
    }
