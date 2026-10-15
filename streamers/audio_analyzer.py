"""Streamer Audio Energy & Scream Peak Detector.

High-performance audio waveform analyzer for detecting:
- Screams, desk slams, and high-energy streamer reactions (e.g. Kai Cenat, IShowSpeed)
- RMS decibel energy spikes above median baseline
- Timestamp windows for dynamic punch-zooms and viral moment candidate scoring
- Zero heavy dependencies (uses FFmpeg 8.x + NumPy PCM streaming)
"""

import subprocess
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class AudioEnergyPeak:
    time_s: float
    duration_s: float
    peak_db: float
    delta_db: float
    is_scream: bool


def analyze_audio_energy(
    media_path: Path,
    sample_rate: int = 16000,
    window_sec: float = 0.5,
    hop_sec: float = 0.25,
    scream_delta_db: float = 6.5
) -> Dict[str, any]:
    """Analyzes audio waveform for instantaneous loudness spikes and scream moments.

    Returns dict with:
        median_db: float
        max_db: float
        peaks: List[AudioEnergyPeak]
        peak_intervals: List[Tuple[float, float]]  # (start, end) suitable for punch zooms
    """
    if not media_path.exists():
        return {"median_db": -60.0, "max_db": -60.0, "peaks": [], "peak_intervals": []}

    cmd = [
        "ffmpeg", "-y", "-i", str(media_path),
        "-f", "f32le", "-ac", "1", "-ar", str(sample_rate),
        "-vn", "-"
    ]

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        raw_audio, _ = proc.communicate(timeout=60)
    except Exception as exc:
        print(f"[!] Audio extraction failed for {media_path.name}: {exc}")
        return {"median_db": -60.0, "max_db": -60.0, "peaks": [], "peak_intervals": []}

    if not raw_audio:
        return {"median_db": -60.0, "max_db": -60.0, "peaks": [], "peak_intervals": []}

    samples = np.frombuffer(raw_audio, dtype=np.float32)
    total_samples = len(samples)
    if total_samples == 0:
        return {"median_db": -60.0, "max_db": -60.0, "peaks": [], "peak_intervals": []}

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
        return {"median_db": -60.0, "max_db": -60.0, "peaks": [], "peak_intervals": []}

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
        "median_db": round(median_db, 1),
        "max_db": round(max_db, 1),
        "peaks": peaks,
        "peak_intervals": intervals
    }
