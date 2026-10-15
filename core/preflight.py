"""P4 - the download pre-flight: container -> decode -> luminance variance.

Three cheap stages run on a freshly downloaded source segment, before any
render spends CPU on it:

1. **Container** - the file must be a readable container with a video stream
   and at least ``MIN_SEGMENT_SECONDS`` of content. Audio may be absent (the
   renderer substitutes a silent bed). A video/audio duration mismatch is
   deliberately NOT a rejection: the class seen in run 37347481817 (two of
   three clips) is repaired by ``video_editor._harmonise_stream_durations``.
2. **Decode** - the first ``DECODE_PROBE_SECONDS`` must decode cleanly; a
   truncated or corrupt segment fails here instead of mid-render.
3. **Luminance variance** - if every sampled frame in the opening
   ``FLAT_SCAN_SECONDS`` is visually flat, the segment opens on a slate or a
   dead frame (the class that ran 3.5s of flat grey/green before the slide in
   the freeze investigation of 2026-10-05). Flat openings render into freeze
   blocks, so the candidate is rejected here and backfill replaces it.

The module is evidence-only on failure details; callers decide what to do.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

MIN_SEGMENT_SECONDS = 3.0
DECODE_PROBE_SECONDS = 2.0
FLAT_SCAN_SECONDS = 3.0
FLAT_FRAME_SAMPLES = 5
# A flat slate measures near zero; real footage (even a dim studio wide shot)
# measures well above this at 160x90. Calibrated against the flat-open classes:
# the 3.5s grey/green slate, black frames, and hard cut-to-colour cards.
LUMINANCE_STDDEV_MIN = 6.0


@dataclass
class PreflightReport:
    ok: bool = True
    stage: str = ""
    reason: str = ""
    video_duration: float = 0.0
    audio_duration: float = 0.0
    luminance_stddev_min: float = 0.0
    luminance_stddev_max: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "stage": self.stage,
            "reason": self.reason,
            "video_duration": round(self.video_duration, 3),
            "audio_duration": round(self.audio_duration, 3),
            "luminance_stddev_min": round(self.luminance_stddev_min, 2),
            "luminance_stddev_max": round(self.luminance_stddev_max, 2),
        }


def _stream_durations(path: Path) -> Optional[Dict[str, float]]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "stream=codec_type,duration",
            "-of", "csv=p=0", str(path),
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30,
    )
    if result.returncode != 0:
        return None
    video = audio = 0.0
    for line in result.stdout.splitlines():
        parts = line.strip().split(",")
        if len(parts) < 2:
            continue
        try:
            duration = float(parts[1])
        except ValueError:
            continue
        if parts[0] == "video" and not video:
            video = duration
        elif parts[0] == "audio" and not audio:
            audio = duration
    return {"video": video, "audio": audio}


def _decodes(path: Path) -> bool:
    result = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(path),
            "-t", str(DECODE_PROBE_SECONDS), "-f", "null", "-",
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120,
    )
    return result.returncode == 0


def _frame_luminance_stddevs(path: Path) -> List[float]:
    """Grey stddev of up to five frames sampled across the opening seconds."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - render job always has numpy
        return []
    result = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(path),
            "-t", str(FLAT_SCAN_SECONDS),
            "-vf", f"fps={FLAT_FRAME_SAMPLES / FLAT_SCAN_SECONDS},scale=160:90",
            "-frames:v", str(FLAT_FRAME_SAMPLES),
            "-f", "rawvideo", "-pix_fmt", "gray", "-",
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120,
    )
    if result.returncode != 0 or not result.stdout:
        return []
    frame_size = 160 * 90
    data = result.stdout
    deviations: List[float] = []
    for index in range(len(data) // frame_size):
        chunk = np.frombuffer(
            data[index * frame_size:(index + 1) * frame_size], dtype=np.uint8
        ).astype(np.float32)
        deviations.append(float(chunk.std()))
    return deviations


def preflight_source(path: Union[str, Path]) -> PreflightReport:
    """Run the three stages; ``ok=False`` names the stage that rejected."""
    path = Path(path)
    if not path.exists() or path.stat().st_size <= 0:
        return PreflightReport(ok=False, stage="container", reason="file missing or empty")

    durations = _stream_durations(path)
    if durations is None:
        return PreflightReport(ok=False, stage="container", reason="not a readable media container")
    if durations["video"] <= 0.0:
        return PreflightReport(
            ok=False, stage="container", reason="no video stream duration",
            audio_duration=durations["audio"],
        )
    if durations["video"] < MIN_SEGMENT_SECONDS:
        return PreflightReport(
            ok=False, stage="container",
            reason=f"video stream is only {durations['video']:.2f}s (< {MIN_SEGMENT_SECONDS})",
            video_duration=durations["video"], audio_duration=durations["audio"],
        )

    if not _decodes(path):
        return PreflightReport(
            ok=False, stage="decode", reason="first 2s do not decode cleanly",
            video_duration=durations["video"], audio_duration=durations["audio"],
        )

    deviations = _frame_luminance_stddevs(path)
    report = PreflightReport(
        video_duration=durations["video"], audio_duration=durations["audio"],
    )
    if deviations:
        report.luminance_stddev_min = float(min(deviations))
        report.luminance_stddev_max = float(max(deviations))
        if report.luminance_stddev_max < LUMINANCE_STDDEV_MIN:
            report.ok = False
            report.stage = "luminance"
            report.reason = (
                f"opening {FLAT_SCAN_SECONDS:.0f}s is visually flat "
                f"(max stddev {report.luminance_stddev_max:.1f} < {LUMINANCE_STDDEV_MIN}); "
                "slate/dead-frame class"
            )
    return report
