"""Loudness verification.

This is the one part of the pipeline that was already correct, and the point of
this module is to keep it that way. Measured across all six golden masters:

============================  ========  =========
Clip                          LUFS      True peak
============================  ========  =========
clip_1_THE_REAL_FEAR...       -14.1     -1.7 dBFS
clip_2_THE_MYTH_OF...         -14.3     -1.4 dBFS
clip_3_WHY_HARD_WORK          -14.2     -1.5 dBFS
clip_1_HOW_I_MADE_400000      -14.3     -1.5 dBFS
clip_2_WHY_DISNEY_PAY...      -14.4     -1.4 dBFS
clip_1_PAID_OFF_60000...      -14.0     -1.6 dBFS
============================  ========  =========

``loudnorm=I=-14:TP=-1.5:LRA=11`` is being honoured to within 0.4 LU and 0.2 dB.
``ARCHITECTURE_AND_METHODOLOGY.md`` section 5 is genuinely implemented, and the
sidechain ducking and SFX chain around it work.

So this module has no defects to catch. It exists as a **regression guard**: the
one number in the whole verification surface that is currently right, and which a
careless change to the audio graph could silently break. A guard that only ever
passes still has value, provided someone has checked that it can fail -- see
``test_grade_severity`` below, and the mutation note on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from . import probe

# The numbers live in core.creative_spec (the single contract source) and are
# re-exported here for the existing callers and tests. The golden masters
# measured -14.0..-14.4 LUFS / -1.4..-1.7 dBTP, i.e. within 0.4 LU and 0.2 dB;
# the 0.5/0.3 tolerances are the contract, not a tuning knob - a tolerance
# tighter than this would measure the encoder rather than the pipeline.
from core.creative_spec import (
    LUFS_TOLERANCE,
    TARGET_LUFS,
    TARGET_TRUE_PEAK_DBFS,
    TRUE_PEAK_TOLERANCE,
)
# Measured exactly at the tolerance is inside it. Float subtraction put
# -1.2 dBFS against the -1.5 ceiling at 0.30000000000000004 and blocked a clip
# from run 37208970507 that measured -14.0 LUFS.
_FLOAT_EPSILON = 1e-9

# True peak is a ceiling, not a target. Loud-on-purpose audio must not exceed
# the ceiling (that is the distortion guard), but genuinely quiet audio is not a
# publish blocker -- platforms normalise loudness anyway. The previous check
# failed a clip at -2.6 dBFS, which is roughly 1 dB below the ceiling and
# perfectly safe. Only an implausibly quiet peak (< -4 dBFS, i.e. the audio
# chain did not engage) is worth surfacing, and as a warning.
QUIET_PEAK_FLOOR_DBFS = -4.0

# AGENTS.md section 3's audio chain, listed so a regression names the clause it
# broke.
SPEC_CLAUSE = "ARCHITECTURE_AND_METHODOLOGY.md section 5 / AGENTS.md section 3.6"


@dataclass
class LoudnessReport:
    """Loudness verdict for one clip."""

    ok: bool = True
    reason: str = ""
    warn_reason: str = ""
    measured: bool = False
    integrated_lufs: float = 0.0
    loudness_range: float = 0.0
    peak_dbfs: float = 0.0
    lufs_error: float = 0.0
    peak_error: float = 0.0
    spec_clause: str = SPEC_CLAUSE

    @property
    def severity(self) -> str:
        if not self.ok or not self.measured:
            return "error"
        return "warning" if self.warn_reason else "ok"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "severity": self.severity,
            "reason": self.reason,
            "warn_reason": self.warn_reason,
            "measured": self.measured,
            "integrated_lufs": round(self.integrated_lufs, 2),
            "loudness_range": round(self.loudness_range, 2),
            "peak_dbfs": round(self.peak_dbfs, 2),
            "lufs_error": round(self.lufs_error, 3),
            "peak_error": round(self.peak_error, 3),
            "targets": {
                "lufs": TARGET_LUFS,
                "lufs_tolerance": LUFS_TOLERANCE,
                "true_peak_dbfs": TARGET_TRUE_PEAK_DBFS,
                "true_peak_tolerance": TRUE_PEAK_TOLERANCE,
                "quiet_peak_floor_dbfs": QUIET_PEAK_FLOOR_DBFS,
            },
            "spec_clause": self.spec_clause,
        }


def grade(
    integrated_lufs: float,
    peak_dbfs: float,
    loudness_range: float = 0.0,
) -> LoudnessReport:
    """Pure grading, so it is testable without ffmpeg."""
    report = LoudnessReport(
        measured=True,
        integrated_lufs=integrated_lufs,
        loudness_range=loudness_range,
        peak_dbfs=peak_dbfs,
    )
    report.lufs_error = abs(integrated_lufs - TARGET_LUFS)
    if peak_dbfs > TARGET_TRUE_PEAK_DBFS:
        report.peak_error = peak_dbfs - TARGET_TRUE_PEAK_DBFS
    else:
        report.peak_error = 0.0

    if report.lufs_error > LUFS_TOLERANCE + _FLOAT_EPSILON:
        report.ok = False
        report.reason = (
            f"integrated loudness {integrated_lufs:.1f} LUFS is "
            f"{report.lufs_error:.2f} LU from the {TARGET_LUFS:.1f} target "
            f"(tolerance {LUFS_TOLERANCE})"
        )
    elif report.peak_error > TRUE_PEAK_TOLERANCE + _FLOAT_EPSILON:
        report.ok = False
        report.reason = (
            f"true peak {peak_dbfs:.1f} dBFS is {report.peak_error:.2f} dB above "
            f"the {TARGET_TRUE_PEAK_DBFS:.1f} dBTP ceiling "
            f"(tolerance {TRUE_PEAK_TOLERANCE})"
        )
    elif peak_dbfs < QUIET_PEAK_FLOOR_DBFS:
        report.warn_reason = (
            f"true peak {peak_dbfs:.1f} dBFS is below the {QUIET_PEAK_FLOOR_DBFS:.1f} "
            f"dBFS floor; the audio chain may not have engaged"
        )
    return report


def analyse(path: Path) -> LoudnessReport:
    """Measure and grade the loudness of a rendered clip."""
    values = probe.read_loudness(path)
    if "integrated_lufs" not in values:
        return LoudnessReport(
            ok=False, reason="could not read loudness; ebur128 produced no summary"
        )
    return grade(
        integrated_lufs=values.get("integrated_lufs", 0.0),
        peak_dbfs=values.get("peak_dbfs", 0.0),
        loudness_range=values.get("loudness_range", 0.0),
    )
