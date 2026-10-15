"""Camera-sway measurement for rendered clips.

The user-visible complaint was shaking: the composed frame oscillated
horizontally and vertically at the motion guarantee's own frequency. Nothing in
the verifier measured it, so both the 107px per-branch drift and the later
15-37px double-oscillator stack shipped.

Method
------
Decode the clip at 15fps, measure the global translation between consecutive
frames with phase correlation, split the trajectory at camera cuts, remove any
linear pan per segment, and read the amplitude of the motion policy's own
frequency band (0.28Hz +/- 0.08). Content-driven movement is broadband and does
not produce a sharp peak in that band, so measuring the band, not total travel,
is what separates policy oscillation from a person leaning.

Thresholds (measured, not guessed)
----------------------------------
Calibrated against the 12-clip golden-master corpus (``tests/fixtures/manifest.json``
carries the human verdicts) plus the artifact of run 37202390593, whose clip #2
was wrongly blocked by the old broadband gate::

    clip                          human      policy   broadband
    high_income_1                 PASS        9.2      16.9   (human note: 35.6px, accepted)
    bathroom_2                    PASS       10.8      22.4   (21.0px note)
    credit_3                      PASS       18.8      18.8   (28.0px note)
    trillion_2                    PASS       13.4      30.5   (37.5px note)
    seven FAIL clips (non-sway)   FAIL      <=10.6    <=33.5
    sixty_k_1                     FAIL       37.1     126.0   (only gross-sway exemplar)
    run 37202390593 clip_2        blocked*   15.0      39.0   (publishable; *old gate)

The old broadband block (32px) sat below content humans accepted and far below
the only gross defect (126px), so publishable clips were lost. The blocks now
sit between the two distributions: broadband 52px keeps ~13px of margin over
the worst accepted content and >2x below the defect; policy 25px already
separates the worst PASS clip (18.8px) from the defect (37.1px). The 39-126px
gap is deliberately unguarded until more runs populate it; warns at 20/38 keep
every occurrence visible in the verdict log.

The metric rejects two transition classes that are not sway, measured on run
37208970507 (which lost 4 of 5 clips to them before this):

* **Directed crop pans.** ``multi_shot_dynamic`` reframes between speakers
  travel 148-350px mostly one way; those segments are skipped
  (``_is_directed_translation``).
* **Short-segment trend residue.** A 2s segment's first FFT bin is 0.5Hz - the
  policy frequency itself - so a single pan or cut reset read as 43-93px of
  "policy sway". A reading now counts only with >= ``MIN_OSCILLATION_CYCLES``.

Still documented, not engineered around: phase correlation reads whatever the
border mask contains; a subject who fills the frame edge contributes their own
motion.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

CODE_SWAY = "sway"
SWAY_WARN_PX = 20.0
# The policy oscillator is 12px; a repair attempt can raise its gain and the
# renderer clamps that at roughly 20px, so 25px is the first value that is
# unambiguously a regression rather than an escalated repair. Corpus margin:
# worst PASS clip reads 18.8px, the gross defect 37.1px.
SWAY_BLOCK_PX = 25.0
# Broadband gate for gross camera shake (the 107px per-branch class) regardless
# of frequency. Corpus-calibrated: human-accepted content reaches 39px
# (run 37202390593 clip #2, measured with this metric), non-sway FAIL clips
# reach 33.5px, and the only gross defect reads 126px. See the module docstring
# for the full table.
BROADBAND_BLOCK_PX = 52.0
BROADBAND_WARN_PX = 38.0
BROADBAND_BAND_HZ = (0.15, 1.5)

ANALYSIS_FPS = 15.0
SCALE_W, SCALE_H = 270, 480
# The metric must be tuned to the oscillator the renderer actually uses, or it
# measures the subject instead: corpus content sway at 0.22-0.33Hz was being
# reported as policy drift while the policy runs at 0.5Hz.
try:
    from core.creative_spec import MOTION_POLICY

    POLICY_BAND_HZ: tuple = (
        MOTION_POLICY.FREQUENCY_HZ - 0.10,
        MOTION_POLICY.FREQUENCY_HZ + 0.10,
    )
except Exception:  # pragma: no cover - defensive
    POLICY_BAND_HZ = (0.40, 0.60)
MIN_SEGMENT_S = 2.0
# A band reading only counts when the segment spans at least this many cycles of
# the peak frequency. Below it the "peak" is trend/transition residue: run
# 37208970507 had a 2.13s segment whose 0.47Hz peak was bin 1 of the segment
# itself - a single crop-pan step read as 93px of "policy sway".
MIN_OSCILLATION_CYCLES = 2.0
# A segment that travels mostly in one direction by a large distance is a crop
# pan (multi_shot_dynamic reframes between speakers), not sway. Measured:
# blocked run clips show net 148-350px at monotonicity 0.61-1.00, while the
# gross-shake control oscillates (monotonicity 0.01-0.14) with a path 20x its
# net. Sweep calibration: the worst accepted non-pan path is ~350px per segment.
PAN_NET_PX = 120.0
PAN_MONOTONIC_RATIO = 0.6
CUT_JUMP_PX = 30.0
CUT_RESPONSE = 0.05


@dataclass
class TrajectorySway:
    """Sway of one axis' trajectory, per segment; worst segment reported."""

    p2p_px: float = 0.0
    freq_hz: float = 0.0
    broadband_p2p_px: float = 0.0
    broadband_freq_hz: float = 0.0
    start_s: float = 0.0
    end_s: float = 0.0
    segments: int = 0
    reason: str = ""


@dataclass
class SwayReport:
    ok: bool = True
    reason: str = ""
    measurable: bool = False
    worst_p2p_px: float = 0.0
    worst_freq_hz: float = 0.0
    worst_broadband_p2p_px: float = 0.0
    worst_broadband_freq_hz: float = 0.0
    worst_axis: str = ""
    worst_start_s: float = 0.0
    worst_end_s: float = 0.0
    segments: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "measurable": self.measurable,
            "worst_p2p_px": round(self.worst_p2p_px, 2),
            "worst_freq_hz": round(self.worst_freq_hz, 4),
            "worst_broadband_p2p_px": round(self.worst_broadband_p2p_px, 2),
            "worst_broadband_freq_hz": round(self.worst_broadband_freq_hz, 4),
            "policy_band_hz": [round(value, 3) for value in POLICY_BAND_HZ],
            "worst_axis": self.worst_axis,
            "worst_start_s": round(self.worst_start_s, 2),
            "worst_end_s": round(self.worst_end_s, 2),
            "segments": self.segments,
        }


def _band_peak(freqs, spectrum, low: float, high: float) -> Tuple[float, float]:
    import numpy as np

    band = (freqs >= low) & (freqs <= high)
    if not band.any():
        return 0.0, 0.0
    index = int(np.argmax(spectrum[band]))
    return float(spectrum[band][index]), float(freqs[band][index])


def _segments_from_jumps(
    trajectory: List[float], fps: float
) -> List[Tuple[int, int]]:
    cuts = [0]
    for index in range(1, len(trajectory)):
        if abs(trajectory[index] - trajectory[index - 1]) > CUT_JUMP_PX:
            cuts.append(index)
    cuts.append(len(trajectory))
    minimum = int(MIN_SEGMENT_S * fps)
    return [
        (start, end)
        for start, end in zip(cuts, cuts[1:])
        if end - start >= minimum
    ]


def _is_directed_translation(values: Any) -> bool:
    """True when the segment travels mostly one way for PAN_NET_PX+ pixels.

    Characterises a crop pan: net displacement large and close to the total
    path. Oscillation (shake) reverses constantly, so its path dwarfs its net.
    """
    import numpy as np

    if len(values) < 2:
        return False
    net = abs(float(values[-1] - values[0]))
    path = float(np.abs(np.diff(values)).sum())
    return path > 0.0 and net >= PAN_NET_PX and (net / path) >= PAN_MONOTONIC_RATIO


def analyse_trajectory(trajectory: List[float], fps: float = ANALYSIS_FPS) -> TrajectorySway:
    """Policy-band and broadband sway of one axis. Pure, so it is unit-testable."""
    import numpy as np

    if len(trajectory) < int(MIN_SEGMENT_S * fps):
        return TrajectorySway(reason="too short to measure")

    segments = _segments_from_jumps(trajectory, fps)
    if not segments:
        return TrajectorySway(reason="no segment long enough after cut splitting")

    worst = TrajectorySway(segments=len(segments))
    for start, end in segments:
        raw = np.asarray(trajectory[start:end], dtype=np.float64)
        if _is_directed_translation(raw):
            continue
        duration_s = (end - start) / fps
        values = raw - np.linspace(raw[0], raw[-1], len(raw))
        windowed = values * np.hanning(len(values))
        spectrum = np.abs(np.fft.rfft(windowed)) / len(values) * 2.0
        freqs = np.fft.rfftfreq(len(values), d=1.0 / fps)
        amp, freq = _band_peak(freqs, spectrum, *POLICY_BAND_HZ)
        if (
            freq > 0.0
            and duration_s * freq >= MIN_OSCILLATION_CYCLES
            and amp * 2.0 > worst.p2p_px
        ):
            worst.p2p_px = amp * 2.0
            worst.freq_hz = freq
            worst.start_s = start / fps
            worst.end_s = end / fps
        broad_amp, broad_freq = _band_peak(freqs, spectrum, *BROADBAND_BAND_HZ)
        if (
            broad_freq > 0.0
            and duration_s * broad_freq >= MIN_OSCILLATION_CYCLES
            and broad_amp * 2.0 > worst.broadband_p2p_px
        ):
            worst.broadband_p2p_px = broad_amp * 2.0
            worst.broadband_freq_hz = broad_freq
    if worst.p2p_px <= 0.0 and worst.broadband_p2p_px <= 0.0:
        worst.reason = "no band peak found"
    return worst


def _decode_gray(path: Path) -> Tuple[List[Any], str]:
    try:
        import numpy as np
    except ImportError:
        return [], "numpy is not installed"
    command = [
        "ffmpeg", "-v", "error", "-i", str(path.resolve()),
        "-vf", f"fps={ANALYSIS_FPS},scale={SCALE_W}:{SCALE_H}:flags=area,format=gray",
        "-f", "rawvideo", "-",
    ]
    try:
        result = subprocess.run(command, capture_output=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as error:
        return [], f"decode failed: {error}"
    if result.returncode != 0 or not result.stdout:
        return [], "decode produced no frames"
    frame_bytes = SCALE_W * SCALE_H
    count = len(result.stdout) // frame_bytes
    if count < 2:
        return [], "fewer than two frames decoded"
    return list(
        np.frombuffer(result.stdout[: count * frame_bytes], dtype=np.uint8).reshape(
            count, SCALE_H, SCALE_W
        )
    ), ""


def analyse(path: Path) -> SwayReport:
    """Measure camera sway for one rendered clip. Never raises."""
    if not path.exists():
        return SwayReport(ok=False, reason=f"file not found: {path}")
    try:
        import cv2
        import numpy as np
    except ImportError:
        return SwayReport(ok=False, reason="OpenCV or numpy is not installed")

    frames, reason = _decode_gray(path)
    if not frames:
        return SwayReport(ok=False, reason=reason)

    # Measure camera motion on the frame borders only. The subject lives in the
    # middle of a portrait crop, and a person's own head/torso sway sits in the
    # same 0.2-0.5Hz band as the motion policy; whole-frame phase correlation
    # read a speaker's sway as camera shake and blocked publishable clips.
    # Background/set occupies the borders, so that is where the camera shows.
    mask = np.ones((SCALE_H, SCALE_W), dtype="float32")
    mask[int(0.10 * SCALE_H): int(0.85 * SCALE_H),
         int(0.15 * SCALE_W): int(0.85 * SCALE_W)] = 0.0

    window = cv2.createHanningWindow((SCALE_W, SCALE_H), cv2.CV_32F)
    dx = [0.0]
    dy = [0.0]
    previous = frames[0].astype("float32") * mask
    for frame in frames[1:]:
        current = frame.astype("float32") * mask
        (shift_x, shift_y), response = cv2.phaseCorrelate(previous, current, window)
        if response >= CUT_RESPONSE and abs(shift_x) <= CUT_JUMP_PX and abs(shift_y) <= CUT_JUMP_PX:
            dx.append(dx[-1] + shift_x)
            dy.append(dy[-1] + shift_y)
        else:
            # A camera cut: restart both trajectories so the jump is not read as motion.
            dx.append(dx[-1])
            dy.append(dy[-1])
            dx[-1] = 0.0
            dy[-1] = 0.0
        previous = current

    scale = 1080.0 / SCALE_W
    worst = SwayReport(measurable=True)
    for axis, trajectory in (("x", dx), ("y", dy)):
        axis_report = analyse_trajectory([value * scale for value in trajectory])
        if axis_report.p2p_px > worst.worst_p2p_px:
            worst.worst_p2p_px = axis_report.p2p_px
            worst.worst_freq_hz = axis_report.freq_hz
            worst.worst_axis = axis
            worst.worst_start_s = axis_report.start_s
            worst.worst_end_s = axis_report.end_s
            worst.segments = axis_report.segments
        if axis_report.broadband_p2p_px > worst.worst_broadband_p2p_px:
            worst.worst_broadband_p2p_px = axis_report.broadband_p2p_px
            worst.worst_broadband_freq_hz = axis_report.broadband_freq_hz
    if not worst.worst_axis and worst.worst_broadband_p2p_px <= 0.0:
        return SwayReport(ok=True, measurable=False, reason="no band peak on either axis")
    if not worst.worst_axis:
        worst.worst_axis = "x"
    worst.measurable = True
    return worst
