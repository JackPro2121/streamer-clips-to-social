"""A/V sync measurement.

There is no A/V sync check anywhere in this codebase today, and AGENTS.md section
3.2 claims "0ms Lipsync Drift ... Video frames and audio decodes are bit-identical".
Neither half of that claim is verifiable with what is in the repo, and one half is
demonstrably false:

* Video trims are quantised to ``:.2f`` (10 ms) while audio trims use ``:.3f``
  (1 ms), so the two can be up to 9 ms apart before anything else happens.
* ``aresample=48000:async=1`` deliberately stretches or squeezes audio to hit the
  video clock. That is the opposite of bit-identical.
* Worst of all, ``video_editor.py`` runs a **second** ``loudnorm`` pass with
  ``-c:v copy`` and no ``aresample``/``asetpts``/``atrim``. The video is
  stream-copied while the audio is re-filtered, and ``loudnorm`` in dynamic mode
  buffers internally. Any PTS shift it introduces slides audio against picture
  silently and permanently.

None of that was caught, because nothing looked.

Method
------
Two independent checks, because neither alone is sufficient.

**1. Structural drift (the one that works, and it is deterministic).**
Compares what the two muxed streams say about themselves: their start times and
their durations. This catches the defects actually present in this codebase:

* the second ``loudnorm`` pass with ``-c:v copy`` (``video_editor.py:116-123``)
  re-filters audio while stream-copying video, so any PTS shift it introduces
  shows up as the two streams disagreeing about duration or start;
* ``aresample=48000:async=1`` deliberately time-stretches audio;
* the 10 ms vs 1 ms trim quantisation mismatch.

It is exact, cheap, and cannot produce a false positive from content.

**Measured on the six golden masters, this found a defect the audit missed:**

===========================  ==================  ==================
Clip                         video duration      audio duration
===========================  ==================  ==================
``clip_1_THE_REAL_FEAR...``  44.70 s             44.70 s   (clean)
``clip_2_THE_MYTH_OF...``    47.50 s             47.64 s   **+140 ms**
``clip_3_WHY_HARD_WORK``     31.03 s             31.00 s   -33 ms
``clip_1_HOW_I_MADE_400K``   43.70 s             43.90 s   **+200 ms**
``clip_2_WHY_DISNEY_PAY``    50.33 s             50.45 s   **+117 ms**
``clip_1_PAID_OFF_60000``   102.13 s            102.14 s   (clean)
===========================  ==================  ==================

**Four of six published clips have the audio and video streams disagreeing about
their own duration by 34-200 ms.** Start times agree exactly (0.0 ms) in every
case, so the drift accumulates over the clip rather than being introduced at the
head -- which is the signature of a resampling or re-encode stage, not of a
seeking bug. AGENTS.md section 3.2 claims "0ms Lipsync Drift"; on this evidence
that is false by up to 200 ms.

**2. Onset correlation (opt-in, and it does not currently work).**
Speech onset lines up with visual motion, so the intent was to build an audio
onset envelope (half-wave rectified log-energy flux) and a baseline-subtracted
video motion envelope, then cross-correlate them.

**Measured result: peak confidence 0.045-0.100 across all six golden masters,
against a 0.15 trust threshold. It produced no usable number on any clip in the
corpus.** Halving the feature choice from RMS to onset flux roughly doubled the
confidence but did not make it actionable. The envelopes are simply not strongly
coupled in a talking-head clip: there is a lot of small continuous motion and a
lot of continuous speech, and their fine structure is not phase-locked.

It is kept, defaulted **off**, as a diagnostic for clips where the structural
check passes but a human still perceives lip-sync error. It is not presented as a
gate, because a check that reports "cannot measure" on every input is not a
check. Anyone enabling it should treat a low confidence as "no result", not as
"no problem".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from . import probe

# 20 ms hops. Fast enough to localise a sync error a human would notice, slow
# enough that the envelope is not dominated by single-frame codec noise.
ENVELOPE_HZ = 50.0
HOP_SECONDS = 1.0 / ENVELOPE_HZ

# Search window. Beyond +/-400 ms nobody perceives lip-sync error as lip-sync
# error; it becomes "the audio is out of sync" as a separate complaint.
MAX_LAG_SECONDS = 0.4

# Below this peak correlation the result is reported as no result at all.
MIN_TRUSTED_CONFIDENCE = 0.15

# Structural tolerance, on the streams' own account of their durations. The
# earlier 40ms was "about one frame" and blocked a clip whose audio container
# ran 65ms past the video (a trailing-padding tail, no content shift -- the
# source-alignment check below is the check that governs actual sync). 80ms
# still blocks the 140-200ms class this guard was written for.
STRUCTURAL_TOLERANCE_MS = 80.0

# Source-alignment tolerance. The rendered clip's audio must start where the
# trim said it would, within about a frame and a half of decode/encode head
# room. This is the deterministic perceptual gate the SyncNet LSE metrics could
# not be calibrated into: it compares the clip's own audio against the source
# segment it was cut from, so it cannot produce a false positive from content.
SOURCE_ALIGN_TOLERANCE_MS = 50.0
SOURCE_ALIGN_MIN_CONFIDENCE = 0.30
# Even when some lag wins the argmax, a flat surface means the winner is an
# artefact of noise, not a measurement: the whitened peak must clearly beat the
# sidelobes (P3 - GCC-PHAT + PSR confidence rule).
SOURCE_ALIGN_MIN_PSR = 6.0
SOURCE_ALIGN_SEARCH_S = 5.0
SOURCE_ALIGN_SAMPLE_RATE = 16000

# Audio decode rate. Low on purpose: we want the envelope, not the waveform.
AUDIO_SAMPLE_RATE = 8000

# Video profile size for the motion envelope. Small, because frame difference is
# a mean and a mean over 400 pixels tracks motion as well as over 200000.
MOTION_WIDTH = 80
MOTION_HEIGHT = 142

_CREATE_NO_WINDOW = 0x08000000 if hasattr(probe.subprocess, "STARTUPINFO") else 0


@dataclass
class SourceAlignment:
    """Where the clip's audio actually starts inside the source segment.

    ``drift_ms`` is ``expected_s - offset_s``: positive means the soundtrack
    **lags** the picture (audio is late), negative means it leads. Confidence is
    the normalised correlation at the peak, so a flat or unrelated signal
    reports ``measured=False`` instead of a fake offset.
    """

    measured: bool = False
    reason: str = ""
    drift_ms: float = 0.0
    offset_s: float = 0.0
    expected_s: float = 0.0
    confidence: float = 0.0

    @property
    def ok(self) -> bool:
        return self.measured and abs(self.drift_ms) <= SOURCE_ALIGN_TOLERANCE_MS

    def as_dict(self) -> Dict[str, Any]:
        return {
            "measured": self.measured,
            "ok": self.ok,
            "reason": self.reason,
            "drift_ms": round(self.drift_ms, 1),
            "offset_s": round(self.offset_s, 3),
            "expected_s": round(self.expected_s, 3),
            "confidence": round(self.confidence, 4),
        }


@dataclass
class AvSyncReport:
    """A/V offset measurement for one clip."""

    ok: bool = True
    reason: str = ""
    offset_ms: float = 0.0
    confidence: float = 0.0
    audio_windows: int = 0
    video_windows: int = 0
    audio_dynamic_range: float = 0.0
    video_dynamic_range: float = 0.0
    # Structural drift, from stream metadata. Always available.
    start_delta_ms: float = 0.0
    duration_delta_ms: float = 0.0
    video_duration: float = 0.0
    audio_duration: float = 0.0
    notes: List[str] = field(default_factory=list)

    @property
    def measurable(self) -> bool:
        return self.ok and self.confidence > 0.0

    @property
    def structural_ok(self) -> bool:
        """The reliable half: do the two streams agree about themselves?"""
        return (
            abs(self.start_delta_ms) <= STRUCTURAL_TOLERANCE_MS
            and abs(self.duration_delta_ms) <= STRUCTURAL_TOLERANCE_MS
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "structural_ok": self.structural_ok,
            "start_delta_ms": round(self.start_delta_ms, 2),
            "duration_delta_ms": round(self.duration_delta_ms, 2),
            "video_duration": round(self.video_duration, 3),
            "audio_duration": round(self.audio_duration, 3),
            "offset_ms": round(self.offset_ms, 2),
            "confidence": round(self.confidence, 4),
            "audio_windows": self.audio_windows,
            "video_windows": self.video_windows,
            "audio_dynamic_range": round(self.audio_dynamic_range, 4),
            "video_dynamic_range": round(self.video_dynamic_range, 4),
            "notes": self.notes,
        }


def _stream_float(stream: Optional[Dict[str, Any]], key: str) -> Optional[float]:
    if not stream:
        return None
    raw = stream.get(key)
    if raw in (None, "", "N/A"):
        return None
    try:
        return float(str(raw))
    except (TypeError, ValueError):
        return None


def structural_drift(path: Path) -> Dict[str, float]:
    """Compare the two muxed streams' own account of start time and duration.

    Deterministic and content-independent, so unlike the correlation check this
    cannot produce a false positive. It is the check that would have caught the
    ``-c:v copy`` loudnorm pass.
    """
    video = probe.video_stream(path)
    audio = probe.audio_stream(path)
    report: Dict[str, float] = {
        "start_delta_ms": 0.0,
        "duration_delta_ms": 0.0,
        "video_duration": 0.0,
        "audio_duration": 0.0,
    }
    if video is None or audio is None:
        return report

    video_start = _stream_float(video, "start_time") or 0.0
    audio_start = _stream_float(audio, "start_time") or 0.0
    video_duration = _stream_float(video, "duration") or 0.0
    audio_duration = _stream_float(audio, "duration") or 0.0

    if not video_duration:
        video_duration = probe.duration_seconds(path)

    report["start_delta_ms"] = (audio_start - video_start) * 1000.0
    report["duration_delta_ms"] = (audio_duration - video_duration) * 1000.0
    report["video_duration"] = video_duration
    report["audio_duration"] = audio_duration
    return report


def _zscore(signal: List[float]) -> List[float]:
    if not signal:
        return []
    count = len(signal)
    mean = sum(signal) / count
    variance = sum((value - mean) ** 2 for value in signal) / count
    std = variance ** 0.5
    if std <= 1e-9:
        return [0.0] * count
    return [(value - mean) / std for value in signal]


def dynamic_range(signal: List[float]) -> float:
    """Coefficient of variation. A flat signal has none, and cannot be correlated.

    Uses ``|mean|`` in the denominator because the envelope is in decibels, so
    the mean is large and negative (around -27 for normalised speech). An
    earlier version guarded with ``mean <= 1e-9`` and therefore reported zero
    dynamic range for every real clip.
    """
    if len(signal) < 2:
        return 0.0
    mean = sum(signal) / len(signal)
    magnitude = abs(mean)
    if magnitude <= 1e-9:
        return 0.0
    variance = sum((value - mean) ** 2 for value in signal) / len(signal)
    return (variance ** 0.5) / magnitude


def _onset_flux(envelope: List[float]) -> List[float]:
    """Half-wave rectified positive flux of a log-energy envelope.

    RMS energy in a talking-head clip is close to constant: the speaker is
    talking for the whole clip and the loudnorm pass has flattened the dynamics.
    Correlating it against frame difference therefore finds nothing, because
    there is no shared structure to find -- both signals are near-flat and their
    peaks are uncorrelated by construction.

    What does co-vary with speech is *onset*: the energy **rise** at a plosive or
    a syllable boundary, and the corresponding visual change. So the feature is
    the positive part of the first difference, which is zero wherever the signal
    is steady or falling, and large exactly at boundaries.
    """
    if len(envelope) < 2:
        return []
    return [max(0.0, envelope[i] - envelope[i - 1]) for i in range(1, len(envelope))]


def audio_envelope(path: Path, hop_seconds: float = HOP_SECONDS) -> List[float]:
    """Audio **onset** envelope, one value per hop.

    Returns half-wave rectified log-energy flux, not RMS. See
    :func:`_onset_flux` for why.
    """
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        return []

    command = [
        probe.ffmpeg_binary(),
        "-v",
        "error",
        "-i",
        str(path.resolve()),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(AUDIO_SAMPLE_RATE),
        "-f",
        "s16le",
        "-",
    ]
    code, data, _stderr = probe._run_binary(command)
    if code != 0 or not data:
        return []

    samples = np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
    if samples.size == 0:
        return []

    hop = max(1, int(AUDIO_SAMPLE_RATE * hop_seconds))
    usable = (samples.size // hop) * hop
    if usable == 0:
        return []
    windows = samples[:usable].reshape(-1, hop)
    rms = np.sqrt(np.mean(np.square(windows), axis=1))
    log_energy = 20.0 * np.log10(rms + 1e-6)
    return _onset_flux([float(value) for value in log_energy])


def video_motion_envelope(
    path: Path, hop_seconds: float = HOP_SECONDS
) -> Tuple[List[float], List[float]]:
    """Video **motion onset** envelope at ``hop_seconds``, plus frame timestamps.

    Motion magnitude is baseline-subtracted before rectification, so that a
    clip which is simply *moving a lot* everywhere does not swamp the envelope
    with a constant offset. Only motion that *rises* relative to the local
    average survives, which is what a gesture or a head turn looks like next to
    the breathing baseline.
    """
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        return [], []

    fps = max(1.0, min(60.0, 1.0 / hop_seconds))
    command = [
        probe.ffmpeg_binary(),
        "-v",
        "error",
        "-i",
        str(path.resolve()),
        "-vf",
        f"fps={fps:.4f},scale={MOTION_WIDTH}:{MOTION_HEIGHT}:flags=area,format=gray",
        "-vsync",
        "0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]
    code, data, _stderr = probe._run_binary(command)
    frame_bytes = MOTION_WIDTH * MOTION_HEIGHT
    if code != 0 or not data or frame_bytes <= 0:
        return [], []

    count = len(data) // frame_bytes
    if count < 3:
        return [], []
    frames = np.frombuffer(data[: count * frame_bytes], dtype=np.uint8).reshape(
        count, MOTION_HEIGHT, MOTION_WIDTH
    )
    deltas = np.abs(frames[1:].astype(np.int16) - frames[:-1].astype(np.int16))
    motion = deltas.mean(axis=(1, 2))
    # Baseline-subtract with a wide window so slow drift is removed but genuine
    # short spikes survive, then rectify.
    window = max(3, int(0.5 / hop_seconds) | 1)
    baseline = np.convolve(motion, np.ones(window) / window, mode="same")
    excess = np.maximum(0.0, motion - baseline)
    envelope = [float(value) for value in excess]
    timestamps = [index / fps for index in range(count)]
    return envelope, timestamps


def best_lag(
    audio: List[float],
    video: List[float],
    hop_seconds: float = HOP_SECONDS,
    max_lag_seconds: float = MAX_LAG_SECONDS,
) -> Tuple[float, float]:
    """Lag (seconds) and normalised peak correlation of the two envelopes.

    A positive lag means the *audio* envelope had to be advanced to match video,
    i.e. audio is running late relative to picture.

    Pure function, so the arithmetic is unit-testable with synthetic envelopes
    where the correct answer is known by construction.
    """
    if len(audio) < 8 or len(video) < 8:
        return 0.0, 0.0

    a = _zscore(audio)
    v = _zscore(video)
    length = min(len(a), len(v))
    a = a[:length]
    v = v[:length]

    max_lag = max(1, int(round(max_lag_seconds / hop_seconds)))
    best_offset = 0
    best_score = -2.0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            left, right = a[: length - lag], v[lag:]
        else:
            left, right = a[-lag:], v[: length + lag]
        if len(left) < 8:
            continue
        score = sum(x * y for x, y in zip(left, right)) / len(left)
        if score > best_score:
            best_score = score
            best_offset = lag
    return best_offset * hop_seconds, best_score


def analyse(
    path: Path,
    hop_seconds: float = HOP_SECONDS,
    max_lag_seconds: float = MAX_LAG_SECONDS,
    with_correlation: bool = False,
) -> AvSyncReport:
    """Measure A/V sync.

    Structural drift runs by default. The onset correlation is **off** by
    default because it produced no usable number on any clip in the golden-master
    corpus (peak confidence 0.045-0.100 against a 0.15 threshold) while costing
    two extra full decodes per clip. See the module docstring.
    """
    if probe.audio_stream(path) is None:
        return AvSyncReport(ok=False, reason="no audio stream to compare against")

    report = AvSyncReport()
    for key, value in structural_drift(path).items():
        setattr(report, key, value)

    if not with_correlation:
        if not report.structural_ok:
            report.notes.append(
                f"streams disagree: audio is {report.duration_delta_ms:+.0f}ms "
                f"{'longer' if report.duration_delta_ms > 0 else 'shorter'} than "
                f"video; start delta {report.start_delta_ms:+.0f}ms"
            )
        return report

    audio = audio_envelope(path, hop_seconds=hop_seconds)
    video, _timestamps = video_motion_envelope(path, hop_seconds=hop_seconds)
    report.audio_windows = len(audio)
    report.video_windows = len(video)

    if len(audio) < 32 or len(video) < 32:
        report.notes.append(
            f"too short to correlate (audio {len(audio)} windows, video {len(video)})"
        )
        return report

    report.audio_dynamic_range = dynamic_range(audio)
    report.video_dynamic_range = dynamic_range(video)

    if report.video_dynamic_range < 0.05:
        report.notes.append(
            "video motion envelope is flat; the clip is effectively a still frame"
        )
        return report
    if report.audio_dynamic_range < 0.05:
        report.notes.append("audio onset envelope is flat; nothing to correlate")
        return report

    offset, score = best_lag(
        audio, video, hop_seconds=hop_seconds, max_lag_seconds=max_lag_seconds
    )
    report.offset_ms = offset * 1000.0
    report.confidence = max(0.0, min(1.0, score))

    if report.confidence < MIN_TRUSTED_CONFIDENCE:
        report.notes.append(
            f"onset correlation peak too weak to trust (confidence "
            f"{report.confidence:.3f} < {MIN_TRUSTED_CONFIDENCE}); the "
            f"{report.offset_ms:+.0f}ms figure is noise, not a measurement. "
            f"Structural drift is still reported."
        )
    return report


def _decode_mono(path: Path, sample_rate: int = SOURCE_ALIGN_SAMPLE_RATE):
    """Decode a file to mono float32 at ``sample_rate``, or None."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        return None
    command = [
        probe.ffmpeg_binary(),
        "-v",
        "error",
        "-i",
        str(path.resolve()),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "-",
    ]
    code, data, _stderr = probe._run_binary(command)
    if code != 0 or not data:
        return None
    samples = np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
    return samples if samples.size else None


def _gcc_phat_offset(clip, window, sample_rate: int = SOURCE_ALIGN_SAMPLE_RATE):
    """GCC-PHAT offset of ``clip`` inside ``window``, plus confidence and PSR.

    Plain cross-correlation of two loudnorm-processed AAC signals lets the
    source's spectral tilt dominate the peak -- run 37213499992 blocked a clip
    at -115ms on a peak of only 0.392, and two clips in run 37329847616 could
    not be aligned at all (peaks 0.076 and 0.143). GCC-PHAT whitens the
    cross-spectrum (divides it by its magnitude) so level and timbre do not
    decide the peak.

    Confidence stays the normalised cross-correlation of the aligned raw
    segments (the existing gate semantics). ``psr`` is the peak-to-sidelobe
    ratio of the whitened surface, excluding a +-50ms main-lobe window: it
    measures whether one lag clearly beats all competitors. Unrelated signals
    have a broad noisy surface and a low PSR even when some lag still wins the
    argmax.
    """
    import numpy as np

    clip = np.asarray(clip, dtype=np.float64)
    window = np.asarray(window, dtype=np.float64)
    if clip.size < sample_rate // 2 or window.size <= clip.size:
        return 0.0, 0.0, 0.0
    clip = clip - clip.mean()
    window = window - window.mean()
    n = clip.size
    m = window.size
    size = 1
    while size < n + m:
        size <<= 1
    cross = np.fft.rfft(window, size) * np.conj(np.fft.rfft(clip, size))
    magnitude = np.abs(cross)
    magnitude[magnitude < 1e-12] = 1e-12
    correlation = np.fft.irfft(cross / magnitude, size)
    valid = correlation[: m - n + 1]
    index = int(np.argmax(valid))
    peak = float(valid[index])
    radius = max(1, int(0.05 * sample_rate))
    sidelobe_mask = np.ones(valid.size, dtype=bool)
    sidelobe_mask[max(0, index - radius): min(valid.size, index + radius + 1)] = False
    sidelobes = valid[sidelobe_mask]
    psr = 0.0
    if sidelobes.size > 8:
        std = float(sidelobes.std())
        if std > 1e-12:
            psr = (peak - float(sidelobes.mean())) / std
    segment = window[index : index + n]
    denominator = float(np.linalg.norm(segment) * np.linalg.norm(clip))
    confidence = float(np.dot(segment, clip) / denominator) if denominator > 0 else 0.0
    return index / float(sample_rate), confidence, psr


def best_source_offset(clip, window, sample_rate: int = SOURCE_ALIGN_SAMPLE_RATE):
    """Offset of ``clip`` inside ``window`` in seconds, plus peak confidence.

    Pure, so the arithmetic is unit-testable with synthetic signals where the
    answer is known by construction. The offset is located with GCC-PHAT (see
    ``_gcc_phat_offset``); the confidence is the normalised correlation of the
    aligned raw segments.
    """
    offset_s, confidence, _psr = _gcc_phat_offset(clip, window, sample_rate)
    return offset_s, confidence


def align_with_source(
    clip_path: Union[str, Path],
    source_path: Union[str, Path],
    expected_start_s: float,
    search_seconds: float = SOURCE_ALIGN_SEARCH_S,
    sample_rate: int = SOURCE_ALIGN_SAMPLE_RATE,
) -> SourceAlignment:
    """Measure where the rendered clip's audio actually starts in its source.

    The renderer is handed a downloaded segment and a trim start. If the clip's
    audio content does not begin at that start, the film is out of sync with the
    edit -- the exact class of defect that shipped when ``asetpts`` ran before
    ``atrim``. This measures it directly, so the verdict can block on it without
    trusting any plan.

    Both paths are normalised here: a caller once handed over a ``str`` (the
    English-audio fallback wrote ``str(fixed_clip)`` into the clip path) and
    every affected clip died with "'str' object has no attribute 'exists'" at
    pixel verification.
    """
    clip_path = Path(clip_path)
    source_path = Path(source_path)
    if not clip_path.exists():
        return SourceAlignment(reason=f"clip not found: {clip_path}")
    if not source_path.exists():
        return SourceAlignment(reason=f"source not found: {source_path}")

    clip = _decode_mono(clip_path, sample_rate)
    if clip is None:
        return SourceAlignment(reason="clip audio could not be decoded")
    source = _decode_mono(source_path, sample_rate)
    if source is None:
        return SourceAlignment(reason="source audio could not be decoded")

    window_start_s = max(0.0, float(expected_start_s) - search_seconds)
    window_end_s = min(
        source.size / sample_rate,
        float(expected_start_s) + clip.size / sample_rate + search_seconds,
    )
    window = source[int(window_start_s * sample_rate): int(window_end_s * sample_rate)]
    if window.size <= clip.size:
        return SourceAlignment(reason="source window is not longer than the clip")

    offset_s, confidence, psr = _gcc_phat_offset(clip, window, sample_rate)
    offset_s += window_start_s
    drift_ms = (float(expected_start_s) - offset_s) * 1000.0
    if confidence < SOURCE_ALIGN_MIN_CONFIDENCE:
        return SourceAlignment(
            measured=False,
            reason=f"correlation peak too weak ({confidence:.3f} < {SOURCE_ALIGN_MIN_CONFIDENCE})",
            offset_s=offset_s,
            expected_s=float(expected_start_s),
            confidence=confidence,
        )
    if psr < SOURCE_ALIGN_MIN_PSR:
        # An argmax on a flat, noisy surface reports a lag -- just not a
        # meaningful one. This is the class that blocked run 37213499992's
        # clip #1 at -115ms on confidence 0.392: an honest "unmeasurable", not
        # a block, is the correct reading.
        return SourceAlignment(
            measured=False,
            reason=(
                f"correlation peak ambiguous (PSR {psr:.1f} < "
                f"{SOURCE_ALIGN_MIN_PSR}); competing lags make the drift unreadable"
            ),
            offset_s=offset_s,
            expected_s=float(expected_start_s),
            confidence=confidence,
        )
    return SourceAlignment(
        measured=True,
        drift_ms=drift_ms,
        offset_s=offset_s,
        expected_s=float(expected_start_s),
        confidence=confidence,
    )
