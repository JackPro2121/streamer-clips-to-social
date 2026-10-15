"""Single cached entry point for ffprobe/ffmpeg.

The codebase previously ran ad-hoc ``subprocess`` probes from a dozen places,
each parsing JSON slightly differently and each re-running ffmpeg on the same
file. This module centralises that: one probe, cached by (path, mtime, size, args)
so repeated verification of an unchanged file is free.

Like every module in this package, it never raises. A probe failure comes back as
a :class:`ProbeResult` with ``ok=False`` and a reason, because a verifier that
throws gets ``try``-wrapped by its caller and quietly stops verifying.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Windows needs this for subprocesses without a console window; harmless on POSIX.
_CREATE_NO_WINDOW = 0x08000000 if hasattr(subprocess, "STARTUPINFO") else 0

_FFMPEG_TIMEOUT_S = 900


class FfmpegUnavailable(RuntimeError):
    """Raised only by :func:`require_ffmpeg`, never during verification."""


def ffmpeg_binary() -> str:
    return shutil.which("ffmpeg") or "ffmpeg"


def ffprobe_binary() -> str:
    return shutil.which("ffprobe") or "ffprobe"


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise FfmpegUnavailable(
            "ffmpeg/ffprobe not on PATH. core/tools/install_deps.sh installs ffmpeg; "
            "verification of rendered pixels is impossible without it."
        )


@dataclass
class ProbeResult:
    """Outcome of a probe. ``ok`` is the only field callers must branch on."""

    ok: bool
    reason: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    stderr: str = ""

    def get(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)


def _run(command: Sequence[str], timeout: int = _FFMPEG_TIMEOUT_S) -> Tuple[int, str, str]:
    try:
        completed = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            creationflags=_CREATE_NO_WINDOW,
        )
    except FileNotFoundError as error:
        return 127, "", str(error)
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout}s"
    return completed.returncode, completed.stdout, completed.stderr


def _run_binary(
    command: Sequence[str], timeout: int = _FFMPEG_TIMEOUT_S
) -> Tuple[int, bytes, str]:
    """Same as :func:`_run` but binary-safe.

    Required for rawvideo. ``text=True`` with ``errors="replace"`` would corrupt
    every byte above 0x7F, which is most of a greyscale frame, and the resulting
    frames would be subtly wrong rather than obviously broken -- the worst
    possible failure mode for a verifier.
    """
    try:
        completed = subprocess.run(
            list(command),
            capture_output=True,
            timeout=timeout,
            creationflags=_CREATE_NO_WINDOW,
        )
    except FileNotFoundError as error:
        return 127, b"", str(error)
    except subprocess.TimeoutExpired:
        return 124, b"", f"timed out after {timeout}s"
    return (
        completed.returncode,
        completed.stdout or b"",
        (completed.stderr or b"").decode("utf-8", errors="replace"),
    )


@lru_cache(maxsize=256)
def _probe_cached(resolved: str, size: int, mtime: float, extra: Tuple[str, ...]) -> ProbeResult:
    command = [ffprobe_binary(), "-v", "error", "-print_format", "json", *extra, resolved]
    code, stdout, stderr = _run(command)
    if code != 0:
        return ProbeResult(False, f"ffprobe exited {code}", stderr=stderr)
    try:
        payload = json.loads(stdout or "{}")
    except json.JSONDecodeError as error:
        return ProbeResult(False, f"ffprobe emitted invalid JSON: {error}", stderr=stderr)
    return ProbeResult(True, payload=payload, stderr=stderr)


def probe_json(path: Path, extra: Sequence[str] = ()) -> ProbeResult:
    """Run ffprobe with JSON output. Cached on file identity plus arguments."""
    try:
        stat = path.stat()
    except OSError as error:
        return ProbeResult(False, f"cannot stat {path}: {error}")
    # mtime is truncated to whole seconds so that a checkout which rewrites mtimes
    # with sub-second jitter does not defeat the cache.
    return _probe_cached(str(path.resolve()), stat.st_size, int(stat.st_mtime), tuple(extra))


def probe_streams(path: Path) -> ProbeResult:
    """Stream and format metadata. Cached."""
    return probe_json(path, ("-show_format", "-show_streams"))


def video_stream(path: Path) -> Optional[Dict[str, Any]]:
    result = probe_streams(path)
    if not result.ok:
        return None
    for stream in result.get("streams", []) or []:
        if stream.get("codec_type") == "video":
            return stream
    return None


def audio_stream(path: Path) -> Optional[Dict[str, Any]]:
    result = probe_streams(path)
    if not result.ok:
        return None
    for stream in result.get("streams", []) or []:
        if stream.get("codec_type") == "audio":
            return stream
    # A clip with no audio track is a defect, not a reason to crash the verifier,
    # so absence is reported as None and judged by the caller.
    return None


def parse_fraction(value: Any, default: float = 0.0) -> float:
    """Parse ffprobe's ``"30000/1001"`` frame-rate form. Never raises."""
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        if "/" in text:
            numerator, _, denominator = text.partition("/")
            den = float(denominator)
            if den == 0.0:
                return default
            return float(numerator) / den
        return float(text)
    except (TypeError, ValueError):
        return default


def duration_seconds(path: Path) -> float:
    result = probe_streams(path)
    if not result.ok:
        return 0.0
    streams = result.get("streams", []) or []
    for stream in streams:
        raw = stream.get("duration")
        if raw:
            try:
                return float(raw)
            except (TypeError, ValueError):
                continue
    try:
        return float((result.get("format", {}) or {}).get("duration", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def dimensions(path: Path) -> Tuple[int, int]:
    stream = video_stream(path)
    if not stream:
        return 0, 0
    try:
        return int(stream.get("width", 0) or 0), int(stream.get("height", 0) or 0)
    except (TypeError, ValueError):
        return 0, 0


_INTERVAL_RE = re.compile(
    r"(?P<kind>black|freeze)_(?P<field>start|end|duration):\s*(?P<value>-?[0-9.]+)"
)


@dataclass
class Interval:
    kind: str
    start: float
    end: float
    duration: float
    terminated: bool

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "terminated": self.terminated,
        }


def _parse_intervals(stderr: str, kind: str, clip_duration: float) -> List[Interval]:
    """Parse blackdetect/freezedetect output into intervals.

    Events are closed in emission order rather than keyed by a parsed position.
    Both filters emit ``<kind>_start`` first and then ``<kind>_duration`` and/or
    ``<kind>_end``, so a single pending-event slot is sufficient and is immune to
    whatever prefix decoration ffmpeg puts on the line.

    The ``terminated`` flag is load-bearing. freezedetect emits ``freeze_start``
    with no matching ``freeze_end``/``freeze_duration`` when a freeze runs to the
    end of the file. A parser that only records events with a closing duration
    reports such a clip as clean, which is precisely the bug recorded in
    HANDOFF section 4 item 8 -- a fully frozen clip looked like a pass.
    """
    intervals: List[Interval] = []
    pending: Optional[Dict[str, float]] = None

    for line in stderr.splitlines():
        for match in _INTERVAL_RE.finditer(line):
            if match.group("kind") != kind:
                continue
            field_name = match.group("field")
            try:
                value = float(match.group("value"))
            except (TypeError, ValueError):
                continue
            if field_name == "start":
                # A start with no close in between means the previous event was
                # unterminated. Close it against the clip end rather than
                # dropping it.
                if pending is not None:
                    intervals.append(_close_interval(pending, clip_duration, terminated=False))
                pending = {"start": value}
            elif pending is not None:
                pending[field_name] = value
                if "duration" in pending or "end" in pending:
                    intervals.append(_close_interval(pending, clip_duration, terminated=True))
                    pending = None

    if pending is not None:
        intervals.append(_close_interval(pending, clip_duration, terminated=False))
    return intervals


def _close_interval(
    slot: Dict[str, float], clip_duration: float, terminated: bool
) -> Interval:
    start = slot.get("start", 0.0)
    if "duration" in slot:
        end = start + slot["duration"]
    elif "end" in slot:
        end = slot["end"]
    else:
        end = clip_duration if clip_duration > 0.0 else start
    return Interval(
        kind="",
        start=start,
        end=end,
        duration=max(0.0, end - start),
        terminated=terminated,
    )


def detect_intervals(
    path: Path,
    filter_spec: str,
    kind: str,
    clip_duration: Optional[float] = None,
) -> List[Interval]:
    """Run a detecting ffmpeg filter and parse its interval output.

    ``filter_spec`` is the filtergraph fragment, e.g.
    ``"freezedetect=n=0.003:d=0.5"``.
    """
    resolved = str(path.resolve())
    total = duration_seconds(path) if clip_duration is None else clip_duration
    command = [
        ffmpeg_binary(),
        "-hide_banner",
        "-nostats",
        "-v",
        "info",
        "-i",
        resolved,
        "-vf",
        filter_spec,
        "-an",
        "-f",
        "null",
        "-",
    ]
    _code, _stdout, stderr = _run(command)
    return _parse_intervals(stderr, kind, total)


def decode_gray_frames(
    path: Path,
    width: int,
    height: int,
    max_frames: int = 240,
    start: Optional[float] = None,
    end: Optional[float] = None,
) -> Tuple[List[Any], str]:
    """Decode evenly sampled greyscale frames at a reduced size.

    Downscaling uses ``flags=area``, a box filter. That matters: a windowed
    resampler bleeds bright content into adjacent black pixels, which would make
    a real pillarbox look narrower than it is. An area average keeps a black
    column black.

    Returns ``(frames, reason)``. On failure ``frames`` is empty and ``reason``
    explains why. numpy is imported lazily so this module stays importable in
    environments without it.
    """
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a hard runtime dep
        return [], "numpy is not installed"

    total = duration_seconds(path)
    if total <= 0.0:
        return [], "could not determine clip duration"

    window_start = 0.0 if start is None else max(0.0, float(start))
    window_end = total if end is None else min(total, float(end))
    span = max(0.05, window_end - window_start)

    # Aim for `max_frames` samples but never exceed 30 fps: past that we are
    # decoding faster than the eye can see and only burning time.
    rate = min(30.0, max(0.5, max_frames / span))
    seek = ["-ss", f"{window_start:.3f}"] if window_start > 0.0 else []
    command = [
        ffmpeg_binary(),
        "-v",
        "error",
        *seek,
        "-t",
        f"{span:.3f}",
        "-i",
        str(path.resolve()),
        "-vf",
        f"fps={rate:.4f},scale={width}:{height}:flags=area,format=gray",
        "-vsync",
        "0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]
    code, stdout_bytes, stderr = _run_binary(command)
    if code != 0:
        return [], f"ffmpeg decode failed ({code}): {stderr.strip()[:300]}"

    frame_bytes = width * height
    if frame_bytes <= 0 or not stdout_bytes:
        return [], "ffmpeg produced no frames"

    count = len(stdout_bytes) // frame_bytes
    if count == 0:
        return [], "ffmpeg produced zero complete frames"
    buffer = np.frombuffer(stdout_bytes[: count * frame_bytes], dtype=np.uint8)
    return list(buffer.reshape(count, height, width)), ""


def decode_colour_frames(
    path: Path,
    width: int,
    height: int,
    max_frames: int = 40,
    start: Optional[float] = None,
    end: Optional[float] = None,
) -> Tuple[List[Any], str]:
    """Decode evenly sampled BGR frames at a reduced size.

    Same contract as :func:`decode_gray_frames` but in colour, for detectors that
    key on a colour signature. ``flags=area`` for the same reason: a windowed
    resampler bleeds colour across an edge and would inflate a saturated-pixel
    mask.
    """
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        return [], "numpy is not installed"

    total = duration_seconds(path)
    if total <= 0.0:
        return [], "could not determine clip duration"

    window_start = 0.0 if start is None else max(0.0, float(start))
    window_end = total if end is None else min(total, float(end))
    span = max(0.05, window_end - window_start)
    rate = min(30.0, max(0.5, max_frames / span))

    seek = ["-ss", f"{window_start:.3f}"] if window_start > 0.0 else []
    command = [
        ffmpeg_binary(),
        "-v",
        "error",
        *seek,
        "-t",
        f"{span:.3f}",
        "-i",
        str(path.resolve()),
        "-vf",
        f"fps={rate:.4f},scale={width}:{height}:flags=area",
        "-vsync",
        "0",
        "-frames:v",
        str(max_frames),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]
    code, data, stderr = _run_binary(command)
    frame_bytes = width * height * 3
    if code != 0 or not data:
        return [], f"ffmpeg colour decode failed ({code}): {stderr.strip()[:200]}"
    count = len(data) // frame_bytes
    if count == 0:
        return [], "ffmpeg produced zero complete frames"
    buffer = np.frombuffer(data[: count * frame_bytes], dtype=np.uint8)
    return list(buffer.reshape(count, height, width, 3)), ""


def read_loudness(path: Path) -> Dict[str, float]:
    """Parse ``ebur128`` summary. Returns an empty dict on failure."""
    command = [
        ffmpeg_binary(),
        "-hide_banner",
        "-nostats",
        "-i",
        str(path.resolve()),
        "-af",
        "ebur128=peak=true:framelog=quiet",
        "-f",
        "null",
        "-",
    ]
    _code, _stdout, stderr = _run(command)
    values: Dict[str, float] = {}
    for line in stderr.splitlines():
        stripped = line.strip()
        for key, prefix in (
            ("integrated_lufs", "I:"),
            ("loudness_range", "LRA:"),
            ("peak_dbfs", "Peak:"),
        ):
            if stripped.startswith(prefix):
                token = stripped[len(prefix) :].split()[0] if stripped[len(prefix) :].split() else ""
                try:
                    values[key] = float(token)
                except (TypeError, ValueError):
                    continue
    return values
