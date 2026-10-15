"""Motion metrics: frozen output, dead output, and the accounting QA got wrong.

The defect this module is built around is measured, not hypothetical. In the
published clip ``clip_1_HOW_I_MADE_400000_IN_A_MONTH.mp4`` freezedetect reports
three events -- 41.1s (0.533s), 41.633s (0.6s), 42.867s (0.5s) -- totalling
**1.63 seconds of frozen video**, 3.7% of a 43.8s clip. The CI run reported
``success`` and editorial QA reported ``passed=True``.

The reason is a single line of arithmetic:

```python
# v1/editorial_qa.py:178, with max_freeze_duration defaulting to 0.8 at :265
if event["duration"] > max_freeze_duration:
    issues.append(QaIssue("freeze_interval", "error", ...))
```

Each event is individually under 0.8s, so nothing fired. The gate was
**per-event maximum** when the meaningful quantity is a **cumulative total**. A
viewer does not experience three 0.5s holds as three acceptable events; they
experience a two-second stall near the end of the clip, which on a 45-second
hook-driven piece is where retention is already fragile.

So this module reports the sum first, the max second, and the fraction of the
clip third -- and treats an unterminated freeze (one that runs to end-of-file)
as fully disqualifying regardless of duration, because a parser that requires a
closing ``freeze_duration`` reports such a clip as clean.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

from . import probe

# Primary gate. 200 ms, not 500 ms: the first-0.25s cover-frame defect measured
# 0.267 s and was invisible to the old d=0.5 floor, so a regression of it would
# pass every check. The strict pass below stays for diagnostics.
FREEZE_FILTER = "n=0.003:d=0.2"
FREEZE_FILTER_STR = f"freezedetect={FREEZE_FILTER}"
FREEZE_FILTER_STRICT = "freezedetect=n=0.001:d=0.3"
BLACK_FILTER_STR = "blackdetect=d=0.15:pix_th=0.10"


@dataclass
class MotionReport:
    """Aggregate motion findings for one clip."""

    clip_duration: float = 0.0
    freeze_events: List[Dict[str, Any]] = field(default_factory=list)
    black_events: List[Dict[str, Any]] = field(default_factory=list)
    cumulative_freeze_s: float = 0.0
    longest_freeze_s: float = 0.0
    cumulative_black_s: float = 0.0
    event_count: int = 0
    has_unterminated_freeze: bool = False
    reason: str = ""

    @property
    def freeze_fraction(self) -> float:
        if self.clip_duration <= 0.0:
            return 0.0
        return self.cumulative_freeze_s / self.clip_duration

    @property
    def black_fraction(self) -> float:
        if self.clip_duration <= 0.0:
            return 0.0
        return self.cumulative_black_s / self.clip_duration

    def as_dict(self) -> Dict[str, Any]:
        return {
            "clip_duration": round(self.clip_duration, 3),
            "event_count": self.event_count,
            "cumulative_freeze_s": round(self.cumulative_freeze_s, 3),
            "longest_freeze_s": round(self.longest_freeze_s, 3),
            "freeze_fraction": round(self.freeze_fraction, 5),
            "cumulative_black_s": round(self.cumulative_black_s, 3),
            "black_fraction": round(self.black_fraction, 5),
            "has_unterminated_freeze": self.has_unterminated_freeze,
            "freeze_events": self.freeze_events,
            "black_events": self.black_events,
            "reason": self.reason,
        }


def summarise(
    freeze_events: Sequence[probe.Interval],
    clip_duration: float = 0.0,
    black_events: Sequence[probe.Interval] = (),
) -> MotionReport:
    """Reduce interval lists to the numbers a threshold can be applied to.

    Pure function, so the accounting bug is unit-testable without ffmpeg.
    """
    report = MotionReport(clip_duration=clip_duration)
    report.freeze_events = [event.as_dict() for event in freeze_events]
    report.black_events = [event.as_dict() for event in black_events]
    report.event_count = len(freeze_events)

    if freeze_events:
        report.cumulative_freeze_s = sum(event.duration for event in freeze_events)
        report.longest_freeze_s = max(event.duration for event in freeze_events)
        report.has_unterminated_freeze = any(not event.terminated for event in freeze_events)
    if black_events:
        report.cumulative_black_s = sum(event.duration for event in black_events)
    return report


def analyse(
    path: Path,
    clip_duration: float | None = None,
    strict: bool = False,
    include_black: bool = True,
) -> MotionReport:
    """Detect freezes (and optionally black intervals) in a rendered clip.

    ``strict=True`` additionally runs a more sensitive filter
    (``n=0.001 d=0.3``) so near-freezes are visible in the report even when they
    do not trip the primary gate. It roughly doubles decode time, so it is off by
    default and used for diagnostics.
    """
    total = probe.duration_seconds(path) if clip_duration is None else clip_duration
    freeze_events = probe.detect_intervals(path, FREEZE_FILTER_STR, "freeze", total)
    if strict:
        for event in probe.detect_intervals(path, FREEZE_FILTER_STRICT, "freeze", total):
            # Only add events the primary filter missed; the stricter filter
            # reports everything the primary one does, plus more.
            if not any(
                abs(event.start - known.start) < 0.05 for known in freeze_events
            ):
                freeze_events.append(event)
        freeze_events.sort(key=lambda item: item.start)

    black_events: List[probe.Interval] = []
    if include_black:
        black_events = probe.detect_intervals(path, BLACK_FILTER_STR, "black", total)

    if not freeze_events and not black_events and total <= 0.0:
        return MotionReport(clip_duration=total, reason="could not determine clip duration")
    return summarise(freeze_events, clip_duration=total, black_events=black_events)
