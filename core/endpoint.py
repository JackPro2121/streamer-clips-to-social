"""Sentence-boundary resolution for clip endpoints, in one place.

Why this module exists
----------------------
The moment normalizer (``viral_detector._normalize_clip_window``) already cuts
windows at sentence boundaries: a word carrying terminal punctuation, or a
spoken pause of ``GAP_BOUNDARY_S``. The render-time "repair"
(``main._extend_moment_to_complete_transcript``) then re-implemented the same
idea with different rules: it looked at *segment* text instead of word
boundaries, used a hard-coded 54-second ceiling instead of the 55-second
contract, and clamped to that ceiling without checking whether the sentence
actually fit. A production run showed the result: clip #1 was moved off a valid
boundary to exactly ``start + 54.0``, mid-sentence, while the log said
"Extended clip endpoint to complete transcript sentence". The QA layer then
flagged the endpoint as incomplete.

This module is the single source of truth for "is this endpoint a sentence
boundary" and "which boundary should this endpoint move to". The moment
normalizer, the render-time repair and the QA warning all call it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from core.creative_spec import (
    ENDPOINT_EXTENSION_MAX_S,
    ENDPOINT_TRIM_MAX_S,
)
from core.transcriber import TranscriptSegment

# A pause at least this long between spoken words is a sentence boundary even
# when the transcriber did not punctuate it.
GAP_BOUNDARY_S = 1.2

# How close an endpoint has to be to a boundary to count as landing on it.
# Covers the 2-decimal rounding of detected moments and word-timestamp jitter.
ENDPOINT_TOLERANCE_S = 0.12

_TERMINAL = (".", "?", "!")
_TRAILING = "\"')]}\u201d\u2019 "


def _terminal_token(raw: str) -> bool:
    """True when the token ends a sentence, ignoring trailing quotes/brackets."""
    return raw.strip().rstrip(_TRAILING).endswith(_TERMINAL)


def _ordered_words(segments: Sequence[TranscriptSegment]):
    return sorted(
        (word for segment in segments for word in segment.words),
        key=lambda word: (word.start, word.end),
    )


def word_boundary_ends(segments: Sequence[TranscriptSegment]) -> List[float]:
    """Ends of words that close a sentence (punctuation or a real pause)."""
    words = _ordered_words(segments)
    boundaries: List[float] = []
    for index, word in enumerate(words):
        next_word = words[index + 1] if index + 1 < len(words) else None
        gap = next_word.start - word.end if next_word else 0.0
        if _terminal_token(word.word) or gap >= GAP_BOUNDARY_S:
            boundaries.append(word.end)
    return boundaries


def text_boundary_ends(segments: Sequence[TranscriptSegment]) -> List[float]:
    """Segment-text boundaries, used only when word timestamps are absent."""
    return [
        segment.end
        for segment in segments
        if segment.text.strip() and _terminal_token(segment.text)
    ]


def sentence_boundary_ends(segments: Sequence[TranscriptSegment]) -> List[float]:
    return sorted(set(word_boundary_ends(segments)) | set(text_boundary_ends(segments)))


def endpoint_is_complete(
    segments: Sequence[TranscriptSegment],
    end: float,
    tolerance: float = ENDPOINT_TOLERANCE_S,
) -> bool:
    """True when ``end`` lands on a sentence boundary within ``tolerance``."""
    return any(abs(boundary - end) <= tolerance for boundary in sentence_boundary_ends(segments))


@dataclass(frozen=True)
class EndpointResolution:
    """The chosen endpoint and what had to happen to reach it.

    ``status`` is one of:

    * ``complete``    - the requested endpoint was already a sentence boundary
    * ``extended``    - moved forward to the first boundary within the contract
    * ``trimmed``     - moved back to the previous boundary (within
                        ``ENDPOINT_TRIM_MAX_S``) because the next sentence did
                        not fit the contract
    * ``incomplete``  - no boundary fit; the requested endpoint is unchanged
    """

    end: float
    status: str


def resolve_endpoint(
    segments: Sequence[TranscriptSegment],
    start: float,
    requested_end: float,
    *,
    ceiling: float,
    min_duration: float,
    max_extension: float = ENDPOINT_EXTENSION_MAX_S,
    max_trim: float = ENDPOINT_TRIM_MAX_S,
    tolerance: float = ENDPOINT_TOLERANCE_S,
) -> EndpointResolution:
    """Move ``requested_end`` onto the best sentence boundary inside the contract.

    Forward extension is preferred. When the completing sentence would run past
    ``ceiling`` (or ``max_extension``), the previous boundary is used only if it
    keeps the ``min_duration`` floor and costs at most ``max_trim`` seconds.
    Otherwise the endpoint is returned unchanged with an honest ``incomplete``
    status - a caller must never report a completion that did not happen.
    """
    if endpoint_is_complete(segments, requested_end, tolerance):
        return EndpointResolution(requested_end, "complete")

    boundaries = sentence_boundary_ends(segments)
    limit = min(requested_end + max_extension, ceiling)
    future = [b for b in boundaries if requested_end - tolerance < b <= limit]
    if future:
        return EndpointResolution(future[0], "extended")

    prior = [b for b in boundaries if start + min_duration <= b <= requested_end - tolerance]
    if prior and requested_end - prior[-1] <= max_trim:
        return EndpointResolution(prior[-1], "trimmed")

    return EndpointResolution(requested_end, "incomplete")
