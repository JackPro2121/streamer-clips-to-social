"""P3 Caption Emphasis: keyword detection and pop styling for karaoke ASS.

Firecrawl research (2026-10-05): Submagic's retention edge is "animated
captions... movement, pacing, and emphasis"; karaoke highlight tracks each
word in sync with speech. We already have karaoke timing; what is missing is
*emphasis* - numbers, money, and power words should hit harder than filler.

This module is the pure decision layer: which words deserve emphasis, and the
ASS override tags that animate them. The subtitle generator applies the tags;
timing stays owned by the caption pipeline.
"""

from __future__ import annotations

import re
from typing import List, Sequence

# Curated power words from retention editing practice: emotional stakes,
# transformation, conflict, money. Short on purpose - a line with three
# emphasised words emphasises nothing.
POWER_WORDS = frozenset(
    {
        "never", "always", "everyone", "nobody", "everything", "nothing",
        "broke", "rich", "poor", "wealth", "debt", "fired", "quit", "lost",
        "secret", "mistake", "truth", "lie", "free", "best", "worst",
        "insane", "crazy", "shocking", "stop", "start", "huge", "massive",
        "million", "billion", "thousand", "dollars", "money", "cash",
        "wrong", "right", "hate", "love", "fear", "fail", "win", "worse",
        "better", "first", "last", "finally", "immediately", "urgent",
    }
)

_NUMERIC = re.compile(r"^\$?\d[\d,.]*%?$")
_STRIP = re.compile(r"^[^A-Za-z0-9$%.]+|[^A-Za-z0-9%]+$")

# A quick scale pop on the emphasised word, easing back to normal. ASS
# timings are centiseconds: over 9cs from the word's start.
EMPHASIS_POP_TAGS = r"{\fscx112\fscy112\t(0,90,\fscx100\fscy100)}"


def _clean(word: str) -> str:
    return _STRIP.sub("", word)


def is_emphasis_word(word: str) -> bool:
    """True for money/number tokens, power words, and short ALL-CAPS."""
    bare = _clean(word)
    if not bare:
        return False
    if _NUMERIC.match(bare):
        return True
    lowered = bare.lower()
    if lowered in POWER_WORDS:
        return True
    # ALL-CAPS acronyms of two letters or more (IRS, ROI, AI), but not
    # single-letter words and not the start of a normal sentence.
    if len(bare) >= 2 and bare.isupper() and bare.isalpha():
        return True
    return False


def mark_emphasis(words: Sequence[str]) -> List[bool]:
    """Per-word emphasis flags, aligned with ``words``."""
    return [is_emphasis_word(word) for word in words]


def emphasised_line(words: Sequence[str]) -> str:
    """Debug/preview form: ``[$40,000]`` marks emphasised words."""
    return " ".join(
        f"[{word}]" if flag else word
        for word, flag in zip(words, mark_emphasis(words))
    )
