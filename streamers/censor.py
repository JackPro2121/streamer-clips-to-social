"""Streamer Profanity & TOS Keyword Masker (V2).

Protects short-form clips from automated TikTok, YouTube Shorts, and Instagram
Reels shadowbans and 'Not Eligible for For You Feed' flags by masking profanity
and high-risk terms in subtitle word timestamps and captions.

Algorithm Safety:
    TikTok and Meta run optical character recognition (OCR) on burned-in subtitles.
    Uncensored profanity (e.g. FUCK, SHIT, BITCH, N-word, violent terms like KILL/SUICIDE)
    immediately triggers algorithmic reach suppression. Masking vowels (F*CK, SH*T)
    preserves human readability and viewer shock value while passing automated moderation.
"""

import re
from typing import List, Dict
from core.transcriber import TranscriptSegment, WordTimestamp

# Target profanity patterns mapped to safe masked replacements
PROFANITY_MAP: Dict[str, str] = {
    # High-severity slurs & explicit terms
    "fuck": "f*ck",
    "fucking": "f*cking",
    "fucked": "f*cked",
    "fucker": "f*cker",
    "motherfucker": "motherf*cker",
    "motherfucking": "motherf*cking",
    "shit": "sh*t",
    "shitting": "sh*tting",
    "bullshit": "bullsh*t",
    "bitch": "b*tch",
    "bitches": "b*tches",
    "bitching": "b*tching",
    "nigga": "n***a",
    "niggas": "n***as",
    "nigger": "n***a",
    "niggers": "n***as",
    "cunt": "c*nt",
    "cunts": "c*nts",
    "pussy": "p*ssy",
    "dick": "d*ck",
    "dicks": "d*cks",
    "cock": "c*ck",
    "asshole": "a**hole",
    "assholes": "a**holes",
    "retard": "r*tard",
    "retarded": "r*tarded",
    # Platform self-harm / violent speech shadowban triggers
    "kill": "k*ll",
    "killing": "k*lling",
    "killed": "k*lled",
    "murder": "m*rder",
    "suicide": "su*cide",
    "rape": "r*pe",
}

def censor_word_text(word: str) -> str:
    """Masks profanity in a single word while preserving case and trailing punctuation."""
    if not word:
        return word

    # Extract leading/trailing punctuation (e.g., '"FUCKING!"' -> '"', 'FUCKING', '!"')
    match = re.match(r"^([^\w]*)([\w\']+)([^\w]*)$", word)
    if not match:
        return word

    prefix, core, suffix = match.groups()
    lower_core = core.lower()

    if lower_core in PROFANITY_MAP:
        replacement = PROFANITY_MAP[lower_core]
        if core.isupper():
            replacement = replacement.upper()
        elif core.istitle():
            replacement = replacement.capitalize()
        return f"{prefix}{replacement}{suffix}"

    return word


def censor_text_line(text: str) -> str:
    """Scans and censors all profanity in a full text line."""
    if not text:
        return text

    words = text.split(" ")
    return " ".join(censor_word_text(w) for w in words)


def censor_transcript_segments(
    segments: List[TranscriptSegment]
) -> List[TranscriptSegment]:
    """Applies smart profanity masking across all words and segments.

    Preserves exact timing, confidence, and monotonic sequence of WordTimestamp
    objects so subtitle bounce animations remain 100% synchronized with audio.
    """
    if not segments:
        return []

    censored_segments: List[TranscriptSegment] = []

    for seg in segments:
        censored_text = censor_text_line(seg.text)
        censored_words: List[WordTimestamp] = []

        for w in (seg.words or []):
            new_word = censor_word_text(w.word)
            censored_words.append(WordTimestamp(
                word=new_word,
                start=w.start,
                end=w.end,
                is_estimated=getattr(w, "is_estimated", False),
            ))

        censored_segments.append(TranscriptSegment(
            start=seg.start,
            end=seg.end,
            text=censored_text,
            words=censored_words,
        ))

    return censored_segments
