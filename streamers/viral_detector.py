"""High-Energy Streamer Viral Detector (V2).

Specialized AI model prompting and heuristic fallback for high-energy streamer content:
- Screaming, rage, desk slams, chair falls, jumping
- Hysterical laughter, comedy fails, unhinged reactions
- Chat donation roasts, troll moments, beef, drama
- High-stakes clutches, jumpscares, chaotic energy

Runs Gemini 3.8 Flash -> Groq -> Ollama -> Streamer heuristic detector.

Defects addressed in this revision
------------------------------------
D11 — 40,000 char truncation replaced with two-pass chunked analysis.
       _prescan_high_energy_zones() selects the top STREAMER_PRE_SCAN_TOP_ZONES
       20-minute windows from the FULL transcript; each is sent to Gemini
       separately, keeping every chunk under STREAMER_TRANSCRIPT_CHUNK_CHARS.
D5  — Duration cap is now enforced BEFORE sentence-boundary snapping so that
       _normalize_clip_window() never destroys a clean endpoint.
D6  — Heuristic fallback end-time is clamped to transcript_end, preventing
       windows that extend beyond the last segment.
Non-det title — random.choice replaced with a hashlib-seeded deterministic
       selection so golden-master tests are stable across runs.
"""

import hashlib
import json
import re
import unicodedata
from typing import List, Tuple

from pydantic import BaseModel, Field

from core.transcriber import TranscriptSegment

# These helpers live in the shared engine. Three of them are *private* upstream
# (`_normalize_clip_window`, `_safe_int`, `strip_emojis` is public but only by
# accident), which makes V2 silently dependent on clips-core's private surface:
# a rename upstream turns into an opaque ImportError at the top of the run.
# Wrap it so the failure names the cause and points at SYNC.md.
# tests/test_v2_audit_round2.py pins every name below so upstream drift fails in
# CI on the PR that causes it rather than in a 90-minute publish job.
try:
    from core.viral_detector import (
        query_gemini_models, query_groq_free_models, query_ollama_cloud_models,
        strip_emojis, _normalize_clip_window, _safe_int
    )
except ImportError as exc:  # pragma: no cover - guard against upstream drift
    raise ImportError(
        "streamers.viral_detector requires the model-query helpers and the "
        "private _normalize_clip_window/_safe_int helpers from core.viral_detector. "
        "If this fails, clips-core renamed or removed one of them: land the change "
        "upstream and sync both pipelines (see SYNC.md). "
        f"Underlying error: {exc}"
    ) from exc
from core.config import GEMINI_API_KEY, GROQ_API_KEY, OLLAMA_API_KEY
from streamers.spec import (
    STREAMER_MAX_CLIP_DURATION_S,
    STREAMER_MIN_CLIP_DURATION_S,
    STREAMER_DURATION_QUANTIZATION_MARGIN_S,
    STREAMER_TRANSCRIPT_CHUNK_CHARS,
    STREAMER_PRE_SCAN_TOP_ZONES,
    STREAMER_PRE_SCAN_WINDOW_S,
)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class StreamerClipCandidate(BaseModel):
    title: str = Field(description="Punchy all-caps viral hook title (under 45 chars)")
    start_time: float = Field(description="Start time in seconds")
    end_time: float = Field(description="End time in seconds")
    duration: float = Field(description="Duration in seconds (15 to 45s)")
    viral_score: int = Field(description="Virality score 1 to 100")
    hook_reason: str = Field(description="Why this streamer moment goes viral")
    social_caption: str = Field(description="TikTok/Shorts engaging caption")
    hashtags: List[str] = Field(description="Relevant hashtags like #streamer #funny #twitch #shorts")
    streamer_energy: str = Field(default="hype", description="rage | funny | clutch | drama | hype")
    sfx_cues: List[Tuple[float, str]] = Field(
        default_factory=list,
        description="Sound effect triggers relative to clip start: [[timestamp_sec, 'whoosh'|'pop'|'ding'], ...]"
    )


# ---------------------------------------------------------------------------
# Keyword taxonomy (used by both AI prompt and heuristic fallback)
# ---------------------------------------------------------------------------

STREAMER_KEYWORDS = {
    "rage": ["bro", "what", "no way", "stop", "swear", "deadass", "shut up", "hell no", "cap"],
    "funny": ["laugh", "bruh", "nah", "cooked", "chat", "gg", "lmaooo", "wild", "tripping"],
    "hype": ["let's go", "clutch", "insane", "unbelievable", "w", "huge", "holy", "god"],
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_chunk_text(zone_segs: List[TranscriptSegment]) -> str:
    """Convert a list of transcript segments into a prompt-safe text block.

    Exists to ensure every Gemini chunk stays under STREAMER_TRANSCRIPT_CHUNK_CHARS
    by truncating at the character budget rather than at segment count.  Truncating
    by character (not by segment) avoids sending a partial line that confuses the model
    about which timestamps are present.
    """
    lines: List[str] = []
    total = 0
    for seg in zone_segs:
        line = f"[{seg.start:.1f}s - {seg.end:.1f}s]: {seg.text}"
        if total + len(line) > STREAMER_TRANSCRIPT_CHUNK_CHARS:
            lines.append("... [zone truncated] ...")
            break
        lines.append(line)
        total += len(line)
    return "\n".join(lines)


def _prescan_high_energy_zones(
    segments: List[TranscriptSegment],
    top_n: int = STREAMER_PRE_SCAN_TOP_ZONES,
    window_s: float = STREAMER_PRE_SCAN_WINDOW_S,
) -> List[Tuple[float, float]]:
    """Fast local pre-scan across the FULL transcript.

    Fixes D11: the old implementation truncated the concatenated transcript text at
    40,000 characters, which for a 4-hour stream covered only the opening ~25 minutes.
    Viral moments deep in a stream were never visible to Gemini.

    This function scores every non-overlapping 20-minute window using cheap
    regex/string operations (no API calls), then returns the top_n windows by score.
    Only those windows are forwarded to Gemini, keeping each prompt under
    STREAMER_TRANSCRIPT_CHUNK_CHARS while covering the full stream breadth.

    Scoring formula (intentionally mimics the heuristic detector so they agree):
        score = exclamation_count * 4
              + all_caps_word_count * 3
              + keyword_hit_count * 5

    Non-overlapping selection: once a window is chosen its segment range is excluded
    from all subsequent candidates, preventing the same peak from being returned twice.

    Returns:
        List of (start_s, end_s) tuples sorted by descending excitement score,
        length <= top_n.
    """
    if not segments:
        return []

    transcript_end = segments[-1].end

    # Build sliding windows at every segment boundary
    scored: List[Tuple[float, float, float]] = []  # (score, start_s, end_s)

    for seg in segments:
        win_start = seg.start
        win_end = min(win_start + window_s, transcript_end)
        window_segs = [s for s in segments if s.start >= win_start and s.end <= win_end]
        if not window_segs:
            continue
        text = " ".join(s.text for s in window_segs)
        lower = text.lower()

        exclamation_count = text.count("!")
        caps_word_count = sum(1 for w in text.split() if w.isupper() and len(w) > 2)
        keyword_hits = sum(
            1 for kw_list in STREAMER_KEYWORDS.values() for kw in kw_list if kw in lower
        )
        score = exclamation_count * 4 + caps_word_count * 3 + keyword_hits * 5
        scored.append((score, win_start, win_end))

    # Sort descending by score
    scored.sort(key=lambda x: x[0], reverse=True)

    # Greedily select non-overlapping windows
    selected: List[Tuple[float, float]] = []
    excluded_ranges: List[Tuple[float, float]] = []

    for score, win_start, win_end in scored:
        if len(selected) >= top_n:
            break
        # Skip if this window overlaps any already-selected window
        overlaps = any(
            not (win_end <= ex_s or win_start >= ex_e)
            for ex_s, ex_e in excluded_ranges
        )
        if overlaps:
            continue
        selected.append((win_start, win_end))
        excluded_ranges.append((win_start, win_end))

    # Return best-first. This IS the documented contract ("sorted by descending
    # excitement score") and two callers depend on it: the Groq and Ollama
    # fallbacks use only zones[0] to stay inside their tighter context windows.
    # A previous `selected.sort(key=lambda x: x[0])` here made zones[0] the
    # EARLIEST window instead of the most energetic one, so both fallbacks were
    # fed the opening minutes of the stream rather than the spike, and the
    # function silently contradicted its own docstring. The Gemini loop labels
    # every window explicitly, so its prompt content is unaffected by ordering.
    return selected


def _deduplicate_candidates(
    candidates: List[StreamerClipCandidate],
) -> List[StreamerClipCandidate]:
    """Remove overlapping clip candidates, keeping the highest-scored one per group.

    When multiple Gemini zone chunks return clips that overlap in time, we must
    collapse them to avoid sending duplicates downstream.  Overlap is defined as
    any intersection of the [start_time, end_time] intervals.
    """
    candidates.sort(key=lambda c: c.viral_score, reverse=True)
    kept: List[StreamerClipCandidate] = []
    for candidate in candidates:
        overlaps = any(
            not (candidate.end_time <= k.start_time or candidate.start_time >= k.end_time)
            for k in kept
        )
        if not overlaps:
            kept.append(candidate)
    return kept


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_streamer_detection_prompt(
    transcript_text: str,
    creator_name: str,
    num_clips: int = 3
) -> str:
    """Construct the Gemini system+user prompt for a single transcript chunk.

    Kept as a standalone function (not inlined) so unit tests can assert on the
    exact prompt string without invoking the API.
    """
    return f"""You are the world's #1 viral short-form video editor for top US Twitch and YouTube streamers (Kai Cenat, IShowSpeed, Jynxzi, CaseOh, xQc).
Your goal is to extract EXACTLY {num_clips} of the MOST EXPLOSIVE, FUNNY, OR UNHINGED VIRAL MOMENTS from this stream by {creator_name}.

### STREAMER TIKTOK VIRALITY FORMULA:
1. **Immediate Chaos / Hook (0-2s)**: The clip MUST start right before a massive spike in energy or right as the funny/shocking situation unfolds.
   - PURE RAGE: Screaming, table slams, rage quitting, unhinged disbelief.
   - COMEDY / FAILS: Getting roasted by chat/donations, failing a game in a ridiculous way, laughing uncontrollably.
   - DRAMA / BEEF: Confrontations, calling someone out, unfiltered hot takes.
   - HYPE / CLUTCH: Insane gaming clutch, massive celebration, unexpected twist.
2. **Optimal Pacing (20s - 45s Sweet Spot)**: Keep it tight and fast-paced! Never drag past 45 seconds.
3. **Standalone Cohesive Scene**: The moment must make complete sense on TikTok/Reels without context.
4. **Clean Complete Boundaries**: Do not cut off in the middle of a scream or punchline.
5. **TikTok Meme Hook Title**: Under 45 characters. Write like top TikTok clip channels and ALWAYS include 1 or 2 viral emojis at the end (e.g. "Nah Kai really started crashing out 💀", "Kai's cooked 😭", "No way Kai responded to this 💀😭", "Bro had Kevin Hart in tears 😂", "Kai Cenat NEVER beating these ALLEGATIONS 💀").
6. **Social Caption**: 1 short punchy sentence matching authentic TikTok viewer tone (e.g. "Bro was not ready for this at all.").
7. **HASHTAGS (STRICT REQUIREMENT)**: EXACTLY 2 to 4 hashtags ONLY! Never more than 4, never less than 2.
   Examples: ["#kaicenat", "#streamer", "#funny", "#fyp"]

### OUTPUT JSON FORMAT:
{{
  "clips": [
    {{
      "title": "Nah Kai really started crashing out 💀",
      "start_time": 120.5,
      "end_time": 155.0,
      "duration": 34.5,
      "viral_score": 95,
      "hook_reason": "High emotional spike and loud streamer reaction",
      "social_caption": "Wait until you see how this ended. Bro was not ready.",
      "hashtags": ["#kaicenat", "#streamer", "#funny", "#fyp"],
      "streamer_energy": "rage",
      "sfx_cues": [[0.1, "whoosh"], [12.0, "pop"], [32.0, "ding"]]
    }}
  ]
}}

STREAM TRANSCRIPT:
{transcript_text}
"""


# ---------------------------------------------------------------------------
# JSON parser / candidate builder
# ---------------------------------------------------------------------------

def parse_streamer_clips_json(
    raw_text: str,
    segments: List[TranscriptSegment],
    num_clips: int
) -> List[StreamerClipCandidate]:
    """Parse a raw Gemini/Groq JSON response into validated StreamerClipCandidate objects.

    Fixes D5: the old code applied the 45-second hard cap AFTER calling
    _normalize_clip_window(), which snaps endpoints to sentence boundaries.  When the
    snapped endpoint was already at a clean sentence break slightly beyond 45 s,
    re-capping it chopped the sentence mid-word.  The correct order is:
        1. Cap end-time to start + STREAMER_MAX_CLIP_DURATION_S (respects EOF via
           transcript_end)
        2. Snap to sentence boundary (_normalize_clip_window)
        3. Re-enforce cap with STREAMER_DURATION_QUANTIZATION_MARGIN_S slack to
           accommodate the 33 ms frame-boundary rounding that snapping may introduce.

    The num_clips ceiling is applied after sorting by viral_score so the caller
    always receives the best clips from this chunk, not just the first N.
    """
    clean_text = raw_text.strip()
    if clean_text.startswith("```json"):
        clean_text = clean_text[7:]
    if clean_text.startswith("```"):
        clean_text = clean_text[3:]
    if clean_text.endswith("```"):
        clean_text = clean_text[:-3]

    parsed = json.loads(clean_text.strip())
    clips_data = parsed.get("clips", [])
    candidates: List[StreamerClipCandidate] = []

    transcript_end = max((s.end for s in segments), default=9999.0)

    for c in clips_data:
        start = float(c.get("start_time", 0.0))
        end = float(c.get("end_time", start + 30.0))

        # D5 FIX — enforce duration cap BEFORE sentence snapping
        max_end = min(transcript_end, start + STREAMER_MAX_CLIP_DURATION_S)
        end = min(end, max_end)

        # Snap to natural sentence boundary
        normalized = _normalize_clip_window(start, end, segments)
        if normalized is not None:
            start, end = normalized
            # Re-enforce the V2 contract AFTER snapping. The shared normalizer
            # clamps against core.config's MIN/MAX_CLIP_DURATION (defaults 30/140s),
            # which are NOT the V2 bounds — so both ends must be re-clamped to
            # STREAMER_* here. Previously only the upper bound was re-applied, so a
            # snap that pulled the window under 30s produced a clip the gate could
            # only reject, after paying for a full render and upload.
            if end - start > STREAMER_MAX_CLIP_DURATION_S + STREAMER_DURATION_QUANTIZATION_MARGIN_S:
                end = start + STREAMER_MAX_CLIP_DURATION_S
            if end - start < STREAMER_MIN_CLIP_DURATION_S:
                end = min(transcript_end, max(end, start + STREAMER_MIN_CLIP_DURATION_S))
                if end - start < STREAMER_MIN_CLIP_DURATION_S:
                    # Too close to the transcript tail to grow forward — grow
                    # backwards rather than emit a sub-contract clip.
                    start = max(0.0, end - STREAMER_MIN_CLIP_DURATION_S)
        else:
            # Resilient fallback for streamer clips (bounds STREAMER_MIN_CLIP_DURATION_S to STREAMER_MAX_CLIP_DURATION_S)
            start = max(0.0, min(start, max(0.0, transcript_end - STREAMER_MIN_CLIP_DURATION_S)))
            end = min(
                transcript_end,
                max(start + STREAMER_MIN_CLIP_DURATION_S, min(end, start + STREAMER_MAX_CLIP_DURATION_S))
            )

        dur = end - start

        # Title formatting: Keep natural TikTok casing or uppercase with emojis
        raw_title = str(c.get("title", "STREAMER VIRAL MOMENT 💀")).strip()
        clean_title = "".join(
            ch for ch in unicodedata.normalize("NFC", raw_title)
            if not unicodedata.category(ch).startswith("C") and ch not in "\r\n\t"
        ).strip()
        if not clean_title:
            clean_title = "STREAMER VIRAL MOMENT 💀"

        # If title doesn't contain an emoji, append an authentic viral emoji based on energy
        energy_emojis = {
            "rage": "💀",
            "funny": "😭",
            "clutch": "🔥",
            "hype": "🔥",
            "drama": "💀😭",
        }
        has_emoji = any(
            (0x1F000 <= ord(ch) <= 0x1FAFF) or (0x2600 <= ord(ch) <= 0x27BF)
            for ch in clean_title
        )
        if not has_emoji:
            energy = str(c.get("streamer_energy", "funny")).lower().strip()
            clean_title = f"{clean_title} {energy_emojis.get(energy, '💀')}"

        caption = strip_emojis(str(c.get("social_caption", "Bro was not ready for this.")))

        # Enforce strictly 2 to 4 hashtags (user requirement)
        raw_hashtags = c.get("hashtags", [])
        if not isinstance(raw_hashtags, list):
            raw_hashtags = [str(raw_hashtags)]
        formatted_hashtags: List[str] = []
        for h in raw_hashtags:
            clean_h = strip_emojis(str(h)).strip()
            if not clean_h:
                continue
            tag = clean_h if clean_h.startswith("#") else f"#{clean_h}"
            tag = re.sub(r"[^\w#]", "", tag)
            if tag and tag not in formatted_hashtags:
                formatted_hashtags.append(tag)

        # Strictly 2 to 4 hashtags
        if len(formatted_hashtags) > 4:
            formatted_hashtags = formatted_hashtags[:4]
        elif len(formatted_hashtags) < 2:
            default_tags = ["#streamer", "#funny", "#fyp", "#viral"]
            for dt in default_tags:
                if dt not in formatted_hashtags and len(formatted_hashtags) < 4:
                    formatted_hashtags.append(dt)
        clean_hashtags = formatted_hashtags[:4]

        raw_sfx = c.get("sfx_cues", [])
        clean_sfx: List[Tuple[float, str]] = []
        for sfx in raw_sfx:
            if isinstance(sfx, (list, tuple)) and len(sfx) == 2:
                try:
                    s_t = float(sfx[0])
                    s_type = str(sfx[1]).lower().strip()
                    if s_type in ("whoosh", "pop", "ding"):
                        clean_sfx.append((s_t, s_type))
                except Exception:
                    pass

        if not clean_sfx:
            clean_sfx = [(0.1, "whoosh"), (max(1.0, dur - 1.5), "ding")]

        candidates.append(StreamerClipCandidate(
            title=clean_title,
            start_time=start,
            end_time=end,
            duration=dur,
            viral_score=_safe_int(c.get("viral_score", 90), default=90),
            hook_reason=strip_emojis(str(c.get("hook_reason", "High energy stream reaction"))),
            social_caption=caption,
            hashtags=clean_hashtags,
            streamer_energy=str(c.get("streamer_energy", "hype")),
            sfx_cues=clean_sfx
        ))

    candidates.sort(key=lambda x: x.viral_score, reverse=True)
    return candidates[:num_clips]


# ---------------------------------------------------------------------------
# Primary detection entry point
# ---------------------------------------------------------------------------

def detect_streamer_viral_moments(
    segments: List[TranscriptSegment],
    creator_name: str = "Streamer",
    num_clips: int = 3
) -> List[StreamerClipCandidate]:
    """Detects viral moments using multi-tier fallback tuned for streamer excitement.

    Fixes D11: the old implementation concatenated segments until the string hit
    40,000 characters and then stopped.  For a 4-hour stream that covers only the
    first ~25 minutes, so clips were always from the opening.

    New approach — two-pass chunked analysis:
      Pass 1 (fast, local): _prescan_high_energy_zones() scores every 20-minute window
        across the FULL transcript using cheap heuristics (exclamations, all-caps,
        keywords).  The top STREAMER_PRE_SCAN_TOP_ZONES non-overlapping windows are
        returned.
      Pass 2 (AI): each selected window is serialised to text (capped at
        STREAMER_TRANSCRIPT_CHUNK_CHARS) and sent to Gemini as a separate prompt.
        Candidates from all chunks are merged, deduplicated, and sorted by viral_score.

    The Groq and Ollama fallbacks still receive only the first prescan zone (to stay
    within their tighter context windows) but that zone is now the HIGHEST-energy
    segment of the stream, not just the opening.
    """
    # --- Pass 1: identify the most energetic windows across the full transcript ---
    zones = _prescan_high_energy_zones(segments)

    if not zones:
        # Edge case: empty or malformed transcript — fall through to heuristic
        print("[-] V2 Streamer Detector: Pre-scan found no zones; running heuristic fallback.")
        return _heuristic_streamer_detector(segments, creator_name=creator_name, num_clips=num_clips)

    # --- Pass 2: query AI models per zone, accumulate candidates ---

    def _query_zone(zone_start: float, zone_end: float) -> str | None:
        """Serialise one zone and return raw model text or None on failure."""
        zone_segs = [s for s in segments if s.start >= zone_start and s.end <= zone_end]
        if not zone_segs:
            return None
        chunk_text = _build_chunk_text(zone_segs)
        return _build_streamer_detection_prompt(chunk_text, creator_name, num_clips)

    all_candidates: List[StreamerClipCandidate] = []

    # 1. Primary: Gemini 3.8 Flash — query every pre-scanned zone
    if GEMINI_API_KEY:
        try:
            print(f"[*] V2 Streamer Detector: Querying Google Gemini Flash across {len(zones)} zone(s)...")
            for zone_start, zone_end in zones:
                prompt = _query_zone(zone_start, zone_end)
                if not prompt:
                    continue
                try:
                    res = query_gemini_models(prompt, GEMINI_API_KEY)
                    if res:
                        parsed = parse_streamer_clips_json(res, segments, num_clips)
                        all_candidates.extend(parsed)
                        print(f"    Zone [{zone_start:.0f}s-{zone_end:.0f}s]: {len(parsed)} candidate(s)")
                except Exception as zone_err:
                    print(f"[-] Gemini zone [{zone_start:.0f}s-{zone_end:.0f}s] failed: {zone_err}")

            if all_candidates:
                deduped = _deduplicate_candidates(all_candidates)
                final = deduped[:num_clips]
                print(f"[+] V2 Streamer Detector: {len(final)} viral moment(s) selected via Gemini!")
                return final
        except Exception as e:
            print(f"[-] Gemini streamer detection failed: {e}")

    # 2. Fallback: Groq — use the single highest-energy zone only
    if GROQ_API_KEY:
        try:
            print("[*] V2 Streamer Detector: Falling back to Groq...")
            best_zone_start, best_zone_end = zones[0]
            prompt = _query_zone(best_zone_start, best_zone_end)
            if prompt:
                res = query_groq_free_models(prompt, GROQ_API_KEY)
                if res:
                    parsed = parse_streamer_clips_json(res, segments, num_clips)
                    if parsed:
                        return parsed
        except Exception as e:
            print(f"[-] Groq streamer detection failed: {e}")

    # 3. Fallback: Ollama — use the single highest-energy zone only
    if OLLAMA_API_KEY:
        try:
            print("[*] V2 Streamer Detector: Falling back to Ollama Cloud...")
            best_zone_start, best_zone_end = zones[0]
            prompt = _query_zone(best_zone_start, best_zone_end)
            if prompt:
                res = query_ollama_cloud_models(prompt, OLLAMA_API_KEY)
                if res:
                    parsed = parse_streamer_clips_json(res, segments, num_clips)
                    if parsed:
                        return parsed
        except Exception:
            pass

    # 4. Fallback: Heuristic streamer density
    print("[*] V2 Streamer Detector: Running streamer excitement heuristic fallback...")
    return _heuristic_streamer_detector(segments, creator_name=creator_name, num_clips=num_clips)


# ---------------------------------------------------------------------------
# Title generator
# ---------------------------------------------------------------------------

def _generate_dynamic_streamer_title(creator_name: str, sample_text: str, energy: str) -> str:
    """Generates an authentic high-CTR TikTok hook title with viral emojis based on text context.

    Replaces the former random.choice() call with a deterministic hashlib-seeded
    index so that golden-master tests produce the same title for the same inputs
    across every run.  The seed is derived from creator_name, the first 40 chars of
    sample_text, and the energy label — three inputs that fully characterise the
    moment — so different moments still produce different titles.
    """
    clean_creator = creator_name.title() if creator_name and creator_name.lower() not in ("streamer", "streamers") else "Kai"
    lower = sample_text.lower()

    if any(k in lower for k in ["beast", "mrbeast", "jimmy"]):
        options = [
            f"Nah {clean_creator} really thought MrBeast was normal 💀",
            f"MrBeast had {clean_creator} completely terrified 😭💀",
            f"{clean_creator} and MrBeast had the wildest stream 💀🔥",
        ]
    elif any(k in lower for k in ["firework", "fireworks", "boom", "explosion", "bomb"]):
        options = [
            f"Nah {clean_creator} really lit fireworks in his room 💀😭",
            f"{clean_creator} almost burned his house down 💀🔥",
            "Bro had the whole room shaking 😭💀",
        ]
    elif any(k in lower for k in ["speed", "ishowspeed"]):
        options = [
            f"Speed and {clean_creator} are completely unhinged 💀😭",
            f"No way Speed did this on {clean_creator}'s stream 💀",
        ]
    elif energy == "rage" or any(k in lower for k in ["shut up", "hell no", "swear", "stop", "deadass"]):
        options = [
            f"Nah {clean_creator} really started crashing out 💀",
            f"{clean_creator} was completely fed up with this 💀😭",
            f"Bro {clean_creator} is actually so mad right now 😭",
            f"{clean_creator} caught screaming at chat 💀",
        ]
    elif energy == "funny" or any(k in lower for k in ["laugh", "bruh", "nah", "cooked", "chat"]):
        options = [
            f"Bro {clean_creator} is actually cooked 😭💀",
            f"Chat had {clean_creator} in tears laughing 😭",
            f"No way {clean_creator} actually did this 💀😭",
            f"{clean_creator} could NOT hold his laugh 😭💀",
        ]
    else:
        options = [
            f"Nah {clean_creator} is NOT real for this 💀😭",
            f"{clean_creator} caught completely off guard 💀",
            f"Wait till you see {clean_creator}'s reaction 😭🔥",
            f"{clean_creator} NEVER beating the allegations 💀",
            "Bro was NOT ready for what happened next 💀",
        ]

    # Deterministic index derived from stable inputs (fixes non-deterministic test failures)
    seed = int(hashlib.md5(
        f"{creator_name}{sample_text[:40]}{energy}".encode()
    ).hexdigest()[:8], 16)
    return options[seed % len(options)]


# ---------------------------------------------------------------------------
# Heuristic fallback detector
# ---------------------------------------------------------------------------

def _heuristic_streamer_detector(
    segments: List[TranscriptSegment],
    creator_name: str = "Streamer",
    num_clips: int = 3
) -> List[StreamerClipCandidate]:
    """Fast rule-based detection for exclamation marks, caps, and streamer slang.

    Fixes D6: the original code computed end = start + window_s without clamping to
    the actual end of the transcript.  For a stream that ends at e.g. 14,400 s, the
    last heuristic window could have end_time = 14,399 + 35 = 14,434 s — beyond EOF.
    This caused downstream clip-extraction to request frames that do not exist.

    The fix: compute transcript_end once from segments and clamp every window's end
    before it is appended to scored_windows.
    """
    scored_windows = []
    window_s = 35.0

    # D6 FIX — derive the true end of transcript once, clamp all windows against it
    transcript_end = max((s.end for s in segments), default=9999.0)

    for i, seg in enumerate(segments):
        start = seg.start
        # Clamp end to transcript_end so no window exceeds EOF
        end = min(start + window_s, transcript_end)
        window_segs = [s for s in segments if s.start >= start and s.end <= end]
        if not window_segs:
            continue
        text = " ".join(s.text for s in window_segs)
        score = 60
        # Exclamation marks & caps indicate shouting/rage
        score += text.count("!") * 4
        score += sum(1 for w in text.split() if w.isupper() and len(w) > 2) * 3
        # Keywords
        lower = text.lower()
        energy = "hype"
        for e_key, kw_list in STREAMER_KEYWORDS.items():
            for kw in kw_list:
                if kw in lower:
                    score += 5
                    energy = e_key

        scored_windows.append((score, start, end, text[:60], energy))

    scored_windows.sort(key=lambda x: x[0], reverse=True)
    results: List[StreamerClipCandidate] = []
    used_times: List[Tuple[float, float]] = []

    clean_creator_tag = f"#{creator_name.lower().replace(' ', '')}" if creator_name and creator_name.lower() != "streamer" else "#kaicenat"

    for score, start, end, sample, energy in scored_windows:
        overlap = any(not (end <= u_s or start >= u_e) for u_s, u_e in used_times)
        if overlap:
            continue
        used_times.append((start, end))

        title = _generate_dynamic_streamer_title(creator_name, sample, energy)

        results.append(StreamerClipCandidate(
            title=title,
            start_time=start,
            end_time=end,
            duration=end - start,
            viral_score=min(95, score),
            hook_reason="High density of vocal excitement and reactions",
            social_caption="Bro was not ready for this. Wait until you see how it ended.",
            hashtags=[clean_creator_tag, "#streamer", "#funny", "#fyp"],
            streamer_energy=energy,
            sfx_cues=[(0.1, "whoosh"), (end - start - 1.5, "ding")]
        ))
        if len(results) >= num_clips:
            break

    return results
