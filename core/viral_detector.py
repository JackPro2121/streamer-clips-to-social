import importlib
import json
import re
import time
from pathlib import Path

import requests
import warnings
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

# Support both google.genai (new official SDK) and google.generativeai
try:
    from google import genai
    from google.genai import types as genai_types
    HAS_NEW_GENAI = True
except ImportError:
    HAS_NEW_GENAI = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        legacy_genai: Any = None
        try:
            legacy_genai = importlib.import_module("google.generativeai")
        except ImportError:
            legacy_genai = None

from core.config import (
    GEMINI_API_KEY, GEMINI_API_KEYS, GROQ_API_KEY, OPENROUTER_API_KEY,
    OLLAMA_API_KEY, OLLAMA_MODEL, OLLAMA_BASE_URL, GEMINI_MODEL_LADDER,
    GROQ_MODEL_LADDER, MIN_CLIP_DURATION, MAX_CLIP_DURATION,
)

from core.transcriber import TranscriptSegment
from core.endpoint import word_boundary_ends as _complete_boundary_ends

# Suppress the non-blocking AFC function-calling advisory notice from google.genai
warnings.filterwarnings("ignore", message=".*Direct use of automatic function calling.*")

class ViralClipCandidate(BaseModel):
    title: str = Field(description="Catchy viral hook title (under 50 chars)")
    start_time: float = Field(description="Start time in seconds")
    end_time: float = Field(description="End time in seconds (must be 30-52s after start_time)")
    duration: float = Field(description="Duration in seconds (30.0 to 52.0)")
    viral_score: int = Field(description="Predicted virality score from 1 to 100")
    hook_reason: str = Field(description="Why this moment grabs immediate viewer attention")
    social_caption: str = Field(description="Ready-to-post engaging caption for TikTok/Reels/Shorts")
    hashtags: List[str] = Field(description="High-traffic relevant hashtags (e.g. ['#podcast', '#viral', '#mindset'])")
    # Who wrote the metadata. "llm" = a model wrote title/caption; "backfill" =
    # the semantic fallback derived them from raw words, and they may be
    # enriched before publish (see main._enrich_backfill_metadata).
    origin: str = Field(default="llm", description="llm or backfill")
    peak_intensity_segments: List[Tuple[float, float]] = Field(
        default_factory=list, 
        description="Segments within the clip (relative to start_time) where emotional intensity peaks, for automatic 1.2x zoom. Format: [[start, end], ...]"
    )
    sfx_cues: List[Tuple[float, str]] = Field(
        default_factory=list,
        description="Sound effect triggers relative to clip start: [[timestamp_sec, 'whoosh'|'pop'|'ding'], ...]"
    )
    keyword_emojis: dict[str, str] = Field(
        default_factory=dict,
        description="Key spoken words mapped to relevant visual emojis, e.g. {'money': '💰', 'growth': '🚀'}"
    )

# High-impact viral keywords mapped to emojis for dynamic subtitle graphics (Finance & Wealth focus)
DEFAULT_KEYWORD_EMOJIS = {
    "money": "💰", "cash": "💵", "dollar": "💵", "rich": "🤑", "wealth": "💎",
    "growth": "🚀", "grow": "🚀", "scale": "📈", "viral": "🔥", "fire": "🔥",
    "mind": "🧠", "brain": "🧠", "think": "💡", "idea": "💡", "secret": "🤫",
    "stop": "🛑", "danger": "⚠️", "warning": "⚠️", "win": "🏆", "winner": "🏆",
    "debt": "💳", "credit": "💳", "invest": "📊", "stock": "📈", "crypto": "🪙",
    "bitcoin": "🪙", "profit": "💸", "bank": "🏦", "save": "🐷", "tax": "📝",
    "million": "💰", "broke": "❌", "rule": "📜", "truth": "💯", "power": "⚡"
}

class ViralDetectionResponse(BaseModel):
    clips: List[ViralClipCandidate]

def format_transcript_with_timestamps(segments: List[TranscriptSegment], max_chars: int = 40000) -> str:
    """Formats transcript segments with start and end timestamps for LLM analysis."""
    lines = []
    total_len = 0
    for seg in segments:
        line = f"[{seg.start:.1f}s - {seg.end:.1f}s]: {seg.text}"
        total_len += len(line)
        if total_len > max_chars:
            lines.append("... [transcript truncated for token budget] ...")
            break
        lines.append(line)
    return "\n".join(lines)

def _gemini_image_part(image_path: Path) -> Optional[Any]:
    """Build an inline image part for the new SDK, or None if unreadable."""
    if not image_path.exists():
        return None
    data = image_path.read_bytes()
    if not data:
        return None
    return genai_types.Part.from_bytes(data=data, mime_type="image/jpeg")


# The attempt budget for ONE decision call.
#
# Real run 37301857585 took 54 minutes end to end: 4 keys x 4 models x up to 3
# attempts each, with a 60s HTTP timeout per request, had no ceiling. The budget
# bounds any single ``query_gemini_models`` call; the caller then falls through
# to the next tier (Ollama -> Groq -> OpenRouter -> semantic). Key rotation
# itself is cheap (429s fail in about a second), so the request ceiling still
# covers the observed recovery path: every key rotated on the first model, then
# the next model serves.
GEMINI_MAX_REQUESTS_PER_CALL = 16
GEMINI_CALL_WALL_CLOCK_S = 90.0


def query_gemini_models(
    prompt: str,
    key: str,
    image_path: Optional[Path] = None,
    context_note: str = "",
    model_ladder: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Queries Gemini across a model ladder and a key chain with exponential backoff.

    The ladder defaults to ``GEMINI_MODEL_LADDER`` (detection). Pass
    ``model_ladder`` for a task-specific set - the director-v2 vision call uses
    ``GEMINI_VISION_MODEL_LADDER`` so a frontier text model is not spent on a
    contact-sheet decision.

    ``key`` is the primary key; extra keys from ``GEMINI_API_KEYS`` are tried
    against the same model before dropping to the next model, so one busy or
    rate-limited key does not cost the whole ladder. Rotation fixes 429s for
    certain (limits are per key/project); capacity 503s only sometimes.

    Every request is charged against ``GEMINI_MAX_REQUESTS_PER_CALL`` and the
    call gives up at ``GEMINI_CALL_WALL_CLOCK_S`` so a wall of transient
    failures cannot multiply into a 50-minute run.

    When `image_path` is supplied the model also receives that image, which is
    how the director v2 pass "watches" the clip. The image rides in the same
    request rather than a second call, so quota is unaffected.
    """
    models_to_try = list(model_ladder) if model_ladder else (list(GEMINI_MODEL_LADDER) or ["gemini-3.5-flash-lite"])
    key_chain = [key] if not key else [key] + [extra for extra in GEMINI_API_KEYS if extra != key]
    deadline = time.monotonic() + GEMINI_CALL_WALL_CLOCK_S
    requests_made = 0
    image_part = _gemini_image_part(image_path) if (HAS_NEW_GENAI and image_path) else None
    if image_path is not None and image_part is None:
        print(f"[-] Image {Path(image_path).name} could not be attached; sending text only.")
    image_bytes = b""
    if image_path is not None and image_part is None and image_path.exists():
        image_bytes = image_path.read_bytes()
    for model_name in models_to_try:
        for key_index, active_key in enumerate(key_chain):
            key_label = f" [key {key_index + 1}/{len(key_chain)}]" if len(key_chain) > 1 else ""
            print(f"[*] Trying Gemini Flash ({model_name}){key_label}...")
            for attempt in range(3):
                if (
                    requests_made >= GEMINI_MAX_REQUESTS_PER_CALL
                    or time.monotonic() >= deadline
                ):
                    print(
                        f"[-] Gemini attempt budget exhausted after {requests_made} "
                        f"requests ({GEMINI_CALL_WALL_CLOCK_S:.0f}s cap); "
                        "falling through to the next tier."
                    )
                    return None
                requests_made += 1
                try:
                    response: Any
                    if HAS_NEW_GENAI:
                        # 60s HTTP timeout so a hung Gemini call can't stall the whole
                        # pipeline (the requests-based fallbacks already time out).
                        client = genai.Client(
                            api_key=active_key,
                            http_options=genai_types.HttpOptions(timeout=60_000),
                        )
                        contents: Any = prompt if image_part is None else [prompt, image_part]
                        response = client.models.generate_content(
                            model=model_name,
                            contents=contents,
                            config=genai_types.GenerateContentConfig(
                                # temperature/top_p/top_k are deprecated sampling
                                # parameters as of the 2026-09-01 changelog; the
                                # model default is used instead.
                                response_mime_type="application/json"
                            )
                        )
                        raw = (response.text or "").strip()
                    elif legacy_genai is not None:
                        legacy_genai.configure(api_key=active_key)
                        model = legacy_genai.GenerativeModel(model_name)
                        legacy_contents: Any = prompt
                        if image_bytes:
                            legacy_contents = [
                                prompt,
                                {"mime_type": "image/jpeg", "data": image_bytes},
                            ]
                        response = model.generate_content(
                            legacy_contents,
                            generation_config=legacy_genai.GenerationConfig(
                                response_mime_type="application/json"
                            ),
                            request_options={"timeout": 60},
                        )
                        raw = (response.text or "").strip()
                    else:
                        return None

                    if raw and len(raw) > 20:
                        if context_note:
                            print(f"[+] Gemini ({model_name}) returned a decision for {context_note}.")
                        return raw
                except Exception as e:
                    err_str = str(e)
                    transient = "503" in err_str or "429" in err_str or "UNAVAILABLE" in err_str
                    if transient and attempt < 2:
                        wait_sec = 2 ** (attempt + 1)
                        if time.monotonic() + wait_sec < deadline:
                            print(f"[-] Model {model_name} busy ({e}). Retrying in {wait_sec}s...")
                            time.sleep(wait_sec)
                        # No time left for a backoff: retry immediately; the
                        # budget check at the top of the loop bounds it.
                    elif transient and key_index + 1 < len(key_chain):
                        print(
                            f"[-] Model {model_name} busy on key {key_index + 1}; "
                            "trying the next key."
                        )
                        break
                    else:
                        print(f"[-] Model {model_name} failed: {e}")
                        break
    return None

def query_groq_free_models(prompt: str, key: str) -> Optional[str]:
    """Queries Groq free tier models (ultra-fast inference, $0 budget)."""
    models = list(GROQ_MODEL_LADDER)
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    for m in models:
        print(f"[*] Trying Groq Free Model ({m})...")
        try:
            payload: Dict[str, Any] = {
                "model": m,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "response_format": {"type": "json_object"}
            }
            r = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=25)
            if r.status_code == 200:
                content = r.json()["choices"][0]["message"]["content"]
                if content and len(content) > 20:
                    print(f"[+] Groq free model ({m}) returned viral moments successfully!")
                    return content
            else:
                print(f"[-] Groq {m} returned status {r.status_code}")
        except Exception as e:
            print(f"[-] Groq {m} error: {e}")
    return None

def query_openrouter_free_models(prompt: str, key: str) -> Optional[str]:
    """Queries OpenRouter verified 100% free models (:free tier)."""
    models = ["meta-llama/llama-3.3-70b-instruct:free", "qwen/qwen-2.5-72b-instruct:free"]
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/JackPro2121/podcasts-clips-to-social",
        "X-Title": "Podcast Clipper"
    }
    for m in models:
        print(f"[*] Trying OpenRouter Free Model ({m})...")
        try:
            payload: Dict[str, Any] = {
                "model": m,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "response_format": {"type": "json_object"}
            }
            r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=25)
            if r.status_code == 200:
                content = r.json()["choices"][0]["message"]["content"]
                if content and len(content) > 20:
                    print(f"[+] OpenRouter free model ({m}) returned viral moments successfully!")
                    return content
            else:
                print(f"[-] OpenRouter {m} returned status {r.status_code}")
        except Exception as e:
            print(f"[-] OpenRouter {m} error: {e}")
    return None

def query_ollama_cloud_models(
    prompt: str,
    key: str,
    model_name: str = OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL
) -> Optional[str]:
    """Queries Ollama Cloud endpoint (e.g., gemma4:31b) with bearer auth."""
    print(f"[*] Trying Ollama Cloud Model ({model_name})...")
    url = f"{base_url.rstrip('/')}/api/chat"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code == 200:
            data = r.json()
            content = data.get("message", {}).get("content", "").strip()
            if content and len(content) > 20:
                print(f"[+] Ollama Cloud model ({model_name}) returned viral moments successfully!")
                return content
        else:
            print(f"[-] Ollama Cloud ({model_name}) returned status {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"[-] Ollama Cloud ({model_name}) error: {e}")
    return None

def _safe_int(value: Any, default: int = 0) -> int:
    """Coerce an LLM-supplied score to int, tolerating floats/float-strings/None.

    A raw ``int("92.5")`` raised ValueError and discarded the ENTIRE parsed
    response (one bad field nuked all clips). We clamp to 1..100 as well.
    """
    try:
        n = int(round(float(value)))
    except (TypeError, ValueError):
        return default
    return max(1, min(100, n))


def strip_emojis(text: str) -> str:
    """Removes all emoji characters to enforce clean signature broadcast aesthetic."""
    emoji_pattern = re.compile(
        "["
        "\U00010000-\U0010ffff"
        "\u2600-\u27bf"
        "\u2300-\u23ff"
        "\u2b50-\u2b55"
        "\u200d"
        "\ufe0f"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r"", text).strip()


def _safe_start_boundary(segments: List[TranscriptSegment], requested: float, lookback: float = 8.0) -> float:
    boundaries = [end for end in _complete_boundary_ends(segments) if requested - lookback <= end <= requested]
    return boundaries[-1] if boundaries else requested


def _safe_end_boundary(
    segments: List[TranscriptSegment],
    requested: float,
    start: float,
    max_end: float,
) -> Optional[float]:
    boundaries = _complete_boundary_ends(segments)
    future = [end for end in boundaries if requested - 0.05 <= end <= max_end]
    if future:
        return future[0]
    prior = [end for end in boundaries if start + MIN_CLIP_DURATION <= end < requested]
    return prior[-1] if prior else None


def _normalize_clip_window(
    start: float,
    end: float,
    segments: List[TranscriptSegment],
) -> Optional[tuple[float, float]]:
    if not segments:
        return None
    transcript_end = max(segment.end for segment in segments)
    if transcript_end < MIN_CLIP_DURATION:
        return None
    start = max(0.0, start)
    start = min(start, transcript_end - MIN_CLIP_DURATION)
    end = max(end, start + MIN_CLIP_DURATION)
    end = min(end, start + MAX_CLIP_DURATION, transcript_end)
    if any(segment.words for segment in segments):
        start = min(_safe_start_boundary(segments, start), start)
        safe_end = _safe_end_boundary(
            segments,
            requested=end,
            start=start,
            max_end=start + MAX_CLIP_DURATION,
        )
        if safe_end is None:
            return None
        end = safe_end
    if end - start < MIN_CLIP_DURATION or end - start > MAX_CLIP_DURATION:
        return None
    if end > transcript_end + 0.01:
        return None
    return round(float(start), 2), round(float(end), 2)


NICHE_PROFILES = {
    "finance": {
        "focus": "personal finance, wealth creation, debt, income, investing, and financial freedom",
        "hashtags": ["#finance", "#money", "#wealth", "#investing", "#financialfreedom"],
    },
    "business": {
        "focus": "business strategy, entrepreneurship, growth, sales, leadership, and ownership",
        "hashtags": ["#business", "#entrepreneur", "#startup", "#growth", "#leadership"],
    },
    "ai_tech": {
        "focus": "artificial intelligence, software, technology, automation, and the future of work",
        "hashtags": ["#ai", "#technology", "#automation", "#startups", "#future"],
    },
    "health_longevity": {
        "focus": "health, fitness, longevity, medical research, habits, and human performance",
        "hashtags": ["#health", "#longevity", "#fitness", "#wellness", "#humanperformance"],
    },
    "real_estate": {
        "focus": "real estate, housing, investing, property markets, and financial independence",
        "hashtags": ["#realestate", "#property", "#investing", "#housing", "#wealth"],
    },
    "mindset": {
        "focus": "mindset, behavior, motivation, discipline, decision-making, and personal growth",
        "hashtags": ["#mindset", "#motivation", "#selfimprovement", "#discipline", "#growth"],
    },
}


def parse_clips_json(
    raw_text: str,
    segments: List[TranscriptSegment],
    num_clips: int,
    default_hashtags: Optional[List[str]] = None,
) -> List[ViralClipCandidate]:
    """Parses raw LLM JSON into validated ViralClipCandidate objects."""
    clean_text = raw_text.strip()
    if clean_text.startswith("```json"):
        clean_text = clean_text[7:]
    if clean_text.startswith("```"):
        clean_text = clean_text[3:]
    if clean_text.endswith("```"):
        clean_text = clean_text[:-3]

    parsed = json.loads(clean_text.strip())
    clips_data = parsed.get("clips", [])
    
    candidates = []
    for c in clips_data:
        start = float(c.get("start_time", 0.0))
        end = float(c.get("end_time", start + 40.0))
        normalized = _normalize_clip_window(start, end, segments)
        if normalized is None:
            continue
        start, end = normalized
        dur = end - start

        candidate_title = strip_emojis(re.sub(r'[^\w\s\-\'\,\.\?]', '', str(c.get("title", "Viral Moment"))).strip().upper())
        clean_caption = strip_emojis(str(c.get("social_caption", "Wait until the end. #shorts")))
        raw_hashtags = c.get("hashtags", ["#podcast", "#viral", "#shorts"])
        clean_hashtags = [strip_emojis(str(h)).strip() for h in raw_hashtags if strip_emojis(str(h)).strip()]

        # Extract or synthesize sfx_cues
        raw_sfx = c.get("sfx_cues", [])
        clean_sfx = []
        for s in raw_sfx:
            if isinstance(s, (list, tuple)) and len(s) == 2:
                try:
                    s_t = float(s[0])
                    s_type = str(s[1]).lower().strip()
                    if s_type in ("whoosh", "pop", "ding"):
                        clean_sfx.append((s_t, s_type))
                except (ValueError, TypeError):
                    continue

        # If no SFX cues provided, synthesize intelligent defaults based on peak segments
        if not clean_sfx:
            clean_sfx.append((0.1, "whoosh"))  # Intro hook transition
            for p_seg in c.get("peak_intensity_segments", []):
                if isinstance(p_seg, (list, tuple)) and len(p_seg) >= 1:
                    clean_sfx.append((float(p_seg[0]), "whoosh"))
            if dur > 15:
                clean_sfx.append((round(dur - 2.0, 1), "ding"))

        raw_kw_emojis = c.get("keyword_emojis", {})
        merged_emojis = dict(DEFAULT_KEYWORD_EMOJIS)
        if isinstance(raw_kw_emojis, dict):
            merged_emojis.update({k.lower().strip(): str(v).strip() for k, v in raw_kw_emojis.items()})

        raw_peaks = c.get("peak_intensity_segments", [])
        peak_segments: List[Tuple[float, float]] = []
        if isinstance(raw_peaks, list):
            for peak in raw_peaks:
                if isinstance(peak, (list, tuple)) and len(peak) >= 2:
                    try:
                        peak_segments.append((float(peak[0]), float(peak[1])))
                    except (TypeError, ValueError):
                        continue

        candidate = ViralClipCandidate(
            title=candidate_title or "VIRAL MOMENT",
            start_time=start,
            end_time=end,
            duration=dur,
            viral_score=_safe_int(c.get("viral_score", 85), default=85),
            hook_reason=strip_emojis(str(c.get("hook_reason", "High engagement segment"))),
            social_caption=clean_caption,
            hashtags=clean_hashtags or default_hashtags or ["#finance", "#money", "#wealth", "#investing", "#financialfreedom"],
            peak_intensity_segments=peak_segments,
            sfx_cues=clean_sfx,
            keyword_emojis=merged_emojis
        )
        candidates.append(candidate)

    candidates.sort(key=lambda x: x.viral_score, reverse=True)
    return candidates[:num_clips]

def detect_viral_moments(
    segments: List[TranscriptSegment],
    num_clips: int = 3,
    api_key: Optional[str] = None,
    niche: str = "finance",
) -> List[ViralClipCandidate]:
    """
    Multi-Tier Zero-Cost Autonomous AI Viral Detection:
    1. Primary: Google Gemini Flash (owner Pro-plan key; free-tier 503s are
       retried with backoff inside query_gemini_models)
    2. Fallback 1: Ollama Cloud (gemma4:31b, dedicated capacity)
    3. Fallback 2: Groq Free Tier (groq/compound-mini, openai/gpt-oss-120b)
    4. Fallback 3: OpenRouter Free Tier (minimax/minimax-m3:free)
    5. Fallback 4: Semantic topic extraction from spoken dialogue
    """
    transcript_text = format_transcript_with_timestamps(segments)
    profile = NICHE_PROFILES.get(niche, NICHE_PROFILES["finance"])
    default_hashtags = list(profile["hashtags"])

    prompt = f"""
You are the world's top viral short-form video editor and content strategist specializing in {profile["focus"]} (on TikTok, Instagram Reels, and YouTube Shorts).
Your goal is to analyze the following podcast transcript and extract EXACTLY {num_clips} distinct, non-overlapping VIRAL moments.

### VIRALITY CRITERIA:
1. **Immediate Scroll-Stopping Hook (0-3s)**: The clip MUST open directly on an immediate attention-grabbing line about {profile["focus"]}. Prioritize either:
   (A) SHOCK & CURIOSITY: An unbelievable revelation, shocking dollar/income number, counter-intuitive statistic, or massive realization (e.g. 'I lost $400,000 in one week doing this').
   (B) CONTRARIAN DEBATE / CONTROVERSY: A heated disagreement, provocative stance, or intense debate that triggers strong opinions and comments.
   Never start on host greetings, pleasantries, chitchat, or filler words ('um', 'uh', 'so', 'you know', 'yeah'). The first 3 seconds decide viral retention on TikTok, Shorts, and Reels. Start at the exact second the core hook begins.
2. **High Emotional Intensity & Retention Pacing**: Fast-paced payoff, intense debate, or shocking breakdown that keeps the audience glued until the final second.
3. **Standalone Cohesion**: The clip must make complete sense on its own without needing the rest of the 2-hour podcast.
4. **Optimal Duration (30-50s Sweet Spot)**: The viral sweet spot for TikTok, Reels, and Shorts is strictly **30 to 50 seconds** (maximum 52 seconds). Never select a clip over 52 seconds. Pick concise, high-retention stories with rapid payoff.
5. **Complete Boundaries**: The start must begin at a complete thought and the end must land after a complete sentence or question. Never end on a partial word, a dangling conjunction, or a sentence fragment.
6. **Exact Timestamps**: Use the provided transcript timestamps to specify precise start_time and end_time.
7. **Punchy Curiosity-Gap Title**: Give each clip an engaging, high-CTR hook title in ALL CAPS (e.g., "THE $100,000 CREDIT CARD MISTAKE", "WHY YOU WILL NEVER RETIRE RICH", "THE 3 MONEY RULES OF MILLIONAIRES"). Max 5-7 words. Never include filler words ("um", "uh", "yeah"), and strictly DO NOT include emojis or special symbols.
8. **STRICTLY NO EMOJIS IN METADATA**: Under NO circumstances use emojis in titles, social captions, or hashtags. Maintain an elite, clean broadcast aesthetic.
9. **Director Cues**:
   - Identify 1-3 "Peak Intensity" segments for automatic 1.2x zoom. Provide relative start and end seconds.
   - Suggest sound effect triggers in "sfx_cues": e.g. [[0.1, "whoosh"], [15.2, "pop"], [32.0, "ding"]].
   - Select 2-5 high-impact keywords for "keyword_emojis" (e.g. {{"money": "💰", "focus": "🎯"}}).
10. **STRICT CLIP COUNT**: You MUST return EXACTLY {num_clips} items in the "clips" array (not 1, but {num_clips} distinct, non-overlapping moments from across the podcast).

### PODCAST TRANSCRIPT:
{transcript_text}

### OUTPUT FORMAT:
Output MUST be valid JSON only matching this schema with EXACTLY {num_clips} items in "clips":
{{
  "clips": [
    {{
      "title": "PUNCHY VIRAL TITLE ONE",
      "start_time": 124.5,
      "end_time": 168.0,
      "duration": 43.5,
      "viral_score": 95,
      "hook_reason": "Opens with a shocking contrarian statement about wealth.",
      "social_caption": "This perspective changes everything. Drop your thoughts below.",
      "hashtags": ["#mindset", "#podcast", "#success", "#reels"],
      "peak_intensity_segments": [[5.0, 12.0], [25.0, 32.0]],
      "sfx_cues": [[0.1, "whoosh"], [5.0, "whoosh"], [20.4, "pop"], [40.0, "ding"]],
      "keyword_emojis": {{"wealth": "💰", "mindset": "🧠"}}
    }}
  ]
}}
Do not include markdown backticks or commentary outside the JSON.
"""

    gemini_key = api_key or GEMINI_API_KEY
    raw_text = None

    # Tier 1: Google Gemini Flash. Primary again per owner direction (Pro-plan
    # key): Gemini writes the richest captions, and query_gemini_models already
    # retries 503/429 with backoff across the model ladder. The paid key removes
    # the free-tier quota cliff that caused three consecutive 503 runs.
    if gemini_key:
        print("[*] Tier 1: Sending transcript to Google Gemini Flash...")
        raw_text = query_gemini_models(prompt, gemini_key)

    # Tier 2: Ollama Cloud (gemma4:31b) - dedicated capacity fallback
    if not raw_text and OLLAMA_API_KEY:
        print(f"[*] Tier 2: Falling back to Ollama Cloud ({OLLAMA_MODEL})...")
        raw_text = query_ollama_cloud_models(prompt, OLLAMA_API_KEY, OLLAMA_MODEL, OLLAMA_BASE_URL)

    # Tier 3: Groq Free Tier
    if not raw_text and GROQ_API_KEY:
        print("[*] Tier 3: Falling back to Groq free models...")
        raw_text = query_groq_free_models(prompt, GROQ_API_KEY)

    # Tier 4: OpenRouter Free Tier
    if not raw_text and OPENROUTER_API_KEY:
        print("[*] Tier 4: Falling back to OpenRouter free models...")
        raw_text = query_openrouter_free_models(prompt, OPENROUTER_API_KEY)

    # Parse JSON if any LLM responded
    if raw_text:
        try:
            candidates = parse_clips_json(
                raw_text,
                segments,
                num_clips,
                default_hashtags=default_hashtags,
            )
            if candidates:
                print(f"[+] Successfully detected {len(candidates)} viral candidates via AI!")
                return candidates
        except Exception as e:
            print(f"[-] Parsing AI response failed ({e}). Falling back to semantic topic detector.")

    # Last resort: semantic dialogue topic extraction
    print("[*] Semantic fallback: using intelligent semantic topic detector.")
    return fallback_rule_based_detector(segments, num_clips, niche=niche)

def fallback_rule_based_detector(segments: List[TranscriptSegment], num_clips: int = 3, niche: str = "finance") -> List[ViralClipCandidate]:
    """Intelligent semantic fallback that extracts meaningful topic titles from transcript speech."""
    if not segments:
        return []

    total_duration = segments[-1].end - segments[0].start
    if total_duration < MIN_CLIP_DURATION:
        return []
    profile = NICHE_PROFILES.get(niche, NICHE_PROFILES["finance"])
    step = total_duration / (num_clips + 1)
    
    candidates = []
    for i in range(num_clips):
        target_start = segments[0].start + step * (i + 0.5)
        # Find closest segment
        closest_seg = min(segments, key=lambda s: abs(s.start - target_start))
        start_t = closest_seg.start
        end_t = min(start_t + 45.0, segments[-1].end)
        normalized = _normalize_clip_window(start_t, end_t, segments)
        if normalized is None:
            continue
        start_t, end_t = normalized
        
        # Extract speech text within this window to form a relevant semantic title
        chunk_words = []
        for s in segments:
            if s.end >= start_t and s.start <= end_t:
                chunk_words.extend(s.text.split())
        
        # Derive a punchy, grammatically sensible title from the opening statement
        stop_words = {
            "um", "uh", "like", "so", "you", "know", "mean", "for", "me", "but", "and", "yeah",
            "well", "actually", "basically", "right", "okay", "just", "the", "a", "an", "to", "in",
            "it", "did", "do", "does", "done", "was", "were", "is", "am", "are", "be", "been",
            "being", "have", "has", "had", "they", "them", "their", "this", "that", "these",
            "those", "what", "how", "why", "who", "when", "where", "which", "want", "wanted",
            "wants", "got", "get", "gets", "say", "said", "would", "could", "should", "some",
            "any", "not", "too", "also", "with", "from", "at", "by", "on", "as", "if"
        }
        raw_sentence = " ".join(chunk_words[:25])
        clean_content = [
            re.sub(r'[^\w\s]', '', w)
            for w in chunk_words[:30]
            if len(w) > 2 and w.lower() not in stop_words
        ]

        derived_title = ""
        first_sentence = re.split(r"[.!?]", raw_sentence)[0].strip()
        first_words = [w for w in first_sentence.split() if w.lower() not in {"um", "uh", "like", "so"}]
        if 3 <= len(first_words) <= 6:
            candidate_hook = " ".join(first_words).upper()
            if len(candidate_hook) <= 45:
                derived_title = re.sub(r'[^\w\s]', '', candidate_hook).strip()

        if not derived_title and len(clean_content) >= 2:
            key_phrase = " ".join(clean_content[:3]).upper()
            derived_title = f"THE TRUTH ABOUT {key_phrase}"
            if len(derived_title) > 42:
                derived_title = f"{key_phrase} STRATEGY"

        if not derived_title:
            focus_val = str(profile.get("focus") or "PODCAST")
            niche_clean = focus_val.split()[0].upper()
            derived_title = f"POWERFUL {niche_clean} INSIGHT #{i+1}"
        
        dur = round(end_t - start_t, 1)

        # Transcript-derived caption for the no-LLM path. The old template
        # repeated the title and appended a generic CTA ("What are your
        # thoughts on this?"), which read as machine-written on a published
        # post; the opening spoken sentences read like a human summary.
        speech_sentences = [
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+", " ".join(chunk_words).strip())
            if part.strip()
        ]
        caption_text = " ".join(speech_sentences[:2]).strip()
        if len(caption_text) > 220:
            caption_text = caption_text[:220].rsplit(" ", 1)[0].rstrip(",;:") + "..."
        if not caption_text:
            caption_text = "A moment from this episode worth hearing twice."
        peaks: List[Tuple[float, float]] = [(5.0, 10.0), (20.0, 25.0)] if dur > 30 else [(3.0, 7.0)]
        sfx = [(0.1, "whoosh"), (peaks[0][0], "whoosh")]
        if dur > 20:
            sfx.append((round(dur - 2.0, 1), "ding"))

        candidates.append(ViralClipCandidate(
            title=derived_title,
            start_time=round(start_t, 1),
            end_time=round(end_t, 1),
            duration=dur,
            viral_score=80 - (i * 5),
            hook_reason="Engaging dialogue section with high-retention speech",
            social_caption=strip_emojis(caption_text),
            hashtags=list(profile["hashtags"]),
            origin="backfill",
            peak_intensity_segments=peaks,
            sfx_cues=sfx,
            keyword_emojis=dict(DEFAULT_KEYWORD_EMOJIS)
        ))
    return candidates
