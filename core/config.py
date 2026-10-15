import os
from pathlib import Path
from typing import Dict, Any, List, Mapping, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def _resolve_project_root() -> Path:
    """Locate the consumer project root that owns ``assets/`` and output dirs.

    ``clips-core`` runs both inside this repo (``core/`` beside ``assets/``) and
    as an installed package under ``site-packages``, where ``__file__`` no
    longer points at a repo root. Order: explicit override, the dev/monorepo
    layout, then the current working directory.
    """
    override = os.getenv("CLIPS_CORE_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    candidate = Path(__file__).resolve().parent.parent
    if (candidate / "assets").is_dir() or (candidate / "pyproject.toml").is_file():
        return candidate
    return Path.cwd()


def _resolve_models_dir() -> Path:
    """Directory holding the ONNX / ``.pth`` weights.

    Weights are large (200 MB+) and never shipped in the package, so the path is
    resolved at runtime. Order: explicit override, the package-relative
    ``models/`` when present (monorepo layout), then ``<root>/models``.
    """
    override = os.getenv("CLIPS_CORE_MODELS_DIR")
    if override:
        return Path(override).expanduser().resolve()
    package_models = Path(__file__).resolve().parent / "models"
    if package_models.is_dir():
        return package_models
    return BASE_DIR / "models"


BASE_DIR = _resolve_project_root()
DOWNLOADS_DIR = BASE_DIR / "downloads"
CLIPS_DIR = BASE_DIR / "clips"
SUBTITLES_DIR = BASE_DIR / "subtitles"
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = Path(os.getenv("CLIPS_CORE_ASSETS_DIR") or (BASE_DIR / "assets"))
AUDIO_ASSETS_DIR = ASSETS_DIR / "audio"
SFX_ASSETS_DIR = ASSETS_DIR / "sfx"
FONTS_DIR = ASSETS_DIR / "fonts"
BROLL_DIR = ASSETS_DIR / "broll"
HISTORY_FILE = DATA_DIR / "history.json"
MODELS_DIR = _resolve_models_dir()

# Ensure directories exist
DOWNLOADS_DIR.mkdir(exist_ok=True, parents=True)
CLIPS_DIR.mkdir(exist_ok=True, parents=True)
SUBTITLES_DIR.mkdir(exist_ok=True, parents=True)
DATA_DIR.mkdir(exist_ok=True, parents=True)
AUDIO_ASSETS_DIR.mkdir(exist_ok=True, parents=True)
SFX_ASSETS_DIR.mkdir(exist_ok=True, parents=True)
FONTS_DIR.mkdir(exist_ok=True, parents=True)
BROLL_DIR.mkdir(exist_ok=True, parents=True)

# API Keys & Credentials
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


def collect_gemini_api_keys(environ: Optional[Mapping[str, str]] = None) -> List[str]:
    """All configured Gemini keys, primary first, deduplicated.

    Rotation is resilience: rate limits (429) are per key/project, so a second
    key can carry a run the first one has exhausted. Capacity 503s are not
    guaranteed to differ per key, but trying one more key costs almost nothing
    when the ladder already has to walk models. Declare extras as
    ``GEMINI_API_KEY_2``..``GEMINI_API_KEY_4`` or comma-separated in
    ``GEMINI_API_KEYS``.
    """
    source = environ if environ is not None else os.environ
    candidates: List[str] = []
    for name in ("GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GEMINI_API_KEY_4"):
        value = (source.get(name) or "").strip()
        if value:
            candidates.append(value)
    for part in (source.get("GEMINI_API_KEYS") or "").split(","):
        part = part.strip()
        if part:
            candidates.append(part)
    ordered: List[str] = []
    for candidate in candidates:
        if candidate not in ordered:
            ordered.append(candidate)
    return ordered


GEMINI_API_KEYS = collect_gemini_api_keys()
BUFFER_ACCESS_TOKEN = (
    os.getenv("BUFFER_ACCESS_TOKEN") or
    os.getenv("BUFFER_API_KEY") or
    ""
)
BUFFER_CHANNEL_IDS = [
    cid.strip() for cid in os.getenv("BUFFER_CHANNEL_IDS", "").split(",") if cid.strip()
]
CHANNEL_WATERMARK = os.getenv("CHANNEL_WATERMARK", "@allinonepodcastsss")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_REPOSITORY = os.getenv("GITHUB_REPOSITORY", "")
APIFY_API_TOKEN = (
    os.getenv("APIFY_API_TOKEN") or
    os.getenv("APIFY_KEY") or
    os.getenv("apify_key") or
    os.getenv("APIFY_API_TOKEN_NEW") or
    ""
)
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "").strip()
# The maintained Streamers actor is retained for explicitly requested full
# downloads. It does not support time ranges, so it must never be used for
# clip extraction. The segment actor accepts startTime/endTime and is invoked
# only after the viral detector selects a short range.
APIFY_FULL_DOWNLOAD_ACTOR_ID = os.getenv(
    "APIFY_FULL_DOWNLOAD_ACTOR_ID", "streamers/youtube-video-downloader"
).strip()
APIFY_SEGMENT_ACTOR_ID = os.getenv(
    "APIFY_SEGMENT_ACTOR_ID", "vidkraken/youtube-video-audio-downloader-reliable"
).strip()
APIFY_TRANSCRIPT_ACTOR_ID = os.getenv(
    "APIFY_TRANSCRIPT_ACTOR_ID", "pintostudio/youtube-transcript-scraper"
).strip()
CLIP_ONLY_MODE = os.getenv("CLIP_ONLY_MODE", "true").lower() in ("true", "1", "yes")
MAX_DISCOVERY_CANDIDATES = max(1, int(os.getenv("MAX_DISCOVERY_CANDIDATES", "12")))
MAX_TRANSCRIPT_FALLBACKS = max(1, int(os.getenv("MAX_TRANSCRIPT_FALLBACKS", "3")))

# Clip length gates. The documented short-form sweet spot is 30-60s, but the
# detector historically accepted up to 140s so it can still return *something*
# when a transcript has no 30-60s window with a complete thought. Widening these
# bounds risks the "no clips were rendered" failure, so they are env-tunable and
# a clip longer than SHORT_FORM_MAX_DURATION is reported as a plan warning
# instead of being silently accepted.
MIN_CLIP_DURATION = float(os.getenv("MIN_CLIP_DURATION", "30"))
MAX_CLIP_DURATION = float(os.getenv("MAX_CLIP_DURATION", "140"))
SHORT_FORM_MAX_DURATION = float(os.getenv("SHORT_FORM_MAX_DURATION", "60"))
if MAX_CLIP_DURATION < MIN_CLIP_DURATION:
    MAX_CLIP_DURATION = MIN_CLIP_DURATION
# Gemini model ladder. Ordered strongest-first; query_gemini_models walks the
# ladder and rotates on 429/503 so a busy or deprecating model never fails a run.
# Override with GEMINI_MODEL_LADDER="gemini-3.8-flash,gemini-3.5-flash-lite".
#
# FREE-TIER MODELS ONLY (owner policy, 2026-10-04). The AI Studio model list
# marks gemini-3.6-flash and gemini-3.5-flash as Paid, so they are excluded.
# Verified live 2026-10-04:
# * gemini-2.0-flash and gemini-2.5-flash-lite return 404 (already shut down).
# * gemini-2.5-flash still answers but is scheduled to shut down 2026-10-16.
# * gemini-3.7-flash is alive but can be congested (504 on the test day);
#   gemini-3.5-flash-lite and gemini-3.1-flash-lite are stable with no
#   announced shutdown before 2027 and give the ladder a durable tail.
GEMINI_MODEL_LADDER = [
    model.strip()
    for model in (
        os.getenv("GEMINI_MODEL_LADDER")
        or "gemini-3.8-flash,gemini-3.7-flash,"
           "gemini-3.5-flash-lite,gemini-3.1-flash-lite"
    ).split(",")
    if model.strip()
]

# Director-v2 "watches" the contact sheet; only that call needs a vision model,
# so it gets its own ladder instead of the detection one. Both entries are
# free-tier: gemini-3.5-flash-lite was verified live with a real frame
# returning valid JSON; gemini-3.1-flash-lite is the fallback.
GEMINI_VISION_MODEL_LADDER = [
    model.strip()
    for model in (
        os.getenv("GEMINI_VISION_MODEL_LADDER")
        or "gemini-3.5-flash-lite,gemini-3.1-flash-lite"
    ).split(",")
    if model.strip()
]
GROQ_API_KEY = (
    os.getenv("GROQ_API_KEY") or
    os.getenv("groq_api_key") or
    ""
)
# Groq model ladder. Ordered strongest-first; query_groq_free_models walks the
# ladder and rotates on errors so a deprecating or rate-limited model never fails a run.
# Override with GROQ_MODEL_LADDER="llama-3.3-70b-versatile,llama-3.1-8b-instant,gemma2-9b-it".
GROQ_MODEL_LADDER = [
    model.strip()
    for model in (
        os.getenv("GROQ_MODEL_LADDER")
        or "llama-3.3-70b-versatile,llama-3.1-8b-instant,gemma2-9b-it"
    ).split(",")
    if model.strip()
]
OPENROUTER_API_KEY = (
    os.getenv("OPENROUTER_API_KEY") or
    os.getenv("openrouter_api_key") or
    ""
)
OLLAMA_API_KEY = (
    os.getenv("OLLAMA_API_KEY") or
    os.getenv("ollama_api_key") or
    ""
)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL") or "gemma4:31b"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL") or "https://ollama.com"
PEXELS_API_KEY = (
    os.getenv("PEXELS_API_KEY") or
    os.getenv("pexels_api_key") or
    ""
)
ENABLE_BROLL = os.getenv("ENABLE_BROLL", "false").lower() in ("true", "1", "yes")
CHOCODATA_API_KEY = (
    os.getenv("CHOCODATA_API_KEY") or
    os.getenv("chocodata_api_key") or
    ""
)
YOUTUBE_COOKIES = os.getenv("YOUTUBE_COOKIES", "")
YTDLP_PROXY = (
    os.getenv("YTDLP_PROXY") or
    os.getenv("HTTP_PROXY") or
    os.getenv("PROXY_URL") or
    ""
)
RAPIDAPI_KEY = (
    os.getenv("RAPIDAPI_KEY") or
    os.getenv("rapidapi_key") or
    ""
)
COBALT_API_URL = os.getenv("COBALT_API_URL", "https://api.cobalt.tools")
COBALT_INSTANCES = [
    url.strip() for url in (
        os.getenv("COBALT_INSTANCES") or
        COBALT_API_URL
    ).split(",") if url.strip()
]
# Leave the local Proof-of-Origin service optional. Sending yt-dlp to a dead
# localhost service makes every extractor client fail before it can try its
# normal authenticated/cookie-based path.
YTDLP_POT_PROVIDER_URL = os.getenv("YTDLP_POT_PROVIDER_URL", "").strip()
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")
SLACK_CHANNEL = os.getenv("SLACK_CHANNEL", "podcast-clip")

# Video Ingestion Standards (Strict Quality Gate)
MIN_VIDEO_HEIGHT = 720
MAX_VIDEO_HEIGHT = 1080

# Visual Retention & Anti-Fingerprint Features
ENABLE_PUNCH_ZOOM = os.getenv("ENABLE_PUNCH_ZOOM", "true").lower() in ("true", "1", "yes")
ENABLE_TOP_HOOK_BADGE = os.getenv("ENABLE_TOP_HOOK_BADGE", "true").lower() in ("true", "1", "yes")
HOOK_BADGE_DURATION = float(os.getenv("HOOK_BADGE_DURATION", "3.0"))  # 3-second initial hook retention badge
HOOK_BADGE_MARGIN_V = int(os.getenv("HOOK_BADGE_MARGIN_V", "95"))     # Safe zone (upper safe zone 90-140px, above speaker forehead)
ENABLE_BGM = os.getenv("ENABLE_BGM", "true").lower() in ("true", "1", "yes")  # Breaks audio fingerprinting
ENABLE_SFX = os.getenv("ENABLE_SFX", "true").lower() in ("true", "1", "yes")  # Transitions & keyword sound design
ENABLE_DYNAMIC_DUCKING = os.getenv("ENABLE_DYNAMIC_DUCKING", "true").lower() in ("true", "1", "yes")  # Sidechain BGM ducking
ENABLE_FILM_GRAIN = os.getenv("ENABLE_FILM_GRAIN", "true").lower() in ("true", "1", "yes")  # Breaks visual pHash
UNIVERSAL_EDITOR_SHADOW = os.getenv("UNIVERSAL_EDITOR_SHADOW", "true").lower() in ("true", "1", "yes")
UNIVERSAL_EDITOR_ENFORCE_QA = os.getenv("UNIVERSAL_EDITOR_ENFORCE_QA", "false").lower() in ("true", "1", "yes")

# Director v2: lets a multimodal model choose per-shot layouts from a contact
# sheet, and executes those choices in the renderer. This is the first feature
# that gives the director authority over pixels rather than only over a veto.
#
# Ships OFF by default on purpose. A wrong layout choice is visible in the
# output, whereas a wrong *window* choice is invisible until someone watches the
# clip, so this needs to be switched on deliberately and reviewed before it is
# trusted in CI. With it off, every path behaves exactly as before.
DIRECTOR_V2_ENABLED = os.getenv("DIRECTOR_V2_ENABLED", "true").lower() in ("true", "1", "yes")
DIRECTOR_V2_TILES = max(4, int(os.getenv("DIRECTOR_V2_TILES", "12")))

# Text-region detection tier: auto | ocr | edge.
# "auto" uses OCR if the package is importable, otherwise the edge heuristic.
# "edge" pins the original pre-OCR behaviour for A/B runs; "ocr" asks for OCR
# explicitly and says so loudly on stderr if it cannot be loaded, so a broken
# setup never degrades silently.
#
# There is deliberately no "mser" option. A zero-dependency MSER tier was built
# and measured first, and rejected: it silently misses glyphs (see the module
# docstring in core/text_detection.py for the numbers).
TEXT_DETECTOR = os.getenv("TEXT_DETECTOR", "auto").strip().lower()

# RapidOCR reports confidence per glyph. Below this, a detection is treated as
# noise; 0.5 keeps decorative marks out of the protected regions without
# discarding genuinely soft text over a busy background.
OCR_CONFIDENCE_MIN = float(os.getenv("OCR_CONFIDENCE_MIN", "0.5"))

# Bound on regions kept per frame. The composition planner turns every region
# into a protected rect, and an unbounded list makes the caption collision check
# fire on everything, which is worse than not detecting text at all.
OCR_MAX_REGIONS_PER_FRAME = max(1, int(os.getenv("OCR_MAX_REGIONS_PER_FRAME", "12")))

# Shot boundary detection: auto | transnet | scenedetect.
# "auto" uses TransNetV2 when its ONNX weights are present at TRANSNET_MODEL_PATH
# and onnxruntime (or OpenCV DNN) can load them, otherwise PySceneDetect.
# PySceneDetect's AdaptiveDetector misses hard cuts on low-motion content, which
# is the common case for a talking-head podcast, so TransNetV2 is the better
# detector when available -- but it is not required, and nothing changes until
# the weights are supplied.
SHOT_DETECTOR = os.getenv("SHOT_DETECTOR", "auto").strip().lower()

# Weights are NEVER downloaded: there is no verified canonical URL for a
# TransNetV2 ONNX export, and this project does not invent URLs. Export from the
# soCzech/TransNetV2 reference implementation and drop the file here.
TRANSNET_MODEL_PATH = os.getenv(
    "TRANSNET_MODEL_PATH", str(MODELS_DIR / "transnetv2.onnx")
)

# Minimum per-frame transition probability to accept a cut. TransNetV2 is
# trained on broadcast material and is over-eager on slow pans, so the default
# sits above the model's own 0.5 argmax.
TRANSNET_MIN_CUT_CONFIDENCE = float(os.getenv("TRANSNET_MIN_CUT_CONFIDENCE", "0.6"))

# Frames per inference window. The reference model is trained on 100-frame
# windows; windows overlap by half so a boundary near an edge is seen whole.
TRANSNET_WINDOW = max(16, int(os.getenv("TRANSNET_WINDOW", "100")))

# Video & Format Defaults (Ultra HD 60FPS Broadcast Studio)
IS_CI = os.getenv("GITHUB_ACTIONS", "false").lower() == "true"

if IS_CI:
    # Optimized for GitHub Actions: Crisp studio quality while maintaining runner memory safety
    OUTPUT_WIDTH = 1080
    OUTPUT_HEIGHT = 1920
    FPS = 30
    VIDEO_CRF = 19  # High visual fidelity (sharp upscaling, no compression artifacts)
    AUDIO_BITRATE = "192k"
else:
    OUTPUT_WIDTH = 1080
    OUTPUT_HEIGHT = 1920
    FPS = 60
    VIDEO_CRF = 16  # Ultra high visual fidelity
    AUDIO_BITRATE = "256k"


# Social Media UI Safe Zone Margins (TikTok, Reels, Shorts)
SAFE_ZONE_TOP = 240       # Reserved for header/search
SAFE_ZONE_BOTTOM = 380    # Reserved for creator handle, captions, audio disc
SAFE_ZONE_RIGHT = 120     # Reserved for like, comment, share icons

SUBTITLE_MARGIN_MIN_V = SAFE_ZONE_BOTTOM
SUBTITLE_MARGIN_MAX_V = OUTPUT_HEIGHT - SAFE_ZONE_TOP - 120
SUBTITLE_LAYOUT_MARGINS = {
    "split_screen": 880,
    "blur_stack": 400,
    "single_smooth": 460,
    "multi_shot_dynamic": 460,
    "dynamic_cut": 460,
}


def get_subtitle_margin_v(layout_mode: str, margin_v: Optional[int] = None) -> int:
    base_margin = SUBTITLE_LAYOUT_MARGINS.get(layout_mode, 460)
    candidate = base_margin if margin_v is None else int(margin_v)
    return max(SUBTITLE_MARGIN_MIN_V, min(SUBTITLE_MARGIN_MAX_V, candidate))


# Audio Mastering (Social Broadcast Standard)
TARGET_LUFS = -14.0       # EBU R128 standard for Instagram/TikTok/Shorts
TARGET_TRUE_PEAK = -1.5   # Prevents compression clipping distortion
HIGHPASS_FREQ = 80        # Cut microphone room rumble
VOCAL_PRESENCE_FREQ = 3000 # Enhance voice clarity
VOCAL_AIR_FREQ = 10000    # High-frequency studio microphone sheen

# Subtitle Color & Aesthetic Themes (ASS color codes: &HAABBGGRR)
# Note: ASS hex format is &H[Alpha][Blue][Green][Red]
SUBTITLE_THEMES: Dict[str, Dict[str, Any]] = {
    "hormozi": {
        "name": "Hormozi Viral",
        "font_name": "Montserrat Black",
        "fallback_font": "Arial Black",
        "font_size": 66,
        "primary_color": "&H00FFFFFF",      # Crisp White
        "highlight_color": "&H0000E6FF",    # Electric Yellow (&H00BBGGRR: Blue 00, Green E6, Red FF)
        "outline_color": "&H00000000",      # Deep Black Outline
        "outline_width": 6.5,
        "shadow_color": "&H90000000",       # Deep Soft Drop Shadow
        "shadow_depth": 3.5,
        "max_words_per_line": 2,            # Alex Hormozi 1-to-2 word dynamic pacing
        "uppercase": True
    },
    "neon_green": {
        "name": "Toxic Neon",
        "font_name": "Montserrat Black",
        "fallback_font": "Arial Black",
        "font_size": 66,
        "primary_color": "&H00FFFFFF",      # Crisp White
        "highlight_color": "&H0033FF22",    # Toxic Lime Green
        "outline_color": "&H00000000",
        "outline_width": 6.5,
        "shadow_color": "&HA0000000",
        "shadow_depth": 3.5,
        "max_words_per_line": 3,
        "uppercase": True
    },
    "luxury_gold": {
        "name": "Luxury Mindset",
        "font_name": "Bebas Neue",
        "fallback_font": "Helvetica",
        "font_size": 68,
        "primary_color": "&H00F5F5F5",      # Ivory White
        "highlight_color": "&H0000D7FF",    # Warm Gold
        "outline_color": "&H00101010",      # Charcoal Deep Outline
        "outline_width": 6.0,
        "shadow_color": "&H80000000",
        "shadow_depth": 3.0,
        "max_words_per_line": 3,
        "uppercase": True
    },
    "cyber_cyan": {
        "name": "Cyber Cyan",
        "font_name": "Anton",
        "fallback_font": "Arial Black",
        "font_size": 66,
        "primary_color": "&H00FFFFFF",      # Crisp White
        "highlight_color": "&H00FFFF00",    # Pure Cyan
        "outline_color": "&H00000000",
        "outline_width": 6.5,
        "shadow_color": "&HA0000000",
        "shadow_depth": 3.5,
        "max_words_per_line": 3,
        "uppercase": True
    },
    "tiktok_streamer": {
        "name": "TikTok Streamer Kinetic",
        "font_name": "Montserrat Black",
        "fallback_font": "Arial Black",
        "font_size": 58,
        "primary_color": "&H00FFFFFF",      # Crisp White
        "highlight_color": "&H0000E6FF",    # Electric Yellow Pop (&H00BBGGRR)
        "outline_color": "&H00000000",      # Deep Crisp Outline
        "outline_width": 5.5,
        "shadow_color": "&H90000000",       # Deep Soft Drop Shadow
        "shadow_depth": 3.0,
        "max_words_per_line": 2,            # 1-to-2 word rapid kinetic bursts
        "uppercase": True
    }
}

