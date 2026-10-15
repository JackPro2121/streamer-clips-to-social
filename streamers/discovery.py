"""Streamer Discovery Engine (V2).

Discovers fresh high-engagement YouTube VODs & streams from top US creators:
- Kai Cenat
- IShowSpeed
- Jynxzi
- CaseOh
- xQc
- Adin Ross
- FlightReacts

Maintains an isolated history store at data/streamer_history.json so V1 podcast
history is never touched or affected.

Defects fixed in this revision
--------------------------------
D19  — Round-robin cursor ensures all 6+ streamers are queried fairly; the old
        code returned after filling max_candidates from the *first* streamer only.
D10  — History is now written atomically (tmp → replace) regardless of crash path;
        corrupted / failed videos are tracked with per-video attempt counters so
        they are blacklisted after STREAMER_HISTORY_MAX_ATTEMPTS failures.
D25  — Atomic os.replace() write; IOError is never silently swallowed.
D4.3 — Freshness filter: VODs older than max_age_hours are skipped at fetch time.
RSS  — Bare `except: pass` in fetch_rss_feed_videos replaced with full logging.
DEDUP — in-run `seen_ids` set prevents duplicate video_ids within a single run.
"""

import json
import os
import re
import subprocess
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.config import DATA_DIR
from streamers.spec import (
    STREAMER_FRESHNESS_HOURS,
    STREAMER_HISTORY_CAP,
    STREAMER_HISTORY_MAX_ATTEMPTS,
    STREAMER_MAX_DISCOVERY_CANDIDATES,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STREAMER_HISTORY_FILE = DATA_DIR / "streamer_history.json"
STREAMER_CURSOR_FILE = DATA_DIR / "streamer_cursor.json"

# Ordered list of roster keys — order determines round-robin rotation.
STREAMER_ROSTER_ORDER: List[str] = [
    "kai_cenat",
    "ishowspeed",
    "jynxzi",
    "caseoh",
    "xqc",
    "adin_ross",
]

STREAMER_CHANNELS: Dict[str, Dict[str, Any]] = {
    "kai_cenat": {
        "name": "Kai Cenat",
        "category": "just_chatting",
        "channel_urls": [
            "https://www.youtube.com/@KaiCenatLive/streams",
            "https://www.youtube.com/@KaiCenatLive/videos",
            "https://www.youtube.com/@KaiCenat/videos",
        ],
        "rss_feeds": [
            "https://www.youtube.com/feeds/videos.xml?channel_id=UCm03u_jV1-G7u7S9ZgE9sOw",
        ],
        "keywords": ["Kai Cenat", "AMP", "Mafia", "stream", "room", "rage"],
    },
    "ishowspeed": {
        "name": "IShowSpeed",
        "category": "irl",
        "channel_urls": [
            "https://www.youtube.com/@LiveSpeedy/videos",
            "https://www.youtube.com/@IShowSpeed/streams",
            "https://www.youtube.com/@IShowSpeed/videos",
        ],
        "rss_feeds": [
            "https://www.youtube.com/feeds/videos.xml?channel_id=UCWsD388ETZql143SFkk9VvA",
        ],
        "keywords": ["Speed", "IShowSpeed", "stream", "bark", "rage", "irl"],
    },
    "jynxzi": {
        "name": "Jynxzi",
        "category": "gaming",
        "channel_urls": [
            "https://www.youtube.com/@Jynxzi/streams",
            "https://www.youtube.com/@Jynxzi/videos",
            "https://www.youtube.com/@JynxziClips/videos",
        ],
        "rss_feeds": [
            "https://www.youtube.com/feeds/videos.xml?channel_id=UC0L_q8F102r3gY4U32bF39A",
        ],
        "keywords": ["Jynxzi", "R6", "Rainbow Six", "reaction", "clip", "gaming"],
    },
    "caseoh": {
        "name": "CaseOh",
        "category": "gaming",
        "channel_urls": [
            "https://www.youtube.com/@CaseOhGames/videos",
            "https://www.youtube.com/@CaseOhGames/streams",
            "https://www.youtube.com/@CaseOh_/videos",
        ],
        "rss_feeds": [
            "https://www.youtube.com/feeds/videos.xml?channel_id=UCp8U9a9_S2lQnQ25h9oT_qA",
        ],
        "keywords": ["CaseOh", "horror", "gameplay", "funny", "rage", "reaction"],
    },
    "xqc": {
        "name": "xQc",
        "category": "just_chatting",
        "channel_urls": [
            "https://www.youtube.com/@xQcOW/videos",
            "https://www.youtube.com/@xQcOW/streams",
            "https://www.youtube.com/@xQcClips/videos",
        ],
        "rss_feeds": [
            "https://www.youtube.com/feeds/videos.xml?channel_id=UCmDTrq0LNgPodDOFZiSbSwA",
        ],
        "keywords": ["xQc", "reacts", "stream", "drama", "debate", "gaming"],
    },
    "adin_ross": {
        "name": "Adin Ross",
        "category": "just_chatting",
        "channel_urls": [
            "https://www.youtube.com/@AdinLive/videos",
            "https://www.youtube.com/@AdinLive/streams",
        ],
        "rss_feeds": [
            "https://www.youtube.com/feeds/videos.xml?channel_id=UCgs_U0_x2BwE_J_7vH3oFzA",
        ],
        "keywords": ["Adin Ross", "stream", "interview", "funny", "reaction"],
    },
}


# ---------------------------------------------------------------------------
# Cursor helpers (D19 — round-robin across all streamers)
# ---------------------------------------------------------------------------


def _load_cursor() -> int:
    """Returns the last-queried roster index so the next run starts one step ahead.

    Persisting the cursor across process restarts is what prevents D19: without
    it every run begins at index-0 (kai_cenat) and exits as soon as max_candidates
    is filled — meaning the remaining 5 streamers are never queried.
    """
    if not STREAMER_CURSOR_FILE.exists():
        return -1
    try:
        data = json.loads(STREAMER_CURSOR_FILE.read_text(encoding="utf-8"))
        return int(data.get("last_index", -1))
    except (json.JSONDecodeError, ValueError, OSError) as exc:
        print(f"[!] Could not read streamer cursor (will reset to 0): {exc}")
        return -1


def _save_cursor(last_index: int) -> None:
    """Atomically persists the cursor so the next run picks up where this one ended.

    Uses the same tmp-then-replace pattern as history writes so a mid-write
    crash cannot produce a truncated cursor file.
    """
    STREAMER_CURSOR_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = STREAMER_CURSOR_FILE.with_suffix(".cursor.tmp")
    try:
        tmp_path.write_text(
            json.dumps({"last_index": last_index}, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp_path, STREAMER_CURSOR_FILE)
    except OSError as exc:
        print(f"[!] WARNING — could not save streamer cursor: {exc}")


# ---------------------------------------------------------------------------
# History helpers (D10, D25 — atomic writes, retry budget)
# ---------------------------------------------------------------------------


class StreamerHistory(dict):
    """Dictionary representing streamer history that also supports `video_id in history`.

    Allows backward-compatible `video_id in history` membership tests while providing
    full access to dict keys: `processed_video_ids`, `attempts`, and `errors`.
    """

    def __contains__(self, key: Any) -> bool:
        if super().__contains__(key):
            return True
        return key in self.get("processed_video_ids", [])


def load_streamer_history() -> Dict[str, Any]:
    """Returns the full history dict, not just IDs.

    Returning the complete dict (instead of a bare list) is required so callers
    can inspect per-video attempt counts and blacklist videos that have exceeded
    STREAMER_HISTORY_MAX_ATTEMPTS (D10).  Supports legacy list-format files by
    up-migrating them transparently.

    Schema::

        {
            "processed_video_ids": ["abc123", ...],
            "attempts": {"xyz789": 2},
            "errors":   {"xyz789": "timeout on attempt 2"}
        }
    """
    if not STREAMER_HISTORY_FILE.exists():
        return StreamerHistory({"processed_video_ids": [], "attempts": {}, "errors": {}})
    try:
        raw = json.loads(STREAMER_HISTORY_FILE.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            # Up-migrate legacy format
            return StreamerHistory({
                "processed_video_ids": [str(v) for v in raw],
                "attempts": {},
                "errors": {},
            })
        if isinstance(raw, dict):
            return StreamerHistory({
                "processed_video_ids": [str(v) for v in raw.get("processed_video_ids", [])],
                "attempts": {str(k): int(v) for k, v in raw.get("attempts", {}).items()},
                "errors": {str(k): str(v) for k, v in raw.get("errors", {}).items()},
            })
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        print(f"[!] WARNING — streamer history unreadable, starting fresh: {exc}")
    return StreamerHistory({"processed_video_ids": [], "attempts": {}, "errors": {}})


def _write_history_atomic(history: Dict[str, Any]) -> None:
    """Writes the history dict to disk atomically via a tmp-then-replace sequence.

    Atomic replacement (D25) prevents a partial write from corrupting the history
    file; if the process is killed between write and replace the old file is
    untouched.  IOError is logged rather than silently swallowed so operators
    can detect disk-full / permission issues immediately.
    """
    STREAMER_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Rolling cap: keep most recent STREAMER_HISTORY_CAP IDs
    history["processed_video_ids"] = history["processed_video_ids"][-STREAMER_HISTORY_CAP:]
    tmp_path = STREAMER_HISTORY_FILE.with_suffix(".tmp")
    try:
        tmp_path.write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp_path, STREAMER_HISTORY_FILE)
    except OSError as exc:
        # Never silently discard this — the caller must know the write failed.
        print(f"[!] WARNING — failed to persist streamer history: {exc}")


def record_streamer_history(video_id: str, title: str = "", creator: str = "") -> None:
    """Records a successfully processed video ID into streamer_history.json.

    Called on the success path only; failures are tracked separately via
    record_failed_attempt so that the attempts counter is not reset on success.
    """
    history = load_streamer_history()
    if video_id not in history["processed_video_ids"]:
        history["processed_video_ids"].append(video_id)
    # Clear error bookkeeping once a video processes successfully
    history["attempts"].pop(video_id, None)
    history["errors"].pop(video_id, None)
    _write_history_atomic(history)


def record_failed_attempt(video_id: str, error_msg: str) -> None:
    """Increments the per-video failure counter and logs the error.

    Once attempts[video_id] reaches STREAMER_HISTORY_MAX_ATTEMPTS the video is
    treated as permanently processed (blacklisted) by discover_streamer_candidates.
    This prevents D10 — a corrupted or unavailable video being retried forever.
    """
    history = load_streamer_history()
    current = history["attempts"].get(video_id, 0) + 1
    history["attempts"][video_id] = current
    history["errors"][video_id] = error_msg
    print(
        f"[!] Attempt {current}/{STREAMER_HISTORY_MAX_ATTEMPTS} recorded for "
        f"video {video_id}: {error_msg}"
    )
    if current >= STREAMER_HISTORY_MAX_ATTEMPTS:
        # Promote to permanently blacklisted
        if video_id not in history["processed_video_ids"]:
            history["processed_video_ids"].append(video_id)
        print(
            f"[-] Video {video_id} permanently blacklisted after "
            f"{STREAMER_HISTORY_MAX_ATTEMPTS} failed attempts."
        )
    _write_history_atomic(history)


# ---------------------------------------------------------------------------
# Freshness helpers (D4.3)
# ---------------------------------------------------------------------------

_UPLOAD_DATE_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})$")


def _parse_upload_date(date_str: str) -> Optional[datetime]:
    """Parses a yt-dlp upload_date string (YYYYMMDD) into a timezone-aware datetime.

    Returns None if parsing fails so callers can choose to include rather than
    incorrectly exclude videos whose date is missing.
    """
    m = _UPLOAD_DATE_RE.match(date_str.strip())
    if not m:
        return None
    try:
        return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), tzinfo=timezone.utc)
    except ValueError:
        return None


def _parse_atom_published(text: str) -> Optional[datetime]:
    """Parses an Atom <published> element (ISO-8601 / RFC-3339) into a UTC datetime.

    Returns None on parse failure so the video is included rather than
    incorrectly dropped.
    """
    # Strip trailing 'Z' and handle +00:00 — Python 3.10 fromisoformat is stricter
    cleaned = text.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def _is_fresh(publish_dt: Optional[datetime], max_age_hours: int) -> bool:
    """Returns True when publish_dt is within max_age_hours of now.

    Treats a missing timestamp as fresh (include-by-default) to avoid
    incorrectly dropping videos from feeds that omit the date field.
    """
    if publish_dt is None:
        return True
    age_hours = (datetime.now(tz=timezone.utc) - publish_dt).total_seconds() / 3600.0
    return age_hours <= max_age_hours


# ---------------------------------------------------------------------------
# URL / ID utilities
# ---------------------------------------------------------------------------


def extract_video_id_from_url(url: str) -> Optional[str]:
    """Extracts YouTube 11-char video ID from any standard URL form.

    Separated from fetch helpers so it can be reused by Firecrawl, yt-dlp,
    and RSS paths without duplicating the regex.
    """
    patterns = [
        r"(?:v=|\/v\/|youtu\.be\/|\/embed\/|\/live\/|\/shorts\/)([a-zA-Z0-9_-]{11})",
        r"^([a-zA-Z0-9_-]{11})$",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------


def fetch_channel_videos_ytdlp(
    channel_url: str,
    limit: int = 5,
    max_age_hours: int = STREAMER_FRESHNESS_HOURS,
) -> List[Dict[str, Any]]:
    """Fetches candidate videos from a YouTube channel via yt-dlp flat-playlist.

    Applies a freshness filter (D4.3): when yt-dlp supplies an upload_date field
    only videos within max_age_hours of now are returned.  Upcoming premieres and
    live-but-unaired streams are excluded, matching the original intent.

    Returns a list of dicts with keys: video_id, url, title, upload_date (may be "").
    """
    candidates: List[Dict[str, Any]] = []
    try:
        cmd = [
            "yt-dlp",
            "--flat-playlist",
            "--no-update",
            "--playlist-end", str(limit),
            "--dump-json",
            channel_url,
        ]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
        if p.returncode == 0 and p.stdout:
            for line in p.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    live_status = data.get("live_status", "")
                    if live_status == "is_upcoming":
                        continue
                    if data.get("duration") is None and live_status != "was_live":
                        continue
                    vid_id = data.get("id")
                    if not vid_id:
                        continue
                    title = data.get("title", "")
                    upload_date_raw: str = data.get("upload_date", "") or ""
                    publish_dt = _parse_upload_date(upload_date_raw) if upload_date_raw else None
                    if not _is_fresh(publish_dt, max_age_hours):
                        print(
                            f"[~] Skipping stale yt-dlp video {vid_id} "
                            f"(uploaded {upload_date_raw}, limit={max_age_hours}h)"
                        )
                        continue
                    candidates.append({
                        "video_id": vid_id,
                        "url": f"https://www.youtube.com/watch?v={vid_id}",
                        "title": title,
                        "upload_date": upload_date_raw,
                    })
                except (json.JSONDecodeError, KeyError) as exc:
                    print(f"[!] yt-dlp JSON parse error for line from {channel_url}: {exc}")
    except subprocess.TimeoutExpired:
        print(f"[-] yt-dlp timed out for channel {channel_url}")
    except Exception as exc:
        print(f"[-] yt-dlp channel discovery note for {channel_url}: {exc}")
    return candidates


def fetch_rss_feed_videos(
    feed_url: str,
    limit: int = 5,
    max_age_hours: int = STREAMER_FRESHNESS_HOURS,
) -> List[Dict[str, Any]]:
    """Fetches candidate videos from a YouTube RSS feed without downloading any video.

    Applies the freshness filter (D4.3) via the Atom <published> element.
    The previously bare `except Exception: pass` is replaced with explicit logging
    so operators know when a feed URL is broken or rate-limited.
    """
    candidates: List[Dict[str, Any]] = []
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        req = urllib.request.Request(feed_url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            content = resp.read()
        root = ET.fromstring(content)
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "yt": "http://www.youtube.com/xml/schemas/2015",
        }
        for entry in root.findall("atom:entry", ns)[:limit]:
            vid_elem = entry.find("yt:videoId", ns)
            title_elem = entry.find("atom:title", ns)
            published_elem = entry.find("atom:published", ns)
            if vid_elem is None or not vid_elem.text:
                continue
            vid_id = vid_elem.text.strip()
            title = title_elem.text.strip() if title_elem is not None and title_elem.text else ""
            published_str = (
                published_elem.text.strip()
                if published_elem is not None and published_elem.text
                else ""
            )
            publish_dt = _parse_atom_published(published_str) if published_str else None
            if not _is_fresh(publish_dt, max_age_hours):
                print(
                    f"[~] Skipping stale RSS video {vid_id} from {feed_url} "
                    f"(published {published_str}, limit={max_age_hours}h)"
                )
                continue
            candidates.append({
                "video_id": vid_id,
                "url": f"https://www.youtube.com/watch?v={vid_id}",
                "title": title,
                "upload_date": published_str,
            })
    except urllib.error.URLError as exc:
        print(f"[-] RSS feed {feed_url} failed: {type(exc).__name__}: {exc}")
    except ET.ParseError as exc:
        print(f"[-] RSS feed {feed_url} failed: {type(exc).__name__}: {exc}")
    except Exception as exc:
        # Catch-all: always log; never silently discard (was bare `except: pass` pre-fix)
        print(f"[-] RSS feed {feed_url} failed: {type(exc).__name__}: {exc}")
    return candidates


def search_stream_via_firecrawl(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Uses Firecrawl API to search for fresh streamer stream VODs on YouTube.

    Kept intentionally separate from the channel/RSS fetchers so that API-key
    availability can gate this tier without affecting the others.
    """
    from core.config import FIRECRAWL_API_KEY  # noqa: PLC0415 — intentional lazy import

    if not FIRECRAWL_API_KEY:
        return []

    headers = {
        "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "StreamerClipper/2.0",
    }
    payload = {
        "query": f"site:youtube.com {query} full stream VOD",
        "limit": limit,
    }
    candidates: List[Dict[str, Any]] = []
    try:
        req = urllib.request.Request(
            "https://api.firecrawl.dev/v1/search",
            headers=headers,
            data=json.dumps(payload).encode("utf-8"),
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        results = data.get("data", [])
        for r in results:
            url = r.get("url", "")
            title = r.get("title", "")
            vid_id = extract_video_id_from_url(url)
            if vid_id and ("watch?v=" in url or "/live/" in url):
                candidates.append({
                    "video_id": vid_id,
                    "url": f"https://www.youtube.com/watch?v={vid_id}",
                    "title": title,
                    "upload_date": "",
                })
    except Exception as exc:
        print(f"[-] Firecrawl search note: {exc}")
    return candidates


# ---------------------------------------------------------------------------
# Main discovery entry point
# ---------------------------------------------------------------------------


def discover_streamer_candidates(
    creator: Optional[str] = None,
    category: Optional[str] = None,
    max_candidates: int = STREAMER_MAX_DISCOVERY_CANDIDATES,
    max_age_hours: int = STREAMER_FRESHNESS_HOURS,
) -> List[Dict[str, Any]]:
    """Discovers fresh, unprocessed streamer VODs via yt-dlp, Firecrawl, and RSS.

    D19 fix — Round-robin across ALL streamers:
        In auto mode (creator=None) the roster is iterated starting one position
        past the cursor stored in STREAMER_CURSOR_FILE.  Candidates are collected
        *across* the full roster until max_candidates is reached, not per-streamer.
        The cursor is saved atomically before returning so the next invocation
        begins at the next streamer.

    D4.3 fix — Freshness filter:
        max_age_hours is forwarded to every fetch helper.  Videos older than this
        window are logged and skipped at fetch time so they never enter the
        candidate list.

    In-run deduplication:
        A `seen_ids` set prevents the same video_id from appearing twice within a
        single discovery run (e.g. if a video appears in both yt-dlp and RSS tiers).

    Args:
        creator:        If supplied, only this creator key is queried (cursor skipped).
        category:       Optional category filter applied when creator is None.
        max_candidates: Hard ceiling on returned candidates.
        max_age_hours:  VODs older than this many hours are excluded.

    Returns:
        List of candidate dicts, each containing:
        creator, category, video_id, url, title, upload_date.
    """
    from core.config import FIRECRAWL_API_KEY  # noqa: PLC0415 — intentional lazy import

    history = load_streamer_history()
    processed_ids: set[str] = set(history["processed_video_ids"])
    # Also blacklist videos that have hit the attempt ceiling
    for vid_id, attempts in history["attempts"].items():
        if attempts >= STREAMER_HISTORY_MAX_ATTEMPTS:
            processed_ids.add(vid_id)

    # Build ordered list of streamers to query
    if creator and creator.lower() in STREAMER_CHANNELS:
        # Single-creator mode: skip cursor logic entirely
        selected_keys = [creator.lower()]
        use_cursor = False
    else:
        # Auto mode: apply category filter then apply round-robin ordering
        filtered_keys = [
            k for k in STREAMER_ROSTER_ORDER
            if k in STREAMER_CHANNELS
            and (category is None or STREAMER_CHANNELS[k].get("category") == category.lower())
        ]
        if not filtered_keys:
            print(f"[-] No streamers matched category={category!r}; returning empty list.")
            return []
        last_index = _load_cursor()
        roster_len = len(filtered_keys)
        # Start from the position *after* last_index, wrapping around
        start_pos = (last_index + 1) % roster_len
        selected_keys = filtered_keys[start_pos:] + filtered_keys[:start_pos]
        use_cursor = True

    discovered: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()  # in-run dedup (DEDUP fix)
    last_queried_index: int = _load_cursor()

    def _maybe_add(s_info: Dict[str, Any], v: Dict[str, Any]) -> bool:
        """Adds a candidate if it passes history and dedup checks.

        Returns True when the candidate was accepted, False otherwise.
        Centralising this logic prevents the same if-block being scattered
        across three tiers.
        """
        vid_id = v["video_id"]
        if vid_id in processed_ids or vid_id in seen_ids:
            return False
        seen_ids.add(vid_id)
        discovered.append({
            "creator": s_info["name"],
            "category": s_info.get("category", "streamer"),
            "video_id": vid_id,
            "url": v["url"],
            "title": v["title"],
            "upload_date": v.get("upload_date", ""),
        })
        return True

    for roster_pos, key in enumerate(selected_keys):
        if len(discovered) >= max_candidates:
            break

        s = STREAMER_CHANNELS[key]

        # Track which index in the *original* filtered_keys list this key corresponds
        # to, so the cursor reflects the actual roster position not the rotated position.
        if use_cursor:
            # Resolve the original roster position of this key for cursor persistence
            filtered_keys_ref = [
                k for k in STREAMER_ROSTER_ORDER
                if k in STREAMER_CHANNELS
                and (category is None or STREAMER_CHANNELS[k].get("category") == category.lower())
            ]
            if key in filtered_keys_ref:
                last_queried_index = filtered_keys_ref.index(key)

        # Tier 1: Real-Time yt-dlp Channel Discovery
        for channel_url in s.get("channel_urls", []):
            if len(discovered) >= max_candidates:
                break
            vids = fetch_channel_videos_ytdlp(channel_url, limit=4, max_age_hours=max_age_hours)
            for v in vids:
                if len(discovered) >= max_candidates:
                    break
                _maybe_add(s, v)

        # Tier 2: Firecrawl Intelligent Search
        if FIRECRAWL_API_KEY and len(discovered) < max_candidates:
            print(f"[*] Firecrawl: Searching fresh stream VODs for {s['name']}...")
            fc_videos = search_stream_via_firecrawl(s["name"], limit=4)
            for v in fc_videos:
                if len(discovered) >= max_candidates:
                    break
                _maybe_add(s, v)

        # Tier 3: RSS Feed parsing fallback
        if len(discovered) < max_candidates:
            for feed_url in s.get("rss_feeds", []):
                if len(discovered) >= max_candidates:
                    break
                rss_vids = fetch_rss_feed_videos(feed_url, limit=6, max_age_hours=max_age_hours)
                for v in rss_vids:
                    if len(discovered) >= max_candidates:
                        break
                    _maybe_add(s, v)

    # Persist cursor so next run starts from the streamer *after* the last one queried
    if use_cursor:
        _save_cursor(last_queried_index)

    return discovered
