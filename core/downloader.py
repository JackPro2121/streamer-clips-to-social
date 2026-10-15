import os
import re
import time
import math
import tempfile
from pathlib import Path
from urllib.parse import quote, urlparse
from typing import Optional, Dict, Any, List, Tuple
import subprocess
import yt_dlp
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from core.config import (
    DOWNLOADS_DIR, APIFY_API_TOKEN, YOUTUBE_COOKIES, YTDLP_PROXY, RAPIDAPI_KEY,
    COBALT_API_URL, COBALT_INSTANCES, MIN_VIDEO_HEIGHT,
    YTDLP_POT_PROVIDER_URL,
    APIFY_FULL_DOWNLOAD_ACTOR_ID, APIFY_SEGMENT_ACTOR_ID,
    APIFY_TRANSCRIPT_ACTOR_ID,
)

# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------
# The old downloader treated every failure identically ("try the next strategy"),
# which meant a permanently-dead video (private/removed/geo-blocked) burned the
# full 7-strategy x full-download budget before giving up. We now distinguish
# permanent failures (abort immediately) from transient ones (rotate client).

def _is_public_https_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        if parsed.scheme.lower() != "https" or not parsed.hostname:
            return False
        import ipaddress
        try:
            address = ipaddress.ip_address(parsed.hostname)
        except ValueError:
            return True
        return not (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
        )
    except (TypeError, ValueError):
        return False


def _is_trusted_apify_url(url: str) -> bool:
    try:
        hostname = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    return (
        hostname == "apify.com"
        or hostname.endswith(".apify.com")
        or hostname == "apify.dev"
        or hostname.endswith(".apify.dev")
    )


class VideoUnavailableError(Exception):
    """Raised when a video is permanently unfetchable (private/removed/geo/age).

    Signals the orchestrator to STOP: no client rotation or paid tier will help.
    """


# Phrases that mean "this video will never download, stop trying".
_PERMANENT_MARKERS = (
    "video unavailable",
    "private video",
    "this video is private",
    "has been removed",
    "account associated with this video has been terminated",
    "video has been removed",
    "who has blocked it",
    "not available in your country",
    "not made this video available",
    "members-only",
    "this live event",
    "premieres in",
    "video is not available",
    "content is not available",
    "drm protected",
)

# Phrases that mean "this specific client got blocked/throttled, rotate".
_TRANSIENT_MARKERS = (
    "sign in to confirm",
    "confirm you're not a bot",
    "http error 429",
    "too many requests",
    "unable to download webpage",
    "read timed out",
    "connection reset",
    "temporarily unavailable",
    "failed to extract",
    "requested format is not available",
    "no video formats found",
    "unable to extract",
    "precondition check failed",
)


def classify_ytdlp_error(exc: Exception) -> str:
    """Return one of: 'permanent' | 'transient' | 'unknown' for a yt-dlp error."""
    msg = str(exc).lower()
    for marker in _PERMANENT_MARKERS:
        if marker in msg:
            return "permanent"
    for marker in _TRANSIENT_MARKERS:
        if marker in msg:
            return "transient"
    return "unknown"


def best_available_height(info: Dict[str, Any]) -> int:
    """Given a yt-dlp info dict (from a metadata-only probe), return the tallest
    downloadable video height on offer. 0 if none / unknown."""
    best = 0
    for f in (info.get("formats") or []):
        if f.get("vcodec") in (None, "none"):
            continue
        h = f.get("height") or 0
        try:
            h = int(h)
        except (TypeError, ValueError):
            h = 0
        if h > best:
            best = h
    # Some extractors expose a top-level height instead of per-format.
    top = info.get("height") or 0
    try:
        top = int(top)
    except (TypeError, ValueError):
        top = 0
    return max(best, top)


def get_video_height(file_path: Path) -> int:
    """Probes video stream height using ffprobe to ensure resolution is >= 720p."""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=height",
            "-of", "csv=p=0",
            str(file_path)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        lines = res.stdout.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line.isdigit():
                return int(line)
        return 0
    except Exception:
        return 0

def get_video_dimensions(file_path: Path) -> Tuple[int, int]:
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=s=x:p=0",
            str(file_path)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        value = res.stdout.strip()
        if "x" in value:
            width, height = value.split("x", 1)
            return int(width), int(height)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return 1920, 1080


def get_video_duration(file_path: Path) -> float:
    """Probes video duration in seconds using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(file_path)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        val = res.stdout.strip()
        return float(val) if val else 0.0
    except Exception:
        return 0.0

def extract_youtube_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from various URL formats."""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:shorts\/)([0-9A-Za-z_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def fetch_youtube_transcript(video_id: str) -> Optional[List[Dict[str, Any]]]:
    """
    Attempt to fetch native YouTube transcript (free & instant).
    Compatible with both youtube-transcript-api 0.x and 1.x.
    Returns list of dicts: [{'text': str, 'start': float, 'duration': float}]
    """
    raw_snippets: Any = None

    # Strategy 1: youtube-transcript-api 1.x instance method
    try:
        api = YouTubeTranscriptApi()
        # 1A. Try listing transcripts first (handles en, en-US, generated captions)
        if hasattr(api, "list"):
            try:
                transcript_list = api.list(video_id)
                t: Any = None
                try:
                    t = transcript_list.find_transcript(['en', 'en-US', 'en-GB'])
                except Exception:
                    t = next(iter(transcript_list), None)
                if t:
                    raw_snippets = t.fetch()
            except Exception as e:
                print(f"[*] api.list() attempt note: {e}")

        # 1B. Direct fetch if list() did not succeed
        if not raw_snippets and hasattr(api, "fetch"):
            try:
                raw_snippets = api.fetch(video_id)
            except Exception as e:
                print(f"[*] api.fetch() attempt note: {e}")
    except Exception as e:
        print(f"[*] YouTubeTranscriptApi initialization note: {e}")

    # Strategy 2: Fallback to legacy static method (0.x)
    if not raw_snippets:
        try:
            if hasattr(YouTubeTranscriptApi, "get_transcript"):
                raw_snippets = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US'])
        except Exception:
            pass

    # Strategy 3: Retry once after a brief pause if rate-limited or challenged
    if not raw_snippets:
        time.sleep(2.0)
        try:
            api = YouTubeTranscriptApi()
            if hasattr(api, "list"):
                t_list = api.list(video_id)
                try:
                    t = t_list.find_transcript(['en', 'en-US', 'en-GB'])
                except Exception:
                    t = next(iter(t_list), None)
                if t:
                    raw_snippets = t.fetch()
        except Exception:
            pass

    if not raw_snippets:
        print(f"[-] Native YouTube transcript unavailable for video {video_id}.")
        return None

    # Normalize snippets into standard dict format
    normalized = []
    for item in raw_snippets:
        text = getattr(item, "text", None) if not isinstance(item, dict) else item.get("text")
        start = getattr(item, "start", 0.0) if not isinstance(item, dict) else item.get("start", 0.0)
        duration = getattr(item, "duration", 0.0) if not isinstance(item, dict) else item.get("duration", 0.0)
        if text:
            normalized.append({
                "text": str(text),
                "start": float(start),
                "duration": float(duration)
            })

    print(f"[+] Retrieved native YouTube transcript ({len(normalized)} snippets).")
    return normalized

def download_via_apify(
    video_url: str,
    output_dir: Path,
    quality: str = "1080",
    api_token: Optional[str] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    output_label: str = "clip",
    moment_start: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """
    Downloads a source video using the configured full-download actor. Targeted
    ranges are delegated to the segment actor because the full-download actor's
    published schema does not include startTime/endTime.
    """
    token = api_token or APIFY_API_TOKEN
    if not token:
        return None

    if start_time is not None and end_time is not None:
        return download_segment_via_apify(
            video_url=video_url,
            output_dir=output_dir,
            start_time=start_time,
            end_time=end_time,
            quality=quality,
            api_token=token,
            output_label=output_label,
            moment_start=moment_start,
        )

    actor_path = quote(APIFY_FULL_DOWNLOAD_ACTOR_ID.replace("/", "~"), safe="~")
    print(f"[*] Dispatching full YouTube download via Apify Actor ({APIFY_FULL_DOWNLOAD_ACTOR_ID})...")
    endpoint = f"https://api.apify.com/v2/acts/{actor_path}/runs"
    
    # Payload for full download vs targeted clip
    payload: Dict[str, Any] = {
        "videos": [{"url": video_url}],
        "preferredQuality": f"{quality}p",
        "storeInKVStore": True,
    }
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    try:
        res = requests.post(endpoint, headers=headers, json=payload, timeout=30)
        if res.status_code not in (200, 201):
            print(f"[-] Apify actor start failed: {res.status_code} - {res.text}")
            return None

        run_id = res.json().get("data", {}).get("id")
        if not run_id:
            print("[-] Could not retrieve Apify run ID.")
            return None

        print(f"[*] Apify run started ({run_id}). Polling for download completion (up to 10 mins)...")
        final_run_data = None
        for attempt in range(150):
            time.sleep(4)
            status_res = requests.get(
                f"https://api.apify.com/v2/actor-runs/{run_id}",
                headers=headers,
                timeout=15
            )
            if status_res.status_code == 200:
                final_run_data = status_res.json().get("data", {})
                status = final_run_data.get("status")
                if attempt % 5 == 0:
                    print(f"[*] Apify download status: {status} (elapsed: {(attempt + 1) * 4}s)")
                if status in ("SUCCEEDED", "FAILED", "TIMED-OUT", "ABORTED"):
                    break

        if not final_run_data or final_run_data.get("status") != "SUCCEEDED":
            print(f"[-] Apify actor finished with non-success state: {final_run_data.get('status') if final_run_data else 'unknown'}")
            return None

        dataset_id = final_run_data.get("defaultDatasetId")

        # Fetch output items
        items_res = requests.get(
            f"https://api.apify.com/v2/datasets/{dataset_id}/items",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30
        )
        items = items_res.json()
        if not items or not isinstance(items, list):
            print("[-] No output items found in Apify dataset.")
            return None

        first_item = items[0]
        direct_url = first_item.get("downloadedFileUrl") or first_item.get("output", {}).get("url")
        if not direct_url:
            print("[-] Apify output missing direct video download URL.")
            return None

        vid_id = first_item.get("id") or extract_youtube_id(video_url) or "video"
        out_file = output_dir / f"{vid_id}.mp4"

        if not _is_public_https_url(direct_url):
            print("[-] Apify returned an unsafe or non-HTTPS download URL.")
            return None
        stream_headers = {}
        if _is_trusted_apify_url(direct_url):
            stream_headers = {"Authorization": f"Bearer {token}"}

        print(f"[*] Streaming high-quality video from Apify storage ({out_file.name})...")
        with requests.get(direct_url, headers=stream_headers, stream=True, timeout=180, allow_redirects=False) as stream_res:
            stream_res.raise_for_status()
            with open(out_file, "wb") as f:
                for chunk in stream_res.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        print(f"[+] Download complete via Apify: {out_file} ({out_file.stat().st_size / (1024*1024):.2f} MB)")
        return {
            "video_path": out_file.resolve(),
            "title": f"YouTube_{vid_id}",
            "duration": float(first_item.get("durationSeconds", 0.0)),
            "is_local": False
        }
    except Exception as e:
        print(f"[-] Apify download encountered an exception: {e}")
        return None


def download_segment_via_apify(
    video_url: str,
    output_dir: Path,
    start_time: float,
    end_time: float,
    quality: str = "1080",
    api_token: Optional[str] = None,
    output_label: str = "clip",
    moment_start: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Download only one selected time range through a segment-capable Actor.

    ``start_time`` may reach back before the moment to provide hook pre-roll
    material. Pass ``moment_start`` so ``segment_start`` still reports the
    moment's offset inside the downloaded file (the file begins at the
    actor's integer grid ``int(start_time)``).
    """
    token = api_token or APIFY_API_TOKEN
    if not token or end_time <= start_time:
        return None

    actor_path = quote(APIFY_SEGMENT_ACTOR_ID.replace("/", "~"), safe="~")
    endpoint = f"https://api.apify.com/v2/acts/{actor_path}/run-sync-get-dataset-items"
    
    # The Apify segment actor strictly requires startTime and endTime to be integers.
    # Flooring start_time and ceiling end_time guarantees the target range is captured.
    int_start = int(start_time)
    int_end = int(math.ceil(end_time))
    if int_end <= int_start:
        int_end = int_start + 1

    payload: Dict[str, Any] = {
        "url": video_url,
        "format": str(quality),
        "startTime": int_start,
        "endTime": int_end,
    }
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    print(f"[*] Requesting Apify source segment {start_time:.2f}s-{end_time:.2f}s (window: {int_start}s-{int_end}s) via {APIFY_SEGMENT_ACTOR_ID}.")

    try:
        started = requests.post(endpoint, headers=headers, json=payload, timeout=60)
        if started.status_code not in (200, 201):
            print(f"[-] Apify segment actor rejected the clip request: {started.status_code} - {started.text[:300]}")
            return None
        items = started.json()
        first_item = items[0] if isinstance(items, list) and items else items
        if not isinstance(first_item, dict):
            print("[-] Apify segment actor returned no job data.")
            return None
        raw_output = first_item.get("output")
        job: Dict[str, Any] = raw_output if isinstance(raw_output, dict) else first_item
        direct_url_value = job.get("downloadUrl")
        poll_url_value = job.get("pollUrl")
        direct_url: Optional[str] = (
            direct_url_value if isinstance(direct_url_value, str) else None
        )
        poll_url: Optional[str] = (
            poll_url_value if isinstance(poll_url_value, str) else None
        )

        if not direct_url and not poll_url:
            print("[-] Apify segment actor response did not include downloadUrl or pollUrl.")
            return None

        if not direct_url and poll_url:
            if not _is_public_https_url(poll_url):
                print(f"[-] Apify returned an unsafe polling URL for host '{urlparse(poll_url).hostname or 'invalid'}'.")
                return None
            poll_headers = {"Authorization": f"Bearer {token}"} if _is_trusted_apify_url(poll_url) else {}
            completed = None
            for attempt in range(75):
                time.sleep(3)
                try:
                    poll = requests.get(poll_url, headers=poll_headers, timeout=20)
                    if poll.status_code != 200:
                        continue
                    completed = poll.json()
                    status = str(completed.get("status", "")).upper()
                    if completed.get("downloadUrl") or status in ("COMPLETED", "DONE", "SUCCESS", "READY", "FINISHED"):
                        direct_url = completed.get("downloadUrl")
                        break
                    if status in ("FAILED", "ERROR", "CANCELLED"):
                        print(f"[-] Apify segment download ended with {status}.")
                        return None
                    if (attempt + 1) % 5 == 0:
                        print(f"  [*] Still waiting for Apify segment ({attempt + 1}/75, status='{status}')...")
                except requests.RequestException:
                    continue

            if not direct_url:
                print("[-] Apify segment download timed out before completion.")
                return None
        if not direct_url:
            print("[-] Apify segment response did not include a valid download URL.")
            return None
        vid_id = extract_youtube_id(video_url) or "video"
        # The label keeps the English-audio fallback from overwriting the video
        # segment: both used to stream to clip_<id>_<range>.mp4, so a fallback
        # fetch replaced the source video with an audio-only file.
        out_file = output_dir / f"{output_label}_{vid_id}_{int_start}s_{int_end}s.mp4"
        print(f"[*] Streaming requested Apify segment to {out_file.name}...")

        if not _is_public_https_url(direct_url):
            print("[-] Apify returned an unsafe or non-HTTPS segment URL.")
            return None
        stream_headers = {}
        if _is_trusted_apify_url(direct_url):
            stream_headers = {"Authorization": f"Bearer {token}"}

        with requests.get(direct_url, headers=stream_headers, stream=True, timeout=180, allow_redirects=False) as stream_res:
            stream_res.raise_for_status()
            with open(out_file, "wb") as f:
                for chunk in stream_res.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        height = get_video_height(out_file)
        from core.preflight import preflight_source

        report = preflight_source(out_file)
        if not report.ok:
            print(
                f"[-] Pre-flight rejected the segment [{report.stage}]: {report.reason}"
            )
            return None
        if report.video_duration and report.audio_duration:
            mismatch_ms = (report.audio_duration - report.video_duration) * 1000.0
            if abs(mismatch_ms) > 80.0:
                print(
                    f"[!] Pre-flight: stream length mismatch {mismatch_ms:+.0f}ms "
                    "(the renderer will harmonise it)."
                )
        anchor = float(moment_start) if moment_start is not None else float(start_time)
        segment_start = max(0.0, anchor - float(int_start))
        return {
            "video_path": out_file.resolve(),
            "title": job.get("title", f"YouTube_{vid_id}"),
            "duration": get_video_duration(out_file) or (end_time - start_time),
            "height": height,
            "is_low_res": 0 < height < MIN_VIDEO_HEIGHT,
            "segment_start": segment_start,
            "segment_duration": end_time - start_time,
            "is_local": False,
        }
    except requests.RequestException as e:
        print(f"[-] Apify segment request failed: {e}")
        return None

# Deduplicated format selectors (were copy-pasted across 7 strategies).
# Strictly prefer avc1 (H.264) to avoid AV1 decoding loops on GitHub Actions runners.
# Enforce English/original audio track priority so multi-dub videos do not download foreign audio.
# Prefer the ORIGINAL audio track over YouTube's auto-dubbed one. A dubbed
# track often carries language=en too (format_note "dubbed-auto"), so the old
# language-first order fetched the dub: yt-dlp issues #11753/#11834 and the
# r/youtubedl report all converge on `format_note*=original` as the reliable
# marker. Old fallbacks are kept at the end so an unknown-operator failure
# cannot regress the fetch.
_AUDIO_EN_PREF = (
    'bestaudio[format_note*=original][language^=en]/'
    'bestaudio[format_note*=original]/'
    'bestaudio[language=en-US][format_note!*=dubbed]/'
    'bestaudio[language^=en][format_note!*=dubbed]/'
    'bestaudio[format_note!*=dubbed]/'
    'bestaudio[language=en-US]/'
    'bestaudio[language=en]/'
    'bestaudio[language^=en]/'
    'bestaudio[format_id*=original]/'
    'bestaudio[format_id=140]/'
    'bestaudio[format_id=251]/'
    'bestaudio[format_id=139]/'
    'bestaudio'
)
_HD_FORMAT = (
    f'bestvideo[vcodec^=avc1][height>=720][height<=1080]+{_AUDIO_EN_PREF}/'
    f'bestvideo[vcodec^=avc1][height<=1080]+{_AUDIO_EN_PREF}/'
    f'bestvideo[height<=1080]+{_AUDIO_EN_PREF}/'
    'best[ext=mp4][height<=1080]/best'
)
_ANY_FORMAT = (
    f'bestvideo[vcodec^=avc1][height<=1080]+{_AUDIO_EN_PREF}/'
    f'bestvideo[height<=1080]+{_AUDIO_EN_PREF}/'
    'best[ext=mp4][height<=1080]/best[vcodec^=avc1]/best'
)
_FAILSAFE_FORMAT = f'bestvideo[vcodec^=avc1]+{_AUDIO_EN_PREF}/bestvideo+{_AUDIO_EN_PREF}/best'
# Local bgutil POT provider. This is only configured after its health check
# succeeds in CI; otherwise yt-dlp must use its normal client fallback.
_POT_ARGS = (
    {'youtubepot-bgutilhttp': {'base_url': [YTDLP_POT_PROVIDER_URL]}}
    if YTDLP_POT_PROVIDER_URL else {}
)


def _ytdlp_client_variants(cookie_path: Optional[Path] = None) -> List[Tuple[str, Dict[str, Any]]]:
    """Generates ordered yt-dlp client extraction options prioritizing bot-bypassing clients."""
    def _client(clients: List[str]) -> Dict[str, Any]:
        return {'extractor_args': {'youtube': {'player_client': clients}, **_POT_ARGS}}

    variants: List[Tuple[str, Dict[str, Any]]] = [
        ("android_vr", _client(['android_vr'])),
        ("tv_embedded", _client(['tv_embedded', 'tv'])),
        ("web-pot", _client(['web'])),
        ("default+mweb", _client(['default', 'mweb'])),
        ("ios", _client(['ios'])),
        ("universal", _client(['android_vr', 'web', 'tv_embedded'])),
    ]
    if cookie_path:
        cookie_extra = _client(['web', 'tv'])
        cookie_extra['cookiefile'] = str(cookie_path)
        variants.append(("cookies", cookie_extra))
    # Fail-safe: android client without POT (no HD guarantee, but usually ungated).
    variants.append(("android-failsafe", {'extractor_args': {'youtube': {'player_client': ['android']}}}))
    return variants


def _write_cookiefile(cookies: str) -> Optional[Path]:
    """Write cookies to a private (0600) temp file OUTSIDE the artifact dir.

    The old code wrote cookies into the downloads/ folder, which is uploaded as a
    CI artifact -- leaking the session. mkstemp + chmod keeps them off the artifact
    path and unreadable to other users; the caller unlinks it in a finally block.
    """
    try:
        fd, tmp = tempfile.mkstemp(prefix="ytc_", suffix=".txt")
        os.write(fd, (cookies.strip() + "\n").encode("utf-8"))
        os.close(fd)
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass  # Windows: best-effort.
        return Path(tmp)
    except Exception as e:
        print(f"[-] Cookie setup warning: {e}")
        return None


def _do_download(
    url: str,
    base_opts: Dict[str, Any],
    extra: Dict[str, Any],
    fmt: str,
    probed_info: Optional[Dict[str, Any]],
    video_id: Optional[str],
    transcript: Optional[Any],
    label: str,
    low_res: bool = False,
) -> Optional[Dict[str, Any]]:
    """Perform exactly one real download with a chosen client + format."""
    opts = {
        **base_opts,
        'format': fmt,
        'postprocessors': [{'key': 'FFmpegVideoConvertor', 'preferedformat': 'mp4'}],
        **extra,
    }
    try:
        print(f"[*] Downloading via '{label}'...")
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            path = ydl.prepare_filename(info)
            if not os.path.exists(path):
                alt = str(Path(path).with_suffix('.mp4'))
                path = alt if os.path.exists(alt) else path
            if not os.path.exists(path):
                print(f"[-] '{label}' reported success but produced no file.")
                return None
        # Probe the real file. h==0 means the probe failed (missing ffprobe /
        # odd container) -- NOT that the video is low-res; fall back to the
        # resolution the metadata probe advertised instead of mislabelling HD.
        h = get_video_height(Path(path))
        real_h = h if h > 0 else (best_available_height(probed_info) if probed_info else 0)
        print(f"[+] Downloaded {real_h or '?'}p via '{label}': {Path(path).name}")
        return {
            'video_path': Path(path).resolve(),
            'title': info.get('title', f"YouTube_{video_id or 'podcast'}"),
            'duration': float(info.get('duration', 0.0) or 0.0),
            'transcript': transcript,
            'video_id': video_id or info.get('id'),
            'height': real_h,
            'is_low_res': low_res or (0 < real_h < 720),
            'is_local': False,
        }
    except VideoUnavailableError:
        raise
    except Exception as e:
        if classify_ytdlp_error(e) == "permanent":
            raise VideoUnavailableError(str(e))
        print(f"[-] Download via '{label}' failed: {e}")
        return None


def download_via_ytdlp(
    url_or_path: str,
    target_dir: Path,
    video_id: Optional[str] = None,
    transcript: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """Probe-first yt-dlp downloader ($0, no Apify compute).

    The previous implementation ran up to SEVEN full downloads of the same
    2-hour video, probing resolution only AFTER each download -- catastrophic on
    a timeboxed/metered CI runner (disk blow-up + 90-min timeout).

    This version inverts the loop:
      1. For each client, run a METADATA-ONLY probe (``download=False`` -- seconds,
         zero video bytes) and classify failures. A *permanent* failure
         (private/removed/geo) raises ``VideoUnavailableError`` and aborts; a
         *transient* block rotates to the next client.
      2. The first client advertising a >=720p stream is downloaded ONCE.
      3. If nobody offers HD, the best sub-HD client is downloaded ONCE.
    Net: one download in the happy path, one in the worst case -- never seven.
    """
    base_opts = {
            'js_runtimes': {'node': {}},
            'outtmpl': str(target_dir / "%(id)s_%(title).50s.%(ext)s"),
            'merge_output_format': 'mp4',
            'socket_timeout': 30,
            'retries': 3,
            'quiet': True,
            'no_warnings': True,
        }
    if YTDLP_PROXY and YTDLP_PROXY.strip():
        base_opts['proxy'] = YTDLP_PROXY.strip()
        print("[*] Routing yt-dlp through configured proxy.")

    cookie_path = (
        _write_cookiefile(YOUTUBE_COOKIES)
        if (YOUTUBE_COOKIES and YOUTUBE_COOKIES.strip())
        else None
    )

    clients = _ytdlp_client_variants(cookie_path)

    try:
        best_sub_hd: Optional[Tuple[str, Dict[str, Any], int]] = None
        for label, extra in clients:
            probe_opts = {**base_opts, 'skip_download': True, **extra}
            try:
                print(f"[*] Probing client '{label}' (metadata only)...")
                with yt_dlp.YoutubeDL(probe_opts) as ydl:
                    info = ydl.extract_info(url_or_path, download=False)
            except VideoUnavailableError:
                raise
            except Exception as e:
                kind = classify_ytdlp_error(e)
                if kind == "permanent":
                    raise VideoUnavailableError(str(e))
                print(f"[-] Client '{label}' probe failed ({kind}): {e}")
                continue

            height = best_available_height(info)
            print(f"[*] Client '{label}' offers up to {height}p.")
            is_failsafe = label == "android-failsafe"
            if height >= 720:
                fmt = _FAILSAFE_FORMAT if is_failsafe else _HD_FORMAT
                result = _do_download(url_or_path, base_opts, extra, fmt, info,
                                      video_id, transcript, label)
                if result:
                    return result
                continue  # HD probe but download died -> next client.
            elif height > 0 and (best_sub_hd is None or height > best_sub_hd[2]):
                best_sub_hd = (label, extra, height)

        # No HD anywhere: take the tallest sub-HD client with a single download.
        if best_sub_hd:
            label, extra, height = best_sub_hd
            print(f"[!] No HD stream available; downloading best {height}p via '{label}'.")
            fmt = _FAILSAFE_FORMAT if label == "android-failsafe" else _ANY_FORMAT
            return _do_download(url_or_path, base_opts, extra, fmt, None,
                                video_id, transcript, label, low_res=True)

        return None
    finally:
        if cookie_path and cookie_path.exists():
            try:
                cookie_path.unlink()
            except Exception:
                pass

def download_via_cobalt(
    video_url: str,
    output_dir: Path,
    quality: str = "1080",
    instances: Optional[List[str]] = None
) -> Optional[Dict[str, Any]]:
    """
    Tier 2 ($0, No Cookies): Downloads YouTube video via Cobalt Engine API.
    Cobalt is an open-source media ingestion service designed specifically to bypass
    YouTube's BotGuard, SABR, and PoToken constraints without requiring user cookies.
    Supports multi-instance fallback, tunnel/redirect streams, and chunked streaming.
    """
    candidate_instances = instances or COBALT_INSTANCES or [COBALT_API_URL]
    vid_id = extract_youtube_id(video_url) or "video"
    out_file = output_dir / f"YouTube_{vid_id}.mp4"

    ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": ua
    }
    payload = {
        "url": video_url,
        "videoQuality": str(quality),
        "youtubeVideoCodec": "h264",
        "downloadMode": "auto"
    }

    for inst_idx, base_url in enumerate(candidate_instances):
        inst_url = base_url.rstrip("/")
        target_endpoint = inst_url if inst_url.endswith("/api") else f"{inst_url}/"
        print(f"[*] Attempting 1080p download via Cobalt instance {inst_idx + 1}/{len(candidate_instances)} ({inst_url})...")
        try:
            res = requests.post(target_endpoint, headers=headers, json=payload, timeout=25)
            if res.status_code not in (200, 201):
                print(f"[-] Cobalt instance {inst_url} returned status {res.status_code}: {res.text[:120]}")
                continue

            data = res.json()
            status = data.get("status")
            stream_url = data.get("url")

            # Direct stream, tunnel, or redirect URL
            if status in ("tunnel", "redirect", "stream") and stream_url:
                if not _is_public_https_url(stream_url):
                    print(f"[-] Cobalt returned an unsafe or non-HTTPS stream URL: {inst_url}")
                    continue
                print(f"[*] Cobalt returned '{status}' stream. Downloading to disk ({out_file.name})...")
                with requests.get(stream_url, headers={"User-Agent": ua}, stream=True, timeout=180, allow_redirects=False) as stream_res:
                    stream_res.raise_for_status()
                    with open(out_file, "wb") as f:
                        for chunk in stream_res.iter_content(chunk_size=2 * 1024 * 1024):
                            if chunk:
                                f.write(chunk)

                if out_file.exists() and out_file.stat().st_size > 1024 * 1024:
                    h = get_video_height(out_file)
                    dur = get_video_duration(out_file)
                    print(f"[+] Cobalt download complete: {out_file.name} ({h}p, {out_file.stat().st_size / (1024*1024):.2f} MB)")
                    return {
                        "video_path": out_file.resolve(),
                        "title": f"YouTube_{vid_id}",
                        "duration": dur,
                        "height": h,
                        "is_low_res": (0 < h < MIN_VIDEO_HEIGHT),
                        "is_local": False
                    }
                else:
                    print("[-] Cobalt stream produced empty or truncated file.")
            elif status == "picker":
                picker_items = data.get("picker", [])
                best_url = None
                for item in picker_items:
                    if item.get("type") in ("video", None) and item.get("url"):
                        best_url = item["url"]
                        break
                if best_url:
                    if not _is_public_https_url(best_url):
                        print(f"[-] Cobalt returned an unsafe or non-HTTPS picker URL: {inst_url}")
                        continue
                    print("[*] Cobalt returned picker items. Downloading stream to disk...")
                    with requests.get(best_url, headers={"User-Agent": ua}, stream=True, timeout=180, allow_redirects=False) as stream_res:
                        stream_res.raise_for_status()
                        with open(out_file, "wb") as f:
                            for chunk in stream_res.iter_content(chunk_size=2 * 1024 * 1024):
                                if chunk:
                                    f.write(chunk)
                    if out_file.exists() and out_file.stat().st_size > 1024 * 1024:
                        h = get_video_height(out_file)
                        dur = get_video_duration(out_file)
                        return {
                            "video_path": out_file.resolve(),
                            "title": f"YouTube_{vid_id}",
                            "duration": dur,
                            "height": h,
                            "is_low_res": (0 < h < MIN_VIDEO_HEIGHT),
                            "is_local": False
                        }
            else:
                err_info = data.get("error", {})
                err_code = err_info.get("code") if isinstance(err_info, dict) else str(err_info)
                print(f"[-] Cobalt instance {inst_url} error ({status}): {err_code}")
        except Exception as e:
            print(f"[-] Cobalt instance {inst_url} exception: {e}")

    return None

def download_via_rapidapi(
    video_url: str,
    output_dir: Path,
    api_key: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Option 2: Downloads YouTube video via RapidAPI YouTube Downloader endpoints.
    Provides 50-100 free requests per month without bot captcha.
    Muxes crystal-clear 1080p MP4 video with high-bitrate M4A audio via FFmpeg stream copy.
    Supports key rotation (comma-separated keys) and robust chunked streaming fallback.
    """
    import subprocess
    raw_keys = api_key or RAPIDAPI_KEY or ""
    keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
    if not keys:
        return None

    vid_id = extract_youtube_id(video_url)
    if not vid_id:
        return None

    ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"

    for key_idx, key in enumerate(keys):
        print(f"[*] Attempting high-speed 1080p download via RapidAPI Downloader (key {key_idx + 1}/{len(keys)})...")
        headers = {
            "x-rapidapi-key": key,
            "x-rapidapi-host": "youtube-media-downloader.p.rapidapi.com"
        }
        url = f"https://youtube-media-downloader.p.rapidapi.com/v2/video/details?videoId={vid_id}"
        try:
            res = requests.get(url, headers=headers, timeout=25)
            if res.status_code == 429:
                print(f"[-] RapidAPI key {key_idx + 1} quota exceeded (429). Trying next key...")
                continue
            if res.status_code != 200:
                print(f"[-] RapidAPI returned status {res.status_code}: {res.text[:150]}")
                continue

            data = res.json()
            videos = data.get("videos", {}).get("items", [])
            audios = data.get("audios", {}).get("items", [])

            # Select best video stream (strictly 1080p or 720p)
            best_video = None
            for q in ["1080p", "720p"]:
                for v in videos:
                    if v.get("quality") == q and v.get("extension") == "mp4" and v.get("url"):
                        best_video = v
                        break
                if best_video:
                    break

            # Select best audio stream
            best_audio = None
            for a in audios:
                if a.get("extension") in ("m4a", "mp4") and a.get("url"):
                    best_audio = a
                    break

            out_file = output_dir / f"YouTube_{vid_id}.mp4"

            if best_video and best_audio:
                print(f"[*] RapidAPI stream found: Video {best_video.get('quality')} + Audio {best_audio.get('extension')}. Muxing via FFmpeg...")
                cmd = [
                    "ffmpeg", "-y",
                    "-user_agent", ua,
                    "-reconnect", "1",
                    "-reconnect_streamed", "1",
                    "-reconnect_delay_max", "5",
                    "-i", best_video["url"],
                    "-user_agent", ua,
                    "-reconnect", "1",
                    "-reconnect_streamed", "1",
                    "-reconnect_delay_max", "5",
                    "-i", best_audio["url"],
                    "-c:v", "copy",
                    "-c:a", "copy",
                    "-movflags", "+faststart",
                    str(out_file)
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                if proc.returncode == 0 and out_file.exists() and out_file.stat().st_size > 1024 * 1024:
                    print(f"[+] Download and 1080p mux complete via RapidAPI: {out_file} ({out_file.stat().st_size / (1024*1024):.2f} MB)")
                    return {
                        "video_path": out_file.resolve(),
                        "title": data.get("title", f"YouTube_{vid_id}"),
                        "duration": float(data.get("lengthSeconds", 0.0)),
                        "is_local": False
                    }
                else:
                    err_msg = proc.stderr[-300:].strip() if proc.stderr else "Unknown error"
                    print(f"[-] Fast FFmpeg mux failed ({proc.returncode}): {err_msg}. Engaging robust chunked streaming fallback...")

                    # Fallback: Download tracks to disk via requests with browser User-Agent and stream chunks
                    temp_v = output_dir / f"tmp_{vid_id}_v.mp4"
                    temp_a = output_dir / f"tmp_{vid_id}_a.m4a"
                    dl_headers = {"User-Agent": ua}
                    try:
                        print(f"[*] Downloading video track via chunked stream ({best_video.get('quality')})...")
                        with requests.get(best_video["url"], headers=dl_headers, stream=True, timeout=180) as r_v:
                            r_v.raise_for_status()
                            with open(temp_v, "wb") as f_v:
                                for chunk in r_v.iter_content(chunk_size=2 * 1024 * 1024):
                                    if chunk:
                                        f_v.write(chunk)

                        print(f"[*] Downloading audio track via chunked stream ({best_audio.get('extension')})...")
                        with requests.get(best_audio["url"], headers=dl_headers, stream=True, timeout=180) as r_a:
                            r_a.raise_for_status()
                            with open(temp_a, "wb") as f_a:
                                for chunk in r_a.iter_content(chunk_size=2 * 1024 * 1024):
                                    if chunk:
                                        f_a.write(chunk)

                        if temp_v.exists() and temp_a.exists() and temp_v.stat().st_size > 512 * 1024:
                            print("[*] Tracks saved locally. Combining with local FFmpeg...")
                            cmd_local = [
                                "ffmpeg", "-y",
                                "-i", str(temp_v),
                                "-i", str(temp_a),
                                "-c:v", "copy",
                                "-c:a", "copy",
                                "-movflags", "+faststart",
                                str(out_file)
                            ]
                            p_loc = subprocess.run(cmd_local, capture_output=True, text=True, timeout=120)
                            if p_loc.returncode == 0 and out_file.exists() and out_file.stat().st_size > 1024 * 1024:
                                print(f"[+] Download and 1080p mux complete via chunked streaming: {out_file} ({out_file.stat().st_size / (1024*1024):.2f} MB)")
                                return {
                                    "video_path": out_file.resolve(),
                                    "title": data.get("title", f"YouTube_{vid_id}"),
                                    "duration": float(data.get("lengthSeconds", 0.0)),
                                    "is_local": False
                                }
                    finally:
                        for tmp_f in [temp_v, temp_a]:
                            if tmp_f.exists():
                                try:
                                    tmp_f.unlink()
                                except Exception:
                                    pass

            # Fallback: If single combined stream exists
            if best_video and best_video.get("hasAudio"):
                print(f"[*] Streaming combined format ({best_video.get('quality')})...")
                with requests.get(best_video["url"], headers={"User-Agent": ua}, stream=True, timeout=180) as stream_res:
                    stream_res.raise_for_status()
                    with open(out_file, "wb") as f:
                        for chunk in stream_res.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                if out_file.exists() and out_file.stat().st_size > 0:
                    print(f"[+] Download complete via RapidAPI: {out_file} ({out_file.stat().st_size / (1024*1024):.2f} MB)")
                    return {
                        "video_path": out_file.resolve(),
                        "title": data.get("title", f"YouTube_{vid_id}"),
                        "duration": float(data.get("lengthSeconds", 0.0)),
                        "is_local": False
                    }
        except Exception as e:
            print(f"[-] RapidAPI download error on key {key_idx + 1}: {e}")

    return None

def download_video(url_or_path: str, output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Multi-tier resilient downloader with strict 720p - 1080p Quality Gate:
    1. Local file check.
    2. Tier 1: Zero-cost yt-dlp (Cloudflare WARP + android_vr / tv_embedded / bgutil POT).
    3. Tier 2: Cobalt Engine Ingestion ($0, no cookies, bypasses BotGuard & PoTokens).
    4. Tier 3: RapidAPI 1080p Muxer fallback (Key rotation).
    5. Tier 4: Apify Actor residential proxy fallback.
    """
    target_dir = output_dir or DOWNLOADS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    # Check if local file
    local_path = Path(url_or_path)
    if local_path.exists() and local_path.is_file():
        h = get_video_height(local_path)
        dur = get_video_duration(local_path)
        return {
            'video_path': local_path.resolve(),
            'title': local_path.stem,
            'duration': dur,
            'transcript': None,
            'video_id': extract_youtube_id(local_path.stem),
            'height': h,
            'is_low_res': (0 < h < MIN_VIDEO_HEIGHT),
            'is_local': True
        }

    # Fetch native transcript if YouTube URL
    video_id = extract_youtube_id(url_or_path)
    transcript = None
    if video_id:
        transcript = fetch_youtube_transcript(video_id)

    fallback_candidate: Optional[Dict[str, Any]] = None

    # Tier 1: Direct yt-dlp ($0 cost, uses bgutil PO Token + android_vr + Proxy if set).
    try:
        ytdlp_result = download_via_ytdlp(url_or_path, target_dir, video_id=video_id, transcript=transcript)
        if ytdlp_result:
            h = ytdlp_result.get('height', 0)
            if h >= MIN_VIDEO_HEIGHT:
                print(f"[+] Tier 1 (yt-dlp) succeeded with verified HD quality ({h}p).")
                return ytdlp_result
            else:
                print(f"[!] Tier 1 result is {h}p (below {MIN_VIDEO_HEIGHT}p). Retaining as fallback and trying Tier 2 (Cobalt)...")
                fallback_candidate = ytdlp_result
    except VideoUnavailableError as e:
        print(f"[-] yt-dlp reports video permanently unavailable: {e}")
    except Exception as e:
        print(f"[-] Tier 1 (yt-dlp) failed: {e}")

    # Tier 2: Cobalt Engine API ($0, Zero Cookies, Bot-Bypass)
    print("[*] Engaging Tier 2: Cobalt Engine API ($0, bot-bypass)...")
    try:
        cobalt_result = download_via_cobalt(url_or_path, target_dir, quality="1080")
        if cobalt_result:
            cobalt_result['transcript'] = transcript
            cobalt_result['video_id'] = video_id
            h = cobalt_result.get('height', 0)
            if h >= MIN_VIDEO_HEIGHT:
                print(f"[+] Tier 2 (Cobalt) succeeded with verified HD quality ({h}p).")
                return cobalt_result
            elif not fallback_candidate or h > fallback_candidate.get('height', 0):
                fallback_candidate = cobalt_result
    except Exception as e:
        print(f"[-] Tier 2 (Cobalt) failed: {e}")

    # Tier 3: RapidAPI Downloader fallback (Option 2)
    if RAPIDAPI_KEY:
        print("[!] Engaging Tier 3: RapidAPI 1080p Downloader fallback...")
        try:
            rapidapi_result = download_via_rapidapi(url_or_path, target_dir)
            if rapidapi_result:
                rapidapi_result['transcript'] = transcript
                rapidapi_result['video_id'] = video_id
                h = rapidapi_result.get('height', 0)
                if h >= MIN_VIDEO_HEIGHT:
                    print(f"[+] Tier 3 (RapidAPI) succeeded with verified HD quality ({h}p).")
                    return rapidapi_result
                elif not fallback_candidate or h > fallback_candidate.get('height', 0):
                    fallback_candidate = rapidapi_result
        except Exception as e:
            print(f"[-] Tier 3 (RapidAPI) failed: {e}")

    # Tier 4: Apify Actor proxy fallback
    if APIFY_API_TOKEN and ("youtube.com" in url_or_path or "youtu.be" in url_or_path):
        print("[!] Engaging Tier 4: Apify Actor proxy downloader...")
        try:
            apify_result = download_via_apify(url_or_path, target_dir, quality="1080")
            if apify_result:
                apify_result['transcript'] = transcript
                apify_result['video_id'] = video_id
                return apify_result
        except Exception as e:
            print(f"[-] Tier 4 (Apify) failed: {e}")

    # If HD wasn't achieved but sub-HD candidate exists:
    if fallback_candidate:
        print(f"[!] Warning: All HD tiers exhausted. Using fallback candidate ({fallback_candidate.get('height', '?')}p).")
        return fallback_candidate

    raise RuntimeError(f"Failed to download video from {url_or_path} across all tiers (yt-dlp, Cobalt, RapidAPI, Apify).")


def fetch_transcript_via_apify(video_url: str, api_token: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
    """Fetch timestamped captions through Apify without downloading video bytes."""
    token = api_token or APIFY_API_TOKEN
    if not token:
        return None
    actor_path = quote(APIFY_TRANSCRIPT_ACTOR_ID.replace("/", "~"), safe="~")
    endpoint = f"https://api.apify.com/v2/acts/{actor_path}/run-sync-get-dataset-items"
    # Verified live 2026-10-05 against pintostudio/youtube-transcript-scraper
    # (5.79M runs): input {"videoUrl": url} -> [{"data": [{start, dur, text}]}].
    # The previous default actor (om_kh/video-transcript-api) went dead, which is
    # why every run fell through to the slow Whisper probe.
    payload: Dict[str, Any] = {"videoUrl": video_url}
    try:
        response = requests.post(
            endpoint,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
            timeout=90,
        )
        response.raise_for_status()
        raw = response.json()
    except requests.RequestException as e:
        print(f"[-] Apify transcript request failed: {e}")
        return None

    records = raw if isinstance(raw, list) else [raw]
    for record in records:
        if not isinstance(record, dict):
            continue
        # Current actor shape: {"data": [{"start": "0.12", "dur": "3.0", "text": ...}]}
        data_segments = record.get("data")
        if isinstance(data_segments, list):
            normalized = []
            for segment in data_segments:
                if not isinstance(segment, dict) or not str(segment.get("text", "")).strip():
                    continue
                try:
                    start_s = float(segment.get("start", 0) or 0)
                    duration_s = float(segment.get("dur", segment.get("duration", 0)) or 0)
                except (TypeError, ValueError):
                    continue
                if duration_s > 0:
                    normalized.append(
                        {"text": str(segment["text"]), "start": start_s, "duration": duration_s}
                    )
            if normalized:
                print(f"[+] Retrieved {len(normalized)} timestamped transcript segments via Apify.")
                return normalized
        transcript_record = record.get("transcripts", record)
        if isinstance(transcript_record, list):
            transcript_record = next((item for item in transcript_record if isinstance(item, dict)), None)
        if not isinstance(transcript_record, dict):
            continue
        raw_segments = transcript_record.get("segments") or transcript_record.get("transcript")
        if not isinstance(raw_segments, list):
            continue
        normalized = []
        for segment in raw_segments:
            if not isinstance(segment, dict) or not str(segment.get("text", "")).strip():
                continue
            start: Any = segment.get("start", segment.get("startMs", 0))
            duration: Any = segment.get("duration")
            if duration is None and segment.get("endMs") is not None:
                duration = float(segment["endMs"]) - float(start)
            try:
                start = float(start) / 1000 if "startMs" in segment else float(start)
                duration = float(duration or 0) / 1000 if "durationMs" in segment else float(duration or 0)
            except (TypeError, ValueError):
                continue
            if duration > 0:
                normalized.append({"text": str(segment["text"]), "start": start, "duration": duration})
        if normalized:
            print(f"[+] Retrieved {len(normalized)} timestamped transcript segments via Apify.")
            return normalized
    print("[!] Apify transcript actor returned no usable timed captions.")
    return None


def fetch_transcript_only(url: str, allow_apify_fallback: bool = False) -> Optional[Dict]:
    """
    Step 0 of the Smart Pipeline: Fetches ONLY the transcript (text) from a
    YouTube video — ZERO video bytes downloaded. Returns transcript + video_id.

    Fallback chain (zero video bytes at every tier):
      Tier 1: YouTubeTranscriptApi  — free, instant, no API key needed
      Tier 2: Apify transcript actor — cheap, residential proxy (if token set)
      Tier 3: Chocodata API          — $0.0009/call, bypasses all bot detection
    """
    video_id = extract_youtube_id(url)
    if not video_id:
        return None

    # Tier 1: Native YouTube captions (free)
    transcript = fetch_youtube_transcript(video_id)

    # Tier 2: Apify transcript actor (cheap, residential proxy)
    if not transcript and allow_apify_fallback:
        print("[*] Native captions unavailable. Trying transcript-only Apify fallback...")
        transcript = fetch_transcript_via_apify(url)

    # Tier 3: Chocodata REST API ($0.0009/call — runs even if Apify is down/exhausted)
    if not transcript:
        try:
            from core.config import CHOCODATA_API_KEY
            from core.transcriber import fetch_transcript_chocodata
            if CHOCODATA_API_KEY:
                print("[*] Trying Chocodata API as transcript fallback (tier 3)...")
                choco_segments = fetch_transcript_chocodata(video_id)
                if choco_segments:
                    # Convert TranscriptSegment objects back to raw dict format
                    transcript = [
                        {"text": s.text, "start": s.start, "duration": s.end - s.start}
                        for s in choco_segments
                    ]
                    print(f"[+] Chocodata returned {len(transcript)} transcript segments.")
        except Exception as e:
            print(f"[*] Chocodata transcript fallback note: {e}")

    if not transcript:
        print("[!] No timed transcript found across all 3 tiers. Targeted clip download unavailable.")
        return None
    return {"video_id": video_id, "transcript": transcript}


def download_clip_segment(
    video_url: str,
    start_time: float,
    end_time: float,
    clip_index: int,
    output_dir: Path,
    moment_start: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """
    Targeted Range Downloader: Downloads ONLY the seconds [start_time, end_time]
    of a YouTube video using yt-dlp's --download-sections flag.

    Instead of downloading a full 2GB podcast, this grabs ONLY the 15-25 MB
    clip window identified by Gemini — saving bandwidth, time, and avoiding
    YouTube's rate-limiter that triggers on long continuous streams.

    The downloaded clip starts at t=0s, so render_viral_clip must be called
    with start_time=0.0 and end_time=(end_time - start_time).

    Returns a dict compatible with download_video() results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_duration = end_time - start_time
    out_file = output_dir / f"clip_{clip_index}_{int(start_time)}s_{int(end_time)}s.mp4"

    # yt-dlp --download-sections takes "*START-END" notation (seconds)
    section_spec = f"*{start_time:.2f}-{end_time:.2f}"
    print(f"[*] Targeted clip download: segment {section_spec} (~{clip_duration:.0f}s) -> {out_file.name}")

    # -------------------------------------------------------------------------
    # STRATEGY 1: Apify Segment Actor (Primary — bypasses datacenter bot detection)
    # -------------------------------------------------------------------------
    if APIFY_API_TOKEN:
        print(f"[*] Attempting primary segment download via Apify actor ({APIFY_SEGMENT_ACTOR_ID})...")
        try:
            apify_result = download_via_apify(
                video_url=video_url,
                output_dir=output_dir,
                start_time=start_time,
                end_time=end_time,
                moment_start=moment_start,
            )
            if apify_result:
                out_path = Path(apify_result["video_path"])
                h = get_video_height(out_path)
                dur = get_video_duration(out_path)
                print(f"  [+] Segment downloaded via Apify ({h}p, {out_path.stat().st_size / (1024*1024):.1f} MB)")
                if h < MIN_VIDEO_HEIGHT or dur <= 0:
                    print(f"  [-] Apify result failed the {MIN_VIDEO_HEIGHT}p/duration quality gate. Trying yt-dlp...")
                else:
                    segment_start = float(apify_result.get("segment_start", 0.0))
                    return {
                        'video_path': out_path.resolve(),
                        'title': f"clip_{clip_index}",
                        'duration': dur or clip_duration,
                        'height': h,
                        'is_low_res': (0 < h < MIN_VIDEO_HEIGHT),
                        'segment_start': segment_start,
                        'segment_duration': clip_duration,
                        'is_local': False,
                    }
        except Exception as e:
            print(f"  [-] Apify segment download failed: {e}. Falling back to yt-dlp...")

    # -------------------------------------------------------------------------
    # STRATEGY 2: Local yt-dlp Multi-Client Waterfall (Fallback)
    # -------------------------------------------------------------------------
    print(f"[*] Attempting segment download via yt-dlp waterfall: {section_spec} (~{clip_duration:.0f}s)...")
    base_opts = {
        'js_runtimes': {'node': {}},
        'outtmpl': str(output_dir / f"clip_{clip_index}_{int(start_time)}s_{int(end_time)}s.%(ext)s"),
        'merge_output_format': 'mp4',
        'download_ranges': yt_dlp.utils.download_range_func(None, [(start_time, end_time)]),
        'force_keyframes_at_cuts': True,
        'quiet': True,
        'no_warnings': True,
    }
    if YTDLP_PROXY and YTDLP_PROXY.strip():
        base_opts['proxy'] = YTDLP_PROXY.strip()

    # Try each client variant in priority order (android_vr first = no BotGuard)
    cookie_value = os.environ.get("YOUTUBE_COOKIES", "") or ""
    cookie_path = (
        _write_cookiefile(cookie_value)
        if cookie_value.strip()
        else None
    )
    clients = _ytdlp_client_variants(cookie_path=cookie_path)
    for label, extra in clients:
        opts = {
            **base_opts,
            'format': _HD_FORMAT,
            'postprocessors': [{'key': 'FFmpegVideoConvertor', 'preferedformat': 'mp4'}],
            **extra,
        }
        try:
            print(f"  [*] Segment download via client '{label}'...")
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.extract_info(video_url, download=True)

            # yt-dlp may produce the file with .mp4 directly or via post-processor
            produced = out_file
            if not produced.exists():
                # Search for any file matching our clip prefix
                candidates = sorted(output_dir.glob(f"clip_{clip_index}_{int(start_time)}s_{int(end_time)}s.*"))
                if candidates:
                    produced = candidates[0]

            if produced.exists() and produced.stat().st_size > 500_000:
                h = get_video_height(produced)
                if not h or h <= 0:
                    print(f"  [-] Client '{label}' produced audio-only or non-video stream. Trying next client...")
                    continue
                if h < MIN_VIDEO_HEIGHT:
                    print(f"  [-] Client '{label}' produced {h}p, below the {MIN_VIDEO_HEIGHT}p gate. Trying next client...")
                    continue
                dur = get_video_duration(produced)
                print(f"  [+] Segment downloaded ({h}p, {produced.stat().st_size / (1024*1024):.1f} MB) via '{label}'")
                # yt-dlp cuts at the requested section start, so the file
                # begins at start_time; the moment sits (start_time - moment)
                # into it. Without a moment anchor the file starts at the clip.
                ytdlp_segment_start = (
                    max(0.0, float(moment_start) - float(start_time))
                    if moment_start is not None
                    else 0.0
                )
                return {
                    'video_path': produced.resolve(),
                    'title': f"clip_{clip_index}",
                    'duration': dur or clip_duration,
                    'height': h,
                    'is_low_res': (0 < h < MIN_VIDEO_HEIGHT),
                    'segment_start': ytdlp_segment_start,
                    'segment_duration': clip_duration,
                    'is_local': False,
                }
            else:
                print(f"  [-] Client '{label}' produced no usable file. Trying next...")
        except Exception as e:
            kind = classify_ytdlp_error(e)
            if kind == "permanent":
                print(f"  [-] Permanent error on segment download: {e}")
                return None
            print(f"  [-] Client '{label}' segment download failed ({kind}): {e}")

    print(f"[-] All segment download strategies failed for {section_spec}.")
    return None


def fetch_english_audio_segment(
    video_url: str,
    output_dir: Path,
    start_time: float,
    end_time: float,
    clip_index: int = 1,
) -> Optional[Path]:
    """
    Downloads specifically the English audio track ([en-US] or original) for a clip time-range.
    Uses yt-dlp with _AUDIO_EN_PREF and download_ranges.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_out_tmpl = str(output_dir / f"audio_en_{clip_index}_{int(start_time)}s_{int(end_time)}s.%(ext)s")
    base_opts: Dict[str, Any] = {
        'js_runtimes': {'node': {}},
        'outtmpl': audio_out_tmpl,
        'download_ranges': yt_dlp.utils.download_range_func(None, [(start_time, end_time)]),
        'force_keyframes_at_cuts': True,
        'quiet': True,
        'no_warnings': True,
    }
    if YTDLP_PROXY and YTDLP_PROXY.strip():
        base_opts['proxy'] = YTDLP_PROXY.strip()

    cookie_value = os.environ.get("YOUTUBE_COOKIES", "") or ""
    cookie_path = (
        _write_cookiefile(cookie_value)
        if cookie_value.strip()
        else None
    )
    clients = _ytdlp_client_variants(cookie_path=cookie_path)
    audio_format = _AUDIO_EN_PREF

    for label, extra in clients:
        opts = {
            **base_opts,
            'format': audio_format,
            **extra,
        }
        try:
            print(f"  [*] Attempting English audio fetch via client '{label}'...")
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.extract_info(video_url, download=True)

            candidates = sorted(output_dir.glob(f"audio_en_{clip_index}_{int(start_time)}s_{int(end_time)}s.*"))
            for candidate in candidates:
                if candidate.exists() and candidate.stat().st_size > 10_000:
                    print(f"  [+] English audio track downloaded ({candidate.stat().st_size / 1024:.1f} KB) via '{label}'")
                    return candidate.resolve()
        except Exception as e:
            print(f"  [-] Client '{label}' audio fetch failed: {e}")

    # Fallback to Apify with audio format if APIFY_API_TOKEN is present
    if APIFY_API_TOKEN:
        print("[*] Trying Apify audio extraction for English track fallback...")
        try:
            apify_audio = download_via_apify(
                video_url=video_url,
                output_dir=output_dir,
                start_time=start_time,
                end_time=end_time,
                quality="audio",
                output_label="audio_en",
            )
            if apify_audio and Path(apify_audio["video_path"]).exists():
                return Path(apify_audio["video_path"]).resolve()
        except Exception as e:
            print(f"[-] Apify audio extraction failed: {e}")

    return None


def remux_video_with_audio(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
) -> bool:
    """
    Replaces the audio track of video_path with audio_path using ffmpeg (c:v copy, c:a aac).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        str(output_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return proc.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0
    except Exception as e:
        print(f"[-] Remux failed: {e}")
        return False


def try_fallback_english_audio(
    video_url: str,
    clip_path: Path,
    output_dir: Path,
    start_time: float,
    end_time: float,
    clip_index: int = 1,
    transcript_text: Optional[str] = None,
) -> Optional[Path]:
    """
    Fallback mechanism when downloaded clip has non-English audio (e.g. Apify pulled foreign dub).
    Fetches the English audio track ([en-US]) and remuxes it with the video stream.
    """
    from core.transcriber import verify_audio_language

    audio_file = fetch_english_audio_segment(
        video_url=video_url,
        output_dir=output_dir,
        start_time=start_time,
        end_time=end_time,
        clip_index=clip_index,
    )
    if not audio_file or not audio_file.exists():
        return None

    remuxed_file = output_dir / f"clip_{clip_index}_en_remuxed.mp4"
    if not remux_video_with_audio(clip_path, audio_file, remuxed_file):
        return None

    try:
        verify_audio_language(remuxed_file, transcript_text=transcript_text)
        return remuxed_file
    except Exception as e:
        print(f"[-] Remuxed audio still failed language verification: {e}")
        return None
