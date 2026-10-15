import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import requests
from core.config import CHOCODATA_API_KEY

@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float
    is_estimated: bool = False

@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    words: List[WordTimestamp]


_WHISPER_MODELS: Dict[tuple[str, str, str], Any] = {}


def is_english_language_code(language: Optional[str]) -> bool:
    normalized = (language or "").strip().lower().replace("_", "-")
    return normalized == "en" or normalized.startswith("en-")


_ENGLISH_STOPWORDS = frozenset({
    "a", "about", "after", "all", "also", "am", "an", "and", "any", "are", "as", "at",
    "be", "because", "been", "but", "by", "can", "do", "does", "for", "from", "get",
    "go", "good", "has", "have", "he", "her", "here", "him", "his", "how", "i", "if",
    "in", "is", "it", "its", "just", "know", "like", "look", "me", "more", "most",
    "my", "new", "no", "not", "now", "of", "on", "one", "only", "or", "other", "our",
    "out", "people", "so", "some", "than", "that", "the", "their", "them", "then",
    "there", "these", "they", "think", "this", "those", "to", "up", "us", "was", "we",
    "well", "were", "what", "when", "which", "who", "why", "will", "with", "would",
    "you", "your",
})


def looks_like_english_text(
    text: str,
    min_words: int = 8,
    min_stopword_ratio: float = 0.08,
) -> bool:
    words = re.findall(r"[A-Za-z']+", text or "")
    if len(words) < min_words:
        return False
    lowered = [word.lower() for word in words]
    stopword_hits = sum(1 for word in lowered if word in _ENGLISH_STOPWORDS)
    if stopword_hits / len(lowered) < min_stopword_ratio:
        return False
    letters = re.findall(r"[^\W\d_]", text or "", flags=re.UNICODE)
    if not letters:
        return False
    latin_letters = sum(1 for letter in letters if letter.isascii() or "\u00c0" <= letter <= "\u024f")
    return latin_letters / len(letters) >= 0.9


def verify_audio_language(
    video_or_audio_path: Path,
    model_size: Optional[str] = None,
    min_probability: float = 0.55,
    transcript_text: Optional[str] = None,
) -> str:
    if model_size is None:
        model_size = "base"
    try:
        from faster_whisper import WhisperModel
    except ImportError as error:
        raise RuntimeError("faster-whisper is required to verify English audio language") from error

    model_key = (model_size, "cpu", "language")
    model = _WHISPER_MODELS.get(model_key)
    if model is None:
        print(f"[*] Initializing language detector ({model_size}, cpu)...")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        _WHISPER_MODELS[model_key] = model

    _, info = model.transcribe(
        str(video_or_audio_path),
        beam_size=1,
        language=None,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    detected = str(getattr(info, "language", "") or "")
    probability = float(getattr(info, "language_probability", 0.0) or 0.0)
    if not is_english_language_code(detected) or probability < min_probability:
        # If the detector is confident (>= 0.70) that the audio is non-English, do not bypass:
        # this catches foreign dubs (e.g. Bengali, Spanish) where metadata transcript was English.
        if transcript_text and looks_like_english_text(transcript_text) and probability < 0.70:
            print(
                f"[!] Language detector suggested '{detected or 'unknown'}' "
                f"(marginal confidence={probability:.2f}) but the clip transcript is clearly English. "
                "Treating the detector result as ambiguous."
            )
            return detected
        raise RuntimeError(
            f"Audio language is not English (detected={detected or 'unknown'}, "
            f"confidence={probability:.2f})"
        )
    print(f"[+] English audio verified ({detected}, confidence={probability:.2f}).")
    return detected

def fetch_transcript_chocodata(video_id: str, api_key: Optional[str] = None) -> Optional[List[TranscriptSegment]]:
    """
    Fetches official timestamped YouTube transcript via Chocodata REST API in 0.4s.
    Bypasses all bot-detection, datacenter blocks, and local Whisper CPU compute.
    Cost: $0.0009 / call.
    """
    key = api_key or CHOCODATA_API_KEY or os.getenv("CHOCODATA_API_KEY", "")
    if not key:
        return None

    url = f"https://api.chocodata.com/api/v1/youtube/transcript?api_key={key}&video_id={video_id}"
    try:
        print(f"[*] Fetching transcript via Chocodata API for video {video_id}...")
        res = requests.get(url, timeout=15)
        if res.status_code == 200:
            data = res.json()
            raw_segments = data.get("segments", [])
            if raw_segments:
                print(f"[+] Chocodata returned {len(raw_segments)} timed segments in <1s!")
                return parse_native_transcript(raw_segments)
            else:
                print("[-] Chocodata: No transcript segments found for this video.")
        else:
            print(f"[-] Chocodata transcript error: {res.status_code} - {res.text[:100]}")
    except Exception as e:
        print(f"[-] Chocodata transcript fetch warning: {e}")
    return None

def parse_native_transcript(raw_transcript: List[Dict[str, Any]]) -> List[TranscriptSegment]:
    """Converts YouTubeTranscriptApi format into TranscriptSegment structures."""
    segments = []
    for item in raw_transcript:
        start = float(item.get("start", 0.0))
        duration = float(item.get("duration", 0.0))
        end = start + duration
        text = str(item.get("text", "")).strip().replace("\n", " ")
        if not text:
            continue
        
        # Approximate word timestamps from segment duration
        words_list = text.split()
        word_objs = []
        if words_list:
            word_duration = duration / len(words_list)
            for idx, w in enumerate(words_list):
                w_start = start + idx * word_duration
                w_end = w_start + word_duration
                word_objs.append(WordTimestamp(word=w, start=w_start, end=w_end, is_estimated=True))
        
        segments.append(TranscriptSegment(
            start=start,
            end=end,
            text=text,
            words=word_objs
        ))
    return segments

def transcribe_audio_whisper(
    video_or_audio_path: Path,
    model_size: Optional[str] = None,
    device: str = "cpu",
    compute_type: str = "int8"
) -> List[TranscriptSegment]:
    """
    Fallback transcription using faster-whisper on CPU.
    Runs free on GitHub Actions using int8 quantization.
    Provides word-level timestamps.
    """
    if model_size is None:
        # base.en (74MB int8) runs in ~3-4s on 4 vCPUs and provides vastly superior word-level timestamp alignment
        model_size = "base.en"

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ImportError("faster-whisper is required for audio transcription when native captions are missing.")

    model_key = (model_size, device, compute_type)
    model = _WHISPER_MODELS.get(model_key)
    if model is None:
        print(f"[*] Initializing Whisper ({model_size}, {device}, {compute_type})...")
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        _WHISPER_MODELS[model_key] = model
    else:
        print(f"[*] Reusing Whisper ({model_size}, {device}, {compute_type})...")

    print(f"[*] Transcribing audio from {video_or_audio_path.name}...")
    whisper_segments, _ = model.transcribe(
        str(video_or_audio_path),
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )

    results: List[TranscriptSegment] = []
    for seg in whisper_segments:
        words = []
        if seg.words:
            for w in seg.words:
                words.append(WordTimestamp(
                    word=w.word.strip(),
                    start=w.start,
                    end=w.end
                ))
        else:
            # Fallback if words not present
            words_split = seg.text.strip().split()
            if words_split:
                dur = (seg.end - seg.start) / len(words_split)
                for i, w in enumerate(words_split):
                    words.append(WordTimestamp(
                        word=w,
                        start=seg.start + i * dur,
                        end=seg.start + (i + 1) * dur,
                        is_estimated=True
                    ))

        results.append(TranscriptSegment(
            start=seg.start,
            end=seg.end,
            text=seg.text.strip(),
            words=words
        ))

    print(f"[+] Transcription complete: {len(results)} segments generated.")
    return results


def align_clip_transcript(
    clip_path: Path,
    render_start: float,
    output_start: float,
    output_end: float,
    fallback_segments: List[TranscriptSegment],
    model_size: Optional[str] = None,
) -> List[TranscriptSegment]:
    if not any(
        word.is_estimated
        and word.end > output_start
        and word.start < output_end
        for segment in fallback_segments
        for word in segment.words
    ):
        return fallback_segments
    aligned = transcribe_audio_whisper(clip_path, model_size=model_size, device="cpu", compute_type="int8")
    if not aligned:
        return fallback_segments
    shifted: List[TranscriptSegment] = []
    clip_file_duration = max(0.0, output_end - output_start + render_start)
    for segment in aligned:
        words = [
            WordTimestamp(
                word=word.word,
                start=max(output_start, min(output_end, output_start + (word.start - render_start))),
                end=max(output_start, min(output_end, output_start + (word.end - render_start))),
                is_estimated=bool(word.is_estimated),
            )
            for word in segment.words
            if word.end > render_start and word.start < clip_file_duration
        ]
        if not words:
            continue
        shifted.append(TranscriptSegment(
            start=words[0].start,
            end=words[-1].end,
            text=segment.text,
            words=words,
        ))
    return shifted or fallback_segments


def get_transcript(
    video_path: Path,
    native_transcript: Optional[List[Dict[str, Any]]] = None,
    video_id: Optional[str] = None
) -> List[TranscriptSegment]:
    """Fetches transcript with multi-tier fallback: Native YouTube -> Chocodata API -> Whisper CPU."""
    if native_transcript:
        print("[+] Using native YouTube transcript (0 compute cost).")
        return parse_native_transcript(native_transcript)

    # Fast Tier 1.5: Chocodata API
    target_id = video_id
    if not target_id:
        match = re.search(r'([0-9A-Za-z_-]{11})', video_path.stem)
        if match:
            target_id = match.group(1)

    if target_id and CHOCODATA_API_KEY:
        choco_segments = fetch_transcript_chocodata(target_id)
        if choco_segments:
            return choco_segments

    print("[!] Native/Chocodata transcript not found. Running faster-whisper speech-to-text...")
    return transcribe_audio_whisper(video_path)
