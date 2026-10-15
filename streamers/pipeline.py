"""V2 Streamer Clipper Pipeline Orchestrator.

Full autonomous pipeline specialised for US Streamers (Kai Cenat, IShowSpeed,
Jynxzi, CaseOh, xQc, Adin Ross):

  - Zero full-video download: transcript-first, targeted 30-45s range downloads
  - SFace identity locking + 3-tier state machine framing
  - TikTok meme hook box + kinetic subtitles
  - Audio scream/peak detection → punch-zoom
  - Quality gate (gate.verify_clip) BETWEEN render and publish
  - BUFFER ISOLATION: reads only STREAMER_BUFFER_* vars, never touches V1 tokens

Defects resolved
----------------
D1  — gate.verify_clip now sits between render_viral_clip and schedule_clip.
D18 — every clip iteration is wrapped in try/except; one failure cannot sink others.
D21 — schedule_clip return value is captured and logged.
D22 — caller validates channel IDs before constructing publisher.
"""

from __future__ import annotations

import gc
import os
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.config import (
    CLIPS_DIR,
    DOWNLOADS_DIR,
    SUBTITLES_DIR,
)
from core.downloader import (
    fetch_transcript_only,
    download_clip_segment,
    get_video_duration,
)
from core.jump_cutter import build_cut_plan, prepare_compacted_clip
from core.subtitle_generator import create_styled_ass_subtitles
from core.thumbnail_generator import generate_clip_thumbnail
from core.transcriber import (
    get_transcript,
    transcribe_audio_whisper,
    TranscriptSegment,
    WordTimestamp,
)
from core.video_editor import render_viral_clip

from streamers.audio_analyzer import analyze_audio_energy
from streamers.censor import censor_transcript_segments
from streamers.discovery import (
    discover_streamer_candidates,
    record_streamer_history,
    record_failed_attempt,
    extract_video_id_from_url,
)
from streamers.editing_enhancer import build_streamer_retention_plan
from streamers.gate import verify_clip, GateResult
from streamers.layout_compositor import build_streamer_framing
from streamers.paths import assert_within, safe_output_name, slugify_title
from streamers.spec import (
    STREAMER_MAX_CLIP_DURATION_S,
    STREAMER_MIN_CLIP_DURATION_S,
)
from streamers.viral_detector import (
    detect_streamer_viral_moments,
    StreamerClipCandidate,
)

# ---------------------------------------------------------------------------
# Runtime constants
# ---------------------------------------------------------------------------
STREAMER_CLIPS_DIR = CLIPS_DIR / "streamers"
STREAMER_CLIPS_DIR.mkdir(parents=True, exist_ok=True)

# Isolated credentials — NEVER use BUFFER_ACCESS_TOKEN or BUFFER_CHANNEL_IDS
STREAMER_BUFFER_ACCESS_TOKEN = os.getenv("STREAMER_BUFFER_ACCESS_TOKEN", "").strip()
STREAMER_BUFFER_CHANNEL_IDS: List[str] = [
    c.strip()
    for c in os.getenv("STREAMER_BUFFER_CHANNEL_IDS", "").split(",")
    if c.strip()
]
STREAMER_WATERMARK = os.getenv("STREAMER_WATERMARK", "@streamersclips").strip()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@dataclass
class _ClipResult:
    """Tracks the outcome of one clip attempt through the full pipeline."""

    clip_index: int
    candidate: StreamerClipCandidate
    rendered: Optional[Path] = None
    thumb_path: Optional[Path] = None
    gate: Optional[GateResult] = None
    published: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.rendered is not None and (self.gate is None or self.gate.passed)

    @property
    def published_ok(self) -> bool:
        return bool(self.published)


def _remap_segments_to_compacted_timeline(
    clip_segments: List[TranscriptSegment],
    cut_plan: Any,
) -> List[TranscriptSegment]:
    """Remap all segment/word timestamps through the dead-air cut plan."""
    remapped: List[TranscriptSegment] = []
    for s in clip_segments:
        new_words = [
            WordTimestamp(
                word=w.word,
                start=cut_plan.remap(w.start),
                end=cut_plan.remap(w.end),
                is_estimated=getattr(w, "is_estimated", False),
            )
            for w in (getattr(s, "words", []) or [])
        ]
        remapped.append(
            TranscriptSegment(
                start=cut_plan.remap(s.start),
                end=cut_plan.remap(s.end),
                text=s.text,
                words=new_words,
            )
        )
    return remapped


def _remap_sfx_cues(
    cues: List[Tuple[float, str]],
    segment_start: float,
    cut_plan: Any,
    clip_duration: float,
) -> List[Tuple[float, str]]:
    """Shift moment-relative SFX cues into the downloaded-file timeline.

    The viral detector emits ``sfx_cues`` relative to the moment start (t=0 at
    the hook), but the renderer works in the downloaded-file timeline, which
    begins ``segment_start`` seconds *before* the moment to provide hook
    pre-roll. Each cue is therefore shifted forward by ``segment_start``, passed
    through the dead-air cut plan (when one was applied), and finally dropped if
    it lands outside the rendered duration.
    """
    offset = float(segment_start or 0.0)
    remapped_cues: List[Tuple[float, str]] = []
    for cue_t, kind in cues:
        file_time = float(cue_t) + offset
        remapped = cut_plan.remap(file_time) if cut_plan.applied else file_time
        if 0.0 <= remapped <= clip_duration:
            remapped_cues.append((remapped, kind))
    return remapped_cues


def _record_video_failure(video_id: str, reason: str) -> None:
    """Record a failed processing attempt so the D10 retry budget can engage.

    Without this, a VOD that consistently fails (region-blocked, corrupt, or
    transcript-less) would be retried on every discovery run forever. Bookkeeping
    is best-effort: a history-write failure must never abort the run.
    """
    if not video_id or video_id == "stream_video":
        return
    try:
        record_failed_attempt(video_id, reason)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[!] Could not record failed attempt for {video_id}: {exc}")


def _build_local_fallback_segments(
    stream_segments: List[TranscriptSegment],
    vm: StreamerClipCandidate,
    clip_duration: float,
) -> List[TranscriptSegment]:
    """Shift stream-level segments to a clip-relative timeline when Whisper fails."""
    shift_offset = max(0.0, vm.start_time - 1.0)
    local: List[TranscriptSegment] = []
    for s in stream_segments:
        if s.end < shift_offset or s.start > shift_offset + clip_duration:
            continue
        shifted_words = [
            WordTimestamp(
                word=w.word,
                start=max(0.0, w.start - shift_offset),
                end=max(0.0, w.end - shift_offset),
                is_estimated=getattr(w, "is_estimated", False),
            )
            for w in (getattr(s, "words", []) or [])
            if w.end >= shift_offset and w.start <= shift_offset + clip_duration
        ]
        local.append(
            TranscriptSegment(
                start=max(0.0, s.start - shift_offset),
                end=min(clip_duration, s.end - shift_offset),
                text=s.text,
                words=shifted_words,
            )
        )
    return local


def _visual_outcome(gate: Optional[GateResult]) -> Optional[bool]:
    """Report the real V2 black-margin outcome, or ``None`` when it never ran.

    ``None`` means "no verdict" (gate never reached / check unavailable) which is
    materially different from ``False`` (check ran and failed).
    """
    if gate is None:
        return None
    if "v2_visual_check_unavailable" in gate.blocking_codes:
        # The audit never produced a verdict (fail-closed). That is "no result",
        # which is materially different from a check that ran and failed.
        return None
    if "v2_black_frame_margin" in gate.blocking_codes:
        return False
    if any(w.startswith("v2_visual_check_unavailable") for w in gate.warnings):
        return None
    return True


def _print_delivery_summary(results: List[_ClipResult], *, run_id: str) -> None:
    """Print a structured delivery summary at the end of the run."""
    passed = [r for r in results if r.succeeded]
    blocked = [r for r in results if r.gate and not r.gate.passed]
    errored = [r for r in results if r.error]
    published_ok = [r for r in results if r.published_ok]

    print("\n" + "═" * 70)
    print(f"  V2 PIPELINE DELIVERY SUMMARY  [run_id={run_id}]")
    print("═" * 70)
    print(f"  Clips rendered  : {len(passed)}")
    print(f"  Gate blocked    : {len(blocked)}")
    print(f"  Errors skipped  : {len(errored)}")
    print(f"  Published       : {len(published_ok)}")
    print("─" * 70)
    for r in results:
        if r.succeeded:
            status = "✓ PASS"
        elif r.gate and not r.gate.passed:
            status = f"✗ GATE({', '.join(r.gate.blocking_codes)})"
        elif r.error:
            status = "✗ ERROR"
        else:
            status = "✗ FAIL"
        pub = "→ PUBLISHED" if r.published_ok else ""
        print(f"  #{r.clip_index}: [{status}] {r.candidate.title[:45]} {pub}")
        if r.error:
            print(f"       error: {r.error}")
        if r.gate and r.gate.verdict_path:
            print(f"       verdict: {r.gate.verdict_path}")
    print("═" * 70 + "\n")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_streamer_pipeline(
    video_url: Optional[str] = None,
    creator_name: Optional[str] = None,
    num_clips: int = 3,
    post_to_buffer: bool = False,
    framing_mode: str = "auto",
    delay_days: Optional[int] = None,
    run_id: Optional[str] = None,
) -> List[Path]:
    """Run the V2 Streamer Viral Clipping Pipeline end-to-end.

    Args:
        video_url:     Direct YouTube VOD/stream URL, or None to auto-discover.
        creator_name:  Target streamer slug (e.g. 'kai_cenat', 'ishowspeed').
                       None = auto-discover across full roster.
        num_clips:     Number of viral clips to extract (1–5, validated by caller).
        post_to_buffer: Schedule clips to Buffer after passing gate. Default False.
        framing_mode:  Layout hint: 'auto', 'single_smooth', 'split_screen',
                       'stacked_webcam', 'fullscreen_room'.
        delay_days:    Days offset for Buffer scheduling (None = queue immediately).
        run_id:        Unique run identifier. Auto-generated if not provided.

    Returns:
        List of Paths to successfully rendered (and gate-passed) clip files.
    """
    if run_id is None:
        run_id = f"v2_{int(time.time())}_{uuid.uuid4().hex[:6]}"

    runs_dir = Path("data") / "runs" / run_id
    runs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"⚡ AUTONOMOUS AI STREAMER VIRAL CLIPPER (PIPELINE V2) — {run_id}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. URL resolution / transcript fetch
    # ------------------------------------------------------------------
    target_url: Optional[str] = video_url
    target_title: str = ""
    target_creator: str = creator_name or "Streamer"
    video_id: str = "stream_video"
    transcript_data: Optional[Dict[str, Any]] = None

    if not target_url:
        print(f"[*] Auto-discovering latest streams (Filter: {creator_name or 'All Top US Streamers'})...")
        try:
            candidates = discover_streamer_candidates(creator=creator_name, max_candidates=10)
        except Exception as exc:
            print(f"[-] Discovery failed: {exc}")
            candidates = []

        if not candidates:
            print("[-] No fresh streamer streams found in discovery feeds.")
            return []

        for cand in candidates:
            cand_url = cand["url"]
            cand_title = cand["title"]
            cand_creator = cand.get("creator", "Streamer")
            cand_id = cand["video_id"]
            print(f"[*] Checking candidate: '{cand_title}' by {cand_creator} ({cand_id})...")
            try:
                t_data = fetch_transcript_only(cand_url, allow_apify_fallback=True)
            except Exception as exc:
                print(f"[-] Transcript fetch failed for {cand_id}: {exc}")
                _record_video_failure(cand_id, f"transcript fetch raised: {exc}")
                continue

            if t_data and t_data.get("transcript"):
                target_url = cand_url
                target_title = cand_title
                target_creator = cand_creator
                video_id = cand_id
                transcript_data = t_data
                print(
                    f"[+] Active transcript ({len(t_data['transcript'])} entries) for "
                    f"'{target_title}' — proceeding to clipping."
                )
                break
            else:
                print(f"[-] No timed transcript for '{cand_title}'. Trying next...")
                _record_video_failure(cand_id, "no timed transcript available")

        if not target_url or not transcript_data:
            print("[-] No candidate stream had an accessible transcript.")
            return []
    else:
        print(f"[*] Using provided stream URL: {target_url}")
        video_id = extract_video_id_from_url(target_url) or "stream_video"
        print("\n--- [1/5] FETCHING TRANSCRIPT (ZERO VIDEO DOWNLOAD) ---")
        try:
            transcript_data = fetch_transcript_only(target_url, allow_apify_fallback=True)
        except Exception as exc:
            print(f"[-] Transcript fetch raised: {exc}")
            _record_video_failure(video_id, f"transcript fetch raised: {exc}")
            return []
        if not transcript_data or not transcript_data.get("transcript"):
            print("[-] Could not retrieve transcript for provided URL.")
            _record_video_failure(video_id, "no transcript retrieved for provided URL")
            return []

    try:
        segments: List[TranscriptSegment] = get_transcript(
            video_path=Path("__transcript_only__"),
            native_transcript=transcript_data["transcript"],
            video_id=video_id,
        )
    except Exception as exc:
        print(f"[-] Transcript parse raised: {exc}")
        _record_video_failure(video_id, f"transcript parse raised: {exc}")
        return []

    if not segments:
        print("[-] Could not parse transcript segments.")
        _record_video_failure(video_id, "transcript parsed to zero segments")
        return []

    print(f"[+] Retrieved {len(segments)} transcript segments. Analysing viral moments...")

    # ------------------------------------------------------------------
    # 2. Viral moment detection
    # ------------------------------------------------------------------
    print("\n--- [2/5] HIGH-ENERGY VIRAL MOMENT DETECTION ---")
    try:
        viral_moments: List[StreamerClipCandidate] = detect_streamer_viral_moments(
            segments, creator_name=target_creator, num_clips=num_clips
        )
    except Exception as exc:
        print(f"[-] Viral detection raised: {exc}")
        traceback.print_exc()
        _record_video_failure(video_id, f"viral detection raised: {exc}")
        return []

    if not viral_moments:
        print("[-] No viral moments detected in stream.")
        _record_video_failure(video_id, "no viral moments detected")
        return []

    print(f"[+] Detected {len(viral_moments)} viral moments:")
    for idx, vm in enumerate(viral_moments, start=1):
        print(f"  #{idx}: '{vm.title}' ({vm.duration:.1f}s, score={vm.viral_score}) — {vm.hook_reason}")

    # ------------------------------------------------------------------
    # 3. Per-clip download → render → gate
    # ------------------------------------------------------------------
    print("\n--- [3/5] TARGETED SEGMENT EXTRACTION & RENDERING ---")

    clip_results: List[_ClipResult] = []

    for idx, vm in enumerate(viral_moments, start=1):
        result = _ClipResult(clip_index=idx, candidate=vm)
        clip_results.append(result)

        print(f"\n>>> Processing Clip #{idx}: '{vm.title}' [{vm.start_time:.1f}s – {vm.end_time:.1f}s]")

        try:
            # 3a. Download targeted segment
            seg_end = min(vm.end_time, vm.start_time + STREAMER_MAX_CLIP_DURATION_S)
            clip_info = download_clip_segment(
                video_url=target_url,
                start_time=max(0.0, vm.start_time - 1.0),
                end_time=seg_end + 1.0,
                clip_index=idx,
                output_dir=DOWNLOADS_DIR,
                moment_start=vm.start_time,
            )
            if not clip_info or "video_path" not in clip_info:
                result.error = "download_clip_segment returned no video_path"
                print(f"[-] {result.error}; skipping clip #{idx}.")
                continue

            clip_path = Path(clip_info["video_path"])
            if not clip_path.exists():
                result.error = f"Downloaded path missing: {clip_path}"
                print(f"[-] {result.error}; skipping clip #{idx}.")
                continue

            clip_duration = get_video_duration(clip_path)

            # The download window is padded (-1s pre-roll / +1s post-roll) so the
            # raw file can exceed the contract: a 45s moment arrives as ~47s.
            # Cap the WORKING duration here so every downstream stage (dead-air
            # cut plan, framing, subtitles, SFX, retention plan, render, EDL)
            # operates on the same in-contract window. render_viral_clip clamps
            # to the source, so the tail is trimmed instead of the finished clip
            # being rejected by the gate (and charged to the VOD retry budget).
            if clip_duration > STREAMER_MAX_CLIP_DURATION_S:
                print(
                    f"[*] Clip #{idx}: capping {clip_duration:.2f}s to the "
                    f"{STREAMER_MAX_CLIP_DURATION_S:.0f}s contract (dropping post-roll)."
                )
                clip_duration = STREAMER_MAX_CLIP_DURATION_S

            # 3b. Transcription (Whisper; fallback to stream segments)
            try:
                whisper_segments = transcribe_audio_whisper(clip_path)
            except Exception as exc:
                print(f"[!] Whisper failed for clip #{idx}: {exc} — falling back to stream segments")
                whisper_segments = None
            gc.collect()

            clip_segments: List[TranscriptSegment] = (
                whisper_segments
                if whisper_segments
                else _build_local_fallback_segments(segments, vm, clip_duration)
            )

            # 3c. Dead-air trimming (jump cuts)
            p_words = [
                (w.start, w.end)
                for s in clip_segments
                for w in (getattr(s, "words", []) or [])
                if 0.0 <= w.start <= clip_duration
            ]
            cut_plan = build_cut_plan(p_words, clip_duration)
            if cut_plan.applied:
                print(f"[*] Pacing: removing {cut_plan.removed_s:.2f}s dead air...")
                prepared = prepare_compacted_clip(clip_path, p_words, clip_duration, clip_path.parent)
                if prepared:
                    clip_path, _, clip_duration = prepared
                    print(f"[+] Pacing applied: {clip_duration:.2f}s")
                    clip_segments = _remap_segments_to_compacted_timeline(clip_segments, cut_plan)
                    gc.collect()

            # 3d. Smart profanity & shadowban keyword masking (TikTok / Reels TOS safety)
            clip_segments = censor_transcript_segments(clip_segments)

            # 3e. Face tracking & layout composition
            framing = build_streamer_framing(
                video_path=clip_path,
                start_time=0.0,
                end_time=clip_duration,
                streamer_name=target_creator,
                force_layout=framing_mode,
            )

            # 3f. Audio energy / scream peak analysis
            try:
                audio_analysis = analyze_audio_energy(clip_path)
                peak_intervals = audio_analysis.get("peak_intervals", [])
                if not audio_analysis.get("available", True):
                    # Surface the degradation: a failed analysis is otherwise
                    # indistinguishable from "no peaks found" in the output.
                    print(
                        f"[!] Clip #{idx}: audio analysis unavailable — "
                        f"punch-zoom disabled for this clip."
                    )
            except Exception as exc:
                print(f"[!] Audio analysis failed for clip #{idx}: {exc} — disabling punch-zoom")
                peak_intervals = []

            if peak_intervals:
                print(f"[*] {len(peak_intervals)} scream/energy peaks detected.")

            # 3g. Styled ASS subtitles
            # video_id is regex-validated upstream, but route it through the same
            # allowlist so no remote-derived string is ever joined raw onto a path.
            ass_path = assert_within(
                SUBTITLES_DIR,
                SUBTITLES_DIR
                / f"streamer_clip_{idx}_{slugify_title(video_id, max_len=24)}.ass",
            )
            try:
                create_styled_ass_subtitles(
                    segments=clip_segments,
                    clip_start=0.0,
                    clip_end=clip_duration,
                    output_ass_path=ass_path,
                    theme_key="tiktok_streamer",
                    watermark=STREAMER_WATERMARK,
                    top_meme_title=vm.title,
                )
            except Exception as exc:
                print(f"[!] Subtitle generation failed for clip #{idx}: {exc}")
                ass_path = None  # render without subtitles if generation fails

            # 3h. Thumbnail
            # Title is untrusted model output — allowlist it before it becomes a path.
            thumb_path = assert_within(
                STREAMER_CLIPS_DIR,
                STREAMER_CLIPS_DIR
                / safe_output_name(f"streamer_clip_{idx}", vm.title, "_thumb.jpg"),
            )
            try:
                generate_clip_thumbnail(
                    clip_path=clip_path,
                    title=vm.title,
                    output_path=thumb_path,
                    badge_text="",
                    watermark=STREAMER_WATERMARK,
                    style="tiktok_meme",
                )
            except Exception as exc:
                print(f"[!] Thumbnail generation failed for clip #{idx}: {exc}")
                thumb_path = None

            # 3i. SFX cue remapping
            # sfx_cues are moment-relative; shift them into the downloaded-file
            # timeline (hook pre-roll) before applying the dead-air cut plan.
            active_sfx = _remap_sfx_cues(
                vm.sfx_cues,
                float(clip_info.get("segment_start", 0.0) or 0.0),
                cut_plan,
                clip_duration,
            )

            # 3j. High-retention editing plan (0.35s velocity hook zoom + pattern interrupts + synced whoosh)
            enhanced_peaks, active_sfx = build_streamer_retention_plan(
                clip_duration=clip_duration,
                peak_intervals=peak_intervals,
                sfx_cues=active_sfx,
                segments=clip_segments,
                inject_velocity_hook=True,
            )

            # 3k. Final FFmpeg render
            # The title comes from the LLM and is untrusted: allowlist it and add a
            # fingerprint so two clips sharing a title cannot collide on disk.
            out_clip_name = safe_output_name(f"clip_{idx}", vm.title, ".mp4")
            out_clip_path = assert_within(
                STREAMER_CLIPS_DIR, STREAMER_CLIPS_DIR / out_clip_name
            )

            rendered = render_viral_clip(
                source_video_path=clip_path,
                start_time=0.0,
                end_time=clip_duration,
                output_clip_path=out_clip_path,
                framing=framing,
                peak_intensity_segments=enhanced_peaks,
                ass_subtitle_path=ass_path,
                burn_subtitles=(ass_path is not None and ass_path.exists()),
                sfx_cues=active_sfx,
            )

            if not rendered or not rendered.exists():
                result.error = "render_viral_clip returned no output"
                print(f"[-] {result.error}; skipping clip #{idx}.")
                continue

            size_mb = rendered.stat().st_size / 1024 / 1024
            print(f"[+] Rendered clip #{idx} ({size_mb:.2f} MB): {rendered.name}")
            try:
                import json as _json
                edl_path = out_clip_path.with_suffix(".edl.json")
                edl_payload = {
                    "schema_version": 1,
                    "pipeline": "v2_streamer",
                    "run_id": run_id,
                    "clip_index": idx,
                    "title": vm.title,
                    "source_url": target_url,
                    "video_id": video_id,
                    "creator": target_creator,
                    "duration_s": round(clip_duration, 3),
                    "video_path": str(rendered),
                    "thumbnail_path": str(thumb_path) if thumb_path else None,
                    "ass_path": str(ass_path) if ass_path else None,
                    "framing_mode": framing_mode,
                    "watermark": STREAMER_WATERMARK,
                    "peak_intervals": peak_intervals,
                    "shots": [
                        {
                            "start": getattr(shot, "start", None),
                            "end": getattr(shot, "end", None),
                            "mode": getattr(shot, "mode", None),
                            "crop_x": getattr(shot, "crop_x", None),
                            "crop_y": getattr(shot, "crop_y", None),
                            "crop_w": getattr(shot, "crop_w", None),
                            "crop_h": getattr(shot, "crop_h", None),
                            "interpolate": False,
                        }
                        for shot in (getattr(framing, "shots", None) or [])
                    ],
                }
                edl_path.write_text(_json.dumps(edl_payload, indent=2, default=str), encoding="utf-8")
            except Exception as exc:
                print(f"[!] Could not write EDL sidecar for clip #{idx}: {exc}")

            if ass_path is not None and ass_path.exists():
                try:
                    import shutil
                    persisted_ass = STREAMER_CLIPS_DIR / (out_clip_path.stem + ".ass")
                    shutil.copyfile(ass_path, persisted_ass)
                except Exception as exc:
                    print(f"[!] Could not persist ASS for clip #{idx}: {exc}")

            # -------------------------------------------------------
            # 3j. QUALITY GATE — nothing passes without verification
            # -------------------------------------------------------
            gate_result = verify_clip(
                rendered,
                run_id=run_id,
                runs_dir=runs_dir,
                min_duration_s=STREAMER_MIN_CLIP_DURATION_S,
                max_duration_s=STREAMER_MAX_CLIP_DURATION_S,
            )
            result.gate = gate_result
            result.rendered = rendered
            result.thumb_path = thumb_path
            if gate_result.passed and os.getenv("V2_DISABLE_PREVIEW", "").lower() not in ("1", "true", "yes"):
                try:
                    import subprocess
                    preview_path = out_clip_path.with_name(out_clip_path.stem + "_preview.mp4")
                    cmd = [
                        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-i", str(rendered), "-vf", "scale=540:960",
                        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
                        "-c:a", "copy", "-movflags", "+faststart", str(preview_path),
                    ]
                    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                    if proc.returncode == 0 and preview_path.exists():
                        print(f"  [+] preview: {preview_path.name}")
                    else:
                        print(f"  [!] preview generation failed: {proc.stderr[-200:]}")
                except Exception as exc:
                    print(f"  [!] preview generation exception: {exc}")

            if not gate_result.passed:
                print(
                    f"[-] Clip #{idx} BLOCKED by gate: {gate_result.blocking_codes}. "
                    f"Verdict at {gate_result.verdict_path}"
                )
                continue

        except Exception as exc:  # pylint: disable=broad-except
            # D18: per-clip exception boundary — one clip failure cannot kill the run
            result.error = f"{type(exc).__name__}: {exc}"
            print(f"[-] Clip #{idx} raised unexpected exception — continuing to next clip.")
            traceback.print_exc()
            continue

    # ------------------------------------------------------------------
    # 3.5 Run manifest (machine-readable production audit)
    # ------------------------------------------------------------------
    try:
        import json as _json
        run_manifest = {
            "run_id": run_id,
            "source_url": target_url,
            "video_id": video_id,
            "creator": target_creator,
            "clips": [
                {
                    "clip_index": r.clip_index,
                    "title": r.candidate.title,
                    "rendered": str(r.rendered) if r.rendered else None,
                    "thumbnail": str(r.thumb_path) if r.thumb_path else None,
                    "passed": bool(r.gate.passed) if r.gate else False,
                    "blocking_codes": list(r.gate.blocking_codes) if r.gate else [],
                    "verdict_path": str(r.gate.verdict_path) if r.gate and r.gate.verdict_path else None,
                    "error": r.error,
                }
                for r in clip_results
            ],
        }
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / "run_manifest.json").write_text(_json.dumps(run_manifest, indent=2, default=str), encoding="utf-8")
    except Exception as exc:
        print(f"[!] Could not write run manifest: {exc}")

    # ------------------------------------------------------------------
    # 3.6 Production report (merged QC summary)
    # ------------------------------------------------------------------
    try:
        import json as _json
        passed = [r for r in clip_results if r.succeeded]
        blocked = [r for r in clip_results if r.gate and not r.gate.passed]
        report = {
            "run_id": run_id,
            "total_clips": len(clip_results),
            "passed": len(passed),
            "blocked": len(blocked),
            "errors": len([r for r in clip_results if r.error]),
            "published_target": "streamer_buffer_only",
            "clips": [
                {
                    "clip_index": r.clip_index,
                    "title": r.candidate.title,
                    "duration_s": r.gate.duration_s if r.gate else 0.0,
                    "passed": bool(r.gate.passed) if r.gate else False,
                    "warnings": list(r.gate.warnings) if r.gate else [],
                    "blocking_codes": list(r.gate.blocking_codes) if r.gate else [],
                    # Report the real black-margin outcome. The previous expression
                    # dereferenced r.gate.__dict__ unconditionally, so any clip that
                    # failed BEFORE the gate raised AttributeError here — and the
                    # surrounding try swallowed it, silently skipping the whole
                    # production report for the run.
                    "visual": _visual_outcome(r.gate),
                    "error": r.error,
                }
                for r in clip_results
            ],
        }
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / "production_report.json").write_text(_json.dumps(report, indent=2, default=str), encoding="utf-8")
    except Exception as exc:
        print(f"[!] Could not write production report: {exc}")

    # ------------------------------------------------------------------
    # 4. Buffer publishing (gate-passed clips only)
    # ------------------------------------------------------------------
    print("\n--- [4/5] BUFFER SOCIAL PUBLISHING ISOLATION ---")

    if post_to_buffer:
        if not STREAMER_BUFFER_ACCESS_TOKEN:
            print("[-] STREAMER_BUFFER_ACCESS_TOKEN not configured. Skipping Buffer dispatch.")
        elif not STREAMER_BUFFER_CHANNEL_IDS:
            # D22: refuse to broadcast to every channel in the account
            print(
                "[-] STREAMER_BUFFER_CHANNEL_IDS not configured. "
                "Refusing to broadcast to all channels. "
                "Set STREAMER_BUFFER_CHANNEL_IDS to one or more channel IDs."
            )
        else:
            from streamers.buffer_publisher import StreamerBufferPublisher

            publisher = StreamerBufferPublisher()
            gate_passed = [r for r in clip_results if r.succeeded and r.gate and r.gate.passed]

            for r in gate_passed:
                print(f"[*] Publishing clip #{r.clip_index}: '{r.candidate.title}'...")
                try:
                    # D21: capture and log the return value
                    dispatch_result = publisher.schedule_clip(
                        clip_path=r.rendered,
                        candidate=r.candidate,
                        thumb_path=r.thumb_path,
                        delay_days=delay_days,
                    )
                    r.published = dispatch_result
                    if dispatch_result:
                        print(f"[+] Clip #{r.clip_index} scheduled: {len(dispatch_result)} channel(s)")
                        for entry in dispatch_result:
                            cid = entry.get("channel_id", "?")
                            svc = entry.get("service", "?")
                            status = entry.get("status", "?")
                            print(f"    → channel={cid} service={svc} status={status}")
                    else:
                        print(f"[-] Clip #{r.clip_index}: schedule_clip returned empty — check publisher logs")
                except Exception as exc:
                    print(f"[-] Buffer dispatch raised for clip #{r.clip_index}: {type(exc).__name__}: {exc}")
    else:
        print("[*] BUFFER ISOLATION ACTIVE:")
        print("    → V1 Podcast Buffer credentials were NOT touched.")
        print("    → Clips saved locally in clips/streamers/ for manual review.")

    # ------------------------------------------------------------------
    # 5. Record history
    # ------------------------------------------------------------------
    print("\n--- [5/5] RECORDING HISTORY ---")
    if target_url:
        succeeded = [r for r in clip_results if r.succeeded]
        if succeeded:
            try:
                record_streamer_history(video_id, title=target_title, creator=target_creator)
            except Exception as exc:
                print(f"[!] History write raised: {exc}")
        else:
            # No publishable clip this run: count it against the retry budget
            # (D10) instead of marking the VOD permanently processed, so a
            # transient failure can be retried up to the attempt ceiling.
            reason = next((r.error for r in clip_results if r.error), "produced no publishable clip")
            _record_video_failure(video_id, reason)

    # ------------------------------------------------------------------
    # 6. Delivery summary
    # ------------------------------------------------------------------
    _print_delivery_summary(clip_results, run_id=run_id)

    # ------------------------------------------------------------------
    # 7. Retention sweep (bounded disk growth)
    # ------------------------------------------------------------------
    # Runs on the scheduled path so the checkout cannot grow forever: expired run
    # directories, download working copies, subtitle stylesheets and derived
    # render artifacts are removed. Rendered clips and thumbnails are
    # deliverables and are deliberately retained. Non-fatal by design — a publish
    # run must never fail because cleanup failed.
    try:
        from streamers.retention import run_retention_sweep

        sweep = run_retention_sweep(
            runs_root=runs_dir.parent,
            downloads_dir=DOWNLOADS_DIR,
            subtitles_dir=SUBTITLES_DIR,
            clips_dir=STREAMER_CLIPS_DIR,
            current_run_id=run_id,
        )
        if sweep["run_dirs"] or sweep["workspace_files"]:
            print(
                f"[*] Retention sweep removed {len(sweep['run_dirs'])} run dir(s) "
                f"and {len(sweep['workspace_files'])} workspace file(s)."
            )
    except Exception as exc:
        print(f"[!] Retention sweep failed (non-fatal): {exc}")

    rendered_ok = [r.rendered for r in clip_results if r.succeeded]
    return rendered_ok
