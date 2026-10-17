"""V2 Streamer Publish Gate.

The single authority that decides whether a rendered clip may be published.

Why this module exists
----------------------
Pipeline V1 (main.py) has an AST-pinned publish gate wiring verdict.verify →
repair.render_with_repair → BufferPublisher. V2 was built alongside it and
inherited none of that, so it originally reached Buffer with nothing between
render_viral_clip and the dispatch call. This module closes that gap: it composes
the existing public verification primitives (core.verification.verdict.verify)
behind a single V2-scoped function that pipeline.py calls instead of invoking
Buffer directly.

Failure policy (fail closed)
----------------------------
This gate is the last line before a live account, so "we could not check" is
treated as "do not publish":

* A required check that cannot run at all (cv2 absent, frames undecodable)
  produces the blocking code ``v2_visual_check_unavailable`` and the clip is
  refused. ``spec.STRICT_VISUAL_CHECKS=0`` downgrades that to a warning; it is an
  explicit, temporary, audited override, not a configuration convenience.
* Every "could not run" path carries a machine-readable ``reason`` (for example
  ``cv2/numpy import failed: ...`` versus ``no decodable frames sampled in ...``)
  so an environment fault is never confused with a content fault.
* Non-blocking checks (caption collision, face stability) still record their
  absence in ``warnings``, so a skipped check cannot read as a clean pass.

V1 contract
-----------
This module does NOT touch or import anything from:
    main.py, core/face_tracker.py, core/viral_detector.py
It composes only from core/verification/ (public API) and v2/streamers/spec.py.

Usage
-----
    from streamers.gate import verify_clip, GateResult

    result = verify_clip(rendered_path, candidate=vm, run_id=run_id)
    if result.passed:
        publisher.schedule_clip(...)
    else:
        log_blocked(result)
"""

from __future__ import annotations

import os
import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from streamers.spec import (
    STRICT_VISUAL_CHECKS,
    STREAMER_MAX_CLIP_DURATION_S,
    STREAMER_MIN_CLIP_DURATION_S,
    STREAMER_OUTPUT_HEIGHT,
    STREAMER_OUTPUT_WIDTH,
)

from core.config import MODELS_DIR as _MODELS_DIR

# YuNet weights live in the shared engine's models dir (env-overridable).
YUNET_MODEL_PATH = _MODELS_DIR / "face_detection_yunet.onnx"

# Face-stability metric. The gate measures the largest horizontal move of the
# largest face's centre between two *adjacent* frames, expressed in face widths,
# in three local windows (roughly 25/50/75% through the clip, 8 frames each).
#
# Adjacency is the whole point of the metric. The first live run's clips swayed
# 4.5px in total, yet the previous implementation — the largest difference
# between samples ~3s apart — read ordinary head movement as a 273-415px "jump"
# and blocked both clips. What a viewer experiences as broken tracking is a face
# per 33ms is a whip-pan, not movement. For high-energy streamers with intentional
# multi-shot camera cuts (wide listening vs close-up reaction), 1.35 bounds jitter
# while allowing genuine crash-out reactions and cut transitions.
FACE_JUMP_MAX_FACE_WIDTHS = float(os.getenv("FACE_JUMP_MAX_FACE_WIDTHS", "1.35"))
FACE_JUMP_WINDOW_FRACTIONS = (0.25, 0.50, 0.75)
FACE_JUMP_FRAMES_PER_WINDOW = 8


# Lazy import: do not fail at module load if ffmpeg is absent in test env
def _import_verdict():
    from core.verification import verdict as _v
    return _v


@dataclass
class GateResult:
    """Outcome of the V2 publish gate for one rendered clip."""

    path: str
    passed: bool = False
    blocking_codes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    verdict_path: Optional[Path] = None
    duration_s: float = 0.0
    dimensions: Dict[str, int] = field(default_factory=dict)
    repair_attempted: bool = False
    error: Optional[str] = None

    def summary_line(self) -> str:
        if self.passed:
            return (
                f"[GATE ✓] {Path(self.path).name} passed "
                f"({self.duration_s:.1f}s, "
                f"{self.dimensions.get('width', '?')}x{self.dimensions.get('height', '?')})"
            )
        codes = ", ".join(self.blocking_codes) if self.blocking_codes else "unknown"
        return (
            f"[GATE ✗] {Path(self.path).name} BLOCKED — {codes}"
        )



def _face_unavailable(reason: str) -> dict:
    """Face-stability could not run. ``reason`` is machine-readable on purpose.

    Every "unavailable" path used to return an anonymous zeroed dict, so the
    caller could not tell an environment problem (cv2 missing) from a content
    problem (no faces found). Downstream logging loses that distinction for good.
    """
    return {
        "available": False,
        "reason": reason,
        "face_presence_ratio": 0.0,
        "max_horizontal_jump_px": 0.0,
        "sampled_frames": 0,
        "warning": "face_stability_unavailable",
    }


def _consecutive_face_jumps(faces):
    """Largest horizontal face-centre jumps between adjacent measured frames.

    ``faces`` is an ordered list of ``(frame_index, centre_x, face_width)`` for
    every frame the check could measure, with ``None`` where no face was found
    (or the frame was too dark to measure). ``None`` breaks the chain, and a
    jump is only counted between entries whose frame indices are *adjacent* —
    those are the only moves that read as one motion to a viewer, so windows
    that do not touch are never compared across the gap.

    Returns ``(max_px, max_face_widths, samples)`` where ``samples`` counts the
    adjacent pairs actually compared, so "0.0px over 0 pairs" (nothing
    measurable) is distinguishable from "0.0px over 40 pairs".
    """
    max_px = 0.0
    max_widths = 0.0
    samples = 0
    prev = None
    for entry in faces:
        if entry is None:
            prev = None
            continue
        idx, cx, width = entry
        if prev is not None and idx == prev[0] + 1:
            jump = abs(cx - prev[1])
            # Relative to the narrower of the two boxes: the conservative read.
            widths = jump / max(1.0, min(width, prev[2]))
            samples += 1
            max_px = max(max_px, jump)
            max_widths = max(max_widths, widths)
        prev = entry
    return max_px, max_widths, samples


def _face_stability_check(path) -> dict:
    """Measure the largest inter-frame face-centre move in three local windows.

    Replaces a metric that sampled frames ~3s apart and compared their centres
    directly: that measures how far the head travelled between samples, not how
    stable the framing is. The first live run's clips swayed 4.5px in total and
    still produced a 273-415px "jump" — every clip blocked for ordinary
    movement (see FACE_JUMP_MAX_FACE_WIDTHS for what now blocks).
    """
    model_path = str(YUNET_MODEL_PATH)
    try:
        import cv2
        import numpy as np  # noqa: F401  (declared optional dependency)
    except Exception as exc:
        return _face_unavailable(f"cv2/numpy import failed: {exc}")

    if not Path(model_path).exists():
        return _face_unavailable(f"YuNet weights not found at {model_path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return _face_unavailable(f"could not open {path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            return _face_unavailable(f"frame count unreadable for {path}")

        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1920)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1080)
        detector = cv2.FaceDetectorYN.create(model_path, "", (w, h), 0.5, 0.3, 5000)

        window = max(2, FACE_JUMP_FRAMES_PER_WINDOW)
        faces: list = []
        read_indices: set = set()
        valid = 0
        seen = 0
        for fraction in FACE_JUMP_WINDOW_FRACTIONS:
            first = int(total_frames * fraction) - window // 2
            first = max(0, min(first, max(0, total_frames - window)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, first)
            last_center = None
            for offset in range(window):
                idx = first + offset
                if idx >= total_frames or idx in read_indices:
                    continue
                read_indices.add(idx)
                ok, frame = cap.read()
                if not ok or frame is None or frame.size == 0:
                    faces.append(None)
                    last_center = None
                    break
                if float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()) < 12.0:
                    faces.append(None)
                    last_center = None
                    continue
                valid += 1
                _, res = detector.detect(frame)
                if res is None or len(res) == 0:
                    faces.append(None)
                    last_center = None
                    continue

                # Target association: pick the face closest to the previously tracked
                # center (within 2.5 face widths) to prevent hopping to gameplay NPC faces.
                if last_center is not None:
                    prev_cx, prev_cy, prev_w = last_center
                    candidates = []
                    for r in res:
                        rcx = float(r[0]) + float(r[2]) / 2.0
                        rcy = float(r[1]) + float(r[3]) / 2.0
                        dist = ((rcx - prev_cx) ** 2 + (rcy - prev_cy) ** 2) ** 0.5
                        if dist <= 2.5 * prev_w:
                            candidates.append((dist, r))
                    if candidates:
                        chosen = min(candidates, key=lambda c: c[0])[1]
                    else:
                        chosen = max(res, key=lambda r: float(r[2]) * float(r[3]))
                else:
                    chosen = max(res, key=lambda r: float(r[2]) * float(r[3]))

                seen += 1
                cx = float(chosen[0]) + float(chosen[2]) / 2.0
                cy = float(chosen[1]) + float(chosen[3]) / 2.0
                cw = float(chosen[2])
                last_center = (cx, cy, cw)
                faces.append((idx, cx, cw))

        max_jump, max_widths, samples = _consecutive_face_jumps(faces)
        presence = (seen / valid) if valid else 0.0
        return {
            "available": True,
            "face_presence_ratio": round(presence, 3),
            "max_horizontal_jump_px": round(max_jump, 2),
            "max_jump_face_widths": round(max_widths, 3),
            "jump_samples": samples,
            "sampled_frames": valid,
            "warning": None,
        }
    finally:
        cap.release()



def _caption_unavailable(reason: str) -> dict:
    """Caption-collision check could not run (see ``_face_unavailable``)."""
    return {
        "available": False,
        "reason": reason,
        "collision": False,
        "warning": "caption_check_unavailable",
    }


def _caption_collision_check(path, model_path=None) -> dict:
    """Sample one frame and flag caption-like white stickers over large faces."""
    if model_path is None:
        model_path = str(YUNET_MODEL_PATH)
    try:
        import cv2
        import numpy as np
    except Exception as exc:
        return _caption_unavailable(f"cv2/numpy import failed: {exc}")

    if not Path(model_path).exists():
        return _caption_unavailable(f"YuNet weights not found at {model_path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return _caption_unavailable(f"could not open {path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            return _caption_unavailable(f"frame count unreadable for {path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            return _caption_unavailable(f"no decodable mid-clip frame in {path}")

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detector = cv2.FaceDetectorYN.create(model_path, "", (w, h), 0.5, 0.3, 5000)
        _, faces = detector.detect(frame)

        # White sticker candidates in heading/meme-box safe region (top 4% to 20% of canvas).
        # Limits search to the actual meme-box band to avoid catching white shirts, room lights,
        # or bright game graphics across the rest of the frame.
        sticker_y_min = int(h * 0.04)
        sticker_y_max = int(h * 0.20)
        white_mask = (gray > 240)
        white_rows = white_mask.sum(axis=1)
        top_band_rows = [i for i in range(sticker_y_min, sticker_y_max) if white_rows[i] > w * 0.15]
        if not top_band_rows:
            return {"available": True, "collision": False, "collision_overlap_px": 0.0}

        y0, y1 = min(top_band_rows), max(top_band_rows)
        band_h = max(1, y1 - y0 + 1)
        white_cols = white_mask[y0:y1+1, :].sum(axis=0)
        x0 = int(np.argmax(white_cols > band_h * 0.3))
        x1 = w - int(np.argmax(white_cols[::-1] > band_h * 0.3))

        collision = False
        overlap_area = 0.0
        if faces is not None and len(faces) > 0:
            largest = max(faces, key=lambda r: float(r[2]) * float(r[3]))
            fx, fy, fw, fh = [float(v) for v in largest[:4]]
            face_area = max(1.0, fw * fh)
            ix0 = max(fx, x0)
            iy0 = max(fy, y0)
            ix1 = min(fx + fw, x1)
            iy1 = min(fy + fh, y1)
            if ix1 > ix0 and iy1 > iy0:
                overlap_area = (ix1 - ix0) * (iy1 - iy0)
                # Significant obscuring collision: requires at least 15% face coverage
                if overlap_area >= 0.15 * face_area:
                    collision = True

        return {
            "available": True,
            "collision": collision,
            "collision_overlap_px": round(float(overlap_area), 2),
            "white_band": [int(y0), int(y1), int(x0), int(x1)],
        }
    finally:
        cap.release()


def _visual_unavailable(reason: str) -> dict:
    """The black-margin audit could not run.

    Returns ``passed: True`` so this function never invents a *content* failure,
    but pairs it with ``available: False`` and a machine-readable ``reason``. The
    publish decision for an unavailable check is made by ``verify_clip`` under the
    ``STRICT_VISUAL_CHECKS`` policy — never silently, and never here.
    """
    return {
        "available": False,
        "reason": reason,
        "passed": True,
        "worst_margin_pct": 0.0,
        "worst_side": "none",
        "sampled_frames": 0,
        "warning": "visual_check_unavailable",
    }


def _visual_black_margin_check(path) -> dict:
    """Sample rendered frames and reject shots with excessive black margins.

    V2 framing must be full-screen 9:16. Partial black borders can appear during
    Tier 3 off-screen fallback or a failed stacked/multi-person composition.
    """
    try:
        import cv2
        import numpy as np  # noqa: F401  (declared optional dependency)
    except Exception as exc:
        return _visual_unavailable(f"cv2/numpy import failed: {exc}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return _visual_unavailable(f"could not open {path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            return _visual_unavailable(f"frame count unreadable for {path}")

        margin_px = 40
        sample_count = min(12, total_frames)
        indices = [0] if sample_count <= 1 else sorted(set(
            int(i * (total_frames - 1) / (sample_count - 1))
            for i in range(sample_count)
        ))

        valid = 0
        side_samples = {"top": [], "bottom": [], "left": [], "right": []}

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                continue
            valid += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]
            margin = min(margin_px, max(8, h // 30), max(8, w // 30))

            # Measure full-length edge bar depth for each side:
            # A real letterbox/pillarbox has contiguous rows or columns at the border
            # that are almost entirely black across their full extent (>=85% black pixels).
            top_bar = 0
            for r in range(margin):
                if (gray[r, :] < 16).mean() >= 0.85:
                    top_bar += 1
                else:
                    break

            bot_bar = 0
            for r in range(h - 1, h - 1 - margin, -1):
                if (gray[r, :] < 16).mean() >= 0.85:
                    bot_bar += 1
                else:
                    break

            left_bar = 0
            for c in range(margin):
                if (gray[:, c] < 16).mean() >= 0.85:
                    left_bar += 1
                else:
                    break

            right_bar = 0
            for c in range(w - 1, w - 1 - margin, -1):
                if (gray[:, c] < 16).mean() >= 0.85:
                    right_bar += 1
                else:
                    break

            side_samples["top"].append(top_bar / margin)
            side_samples["bottom"].append(bot_bar / margin)
            side_samples["left"].append(left_bar / margin)
            side_samples["right"].append(right_bar / margin)

        if valid == 0:
            return _visual_unavailable(f"no decodable frames sampled in {path}")

        # Persistent margin: compute median bar percentage per side across sampled frames.
        side_medians = {
            side: float(np.median(vals)) * 100.0 for side, vals in side_samples.items()
        }
        worst_side = max(side_medians, key=side_medians.get)
        worst_pct = side_medians[worst_side]

        return {
            "available": True,
            "passed": worst_pct < 20.0,
            "worst_margin_pct": round(worst_pct, 2),
            "worst_side": worst_side,
            "sampled_frames": valid,
        }
    finally:
        cap.release()


def verify_clip(
    path: Path,
    *,
    run_id: str = "",
    runs_dir: Optional[Path] = None,
    min_duration_s: float = STREAMER_MIN_CLIP_DURATION_S,
    max_duration_s: float = STREAMER_MAX_CLIP_DURATION_S,
) -> GateResult:
    """Run every quality check against a rendered V2 clip.

    Parameters
    ----------
    path:
        Absolute path to the rendered MP4 clip.
    run_id:
        Identifier for the current pipeline run (used in verdict.json path).
    runs_dir:
        Override directory for verdict.json output. Defaults to
        ``data/runs/<run_id>/``.
    min_duration_s / max_duration_s:
        Duration contract for this clip (defaults to V2 spec: 30–45 s).

    Returns
    -------
    GateResult:
        .passed is True iff the clip meets every blocking criterion.
        .verdict_path points to the emitted verdict.json (always written).
    """
    result = GateResult(path=str(path))

    if not path.exists():
        result.error = f"Rendered clip not found: {path}"
        result.blocking_codes.append("output_missing")
        print(f"  [gate] ✗ {result.error}")
        return result

    # ------------------------------------------------------------------
    # 1. Run the full pixel-level verifier
    # ------------------------------------------------------------------
    verdict_mod = None
    raw_verdict = None
    try:
        verdict_mod = _import_verdict()
        raw_verdict = verdict_mod.verify(
            path,
            expected_width=STREAMER_OUTPUT_WIDTH,
            expected_height=STREAMER_OUTPUT_HEIGHT,
            with_mirror=False,   # Streamer clips: mirror check off (fast cam flips)
            with_captions=True,
            with_sway=True,
            min_duration_s=min_duration_s,
            max_duration_s=max_duration_s,
        )
        result.duration_s = raw_verdict.duration_s
        result.dimensions = raw_verdict.dimensions or {}
    except Exception as exc:  # pragma: no cover
        result.error = f"verdict.verify raised: {exc}"
        result.blocking_codes.append("verifier_error")
        print(f"  [gate] ✗ verifier exception — {exc}")
        traceback.print_exc()
        return result

    # ------------------------------------------------------------------
    # 2. Enforce V2-specific duration contract
    #    (verdict.py's own CONTRACT_MAX is 55s; we need 45s for V2)
    # ------------------------------------------------------------------
    dur = raw_verdict.duration_s
    duration_ok = (min_duration_s - 0.15) <= dur <= (max_duration_s + 0.15)
    if not duration_ok:
        raw_verdict.add(
            verdict_mod.Finding(
                code="v2_duration_out_of_contract",
                severity="error",
                message=(
                    f"V2 duration {dur:.2f}s outside [{min_duration_s:.0f}s, "
                    f"{max_duration_s:.0f}s] contract"
                ),
                spec_clause="streamers/spec.py STREAMER_MAX_CLIP_DURATION_S",
                metrics={"duration_s": round(dur, 3)},
            )
        )

    # Enforce strict zero-drift A/V sync (stops any lip-sync drift before publish)
    av_sync_metrics = getattr(raw_verdict, "metrics", {}).get("av_sync", {})
    dur_delta = abs(float(av_sync_metrics.get("duration_delta_ms", 0.0) or 0.0))
    start_delta = abs(float(av_sync_metrics.get("start_delta_ms", 0.0) or 0.0))
    if dur_delta > 40.0 or start_delta > 35.0:
        raw_verdict.add(
            verdict_mod.Finding(
                code="v2_av_sync_drift",
                severity="error",
                message=(
                    f"V2 A/V sync drift detected: duration delta {dur_delta:.1f}ms, "
                    f"start delta {start_delta:.1f}ms exceeds strict 40ms broadcast limit"
                ),
                spec_clause="streamer frame-accurate lip-sync",
                metrics=av_sync_metrics,
            )
        )

    caption_collision = _caption_collision_check(path)
    if caption_collision.get("available") and caption_collision.get("collision"):
        raw_verdict.add(
            verdict_mod.Finding(
                code="v2_caption_face_collision",
                severity="error",
                message="V2 caption sticker overlaps a sampled face box",
                spec_clause="streamer caption safe-zone",
                metrics=caption_collision,
            )
        )

    face_stability = _face_stability_check(path)
    if (
        face_stability.get("available")
        and face_stability.get("max_jump_face_widths", 0.0) > FACE_JUMP_MAX_FACE_WIDTHS
    ):
        raw_verdict.add(
            verdict_mod.Finding(
                code="v2_excessive_face_jump",
                severity="error",
                message=(
                    f"V2 face jump {face_stability['max_horizontal_jump_px']:.1f}px "
                    f"({face_stability['max_jump_face_widths']:.2f} face widths) between "
                    f"adjacent frames exceeds {FACE_JUMP_MAX_FACE_WIDTHS:.2f} face widths"
                ),
                spec_clause="streamer face-tracking stability",
                metrics=face_stability,
            )
        )
    if face_stability.get("available") and face_stability.get("face_presence_ratio", 0.0) < 0.5:
        raw_verdict.add(
            verdict_mod.Finding(
                code="v2_low_face_presence",
                severity="warning",
                message=(
                    f"V2 face stability found face in only "
                    f"{face_stability['face_presence_ratio'] * 100:.0f}% of sampled frames"
                ),
                spec_clause="streamer face-lock quality",
                metrics=face_stability,
            )
        )

    # ------------------------------------------------------------------
    # 2b. Visual black-margin audit
    # ------------------------------------------------------------------
    visual_check = _visual_black_margin_check(path)
    visual_unavailable_reason: Optional[str] = None
    if visual_check.get('available') and not visual_check.get('passed'):
        raw_verdict.add(
            verdict_mod.Finding(
                code='v2_black_frame_margin',
                severity='error',
                message=(
                    f'V2 visual check found {visual_check["worst_margin_pct"]:.1f}% '
                    f'black margin ({visual_check["worst_side"]})'
                ),
                spec_clause='streamer 9:16 full-frame composition',
                metrics=visual_check,
            )
        )
    elif not visual_check.get('available'):
        visual_unavailable_reason = str(visual_check.get('reason') or 'unknown')

    # ------------------------------------------------------------------
    # 3. Collect findings
    # ------------------------------------------------------------------
    blocking = [f.code for f in raw_verdict.findings if f.blocks]
    warnings = [f.code for f in raw_verdict.findings if f.severity == "warning"]
    if 'v2_visual_check_unavailable' in getattr(raw_verdict, 'warnings', []):
        warnings.append('v2_visual_check_unavailable')

    # ------------------------------------------------------------------
    # 2c. Policy for checks that could not run
    # ------------------------------------------------------------------
    # An unverifiable clip must not reach a live account. The black-margin check
    # IS the V2 composition guarantee, and a check that silently skipped is
    # indistinguishable from a check that passed in every log, artifact and
    # metric downstream. Fail closed by default (spec.STRICT_VISUAL_CHECKS).
    if visual_unavailable_reason is not None:
        warnings.append(f'v2_visual_check_unavailable:{visual_unavailable_reason}')
        if STRICT_VISUAL_CHECKS:
            blocking.append('v2_visual_check_unavailable')
            print(
                f"  [gate] ✗ visual check unavailable "
                f"({visual_unavailable_reason}) — failing closed"
            )

    # Non-blocking checks: record their absence explicitly so the coverage gap is
    # visible in the verdict and the production report instead of reading as a
    # clean pass.
    if not face_stability.get('available'):
        warnings.append(
            f"v2_face_stability_unavailable:{face_stability.get('reason', 'unknown')}"
        )
    if not caption_collision.get('available'):
        warnings.append(
            f"v2_caption_check_unavailable:{caption_collision.get('reason', 'unknown')}"
        )

    result.blocking_codes = blocking
    result.warnings = warnings
    result.passed = len(blocking) == 0

    # ------------------------------------------------------------------
    # 4. Write verdict.json
    # ------------------------------------------------------------------
    try:
        if runs_dir is None:
            runs_dir = Path("data") / "runs" / (run_id or f"run_{int(time.time())}")
        runs_dir.mkdir(parents=True, exist_ok=True)
        verdict_file = runs_dir / f"{path.stem}_verdict.json"

        payload: Dict[str, Any] = {
            "schema_version": 1,
            "pipeline": "v2_streamer",
            "clip": path.name,
            "passed": result.passed,
            "duration_s": round(dur, 3),
            "dimensions": result.dimensions,
            "blocking_codes": blocking,
            "warnings": warnings,
            "findings": [f.as_dict() for f in raw_verdict.findings],
            "metrics": {**getattr(raw_verdict, "metrics", {}), "visual": visual_check, "face_stability": face_stability, "caption_collision": caption_collision},
        }
        verdict_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        result.verdict_path = verdict_file
    except Exception as exc:  # pragma: no cover
        print(f"  [gate] ⚠ Could not write verdict.json: {exc}")

    # ------------------------------------------------------------------
    # 5. Print summary
    # ------------------------------------------------------------------
    try:
        print(f"  {result.summary_line()}")
        if warnings:
            print(f"  [gate] ⚠ Warnings: {', '.join(warnings)}")
        if blocking:
            print(f"  [gate] ✗ Blocking codes: {', '.join(blocking)}")
    except UnicodeEncodeError:
        ascii_summary = result.summary_line().replace("✓", "[PASS]").replace("✗", "[FAIL]")
        print(f"  {ascii_summary}")
        if warnings:
            print(f"  [gate] ! Warnings: {', '.join(warnings)}")
        if blocking:
            print(f"  [gate] X Blocking codes: {', '.join(blocking)}")

    return result
