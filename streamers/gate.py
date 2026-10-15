"""V2 Streamer Publish Gate.

The single authority that decides whether a rendered clip may be published.

Why this module exists
----------------------
Pipeline V1 (main.py) has an AST-pinned publish gate wiring verdict.verify →
repair.render_with_repair → BufferPublisher. V2 was built alongside it and
inherited none of that: pipeline.py:318-325 reaches Buffer with nothing between
render_viral_clip and the dispatch call.

This module closes that gap. It composes the existing public verification
primitives (core.verification.verdict.verify) behind a single V2-scoped function
that pipeline.py calls instead of invoking Buffer directly.

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

import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from streamers.spec import (
    STREAMER_MAX_CLIP_DURATION_S,
    STREAMER_MIN_CLIP_DURATION_S,
    STREAMER_OUTPUT_HEIGHT,
    STREAMER_OUTPUT_WIDTH,
)

from core.config import MODELS_DIR as _MODELS_DIR

# YuNet weights live in the shared engine's models dir (env-overridable).
YUNET_MODEL_PATH = _MODELS_DIR / "face_detection_yunet.onnx"


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



def _face_stability_check(path) -> dict:
    """Sample frames and measure face/track instability in the rendered clip."""
    model_path = str(YUNET_MODEL_PATH)
    try:
        import cv2
        import numpy as np
    except Exception:
        return {
            "available": False,
            "face_presence_ratio": 0.0,
            "max_horizontal_jump_px": 0.0,
            "sampled_frames": 0,
            "warning": "face_stability_unavailable",
        }

    if not __import__("pathlib").Path(model_path).exists():
        return {
            "available": False,
            "face_presence_ratio": 0.0,
            "max_horizontal_jump_px": 0.0,
            "sampled_frames": 0,
            "warning": "face_stability_unavailable",
        }

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {
            "available": False,
            "face_presence_ratio": 0.0,
            "max_horizontal_jump_px": 0.0,
            "sampled_frames": 0,
            "warning": "face_stability_unavailable",
        }

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            return {
                "available": False,
                "face_presence_ratio": 0.0,
                "max_horizontal_jump_px": 0.0,
                "sampled_frames": 0,
                "warning": "face_stability_unavailable",
            }

        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1920)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1080)
        detector = cv2.FaceDetectorYN.create(model_path, "", (w, h), 0.5, 0.3, 5000)
        sample_count = min(12, total_frames)
        start_idx = max(0, int(total_frames * 0.15))
        end_idx = max(start_idx + 1, int(total_frames * 0.85))
        indices = []
        span = max(1, end_idx - start_idx)
        for i in range(sample_count):
            idx = start_idx + int(i * (span - 1) / max(sample_count - 1, 1))
            if start_idx <= idx < total_frames:
                indices.append(idx)
        indices = sorted(set(indices)) if indices else [max(0, total_frames // 2)]

        prior_cx = None
        max_jump = 0.0
        seen = 0
        valid = 0
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                continue
            if float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()) < 12.0:
                continue
            valid += 1
            _, res = detector.detect(frame)
            if res is None or len(res) == 0:
                continue
            # Choose largest face.
            largest = max(res, key=lambda r: float(r[2]) * float(r[3]))
            cx = float(largest[0]) + float(largest[2]) / 2.0
            seen += 1
            if prior_cx is not None:
                jump = abs(cx - prior_cx)
                if jump > max_jump:
                    max_jump = jump
            prior_cx = cx

        presence = (seen / valid) if valid else 0.0
        return {
            "available": True,
            "face_presence_ratio": round(presence, 3),
            "max_horizontal_jump_px": round(max_jump, 2),
            "sampled_frames": valid,
            "warning": None,
        }
    finally:
        cap.release()



def _caption_collision_check(path, model_path=None) -> dict:
    """Sample one frame and flag caption-like white stickers over large faces."""
    if model_path is None:
        model_path = str(YUNET_MODEL_PATH)
    try:
        import cv2
        import numpy as np
        from pathlib import Path as _Path
    except Exception:
        return {"available": False, "collision": False, "warning": "caption_check_unavailable"}

    if not _Path(model_path).exists():
        return {"available": False, "collision": False, "warning": "caption_check_unavailable"}

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"available": False, "collision": False, "warning": "caption_check_unavailable"}
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            return {"available": False, "collision": False, "warning": "caption_check_unavailable"}
        cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            return {"available": False, "collision": False, "warning": "caption_check_unavailable"}

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detector = cv2.FaceDetectorYN.create(model_path, "", (w, h), 0.5, 0.3, 5000)
        _, faces = detector.detect(frame)

        # White sticker candidates above heading/meme-box regions in the upper half.
        white_mask = (gray > 240)
        white_rows = white_mask.sum(axis=1)
        top_band_rows = [i for i in range(h // 2) if white_rows[i] > w * 0.05]
        if not top_band_rows:
            return {"available": True, "collision": False, "collision_ratio": 0.0}

        y0, y1 = min(top_band_rows), max(top_band_rows)
        white_cols = white_mask[y0:y1+1, :].sum(axis=0)
        x0 = int(np.argmax(white_cols > w * 0.01))
        x1 = w - int(np.argmax(white_cols[::-1] > w * 0.01))

        collision = False
        overlap_area = 0.0
        if faces is not None and len(faces) > 0:
            largest = max(faces, key=lambda r: float(r[2]) * float(r[3]))
            fx, fy, fw, fh = [float(v) for v in largest[:4]]
            ix0 = max(fx, x0); iy0 = max(fy, y0)
            ix1 = min(fx + fw, x1); iy1 = min(fy + fh, y1)
            if ix1 > ix0 and iy1 > iy0:
                collision = True
                overlap_area = (ix1 - ix0) * (iy1 - iy0)

        return {
            "available": True,
            "collision": collision,
            "collision_overlap_px": round(float(overlap_area), 2),
            "white_band": [int(y0), int(y1), int(x0), int(x1)],
        }
    finally:
        cap.release()


def _visual_black_margin_check(path) -> dict:
    """Sample rendered frames and reject shots with excessive black margins.

    V2 framing must be full-screen 9:16. Partial black borders can appear during
    Tier 3 off-screen fallback or a failed stacked/multi-person composition.
    """
    try:
        import cv2
        import numpy as np
    except Exception:
        return {
            "available": False,
            "passed": True,
            "worst_margin_pct": 0.0,
            "worst_side": "none",
            "sampled_frames": 0,
            "warning": "visual_check_unavailable",
        }

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {
            "available": False,
            "passed": True,
            "worst_margin_pct": 0.0,
            "worst_side": "none",
            "sampled_frames": 0,
            "warning": "visual_check_unavailable",
        }

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            return {
                "available": False,
                "passed": True,
                "worst_margin_pct": 0.0,
                "worst_side": "none",
                "sampled_frames": 0,
                "warning": "visual_check_unavailable",
            }

        margin_px = 40
        sample_count = min(12, total_frames)
        indices = [0] if sample_count <= 1 else sorted(set(
            int(i * (total_frames - 1) / (sample_count - 1))
            for i in range(sample_count)
        ))

        worst_pct = 0.0
        worst_side = "none"
        valid = 0

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                continue
            valid += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]
            margin = min(margin_px, max(8, h // 30), max(8, w // 30))

            top = float((gray[:margin, :] < 16).mean())
            bottom = float((gray[h - margin:, :] < 16).mean())
            left = float((gray[:, :margin] < 16).mean())
            right = float((gray[:, w - margin:] < 16).mean())

            for side, value in (("top", top), ("bottom", bottom), ("left", left), ("right", right)):
                if value > worst_pct:
                    worst_pct = value
                    worst_side = side

        if valid == 0:
            return {
                "available": False,
                "passed": True,
                "worst_margin_pct": 0.0,
                "worst_side": "none",
                "sampled_frames": 0,
                "warning": "visual_check_unavailable",
            }

        worst_pct *= 100.0
        return {
            "available": True,
            "passed": worst_pct < 10.0,
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
    visual_check = {
        'available': False,
        'passed': True,
        'worst_margin_pct': 0.0,
        'worst_side': 'none',
        'sampled_frames': 0,
    }

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
    if face_stability.get("available") and face_stability.get("max_horizontal_jump_px", 0) > 150.0:
        raw_verdict.add(
            verdict_mod.Finding(
                code="v2_excessive_face_jump",
                severity="error",
                message=(
                    f"V2 face jump {face_stability['max_horizontal_jump_px']:.1f}px exceeds 150px"
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
        raw_verdict.warnings.append('v2_visual_check_unavailable')

    # ------------------------------------------------------------------
    # 3. Collect findings
    # ------------------------------------------------------------------
    blocking = [f.code for f in raw_verdict.findings if f.blocks]
    warnings = [f.code for f in raw_verdict.findings if f.severity == "warning"]
    if 'v2_visual_check_unavailable' in getattr(raw_verdict, 'warnings', []):
        warnings.append('v2_visual_check_unavailable')

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
    print(f"  {result.summary_line()}")
    if warnings:
        print(f"  [gate] ⚠ Warnings: {', '.join(warnings)}")
    if blocking:
        print(f"  [gate] ✗ Blocking codes: {', '.join(blocking)}")

    return result
