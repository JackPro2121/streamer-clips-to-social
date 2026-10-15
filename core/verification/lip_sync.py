"""SyncNet lip-sync measurement for rendered clips.

The renderer-level A/V regression (``tests/test_av_sync_render.py``) proves the
pipeline does not *introduce* an offset. This module measures whether the
finished clip *is* in sync, end to end, using the standard SyncNet metrics:

* **LSE-C** -- maximum confidence across face tracks (higher is better).
* **LSE-D** -- minimum L1 distance across face tracks (lower is better).

Implementation: the MIT-licensed ``syncnet-python`` package with the official
Oxford VGG SyncNet and S3FD checkpoints fetched by ``core/tools/fetch_models.py``
(SHA-256 pinned). It runs in the ML tier (``requirements-ml.txt``); the render
job does not install torch.

Thresholds
----------
``PROVISIONAL_*`` values come from the wrapper package's published bands. They
are explicitly provisional: the published bands were calibrated on talking-face
benchmarks, not on loudnorm-processed vertical podcast crops, so
``core/tools/syncnet_calibrate.py`` measures the golden-master corpus and the
constants are repinned from that evidence. Until then the verdict reports the
metric at ``warning`` severity instead of blocking.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CODE_LIP_SYNC = "lip_sync"

# Provisional bands (see module docstring). Calibration evidence will replace
# these; they are named so the replacement is a one-line change.
PROVISIONAL_GOOD_C = 3.5
PROVISIONAL_GOOD_D = 7.0
PROVISIONAL_FAIR_C = 2.0
PROVISIONAL_FAIR_D = 10.0

# A measurement that cannot run is reported as such, never as a pass. Mirrors
# the caption gate's fail-closed behaviour.
MIN_TRACK_SECONDS = 1.0


@dataclass
class LipSyncReport:
    ok: bool = True
    reason: str = ""
    measurable: bool = False
    has_face: bool = False
    lse_c: float = 0.0
    lse_d: float = 0.0
    quality: str = "UNKNOWN"
    tracks: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "measurable": self.measurable,
            "has_face": self.has_face,
            "lse_c": round(self.lse_c, 4),
            "lse_d": round(self.lse_d, 4),
            "quality": self.quality,
            "tracks": self.tracks,
        }


def quality_for(lse_c: float, lse_d: float) -> str:
    """Map the two metrics onto a band. Pure, so the banding is unit-testable."""
    if lse_c > PROVISIONAL_GOOD_C and lse_d < PROVISIONAL_GOOD_D:
        return "GOOD"
    if lse_c > PROVISIONAL_FAIR_C and lse_d < PROVISIONAL_FAIR_D:
        return "FAIR"
    return "POOR"


def models_dir() -> Path:
    override = os.getenv("LIP_SYNC_MODELS_DIR")
    if override:
        return Path(override)
    from core.config import MODELS_DIR

    return MODELS_DIR


_pipeline_cache: Optional[Any] = None


def _load_pipeline(models: Path):
    global _pipeline_cache
    if _pipeline_cache is not None:
        return _pipeline_cache, ""
    try:
        from syncnet_python.syncnet_pipeline import PipelineConfig, SyncNetPipeline
    except ImportError as error:
        return None, f"ML tier not installed ({error}); pip install -r requirements-ml.txt"
    s3fd = models / "sfd_face.pth"
    syncnet = models / "syncnet_v2.model"
    missing = [name for name in (s3fd, syncnet) if not name.exists()]
    if missing:
        return None, (
            "checkpoints missing: "
            + ", ".join(path.name for path in missing)
            + "; run python core/tools/fetch_models.py --fetch all"
        )
    config = PipelineConfig(
        s3fd_weights=str(s3fd),
        syncnet_weights=str(syncnet),
        frame_rate=25,
        batch_size=10,
        min_track=10,
    )
    _pipeline_cache = SyncNetPipeline(config, device="cpu")
    return _pipeline_cache, ""


def analyse(path: Path, models: Optional[Path] = None) -> LipSyncReport:
    """Measure lip-sync for one rendered clip. Never raises."""
    if not path.exists():
        return LipSyncReport(ok=False, reason=f"file not found: {path}")

    pipeline, reason = _load_pipeline(models or models_dir())
    if pipeline is None:
        return LipSyncReport(ok=False, reason=reason)

    try:
        offsets, confidences, distances, max_conf, min_dist, _faces, has_face = pipeline.inference(
            str(path)
        )
    except Exception as error:  # noqa: BLE001 - a verifier reports, never crashes
        return LipSyncReport(ok=False, reason=f"SyncNet inference failed: {type(error).__name__}: {error}")

    if not has_face or not offsets:
        return LipSyncReport(
            ok=True,
            measurable=False,
            has_face=bool(has_face),
            reason="no face track long enough to measure",
        )

    report = LipSyncReport(
        ok=True,
        measurable=True,
        has_face=True,
        lse_c=float(max_conf),
        lse_d=float(min_dist),
        quality=quality_for(float(max_conf), float(min_dist)),
        tracks=[
            {
                "offset_frames": int(offset),
                "offset_ms": round(int(offset) / 25.0 * 1000.0, 1),
                "confidence": round(float(conf), 4),
                "distance": round(float(dist), 4),
            }
            for offset, conf, dist in zip(offsets, confidences, distances)
        ],
    )
    return report
