#!/usr/bin/env python3
"""
Calibrate the SyncNet lip-sync gate on the golden-master corpus.

Run on CI (the ML tier needs torch). Produces, per clip, the two standard
audio-visual sync metrics:

* **LSE-C** -- max confidence across face tracks. Higher is better.
* **LSE-D** -- min L1 distance across face tracks. Lower is better.

The published "typical" thresholds (GOOD > 3.5 / < 7.0) were calibrated on
talking-face benchmarks, not on loudnorm-processed vertical podcast crops. So
the gate must be calibrated on this project's own corpus: clean PASS fixtures
vs the pre-fix clips that were reported as out of sync. The table this prints is
the evidence for wherever the threshold ends up.

    python core/tools/syncnet_calibrate.py --max-seconds 20 --output calibration.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ``sys.path`` needs the pipeline repo root for ``core.verification`` to
# import; the model registry lives in the engine root (``core/``).
_ENGINE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = _ENGINE_ROOT.parent if _ENGINE_ROOT.name == "core" else _ENGINE_ROOT
sys.path.insert(0, str(REPO_ROOT))

from core.verification import fixtures  # noqa: E402

DEFAULT_MODELS_DIR = _ENGINE_ROOT / "models"


def _trim(video: Path, seconds: float) -> Path:
    handle = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    handle.close()
    target = Path(handle.name)
    command = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(video), "-t", f"{seconds:.2f}",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
        "-c:a", "aac", str(target),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=600)
    if result.returncode != 0 or not target.exists():
        raise RuntimeError(f"trim failed: {result.stderr[:300]}")
    return target


def calibrate(ids: Optional[List[str]], max_seconds: float, models_dir: Path) -> List[Dict[str, Any]]:
    from syncnet_python.syncnet_pipeline import PipelineConfig, SyncNetPipeline

    masters = fixtures.available()
    if ids:
        masters = [master for master in masters if master.id in ids]
    if not masters:
        raise SystemExit("no fixtures available; run core/tools/fetch_fixtures.py first")

    config = PipelineConfig(
        s3fd_weights=str(models_dir / "sfd_face.pth"),
        syncnet_weights=str(models_dir / "syncnet_v2.model"),
        frame_rate=25,
        batch_size=10,
        min_track=10,
    )
    pipeline = SyncNetPipeline(config, device="cpu")

    rows: List[Dict[str, Any]] = []
    for master in masters:
        video = _trim(master.path, max_seconds) if max_seconds > 0 else master.path
        started = time.time()
        row: Dict[str, Any] = {
            "id": master.id,
            "file": master.file,
            "human_verdict": master.human_verdict,
            "human_defects": master.human_defects,
        }
        try:
            offsets, confs, dists, max_conf, min_dist, _faces, has_face = pipeline.inference(str(video))
            tracks = [
                {
                    "offset_frames": int(offset),
                    "offset_ms": round(int(offset) / 25.0 * 1000.0, 1),
                    "confidence": round(float(conf), 4),
                    "distance": round(float(dist), 4),
                }
                for offset, conf, dist in zip(offsets, confs, dists)
            ]
            row.update(
                {
                    "ok": bool(has_face),
                    "has_face": bool(has_face),
                    "track_count": len(tracks),
                    "lse_c": round(float(max_conf), 4),
                    "lse_d": round(float(min_dist), 4),
                    "tracks": tracks,
                }
            )
        except Exception as error:  # noqa: BLE001 - calibration reports, never aborts
            row.update({"ok": False, "error": f"{type(error).__name__}: {error}"})
        row["seconds"] = round(time.time() - started, 1)
        if max_seconds > 0 and video != master.path:
            video.unlink(missing_ok=True)
        rows.append(row)
        print(f"[{row['id']}] lse_c={row.get('lse_c')} lse_d={row.get('lse_d')} "
              f"tracks={row.get('track_count')} verdict={row['human_verdict']} ({row['seconds']}s)", flush=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", nargs="*", default=None)
    parser.add_argument("--max-seconds", type=float, default=20.0, help="0 = full clips")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--output", type=Path, default=Path("syncnet_calibration.json"))
    args = parser.parse_args()

    rows = calibrate(args.ids, args.max_seconds, args.models_dir)
    args.output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    print("\n### SyncNet calibration")
    print()
    print("| id | human | has_face | tracks | LSE-C (higher=better) | LSE-D (lower=better) | offsets (ms) |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for row in sorted(rows, key=lambda item: item["id"]):
        offsets = ", ".join(str(track["offset_ms"]) for track in row.get("tracks", [])) or "-"
        print(
            "| `{}` | {} | {} | {} | {} | {} | {} |".format(
                row["id"],
                row["human_verdict"],
                row.get("has_face"),
                row.get("track_count", 0),
                row.get("lse_c", "-"),
                row.get("lse_d", "-"),
                offsets,
            )
        )
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
