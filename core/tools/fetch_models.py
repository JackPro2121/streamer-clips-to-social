#!/usr/bin/env python3
"""
Fetch the pretrained models the deep-verification tier needs.

Every URL below is the model's **official publisher**:

* SyncNet + S3FD: Oxford VGG, the canonical source referenced by
  ``joonson/syncnet_python``'s own ``download_model.sh``.
* TransNetV2: exported from the proven ``transnetv2-pytorch`` model by
  ``core/tools/export_transnetv2_onnx.py`` (window- and video-level parity proof)
  and hosted on this project's immutable ``golden-masters-v1`` release, so the
  torch-free render job can run it through onnxruntime.

The SHA-256 and byte count of each file were recorded on first download from
that official source and are pinned here, so a later fetch that returns a
different payload fails loudly instead of silently poisoning every metric that
depends on the model.

    python core/tools/fetch_models.py --list
    python core/tools/fetch_models.py --fetch all
    python core/tools/fetch_models.py --verify
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

# ``tools/`` lives inside the clips-core root: ``core/`` when the engine is
# mounted as a submodule in a pipeline repo, and the repo root when clips-core
# is checked out on its own. ``parents[1]`` is that root in both layouts.
_ENGINE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = _ENGINE_ROOT / "models"
CHUNK = 1 << 20

VGG = "https://www.robots.ox.ac.uk/~vgg/software/lipsync/data"
RELEASE_BASE = "https://github.com/JackPro2121/podcasts-clips-to-social/releases/download"

MODELS: Dict[str, Dict[str, Any]] = {
    "syncnet_v2.model": {
        "url": f"{VGG}/syncnet_v2.model",
        "sha256": "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442",
        "bytes": 54573114,
        "purpose": "SyncNet audio-visual sync + active speaker (MIT package, official checkpoint)",
    },
    "sfd_face.pth": {
        "url": f"{VGG}/sfd_face.pth",
        "sha256": "d54a87c2b7543b64729c9a25eafd188da15fd3f6e02f0ecec76ae1b30d86c491",
        "bytes": 89844381,
        "purpose": "S3FD face detector required by the SyncNet pipeline",
    },
    "transnetv2.onnx": {
        "url": f"{RELEASE_BASE}/golden-masters-v1/transnetv2.onnx",
        "sha256": "e80ce1264ce71d96b21ae641c8b593950599d8c25a7320e810efe06049b65295",
        "bytes": 31996324,
        "purpose": (
            "TransNetV2 shot-boundary detection, exported from transnetv2-pytorch by "
            "core/tools/export_transnetv2_onnx.py with window- and video-level parity proof"
        ),
    },
}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, allow_missing: bool = False) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "fetch_models/1"})
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            with temporary.open("wb") as handle:
                while True:
                    chunk = response.read(CHUNK)
                    if not chunk:
                        break
                    handle.write(chunk)
    except urllib.error.URLError as error:
        temporary.unlink(missing_ok=True)
        if allow_missing:
            print(f"[!] skipped unavailable model {url}: {error}")
            return False
        raise SystemExit(f"download failed for {url}: {error}") from error
    temporary.replace(destination)
    return True


def process(models_dir: Path, fetch: List[str], verify_only: bool, allow_missing: bool) -> int:
    """Fetch or verify models.

    Selection semantics, because a fetch of one model must not fail because an
    unrelated one is absent (the render job fetches only ``transnetv2.onnx``;
    the SyncNet checkpoints belong to the ML tier):

    * ``--fetch all``            -> every registry entry is required
    * ``--fetch name [name...]`` -> only the named entries are required; others
                                    are verified if present and skipped if not
    * ``--verify`` (no fetch)    -> every registry entry is required
    """
    failures: List[str] = []
    fetch_all = "all" in fetch
    required = (
        set(MODELS) if (fetch_all or not fetch) else {name for name in fetch if name in MODELS}
    )

    for name, meta in MODELS.items():
        target = models_dir / name
        expected = meta["sha256"]
        needed = name in required

        if target.exists():
            actual = sha256_of(target)
            size = target.stat().st_size
            if actual != expected or size != meta["bytes"]:
                failures.append(
                    f"{name}: on-disk {size} bytes sha {actual[:12]}... != pinned "
                    f"{meta['bytes']} bytes sha {expected[:12]}..."
                )
            else:
                print(f"[ok] {name}: sha256 verified ({size:,} bytes)")
            continue

        if not needed:
            continue

        if verify_only:
            if not allow_missing:
                failures.append(f"{name}: missing (run --fetch {name})")
            else:
                print(f"[!] {name}: missing (ignored with --allow-missing)")
            continue

        print(f"[>] {name} <- {meta['url']}")
        if not download(meta["url"], target, allow_missing=allow_missing):
            continue
        actual = sha256_of(target)
        size = target.stat().st_size
        if actual != expected or size != meta["bytes"]:
            failures.append(
                f"{name}: downloaded {size} bytes sha {actual[:12]}... != pinned "
                f"{meta['bytes']} bytes sha {expected[:12]}..."
            )
            continue
        print(f"[+] {name}: fetched, sha256 verified ({size:,} bytes)")

    if failures:
        print("\nFAILURES:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print(f"\n[+] {len(MODELS)} model(s) accounted for in {models_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="show the model registry")
    parser.add_argument(
        "--fetch",
        nargs="+",
        default=[],
        metavar="NAME",
        help="model name(s) to download, or 'all'",
    )
    parser.add_argument("--verify", action="store_true", help="verify only")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    if args.list:
        for name, meta in MODELS.items():
            print(f"{name}\n    url:     {meta['url']}\n    bytes:   {meta['bytes']:,}\n    purpose: {meta['purpose']}")
        return 0

    return process(args.models_dir, args.fetch, args.verify, args.allow_missing)


if __name__ == "__main__":
    raise SystemExit(main())
