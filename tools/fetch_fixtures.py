#!/usr/bin/env python3
"""
Fetch the golden-master clip corpus and record its SHA-256 in the manifest.

The fixtures are the real published artifacts, served from public GitHub Release
URLs. They are deliberately not tracked in git (~140 MB), so this script is the
only way to obtain them, and it verifies every byte.

    python tools/fetch_fixtures.py            # fetch anything missing or stale
    python tools/fetch_fixtures.py --verify   # verify only, download nothing
    python tools/fetch_fixtures.py --record   # (re)write sha256 into the manifest

`--record` is the only supported way to change a hash, and it must be a
deliberate act: a golden master that silently changes underneath the visual
regression suite makes every assertion in it meaningless.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"
MANIFEST_PATH = FIXTURE_DIR / "manifest.json"
CHUNK = 1 << 20


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest() -> Dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"manifest not found: {MANIFEST_PATH}")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def save_manifest(manifest: Dict[str, Any]) -> None:
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def download(url: str, destination: Path, allow_missing: bool = False) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "fetch_fixtures/1"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with temporary.open("wb") as handle:
                while True:
                    chunk = response.read(CHUNK)
                    if not chunk:
                        break
                    handle.write(chunk)
    except urllib.error.URLError as error:
        temporary.unlink(missing_ok=True)
        if allow_missing:
            print(f"[!] skipped unavailable fixture {url}: {error}")
            return False
        raise SystemExit(f"download failed for {url}: {error}") from error
    temporary.replace(destination)
    return True


def url_for(entry: Dict[str, Any], base: str) -> str:
    return f"{base}/{entry['release_tag']}/{entry['file']}"


def process(manifest: Dict[str, Any], verify_only: bool, record: bool, allow_missing: bool = False) -> int:
    base = manifest["release_base"]
    entries: List[Dict[str, Any]] = list(manifest.get("clips", []))
    failures: List[str] = []
    changed = False

    for entry in entries:
        name = entry["file"]
        target = FIXTURE_DIR / name
        expected: Optional[str] = entry.get("sha256")

        if target.exists():
            actual = sha256_of(target)
            if expected and actual != expected:
                failures.append(
                    f"{name}: on-disk sha256 {actual[:12]}... != manifest {expected[:12]}..."
                )
                continue
            if not expected:
                print(f"[=] {name}: present, recording sha256")
                entry["sha256"] = actual
                entry["bytes"] = target.stat().st_size
                changed = True
                continue
            print(f"[ok] {name}: sha256 verified")
            continue

        if verify_only:
            if not allow_missing:
                failures.append(f"{name}: missing (run without --verify to fetch)")
            else:
                print(f"[!] {name}: missing (ignored with --allow-missing)")
            continue

        url = url_for(entry, base)
        print(f"[>] {name} <- {url}")
        ok = download(url, target, allow_missing=allow_missing)
        if not ok:
            continue
        actual = sha256_of(target)
        size = target.stat().st_size
        if expected and actual != expected:
            failures.append(
                f"{name}: downloaded sha256 {actual[:12]}... != manifest {expected[:12]}..."
            )
            continue
        if not expected:
            entry["sha256"] = actual
            entry["bytes"] = size
            changed = True
            print(f"[+] {name}: fetched, sha256 recorded ({size:,} bytes)")
        else:
            print(f"[ok] {name}: fetched, sha256 verified ({size:,} bytes)")

    if record and changed:
        save_manifest(manifest)
        print(f"[*] manifest updated: {MANIFEST_PATH}")
    elif changed:
        print(
            "[!] sha256 values were discovered but not written. "
            "Re-run with --record to persist them."
        )

    if failures:
        print("\nFAILURES:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print(f"\n[+] {len(entries)} fixture(s) accounted for.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="verify only, download nothing")
    parser.add_argument(
        "--record", action="store_true", help="write newly discovered sha256 values to the manifest"
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="do not fail if remote fixtures are missing (e.g. pruned by release cleaner)",
    )
    args = parser.parse_args()
    return process(load_manifest(), args.verify, args.record, allow_missing=args.allow_missing)


if __name__ == "__main__":
    raise SystemExit(main())
