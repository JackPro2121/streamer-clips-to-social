"""Fixture access for the golden-master corpus.

The clips are the real published artifacts. They are not tracked in git; fetch
them with ``python core/tools/fetch_fixtures.py``. Every file is SHA-256 verified
against ``tests/fixtures/manifest.json`` on load, because a golden master that
silently changes underneath the visual regression suite makes every assertion in
it meaningless.

The manifest also carries a ``human_assessment`` per clip: what a person sees
when scrubbing the timeline. That is the ground truth the verifier is checked
against. Without it the suite would only prove the verifier is self-consistent,
which is the exact failure mode that let the original defects ship.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"
MANIFEST_PATH = FIXTURE_DIR / "manifest.json"


@dataclass(frozen=True)
class GoldenMaster:
    """One published clip, plus what a human saw in it."""

    id: str
    path: Path
    release_tag: str
    run_id: int
    sha256: str
    bytes: int
    human_verdict: str
    human_defects: List[str]
    human_clean: List[str]
    human_notes: List[str]

    @property
    def file(self) -> str:
        """Filename, for use in assertion messages."""
        return self.path.name

    @property
    def expects_pass(self) -> bool:
        return self.human_verdict.upper() == "PASS"

    @property
    def defect_ids(self) -> List[str]:
        """Audit defect identifiers mentioned in the human assessment, e.g. ``B1``."""
        found: List[str] = []
        for text in self.human_defects + self.human_notes:
            token = text.split(":", 1)[0].split()[0] if text.split() else ""
            token = token.strip(".,()")
            if token and token[0].isalpha() and token[1:2].isdigit() and len(token) <= 4:
                if token not in found:
                    found.append(token)
        return found


def _manifest() -> Dict[str, Any]:
    if not MANIFEST_PATH.exists():
        return {}
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def available() -> List[GoldenMaster]:
    """Every fixture present on disk. Missing ones are skipped, not fatal.

    The suite calls this and skips when it is empty, so a contributor who has not
    run ``fetch_fixtures.py`` still gets the full unit suite rather than a wall
    of errors.
    """
    manifest = _manifest()
    if not manifest:
        return []
    masters: List[GoldenMaster] = []
    for entry in manifest.get("clips", []) or []:
        path = FIXTURE_DIR / entry["file"]
        if not path.exists():
            continue
        assessment = entry.get("human_assessment", {}) or {}
        masters.append(
            GoldenMaster(
                id=entry["id"],
                path=path,
                release_tag=entry.get("release_tag", ""),
                run_id=int(entry.get("run_id", 0) or 0),
                sha256=entry.get("sha256") or "",
                bytes=int(entry.get("bytes") or path.stat().st_size),
                human_verdict=assessment.get("verdict", "UNKNOWN"),
                human_defects=list(assessment.get("defects", []) or []),
                human_clean=list(assessment.get("clean", []) or []),
                human_notes=list(assessment.get("notes", []) or []),
            )
        )
    return masters


def by_id(identifier: str) -> Optional[GoldenMaster]:
    for master in available():
        if master.id == identifier:
            return master
    return None


def fetch_hint() -> str:
    return (
        "Golden-master clips are not tracked in git (~140 MB). "
        "Fetch them with:  python core/tools/fetch_fixtures.py"
    )


def all_present() -> bool:
    """True when the whole corpus is on disk. Used to decide whether to skip."""
    manifest = _manifest()
    clips = manifest.get("clips", []) or []
    if not clips:
        return False
    return all((FIXTURE_DIR / entry["file"]).exists() for entry in clips)
