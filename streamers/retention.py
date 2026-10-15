"""Bounded retention for V2 run artifacts.

Why this module exists
----------------------
The publish job runs unattended on a schedule. Before this module nothing in V2
ever deleted anything, so every run permanently added:

* ``downloads/streamer_clip_<n>.mp4``   — Whisper/FFmpeg working copy
* ``subtitles/streamer_clip_<n>_*.ass`` — subtitle stylesheet
* ``clips/streamers/*_preview.mp4``     — derived 540x960 preview
* ``clips/streamers/*_edl.json``        — edit-decision sidecar
* ``data/runs/<run_id>/``               — verdict.json + run_manifest.json + report

with no ceiling. On an ephemeral CI runner that is mostly disk pressure during a
run; on a self-hosted or cron checkout it is an unmonitored grow-forever path.

Deliberately retained
---------------------
``clips/streamers/*.mp4`` (rendered clips) and ``*_thumb.jpg`` are the
*deliverables* — they are uploaded to GitHub Releases and reviewed by a human —
so they are never deleted here. Use the release-cleanup workflow for those.

Safety rules
------------
* Only paths strictly inside the directories handed to these functions are ever
  touched; anything that resolves outside its root is ignored.
* Symlinks are skipped, so a planted link cannot redirect a delete.
* The run currently executing is always preserved.
* Deletions are logged; failures are logged and never raise.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

# How many most-recent run directories to keep under data/runs/.
DEFAULT_KEEP_RUNS: int = 20
# Delete run directories older than this regardless of the count above.
DEFAULT_RUN_AGE_DAYS: float = 30.0
# Working copies are pure scratch — a much shorter window is enough.
DEFAULT_WORKSPACE_AGE_DAYS: float = 7.0

_SECONDS_PER_DAY = 86400.0


def _within(base: Path, candidate: Path) -> bool:
    """True if ``candidate`` resolves to something strictly inside ``base``."""
    try:
        base_resolved = base.resolve()
        candidate_resolved = candidate.resolve()
    except OSError:
        return False
    if candidate_resolved == base_resolved:
        return False
    return base_resolved in candidate_resolved.parents


def _age_days(path: Path, now: float) -> float:
    try:
        return max(0.0, (now - path.stat().st_mtime) / _SECONDS_PER_DAY)
    except OSError:
        return 0.0


def _newest_age_days(directory: Path, now: float) -> float:
    """Age of the newest *entry* inside ``directory``.

    A directory's own mtime only moves when an entry is added or removed, so it
    reports "something was written here recently" even after the contents have
    been back-dated or archived, and a run whose directory was re-created looks
    newer than it is. Ranking on the newest child is the honest "last activity"
    measure; an empty directory falls back to its own mtime.
    """
    newest = 0.0
    saw_child = False
    try:
        for child in directory.iterdir():
            saw_child = True
            try:
                newest = max(newest, child.stat().st_mtime)
            except OSError:
                continue
    except OSError:
        pass

    if not saw_child:
        try:
            newest = directory.stat().st_mtime
        except OSError:
            newest = 0.0

    if newest <= 0.0:
        return 0.0
    return max(0.0, (now - newest) / _SECONDS_PER_DAY)


def prune_run_dirs(
    runs_root: Path,
    *,
    current_run_id: Optional[str] = None,
    keep: int = DEFAULT_KEEP_RUNS,
    max_age_days: float = DEFAULT_RUN_AGE_DAYS,
    dry_run: bool = False,
) -> List[Path]:
    """Delete old ``<runs_root>/<run_id>`` directories.

    Retains the ``keep`` most recently modified directories, always retains the
    run that is currently executing, and additionally expires anything older than
    ``max_age_days`` (pass ``0`` to disable the age rule).
    """
    runs_root = Path(runs_root)
    if not runs_root.is_dir():
        return []

    now = time.time()
    candidates: List[Path] = []
    for child in runs_root.iterdir():
        if not child.is_dir() or child.is_symlink():
            continue
        if not _within(runs_root, child):
            continue
        if current_run_id and child.name == current_run_id:
            continue
        candidates.append(child)

    # Newest first, so `position` is a recency rank.
    candidates.sort(key=lambda p: _newest_age_days(p, now))

    removed: List[Path] = []
    for position, path in enumerate(candidates):
        age = _newest_age_days(path, now)
        over_count = position >= keep
        over_age = max_age_days > 0 and age > max_age_days
        if not (over_count or over_age):
            continue
        if dry_run:
            removed.append(path)
            continue
        try:
            shutil.rmtree(path)
            removed.append(path)
        except OSError as exc:
            print(f"[!] Retention: could not remove run dir {path}: {exc}")
    return removed


def prune_workspace_files(
    directory: Path,
    *,
    patterns: Sequence[str],
    current_run_id: Optional[str] = None,
    max_age_days: float = DEFAULT_WORKSPACE_AGE_DAYS,
    dry_run: bool = False,
) -> List[Path]:
    """Delete scratch files matching ``patterns`` inside ``directory``.

    Only direct children are considered, symlinks are skipped, and a path that
    resolves outside ``directory`` is left alone.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []

    now = time.time()
    removed: List[Path] = []
    seen: Set[Path] = set()

    for pattern in patterns:
        for path in directory.glob(pattern):
            if path in seen:
                continue
            seen.add(path)
            if not path.is_file() or path.is_symlink():
                continue
            if not _within(directory, path):
                continue
            if current_run_id and current_run_id in path.name:
                continue
            if max_age_days > 0 and _age_days(path, now) <= max_age_days:
                continue
            if dry_run:
                removed.append(path)
                continue
            try:
                path.unlink()
                removed.append(path)
            except OSError as exc:
                print(f"[!] Retention: could not remove {path}: {exc}")
    return removed


def run_retention_sweep(
    *,
    runs_root: Path,
    downloads_dir: Optional[Path] = None,
    subtitles_dir: Optional[Path] = None,
    clips_dir: Optional[Path] = None,
    current_run_id: Optional[str] = None,
    keep_runs: int = DEFAULT_KEEP_RUNS,
    workspace_age_days: float = DEFAULT_WORKSPACE_AGE_DAYS,
    dry_run: bool = False,
) -> Dict[str, List[Path]]:
    """Apply the V2 retention policy and return everything that was removed."""
    removed: Dict[str, List[Path]] = {"run_dirs": [], "workspace_files": []}

    removed["run_dirs"] = prune_run_dirs(
        runs_root, current_run_id=current_run_id, keep=keep_runs, dry_run=dry_run
    )

    if downloads_dir is not None:
        removed["workspace_files"] += prune_workspace_files(
            downloads_dir,
            patterns=["streamer_clip_*.mp4", "*.part", "*.ytdl"],
            current_run_id=current_run_id,
            max_age_days=workspace_age_days,
            dry_run=dry_run,
        )

    if subtitles_dir is not None:
        removed["workspace_files"] += prune_workspace_files(
            subtitles_dir,
            patterns=["streamer_clip_*.ass"],
            current_run_id=current_run_id,
            max_age_days=workspace_age_days,
            dry_run=dry_run,
        )

    if clips_dir is not None:
        # Derived render artifacts only. Rendered clips (*.mp4) and thumbnails
        # (*_thumb.jpg) are deliverables and are never matched here.
        removed["workspace_files"] += prune_workspace_files(
            clips_dir,
            patterns=["*_preview.mp4", "*_edl.json"],
            current_run_id=current_run_id,
            max_age_days=workspace_age_days,
            dry_run=dry_run,
        )

    return removed
