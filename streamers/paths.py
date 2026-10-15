"""Safe filesystem naming for V2 outputs.

Why this module exists
----------------------
Clip titles are **untrusted input**. They originate from an LLM (Gemini/Groq/
Ollama) reasoning over a third-party stream transcript, and the pipeline joins
them straight onto a directory to build clip, thumbnail, subtitle and sidecar
filenames. Before this module, the only sanitisation was
``title.replace(" ", "_")``, which leaves ``/ \\ : | " * ? < >`` intact:

* ``"RANKED #1 / Kai vs Speed"`` produced ``clips/streamers/clip_2_RANKED_#1_/_...``
  — a clip written *outside* ``clips/streamers/`` (both the renderer and the
  thumbnail generator call ``mkdir(parents=True)``).
* On Windows, ``|`` raised ``OSError`` and ``:`` silently wrote an NTFS
  alternate data stream, uploading a 0-byte asset.

Contract
--------
* :func:`slugify_title` — allowlist ``[A-Za-z0-9_-]``, deterministic, never
  empty, never longer than ``max_len``.
* :func:`safe_output_name` — ``<prefix>_<slug>_<fingerprint><suffix>``; the
  fingerprint keeps two clips with the same title from colliding on disk.
* :func:`assert_within` — hard containment guard so a naming mistake can never
  write outside the intended directory.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

# Anything outside this set becomes a single underscore.
_UNSAFE_RUN = re.compile(r"[^A-Za-z0-9_-]+")

# Default slug budget. Short enough to keep paths readable, long enough to stay
# recognisable in the artifact listing.
DEFAULT_SLUG_LEN = 40

# Fallback slug when a title reduces to nothing (empty, emoji-only, CJK-only).
_FALLBACK_SLUG = "clip"


def slugify_title(title: str, *, max_len: int = DEFAULT_SLUG_LEN) -> str:
    """Reduce an untrusted title to a safe, deterministic filename component.

    Non-ASCII characters are dropped rather than transliterated so the result is
    byte-identical on every platform and filesystem.
    """
    ascii_only = (title or "").encode("ascii", "ignore").decode("ascii")
    collapsed = _UNSAFE_RUN.sub("_", ascii_only).strip("_-")
    if not collapsed:
        return _FALLBACK_SLUG
    if len(collapsed) > max_len:
        collapsed = collapsed[:max_len].rstrip("_-")
    return collapsed or _FALLBACK_SLUG


def title_fingerprint(title: str, *, length: int = 6) -> str:
    """Short, stable hash of the raw title — disambiguates sibling filenames."""
    digest = hashlib.sha256((title or "").encode("utf-8")).hexdigest()
    return digest[:length]


def safe_output_name(
    prefix: str,
    untrusted: str,
    suffix: str,
    *,
    max_len: int = DEFAULT_SLUG_LEN,
) -> str:
    """Build ``<prefix>_<slug>_<fingerprint><suffix>`` from an untrusted title."""
    return f"{prefix}_{slugify_title(untrusted, max_len=max_len)}_{title_fingerprint(untrusted)}{suffix}"


def assert_within(base: Path, candidate: Path) -> Path:
    """Return ``candidate`` unchanged, or raise if it escapes ``base``.

    Last line of defence: even if a caller forgets to slugify, an output can
    never be written outside the directory the pipeline owns.
    """
    resolved_base = base.resolve()
    resolved = candidate.resolve()
    if resolved != resolved_base and resolved_base not in resolved.parents:
        raise ValueError(f"refusing to write outside {resolved_base}: {resolved}")
    return candidate
