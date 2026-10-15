"""Streamer Pipeline V2 — Creative Specification Overlay.

Single source of truth for every numeric constraint in the V2 streamer pipeline.
All V2 bounds are equal to or tighter than the global creative_spec (V1) values.
``validate_spec()`` runs at import time and makes it impossible to accidentally
widen them.

Mapping to defect register
--------------------------
D26 — Four conflicting duration bounds resolved here.
D5  — Endpoint clamping must use STREAMER_MAX_CLIP_DURATION_S, not config defaults.
D1  — gate.py imports this for publish-gate enforcement.
"""

from __future__ import annotations

import os
from typing import List

from core import creative_spec

# ---------------------------------------------------------------------------
# Clip duration (governs all four previously-conflicting bounds)
# ---------------------------------------------------------------------------
# The workflow sets MAX_CLIP_DURATION=45, config.MAX_CLIP_DURATION defaults to
# 140, creative_spec caps at 55s, and viral_detector docstring says 15-45s.
# V2 normalises to 30-45s everywhere. This is the one place to change it.
STREAMER_MIN_CLIP_DURATION_S: float = float(
    os.getenv("STREAMER_MIN_CLIP_DURATION", "30.0")
)
STREAMER_MAX_CLIP_DURATION_S: float = float(
    os.getenv("STREAMER_MAX_CLIP_DURATION", "45.0")
)
# How far endpoint snapping may push the boundary backward (natural sentence)
STREAMER_ENDPOINT_TRIM_MAX_S: float = 2.0
# Frame-boundary quantisation slack (29.97 fps ≈ 33 ms per frame)
STREAMER_DURATION_QUANTIZATION_MARGIN_S: float = 0.1

# ---------------------------------------------------------------------------
# Hashtags (AGENTS.md commandment 3)
# ---------------------------------------------------------------------------
STREAMER_HASHTAG_MIN: int = 2
STREAMER_HASHTAG_MAX: int = 4

# ---------------------------------------------------------------------------
# Face tracking / 3-Tier State Machine (AGENTS.md V2 §SFace)
# ---------------------------------------------------------------------------
# Minimum seconds a face must be absent before TIER-2 → TIER-3 transition
STREAMER_TIER2_HOLD_S: float = 1.5
# Minimum seconds a face track must persist before we treat it as a real person
# (prevents a single streamer walking across the room from triggering split-screen)
STREAMER_IDENTITY_MIN_DWELL_S: float = 2.0
# SFace cosine similarity floor. Uncalibrated; can be overridden per clip.
STREAMER_IDENTITY_SIMILARITY_FLOOR: float = 0.38

# ---------------------------------------------------------------------------
# Layout / geometry
# ---------------------------------------------------------------------------
# Output canvas
STREAMER_OUTPUT_WIDTH: int = 1080
STREAMER_OUTPUT_HEIGHT: int = 1920

# Safe-zone pixel offsets (absolute on the 1080x1920 canvas).
# Subtitles and hook box must stay inside these margins.
STREAMER_SAFE_TOP_PX: int = 240    # search bar / audio icon
STREAMER_SAFE_BOTTOM_PX: int = 380  # TikTok handle / captions / disc
STREAMER_SAFE_RIGHT_PX: int = 120   # like / comment / share buttons

# Maximum pane width for disjoint 2-person split-screen on a 1920-wide source.
# Derived: floor(source_w / 2) rounded down to even → 960 px.
# Pane height for 9:8 aspect: 960 * 8 / 9 = 853.3 → 852 px (even).
STREAMER_SPLIT_MAX_PANE_W: int = 960
STREAMER_SPLIT_PANE_ASPECT_W: int = 9
STREAMER_SPLIT_PANE_ASPECT_H: int = 8

# ---------------------------------------------------------------------------
# Audio / punch-zoom
# ---------------------------------------------------------------------------
# Audio RMS threshold (dB above median) that triggers a punch-zoom
STREAMER_SCREAM_DELTA_DB: float = 6.5
# Maximum zoom factor applied during a scream peak (1.0 = no zoom)
STREAMER_PUNCH_ZOOM_FACTOR: float = 1.20
# Maximum number of punch-zoom segments per clip (keeps filtergraph sane in CI)
STREAMER_PUNCH_ZOOM_MAX_SEGMENTS: int = 3

# ---------------------------------------------------------------------------
# Discovery / history
# ---------------------------------------------------------------------------
# Maximum candidates to collect per discovery run (across the whole roster)
STREAMER_MAX_DISCOVERY_CANDIDATES: int = 10
# How many processing failures before a video is permanently blacklisted
STREAMER_HISTORY_MAX_ATTEMPTS: int = 3
# Rolling cap on history file entries
STREAMER_HISTORY_CAP: int = 500
# Freshness window: only consider VODs published within this many hours
STREAMER_FRESHNESS_HOURS: int = 72

# ---------------------------------------------------------------------------
# Viral detection
# ---------------------------------------------------------------------------
# Maximum characters per Gemini prompt chunk (leaves room for system + JSON)
STREAMER_TRANSCRIPT_CHUNK_CHARS: int = 35_000
# Number of high-energy zones the pre-scan selects for Gemini analysis
STREAMER_PRE_SCAN_TOP_ZONES: int = 3
# Duration of each pre-scan analysis window (seconds)
STREAMER_PRE_SCAN_WINDOW_S: float = 1200.0   # 20 minutes

# ---------------------------------------------------------------------------
# Publish-gate policy
# ---------------------------------------------------------------------------
# What should the gate do when a *required* visual check cannot run at all —
# cv2 absent, weights missing, or no decodable frames?
#
# Fail CLOSED (default). An unverifiable clip must not reach a live account, and
# a silently-skipped check is indistinguishable from a passing one in every log,
# artifact and metric downstream, so fail-open is indistinguishable from success.
# Set STRICT_VISUAL_CHECKS=0 only as an explicit, temporary, audited override.
STRICT_VISUAL_CHECKS: bool = os.getenv("STRICT_VISUAL_CHECKS", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

# ---------------------------------------------------------------------------
# Invariants — the V2 overlay may never be looser than the shared V1 spec
# ---------------------------------------------------------------------------
def validate_spec() -> None:
    """Raise ``ValueError`` if a V2 bound is looser than the shared V1 bound.

    Deliberately *not* an ``assert``: assertions are stripped by ``python -O``,
    which would silently disable every guard below and let a widened V2 bound
    reach production. All violations are reported together so one run fixes all
    of them instead of failing on the first.
    """
    violations: List[str] = []

    if STREAMER_MIN_CLIP_DURATION_S < creative_spec.MIN_CLIP_DURATION_S:
        violations.append(
            f"V2 min duration ({STREAMER_MIN_CLIP_DURATION_S}s) is looser than "
            f"V1 ({creative_spec.MIN_CLIP_DURATION_S}s)"
        )
    if STREAMER_MAX_CLIP_DURATION_S > creative_spec.MAX_CLIP_DURATION_S:
        violations.append(
            f"V2 max duration ({STREAMER_MAX_CLIP_DURATION_S}s) exceeds "
            f"V1 ({creative_spec.MAX_CLIP_DURATION_S}s)"
        )
    if STREAMER_MIN_CLIP_DURATION_S > STREAMER_MAX_CLIP_DURATION_S:
        violations.append(
            f"V2 duration window is inverted: min {STREAMER_MIN_CLIP_DURATION_S}s > "
            f"max {STREAMER_MAX_CLIP_DURATION_S}s"
        )
    if STREAMER_OUTPUT_WIDTH != creative_spec.OUTPUT_WIDTH:
        violations.append(
            f"V2 output width {STREAMER_OUTPUT_WIDTH} diverges from "
            f"creative_spec {creative_spec.OUTPUT_WIDTH}"
        )
    if STREAMER_OUTPUT_HEIGHT != creative_spec.OUTPUT_HEIGHT:
        violations.append(
            f"V2 output height {STREAMER_OUTPUT_HEIGHT} diverges from "
            f"creative_spec {creative_spec.OUTPUT_HEIGHT}"
        )

    if violations:
        raise ValueError(
            "V2 creative spec is looser than the shared V1 spec:\n  - "
            + "\n  - ".join(violations)
        )


# Fail fast at import: a misconfigured overlay must never reach a render.
validate_spec()
