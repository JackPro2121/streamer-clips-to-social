"""Pixel-level verification of rendered output.

This package exists because of a specific, measured failure: the pipeline shipped a
clip containing 1.63 seconds of frozen video inside a frame that was 50% dead
black, and editorial QA reported ``passed=True``. The cause was structural, not a
tuning mistake. Three separate opinions existed about every clip --

* the **plan** (``EditPlan`` / ``CompositionPlan`` / ``ShotPlan``),
* the **renderer** (``video_editor.build_video_filtergraph``), and
* the **verifier** (``editorial_qa``) --

and nothing ever compared what the renderer produced against what the spec asked
for. The verifier was the weakest of the three, because it inspected a plan that
the same code had written moments earlier. Agreement between it and the renderer
was therefore guaranteed by construction, which is why "QA passed" carried almost
no information about whether the video was watchable.

Everything here reads the **rendered pixels**. That is the whole point.

Module map, and the defect each one is responsible for catching:

============================  ==========================================
Module                        Catches
============================  ==========================================
:mod:`~.probe`                one cached entry point for ffprobe/ffmpeg
:mod:`~.geometry`             pillarboxing / letterboxing / wrong aspect
:mod:`~.motion`               frozen output, dead output, scene sanity
:mod:`~.loudness`             EBU R128 drift (already correct; lock it in)
:mod:`~.av_sync`              A/V drift -- no check for this exists today
:mod:`~.captions`             captions on faces, off safe zone, on dividers
:mod:`~.mirror`               selfie-flipped source footage
:mod:`~.typography`           hook-badge plate alignment, overflow, contrast
:mod:`~.verdict`              aggregation, thresholds, blocking decisions
:mod:`~.fixtures`             the golden-master corpus
============================  ==========================================

Design rules for this package:

1. **Pure functions where possible.** Every metric is computable from a numpy
   array with no ffmpeg invocation, so it can be unit-tested against synthetic
   input. ffmpeg is only used to *get* pixels.
2. **Never raise on a malformed input.** A verifier that crashes is a verifier
   that gets wrapped in ``try/except`` by a caller and silently skipped. Failures
   are returned as data.
3. **Thresholds live in :mod:`core.creative_spec`, not here.** This module decides
   *what* the number is; the spec decides whether it is acceptable.
4. **Report the worst sample, not the mean.** A clip that is pillarboxed for 2 of
   45 seconds is a broken clip.
"""

from __future__ import annotations

__all__ = [
    "av_sync",
    "captions",
    "fixtures",
    "geometry",
    "loudness",
    "mirror",
    "motion",
    "probe",
    "typography",
    "verdict",
]
