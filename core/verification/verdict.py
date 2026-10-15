"""The single authority on whether a rendered clip is publishable.

Why this module exists
----------------------
Three independent opinions existed about every clip: the ``EditPlan``, the
renderer, and ``editorial_qa``. Nothing compared what the renderer produced
against what the spec asked for, and the verifier was the weakest of the three
because it inspected a plan the same code had just written. Agreement between it
and the renderer was guaranteed by construction.

The measured consequence, on six real published clips:

* 370 tests green, editorial QA ``passed=True``, CI run ``success``
* yet one clip shipped with 1.63 s of frozen video inside a frame that was 50%
  dead black
* four of six shipped with audio and video streams disagreeing about their own
  duration by 34-200 ms

Every one of those is visible in a rendered frame or a stream header. None of
them was in the plan the QA checked.

Design
------
``verdict.json`` is the output. It is:

* **complete** -- every metric from every module, pass or fail, not just the
  failures, so a regression is visible as a number moving rather than as an
  absence;
* **machine-readable** -- so CI can gate on it, a later run can diff against it,
  and a human can read it;
* **graded** -- ``error`` blocks publication, ``warning`` is surfaced, ``info`` is
  recorded. Collapsing these into one boolean is how the original gate managed to
  be both too lax (0.8 s per-event freeze tolerance) and unreadable.

This module does not decide what is acceptable; the thresholds live in each
metric module and, ultimately, in ``core.creative_spec``. It decides what is
*blocking*, and it is the only thing ``main.py`` is permitted to consult before
uploading or scheduling.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import av_sync, fixtures, geometry, loudness, mirror, motion, probe, sway

SCHEMA_VERSION = 1

# Codes are stable strings so a later run can diff them. They deliberately
# mirror the existing editorial_qa vocabulary where one applies, so the log
# output reads consistently with what the pipeline already prints.
CODE_OUTPUT_UNREADABLE = "output_unreadable"
CODE_ASPECT_WRONG = "aspect_wrong"
CODE_PILLARBOX = "pillarbox"
CODE_LETTERBOX = "letterbox"
CODE_FREEZE = "freeze_cumulative"
CODE_FREEZE_UNTERMINATED = "freeze_unterminated"
CODE_BLACK_INTERVAL = "black_interval"
CODE_AV_STRUCTURAL_DRIFT = "av_structural_drift"
CODE_AV_SOURCE_DRIFT = "av_source_drift"
CODE_LOUDNESS = "loudness"
CODE_MIRRORED = "mirrored_source"
CODE_CAPTION_ON_FACE = "caption_on_face"
CODE_DURATION = "duration_out_of_contract"
CODE_CHECK_UNRUNNABLE = "check_unrunnable"

# A caption covering more than this fraction of the **upper** face (brow through
# eyes) is unreadable. Measured on the golden masters:
#
#   broke_1       100.0%   caption entirely across the brow and eyes
#   grinding_2    100.0%
#   hardwork_3    100.0%
#   four_hundred_k 37.3%
#   disney_2        0.0%   caption sits low, on the wall art, not the face
#   sixty_k_1       0.0%   faces are small in a two-pane split
#
# The band is deliberately the upper face rather than the whole face. In a tight
# portrait crop the face fills the frame, so whole-face overlap saturates at 100%
# on five of six clips and distinguishes nothing.
CAPTION_ON_FACE_BLOCK_PCT = 15.0
CAPTION_ON_FACE_WARN_PCT = 5.0

# AGENTS.md 2.C: "Strictly 30 to 50 seconds (sweet spot). Maximum 55 seconds."
# Note the gap with the code, which caps at 140s (config.MAX_CLIP_DURATION) and
# 148s after main._extend_moment_to_complete_transcript. See
# AGENTS_COMPLIANCE_AUDIT.md section 2.C.
CONTRACT_MIN_DURATION_S = 30.0
CONTRACT_MAX_DURATION_S = 55.0


@dataclass
class Finding:
    """One graded observation about a rendered clip."""

    code: str
    severity: str  # error | warning | info
    message: str
    spec_clause: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def blocks(self) -> bool:
        return self.severity == "error"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "spec_clause": self.spec_clause,
            "metrics": self.metrics,
        }


@dataclass
class Verdict:
    """Everything known about one rendered clip."""

    path: str = ""
    passed: bool = True
    readable: bool = True
    reason: str = ""
    duration_s: float = 0.0
    dimensions: Dict[str, int] = field(default_factory=dict)
    findings: List[Finding] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def add(self, finding: Finding) -> None:
        self.findings.append(finding)
        if finding.blocks:
            self.passed = False

    @property
    def blocking(self) -> List[Finding]:
        return [finding for finding in self.findings if finding.blocks]

    @property
    def warnings(self) -> List[Finding]:
        return [finding for finding in self.findings if finding.severity == "warning"]

    @property
    def blocking_codes(self) -> List[str]:
        return [finding.code for finding in self.blocking]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "path": self.path,
            "passed": self.passed,
            "readable": self.readable,
            "reason": self.reason,
            "duration_s": round(self.duration_s, 3),
            "dimensions": self.dimensions,
            "blocking_codes": self.blocking_codes,
            "findings": [finding.as_dict() for finding in self.findings],
            "metrics": self.metrics,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, sort_keys=False)

    def write(self, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.to_json() + "\n", encoding="utf-8")
        return destination

    def summary(self) -> str:
        """One block of human-readable text, for the CI job summary."""
        state = "PASS" if self.passed else "FAIL"
        lines = [
            f"### Video verdict: {state}",
            "",
            f"- file: `{self.path}`",
            f"- duration: {self.duration_s:.2f}s",
            f"- dimensions: {self.dimensions.get('width')}x{self.dimensions.get('height')}",
            "",
        ]
        if not self.findings:
            lines.append("No findings.")
        else:
            lines.append("| severity | code | message |")
            lines.append("| --- | --- | --- |")
            for finding in self.findings:
                message = finding.message.replace("|", "/")
                lines.append(f"| {finding.severity} | `{finding.code}` | {message} |")
        return "\n".join(lines)


def _duration_finding(
    duration_s: float,
    min_duration_s: float = CONTRACT_MIN_DURATION_S,
    max_duration_s: float = CONTRACT_MAX_DURATION_S,
) -> Optional[Finding]:
    if duration_s <= 0.0:
        return None
    if min_duration_s <= duration_s <= max_duration_s:
        return None
    return Finding(
        code=CODE_DURATION,
        severity="error",
        message=(
            f"duration {duration_s:.2f}s is outside the "
            f"{min_duration_s:.0f}-{max_duration_s:.0f}s contract"
        ),
        spec_clause="AGENTS.md 2.C",
        metrics={"duration_s": round(duration_s, 3)},
    )


def verify(
    path: Path,
    with_correlation: bool = False,
    with_mirror: bool = True,
    with_captions: bool = True,
    with_sway: bool = True,
    source_path: Optional[Path] = None,
    source_expected_start: Optional[float] = None,
    expected_width: Optional[int] = None,
    expected_height: Optional[int] = None,
    min_duration_s: Optional[float] = None,
    max_duration_s: Optional[float] = None,
) -> Verdict:
    """Run every check against a rendered clip and return a single verdict.

    Every module is best-effort: a module that cannot measure reports ``info``
    rather than blocking, because a verifier that blocks on its own failure to
    run trains people to disable it.
    """
    verdict = Verdict(path=str(path))

    if not path.exists():
        verdict.readable = False
        verdict.passed = False
        verdict.reason = f"file not found: {path}"
        verdict.add(
            Finding(CODE_OUTPUT_UNREADABLE, "error", verdict.reason, "internal")
        )
        return verdict

    width, height = probe.dimensions(path)
    verdict.dimensions = {"width": width, "height": height}
    verdict.duration_s = probe.duration_seconds(path)

    if width <= 0 or height <= 0 or verdict.duration_s <= 0.0:
        verdict.readable = False
        verdict.passed = False
        verdict.reason = "no decodable video stream"
        verdict.add(
            Finding(
                CODE_OUTPUT_UNREADABLE,
                "error",
                f"{verdict.reason} (ffprobe reported {width}x{height}, "
                f"{verdict.duration_s:.2f}s)",
                "internal",
            )
        )
        return verdict

    # --- geometry -----------------------------------------------------------
    geometry_report = geometry.analyse(path)
    verdict.metrics["geometry"] = geometry_report.as_dict()
    if not geometry_report.ok:
        verdict.add(
            Finding(
                CODE_OUTPUT_UNREADABLE,
                "error",
                f"geometry could not be measured: {geometry_report.reason}",
                "internal",
                geometry_report.as_dict(),
            )
        )
    else:
        if geometry_report.blocking:
            verdict.add(
                Finding(
                    CODE_PILLARBOX,
                    "error",
                    (
                        f"{geometry_report.worst_vertical_bar_pct:.1f}% of the frame "
                        f"width is dead black, sustained across "
                        f"{geometry_report.bar_samples}/{geometry_report.samples} "
                        f"samples"
                    ),
                    "AGENTS.md 3.3",
                    geometry_report.as_dict(),
                )
            )
        elif geometry_report.warning:
            verdict.add(
                Finding(
                    CODE_PILLARBOX,
                    "warning",
                    (
                        f"{geometry_report.worst_vertical_bar_pct:.1f}% inset, below "
                        f"the {geometry.BLOCKING_BAR_PCT}% blocking threshold. For a "
                        f"blur_stack shot this is the designed dark background"
                    ),
                    "AGENTS.md 3.3",
                    geometry_report.as_dict(),
                )
            )
        if geometry_report.worst_horizontal_bar_pct >= geometry.BLOCKING_BAR_PCT:
            verdict.add(
                Finding(
                    CODE_LETTERBOX,
                    "error",
                    f"{geometry_report.worst_horizontal_bar_pct:.1f}% of the frame "
                    f"height is dead black",
                    "AGENTS.md 3.3",
                    geometry_report.as_dict(),
                )
            )
        if expected_width and expected_height and (
            width != expected_width or height != expected_height
        ):
            verdict.add(
                Finding(
                    CODE_ASPECT_WRONG,
                    "error",
                    f"expected {expected_width}x{expected_height}, got {width}x{height}",
                    "AGENTS.md 3.3",
                    {"width": width, "height": height},
                )
            )

    # --- motion -------------------------------------------------------------
    motion_report = motion.analyse(path, clip_duration=verdict.duration_s)
    verdict.metrics["motion"] = motion_report.as_dict()
    if motion_report.cumulative_freeze_s > 0.0:
        verdict.add(
            Finding(
                CODE_FREEZE,
                "error" if motion_report.cumulative_freeze_s > 0.3 else "warning",
                (
                    f"{motion_report.cumulative_freeze_s:.2f}s of frozen video across "
                    f"{motion_report.event_count} events "
                    f"({motion_report.freeze_fraction:.1%} of the clip); longest "
                    f"{motion_report.longest_freeze_s:.2f}s"
                ),
                "AGENTS.md 3.1",
                motion_report.as_dict(),
            )
        )
    if motion_report.has_unterminated_freeze:
        verdict.add(
            Finding(
                CODE_FREEZE_UNTERMINATED,
                "error",
                "a freeze runs to end-of-file and was not closed by freezedetect",
                "AGENTS.md 3.1",
                motion_report.as_dict(),
            )
        )
    if motion_report.cumulative_black_s > 0.5:
        verdict.add(
            Finding(
                CODE_BLACK_INTERVAL,
                "error",
                f"{motion_report.cumulative_black_s:.2f}s of full-frame black",
                "AGENTS.md 3.1",
                motion_report.as_dict(),
            )
        )

    # --- A/V sync -----------------------------------------------------------
    sync_report = av_sync.analyse(path, with_correlation=with_correlation)
    verdict.metrics["av_sync"] = sync_report.as_dict()
    if not sync_report.structural_ok:
        verdict.add(
            Finding(
                CODE_AV_STRUCTURAL_DRIFT,
                "error",
                (
                    f"audio is {sync_report.duration_delta_ms:+.0f}ms "
                    f"{'longer' if sync_report.duration_delta_ms > 0 else 'shorter'} "
                    f"than video; start delta {sync_report.start_delta_ms:+.0f}ms. "
                    f"AGENTS.md 3.2 claims 0ms drift."
                ),
                "AGENTS.md 3.2",
                sync_report.as_dict(),
            )
        )

    # --- audio content alignment against the source segment ------------------
    # The deterministic perceptual check. SyncNet's LSE metrics could not be
    # calibrated onto this domain (measured on the corpus), so the gate compares
    # the clip's own audio against the segment it was cut from and reports the
    # exact drift in milliseconds.
    if source_path is not None and source_expected_start is not None:
        alignment = av_sync.align_with_source(path, source_path, source_expected_start)
        verdict.metrics["source_alignment"] = alignment.as_dict()
        if not alignment.measured:
            verdict.add(
                Finding(
                    CODE_CHECK_UNRUNNABLE,
                    "warning",
                    f"source alignment could not be measured: {alignment.reason}",
                    "internal",
                    alignment.as_dict(),
                )
            )
        elif abs(alignment.drift_ms) > av_sync.SOURCE_ALIGN_TOLERANCE_MS:
            verdict.add(
                Finding(
                    CODE_AV_SOURCE_DRIFT,
                    "error",
                    (
                        f"rendered soundtrack is {alignment.drift_ms:+.0f}ms relative to the "
                        f"trim (positive = audio late; expected start "
                        f"{alignment.expected_s:.3f}s; tolerance "
                        f"{av_sync.SOURCE_ALIGN_TOLERANCE_MS:.0f}ms; "
                        f"confidence {alignment.confidence:.3f})"
                    ),
                    "AGENTS.md 3.2",
                    alignment.as_dict(),
                )
            )

    # --- loudness -----------------------------------------------------------
    loudness_report = loudness.analyse(path)
    verdict.metrics["loudness"] = loudness_report.as_dict()
    if loudness_report.measured and not loudness_report.ok:
        verdict.add(
            Finding(
                CODE_LOUDNESS,
                "error",
                loudness_report.reason,
                loudness_report.spec_clause,
                loudness_report.as_dict(),
            )
        )
    elif loudness_report.measured and loudness_report.warn_reason:
        verdict.add(
            Finding(
                CODE_LOUDNESS,
                "warning",
                loudness_report.warn_reason,
                loudness_report.spec_clause,
                loudness_report.as_dict(),
            )
        )

    # --- mirror -------------------------------------------------------------
    if with_mirror:
        # The pipeline's own burned-in captions must be excluded first. libass
        # renders them after the frame is assembled, so they are always correctly
        # oriented; on the first CI run they outvoted the one genuinely mirrored
        # element and the detector returned a false negative on a clip that is
        # demonstrably mirrored.
        caption_band: List[Any] = []
        caption_reason = ""
        try:
            from . import captions as captions_module

            caption_band, caption_reason = captions_module.detect_caption_regions(path)
        except Exception as error:  # pragma: no cover - defensive
            caption_reason = f"caption band detection failed: {error}"

        verdict.metrics["caption_band"] = {
            "found": bool(caption_band),
            "reason": caption_reason,
            "rects": [rect.as_dict() for rect in caption_band],
        }

        mirror_report = mirror.analyse(path, exclude_rects=caption_band or None)
        verdict.metrics["mirror"] = mirror_report.as_dict()
        if mirror_report.mirrored:
            verdict.add(
                Finding(
                    CODE_MIRRORED,
                    "error",
                    mirror_report.reason,
                    "output quality",
                    mirror_report.as_dict(),
                )
            )
        elif not mirror_report.measurable:
            # Warning, not info. Measured: with the pipeline's own captions
            # excluded, no clip in the corpus has any machine-readable source
            # text, so mirror state is undecidable from rendered output. That is
            # a real capability gap and it must not read as a pass.
            verdict.add(
                Finding(
                    CODE_MIRRORED,
                    "warning",
                    (
                        f"mirror state not determinable from rendered output: "
                        f"{mirror_report.reason} Selfie-flipped source footage would "
                        f"not be caught by this report."
                    ),
                    "output quality",
                    mirror_report.as_dict(),
                )
            )
        if not caption_band:
            verdict.metrics.setdefault("notes", []).append(
                f"caption band not located ({caption_reason}); mirror detection "
                f"ran without caption exclusion and may under-report"
            )

    # --- captions vs faces --------------------------------------------------
    # The check editorial_qa cannot make. It compares shot.caption_rect against
    # shot.protected_regions, both written by the same code minutes apart, so it
    # confirms the plan. The caption that lands on the face is produced by the
    # collision-avoidance *relocation* those same regions triggered, so no
    # plan-only check can see it.
    caption_payload: Dict[str, Any] = {}
    if with_captions:
        try:
            from . import captions as captions_module
            from . import faces as faces_module

            face_report = faces_module.analyse(path, samples=14)
            caption_payload["faces"] = face_report.as_dict()
            if face_report.any_faces:
                avoid = faces_module.avoid_rects(face_report, upper_only=True)
                on_face, at, measured = captions_module.measure_on_face(
                    path, avoid, max_samples=20
                )
                caption_payload["on_face_pct"] = round(on_face, 2)
                caption_payload["on_face_at_pct"] = round(at, 1)
                caption_payload["frames_measured"] = measured
                caption_payload["region"] = "upper face (brow through eyes)"
                if on_face > CAPTION_ON_FACE_BLOCK_PCT:
                    verdict.add(
                        Finding(
                            CODE_CAPTION_ON_FACE,
                            "error",
                            (
                                f"{on_face:.0f}% of burned-in caption pixels sit on the "
                                f"subject's brow/eyes (worst at {at:.0f}% through the "
                                f"clip). Captions are unreadable there."
                            ),
                            "AGENTS.md 2.A / 3.4",
                            caption_payload,
                        )
                    )
                elif on_face > CAPTION_ON_FACE_WARN_PCT:
                    verdict.add(
                        Finding(
                            CODE_CAPTION_ON_FACE,
                            "warning",
                            f"{on_face:.0f}% of caption pixels touch the upper face",
                            "AGENTS.md 2.A / 3.4",
                            caption_payload,
                        )
                    )
            else:
                caption_payload["on_face_pct"] = None
                captions_present = bool(
                    (verdict.metrics.get("caption_band") or {}).get("found")
                )
                if not captions_present:
                    try:
                        from . import captions as captions_module

                        found, _reason = captions_module.detect_caption_regions(path)
                        captions_present = bool(found)
                    except Exception:
                        captions_present = False
                caption_payload["captions_present"] = captions_present
                # A check that cannot run must not read as a pass. When captions
                # are burned in, the face-avoidance contract is unverifiable, so
                # this blocks; when there are no captions there is nothing to
                # check and a warning is honest.
                severity = "error" if captions_present else "warning"
                verdict.add(
                    Finding(
                        CODE_CHECK_UNRUNNABLE,
                        severity,
                        (
                            f"caption placement could not be measured: "
                            f"{face_report.reason}. "
                            + (
                                "Burned-in captions were detected, so the "
                                "face-avoidance contract cannot be verified."
                                if captions_present
                                else "No burned-in captions were detected."
                            )
                        ),
                        "internal",
                        caption_payload,
                    )
                )
        except Exception as error:  # pragma: no cover - defensive
            caption_payload["error"] = str(error)
    verdict.metrics["captions"] = caption_payload

    # --- camera sway --------------------------------------------------------
    # Thresholds are calibrated on the golden-master corpus plus the run
    # 37202390593 artifact; the evidence table lives in sway.py's docstring.
    # Re-tighten only when 2-3 more published runs populate the 39-126px gap.
    if with_sway:
        sway_report = sway.analyse(path)
        verdict.metrics["sway"] = sway_report.as_dict()
        if not sway_report.ok:
            verdict.add(
                Finding(
                    CODE_CHECK_UNRUNNABLE,
                    "warning",
                    f"camera sway could not be measured: {sway_report.reason}",
                    "internal",
                    sway_report.as_dict(),
                )
            )
        elif sway_report.measurable and (
            sway_report.worst_p2p_px > sway.SWAY_BLOCK_PX
            or sway_report.worst_broadband_p2p_px > sway.BROADBAND_BLOCK_PX
        ):
            verdict.add(
                Finding(
                    sway.CODE_SWAY,
                    "error",
                    (
                        f"camera sway {sway_report.worst_p2p_px:.0f}px peak-to-peak in the "
                        f"policy band on the {sway_report.worst_axis} axis at "
                        f"{sway_report.worst_freq_hz:.2f}Hz "
                        f"({sway_report.worst_start_s:.1f}-{sway_report.worst_end_s:.1f}s); "
                        f"broadband {sway_report.worst_broadband_p2p_px:.0f}px at "
                        f"{sway_report.worst_broadband_freq_hz:.2f}Hz. Policy block "
                        f"<= {sway.SWAY_BLOCK_PX:.0f}px; broadband block "
                        f"<= {sway.BROADBAND_BLOCK_PX:.0f}px; warn at policy "
                        f">{sway.SWAY_WARN_PX:.0f}px / broadband "
                        f">{sway.BROADBAND_WARN_PX:.0f}px"
                    ),
                    "AGENTS.md 3.1",
                    sway_report.as_dict(),
                )
            )
        elif sway_report.measurable and (
            sway_report.worst_p2p_px > sway.SWAY_WARN_PX
            or sway_report.worst_broadband_p2p_px > sway.BROADBAND_WARN_PX
        ):
            verdict.add(
                Finding(
                    sway.CODE_SWAY,
                    "warning",
                    (
                        f"camera sway {sway_report.worst_p2p_px:.0f}px peak-to-peak in the "
                        f"policy band on the {sway_report.worst_axis} axis at "
                        f"{sway_report.worst_freq_hz:.2f}Hz; broadband "
                        f"{sway_report.worst_broadband_p2p_px:.0f}px at "
                        f"{sway_report.worst_broadband_freq_hz:.2f}Hz"
                    ),
                    "AGENTS.md 3.1",
                    sway_report.as_dict(),
                )
            )

    # --- duration contract --------------------------------------------------
    min_dur = CONTRACT_MIN_DURATION_S if min_duration_s is None else min_duration_s
    max_dur = CONTRACT_MAX_DURATION_S if max_duration_s is None else max_duration_s
    duration_finding = _duration_finding(
        verdict.duration_s, min_duration_s=min_dur, max_duration_s=max_dur
    )
    if duration_finding is not None:
        verdict.add(duration_finding)

    return verdict


def verify_corpus(
    master_ids: Optional[List[str]] = None,
    with_mirror: bool = True,
    with_captions: bool = True,
    **kwargs: Any,
) -> Dict[str, Verdict]:
    """Verify every golden master. Used by the CI visual-verification job."""
    results: Dict[str, Verdict] = {}
    for master in fixtures.available():
        if master_ids and master.id not in master_ids:
            continue
        results[master.id] = verify(
            master.path,
            with_mirror=with_mirror,
            with_captions=with_captions,
            expected_width=1080,
            expected_height=1920,
            **kwargs,
        )
    return results


def corpus_report(results: Dict[str, Verdict]) -> Dict[str, Any]:
    """Aggregate corpus results, including whether the verifier agrees with humans.

    The agreement check is the important part. The golden-master manifest records
    what a person saw on the timeline. If the verifier's verdict does not match,
    then the *verifier* is wrong -- not the human, and not the threshold.
    """
    rows: List[Dict[str, Any]] = []
    disagreements: List[Dict[str, Any]] = []
    for master_id, verdict in sorted(results.items()):
        master = fixtures.by_id(master_id)
        expected_pass = master.expects_pass if master is not None else None
        agrees = (expected_pass is None) or (expected_pass == verdict.passed)
        row = {
            "id": master_id,
            "file": master.file if master is not None else "",
            "duration_s": round(verdict.duration_s, 2),
            "verifier_passed": verdict.passed,
            "human_verdict": expected_pass,
            "agrees": agrees,
            "blocking_codes": verdict.blocking_codes,
        }
        rows.append(row)
        if not agrees:
            disagreements.append(row)
    return {
        "schema_version": SCHEMA_VERSION,
        "clips": rows,
        "disagreements": disagreements,
        "agreement_rate": (
            1.0 - (len(disagreements) / len(rows)) if rows else 0.0
        ),
    }
