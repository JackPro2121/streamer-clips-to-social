"""Regression coverage for the static-shot motion guarantee.

Background: CI run 36171365900 rendered a 96s clip that editorial_qa rejected with
`freeze_interval` (severity error), which discarded the only clip and aborted the
run with `RuntimeError: No clips were rendered`. Root cause: a `portrait_face` shot
whose face timeline could not be resolved into a moving crop was emitted as a fixed
integer crop box (`crop=526:938:494:39`) with no `t` term, so statically-held source
content produced a literally frozen output.

The first fix added a per-branch oscillator. The corpus later measured 15-37px of
camera sway because that oscillator (0.286 Hz) and the composed-frame guarantee
(0.28 Hz) ran at the same time and added together. Motion is now owned by exactly
one mechanism -- ``_motion_guarantee_filter`` on the composed frame -- and these
tests pin that:

* a static shot's crop contains no oscillator of its own,
* the composed graph carries exactly one motion layer (two out-of-phase axes),
* a rendered still source produces zero freeze events at the strict filter.
"""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from core.face_tracker import FramingDecision, ShotPlan
from core.video_editor import build_video_filtergraph

SOURCE_W, SOURCE_H = 1920, 1080


def _shot_crop_segment(shot: ShotPlan, active_w: int = SOURCE_W, active_h: int = SOURCE_H) -> str:
    """The crop filter emitted for a single-shot multi_shot_dynamic decision."""
    decision = FramingDecision(
        mode="multi_shot_dynamic",
        face_count=1,
        shots=[shot],
        video_width=SOURCE_W,
        video_height=SOURCE_H,
        active_x=0,
        active_y=0,
        active_w=active_w,
        active_h=active_h,
    )
    graph = build_video_filtergraph(decision, ass_subtitle_path=None, burn_subtitles=False)
    segments = [part for part in graph.split(";") if "crop=" in part and "v_shot_0" in part]
    assert len(segments) == 1, f"expected exactly one shot filter, got {segments}"
    return segments[0]


class TestStaticShotCarriesNoOwnDrift(unittest.TestCase):
    """The per-branch oscillator is retired; the composed layer owns motion."""

    def test_static_portrait_shot_crop_is_static(self):
        segment = _shot_crop_segment(
            ShotPlan(start=0.0, end=2.5, mode="portrait_face", crop_x=454,
                     margin_v=460, zoom_factor=1.15, face_centers_timeline=[])
        )
        # The crop box must be fixed integers; any `t` term in the shot filter is
        # a second oscillator, which is exactly what produced the 15-37px sway.
        self.assertIn("crop=526:938:", segment)
        self.assertNotIn("asin(sin(", segment)
        self.assertNotIn("sin(t", segment)

    def test_static_portrait_shot_without_zoom_is_static_too(self):
        segment = _shot_crop_segment(
            ShotPlan(start=0.0, end=3.0, mode="portrait_face", crop_x=441,
                     margin_v=460, zoom_factor=1.0, face_centers_timeline=[])
        )
        self.assertNotIn("asin(sin(", segment)

    def test_resolved_face_panning_branch_is_unchanged(self):
        """Shots that already had motion must keep their timeline-driven crop."""
        decision = FramingDecision(
            mode="multi_shot_dynamic",
            face_count=1,
            video_width=SOURCE_W, video_height=SOURCE_H,
            active_x=0, active_y=0, active_w=SOURCE_W, active_h=SOURCE_H,
            shots=[ShotPlan(start=0.0, end=6.0, mode="portrait_face", crop_x=400,
                            margin_v=460, zoom_factor=1.0,
                            face_centers_timeline=[(0.0, 300), (6.0, 900)])],
        )
        graph = build_video_filtergraph(decision, ass_subtitle_path=None, burn_subtitles=False)
        self.assertIn("min(1.0,max(0.0,t/6.00))", graph)
        self.assertNotIn("asin(sin(t*1.8))", graph)


class TestComposedGraphCarriesExactlyOneGuarantee(unittest.TestCase):
    def test_one_motion_layer_two_axes(self):
        decision = FramingDecision(
            mode="multi_shot_dynamic",
            face_count=1,
            video_width=SOURCE_W, video_height=SOURCE_H,
            active_x=0, active_y=0, active_w=SOURCE_W, active_h=SOURCE_H,
            shots=[ShotPlan(start=0.0, end=2.5, mode="portrait_face", crop_x=657,
                            crop_y=0, crop_w=607, crop_h=1080)],
        )
        graph = build_video_filtergraph(decision, ass_subtitle_path=None, burn_subtitles=False)
        self.assertIn("[motion_out]", graph)
        self.assertEqual(
            graph.count("asin(sin("), 2,
            "the composed guarantee is one layer with two out-of-phase axes",
        )


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),
                     "ffmpeg/ffprobe required for the render-level freeze assertion")
class TestRenderedShotIsNeverFrozen(unittest.TestCase):
    """Render-level proof: a static source must not yield a freezedetect event.

    The filter is the strict one (n=0.003 d=0.2), not the 0.5s floor: the
    0.267s cover-freeze class shipped under a 0.5 floor once already.
    """

    QA_FREEZE_FILTER = "freezedetect=n=0.003:d=0.2"

    def _render_still_source(self, tmp: Path) -> Path:
        import os
        os.environ.setdefault("GITHUB_ACTIONS", "true")
        still = tmp / "still.png"
        src = tmp / "still.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
             "-i", f"testsrc2=size={SOURCE_W}x{SOURCE_H}:rate=1",
             "-frames:v", "1", str(still)],
            check=True, timeout=300,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-loop", "1", "-framerate", "30",
             "-t", "3.0", "-i", str(still), "-c:v", "libx264", "-crf", "18",
             "-pix_fmt", "yuv420p", str(src)],
            check=True, timeout=300,
        )
        return src

    def test_frozen_source_renders_without_freeze_event(self):
        from core.config import FPS

        tmp = Path(tempfile.mkdtemp())
        try:
            src = self._render_still_source(tmp)
            decision = FramingDecision(
                mode="multi_shot_dynamic", face_count=1,
                video_width=SOURCE_W, video_height=SOURCE_H,
                active_x=0, active_y=0, active_w=SOURCE_W, active_h=SOURCE_H,
                shots=[ShotPlan(start=0.0, end=3.0, mode="portrait_face", crop_x=454,
                                margin_v=460, zoom_factor=1.15, face_centers_timeline=[])],
            )
            graph = build_video_filtergraph(decision, ass_subtitle_path=None, burn_subtitles=False)
            out = tmp / "out.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-i", str(src),
                 "-filter_complex", graph, "-map", "[outv]",
                 "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
                 "-pix_fmt", "yuv420p", "-r", str(FPS), str(out)],
                check=True, timeout=900,
            )
            probe = subprocess.run(
                ["ffmpeg", "-hide_banner", "-nostats", "-i", str(out),
                 "-vf", self.QA_FREEZE_FILTER, "-an", "-f", "null", "NUL"],
                capture_output=True, text=True, timeout=600,
            )
            violations = [
                line for line in probe.stderr.splitlines()
                if "freeze_duration" in line or "freeze_start" in line
            ]
            self.assertEqual(
                violations, [],
                f"rendered a frozen interval from a static source: {violations}",
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
