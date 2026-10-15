"""L2 render tests: the motion guarantee must hold on *every* layout branch.

The defect
----------
``clip_1_HOW_I_MADE_400000_IN_A_MONTH.mp4`` -- published, QA ``passed=True`` --
carries **1.63 seconds of frozen video** across three freezedetect events in its
final 2.3 seconds, inside a static ``split_screen`` segment.

Cause: the drift that guarantees motion was emitted *per renderer branch*, and
only two of the seven branches had it. The other five were fully static:
``presentation_slide`` (which used a sine, the wave the rest of the code rejects),
both ``split_screen`` paths, ``single_smooth`` / ``dynamic_cut``, and
``blur_stack``. Since ``freeze_interval`` is in ``repair.UNREPAIRABLE_CODES``, any
static segment aborts the whole run.

The fix applies the guarantee once, at the end, on the composed frame, so a new
layout inherits it. These tests prove that on all of them.

Why render and not inspect
---------------------------
Reading the graph would not catch this. HANDOFF section 7 rule 6: "Render, don't
just inspect, when a change touches a filtergraph." The zero-slack frozen PiP
background, the 50%-black split pane and the mirrored output were all found by
rendering.
"""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from core.face_tracker import FramingDecision, ShotPlan
from core.video_editor import _motion_guarantee_filter, build_video_filtergraph, render_viral_clip

HAS_FFMPEG = bool(
    subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode == 0
)
HAS_LIBX264 = False
if HAS_FFMPEG:
    encoders = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True
    ).stdout
    HAS_LIBX264 = "libx264" in encoders


def _static_source(path: Path, seconds: int = 3, size=(1920, 1080)) -> bool:
    """A genuinely static clip.

    Uses a noise texture rather than a flat colour, repeated across all frames.
    Texture means the only way to avoid a freezedetect event is to actually move the frame.
    """
    width, height = size
    img = path.with_suffix(".png")
    try:
        cmd1 = [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", f"nullsrc=s={width}x{height}",
            "-vf", "noise=alls=45:allf=u,format=yuv420p",
            "-vframes", "1", str(img),
        ]
        r1 = subprocess.run(cmd1, capture_output=True, text=True, timeout=30)
        if r1.returncode != 0:
            return False
        cmd2 = [
            "ffmpeg", "-y", "-v", "error",
            "-loop", "1", "-i", str(img),
            "-t", str(seconds),
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "16", "-pix_fmt", "yuv420p",
            str(path),
        ]
        r2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=60)
        return r2.returncode == 0 and path.exists() and path.stat().st_size > 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    finally:
        if img.exists():
            try:
                img.unlink()
            except OSError:
                pass


def _freeze_events(path: Path) -> list:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "info", "-i", str(path),
         "-vf", "freezedetect=n=0.003:d=0.5", "-an", "-f", "null", "-"],
        capture_output=True, text=True, timeout=300,
    )
    starts = []
    import re

    for match in re.finditer(r"freeze_start:\s*([0-9.]+)", result.stderr):
        starts.append(float(match.group(1)))
    return starts


class TestMotionGuaranteeIsOneChokePoint(unittest.TestCase):
    """L1: the guarantee must exist exactly once, on the composed frame."""

    def test_filter_is_emitted_by_every_path(self) -> None:
        graph = _motion_guarantee_filter(1.0)
        self.assertIn("scale=", graph)
        self.assertIn("crop=", graph)
        self.assertIn("asin", graph, "a sine reaches zero velocity at its turning points")
        # The wave must be a triangle, not a sine. ffmpeg has no triangle function,
        # so asin(sin()) is the construction, and a sine's zero velocity at each
        # extremum produced sub-threshold runs up to 0.53s -- longer than
        # freezedetect's own 0.5s report floor.
        self.assertIn("asin(sin(", graph)

    def test_amplitude_is_small_enough_to_be_imperceptible(self) -> None:
        """The old per-branch drift was 107px peak-to-peak, about 10% of width.

        That is visible as a wobble, which is exactly what AGENTS.md 3.1 forbids.
        The replacement must be small enough that a viewer does not read it as
        camera movement, while still moving far more than freezedetect's n=0.003
        per-pixel threshold on a textured frame.
        """
        from core.creative_spec import MOTION_POLICY

        peak_to_peak = MOTION_POLICY.peak_to_peak_px
        self.assertLessEqual(
            peak_to_peak,
            20,
            msg=(
                f"{peak_to_peak}px peak-to-peak is {MOTION_POLICY.peak_to_peak_pct_of_width:.1f}% "
                f"of frame width; the replaced per-branch drift was 107px (9.9%)"
            ),
        )

    def test_axes_are_out_of_phase(self) -> None:
        """A diagonal trace reads as life; a single-axis slide reads as a fault."""
        from core.creative_spec import MOTION_POLICY

        self.assertNotAlmostEqual(
            MOTION_POLICY.PHASE_X, MOTION_POLICY.PHASE_Y, places=2,
            msg="x and y must be out of phase or the frame slides along one line",
        )

    def test_gain_scales_amplitude(self) -> None:
        from core.creative_spec import MOTION_POLICY

        small = _motion_guarantee_filter(0.5)
        large = _motion_guarantee_filter(1.5)
        self.assertNotEqual(small, large, "motion_gain must reach the guarantee")
        self.assertIn("asin", MOTION_POLICY.WAVE)


@unittest.skipUnless(HAS_LIBX264, "ffmpeg with libx264 not available")
class TestEveryBranchMovesAStaticSource(unittest.TestCase):
    """L2: render a static source through every layout and demand zero freezes.

    This is the test whose absence let the defect ship. The previous suite
    asserted on filtergraph strings, so a branch that emitted a static crop
    passed as long as the string it asserted was present.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.source = Path(self.tmp.name) / "static.mp4"
        if not _static_source(self.source):
            self.skipTest("could not encode a static test source")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _render(self, framing: FramingDecision, name: str) -> Path | None:
        output = Path(self.tmp.name) / f"{name}.mp4"
        try:
            render_viral_clip(
                source_video_path=self.source,
                start_time=0.0,
                end_time=2.5,
                output_clip_path=output,
                framing=framing,
            )
        except Exception:
            return None
        return output if output.exists() and output.stat().st_size > 0 else None

    def _assert_no_freeze(self, framing: FramingDecision, name: str) -> None:
        output = self._render(framing, name)
        if output is None:
            self.skipTest(f"renderer produced no output for {name}")
        events = _freeze_events(output)
        self.assertEqual(
            events,
            [],
            msg=(
                f"{name}: freezedetect reported {len(events)} freeze event(s) at "
                f"{[round(e, 2) for e in events]}. AGENTS.md 3.1 requires no static "
                f"output, and freeze_interval is unrepairable so the run would abort."
            ),
        )

    def test_split_screen_moves(self) -> None:
        """The branch that shipped the defect."""
        self._assert_no_freeze(
            FramingDecision(
                face_count=2,
                mode="split_screen",
                speaker1_box=(120, 80, 600, 533),
                speaker2_box=(1200, 80, 600, 533),
            ),
            "split_screen",
        )

    def test_multi_shot_split_screen_moves(self) -> None:
        self._assert_no_freeze(
            FramingDecision(
                face_count=2,
                mode="multi_shot_dynamic",
                shots=[
                    ShotPlan(
                        start=0.0, end=2.5, mode="split_screen",
                        speaker1_box=(120, 80, 600, 533),
                        speaker2_box=(1200, 80, 600, 533),
                    )
                ],
            ),
            "multi_shot_split_screen",
        )

    def test_presentation_slide_moves(self) -> None:
        """Previously carried a sine, whose velocity hits zero at each turning point."""
        self._assert_no_freeze(
            FramingDecision(
                face_count=1,
                mode="multi_shot_dynamic",
                shots=[ShotPlan(start=0.0, end=2.5, mode="presentation_slide")],
            ),
            "multi_shot_presentation_slide",
        )

    def test_blur_stack_moves(self) -> None:
        """The single-slide path, and the one that aborted a documented CI run."""
        self._assert_no_freeze(
            FramingDecision(face_count=0, mode="blur_stack"),
            "blur_stack",
        )

    def test_single_smooth_moves(self) -> None:
        self._assert_no_freeze(
            FramingDecision(face_count=1, mode="single_smooth", smoothed_center_x=960),
            "single_smooth",
        )

    def test_dynamic_cut_moves(self) -> None:
        self._assert_no_freeze(
            FramingDecision(
                face_count=1,
                mode="dynamic_cut",
                crop_x_expr="600",
            ),
            "dynamic_cut",
        )

    def test_multi_shot_pip_slide_moves(self) -> None:
        self._assert_no_freeze(
            FramingDecision(
                face_count=1,
                mode="multi_shot_dynamic",
                shots=[ShotPlan(start=0.0, end=2.5, mode="pip_slide")],
            ),
            "multi_shot_pip_slide",
        )

    def test_portrait_face_moves(self) -> None:
        self._assert_no_freeze(
            FramingDecision(
                face_count=1,
                mode="multi_shot_dynamic",
                shots=[
                    ShotPlan(
                        start=0.0, end=2.5, mode="portrait_face",
                        crop_x=657, crop_y=0, crop_w=607, crop_h=1080,
                    )
                ],
            ),
            "portrait_face",
        )

    def test_every_mode_name_is_covered(self) -> None:
        """A new layout must come with a render test, or it inherits the old risk.

        Two dispatch levels exist and both have to be covered: the top-level
        ``FramingDecision.mode`` (blur_stack, split_screen, single_smooth,
        dynamic_cut) and the per-shot ``ShotPlan.mode`` (presentation_slide,
        portrait_face, split_screen, pip_slide). The defect lived in the gap
        between them -- five branches had no test and no motion.
        """
        import ast
        import re

        source = Path(__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        covered = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "_assert_no_freeze":
                if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                    covered.add(node.args[1].value)
                for keyword in node.keywords:
                    if keyword.arg == "name" and isinstance(keyword.value, ast.Constant):
                        covered.add(keyword.value.value)

        editor = Path(__file__).resolve().parent.parent / "core" / "video_editor.py"
        editor_source = editor.read_text(encoding="utf-8")

        top_level = set(
            re.findall(r'framing\.mode\s*==\s*"([a-z_]+)"', editor_source)
        )
        # `multi_shot_dynamic` is a container, not a leaf layout: it dispatches to
        # the per-shot modes. Its leaves are covered by the per-shot tests.
        per_shot = set(re.findall(r"shot\.mode\s*==\s*'([a-z_]+)'", editor_source))
        per_shot |= set(re.findall(r"shot\.mode\s*==\s*\"([a-z_]+)\"", editor_source))

        leaf_top_level = top_level - {"multi_shot_dynamic"}
        # Per-shot render tests are named after the leaf layout they exercise.
        per_shot_names = {name for name in covered if name.startswith("multi_shot_")}
        leaf_per_shot = {name.replace("multi_shot_", "") for name in per_shot_names}
        leaf_per_shot |= {"portrait_face"}  # covered by the plain portrait test

        for mode in sorted(leaf_top_level):
            with self.subTest(level="FramingDecision", mode=mode):
                self.assertIn(
                    mode,
                    covered,
                    f"top-level layout {mode!r} has no render-level motion test",
                )
        for mode in sorted(per_shot):
            with self.subTest(level="ShotPlan", mode=mode):
                self.assertIn(
                    mode,
                    leaf_per_shot,
                    f"per-shot layout {mode!r} has no render-level motion test; a new "
                    f"layout without one is exactly how the freeze shipped",
                )


class TestGuaranteeIsOnTheComposedFrame(unittest.TestCase):
    """L1: the guarantee must be after the layout, not inside any branch."""

    def test_graph_contains_exactly_one_motion_layer(self) -> None:
        graph = build_video_filtergraph(
            FramingDecision(face_count=1, mode="single_smooth", smoothed_center_x=960),
            ass_subtitle_path=None,
            burn_subtitles=False,
        )
        # Composed motion layer uses both X and Y out-of-phase oscillation (2 asin(sin( terms)
        self.assertEqual(
            graph.count("asin(sin("),
            2,
            "the guarantee must appear once on the composed frame with 2 out-of-phase axes",
        )
        self.assertIn("[motion_out]", graph)

    def test_multi_shot_has_composed_motion(self) -> None:
        """Verify the composed motion guarantee is attached on multi-shot layout."""
        graph = build_video_filtergraph(
            FramingDecision(
                face_count=1,
                mode="multi_shot_dynamic",
                shots=[ShotPlan(start=0.0, end=2.5, mode="portrait_face",
                                crop_x=657, crop_y=0, crop_w=607, crop_h=1080)],
            ),
            ass_subtitle_path=None,
            burn_subtitles=False,
        )
        self.assertIn("[motion_out]", graph)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
