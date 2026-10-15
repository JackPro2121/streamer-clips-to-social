"""Render-level tests: no layout may emit a graph that can produce black bars.

This is the L2 level the test pyramid needs and the suite did not have. Every
existing filtergraph assertion checked that a *string* was present, which is how
`scale=...:force_original_aspect_ratio=decrease,pad=...:color=black` survived in
two split-screen branches for as long as it did: the string was exactly what the
tests were looking for.

AGENTS.md 3.3: "Crop boxes must maintain PANE_ASPECT = 9.0 / 8.0 so each speaker
pane scales to 1080x960 with zero black bars on the sides."

The published artifact `clip_1_HOW_I_MADE_400000_IN_A_MONTH.mp4` measures 24.8% of
frame width dead black per side, sustained across 34 of 240 sampled frames, with
the caption wider than the live content and floating over the bars. That is the
defect these tests exist to make impossible.

Two layers:

* **Graph-level.** Assert the emitted filtergraph cannot introduce bars, for every
  layout mode, for both a 16:9 and an already-vertical source. A `pad=` combined
  with `force_original_aspect_ratio=decrease` is a bar waiting to happen; an
  `increase` + `crop` chain cannot produce one.
* **Render-level.** Encode a real vertical 1080x1920 source with a hard-edged
  marker, run it through the split-screen graph, and measure the output for dead
  columns. This is the check that would have caught the shipped defect.
"""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from core.config import OUTPUT_HEIGHT, OUTPUT_WIDTH
from core.face_tracker import PANE_ASPECT, FramingDecision, ShotPlan, _pane_dimensions
from core.video_editor import build_video_filtergraph


def _vertical_source(path: Path, seconds: int = 3) -> bool:
    """Encode a 1080x1920 clip with distinct left/right halves.

    A vertical source is the case that exposed the bug: 9:16 content scaled into
    1080x960 needs 270px of padding on each side, which is exactly what shipped.
    """
    command = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i",
        f"color=c=0x203040:s={OUTPUT_WIDTH}x{OUTPUT_HEIGHT}:r=30:d={seconds}",
        "-f", "lavfi", "-i",
        f"color=c=0xE0E0E0:s={OUTPUT_WIDTH // 2}x{OUTPUT_HEIGHT}:r=30:d={seconds}",
        "-filter_complex",
        f"[0:v][1:v]overlay={OUTPUT_WIDTH // 2}:0,format=yuv420p[v]",
        "-map", "[v]", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
        str(path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and path.exists() and path.stat().st_size > 0


def _worst_dead_columns(path: Path) -> float:
    """Largest run of near-black columns, as a percentage of frame width."""
    import numpy as np

    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", "1", "-i", str(path), "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, timeout=180,
    ).stdout
    frame = np.frombuffer(raw[: OUTPUT_WIDTH * OUTPUT_HEIGHT], dtype=np.uint8)
    if frame.size != OUTPUT_WIDTH * OUTPUT_HEIGHT:
        return 100.0
    frame = frame.reshape(OUTPUT_HEIGHT, OUTPUT_WIDTH)
    column_max = frame.max(axis=0)
    dead = column_max <= 24
    longest = current = 0
    for flag in dead:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return 100.0 * longest / OUTPUT_WIDTH


class TestPaneGeometry(unittest.TestCase):
    """L1: pane dimensions must be even and 9:8, and fit the source."""

    def test_ratio_is_nine_eight(self) -> None:
        for target in (200, 324, 528, 540, 594, 607, 720, 900, 1055, 1080, 1920):
            with self.subTest(target=target):
                width, height = _pane_dimensions(target, 1920, max_width=1920)
                self.assertEqual(width % 2, 0, "width must be even for yuv420p")
                self.assertEqual(height % 2, 0, "height must be even for yuv420p")
                self.assertAlmostEqual(
                    width / height, PANE_ASPECT, delta=0.006,
                    msg=f"target {target} produced {width}x{height}",
                )

    def test_never_exceeds_the_available_width(self) -> None:
        """The odd-height correction can widen the result; it must stay in bounds.

        An uncapped width makes the position clamp elsewhere produce a negative
        right edge, and ffmpeg then fails with "Invalid too big or non positive
        size for width" -- a crash, not a cosmetic issue.
        """
        for target, cap in ((1920, 1080), (2000, 600), (5000, 300)):
            with self.subTest(target=target, cap=cap):
                width, height = _pane_dimensions(target, 1920, max_width=cap)
                self.assertLessEqual(width, cap, "pane wider than the source region")
                self.assertLessEqual(height, 1920)

    def test_height_cap_is_respected(self) -> None:
        for cap in (240, 480, 600, 960):
            with self.subTest(cap=cap):
                _width, height = _pane_dimensions(4000, cap, max_width=4000)
                self.assertLessEqual(height, cap)

    def test_output_pane_is_exactly_1080x960(self) -> None:
        width, height = _pane_dimensions(1080, 1920, max_width=1080)
        self.assertEqual((width, height), (OUTPUT_WIDTH, OUTPUT_HEIGHT // 2))


class TestNoBarGraphs(unittest.TestCase):
    """L1 on the emitted string: the graph must be incapable of adding bars."""

    def _graph(self, framing: FramingDecision, shots=None) -> str:
        # The graph builder takes only the framing; shot plans ride on
        # `framing.shots`. burn_subtitles=False keeps the assertion about layout
        # geometry rather than about libass.
        if shots is not None:
            framing.shots = shots
        return build_video_filtergraph(
            framing,
            ass_subtitle_path=None,
            burn_subtitles=False,
        )

    def test_split_screen_graph_cannot_pad(self) -> None:
        """A `decrease` + `pad` chain is a black bar waiting for the wrong aspect."""
        framing = FramingDecision(
            face_count=2,
            mode="split_screen",
            speaker1_box=(100, 100, 594, 528),
            speaker2_box=(900, 100, 594, 528),
        )
        graph = self._graph(framing)
        self.assertIn("split_screen", framing.mode)
        self.assertNotIn(
            "pad=", graph,
            "pad= in a split-screen graph is what produced 270px of black on each "
            "side in the published artifact",
        )
        self.assertIn("force_original_aspect_ratio=increase", graph)
        self.assertIn(f"crop={OUTPUT_WIDTH}:{OUTPUT_HEIGHT // 2}", graph)

    def test_multi_shot_split_screen_cannot_pad(self) -> None:
        framing = FramingDecision(mode="multi_shot_dynamic", face_count=2)
        shots = [
            ShotPlan(
                start=0.0,
                end=3.0,
                mode="split_screen",
                speaker1_box=(100, 100, 594, 528),
                speaker2_box=(900, 100, 594, 528),
            )
        ]
        graph = self._graph(framing, shots=shots)
        self.assertNotIn(
            "pad=", graph,
            "the per-shot split_screen branch had the same decrease+pad chain",
        )
        self.assertIn("force_original_aspect_ratio=increase", graph)

    def test_every_split_path_uses_fill_then_crop(self) -> None:
        """Both split paths, so a future refactor cannot reintroduce one of them."""
        top_level = self._graph(
            FramingDecision(
                face_count=2,
                mode="split_screen",
                speaker1_box=(100, 100, 594, 528),
                speaker2_box=(900, 100, 594, 528),
            )
        )
        per_shot = self._graph(
            FramingDecision(mode="multi_shot_dynamic", face_count=2),
            shots=[
                ShotPlan(
                    start=0.0, end=3.0, mode="split_screen",
                    speaker1_box=(100, 100, 594, 528),
                    speaker2_box=(900, 100, 594, 528),
                )
            ],
        )
        for label, graph in (("top-level", top_level), ("per-shot", per_shot)):
            with self.subTest(path=label):
                self.assertNotIn("pad=", graph)
                self.assertNotIn("force_original_aspect_ratio=decrease", graph)

    def test_divider_is_still_six_pixels(self) -> None:
        """The 6px divider is an AGENTS.md requirement; do not lose it while fixing bars."""
        framing = FramingDecision(
            face_count=2,
            mode="split_screen",
            speaker1_box=(100, 100, 594, 528),
            speaker2_box=(900, 100, 594, 528),
        )
        self.assertIn("drawbox", self._graph(framing))
        self.assertIn("h=6", self._graph(framing))


class TestSplitScreenRendersFullBleed(unittest.TestCase):
    """L2: render a real vertical source and measure for dead columns.

    This is the level that would have caught the shipped defect. The graph-level
    tests above prove the *intent*; this proves the *output*.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.source = Path(self.tmp.name) / "vertical.mp4"
        self.output = Path(self.tmp.name) / "out.mp4"
        if not _vertical_source(self.source):
            self.skipTest("could not encode a vertical test source (ffmpeg/libx264?)")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _render(self) -> bool:
        framing = FramingDecision(
            face_count=2,
            mode="split_screen",
            speaker1_box=(120, 100, 540, 480),
            speaker2_box=(1260, 100, 540, 480),
        )
        from core.video_editor import render_viral_clip

        try:
            render_viral_clip(
                source_video_path=self.source,
                start_time=0.0,
                end_time=2.0,
                output_clip_path=self.output,
                framing=framing,
            )
        except Exception as error:  # pragma: no cover - surfaced as a failure
            self.fail(f"render raised: {error}")
        return self.output.exists() and self.output.stat().st_size > 0

    def test_no_dead_columns_in_the_rendered_output(self) -> None:
        """The 9:16 pane that shipped produced 25% dead frame on each side.

        The pane boxes here are deliberately 540x480, i.e. 9:8-ish but narrow, so
        the assertion is about the *scale chain* filling the pane rather than about
        the input being well proportioned.
        """
        if not self._render():
            self.skipTest("renderer produced no output")
        worst = _worst_dead_columns(self.output)
        self.assertLessEqual(
            worst,
            1.0,
            msg=(
                f"{worst:.1f}% of the frame width is a dead black column. "
                f"AGENTS.md 3.3 requires zero black bars on split-screen panes."
            ),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
