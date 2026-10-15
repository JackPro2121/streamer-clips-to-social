"""P1 Jump-Cut Engine: dead-air cuts from word timestamps, purely.

Research basis: dead-air removal is the industry-standard editing upgrade
(TimeBolt/SavvyCut/Jump Cutter) and pacing is retention (50-60% drop-off in
the first 3s). The engine must never ship a clip below the 30s contract floor,
so that guard has its own test.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.jump_cutter import build_cut_plan

HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


class TestCutPlan(unittest.TestCase):
    def test_long_gap_is_cut_with_padding(self):
        words = [(0.0, 1.0), (3.0, 4.0)]
        plan = build_cut_plan(words, 4.0, min_duration_s=1.0)
        self.assertTrue(plan.applied)
        self.assertEqual(len(plan.keep_segments), 2)
        self.assertAlmostEqual(plan.keep_segments[0][0], 0.0)
        self.assertAlmostEqual(plan.keep_segments[0][1], 1.08, places=2)
        self.assertAlmostEqual(plan.keep_segments[1][0], 2.92, places=2)
        self.assertAlmostEqual(plan.keep_segments[1][1], 4.0, places=2)
        self.assertAlmostEqual(plan.removed_s, 1.84, places=2)
        self.assertAlmostEqual(plan.duration_s, 2.16, places=2)

    def test_small_gaps_are_speech_rhythm_not_cuts(self):
        words = [(0.0, 1.0), (1.3, 2.0), (2.2, 3.0)]
        plan = build_cut_plan(words, 3.0, min_duration_s=1.0)
        self.assertFalse(plan.applied)
        self.assertEqual(plan.keep_segments, [(0.0, 3.0)])
        self.assertEqual(plan.removed_s, 0.0)

    def test_floor_guard_refuses_a_short_result(self):
        words = [(0.0, 1.0), (3.0, 4.0), (10.0, 11.0)]
        plan = build_cut_plan(words, 31.0)
        self.assertFalse(plan.applied, "cutting to 3.4s must be refused (floor 30s)")
        self.assertEqual(plan.keep_segments, [(0.0, 31.0)])

    def test_no_words_is_identity(self):
        plan = build_cut_plan([], 45.0)
        self.assertFalse(plan.applied)
        self.assertEqual(plan.keep_segments, [(0.0, 45.0)])

    def test_head_and_tail_dead_air_are_removed(self):
        words = [(2.0, 40.0)]
        plan = build_cut_plan(words, 45.0, min_duration_s=1.0)
        self.assertTrue(plan.applied)
        self.assertAlmostEqual(plan.keep_segments[0][0], 1.92, places=2)
        self.assertAlmostEqual(plan.keep_segments[0][1], 40.08, places=2)
        self.assertAlmostEqual(plan.removed_s, 6.84, places=2)

    def test_head_and_tail_shorter_than_the_threshold_are_kept(self):
        # 0.32s of silence at each edge is rhythm, not dead air: identity.
        words = [(0.4, 44.6)]
        plan = build_cut_plan(words, 45.0, min_duration_s=1.0)
        self.assertFalse(plan.applied, "0.4s edges are rhythm, not dead air")
        self.assertEqual(plan.keep_segments, [(0.0, 45.0)])


class TestRemap(unittest.TestCase):
    def test_remap_shifts_after_the_cut(self):
        plan = build_cut_plan([(0.0, 1.0), (3.0, 4.0)], 4.0, min_duration_s=1.0)
        self.assertAlmostEqual(plan.remap(0.5), 0.5, places=3)
        self.assertAlmostEqual(plan.remap(3.5), 1.66, places=2)
        self.assertAlmostEqual(plan.remap(4.0), 2.16, places=2)

    def test_remap_snaps_removed_gaps_forward(self):
        plan = build_cut_plan([(0.0, 1.0), (3.0, 4.0)], 4.0, min_duration_s=1.0)
        # 2.0s sits inside the removed gap; it snaps to the next segment start.
        self.assertAlmostEqual(plan.remap(2.0), 1.08, places=2)

    def test_remap_of_identity_is_a_no_op(self):
        plan = build_cut_plan([(0.0, 1.0), (1.3, 2.9)], 3.0, min_duration_s=1.0)
        self.assertFalse(plan.applied)
        for moment in (0.0, 1.15, 2.9):
            self.assertAlmostEqual(plan.remap(moment), moment, places=6)


@unittest.skipUnless(HAS_FFMPEG, "ffmpeg required")
class TestCompactMedia(unittest.TestCase):
    """End-to-end proof of the cut mechanics on real media.

    A 4s source carries two tone bursts (centre 0.7s and 3.325s). The plan
    removes 2.14s of dead air; the compacted file must contain exactly the two
    bursts, each at its ``plan.remap()`` position - the same call the caption
    layer will use.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmp.name)
        self.source = self.workspace / "source.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=10:duration=4",
                "-f", "lavfi", "-i", "sine=frequency=880:duration=0.4",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=0.25",
                "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
                "-filter_complex",
                "[1:a]adelay=500|500[b1];[2:a]adelay=3200|3200[b2];"
                "[3:a]atrim=duration=4[base];"
                "[base][b1][b2]amix=inputs=3:duration=first:normalize=0[a]",
                "-map", "0:v", "-map", "[a]", "-t", "4",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                str(self.source),
            ],
            capture_output=True,
            check=True,
            timeout=180,
        )

    def tearDown(self):
        self.tmp.cleanup()

    def _burst_centres(self, path: Path) -> list[float]:
        raw = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-i", str(path),
                "-ac", "1", "-ar", "16000", "-f", "s16le", "-",
            ],
            capture_output=True,
        ).stdout
        samples = (
            np.frombuffer(raw[: len(raw) // 2 * 2], dtype="<i2").astype(np.float32)
            / 32768.0
        )
        hop = 320  # 20ms
        rms = np.asarray(
            [
                float(np.sqrt(np.mean(samples[i:i + hop] ** 2)))
                for i in range(0, max(1, len(samples) - hop), hop)
            ]
        )
        active = rms > max(0.01, float(rms.max()) * 0.3)
        centres: list[float] = []
        run_start = None
        for index, on in enumerate(active):
            if on and run_start is None:
                run_start = index
            if not on and run_start is not None:
                centres.append((run_start + index - 1) / 2 * 0.02)
                run_start = None
        if run_start is not None:
            centres.append((run_start + len(active) - 1) / 2 * 0.02)
        return centres

    def test_source_offset_shifts_the_window_into_the_file(self):
        # Same file, but the clip window starts 0.5s in: clip-relative words
        # (0.0-0.4) and (2.7-2.95) must compact the FILE's 0.5-0.9 and
        # 3.2-3.45 bursts, landing at remap() positions in the output.
        from core.jump_cutter import compact_media

        plan = build_cut_plan([(0.0, 0.4), (2.7, 2.95)], 3.5, min_duration_s=1.0)
        self.assertTrue(plan.applied)
        output = self.workspace / "compact_offset.mp4"
        self.assertTrue(compact_media(self.source, plan, output, source_offset_s=0.5))
        centres = self._burst_centres(output)
        self.assertEqual(len(centres), 2, f"expected two bursts, got {centres}")
        self.assertAlmostEqual(centres[0], plan.remap(0.2), delta=0.12)
        self.assertAlmostEqual(centres[1], plan.remap(2.825), delta=0.12)

    def test_compacted_media_places_every_burst_at_its_remapped_time(self):
        from core.jump_cutter import compact_media

        source_centres = self._burst_centres(self.source)
        self.assertEqual(len(source_centres), 2, source_centres)
        plan = build_cut_plan([(0.5, 0.9), (3.2, 3.45)], 4.0, min_duration_s=1.0)
        self.assertTrue(plan.applied)
        output = self.workspace / "compact.mp4"
        self.assertTrue(compact_media(self.source, plan, output))

        centres = self._burst_centres(output)
        self.assertEqual(len(centres), 2, f"expected exactly two bursts, got {centres}")
        for original, compacted in zip(source_centres, centres):
            self.assertAlmostEqual(
                compacted, plan.remap(original), delta=0.12,
                msg=f"burst at {original:.2f}s moved to {compacted:.2f}s",
            )


@unittest.skipUnless(HAS_FFMPEG, "ffmpeg required")
class TestPrepareCompactedClip(unittest.TestCase):
    """The one call the render path will make: source + words -> compacted."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmp.name)
        self.source = self.workspace / "source.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=10:duration=4",
                "-f", "lavfi", "-i", "sine=frequency=880:duration=4",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                str(self.source),
            ],
            capture_output=True,
            check=True,
            timeout=180,
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_prepare_returns_compacted_media_and_remapped_words(self):
        from core.jump_cutter import prepare_compacted_clip

        words = [(0.5, 0.9), (3.2, 3.45)]
        prepared = prepare_compacted_clip(
            self.source, words, 4.0, self.workspace, min_duration_s=1.0
        )
        self.assertIsNotNone(prepared)
        assert prepared is not None
        path, remapped, duration = prepared
        self.assertTrue(path.exists())
        self.assertAlmostEqual(duration, 1.86, places=2)
        self.assertEqual(len(remapped), 2)
        self.assertAlmostEqual(remapped[0][0], 0.5, places=2)
        self.assertAlmostEqual(remapped[1][0], 1.06, places=2)
        self.assertAlmostEqual(remapped[1][1], 1.31, places=2)
        # The rendered clip must fit the contract floor and the cut plan.
        self.assertGreaterEqual(duration, 1.0)

    def test_no_cut_needed_returns_none(self):
        from core.jump_cutter import prepare_compacted_clip

        prepared = prepare_compacted_clip(
            self.source, [(0.0, 3.9)], 4.0, self.workspace, min_duration_s=1.0
        )
        self.assertIsNone(prepared)

    def test_floor_guard_returns_none(self):
        from core.jump_cutter import prepare_compacted_clip

        prepared = prepare_compacted_clip(
            self.source, [(0.5, 0.9), (3.2, 3.45)], 4.0, self.workspace
        )
        self.assertIsNone(prepared, "4s clip cannot afford cuts under the 30s floor")


if __name__ == "__main__":
    unittest.main()
