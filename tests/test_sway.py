"""Sway metric arithmetic. The render path is covered by the motion tests."""

from __future__ import annotations

import unittest

import numpy as np

from core.verification import sway


def _triangle(fps: float, seconds: float, freq: float, amplitude: float) -> np.ndarray:
    """The renderer's triangle construction: 0.6366*asin(sin(w*t))."""
    t = np.arange(int(fps * seconds)) / fps
    return amplitude * 0.6366 * np.arcsin(np.sin(2 * np.pi * freq * t))


class TestTrajectorySway(unittest.TestCase):
    def test_policy_band_triangle_is_detected(self) -> None:
        report = sway.analyse_trajectory(_triangle(15, 20, 0.5, 8.0).tolist())
        self.assertGreater(report.p2p_px, 3.0, "an 8px policy triangle must register")
        self.assertAlmostEqual(report.freq_hz, 0.5, delta=0.06)

    def test_bigger_amplitude_reads_bigger(self) -> None:
        small = sway.analyse_trajectory(_triangle(15, 20, 0.5, 3.0).tolist())
        large = sway.analyse_trajectory(_triangle(15, 20, 0.5, 12.0).tolist())
        self.assertGreater(large.p2p_px, small.p2p_px)

    def test_content_frequency_is_not_policy_sway(self) -> None:
        """A speaker's own 0.3Hz sway must not read as the 0.5Hz policy oscillator.

        Measured on a real run: 53px of content sway at 0.33Hz was blocked as
        camera shake, which cost a publishable clip.
        """
        report = sway.analyse_trajectory(_triangle(15, 20, 0.3, 20.0).tolist())
        self.assertLess(report.p2p_px, sway.SWAY_BLOCK_PX)
        self.assertGreater(report.broadband_p2p_px, 5.0)

    def test_short_trajectory_is_reported_not_measured(self) -> None:
        report = sway.analyse_trajectory([0.0] * 10)
        self.assertIn("too short", report.reason)
        self.assertEqual(report.p2p_px, 0.0)

    def test_camera_cut_splits_segments(self) -> None:
        trajectory = _triangle(15, 20, 0.5, 8.0)
        trajectory[150:] += 100.0
        report = sway.analyse_trajectory(trajectory.tolist())
        self.assertGreaterEqual(report.segments, 2)

    def test_steady_pan_is_not_sway(self) -> None:
        """A linear pan is removed before the band is read, or tracking reads as shake."""
        fps, seconds = 15, 20
        t = np.arange(int(fps * seconds)) / fps
        report = sway.analyse_trajectory((3.0 * t).tolist())
        self.assertLess(report.p2p_px, 2.0, f"a pure pan read as {report.p2p_px:.1f}px sway")


class TestCalibratedBoundaries(unittest.TestCase):
    """Thresholds sit between human-accepted content and the gross defect.

    Calibration source: the fixture manifest's human verdicts (accepted content
    reaches 39px broadband on run 37202390593's wrongly blocked clip #2; the
    only gross-sway exemplar reads 126px) plus the corpus table in
    ``sway.py``'s docstring. These tests pin the warn/block split with synthetic
    trajectories; the real corpus is pinned in ``test_visual_regressions``.
    """

    def test_policy_warn_level_does_not_block(self) -> None:
        report = sway.analyse_trajectory(_triangle(15, 20, 0.5, 27.0).tolist())
        self.assertGreater(report.p2p_px, sway.SWAY_WARN_PX)
        self.assertLessEqual(report.p2p_px, sway.SWAY_BLOCK_PX)

    def test_policy_regression_class_blocks(self) -> None:
        report = sway.analyse_trajectory(_triangle(15, 20, 0.5, 34.0).tolist())
        self.assertGreater(report.p2p_px, sway.SWAY_BLOCK_PX)

    def test_broadband_warn_level_does_not_block(self) -> None:
        report = sway.analyse_trajectory(_triangle(15, 20, 0.3, 55.0).tolist())
        self.assertGreater(report.broadband_p2p_px, sway.BROADBAND_WARN_PX)
        self.assertLessEqual(report.broadband_p2p_px, sway.BROADBAND_BLOCK_PX)

    def test_gross_broadband_class_blocks(self) -> None:
        report = sway.analyse_trajectory(_triangle(15, 20, 0.3, 70.0).tolist())
        self.assertGreater(report.broadband_p2p_px, sway.BROADBAND_BLOCK_PX)


class TestTransitionRejection(unittest.TestCase):
    """Pans and short-segment residues are not sway.

    Run 37208970507 lost 4 of 5 clips to segments that were pure crop pans
    (``multi_shot_dynamic`` reframes between speakers) or to short-segment
    bin-1 residue: a 2.13s segment's 0.47Hz peak is one cycle of the segment
    itself. Both classes read as 43-105px "sway" before this.
    """

    def test_eased_crop_pan_is_not_sway(self) -> None:
        t = np.arange(int(15 * 2.13)) / 15
        eased = (225.0 * (1 - np.exp(-t / 0.45))).tolist()
        report = sway.analyse_trajectory(eased)
        self.assertLess(report.p2p_px, 3.0)
        self.assertLess(report.broadband_p2p_px, 3.0)

    def test_directed_long_pan_is_skipped(self) -> None:
        pan = (200.0 * np.arange(int(15 * 8)) / int(15 * 8)).tolist()
        report = sway.analyse_trajectory(pan)
        self.assertLess(report.p2p_px, 3.0)
        self.assertLess(report.broadband_p2p_px, 3.0)

    def test_short_segment_step_residue_needs_two_cycles(self) -> None:
        step = [float(v) for v in list(range(21)) + [0.0] * 9]
        report = sway.analyse_trajectory(step)
        self.assertLess(report.p2p_px, 3.0)
        self.assertLess(report.broadband_p2p_px, 3.0)

    def test_sustained_gross_oscillation_still_blocks(self) -> None:
        report = sway.analyse_trajectory(_triangle(15, 20, 0.3, 70.0).tolist())
        self.assertGreater(report.broadband_p2p_px, sway.BROADBAND_BLOCK_PX)


class TestReportShape(unittest.TestCase):
    def test_missing_file_is_not_a_crash(self) -> None:
        from pathlib import Path

        report = sway.analyse(Path("does_not_exist.mp4"))
        self.assertFalse(report.ok)
        self.assertIn("file not found", report.reason)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
