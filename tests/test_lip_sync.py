"""Lip-sync gate arithmetic and reporting, without torch.

The heavy SyncNet stack belongs to the ML tier; these tests pin the parts that
must be true in every environment: the banding, the fail-closed reporting when
the tier or checkpoints are missing, and the metric mapping from the pipeline's
return tuple.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from core.verification import lip_sync


class TestQualityBands(unittest.TestCase):
    def test_bands(self) -> None:
        self.assertEqual(lip_sync.quality_for(5.0, 5.0), "GOOD")
        self.assertEqual(lip_sync.quality_for(2.5, 8.0), "FAIR")
        self.assertEqual(lip_sync.quality_for(1.5, 13.0), "POOR")


class TestAnalyse(unittest.TestCase):
    def test_missing_file_is_not_a_crash(self) -> None:
        report = lip_sync.analyse(Path("does_not_exist.mp4"))
        self.assertFalse(report.ok)
        self.assertIn("file not found", report.reason)

    def test_missing_tier_or_checkpoints_is_reported_not_passed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            empty_models = Path(directory)
            candidate = empty_models / "clip.mp4"
            candidate.write_bytes(b"not a real video")
            pipeline, reason = lip_sync._load_pipeline(empty_models)
        if pipeline is not None:
            self.fail("no pipeline can exist without checkpoints")
        self.assertTrue(reason, "an unrunnable check must name what is missing")

    def test_pipeline_tuple_maps_onto_the_report(self) -> None:
        class FakePipeline:
            def inference(self, path):
                return (
                    [-1, 2],
                    [4.2, 3.1],
                    [5.5, 4.9],
                    4.2,
                    4.9,
                    "[]",
                    True,
                )

        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "clip.mp4"
            candidate.write_bytes(b"not a real video")
            with mock.patch.object(lip_sync, "_load_pipeline", return_value=(FakePipeline(), "")):
                report = lip_sync.analyse(candidate)

        self.assertTrue(report.ok)
        self.assertTrue(report.measurable)
        self.assertEqual(report.lse_c, 4.2)
        self.assertEqual(report.lse_d, 4.9)
        self.assertEqual(report.quality, "GOOD")
        self.assertEqual(len(report.tracks), 2)
        self.assertEqual(report.tracks[0]["offset_frames"], -1)
        self.assertEqual(report.tracks[1]["offset_ms"], 80.0)

    def test_no_face_track_is_unmeasurable_but_not_a_pass(self) -> None:
        class NoFacePipeline:
            def inference(self, path):
                return ([], [], [], 0.0, 0.0, "", False)

        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "clip.mp4"
            candidate.write_bytes(b"not a real video")
            with mock.patch.object(lip_sync, "_load_pipeline", return_value=(NoFacePipeline(), "")):
                report = lip_sync.analyse(candidate)

        self.assertTrue(report.ok)
        self.assertFalse(report.measurable)
        self.assertIn("no face track", report.reason)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
