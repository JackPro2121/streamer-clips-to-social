"""
Unit and integration tests for Phase 1c: Shot boundary detection.

Tests the TransNetV2 detector with fallback to PySceneDetect, ensuring
that timeline partition guarantees, minimum shot durations, and model
availability boundaries are respected.
"""
from pathlib import Path
import tempfile
from typing import Any, List
import unittest
from unittest import mock
import numpy as np

from core.shot_detection import (
    ShotBoundaryResult,
    TransNetV2Detector,
    detect_shot_boundaries,
    normalise_shots,
    reset_shot_detector_cache,
)
from core import scene_classifier


class TestNormaliseShots(unittest.TestCase):
    def test_normalise_shots_empty_yields_single_partition(self) -> None:
        shots = normalise_shots([], 0.0, 10.0, min_shot_duration=0.5)
        self.assertEqual(shots, [(0.0, 10.0)])

    def test_normalise_shots_clamps_boundaries_to_window(self) -> None:
        raw = [(-2.0, 3.5), (3.5, 7.0), (7.0, 15.0)]
        shots = normalise_shots(raw, 0.0, 10.0, min_shot_duration=0.4)
        self.assertEqual(shots[0][0], 0.0)
        self.assertEqual(shots[-1][1], 10.0)
        self.assertEqual(shots[0], (0.0, 3.5))
        self.assertEqual(shots[1], (3.5, 7.0))
        self.assertEqual(shots[2], (7.0, 10.0))

    def test_normalise_shots_merges_sub_minimum_shots(self) -> None:
        raw = [(0.0, 2.0), (2.0, 2.2), (2.2, 5.0)]
        # 2.0-2.2 is 0.2s < min 0.5s -> merged into predecessor (0.0, 2.2)
        shots = normalise_shots(raw, 0.0, 5.0, min_shot_duration=0.5)
        self.assertEqual(len(shots), 2)
        self.assertEqual(shots[0], (0.0, 2.2))
        self.assertEqual(shots[1], (2.2, 5.0))

    def test_normalise_shots_merges_trailing_short_shot(self) -> None:
        raw = [(0.0, 4.8), (4.8, 5.0)]
        # Trailing 0.2s merged into predecessor
        shots = normalise_shots(raw, 0.0, 5.0, min_shot_duration=0.5)
        self.assertEqual(len(shots), 1)
        self.assertEqual(shots[0], (0.0, 5.0))

    def test_normalise_shots_strict_partition_total_span(self) -> None:
        raw = [(1.0, 2.5), (2.5, 4.0), (4.0, 7.5), (7.5, 9.0)]
        shots = normalise_shots(raw, 0.0, 10.0, min_shot_duration=0.4)
        total = sum(end - start for start, end in shots)
        self.assertAlmostEqual(total, 10.0, places=2)
        # Check continuity (no gaps)
        for i in range(len(shots) - 1):
            self.assertAlmostEqual(shots[i][1], shots[i + 1][0], places=2)


class TestTransNetV2Detector(unittest.TestCase):
    def test_try_create_returns_none_when_weights_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing_model = Path(tmp_dir) / "non_existent.onnx"
            with mock.patch("core.shot_detection.TRANSNET_MODEL_PATH", str(missing_model)):
                detector = TransNetV2Detector.try_create(fps=30.0)
                self.assertIsNone(detector)

    def test_prepare_returns_uint8_rgb_for_the_exported_graph(self) -> None:
        # The exported ONNX normalises internally; the runner must hand it the
        # same uint8 RGB the export's parity proof measured.
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame[500:600, 500:600] = 255
        prepared = TransNetV2Detector._prepare(frame)
        self.assertEqual(prepared.shape, (27, 48, 3))
        self.assertEqual(prepared.dtype, np.uint8)

    def test_transition_strengths_with_mock_session(self) -> None:
        class DummySession:
            def get_outputs(self) -> List[Any]:
                class OutputMeta:
                    def __init__(self, name: str) -> None:
                        self.name = name
                return [OutputMeta("single_frame"), OutputMeta("many_hot")]

            def get_inputs(self) -> List[Any]:
                class InputMeta:
                    name = "frames"
                return [InputMeta()]

            def run(self, output_names: Any, input_feed: Any) -> List[np.ndarray]:
                batch = list(input_feed.values())[0]
                batch_len = batch.shape[1]
                preds = np.zeros((1, batch_len, 1), dtype=np.float32)
                # Reference windowing pads 25 frames at the head, so clip frame 5
                # sits at padded index 30, inside the kept middle [25:75].
                if batch_len > 30:
                    preds[0, 30, 0] = 0.95
                return [preds]

        detector = TransNetV2Detector(session=DummySession(), fps=10.0)
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(20)]
        strengths = detector._transition_strengths(frames)
        self.assertEqual(len(strengths), 20)
        self.assertGreater(strengths[5], 0.9)

    def test_shots_splits_on_confidence_threshold(self) -> None:
        detector = TransNetV2Detector(session=mock.MagicMock(), fps=10.0)
        dummy_strengths = [0.0] * 20
        dummy_strengths[10] = 0.95
        detector._transition_strengths = mock.MagicMock(return_value=dummy_strengths)

        shots, cuts, conf = detector.shots(
            frames=[np.zeros((10, 10, 3))] * 20,
            start_sec=0.0,
            end_sec=2.0,
            min_shot_duration=0.3,
        )
        self.assertGreaterEqual(cuts, 1)
        self.assertEqual(len(shots), 2)
        self.assertGreater(conf, 0.9)


class TestDetectShotBoundaries(unittest.TestCase):
    def setUp(self) -> None:
        reset_shot_detector_cache()

    def test_fallback_to_pyscenedetect_when_model_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dummy_video = Path(tmp_dir) / "dummy.mp4"
            dummy_video.write_bytes(b"dummy")

            with mock.patch("core.shot_detection.TRANSNET_MODEL_PATH", str(Path(tmp_dir) / "absent.onnx")):
                with mock.patch("core.shot_detection._detect_with_pyscenedetect") as mock_psd:
                    mock_psd.return_value = ([(0.0, 5.0)], "scenedetect")
                    result = detect_shot_boundaries(dummy_video, 0.0, 5.0)
                    self.assertIsInstance(result, ShotBoundaryResult)
                    self.assertEqual(result.source, "scenedetect")
                    self.assertEqual(result.shots, [(0.0, 5.0)])

    def test_scene_classifier_detect_clip_shots_delegation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dummy_video = Path(tmp_dir) / "dummy.mp4"
            dummy_video.write_bytes(b"dummy")

            with mock.patch("core.shot_detection.detect_shot_boundaries") as mock_det:
                mock_det.return_value = ShotBoundaryResult(
                    shots=[(0.0, 3.0), (3.0, 6.0)],
                    source="transnetv2",
                    transition_count=1,
                    mean_confidence=0.88,
                )
                shots = scene_classifier.detect_clip_shots(dummy_video, 0.0, 6.0)
                self.assertEqual(shots, [(0.0, 3.0), (3.0, 6.0)])
