"""TransNetV2 ONNX runner: windowing must match the reference implementation.

The export's parity proof measured the model against the reference padding,
100/50 windowing and middle-50 splice. If the runner uses a different scheme --
as the previous one did, averaging overlapping windows with no padding -- the
boundary timestamps stop matching what the parity proof validated.

These tests use a stub session, so they run everywhere without the 30MB model.
"""

from __future__ import annotations

import unittest

import numpy as np

from core.shot_detection import TransNetV2Detector


class _StubInput:
    def __init__(self, name, shape, type_name):
        self.name = name
        self.shape = shape
        self.type = type_name


class _StubOutput:
    def __init__(self, name):
        self.name = name


class _StubSession:
    """Returns the call index as every frame's probability, so the splice is checkable."""

    def __init__(self) -> None:
        self.calls = 0
        self.batch_shapes = []
        self._inputs = [_StubInput("frames", [1, 100, 27, 48, 3], "tensor(uint8)")]
        self._outputs = [_StubOutput("single_frame"), _StubOutput("many_hot")]

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, names, feed):
        self.batch_shapes.append(feed["frames"].shape)
        self.calls += 1
        return [np.full((1, 100, 1), float(self.calls), dtype=np.float32)]


def _frames(count: int):
    return [np.zeros((27, 48, 3), dtype=np.uint8) for _ in range(count)]


class TestWindowing(unittest.TestCase):
    def test_splice_order_and_length(self) -> None:
        session = _StubSession()
        detector = TransNetV2Detector(session, fps=30.0)
        strengths = detector._transition_strengths(_frames(220))
        # 220 frames -> head pad 25, tail pad 25+50-(220%50)=55, 300 padded,
        # windows at 0,50,100,150,200 -> five calls, each contributing 50.
        self.assertEqual(session.calls, 5)
        self.assertEqual(len(strengths), 220)
        self.assertEqual(strengths[49], 1.0)
        self.assertEqual(strengths[50], 2.0)
        self.assertEqual(strengths[199], 4.0)
        self.assertEqual(strengths[200], 5.0)
        for shape in session.batch_shapes:
            self.assertEqual(tuple(shape), (1, 100, 27, 48, 3))

    def test_exact_multiple_tail_padding(self) -> None:
        session = _StubSession()
        detector = TransNetV2Detector(session, fps=30.0)
        strengths = detector._transition_strengths(_frames(100))
        # remainder 0 -> tail pad 25; 150 padded; windows at 0 and 50.
        self.assertEqual(session.calls, 2)
        self.assertEqual(len(strengths), 100)
        self.assertEqual(strengths[0], 1.0)
        self.assertEqual(strengths[99], 2.0)

    def test_prepare_is_uint8_rgb(self) -> None:
        prepared = TransNetV2Detector._prepare(np.zeros((1080, 1920, 3), dtype=np.uint8))
        self.assertEqual(prepared.shape, (27, 48, 3))
        self.assertEqual(prepared.dtype, np.uint8)

    def test_short_sequence_returns_zeros(self) -> None:
        detector = TransNetV2Detector(_StubSession(), fps=30.0)
        self.assertEqual(detector._transition_strengths(_frames(1)), [0.0])


class TestShotsMapping(unittest.TestCase):
    def test_boundary_uses_frame_index_minus_one(self) -> None:
        detector = TransNetV2Detector(_StubSession(), fps=10.0)
        strengths = [0.0] * 100
        strengths[50] = 0.9
        detector._transition_strengths = lambda frames: strengths  # type: ignore[assignment]
        shots, cuts, confidence = detector.shots(
            _frames(100), start_sec=0.0, end_sec=10.0, min_shot_duration=0.4
        )
        self.assertEqual(cuts, 1)
        self.assertEqual(shots, [(0.0, 4.9), (4.9, 10.0)])
        self.assertAlmostEqual(confidence, 0.9, places=4)

    def test_weak_probability_is_not_a_cut(self) -> None:
        detector = TransNetV2Detector(_StubSession(), fps=10.0)
        strengths = [0.0] * 100
        strengths[50] = 0.2
        detector._transition_strengths = lambda frames: strengths  # type: ignore[assignment]
        shots, cuts, _ = detector.shots(
            _frames(100), start_sec=0.0, end_sec=10.0, min_shot_duration=0.4
        )
        self.assertEqual(cuts, 0)
        self.assertEqual(shots, [(0.0, 10.0)])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
