"""The caption gate's face detector must be wired, not merely present.

Two failures motivated these tests, both of which let a blocking caption-on-face
defect ship as PASS:

1. **Stale shared detector state.** ``face_tracker.detect_faces_in_frame`` calls
   ``setInputSize`` with the full frame before every detect. The verifier reuses
   the same cached detector through ``get_face_detector()`` but fed a 320x320
   image without resetting the size, so YuNet returned nothing on CI while the
   identical code found faces locally. The mutation guard: remove the
   ``setInputSize`` call and ``test_input_size_is_set_before_detect`` fails.

2. **Wrong-backend silence.** ``get_face_detector`` falls back to MediaPipe and
   Haar, which have no ``detect()``. ``detect_in_frame`` caught the missing
   method and returned ``[]`` -- indistinguishable from "no faces". ``_load_yunet``
   now names the backend it cannot use.
"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from core.verification.faces import _load_yunet, detect_in_frame


class RecordingDetector:
    def __init__(self):
        self.input_sizes = []
        self.frame_shapes = []

    def setInputSize(self, size):
        self.input_sizes.append(tuple(size))

    def detect(self, frame):
        self.frame_shapes.append(tuple(frame.shape[:2]))
        return 1, None


class TestDetectInFrame(unittest.TestCase):
    def test_input_size_is_set_before_detect(self) -> None:
        detector = RecordingDetector()
        frame = np.zeros((480, 270, 3), dtype=np.uint8)
        detect_in_frame(frame, detector)
        self.assertEqual(
            detector.input_sizes,
            [(320, 320)],
            "detect_in_frame must reset the shared detector's input size",
        )
        self.assertEqual(detector.frame_shapes[0], (320, 320))

    def test_backend_without_detect_returns_empty_not_crash(self) -> None:
        class HaarLike:
            pass

        self.assertEqual(
            detect_in_frame(np.zeros((10, 10, 3), dtype=np.uint8), HaarLike()),
            [],
        )


class TestLoadDetector(unittest.TestCase):
    def test_load_yunet_rejects_backends_without_detect(self) -> None:
        with mock.patch(
            "core.face_tracker.get_face_detector", return_value=object()
        ):
            detector, reason = _load_yunet()
        self.assertIsNone(detector)
        self.assertIn("detect", reason)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
