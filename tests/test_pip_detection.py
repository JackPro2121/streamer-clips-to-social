"""Tests for picture-in-picture host detection on slide-with-camera shots."""
import unittest

from core.face_tracker import (
    PIP_MAX_HOST_AREA_RATIO,
    PIP_MIN_PERSISTENCE,
    FaceBox,
    _picture_in_picture_host_box,
)


def _face(x, y, w, h):
    return FaceBox(x=x, y=y, w=w, h=h, center_x=x + w // 2, center_y=y + h // 2)


class TestPictureInPictureHostBox(unittest.TestCase):
    def test_no_faces_returns_none(self):
        self.assertIsNone(_picture_in_picture_host_box([]))
        self.assertIsNone(_picture_in_picture_host_box([[], []]))

    def test_persistent_single_face_yields_the_median_box(self):
        samples = [[_face(1200, 120, 200, 200)] for _ in range(5)]
        self.assertEqual(_picture_in_picture_host_box(samples), (1200, 120, 200, 200))

    def test_outlier_detection_cannot_move_the_inset(self):
        samples = [[_face(1200, 120, 200, 200)] for _ in range(4)]
        samples.append([_face(300, 40, 900, 900)])
        box = _picture_in_picture_host_box(samples)
        self.assertIsNotNone(box)
        self.assertEqual(box, (1200, 120, 200, 200))

    def test_transient_face_is_rejected(self):
        samples = [[] for _ in range(9)]
        samples.append([_face(1200, 120, 200, 200)])
        self.assertIsNone(_picture_in_picture_host_box(samples))

    def test_two_faces_in_a_frame_is_rejected(self):
        samples = [[_face(100, 100, 200, 200), _face(1400, 100, 200, 200)]
                   for _ in range(4)]
        self.assertIsNone(_picture_in_picture_host_box(samples))

    def test_tiny_face_is_rejected(self):
        samples = [[_face(10, 10, 4, 4)] for _ in range(5)]
        self.assertIsNone(_picture_in_picture_host_box(samples))

    def test_persistence_threshold_is_half(self):
        self.assertEqual(PIP_MIN_PERSISTENCE, 0.5)
        samples = [[] for _ in range(2)]
        samples += [[_face(1200, 120, 200, 200)] for _ in range(2)]
        # exactly half -> accepted
        self.assertIsNotNone(_picture_in_picture_host_box(samples))
        samples2 = [[] for _ in range(3)]
        samples2 += [[_face(1200, 120, 200, 200)] for _ in range(2)]
        # below half -> rejected
        self.assertIsNone(_picture_in_picture_host_box(samples2))

    def test_area_budget_is_small_enough_to_be_an_inset(self):
        # A 1920x1080 frame: 12% is 248832 px2. A typical corner camera inset
        # (about 400x400) must qualify; a real co-presenter (700x700) must not.
        frame_area = 1920 * 1080
        self.assertLessEqual(400 * 400, frame_area * PIP_MAX_HOST_AREA_RATIO)
        self.assertGreater(700 * 700, frame_area * PIP_MAX_HOST_AREA_RATIO)


if __name__ == "__main__":
    unittest.main()
