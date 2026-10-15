"""Unit tests for smart dynamic speaker switching (OpusClip Pro style).

Verifies active speaker detection, dwell-time enforcement, interjection
filtering, and debate/split-screen transitions in core/face_tracker.py.
"""

from __future__ import annotations

import unittest
from core.face_tracker import FaceBox, _segment_two_speaker_shots
from core.creative_spec import SPEAKER_MIN_DWELL_S


class TestSmartSpeakerSwitching(unittest.TestCase):
    def setUp(self) -> None:
        self.active_x = 0
        self.active_y = 0
        self.active_w = 1920
        self.active_h = 1080
        self.target_crop_w = int(self.active_h * (9 / 16))  # 607 -> 606

        # Two stable speaker faces (Speaker 1 on left at x=350, Speaker 2 on right at x=1350)
        self.f1 = FaceBox(x=250, y=300, w=200, h=200, center_x=350, center_y=400)
        self.f2 = FaceBox(x=1250, y=300, w=200, h=200, center_x=1350, center_y=400)
        self.valid_face_samples = [[self.f1, self.f2] for _ in range(30)]
        self.two_speaker_samples = [[self.f1, self.f2] for _ in range(30)]

    def test_active_speaker_alternation(self) -> None:
        """When Speaker 1 speaks then Speaker 2 speaks, camera cuts from S1 to S2."""
        # 6.0 seconds total at 0.2s sample step (30 samples)
        # 0.0s - 3.0s: S1 active (m1=5.0, m2=0.2)
        # 3.0s - 6.0s: S2 active (m1=0.2, m2=5.0)
        samples = []
        for i in range(30):
            t = round(i * 0.2, 2)
            if t < 3.0:
                mouth = [5.0, 0.2]
            else:
                mouth = [0.2, 5.0]
            samples.append((t, [self.f1, self.f2], False, mouth))

        shots = _segment_two_speaker_shots(
            rel_s=0.0,
            rel_e=6.0,
            shot_samples=samples,
            valid_face_samples=self.valid_face_samples,
            two_speaker_samples=self.two_speaker_samples,
            target_crop_w=self.target_crop_w,
            active_x=self.active_x,
            active_y=self.active_y,
            active_w=self.active_w,
            active_h=self.active_h,
        )

        self.assertEqual(len(shots), 2)
        # First shot: Speaker 1 portrait
        self.assertEqual(shots[0].mode, "portrait_face")
        self.assertAlmostEqual(shots[0].start, 0.0)
        self.assertAlmostEqual(shots[0].end, 3.0, places=1)
        self.assertLess(shots[0].crop_x, 600)  # Centered on S1 at x=350

        # Second shot: Speaker 2 portrait
        self.assertEqual(shots[1].mode, "portrait_face")
        self.assertAlmostEqual(shots[1].start, 3.0, places=1)
        self.assertAlmostEqual(shots[1].end, 6.0)
        self.assertGreater(shots[1].crop_x, 800)  # Centered on S2 at x=1350

    def test_interjection_filtering_no_rapid_ping_pong(self) -> None:
        """Brief interjections (<= 0.45s) must not trigger camera cuts away from main speaker."""
        # Speaker 1 speaks continuously for 5.0s, with Speaker 2 interjecting for 0.3s at t=2.0s
        samples = []
        for i in range(25):
            t = round(i * 0.2, 2)
            if 2.0 <= t <= 2.2:
                mouth = [0.5, 4.5]  # S2 quick interjection (0.2s duration)
            else:
                mouth = [4.5, 0.3]  # S1 dominant
            samples.append((t, [self.f1, self.f2], False, mouth))

        shots = _segment_two_speaker_shots(
            rel_s=0.0,
            rel_e=5.0,
            shot_samples=samples,
            valid_face_samples=self.valid_face_samples,
            two_speaker_samples=self.two_speaker_samples,
            target_crop_w=self.target_crop_w,
            active_x=self.active_x,
            active_y=self.active_y,
            active_w=self.active_w,
            active_h=self.active_h,
        )

        # Interjection must be filtered out: exactly 1 continuous shot on Speaker 1
        self.assertEqual(len(shots), 1)
        self.assertEqual(shots[0].mode, "portrait_face")
        self.assertEqual(shots[0].start, 0.0)
        self.assertEqual(shots[0].end, 5.0)
        self.assertLess(shots[0].crop_x, 600)

    def test_simultaneous_debate_triggers_split_screen(self) -> None:
        """When both speakers speak simultaneously, switch to 9:8 split-screen."""
        # Both speakers have high lip motion throughout 4.0s
        samples = []
        for i in range(20):
            t = round(i * 0.2, 2)
            samples.append((t, [self.f1, self.f2], False, [4.0, 4.2]))

        shots = _segment_two_speaker_shots(
            rel_s=0.0,
            rel_e=4.0,
            shot_samples=samples,
            valid_face_samples=self.valid_face_samples,
            two_speaker_samples=self.two_speaker_samples,
            target_crop_w=self.target_crop_w,
            active_x=self.active_x,
            active_y=self.active_y,
            active_w=self.active_w,
            active_h=self.active_h,
        )

        self.assertEqual(len(shots), 1)
        self.assertEqual(shots[0].mode, "split_screen")
        self.assertIsNotNone(shots[0].speaker1_box)
        self.assertIsNotNone(shots[0].speaker2_box)
        # Split panes must maintain 9:8 aspect ratio
        s1_box = shots[0].speaker1_box
        self.assertAlmostEqual(s1_box[2] / s1_box[3], 9.0 / 8.0, places=2)

    def test_min_dwell_enforcement(self) -> None:
        """Transitions must respect minimum dwell time of at least SPEAKER_MIN_DWELL_S."""
        samples = []
        # Attempt rapid alternating speech every 0.4s
        for i in range(25):
            t = round(i * 0.2, 2)
            speaker_idx = (int(t / 0.4)) % 2
            mouth = [5.0, 0.1] if speaker_idx == 0 else [0.1, 5.0]
            samples.append((t, [self.f1, self.f2], False, mouth))

        shots = _segment_two_speaker_shots(
            rel_s=0.0,
            rel_e=5.0,
            shot_samples=samples,
            valid_face_samples=self.valid_face_samples,
            two_speaker_samples=self.two_speaker_samples,
            target_crop_w=self.target_crop_w,
            active_x=self.active_x,
            active_y=self.active_y,
            active_w=self.active_w,
            active_h=self.active_h,
        )

        for s in shots:
            duration = s.end - s.start
            self.assertGreaterEqual(
                duration,
                SPEAKER_MIN_DWELL_S - 0.05,
                f"Shot duration {duration:.2f}s is less than min dwell {SPEAKER_MIN_DWELL_S}s",
            )

    def test_zero_gap_continuity(self) -> None:
        """Consecutive shots must connect seamlessly with zero time drift or gaps."""
        samples = []
        for i in range(35):
            t = round(i * 0.2, 2)
            if t < 2.5:
                mouth = [4.0, 0.2]
            elif t < 4.5:
                mouth = [4.0, 4.0]
            else:
                mouth = [0.2, 4.0]
            samples.append((t, [self.f1, self.f2], False, mouth))

        shots = _segment_two_speaker_shots(
            rel_s=0.0,
            rel_e=7.0,
            shot_samples=samples,
            valid_face_samples=self.valid_face_samples,
            two_speaker_samples=self.two_speaker_samples,
            target_crop_w=self.target_crop_w,
            active_x=self.active_x,
            active_y=self.active_y,
            active_w=self.active_w,
            active_h=self.active_h,
        )

        self.assertGreater(len(shots), 1)
        self.assertEqual(shots[0].start, 0.0)
        self.assertEqual(shots[-1].end, 7.0)
        for i in range(len(shots) - 1):
            self.assertEqual(
                shots[i].end,
                shots[i + 1].start,
                f"Gap detected between shot {i} and {i+1}: {shots[i].end} != {shots[i+1].start}",
            )


if __name__ == "__main__":
    unittest.main()
