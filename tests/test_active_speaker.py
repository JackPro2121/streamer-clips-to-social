"""ASD-Lite decision logic: pick the face whose mouth moves with the voice.

The defect this feeds (run 37347481817, clip_2): the crop showed a face whose
mouth motion correlated with the voice at +0.09 with no lag alignment -- the
visible face was not the speaker. Composition must consult who is talking; the
decision core is pure so the talker is known by construction in these tests.

Synthetic series are non-negative on purpose: real mouth motion and audio RMS
envelopes are magnitudes.
"""

from __future__ import annotations

import unittest

import numpy as np

from core.active_speaker import (
    Track,
    pick_speaker,
    scores_from_series,
    shot_speaker_targets,
    windowed_speakers,
)


def _positive(size: int, seed: int, scale: float = 1.0) -> list[float]:
    return list(np.abs(np.random.default_rng(seed).standard_normal(size)) * scale)


def _follower(envelope: list[float], seed: int, gain: float = 3.0) -> list[float]:
    noise = np.random.default_rng(seed).standard_normal(len(envelope)) * 0.1
    return list(np.asarray(envelope) * gain + noise)


class TestScoresFromSeries(unittest.TestCase):
    def test_follower_scores_above_noise(self):
        envelope = _positive(40, 1)
        follower = _follower(envelope, 2, gain=1.5)
        stranger = _positive(40, 3, scale=0.5)
        scores = scores_from_series([follower, stranger], envelope)
        self.assertGreater(scores[0], 0.8)
        self.assertLess(abs(scores[1]), 0.5)

    def test_short_series_returns_zero(self):
        self.assertEqual(scores_from_series([[1.0, 2.0]], [1.0, 2.0]), [0.0])


class TestPickSpeaker(unittest.TestCase):
    def test_picks_the_follower(self):
        envelope = _positive(40, 4)
        follower = _follower(envelope, 5)
        stranger = _positive(40, 6, scale=0.5)
        self.assertEqual(pick_speaker([stranger, follower], envelope), 1)

    def test_returns_minus_one_when_nobody_follows(self):
        envelope = _positive(40, 7)
        self.assertEqual(
            pick_speaker([_positive(40, 8, scale=0.5), _positive(40, 9, scale=0.5)], envelope),
            -1,
        )

    def test_a_still_face_cannot_win(self):
        # A frozen "face" (photo/poster) with no mouth movement is not a
        # speaker even when its constant series would otherwise score.
        envelope = _positive(40, 10)
        still = [0.01] * 40
        follower = _follower(envelope, 11)
        self.assertEqual(pick_speaker([still, follower], envelope), 1)


class TestWindowedSpeakers(unittest.TestCase):
    def test_speaker_switch_between_windows(self):
        fps = 10.0
        window = 1.0
        first = _positive(50, 12)
        second = _positive(50, 13)
        # Face 0 talks in the first 2.5s, face 1 in the last 2.5s.
        envelope = first[:25] + second[25:]
        face0 = _follower(first, 14)[:25] + _positive(25, 15, scale=0.5)
        face1 = _positive(25, 16, scale=0.5) + _follower(second, 17)[25:]
        windows = windowed_speakers([face0, face1], envelope, fps, window)
        self.assertGreaterEqual(len(windows), 4)
        self.assertEqual(windows[0][1], 0)
        self.assertEqual(windows[-1][1], 1)

    def test_no_confidence_windows_report_minus_one(self):
        envelope = _positive(30, 18)
        windows = windowed_speakers([_positive(30, 19, scale=0.5)], envelope, 10.0, 1.0)
        self.assertTrue(windows)
        self.assertTrue(all(index == -1 for _start, index, _score in windows))


class TestShotSpeakerTargets(unittest.TestCase):
    """The integration core: shot span -> talking track -> crop centre."""

    @staticmethod
    def _track(motion: list[float], x: list[float]) -> Track:
        return Track(motion=list(motion), x=list(x))

    def test_shot_target_lands_on_the_talking_face(self):
        fps = 10.0
        envelope = _positive(80, 20)
        talker = _follower(envelope, 21)
        listener = _positive(80, 22, scale=0.5)
        # Analysis space is 480 wide; the talker sits right (x=400), which maps
        # to 1600 in a 1920-wide source. The nearest shot face box centre is
        # 1550, not the listener's 250.
        tracks = [
            self._track(listener, [80.0] * 80),
            self._track(talker, [400.0] * 80),
        ]
        boxes = [[(1400.0, 0.0, 300.0, 300.0), (100.0, 0.0, 300.0, 300.0)]]
        targets = shot_speaker_targets(
            tracks, envelope, [(0.0, 8.0)], boxes, (1920, 1080), fps
        )
        self.assertIsNotNone(targets[0])
        assert targets[0] is not None
        self.assertAlmostEqual(targets[0], 1550.0, delta=50.0)

    def test_no_confident_talker_leaves_the_shot_alone(self):
        envelope = _positive(40, 23)
        tracks = [self._track(_positive(40, 24, scale=0.5), [100.0] * 40)]
        targets = shot_speaker_targets(
            tracks, envelope, [(0.0, 4.0)], [[(0.0, 0.0, 100.0, 100.0)]],
            (1920, 1080), 10.0,
        )
        self.assertIsNone(targets[0])

    def test_absent_face_boxes_fall_back_to_the_talker_position(self):
        # Run 37466588808 had zero face boxes on every shot; giving up there
        # silently disabled the whole fix. The talker track's own position is
        # a face centre: x=200 in the 480-wide analysis maps to 800 in 1920.
        envelope = _positive(40, 25)
        talker = _follower(envelope, 26)
        tracks = [self._track(talker, [200.0] * 40)]
        targets = shot_speaker_targets(
            tracks, envelope, [(0.0, 4.0)], [[]], (1920, 1080), 10.0
        )
        self.assertIsNotNone(targets[0])
        assert targets[0] is not None
        self.assertAlmostEqual(targets[0], 800.0, delta=1.0)

    def test_vote_is_confidence_weighted_across_windows(self):
        fps = 10.0
        envelope = _positive(60, 27)
        # Two faces, both following, but track 1 only in the middle third.
        track0 = _follower(envelope, 28)
        track1 = _positive(20, 29, scale=0.5) + _follower(envelope[20:40], 30) + _positive(20, 31, scale=0.5)
        tracks = [self._track(track0, [60.0] * 60), self._track(track1, [420.0] * 60)]
        boxes = [[(100.0, 0.0, 200.0, 200.0), (1700.0, 0.0, 200.0, 200.0)]]
        targets = shot_speaker_targets(
            tracks, envelope, [(0.0, 6.0)], boxes, (1920, 1080), fps
        )
        self.assertIsNotNone(targets[0])

    def test_varying_track_lengths_do_not_collapse_windows(self):
        fps = 10.0
        envelope = _positive(50, 42)
        # Track 0: present and following for 50 frames
        track0 = _follower(envelope, 43)
        # Track 1: spurious 1-frame detection!
        track1 = [2.0]
        # Track 2: brief 10-frame face detection
        track2 = _positive(10, 44, scale=0.5)

        windows = windowed_speakers([track0, track1, track2], envelope, fps, window_s=1.0)
        # Must produce 5 windows (for 50 frames at 10 fps with 1s window)
        self.assertEqual(len(windows), 5)
        # Track 0 must win all windows despite track1 and track2 being short
        self.assertTrue(all(speaker == 0 for _start, speaker, _score in windows))

    def test_shot_target_ignores_zero_positions_and_uses_shot_window(self):
        fps = 10.0
        envelope = _positive(60, 50)
        talker = _follower(envelope, 51)
        # Talker is at x=100 in first half (0-3s), x=400 in second half (3-6s), with 0.0 on absent frames
        positions = [100.0] * 30 + [400.0] * 30
        positions[5] = 0.0  # simulated dropped frame
        positions[35] = 0.0  # simulated dropped frame

        tracks = [self._track(talker, positions)]
        # Shot 1: 0.0s - 3.0s -> should pick ~100 in analysis space -> ~400 in 1920
        # Shot 2: 3.0s - 6.0s -> should pick ~400 in analysis space -> ~1600 in 1920
        targets = shot_speaker_targets(
            tracks, envelope, [(0.0, 3.0), (3.0, 6.0)], [[], []], (1920, 1080), fps
        )
        self.assertIsNotNone(targets[0])
        self.assertIsNotNone(targets[1])
        assert targets[0] is not None and targets[1] is not None
        self.assertAlmostEqual(targets[0], 100.0 * 1920 / 480, delta=1.0)
        self.assertAlmostEqual(targets[1], 400.0 * 1920 / 480, delta=1.0)


if __name__ == "__main__":
    unittest.main()
