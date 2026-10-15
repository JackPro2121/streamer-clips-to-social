"""Regression tests for the V2 streamer pipeline audit fixes.

Covers three defects found during the in-depth V2 audit:

  H1 — SFX cues were never shifted from the moment timeline into the
       downloaded-file timeline (hook pre-roll), so every cue rendered ~1s
       early.  Fixed by ``pipeline._remap_sfx_cues`` + passing ``moment_start``.
  H2 — ``discovery.record_failed_attempt`` was dead in production, so failing
       VODs were retried forever and successful-but-empty runs were marked
       processed.  Fixed by wiring the retry budget through the pipeline.
  H3 — a forced ``split_screen`` clobbered a *detected* ``stacked_webcam``
       layout, discarding the detected webcam/content boxes in favour of
       synthetic half-frame panes.

V1 behaviour is intentionally not exercised here.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.face_tracker import FramingDecision
from streamers.discovery import load_streamer_history
from streamers.layout_compositor import (
    StreamLayoutInfo,
    build_streamer_framing,
)
from streamers.pipeline import _remap_sfx_cues, run_streamer_pipeline

_TEST_URL = "https://www.youtube.com/watch?v=abcdefghijk"
_TEST_VIDEO_ID = "abcdefghijk"


class _FakeCutPlan:
    """Minimal stand-in for jump_cutter.CutPlan."""

    def __init__(self, applied: bool = False, shift: float = 0.0) -> None:
        self.applied = applied
        self._shift = shift

    def remap(self, t: float) -> float:
        return t - self._shift


class TestSfxCueRemap(unittest.TestCase):
    """H1 — moment-relative cues must be shifted into the file timeline."""

    def test_cues_shifted_by_segment_start(self):
        cues = [(0.0, "whoosh"), (5.0, "pop")]
        out = _remap_sfx_cues(cues, segment_start=1.0, cut_plan=_FakeCutPlan(), clip_duration=47.0)
        self.assertEqual(out, [(1.0, "whoosh"), (6.0, "pop")])

    def test_zero_segment_start_is_identity(self):
        cues = [(0.1, "whoosh"), (12.0, "ding")]
        out = _remap_sfx_cues(cues, segment_start=0.0, cut_plan=_FakeCutPlan(), clip_duration=45.0)
        self.assertEqual(out, cues)

    def test_cues_pass_through_cut_plan_after_shift(self):
        # cue 2.0 + offset 1.0 = 3.0 in file time; cut plan removes 0.5s -> 2.5
        out = _remap_sfx_cues(
            [(2.0, "pop")],
            segment_start=1.0,
            cut_plan=_FakeCutPlan(applied=True, shift=0.5),
            clip_duration=47.0,
        )
        self.assertEqual(out, [(2.5, "pop")])

    def test_cues_outside_rendered_duration_are_dropped(self):
        cues = [(1.0, "whoosh"), (40.0, "ding")]
        out = _remap_sfx_cues(cues, segment_start=1.0, cut_plan=_FakeCutPlan(), clip_duration=10.0)
        # 1.0 + 1.0 = 2.0 (kept); 40.0 + 1.0 = 41.0 > 10.0 (dropped)
        self.assertEqual(out, [(2.0, "whoosh")])


class TestRetryBudgetWiring(unittest.TestCase):
    """H2 — failed runs must feed the D10 retry budget, not pretend success."""

    def _run_with_history_file(self, hist_path: Path, **patches):
        with patch("streamers.discovery.STREAMER_HISTORY_FILE", hist_path):
            with patch.multiple("streamers.pipeline", **patches):
                return run_streamer_pipeline(
                    video_url=_TEST_URL,
                    creator_name="kai_cenat",
                    num_clips=1,
                    run_id="test_run_audit",
                )

    def test_transcript_failure_records_an_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            hist = Path(tmp) / "streamer_history.json"
            self._run_with_history_file(hist, fetch_transcript_only=lambda *a, **k: None)
            with patch("streamers.discovery.STREAMER_HISTORY_FILE", hist):
                data = load_streamer_history()
            self.assertEqual(data["attempts"].get(_TEST_VIDEO_ID), 1)
            self.assertNotIn(_TEST_VIDEO_ID, data["processed_video_ids"])

    def test_no_viral_moments_records_an_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            hist = Path(tmp) / "streamer_history.json"
            self._run_with_history_file(
                hist,
                fetch_transcript_only=lambda *a, **k: {
                    "transcript": [{"text": "hello there friends", "start": 0.0, "duration": 5.0}]
                },
                detect_streamer_viral_moments=lambda *a, **k: [],
            )
            with patch("streamers.discovery.STREAMER_HISTORY_FILE", hist):
                data = load_streamer_history()
            self.assertEqual(data["attempts"].get(_TEST_VIDEO_ID), 1)
            self.assertNotIn(_TEST_VIDEO_ID, data["processed_video_ids"])

    def test_three_failures_blacklist_the_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            hist = Path(tmp) / "streamer_history.json"
            for _ in range(3):
                self._run_with_history_file(hist, fetch_transcript_only=lambda *a, **k: None)
            with patch("streamers.discovery.STREAMER_HISTORY_FILE", hist):
                data = load_streamer_history()
            self.assertEqual(data["attempts"].get(_TEST_VIDEO_ID), 3)
            # At the ceiling the video is promoted to permanently processed.
            self.assertIn(_TEST_VIDEO_ID, data["processed_video_ids"])


class TestForcedLayoutGuards(unittest.TestCase):
    """H3 — forcing a layout must not discard a confidently detected layout."""

    _WEBCAM = (100, 100, 400, 300)
    _CONTENT = (0, 0, 1920, 1080)

    def test_forced_split_preserves_detected_stacked_webcam(self):
        with patch(
            "streamers.layout_compositor.detect_stream_layout",
            return_value=StreamLayoutInfo(
                layout_type="stacked_webcam",
                webcam_box=self._WEBCAM,
                content_box=self._CONTENT,
            ),
        ):
            framing = build_streamer_framing(
                video_path=Path("nonexistent.mp4"),
                start_time=0.0,
                end_time=10.0,
                force_layout="split_screen",
            )
        self.assertEqual(framing.mode, "split_screen")
        # Detected boxes preserved — NOT synthetic half-frame panes (0,0,960,*).
        self.assertEqual(framing.speaker1_box, self._WEBCAM)
        self.assertEqual(framing.speaker2_box, self._CONTENT)

    def test_forced_split_uses_detected_multi_person_boxes(self):
        boxes = ((0, 0, 960, 852), (960, 0, 960, 852))
        with patch(
            "streamers.layout_compositor.detect_stream_layout",
            return_value=StreamLayoutInfo(
                layout_type="multi_person_room",
                speaker1_box=boxes[0],
                speaker2_box=boxes[1],
            ),
        ):
            framing = build_streamer_framing(
                video_path=Path("nonexistent.mp4"),
                start_time=0.0,
                end_time=10.0,
                force_layout="split_screen",
            )
        self.assertEqual(framing.mode, "split_screen")
        self.assertEqual(framing.speaker1_box, boxes[0])
        self.assertEqual(framing.speaker2_box, boxes[1])

    def test_forced_split_on_single_person_still_forces_split(self):
        # Explicit force is a documented operator override: a single-person
        # source still produces synthetic split panes (legacy semantics kept).
        with patch(
            "streamers.layout_compositor.detect_stream_layout",
            return_value=StreamLayoutInfo(layout_type="fullscreen_room"),
        ):
            framing = build_streamer_framing(
                video_path=Path("nonexistent.mp4"),
                start_time=0.0,
                end_time=10.0,
                force_layout="split_screen",
            )
        self.assertEqual(framing.mode, "split_screen")
        self.assertIsNotNone(framing.speaker1_box)
        self.assertIsNotNone(framing.speaker2_box)

    def test_forced_stacked_without_boxes_falls_back_to_detected(self):
        sentinel = FramingDecision(mode="single_smooth", face_count=1)
        with patch(
            "streamers.layout_compositor.detect_stream_layout",
            return_value=StreamLayoutInfo(layout_type="fullscreen_room"),
        ), patch(
            "streamers.layout_compositor.analyze_streamer_clip_framing",
            return_value=sentinel,
        ) as fallback:
            framing = build_streamer_framing(
                video_path=Path("nonexistent.mp4"),
                start_time=0.0,
                end_time=10.0,
                force_layout="stacked_webcam",
            )
        self.assertIs(framing, sentinel)
        fallback.assert_called_once()


if __name__ == "__main__":
    unittest.main()
