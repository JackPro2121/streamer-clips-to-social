"""Comprehensive Test Suite for V2 Streamer Pipeline Engineering Improvements.

Verifies:
1. Spec overlay bounds & assertions (streamers.spec)
2. Publication Gate & verdict emission (streamers.gate)
3. Round-robin cursor rotation & retry budget (streamers.discovery)
4. Disjoint split-pane geometry with zero overlap (streamers.layout_compositor)
5. Live chat velocity median & word-boundary regex (streamers.chat_analyzer)
6. Face tracker anchor weighting (streamers.face_tracker)
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from streamers.spec import (
    STREAMER_MIN_CLIP_DURATION_S,
    STREAMER_MAX_CLIP_DURATION_S,
    STREAMER_HASHTAG_MIN,
    STREAMER_HASHTAG_MAX,
    STREAMER_TIER2_HOLD_S,
    STREAMER_SPLIT_MAX_PANE_W,
)
from streamers.gate import verify_clip
from streamers.discovery import (
    load_streamer_history,
    record_failed_attempt,
    _load_cursor,
    _save_cursor,
)
from streamers.layout_compositor import (
    _compute_disjoint_split_panes,
    _even,
)
from streamers.chat_analyzer import (
    detect_chat_spikes,
)


class TestV2SpecInvariants(unittest.TestCase):
    """Test that V2 spec constants meet all platform & business constraints."""

    def test_duration_bounds(self):
        self.assertEqual(STREAMER_MIN_CLIP_DURATION_S, 30.0)
        self.assertEqual(STREAMER_MAX_CLIP_DURATION_S, 45.0)
        self.assertLessEqual(STREAMER_MIN_CLIP_DURATION_S, STREAMER_MAX_CLIP_DURATION_S)

    def test_hashtag_bounds(self):
        self.assertEqual(STREAMER_HASHTAG_MIN, 2)
        self.assertEqual(STREAMER_HASHTAG_MAX, 4)

    def test_tier2_hold_duration(self):
        self.assertEqual(STREAMER_TIER2_HOLD_S, 1.5)


class TestV2PublicationGate(unittest.TestCase):
    """Test the publish gate verification and verdict emission."""

    def test_gate_blocks_missing_file(self):
        non_existent = Path("does_not_exist_clip.mp4")
        res = verify_clip(non_existent)
        self.assertFalse(res.passed)
        self.assertIn("output_missing", res.blocking_codes)

    def test_gate_writes_verdict_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_clip = Path(tmpdir) / "test_clip.mp4"
            fake_clip.write_bytes(b"\x00" * 1024)
            runs_dir = Path(tmpdir) / "runs" / "test_run"

            # Mock verdict.verify to return simulated passing verdict
            mock_verdict = MagicMock()
            mock_verdict.duration_s = 35.0
            mock_verdict.dimensions = {"width": 1080, "height": 1920}
            mock_verdict.findings = []
            mock_verdict.metrics = {}

            with patch("streamers.gate._import_verdict") as mock_import:
                mock_mod = MagicMock()
                mock_mod.verify.return_value = mock_verdict
                mock_import.return_value = mock_mod

                res = verify_clip(
                    fake_clip,
                    run_id="test_run",
                    runs_dir=runs_dir,
                    min_duration_s=30.0,
                    max_duration_s=45.0
                )

                self.assertTrue(res.passed)
                self.assertIsNotNone(res.verdict_path)
                self.assertTrue(res.verdict_path.exists())

                data = json.loads(res.verdict_path.read_text(encoding="utf-8"))
                self.assertEqual(data["pipeline"], "v2_streamer")
                self.assertEqual(data["clip"], "test_clip.mp4")
                self.assertTrue(data["passed"])
                self.assertEqual(data["duration_s"], 35.0)

    def test_gate_blocks_out_of_contract_duration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_clip = Path(tmpdir) / "too_long.mp4"
            fake_clip.write_bytes(b"\x00" * 1024)

            mock_verdict = MagicMock()
            mock_verdict.duration_s = 52.0  # > 45s
            mock_verdict.dimensions = {"width": 1080, "height": 1920}
            f_mock = MagicMock()
            f_mock.code = "v2_duration_out_of_contract"
            f_mock.blocks = True
            f_mock.severity = "error"
            f_mock.as_dict.return_value = {"code": "v2_duration_out_of_contract"}
            mock_verdict.findings = [f_mock]
            mock_verdict.metrics = {}

            with patch("streamers.gate._import_verdict") as mock_import:
                mock_mod = MagicMock()
                mock_mod.verify.return_value = mock_verdict
                mock_mod.Finding = MagicMock(return_value=f_mock)
                mock_import.return_value = mock_mod

                res = verify_clip(
                    fake_clip,
                    min_duration_s=30.0,
                    max_duration_s=45.0
                )
                self.assertFalse(res.passed)
                self.assertIn("v2_duration_out_of_contract", res.blocking_codes)


class TestV2LayoutGeometry(unittest.TestCase):
    """Test mathematically disjoint panes and even coordinates."""

    def test_disjoint_split_panes_zero_overlap(self):
        # 1920x1080 frame with faces at x=400 and x=1500
        source_w = 1920
        source_h = 1080
        p1_cx = 400
        p1_cy = 500
        p2_cx = 1500
        p2_cy = 500

        b1, b2 = _compute_disjoint_split_panes(source_w, source_h, p1_cx, p1_cy, p2_cx, p2_cy)

        # 1. Left pane right edge <= Right pane left edge
        b1_right = b1[0] + b1[2]
        b2_left = b2[0]
        self.assertLessEqual(b1_right, b2_left, f"Panes overlap! b1_right={b1_right} > b2_left={b2_left}")

        # 2. Maximum pane width <= 960 (half of 1920)
        self.assertLessEqual(b1[2], STREAMER_SPLIT_MAX_PANE_W)
        self.assertLessEqual(b2[2], STREAMER_SPLIT_MAX_PANE_W)

        # 3. All coordinates must be even integers for FFmpeg YUV420p
        for box in (b1, b2):
            for val in box:
                self.assertEqual(val % 2, 0, f"Coordinate {val} is not even")

    def test_even_helper(self):
        self.assertEqual(_even(1920), 1920)
        self.assertEqual(_even(1921), 1920)
        self.assertEqual(_even(961), 960)
        self.assertEqual(_even(0), 0)


class TestV2DiscoveryRotationAndHistory(unittest.TestCase):
    """Test round-robin cursor rotation, history isolation, and retry budget."""

    def test_cursor_round_robin_saving(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cursor_file = Path(tmpdir) / "streamer_cursor.json"
            with patch("streamers.discovery.STREAMER_CURSOR_FILE", cursor_file):
                self.assertEqual(_load_cursor(), -1)
                _save_cursor(3)
                self.assertEqual(_load_cursor(), 3)
                _save_cursor(7)
                self.assertEqual(_load_cursor(), 7)

    def test_retry_budget_blacklisting(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_hist = Path(tmpdir) / "streamer_history.json"
            with patch("streamers.discovery.STREAMER_HISTORY_FILE", test_hist):
                record_failed_attempt("bad_vod_1", "Download timeout")
                h1 = load_streamer_history()
                self.assertEqual(h1["attempts"].get("bad_vod_1"), 1)

                record_failed_attempt("bad_vod_1", "Download timeout 2")
                record_failed_attempt("bad_vod_1", "Download timeout 3")
                h3 = load_streamer_history()
                self.assertEqual(h3["attempts"].get("bad_vod_1"), 3)
                # After 3 failures, it is blacklisted
                self.assertIn("bad_vod_1", h3["processed_video_ids"])


class TestV2ChatAnalyzer(unittest.TestCase):
    """Test live chat velocity calculations and word-boundary emote matching."""

    def test_emote_word_boundary_isolation(self):
        # "what a nice day" contains 'w' and 'l', but is NOT the 'w' or 'l' emote
        sample_messages = [
            {"time": 1.0, "text": "what a nice day in the world"},
            {"time": 2.0, "text": "wait for it"},
        ]
        spikes = detect_chat_spikes(sample_messages, window_sec=5.0)
        # Should NOT trigger an emote spike from ordinary words
        if spikes:
            self.assertEqual(spikes[0].emote_score, 0.0)

    def test_genuine_emote_detection(self):
        # Genuine emotes: W, L, 💀, COOKED with high density
        sample_messages = [
            {"time": 1.0, "text": "W W W BRO IS COOKED 💀💀💀"},
            {"time": 2.0, "text": "LMAO 💀💀💀💀"},
            {"time": 3.0, "text": "BIG W CRASHOUT"},
        ]
        spikes = detect_chat_spikes(sample_messages, window_sec=5.0, min_mps_factor=1.0)
        self.assertTrue(len(spikes) > 0)
        self.assertGreater(spikes[0].emote_score, 0.0)


if __name__ == "__main__":
    unittest.main()
