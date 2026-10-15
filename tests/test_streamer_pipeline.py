import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from streamers.discovery import (
    STREAMER_CHANNELS, extract_video_id_from_url, load_streamer_history,
    record_streamer_history
)
from streamers.viral_detector import (
    parse_streamer_clips_json, _heuristic_streamer_detector
)
from core.transcriber import TranscriptSegment


class TestStreamerPipeline(unittest.TestCase):
    def test_streamer_channels_configured(self):
        """Top US streamers should be registered with channel URLs and keywords."""
        expected = ["kai_cenat", "ishowspeed", "jynxzi", "caseoh", "xqc", "adin_ross"]
        for streamer in expected:
            self.assertIn(streamer, STREAMER_CHANNELS)
            self.assertTrue(len(STREAMER_CHANNELS[streamer]["channel_urls"]) > 0)
            self.assertTrue(len(STREAMER_CHANNELS[streamer]["keywords"]) > 0)

    def test_extract_video_id(self):
        """URL parser should extract standard 11-char YouTube IDs."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        self.assertEqual(extract_video_id_from_url(url), "dQw4w9WgXcQ")
        live_url = "https://www.youtube.com/live/12345678901"
        self.assertEqual(extract_video_id_from_url(live_url), "12345678901")

    def test_streamer_history_isolation(self):
        """Streamer history should persist without touching main history file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_hist = Path(tmpdir) / "streamer_history.json"
            with patch("streamers.discovery.STREAMER_HISTORY_FILE", test_hist):
                record_streamer_history("test_video_123", "Test Title", "Kai Cenat")
                history = load_streamer_history()
                self.assertIn("test_video_123", history)

    def test_parse_streamer_clips_json(self):
        """JSON parser should extract candidates and format titles in uppercase without emojis."""
        raw_json = json.dumps({
            "clips": [
                {
                    "title": "🔥 KAI GOES CRAZY ON STREAM!!",
                    "start_time": 10.0,
                    "end_time": 40.0,
                    "duration": 30.0,
                    "viral_score": 95,
                    "hook_reason": "High emotional spike",
                    "social_caption": "Bro was stunned. #shorts",
                    "hashtags": ["#streamer", "#twitch"],
                    "streamer_energy": "rage",
                    "sfx_cues": [[0.1, "whoosh"], [28.0, "ding"]]
                }
            ]
        })
        dummy_segments = [TranscriptSegment(start=0.0, end=60.0, text="hello world bro what are you doing", words=[])]
        parsed = parse_streamer_clips_json(raw_json, dummy_segments, num_clips=1)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0].title, "🔥 KAI GOES CRAZY ON STREAM!!")
        self.assertEqual(parsed[0].streamer_energy, "rage")
        self.assertEqual(len(parsed[0].sfx_cues), 2)
        self.assertTrue(2 <= len(parsed[0].hashtags) <= 4)

    def test_hashtag_bounds_enforced(self):
        """Hashtags must strictly be between 2 and 4 tags, never more than 4."""
        dummy_segments = [TranscriptSegment(start=0.0, end=60.0, text="test", words=[])]
        
        # Test case with 7 hashtags -> must be capped at 4
        json_many = json.dumps({
            "clips": [{
                "title": "Nah Kai really crashed out",
                "start_time": 10.0,
                "end_time": 40.0,
                "hashtags": ["#tag1", "#tag2", "#tag3", "#tag4", "#tag5", "#tag6", "#tag7"]
            }]
        })
        parsed_many = parse_streamer_clips_json(json_many, dummy_segments, num_clips=1)
        self.assertEqual(len(parsed_many[0].hashtags), 4)

        # Test case with 1 hashtag -> must have at least 2 tags
        json_few = json.dumps({
            "clips": [{
                "title": "Nah Kai really crashed out",
                "start_time": 10.0,
                "end_time": 40.0,
                "hashtags": ["#onlyone"]
            }]
        })
        parsed_few = parse_streamer_clips_json(json_few, dummy_segments, num_clips=1)
        self.assertTrue(2 <= len(parsed_few[0].hashtags) <= 4)

    def test_heuristic_streamer_detector(self):
        """Heuristic detector should prioritize shouting and exclamation marks."""
        segments = [
            TranscriptSegment(start=0.0, end=10.0, text="normal quiet conversation right here.", words=[]),
            TranscriptSegment(start=15.0, end=45.0, text="BRO WHAT ARE YOU DOING?! NO WAY! CHAT HE IS COOKED! LET'S GOOO!", words=[]),
            TranscriptSegment(start=50.0, end=60.0, text="okay back to regular chat.", words=[]),
        ]
        candidates = _heuristic_streamer_detector(segments, num_clips=1)
        self.assertTrue(len(candidates) >= 1)
        self.assertGreaterEqual(candidates[0].viral_score, 70)

    def test_dead_air_remapping_synchronizes_captions(self):
        """When dead-air silence is cut, words must shift onto the compacted timeline."""
        from core.jump_cutter import build_cut_plan
        from core.transcriber import WordTimestamp

        # Segment with speech at 0-1s, 2s silence (1-3s), speech at 3-4s
        words = [
            WordTimestamp(word="hello", start=0.0, end=1.0),
            WordTimestamp(word="world", start=3.0, end=4.0),
        ]
        p_words = [(w.start, w.end) for w in words]
        cut_plan = build_cut_plan(p_words, clip_duration_s=4.0, min_duration_s=1.0)
        self.assertTrue(cut_plan.applied)
        self.assertAlmostEqual(cut_plan.removed_s, 1.84, places=2)

        # Before cut: 'world' was at 3.0s
        # After cut: 'world' must be shifted to ~1.16s (3.0 - 1.84)
        remapped_start = cut_plan.remap(3.0)
        self.assertAlmostEqual(remapped_start, 3.0 - cut_plan.removed_s, places=2)

    def test_multi_person_split_screen_framing(self):
        """Multi-person room framing must produce split_screen mode with 2 speakers and ZERO blurry presentation_slide."""
        from streamers.layout_compositor import build_streamer_framing
        import numpy as np
        import cv2

        with tempfile.TemporaryDirectory() as tmpdir:
            test_vid = Path(tmpdir) / "collab.mp4"
            # Create a 2-second 1280x720 dummy video
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(test_vid), fourcc, 10.0, (1280, 720))
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            for _ in range(20):
                out.write(frame)
            out.release()

            framing = build_streamer_framing(
                video_path=test_vid,
                start_time=0.0,
                end_time=2.0,
                streamer_name="Kai Cenat",
                force_layout="multi_person_room"
            )

            self.assertEqual(framing.mode, "split_screen")
            self.assertIsNotNone(framing.speaker1_box)
            self.assertIsNotNone(framing.speaker2_box)
            self.assertNotEqual(framing.mode, "presentation_slide")
            # Verify 9:8 ratio of panes
            s1_w, s1_h = framing.speaker1_box[2], framing.speaker1_box[3]
            self.assertAlmostEqual(s1_w / s1_h, 9.0 / 8.0, delta=0.05)

    def test_tiktok_meme_box_sanitizes_emojis(self):
        """Top TikTok meme box must strip emojis and astral unicode in ASS to prevent '▯' tofu rectangles."""
        from core.subtitle_generator import create_styled_ass_subtitles
        from core.transcriber import TranscriptSegment, WordTimestamp

        with tempfile.TemporaryDirectory() as tmpdir:
            ass_out = Path(tmpdir) / "test.ass"
            segments = [
                TranscriptSegment(
                    start=0.0,
                    end=5.0,
                    text="what is going on bro",
                    words=[WordTimestamp(word="what", start=0.0, end=1.0)]
                )
            ]
            raw_title_with_emojis = "Kai realized MrBeast is a giant 💀 😫 \ufe0f"
            create_styled_ass_subtitles(
                segments=segments,
                clip_start=0.0,
                clip_end=5.0,
                output_ass_path=ass_out,
                theme_key="tiktok_streamer",
                top_meme_title=raw_title_with_emojis
            )

            ass_content = ass_out.read_text(encoding="utf-8")
            self.assertIn("Style: TikTokMemeBox", ass_content)
            # Find the dialogue line for TikTokMemeBox
            meme_lines = [line for line in ass_content.splitlines() if "TikTokMemeBox" in line and line.startswith("Dialogue:")]
            self.assertTrue(len(meme_lines) >= 1)
            # Emojis must be completely stripped from the ASS dialogue text
            self.assertNotIn("💀", meme_lines[0])
            self.assertNotIn("😫", meme_lines[0])
            self.assertIn("Kai realized MrBeast is a giant", meme_lines[0].replace(r"\N", " "))


if __name__ == "__main__":
    unittest.main()

