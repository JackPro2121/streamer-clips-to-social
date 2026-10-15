"""Tests for the Alex Hormozi kinetic subtitle aesthetic and hook badge styling.

Verifies:
1. 1-to-2 words per line pacing (with up to 3 words on fast speech >190 WPM).
2. Word-level color coding:
   - Danger / loss / risk -> &H003333FF (#FF3333)
   - Money / financial / numbers -> &H0033FF22 (#22FF33)
   - Power / authority -> &H0000D7FF (#FFD700)
   - General high-energy -> &H0000E6FF (#FFE600)
3. Hook badge duration matches creative_spec (3.0s).
4. No unrenderable emoji glyphs burned into Montserrat Black output.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.creative_spec import (
    HOOK_BADGE_DURATION_S,
    MAX_WORDS_PER_CAPTION,
    MAX_WORDS_PER_CAPTION_FAST_SPEECH,
    FAST_SPEECH_WPM_THRESHOLD,
    hex_to_ass_color,
    HIGHLIGHT_YELLOW,
    HIGHLIGHT_NEON_GREEN,
    DANGER_RED,
    POWER_GOLD,
)
from core.subtitle_generator import (
    create_styled_ass_subtitles,
    DANGER_KEYWORDS,
    MONEY_KEYWORDS,
    POWER_KEYWORDS,
    ASS_COLOR_DANGER_RED,
    ASS_COLOR_NEON_GREEN,
    ASS_COLOR_POWER_GOLD,
    ASS_COLOR_YELLOW,
)
from core.transcriber import TranscriptSegment, WordTimestamp


class TestHormoziSubtitleAesthetic(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.out_ass = Path(self.tmp.name) / "test_hormozi.ass"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_colors_match_creative_spec(self) -> None:
        self.assertEqual(ASS_COLOR_YELLOW, hex_to_ass_color(HIGHLIGHT_YELLOW))
        self.assertEqual(ASS_COLOR_NEON_GREEN, hex_to_ass_color(HIGHLIGHT_NEON_GREEN))
        self.assertEqual(ASS_COLOR_DANGER_RED, hex_to_ass_color(DANGER_RED))
        self.assertEqual(ASS_COLOR_POWER_GOLD, hex_to_ass_color(POWER_GOLD))
        self.assertEqual(HOOK_BADGE_DURATION_S, 3.0)
        self.assertEqual(MAX_WORDS_PER_CAPTION, 2)
        self.assertEqual(MAX_WORDS_PER_CAPTION_FAST_SPEECH, 3)
        self.assertEqual(FAST_SPEECH_WPM_THRESHOLD, 190)
        self.assertIn("money", MONEY_KEYWORDS)
        self.assertIn("secret", POWER_KEYWORDS)

    def test_trap_and_danger_keywords_render_red(self) -> None:
        """The audit found that 'trap' was previously missing from DANGER_KEYWORDS."""
        self.assertIn("trap", DANGER_KEYWORDS)
        self.assertIn("scared", DANGER_KEYWORDS)
        self.assertIn("broke", DANGER_KEYWORDS)

        words = [
            WordTimestamp(word="TRAP", start=1.0, end=1.5),
            WordTimestamp(word="HOUSE", start=1.5, end=2.0),
        ]
        segments = [TranscriptSegment(start=1.0, end=2.0, text="TRAP HOUSE", words=words)]
        create_styled_ass_subtitles(
            segments=segments,
            clip_start=0.0,
            clip_end=5.0,
            output_ass_path=self.out_ass,
            theme_key="hormozi",
        )
        content = self.out_ass.read_text(encoding="utf-8")
        self.assertIn(f"{{\\c{ASS_COLOR_DANGER_RED}}}TRAP", content)

    def test_money_and_currency_render_neon_green(self) -> None:
        words = [
            WordTimestamp(word="$400K", start=1.0, end=1.5),
            WordTimestamp(word="PROFIT", start=1.5, end=2.0),
        ]
        segments = [TranscriptSegment(start=1.0, end=2.0, text="$400K PROFIT", words=words)]
        create_styled_ass_subtitles(
            segments=segments,
            clip_start=0.0,
            clip_end=5.0,
            output_ass_path=self.out_ass,
            theme_key="hormozi",
        )
        content = self.out_ass.read_text(encoding="utf-8")
        self.assertIn(f"{{\\c{ASS_COLOR_NEON_GREEN}}}$400K", content)
        self.assertIn(f"{{\\c{ASS_COLOR_NEON_GREEN}}}PROFIT", content)

    def test_power_keywords_render_gold(self) -> None:
        words = [
            WordTimestamp(word="SECRET", start=1.0, end=1.5),
            WordTimestamp(word="BLUEPRINT", start=1.5, end=2.0),
        ]
        segments = [TranscriptSegment(start=1.0, end=2.0, text="SECRET BLUEPRINT", words=words)]
        create_styled_ass_subtitles(
            segments=segments,
            clip_start=0.0,
            clip_end=5.0,
            output_ass_path=self.out_ass,
            theme_key="hormozi",
        )
        content = self.out_ass.read_text(encoding="utf-8")
        self.assertIn(f"{{\\c{ASS_COLOR_POWER_GOLD}}}SECRET", content)
        self.assertIn(f"{{\\c{ASS_COLOR_POWER_GOLD}}}BLUEPRINT", content)

    def test_one_to_two_words_chunking_on_standard_speech(self) -> None:
        """Hormozi style chunks into 1 to 2 words per pop."""
        words = [
            WordTimestamp(word="THIS", start=0.5, end=0.9),
            WordTimestamp(word="ONE", start=0.9, end=1.3),
            WordTimestamp(word="DECISION", start=1.3, end=1.8),
            WordTimestamp(word="CHANGED", start=1.8, end=2.2),
            WordTimestamp(word="EVERYTHING", start=2.2, end=2.8),
        ]
        segments = [TranscriptSegment(start=0.5, end=2.8, text="THIS ONE DECISION CHANGED EVERYTHING", words=words)]
        create_styled_ass_subtitles(
            segments=segments,
            clip_start=0.0,
            clip_end=5.0,
            output_ass_path=self.out_ass,
            theme_key="hormozi",
        )
        content = self.out_ass.read_text(encoding="utf-8")
        dialogue_lines = [line for line in content.splitlines() if line.startswith("Dialogue: 0,")]
        # 5 words chunked into max 2 words per line -> 3 dialogue lines
        self.assertEqual(len(dialogue_lines), 3)

    def test_hook_badge_duration_is_three_seconds(self) -> None:
        words = [WordTimestamp(word="HELLO", start=0.5, end=1.0)]
        segments = [TranscriptSegment(start=0.5, end=1.0, text="HELLO", words=words)]
        create_styled_ass_subtitles(
            segments=segments,
            clip_start=0.0,
            clip_end=10.0,
            output_ass_path=self.out_ass,
            theme_key="hormozi",
            header_title="HOW TO GET RICH",
        )
        content = self.out_ass.read_text(encoding="utf-8")
        badge_lines = [line for line in content.splitlines() if "TopHeader" in line and "Dialogue:" in line]
        self.assertEqual(len(badge_lines), 1)
        # End timestamp should be 0:00:03.00
        self.assertIn("0:00:00.00,0:00:03.00,TopHeader", badge_lines[0])

    def test_no_raw_emoji_glyphs_burned(self) -> None:
        words = [
            WordTimestamp(word="CASH", start=0.5, end=1.0),
        ]
        segments = [TranscriptSegment(start=0.5, end=1.0, text="CASH", words=words)]
        create_styled_ass_subtitles(
            segments=segments,
            clip_start=0.0,
            clip_end=5.0,
            output_ass_path=self.out_ass,
            theme_key="hormozi",
            keyword_emojis={"cash": "💰"},
        )
        content = self.out_ass.read_text(encoding="utf-8")
        self.assertNotIn("💰", content)
        self.assertNotIn("\ufffd", content)


if __name__ == "__main__":
    unittest.main()
