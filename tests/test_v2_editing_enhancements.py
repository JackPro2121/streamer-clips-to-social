"""Test Suite for V2 Streamer Editing Enhancements.

Tests:
1. Smart profanity & shadowban keyword masking (streamers.censor)
2. Velocity hook snap-zoom (0.0s to 0.35s) injection (streamers.editing_enhancer)
3. 4-second rule pattern interrupt generation (streamers.editing_enhancer)
4. Monotonic timestamp and casing preservation
"""

import unittest
from core.transcriber import TranscriptSegment, WordTimestamp
from streamers.censor import (
    censor_word_text,
    censor_transcript_segments,
)
from streamers.editing_enhancer import (
    build_streamer_retention_plan,
)


class TestV2ProfanityCensor(unittest.TestCase):
    """Test smart masking of vulgarities and TOS triggers."""

    def test_censor_word_variations(self):
        # Exact lower
        self.assertEqual(censor_word_text("fuck"), "f*ck")
        self.assertEqual(censor_word_text("shit"), "sh*t")
        self.assertEqual(censor_word_text("bitch"), "b*tch")
        self.assertEqual(censor_word_text("nigga"), "n***a")
        self.assertEqual(censor_word_text("kill"), "k*ll")

        # Uppercase preserved
        self.assertEqual(censor_word_text("FUCK"), "F*CK")
        self.assertEqual(censor_word_text("SHIT"), "SH*T")
        self.assertEqual(censor_word_text("KILL"), "K*LL")

        # Capitalized preserved
        self.assertEqual(censor_word_text("Fucking"), "F*cking")
        self.assertEqual(censor_word_text("Bullshit"), "Bullsh*t")

    def test_punctuation_preservation(self):
        self.assertEqual(censor_word_text('"FUCKING!"'), '"F*CKING!"')
        self.assertEqual(censor_word_text("what?!"), "what?!")
        self.assertEqual(censor_word_text("shit,"), "sh*t,")

    def test_clean_words_untouched(self):
        clean_words = ["hello", "streamer", "game", "clutch", "funny", "win"]
        for w in clean_words:
            self.assertEqual(censor_word_text(w), w)

    def test_transcript_segments_censoring(self):
        segments = [
            TranscriptSegment(
                start=0.0,
                end=3.0,
                text="BRO ARE YOU FUCKING SERIOUS RIGHT NOW",
                words=[
                    WordTimestamp("BRO", 0.0, 0.4),
                    WordTimestamp("ARE", 0.4, 0.8),
                    WordTimestamp("YOU", 0.8, 1.2),
                    WordTimestamp("FUCKING", 1.2, 1.8),
                    WordTimestamp("SERIOUS", 1.8, 2.4),
                    WordTimestamp("RIGHT", 2.4, 2.7),
                    WordTimestamp("NOW", 2.7, 3.0),
                ]
            )
        ]

        censored = censor_transcript_segments(segments)
        self.assertEqual(len(censored), 1)
        self.assertIn("F*CKING", censored[0].text)
        self.assertEqual(censored[0].words[3].word, "F*CKING")
        # Ensure timestamps are 100% preserved
        self.assertEqual(censored[0].words[3].start, 1.2)
        self.assertEqual(censored[0].words[3].end, 1.8)


class TestV2EditingRetentionPlan(unittest.TestCase):
    """Test velocity hook injection and pattern interrupt generation."""

    def test_velocity_hook_injection(self):
        # Clip with no initial peak
        raw_peaks = [(10.0, 12.0)]
        raw_sfx = [(15.0, "ding")]

        enhanced_zooms, enhanced_sfx = build_streamer_retention_plan(
            clip_duration=30.0,
            peak_intervals=raw_peaks,
            sfx_cues=raw_sfx,
            inject_velocity_hook=True,
        )

        # First zoom interval must be at t=0.0s for the instant snap hook
        self.assertEqual(enhanced_zooms[0], (0.0, 0.35))
        # Initial whoosh SFX must be present at frame 0
        self.assertEqual(enhanced_sfx[0], (0.0, "whoosh"))

    def test_pattern_interrupts_fill_long_lulls(self):
        # Clip of 35 seconds with 15-second visual lull between t=5s and t=20s
        raw_peaks = [(2.0, 4.0), (22.0, 24.0)]
        raw_sfx = []

        enhanced_zooms, _ = build_streamer_retention_plan(
            clip_duration=35.0,
            peak_intervals=raw_peaks,
            sfx_cues=raw_sfx,
            inject_velocity_hook=True,
            max_static_gap_s=6.0,
        )

        # There must be an injected pattern interrupt zoom between 4s and 22s
        injected = [z for z in enhanced_zooms if 4.0 < z[0] < 22.0]
        self.assertTrue(len(injected) > 0, "Pattern interrupt was not injected during visual lull")


if __name__ == "__main__":
    unittest.main()
