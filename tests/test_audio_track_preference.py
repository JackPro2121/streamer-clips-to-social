"""The audio-track selector must prefer the ORIGINAL track over YouTube auto-dubs.

Research (yt-dlp issues #11753 / #11834, r/youtubedl): a dubbed track carries
``format_note`` "dubbed-auto" and often ``language=en`` too, so language-first
ordering fetches the dub with mismatched lips. ``format_note*=original`` is the
reliable marker; the selector keeps old fallbacks afterward.
"""

from __future__ import annotations

import unittest

from core import downloader

try:
    from yt_dlp.utils import match_filter_func

    HAS_YTDLP = True
except ImportError:  # pragma: no cover
    HAS_YTDLP = False


class TestAudioTrackPreference(unittest.TestCase):
    def test_original_preference_comes_before_language_only(self) -> None:
        selector = downloader._AUDIO_EN_PREF
        self.assertIn("[format_note*=original]", selector)
        self.assertLess(
            selector.index("[format_note*=original]"),
            selector.index("[language=en-US]"),
            "original-track filter must outrank language-only filters",
        )

    def test_dubbed_exclusion_appears_before_the_language_only_fallbacks(self) -> None:
        selector = downloader._AUDIO_EN_PREF
        self.assertIn("[format_note!*=dubbed]", selector)
        self.assertLess(
            selector.index("[format_note!*=dubbed]"),
            selector.index("[format_id=140]"),
        )

    @unittest.skipUnless(HAS_YTDLP, "yt-dlp required to validate selector syntax")
    def test_every_alternative_parses(self) -> None:
        for alternative in downloader._AUDIO_EN_PREF.split("/"):
            with self.subTest(alternative=alternative):
                match_filter_func(alternative)  # raises on invalid syntax


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
