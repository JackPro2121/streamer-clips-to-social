"""P3 caption emphasis: numbers, money and power words hit harder.

Research: Submagic's retention edge is animated captions with emphasis; a line
that emphasises everything emphasises nothing, so the rules are deliberately
narrow.
"""

from __future__ import annotations

import unittest

from core.caption_emphasis import (
    EMPHASIS_POP_TAGS,
    emphasised_line,
    is_emphasis_word,
    mark_emphasis,
)


class TestIsEmphasisWord(unittest.TestCase):
    def test_numbers_and_money(self):
        for token in ("$40,000", "90%", "3", "$5", "1,200", "$1.5"):
            self.assertTrue(is_emphasis_word(token), token)

    def test_power_words_case_insensitive_and_punctuated(self):
        for token in ("Never", "BROKE,", "wealth.", "quit!", "Million"):
            self.assertTrue(is_emphasis_word(token), token)

    def test_acronyms_hit_but_sentence_lead_and_single_letters_do_not(self):
        self.assertTrue(is_emphasis_word("IRS"))
        self.assertTrue(is_emphasis_word("ROI,"))
        self.assertFalse(is_emphasis_word("I"))
        self.assertFalse(is_emphasis_word("A"))

    def test_ordinary_words_do_not(self):
        for token in ("the", "and", "because", "really", "going", "So"):
            self.assertFalse(is_emphasis_word(token), token)

    def test_empty_and_symbols_do_not(self):
        for token in ("", "---", "..."):
            self.assertFalse(is_emphasis_word(token), token)


class TestMarkEmphasis(unittest.TestCase):
    def test_flags_align_with_the_word_list(self):
        words = ["I", "was", "$40,000", "in", "debt,", "and", "nobody", "knew"]
        self.assertEqual(
            mark_emphasis(words),
            [False, False, True, False, True, False, True, False],
        )

    def test_preview_helper_marks_exactly_the_flags(self):
        line = emphasised_line(["I", "was", "$40,000", "in", "debt"])
        self.assertEqual(line, "I was [$40,000] in [debt]")


class TestPopTags(unittest.TestCase):
    def test_pop_tags_scale_and_settle(self):
        self.assertIn(r"\fscx112", EMPHASIS_POP_TAGS)
        self.assertIn(r"\t(0,90,\fscx100", EMPHASIS_POP_TAGS)


if __name__ == "__main__":
    unittest.main()
