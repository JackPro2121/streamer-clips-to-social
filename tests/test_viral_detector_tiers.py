"""Tier order in ``detect_viral_moments``: Ollama Cloud is the primary model.

Why: three consecutive runs (2026-10-03/04) had every Gemini model return 503;
detection fell through to the semantic fallback each time. Ollama Cloud
(gemma4:31b) now answers the same prompt from dedicated capacity behind our own
key, so it is Tier 1 and Gemini is the first fallback. These tests pin the
order, the short-circuit behaviour, and the final semantic fallback.
"""

from __future__ import annotations

import unittest
from contextlib import ExitStack
from unittest import mock

from core.transcriber import TranscriptSegment, WordTimestamp
from core.viral_detector import detect_viral_moments


def _segments():
    return [
        TranscriptSegment(
            start=0.0,
            end=8.0,
            text="We saved a lot of money with this one trick.",
            words=[WordTimestamp(word="We", start=0.0, end=0.4)],
        ),
        TranscriptSegment(
            start=8.0,
            end=16.0,
            text="That is how the wealthy keep their taxes low.",
            words=[WordTimestamp(word="That", start=8.0, end=8.4)],
        ),
    ]


def _detect(responses, keys=None, parsed_ok=True):
    from core import viral_detector as vd

    keys = keys or {
        "OLLAMA_API_KEY": "ollama-key",
        "GEMINI_API_KEY": "gemini-key",
        "GROQ_API_KEY": "groq-key",
        "OPENROUTER_API_KEY": "openrouter-key",
    }
    mocks = {
        "ollama": mock.Mock(return_value=responses.get("ollama")),
        "gemini": mock.Mock(return_value=responses.get("gemini")),
        "groq": mock.Mock(return_value=responses.get("groq")),
        "openrouter": mock.Mock(return_value=responses.get("openrouter")),
        "parse": mock.Mock(return_value=[object()] if parsed_ok else []),
        "fallback": mock.Mock(return_value=["SEMANTIC"]),
    }
    with mock.patch.object(vd, "OLLAMA_API_KEY", keys["OLLAMA_API_KEY"]), mock.patch.object(
        vd, "GEMINI_API_KEY", keys["GEMINI_API_KEY"]
    ), mock.patch.object(vd, "GROQ_API_KEY", keys["GROQ_API_KEY"]), mock.patch.object(
        vd, "OPENROUTER_API_KEY", keys["OPENROUTER_API_KEY"]
    ), mock.patch.object(vd, "query_ollama_cloud_models", mocks["ollama"]), mock.patch.object(
        vd, "query_gemini_models", mocks["gemini"]
    ), mock.patch.object(vd, "query_groq_free_models", mocks["groq"]), mock.patch.object(
        vd, "query_openrouter_free_models", mocks["openrouter"]
    ), mock.patch.object(vd, "parse_clips_json", mocks["parse"]), mock.patch.object(
        vd, "fallback_rule_based_detector", mocks["fallback"]
    ):
        result = detect_viral_moments(_segments(), num_clips=2)
    return result, mocks


class TestTierOrder(unittest.TestCase):
    def test_gemini_is_primary_and_short_circuits_everything(self) -> None:
        result, mocks = _detect({"gemini": "GEMINI_RAW"})
        mocks["gemini"].assert_called_once()
        mocks["ollama"].assert_not_called()
        mocks["groq"].assert_not_called()
        mocks["openrouter"].assert_not_called()
        mocks["fallback"].assert_not_called()
        mocks["parse"].assert_called_once()
        self.assertEqual(mocks["parse"].call_args.args[0], "GEMINI_RAW")
        self.assertEqual(result, mocks["parse"].return_value)

    def test_ollama_is_the_first_fallback(self) -> None:
        result, mocks = _detect({"gemini": None, "ollama": "OLLAMA_RAW"})
        mocks["gemini"].assert_called_once()
        mocks["ollama"].assert_called_once()
        mocks["groq"].assert_not_called()
        mocks["openrouter"].assert_not_called()
        self.assertEqual(mocks["parse"].call_args.args[0], "OLLAMA_RAW")
        self.assertEqual(result, mocks["parse"].return_value)

    def test_groq_then_openrouter_follow(self) -> None:
        _, mocks = _detect({"gemini": None, "ollama": None, "groq": "GROQ_RAW"})
        mocks["openrouter"].assert_not_called()
        self.assertEqual(mocks["parse"].call_args.args[0], "GROQ_RAW")

        _, mocks = _detect(
            {"gemini": None, "ollama": None, "groq": None, "openrouter": "OPENROUTER_RAW"}
        )
        mocks["openrouter"].assert_called_once()
        self.assertEqual(mocks["parse"].call_args.args[0], "OPENROUTER_RAW")

    def test_semantic_fallback_when_every_ai_tier_fails(self) -> None:
        result, mocks = _detect(
            {"gemini": None, "ollama": None, "groq": None, "openrouter": None}
        )
        mocks["fallback"].assert_called_once()
        self.assertEqual(result, ["SEMANTIC"])

    def test_gemini_is_skipped_without_a_key(self) -> None:
        keys = {
            "OLLAMA_API_KEY": "ollama-key",
            "GEMINI_API_KEY": "",
            "GROQ_API_KEY": "",
            "OPENROUTER_API_KEY": "",
        }
        result, mocks = _detect({"ollama": "OLLAMA_RAW"}, keys=keys)
        mocks["gemini"].assert_not_called()
        mocks["ollama"].assert_called_once()
        self.assertEqual(result, mocks["parse"].return_value)


class TestFallbackCaption(unittest.TestCase):
    """The no-LLM caption must read like a summary, not a template.

    A scheduled run published "SAM NEW HOUSE WAS BORN / SAM NEW HOUSE WAS BORN /
    What are your thoughts on this? Let us know below." - the derived title
    repeated twice around a generic CTA. The fallback now uses the opening
    spoken sentences.
    """

    def _caption_segments(self):
        return [
            TranscriptSegment(
                start=0.0,
                end=20.0,
                text="Money mistakes cost more than fees.",
                words=[
                    WordTimestamp(word="Money", start=0.0, end=0.5),
                    WordTimestamp(word="fees.", start=19.5, end=20.0),
                ],
            ),
            TranscriptSegment(
                start=20.0,
                end=45.0,
                text="But holding cash during a crisis is how fortunes are made.",
                words=[
                    WordTimestamp(word="But", start=20.0, end=20.4),
                    WordTimestamp(word="made.", start=44.5, end=45.0),
                ],
            ),
            TranscriptSegment(
                start=45.0,
                end=60.0,
                text="Patience pays.",
                words=[
                    WordTimestamp(word="Patience", start=45.0, end=45.5),
                    WordTimestamp(word="pays.", start=59.5, end=60.0),
                ],
            ),
        ]

    def test_caption_is_transcript_derived_and_not_generic(self) -> None:
        from core.viral_detector import fallback_rule_based_detector

        candidates = fallback_rule_based_detector(self._caption_segments(), num_clips=1)
        self.assertTrue(candidates, "the 40s window must produce a candidate")
        self.assertEqual(candidates[0].origin, "backfill")
        caption = candidates[0].social_caption
        self.assertNotIn("What are your thoughts", caption)
        self.assertIn("holding cash during a crisis", caption)
        self.assertLessEqual(len(caption), 240)


class TestGeminiAttemptBudget(unittest.TestCase):
    """P2: one Gemini decision call is bounded in requests and in wall clock.

    Real run 37301857585 took 54 minutes end to end: 4 keys x 4 models x up to
    3 attempts each with a 60s per-request timeout and no ceiling. The budget
    keeps the fast key rotation (which fixes 429s) but caps the worst case; the
    pipeline then falls through to Ollama/Groq/semantic.
    """

    def _patches(self, vd, ladder, keys):
        return [
            mock.patch.object(vd, "GEMINI_MODEL_LADDER", ladder),
            mock.patch.object(vd, "GEMINI_API_KEYS", keys),
            mock.patch.object(vd, "HAS_NEW_GENAI", True),
            mock.patch.object(vd, "genai", mock.MagicMock()),
            mock.patch.object(vd, "genai_types", mock.MagicMock()),
            mock.patch.object(vd.time, "sleep", lambda _seconds: None),
        ]

    def test_transient_failures_stop_at_the_request_ceiling(self):
        from core import viral_detector as vd

        fake_client = mock.MagicMock()
        fake_client.models.generate_content.side_effect = RuntimeError("503 UNAVAILABLE")
        with ExitStack() as stack:
            for patch in self._patches(vd, ["m1", "m2", "m3", "m4"], ["k2", "k3", "k4"]):
                stack.enter_context(patch)
            vd.genai.Client.return_value = fake_client
            result = vd.query_gemini_models("prompt", "primary")
        self.assertIsNone(result)
        self.assertEqual(
            fake_client.models.generate_content.call_count,
            vd.GEMINI_MAX_REQUESTS_PER_CALL,
        )

    def test_wall_clock_cap_stops_even_fast_failures(self):
        from core import viral_detector as vd

        clock = {"now": 0.0}

        def advancing_monotonic():
            clock["now"] += 45.0
            return clock["now"]

        fake_client = mock.MagicMock()
        fake_client.models.generate_content.side_effect = RuntimeError("503 UNAVAILABLE")
        with ExitStack() as stack:
            for patch in self._patches(vd, ["m1", "m2", "m3", "m4"], ["k2", "k3", "k4"]):
                stack.enter_context(patch)
            stack.enter_context(
                mock.patch.object(vd.time, "monotonic", side_effect=advancing_monotonic)
            )
            vd.genai.Client.return_value = fake_client
            result = vd.query_gemini_models("prompt", "primary")
        self.assertIsNone(result)
        self.assertLess(fake_client.models.generate_content.call_count, 4)

    def test_success_before_the_budget_is_returned_unchanged(self):
        from core import viral_detector as vd

        fake_client = mock.MagicMock()
        fake_client.models.generate_content.return_value.text = "x" * 40
        with ExitStack() as stack:
            for patch in self._patches(vd, ["m1", "m2"], ["k2"]):
                stack.enter_context(patch)
            vd.genai.Client.return_value = fake_client
            result = vd.query_gemini_models("prompt", "primary")
        self.assertEqual(result, "x" * 40)
        self.assertEqual(fake_client.models.generate_content.call_count, 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
