"""Regression tests for the second V2 audit round (P0/P1/P2 remediation).

Each test names the finding it pins, so the mapping survives refactors:

    P0-1  rendered clips overran the 30-45s contract by the download padding
    P0-2  production_report.json was silently skipped (AttributeError on gate.__dict__)
    P0-3  LLM titles became filenames with no allowlist
    P1-1  zones[0] (Groq/Ollama fallback) was the EARLIEST window, not the best
    P1-2  an unavailable visual check published as if it had passed
    P1-3  the publisher retried ambiguous transport errors (duplicate posts)
    P1-4  the V2 duration bounds depended on the shared engine's env namespace
    P1-5  an unset/typo'd channel list broadcast to every connected channel
    P1-6  an unknown --streamer silently clipped the whole roster
    P1-7  nothing ever deleted anything (unbounded disk growth)
    P2-3  the vertical-source branch failed its own disjointness invariant
    P2-4  a failed audio analysis was indistinguishable from "no peaks"
    P2-7  spec invariants used `assert` (stripped by `python -O`)
    P2-8  V2 imports private names from the shared engine
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.transcriber import TranscriptSegment, WordTimestamp  # noqa: E402
from streamers import gate as gate_mod  # noqa: E402
from streamers import paths as paths_mod  # noqa: E402
from streamers import retention as retention_mod  # noqa: E402
from streamers import spec as spec_mod  # noqa: E402
from streamers.gate import GateResult, verify_clip  # noqa: E402


# ---------------------------------------------------------------------------
# P0-3 — untrusted titles must never become paths
# ---------------------------------------------------------------------------
class TestUntrustedTitleNaming(unittest.TestCase):
    def test_slug_removes_path_separators_and_traversal(self):
        for title in [
            "a/b",
            "a\\b",
            "../../etc/passwd",
            "C:\\Windows\\system32",
            "..",
            "....//....//x",
        ]:
            slug = paths_mod.slugify_title(title)
            self.assertNotIn("/", slug, title)
            self.assertNotIn("\\", slug, title)
            self.assertNotIn("..", slug, title)

    def test_slug_is_windows_reserved_char_safe(self):
        slug = paths_mod.slugify_title('RANKED #1 / Kai vs Speed: PART 2 | "final" <a>?*')
        for ch in '<>:"/\\|?*':
            self.assertNotIn(ch, slug)

    def test_slug_is_never_empty(self):
        for title in ["", "   ", "💀💀💀", "مرحبا", "???", "-_-_"]:
            self.assertTrue(paths_mod.slugify_title(title), repr(title))

    def test_slug_is_deterministic_and_bounded(self):
        first = paths_mod.slugify_title("x" * 500)
        second = paths_mod.slugify_title("x" * 500)
        self.assertEqual(first, second)
        self.assertLessEqual(len(first), 40)

    def test_safe_output_name_disambiguates_titles_that_slug_identically(self):
        # Both slug to "Kai_wins" but are different raw titles: the fingerprint
        # must keep them from colliding on disk.
        one = paths_mod.safe_output_name("clip_1", "Kai / wins", ".mp4")
        two = paths_mod.safe_output_name("clip_1", "Kai \\ wins", ".mp4")
        self.assertNotEqual(one, two)
        self.assertTrue(one.endswith(".mp4"))

    def test_assert_within_allows_children_and_blocks_escapes(self):
        base = Path("clips")
        self.assertEqual(
            paths_mod.assert_within(base, base / "ok.mp4"), base / "ok.mp4"
        )
        for escape in [Path("clips/../escape.mp4"), Path("/etc/passwd"), Path("other/x.mp4")]:
            with self.assertRaises(ValueError):
                paths_mod.assert_within(base, escape)


# ---------------------------------------------------------------------------
# P1-2 — the publish gate must fail closed
# ---------------------------------------------------------------------------
class TestGateFailClosedPolicy(unittest.TestCase):
    @staticmethod
    def _mocked_verdict_module(findings=None) -> MagicMock:
        mock_verdict = MagicMock()
        mock_verdict.duration_s = 35.0
        mock_verdict.dimensions = {"width": 1080, "height": 1920}
        mock_verdict.findings = list(findings or [])
        mock_verdict.metrics = {}
        mock_verdict.warnings = []
        mock_mod = MagicMock()
        mock_mod.verify.return_value = mock_verdict
        return mock_mod

    def _fake_clip(self, tmp: str) -> Path:
        clip = Path(tmp) / "clip.mp4"
        clip.write_bytes(b"\x00" * 64)
        return clip

    def test_unavailable_visual_check_blocks_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            clip = self._fake_clip(tmp)
            with patch(
                "streamers.gate._import_verdict",
                return_value=self._mocked_verdict_module(),
            ), patch(
                "streamers.gate._visual_black_margin_check",
                return_value=gate_mod._visual_unavailable("cv2/numpy import failed: boom"),
            ):
                res = verify_clip(clip, runs_dir=Path(tmp) / "runs")

            self.assertFalse(res.passed, "an unverifiable clip must not publish")
            self.assertIn("v2_visual_check_unavailable", res.blocking_codes)
            self.assertTrue(
                any(w.startswith("v2_visual_check_unavailable:") for w in res.warnings)
            )

    def test_policy_override_is_explicit_and_only_a_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            clip = self._fake_clip(tmp)
            with patch(
                "streamers.gate._import_verdict",
                return_value=self._mocked_verdict_module(),
            ), patch("streamers.gate.STRICT_VISUAL_CHECKS", False), patch(
                "streamers.gate._visual_black_margin_check",
                return_value=gate_mod._visual_unavailable("boom"),
            ):
                res = verify_clip(clip, runs_dir=Path(tmp) / "runs")

            self.assertTrue(res.passed)
            self.assertNotIn("v2_visual_check_unavailable", res.blocking_codes)
            self.assertTrue(
                any(w.startswith("v2_visual_check_unavailable:") for w in res.warnings)
            )

    def test_available_and_clean_visual_check_passes_with_no_findings(self):
        clean = {
            "available": True,
            "passed": True,
            "worst_margin_pct": 1.0,
            "worst_side": "top",
            "sampled_frames": 12,
        }
        with tempfile.TemporaryDirectory() as tmp:
            clip = self._fake_clip(tmp)
            with patch(
                "streamers.gate._import_verdict",
                return_value=self._mocked_verdict_module(),
            ), patch("streamers.gate._visual_black_margin_check", return_value=clean):
                res = verify_clip(clip, runs_dir=Path(tmp) / "runs")

            self.assertTrue(res.passed)
            self.assertEqual(res.blocking_codes, [])

    def test_analyzers_report_a_reason_instead_of_a_bare_zero(self):
        # The three "could not run" helpers must stay distinguishable.
        self.assertEqual(gate_mod._visual_unavailable("why")["reason"], "why")
        self.assertEqual(gate_mod._face_unavailable("why")["reason"], "why")
        self.assertEqual(gate_mod._caption_unavailable("why")["reason"], "why")
        for helper in (
            gate_mod._visual_unavailable,
            gate_mod._face_unavailable,
            gate_mod._caption_unavailable,
        ):
            result = helper("why")
            self.assertFalse(result["available"])
            self.assertIn("warning", result)


# ---------------------------------------------------------------------------
# P0-2 — the production report must actually be written
# ---------------------------------------------------------------------------
class TestProductionReportVisualOutcome(unittest.TestCase):
    @staticmethod
    def _gate(blocking=(), warnings=()) -> GateResult:
        return GateResult(
            path="clip.mp4", blocking_codes=list(blocking), warnings=list(warnings)
        )

    def test_missing_gate_is_none(self):
        from streamers.pipeline import _visual_outcome

        self.assertIsNone(_visual_outcome(None))

    def test_unavailable_check_is_none_not_false(self):
        from streamers.pipeline import _visual_outcome

        gate = self._gate(blocking=["v2_visual_check_unavailable"])
        self.assertIsNone(_visual_outcome(gate))

    def test_unavailable_check_in_warnings_is_none(self):
        from streamers.pipeline import _visual_outcome

        gate = self._gate(warnings=["v2_visual_check_unavailable:cv2 missing"])
        self.assertIsNone(_visual_outcome(gate))

    def test_black_margin_failure_is_false(self):
        from streamers.pipeline import _visual_outcome

        self.assertFalse(_visual_outcome(self._gate(blocking=["v2_black_frame_margin"])))

    def test_clean_gate_is_true(self):
        from streamers.pipeline import _visual_outcome

        self.assertTrue(_visual_outcome(self._gate()))

    def test_report_shape_is_serialisable_for_a_failed_clip(self):
        """The report must serialise even when a clip never reached the gate."""
        from streamers.pipeline import _ClipResult

        failed = _ClipResult(clip_index=1, candidate=MagicMock(title="T"), error="boom")
        self.assertIsNone(failed.gate)
        self.assertIsNone(failed.rendered)
        self.assertIsNone(failed.thumb_path)
        # The old code did `r.gate.__dict__.get(...)` unconditionally, which
        # raised AttributeError on exactly this object.
        json.dumps({"visual": None, "error": failed.error})


# ---------------------------------------------------------------------------
# P1-4 / P2-7 — the V2 overlay owns its bounds, and says so loudly
# ---------------------------------------------------------------------------
class TestSpecValidation(unittest.TestCase):
    def test_validate_spec_accepts_the_defaults(self):
        spec_mod.validate_spec()

    def test_validate_spec_raises_valueerror_not_assertionerror(self):
        # `assert` disappears under `python -O`; a real raise does not.
        with patch.object(spec_mod, "STREAMER_MAX_CLIP_DURATION_S", 10_000.0):
            with self.assertRaises(ValueError) as ctx:
                spec_mod.validate_spec()
        self.assertIn("looser than", str(ctx.exception))

    def test_validate_spec_reports_inverted_window(self):
        with patch.object(spec_mod, "STREAMER_MIN_CLIP_DURATION_S", 90.0), patch.object(
            spec_mod, "STREAMER_MAX_CLIP_DURATION_S", 45.0
        ):
            with self.assertRaises(ValueError) as ctx:
                spec_mod.validate_spec()
        self.assertIn("inverted", str(ctx.exception))

    def test_strict_visual_checks_defaults_to_on(self):
        self.assertTrue(spec_mod.STRICT_VISUAL_CHECKS)


class TestClipDurationContract(unittest.TestCase):
    """P0-1 / P1-4 — every produced candidate lands inside 30-45s."""

    @staticmethod
    def _segments(end_s: float = 1200.0):
        return [
            TranscriptSegment(
                start=0.0,
                end=end_s,
                text="bro what",
                words=[WordTimestamp("bro", 0.0, 1.0), WordTimestamp("what", 1.0, 2.0)],
            )
        ]

    @staticmethod
    def _raw(start: float, end: float) -> str:
        return json.dumps(
            {
                "clips": [
                    {
                        "start_time": start,
                        "end_time": end,
                        "title": "BRO WHAT",
                        "viral_score": 90,
                        "hook_reason": "rage",
                        "social_caption": "insane",
                        "hashtags": ["#streamer", "#clips"],
                    }
                ]
            }
        )

    def test_oversized_window_is_clamped_after_snapping(self):
        from streamers import viral_detector as vd

        with patch.object(vd, "_normalize_clip_window", return_value=(100.0, 260.0)):
            out = vd.parse_streamer_clips_json(self._raw(100.0, 200.0), self._segments(), 1)

        self.assertTrue(out)
        self.assertLessEqual(
            out[0].duration, vd.STREAMER_MAX_CLIP_DURATION_S + 0.01, out[0].duration
        )

    def test_window_below_the_v2_minimum_is_grown(self):
        # The shared normalizer only guarantees core.config's MIN (30s). If the V2
        # minimum is raised, the window must still be grown to it — previously
        # only the upper bound was re-applied after snapping.
        from streamers import viral_detector as vd

        with patch.object(vd, "_normalize_clip_window", return_value=(100.0, 132.0)), patch.object(
            vd, "STREAMER_MIN_CLIP_DURATION_S", 40.0
        ):
            out = vd.parse_streamer_clips_json(self._raw(100.0, 132.0), self._segments(), 1)

        self.assertTrue(out)
        self.assertGreaterEqual(out[0].duration, 40.0 - 0.01, out[0].duration)

    def test_window_near_the_transcript_tail_still_meets_the_minimum(self):
        from streamers import viral_detector as vd

        # end_time pinned to the very end of the transcript: cannot grow forward,
        # so the window must be grown backwards instead.
        with patch.object(vd, "_normalize_clip_window", return_value=(1175.0, 1200.0)):
            out = vd.parse_streamer_clips_json(self._raw(1175.0, 1200.0), self._segments(1200.0), 1)

        self.assertTrue(out)
        self.assertGreaterEqual(out[0].duration, 30.0 - 0.01, out[0].duration)
        self.assertLessEqual(out[0].end_time, 1200.01)


# ---------------------------------------------------------------------------
# P1-1 — the pre-scan must be best-first
# ---------------------------------------------------------------------------
class TestPrescanZoneOrdering(unittest.TestCase):
    def test_loudest_zone_is_returned_first(self):
        from streamers import viral_detector as vd

        segments = []
        for t in range(0, 4000, 10):
            loud = 2000 <= t < 2200
            segments.append(
                TranscriptSegment(
                    start=float(t),
                    end=float(t + 10),
                    text="BRO WHAT NO WAY INSANE!!!" if loud else "so anyway i was saying",
                    words=[],
                )
            )

        zones = vd._prescan_high_energy_zones(segments, top_n=3, window_s=200.0)

        self.assertTrue(zones)
        # The old chronological sort made zones[0] the OPENING window, which is
        # what the Groq/Ollama fallbacks were then fed.
        self.assertGreaterEqual(zones[0][0], 1900.0, f"expected the loud window first, got {zones}")

    def test_empty_transcript_returns_no_zones(self):
        from streamers import viral_detector as vd

        self.assertEqual(vd._prescan_high_energy_zones([]), [])


# ---------------------------------------------------------------------------
# P1-6 — unknown streamer must not degrade into whole-roster auto mode
# ---------------------------------------------------------------------------
class TestDiscoveryCreatorValidation(unittest.TestCase):
    def test_unknown_creator_raises_before_any_discovery(self):
        from streamers.discovery import discover_streamer_candidates

        with self.assertRaises(ValueError) as ctx:
            discover_streamer_candidates(creator="kai_cenatt", max_candidates=1)
        self.assertIn("Unknown streamer", str(ctx.exception))

    def test_valid_creator_is_accepted(self):
        from streamers.discovery import STREAMER_CHANNELS, discover_streamer_candidates

        self.assertIn("kai_cenat", STREAMER_CHANNELS)
        # Every network tier is stubbed so this asserts only the roster guard:
        # a valid, oddly-cased creator must NOT raise.
        with patch(
            "streamers.discovery.load_streamer_history",
            return_value={"processed_video_ids": [], "attempts": {}},
        ), patch(
            "streamers.discovery.fetch_channel_videos_ytdlp", return_value=[]
        ), patch(
            "streamers.discovery.fetch_rss_feed_videos", return_value=[]
        ), patch(
            "streamers.discovery.search_stream_via_firecrawl", return_value=[]
        ):
            result = discover_streamer_candidates(creator="  KAI_CENAT  ", max_candidates=1)

        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# P1-3 / P1-5 — the publisher fails closed and never duplicates
# ---------------------------------------------------------------------------
class TestPublisherFailClosed(unittest.TestCase):
    CHANNELS = [{"id": "c1", "name": "Main", "service": "tiktok"}]

    def _publisher(self, channel_ids):
        from streamers.buffer_publisher import StreamerBufferPublisher

        pub = StreamerBufferPublisher(access_token="token", channel_ids=channel_ids)
        pub.get_channels = lambda: list(self.CHANNELS)  # type: ignore[method-assign]
        return pub

    @staticmethod
    def _candidate():
        from streamers.viral_detector import StreamerClipCandidate

        return StreamerClipCandidate(
            title="T",
            start_time=0.0,
            end_time=35.0,
            duration=35.0,
            viral_score=90,
            hook_reason="r",
            social_caption="c",
            hashtags=["#a", "#b"],
        )

    def test_refuses_when_no_target_channel_is_configured(self):
        pub = self._publisher([])
        with patch(
            "streamers.buffer_publisher.upload_clip_to_github_release",
            return_value="https://example.invalid/v.mp4",
        ):
            out = pub.schedule_clip(Path("clip.mp4"), self._candidate())

        self.assertEqual(out, [], "must not broadcast to every connected channel")
        self.assertIn("no target channel", pub.last_error or "")

    def test_refuses_unknown_channel_ids(self):
        pub = self._publisher(["cX"])
        with patch(
            "streamers.buffer_publisher.upload_clip_to_github_release",
            return_value="https://example.invalid/v.mp4",
        ):
            out = pub.schedule_clip(Path("clip.mp4"), self._candidate())

        self.assertEqual(out, [])
        self.assertIn("unknown channel", pub.last_error or "")

    def test_ambiguous_transport_error_is_not_retried(self):
        import requests

        pub = self._publisher(["c1"])
        with patch(
            "streamers.buffer_publisher.upload_clip_to_github_release",
            return_value="https://example.invalid/v.mp4",
        ), patch("streamers.buffer_publisher.time.sleep") as sleeper, patch(
            "streamers.buffer_publisher.requests.post",
            side_effect=requests.exceptions.Timeout("timed out"),
        ):
            out = pub.schedule_clip(Path("clip.mp4"), self._candidate())

        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["status"], "unknown")
        self.assertFalse(out[0]["retry_safe"])
        sleeper.assert_not_called()
        self.assertEqual(
            out[0]["video_url"], "https://example.invalid/v.mp4"
        )

    def test_unparseable_body_is_not_retried(self):
        pub = self._publisher(["c1"])
        fake_response = MagicMock()
        fake_response.text = "<html>gateway</html>"
        fake_response.json.side_effect = ValueError("not json")
        with patch(
            "streamers.buffer_publisher.upload_clip_to_github_release",
            return_value="https://example.invalid/v.mp4",
        ), patch("streamers.buffer_publisher.time.sleep") as sleeper, patch(
            "streamers.buffer_publisher.requests.post", return_value=fake_response
        ):
            out = pub.schedule_clip(Path("clip.mp4"), self._candidate())

        self.assertEqual(out[0]["status"], "unknown")
        sleeper.assert_not_called()

    def test_successful_post_is_recorded_once(self):
        pub = self._publisher(["c1"])
        ok = MagicMock()
        ok.text = ""
        ok.json.return_value = {
            "data": {"createPost": {"post": {"id": "p1", "status": "scheduled", "dueAt": None}}}
        }
        with patch(
            "streamers.buffer_publisher.upload_clip_to_github_release",
            return_value="https://example.invalid/v.mp4",
        ), patch("streamers.buffer_publisher.requests.post", return_value=ok):
            out = pub.schedule_clip(Path("clip.mp4"), self._candidate())

        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["status"], "scheduled")
        self.assertEqual(out[0]["post_id"], "p1")


# ---------------------------------------------------------------------------
# P1-7 — bounded retention
# ---------------------------------------------------------------------------
class TestRetentionSweep(unittest.TestCase):
    @staticmethod
    def _make(path: Path, age_days: float = 0.0) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
        stamp = time.time() - age_days * 86400.0
        os.utime(path, (stamp, stamp))
        return path

    def test_keeps_the_newest_runs_and_the_active_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            for i in range(5):
                self._make(root / f"run{i}" / "verdict.json", age_days=float(i))

            removed = retention_mod.prune_run_dirs(root, keep=2, current_run_id="run0")

            self.assertEqual(sorted(p.name for p in removed), ["run3", "run4"])
            self.assertTrue((root / "run0").exists(), "active run must survive")
            self.assertTrue((root / "run1").exists())

    def test_age_rule_expires_below_the_count_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            self._make(root / "old" / "v.json", age_days=90)
            self._make(root / "new" / "v.json", age_days=1)

            removed = retention_mod.prune_run_dirs(root, keep=100, max_age_days=30)

            self.assertEqual([p.name for p in removed], ["old"])
            self.assertTrue((root / "new").exists())

    def test_never_touches_anything_outside_the_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "clips"
            root.mkdir()
            outside = self._make(Path(tmp) / "outside.mp4", age_days=400)

            removed = retention_mod.prune_workspace_files(
                root, patterns=["*.mp4"], max_age_days=0
            )

            self.assertEqual(removed, [])
            self.assertTrue(outside.exists())

    def test_only_derived_artifacts_are_pruned_not_deliverables(self):
        with tempfile.TemporaryDirectory() as tmp:
            clips = Path(tmp)
            self._make(clips / "clip_1_x.mp4", age_days=30)          # deliverable
            self._make(clips / "clip_1_x_thumb.jpg", age_days=30)    # deliverable
            self._make(clips / "clip_1_x_preview.mp4", age_days=30)  # derived
            self._make(clips / "clip_1_x_edl.json", age_days=30)     # derived

            removed = retention_mod.prune_workspace_files(
                clips, patterns=["*_preview.mp4", "*_edl.json"], max_age_days=7
            )

            self.assertEqual(
                sorted(p.name for p in removed),
                ["clip_1_x_edl.json", "clip_1_x_preview.mp4"],
            )
            self.assertTrue((clips / "clip_1_x.mp4").exists())
            self.assertTrue((clips / "clip_1_x_thumb.jpg").exists())

    def test_recent_files_are_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            clips = Path(tmp)
            fresh = self._make(clips / "clip_1_preview.mp4", age_days=0)

            removed = retention_mod.prune_workspace_files(
                clips, patterns=["*_preview.mp4"], max_age_days=7
            )

            self.assertEqual(removed, [])
            self.assertTrue(fresh.exists())

    def test_dry_run_reports_without_deleting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            self._make(root / "r0" / "v.json", age_days=0)
            self._make(root / "r1" / "v.json", age_days=5)

            removed = retention_mod.prune_run_dirs(root, keep=1, dry_run=True)

            self.assertEqual(len(removed), 1)
            self.assertTrue(removed[0].exists())

    def test_missing_root_is_a_no_op(self):
        self.assertEqual(retention_mod.prune_run_dirs(Path("no/such/dir")), [])
        self.assertEqual(
            retention_mod.prune_workspace_files(Path("no/such/dir"), patterns=["*"]), []
        )

    def test_sweep_returns_both_buckets(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            downloads = Path(tmp) / "downloads"
            subtitles = Path(tmp) / "subtitles"
            clips = Path(tmp) / "clips"
            self._make(runs / "r0" / "v.json", age_days=0)
            self._make(runs / "r1" / "v.json", age_days=40)
            self._make(downloads / "streamer_clip_1.mp4", age_days=40)
            self._make(subtitles / "streamer_clip_1_x.ass", age_days=40)
            self._make(clips / "c_preview.mp4", age_days=40)

            sweep = retention_mod.run_retention_sweep(
                runs_root=runs,
                downloads_dir=downloads,
                subtitles_dir=subtitles,
                clips_dir=clips,
                current_run_id="r0",
            )

            self.assertEqual([p.name for p in sweep["run_dirs"]], ["r1"])
            self.assertEqual(len(sweep["workspace_files"]), 3)
            self.assertTrue((runs / "r0").exists())


# ---------------------------------------------------------------------------
# P2-3 — vertical sources must produce genuinely disjoint panes
# ---------------------------------------------------------------------------
class TestVerticalSourceSplit(unittest.TestCase):
    def test_vertical_source_stacks_disjoint_full_width_panes(self):
        from streamers.layout_compositor import _compute_disjoint_split_panes

        b1, b2 = _compute_disjoint_split_panes(608, 1080, 300, 400, 300, 700)

        self.assertEqual(b1[0], 0)
        self.assertEqual(b2[0], 0)
        self.assertEqual(b1[2], 608, "stacked panes must use the full source width")
        self.assertEqual(b1[2], b2[2])
        # The invariant that actually applies to a vertical stack.
        self.assertLessEqual(b1[1] + b1[3], b2[1] + 2)

    def test_horizontal_source_remains_strictly_disjoint(self):
        from streamers.layout_compositor import _compute_disjoint_split_panes

        b1, b2 = _compute_disjoint_split_panes(1920, 1080, 400, 540, 1500, 540)

        self.assertLessEqual(b1[0] + b1[2], b2[0] + 2)
        for box in (b1, b2):
            for value in box:
                self.assertEqual(value % 2, 0, f"odd coordinate in {box}")


# ---------------------------------------------------------------------------
# P2-4 — failed analysis must be distinguishable from "no peaks"
# ---------------------------------------------------------------------------
class TestAudioAnalyzerUnavailable(unittest.TestCase):
    def test_missing_media_reports_unavailable(self):
        from streamers.audio_analyzer import analyze_audio_energy

        result = analyze_audio_energy(Path("definitely_missing_media_file.mp4"))

        self.assertIn("available", result)
        self.assertFalse(result["available"])
        self.assertEqual(result["peak_intervals"], [])
        # Backwards-compatible keys consumers already rely on.
        for key in ("median_db", "max_db", "peaks", "peak_intervals"):
            self.assertIn(key, result)


# ---------------------------------------------------------------------------
# P2-8 — the private engine surface V2 depends on must stay pinned
# ---------------------------------------------------------------------------
class TestEngineHelperContract(unittest.TestCase):
    REQUIRED = (
        "query_gemini_models",
        "query_groq_free_models",
        "query_ollama_cloud_models",
        "strip_emojis",
        "_normalize_clip_window",
        "_safe_int",
    )

    def test_every_engine_helper_v2_imports_still_exists(self):
        import core.viral_detector as engine

        for name in self.REQUIRED:
            self.assertTrue(
                hasattr(engine, name),
                f"core.viral_detector.{name} is missing — clips-core renamed or "
                f"removed a helper streamers/viral_detector.py imports. Land the "
                f"change upstream and sync both pipelines (see SYNC.md).",
            )


# ---------------------------------------------------------------------------
# Concurrency safety of the new naming helper (used on the render hot path)
# ---------------------------------------------------------------------------
class TestNamingUnderConcurrency(unittest.TestCase):
    def test_fingerprints_are_stable_under_thread_pool(self):
        titles = [f"clip / {i}" for i in range(64)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            rows = list(pool.map(paths_mod.slugify_title, titles))
        self.assertEqual(rows, [paths_mod.slugify_title(t) for t in titles])


if __name__ == "__main__":
    unittest.main()
