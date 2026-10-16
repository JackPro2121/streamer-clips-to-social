"""Regression tests for the third V2 remediation round — the first live run.

Evidence base: workflow run 38758861858 processed pLO8AVvIbkg ("MONDAY NIGHT
STREAM", 17,426s). Two clips rendered, zero published. Reading the rendered
artifacts (frames + verdict.json) showed the gate was right to be suspicious
but wrong about what it was measuring:

    F-A  selection could aim a clip window at the stream tail. Clip 2's window
         closed 42s before EOF and the render carried a ~0.5s black+frozen tail
         (freeze_unterminated + a 98.7%-black final frame) — dead air that
         selection should never have chosen.
    F-B  _face_stability_check compared the largest face's centre between
         samples ~3.2s apart, so ordinary head movement (measured sway 4.5px)
         read as a 273-415px "jump" and blocked both live clips.
    F-C  synthesised webcam/content boxes were never checked against pixels:
         clip 1's webcam crop included ~10 source rows of dark window chrome,
         which the 3.38x pane upscale turned into a persistent 66%-black band
         across the top of the frame.
    F-D  one boundary frame could block a clip: the black-margin audit scored
         the *fraction of near-black pixels* inside an edge band, so dark
         content (chat panels, dark game scenes) read as a bar and a single
         black final frame failed a 38s clip.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.transcriber import TranscriptSegment, WordTimestamp  # noqa: E402

try:
    import cv2  # noqa: F401
    HAS_CV2 = True
except Exception:  # pragma: no cover - cv2 ships in requirements
    HAS_CV2 = False

HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _plain_segments(end_s: float, step: float = 10.0, texts=None):
    """Build deterministic segments every ``step`` seconds from 0 to ``end_s``."""
    segs = []
    t = 0.0
    while t < end_s:
        e = min(t + step, end_s)
        text = (texts or {}).get(t, "bro what")
        segs.append(
            TranscriptSegment(
                start=t,
                end=e,
                text=text,
                words=[WordTimestamp("bro", t, t + 1.0)],
            )
        )
        t = e
    return segs


def _raw_clip_json(start: float, end: float) -> str:
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


# ---------------------------------------------------------------------------
# F-A — selection must keep clear of the stream tail
# ---------------------------------------------------------------------------
class TestTailExclusion(unittest.TestCase):
    """F-A — no detector path may aim a window at the stream's outro."""

    def test_effective_end_excludes_the_configured_tail(self):
        from streamers import viral_detector as vd

        segs = _plain_segments(2000.0)
        self.assertAlmostEqual(
            vd.effective_transcript_end(segs),
            2000.0 - vd.STREAMER_TAIL_EXCLUSION_S,
            places=3,
        )

    def test_effective_end_keeps_short_transcripts_intact(self):
        from streamers import viral_detector as vd

        segs = _plain_segments(100.0)
        self.assertAlmostEqual(vd.effective_transcript_end(segs), 100.0, places=3)
        # Empty transcripts keep the existing 9999s sentinel (no clamping).
        self.assertEqual(vd.effective_transcript_end([]), 9999.0)

    def test_llm_candidate_inside_the_tail_is_dropped(self):
        from streamers import viral_detector as vd

        segs = _plain_segments(2000.0)
        tail_start = 2000.0 - vd.STREAMER_TAIL_EXCLUSION_S + 20.0
        with patch.object(
            vd, "_normalize_clip_window", return_value=(tail_start, tail_start + 40.0)
        ):
            out = vd.parse_streamer_clips_json(
                _raw_clip_json(tail_start, tail_start + 40.0), segs, 1
            )

        self.assertEqual(out, [], "a tail-timed candidate must not be relocated")

    def test_llm_candidate_before_the_tail_is_kept_and_clamped(self):
        from streamers import viral_detector as vd

        segs = _plain_segments(2000.0)
        effective_end = 2000.0 - vd.STREAMER_TAIL_EXCLUSION_S
        start, end = 1700.0, 1740.0
        with patch.object(vd, "_normalize_clip_window", return_value=(start, end)):
            out = vd.parse_streamer_clips_json(_raw_clip_json(start, end), segs, 1)

        self.assertEqual(len(out), 1)
        self.assertLessEqual(out[0].end_time, effective_end + 0.01)

    def test_heuristic_windows_never_reach_the_raw_tail(self):
        from streamers import viral_detector as vd

        loud_at = 1900.0
        segs = _plain_segments(2000.0, texts={loud_at: "NO WAY! BRO! WHAT! WHAT! NO!"})
        out = vd._heuristic_streamer_detector(segs, creator_name="Jynxzi", num_clips=3)

        self.assertTrue(out, "the heuristic detector must still return candidates")
        effective_end = 2000.0 - vd.STREAMER_TAIL_EXCLUSION_S
        for c in out:
            self.assertLessEqual(c.end_time, effective_end + 0.01)
            self.assertFalse(
                c.start_time <= loud_at + 5.0 < c.end_time,
                "a candidate covers the excluded tail segment",
            )

    def test_prescan_zones_never_cover_the_raw_tail(self):
        from streamers import viral_detector as vd

        loud_at = 1900.0
        segs = _plain_segments(2000.0, texts={loud_at: "NO WAY! BRO! WHAT! WHAT! NO!"})
        zones = vd._prescan_high_energy_zones(segs, top_n=3)

        self.assertTrue(zones)
        effective_end = 2000.0 - vd.STREAMER_TAIL_EXCLUSION_S
        for start, end in zones:
            self.assertLessEqual(end, effective_end + 0.01)
            self.assertFalse(start <= loud_at + 5.0 < end)

    def test_spec_rejects_a_nonsense_tail_exclusion(self):
        from streamers import spec as spec_mod

        with patch.object(spec_mod, "STREAMER_TAIL_EXCLUSION_S", -5.0):
            with self.assertRaises(ValueError) as negative:
                spec_mod.validate_spec()
        self.assertIn("tail exclusion", str(negative.exception))

        with patch.object(
            spec_mod, "STREAMER_TAIL_EXCLUSION_S", spec_mod.STREAMER_MAX_TAIL_EXCLUSION_S + 1.0
        ):
            with self.assertRaises(ValueError) as oversized:
                spec_mod.validate_spec()
        self.assertIn("silently discards usable transcript", str(oversized.exception))


# ---------------------------------------------------------------------------
# F-A (repair half) — a render must not carry the stream's dead tail to the gate
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAS_FFMPEG and HAS_CV2, "ffmpeg and cv2 are required")
class TestBoundaryRepair(unittest.TestCase):
    """C2 — dead head/tail frames are removed before the gate sees the file."""

    @staticmethod
    def _duration_s(path: Path) -> float:
        cap = cv2.VideoCapture(str(path))
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            cap.release()
        return frames / fps if fps > 0 else 0.0

    @staticmethod
    def _make_clip(
        path: Path, *, head_black_s: float, body_s: float, tail_black_s: float
    ) -> None:
        total = head_black_s + body_s + tail_black_s
        vf = (
            f"color=c=black:s=270x480:d={total}:r=10,"
            f"drawbox=x=0:y=0:w=270:h=480:color=gray:t=fill:"
            f"enable='between(t,{head_black_s},{head_black_s + body_s})'"
        )
        proc = subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", vf,
                "-f", "lavfi", "-i", f"sine=frequency=440:duration={total}",
                "-shortest",
                "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "96k",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if proc.returncode != 0:
            raise AssertionError(proc.stderr[-400:])

    def test_dead_head_and_tail_are_trimmed(self):
        from streamers.repair import trim_dead_boundaries

        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp) / "clip.mp4"
            self._make_clip(clip, head_black_s=1.0, body_s=4.0, tail_black_s=0.5)
            before = self._duration_s(clip)

            res = trim_dead_boundaries(clip, min_duration_s=3.0)

            self.assertTrue(res.changed)
            self.assertAlmostEqual(res.head_trim_s, 1.0, delta=0.2)
            self.assertAlmostEqual(res.tail_trim_s, 0.5, delta=0.2)
            after = self._duration_s(clip)
            self.assertAlmostEqual(after, 4.0, delta=0.3)
            self.assertLess(after, before)

            # Idempotent — the dead frames are gone, a second pass finds nothing.
            again = trim_dead_boundaries(clip, min_duration_s=3.0)
            self.assertFalse(again.changed)
            self.assertIsNone(again.skipped_reason)

    def test_trim_that_would_break_the_floor_is_refused(self):
        from streamers.repair import trim_dead_boundaries

        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp) / "clip.mp4"
            self._make_clip(clip, head_black_s=1.0, body_s=3.0, tail_black_s=0.5)
            digest = hashlib.sha256(clip.read_bytes()).hexdigest()

            res = trim_dead_boundaries(clip, min_duration_s=4.0)

            self.assertFalse(res.changed)
            self.assertIn("duration floor", res.skipped_reason or "")
            self.assertEqual(hashlib.sha256(clip.read_bytes()).hexdigest(), digest)

    def test_clean_clip_is_left_alone(self):
        from streamers.repair import trim_dead_boundaries

        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp) / "clip.mp4"
            self._make_clip(clip, head_black_s=0.0, body_s=5.0, tail_black_s=0.0)
            digest = hashlib.sha256(clip.read_bytes()).hexdigest()

            res = trim_dead_boundaries(clip, min_duration_s=3.0)

            self.assertFalse(res.changed)
            self.assertIsNone(res.skipped_reason)
            self.assertEqual(hashlib.sha256(clip.read_bytes()).hexdigest(), digest)

    def test_all_black_clip_is_refused(self):
        from streamers.repair import trim_dead_boundaries

        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp) / "clip.mp4"
            proc = subprocess.run(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", "color=c=black:s=270x480:d=2:r=10",
                    "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p",
                    str(clip),
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr[-400:])

            res = trim_dead_boundaries(clip, min_duration_s=3.0)

            self.assertFalse(res.changed)
            self.assertIn("dead", res.skipped_reason or "")


if __name__ == "__main__":
    unittest.main()
