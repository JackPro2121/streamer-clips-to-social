"""End-to-end gate verification on real rendered media.

Why this file exists
--------------------
Every other V2 gate test mocks ``_import_verdict``. That is fine for the verdict
bookkeeping it targets, but it left the gate's *real* behaviour on actual pixels
unverified in CI: the fail-closed policy, the black-margin audit and the
duration contract were all exercised only against mocks. A regression that made
the visual check report ``available: False`` in production — cv2 missing, frames
undecodable, an ffmpeg colour-range change — would have passed the whole suite.

These tests synthesise real 9:16 clips with ffmpeg and run the real
``verify_clip`` path over real decoded frames. They are the "corpus" the audit
found missing (``verdict.verify_corpus`` was called by nothing and there was no
``tests/fixtures/``).

They skip when ffmpeg is unavailable, matching the rest of the suite.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from streamers.gate import verify_clip  # noqa: E402

HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))

# Short clip: the sampled-frame logic only needs a handful of decodable frames,
# and the duration contract is passed in explicitly.
_CLIP_SECONDS = 4.0
_MIN_S = 3.0
_MAX_S = 5.0


def _run_ffmpeg(args: list[str]) -> bool:
    proc = subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=180,
    )
    return proc.returncode == 0


@unittest.skipUnless(HAS_FFMPEG, "ffmpeg and ffprobe are required")
class TestGateOnRealMedia(unittest.TestCase):
    def _make_clip(self, out_path: Path, *, with_black_bars: bool) -> None:
        video = f"color=c=black:s=1080x1920:d={_CLIP_SECONDS}:r=10"
        if with_black_bars:
            # 200px of pure black top and bottom around a grey field: the
            # black-margin audit samples the outer 36px band, so both bands are
            # solidly inside its sample window.
            graph = "drawbox=x=0:y=200:w=1080:h=1520:color=gray:t=fill"
        else:
            graph = "drawbox=x=0:y=0:w=1080:h=1920:color=gray:t=fill"

        ok = _run_ffmpeg(
            [
                "-f", "lavfi", "-i", video,
                "-f", "lavfi", "-i", f"sine=frequency=440:duration={_CLIP_SECONDS}",
                "-vf", graph,
                "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "128k",
                "-shortest",
                str(out_path),
            ]
        )
        self.assertTrue(ok, f"ffmpeg could not synthesise {out_path.name}")
        self.assertTrue(out_path.exists() and out_path.stat().st_size > 0)

    def test_valid_full_frame_clip_runs_the_real_visual_audit_and_passes_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp) / "full_frame.mp4"
            self._make_clip(clip, with_black_bars=False)

            res = verify_clip(
                clip,
                run_id="e2e_full_frame",
                runs_dir=Path(tmp) / "runs",
                min_duration_s=_MIN_S,
                max_duration_s=_MAX_S,
            )

            # The audit must have actually RAN on decoded pixels...
            self.assertNotIn(
                "v2_visual_check_unavailable",
                res.blocking_codes,
                "the black-margin audit did not run on real media — fail-closed "
                "would block every clip in production",
            )
            # ...and it must not have flagged a full-frame 9:16 clip.
            self.assertNotIn("v2_black_frame_margin", res.blocking_codes)

            payload = res.verdict_path.read_text(encoding="utf-8")
            self.assertIn('"visual"', payload)
            self.assertIn('"available": true', payload)

    def test_letterboxed_clip_is_blocked_by_the_real_visual_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp) / "letterboxed.mp4"
            self._make_clip(clip, with_black_bars=True)

            res = verify_clip(
                clip,
                run_id="e2e_letterboxed",
                runs_dir=Path(tmp) / "runs",
                min_duration_s=_MIN_S,
                max_duration_s=_MAX_S,
            )

            self.assertFalse(res.passed)
            self.assertIn("v2_black_frame_margin", res.blocking_codes)

    def test_real_duration_outside_contract_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp) / "full_frame.mp4"
            self._make_clip(clip, with_black_bars=False)

            # Same real clip, contract that it cannot satisfy.
            res = verify_clip(
                clip,
                run_id="e2e_duration",
                runs_dir=Path(tmp) / "runs",
                min_duration_s=30.0,
                max_duration_s=45.0,
            )

            self.assertFalse(res.passed)
            self.assertIn("v2_duration_out_of_contract", res.blocking_codes)


@unittest.skipUnless(HAS_FFMPEG, "ffmpeg and ffprobe are required")
class TestVerdictCorpusApiIsStillWired(unittest.TestCase):
    """The corpus verifier exists but nothing invoked it — pin the API."""

    def test_verify_corpus_is_importable_and_callable(self):
        from core.verification import verdict

        self.assertTrue(callable(getattr(verdict, "verify_corpus", None)))


if __name__ == "__main__":
    unittest.main()
