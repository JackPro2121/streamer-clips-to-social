"""Source-alignment: the clip's audio must start where the trim said it would.

This is the deterministic perceptual A/V gate. SyncNet's LSE metrics could not
be calibrated onto this domain (the corpus calibration showed the published
bands do not transfer to loudnorm-processed vertical crops), so the gate
compares the clip's own audio against the source segment it was cut from.

The negative control is the important test: a clip whose audio is deliberately
delayed by 300ms must be reported as out of tolerance, or the gate is theatre.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.verification import av_sync

HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
SOURCE_SECONDS = 8.0
CLIP_START = 1.3
CLIP_END = 6.3


def _build_source(path: Path) -> bool:
    command = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30",
        "-f", "lavfi", "-i", "anoisesrc=color=pink:amplitude=0.5:sample_rate=48000",
        "-t", str(SOURCE_SECONDS),
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    return result.returncode == 0 and path.exists() and path.stat().st_size > 0


def _render_clip(source: Path, output: Path) -> None:
    from core.face_tracker import FramingDecision
    from core.video_editor import render_viral_clip

    render_viral_clip(
        source_video_path=source,
        start_time=CLIP_START,
        end_time=CLIP_END,
        output_clip_path=output,
        framing=FramingDecision(
            face_count=1,
            mode="single_smooth",
            smoothed_center_x=320,
            video_width=640,
            video_height=360,
            active_x=0,
            active_y=0,
            active_w=640,
            active_h=360,
        ),
        ass_subtitle_path=None,
        burn_subtitles=False,
    )


class TestBestSourceOffset(unittest.TestCase):
    def test_known_offset_is_recovered(self) -> None:
        sample_rate = 16000
        rng = np.random.default_rng(4)
        source = rng.standard_normal(sample_rate * 10).astype(np.float32)
        start = sample_rate * 3
        clip = source[start:start + sample_rate * 4]
        offset, confidence = av_sync.best_source_offset(clip, source)
        self.assertAlmostEqual(offset, 3.0, delta=0.01)
        self.assertGreater(confidence, 0.95)

    def test_unrelated_signal_is_low_confidence(self) -> None:
        rng = np.random.default_rng(5)
        clip = rng.standard_normal(16000 * 3).astype(np.float32)
        other = rng.standard_normal(16000 * 8).astype(np.float32)
        _offset, confidence = av_sync.best_source_offset(clip, other)
        self.assertLess(confidence, av_sync.SOURCE_ALIGN_MIN_CONFIDENCE)


class TestGccPhatPsr(unittest.TestCase):
    """P3: GCC-PHAT locates the lag; PSR decides whether to trust the winner.

    Run 37213499992 blocked a clip at -115ms on a raw-waveform peak of 0.392,
    and two clips in run 37329847616 could not be aligned at all (0.076/0.143).
    Whitening removes the spectral-tilt bias; the peak-to-sidelobe ratio then
    separates "one lag clearly wins" from "some lag won an argmax on noise".
    """

    def test_related_signal_has_a_sharp_peak(self) -> None:
        sample_rate = 16000
        rng = np.random.default_rng(11)
        source = rng.standard_normal(sample_rate * 10).astype(np.float32)
        start = sample_rate * 4
        clip = source[start:start + sample_rate * 3]
        offset, confidence, psr = av_sync._gcc_phat_offset(clip, source, sample_rate)
        self.assertAlmostEqual(offset, 4.0, delta=0.01)
        self.assertGreater(psr, av_sync.SOURCE_ALIGN_MIN_PSR)
        self.assertGreater(confidence, av_sync.SOURCE_ALIGN_MIN_CONFIDENCE)

    def test_unrelated_signal_has_a_flat_surface(self) -> None:
        rng = np.random.default_rng(12)
        clip = rng.standard_normal(16000 * 3).astype(np.float32)
        other = rng.standard_normal(16000 * 8).astype(np.float32)
        _offset, _confidence, psr = av_sync._gcc_phat_offset(clip, other, 16000)
        self.assertLess(psr, av_sync.SOURCE_ALIGN_MIN_PSR)


@unittest.skipUnless(HAS_FFMPEG, "ffmpeg/ffprobe required")
class TestAlignWithSource(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmp.name)
        self.source = self.workspace / "source.mp4"
        if not _build_source(self.source):
            self.skipTest("could not build the alignment source")
        self.clip = self.workspace / "rendered.mp4"
        _render_clip(self.source, self.clip)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_renderer_alignment_is_within_tolerance(self) -> None:
        alignment = av_sync.align_with_source(self.clip, self.source, CLIP_START)
        self.assertTrue(alignment.measured, alignment.reason)
        self.assertLessEqual(
            abs(alignment.drift_ms),
            av_sync.SOURCE_ALIGN_TOLERANCE_MS,
            f"renderer drifted {alignment.drift_ms:+.0f}ms (confidence {alignment.confidence:.3f})",
        )
        self.assertTrue(alignment.ok)
        self.assertGreater(alignment.confidence, 0.5)

    def test_delayed_audio_is_detected(self) -> None:
        """Negative control: +300ms of audio delay must fail the gate."""
        shifted = self.workspace / "shifted.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(self.clip),
             "-c:v", "copy", "-af", "adelay=300:all=1", "-c:a", "aac", str(shifted)],
            check=True, timeout=300,
        )
        alignment = av_sync.align_with_source(shifted, self.source, CLIP_START)
        self.assertTrue(alignment.measured, alignment.reason)
        self.assertGreater(alignment.drift_ms, 200.0, "a 300ms delay must read as positive drift")
        self.assertLess(alignment.drift_ms, 450.0)
        self.assertFalse(alignment.ok, "the gate must reject a delayed clip")

    def test_missing_source_is_reported_not_crashed(self) -> None:
        alignment = av_sync.align_with_source(self.clip, self.workspace / "nope.mp4", 0.0)
        self.assertFalse(alignment.measured)
        self.assertIn("not found", alignment.reason)

    def test_string_paths_measure_the_same(self) -> None:
        """The English-audio fallback once wrote str(fixed_clip) into the clip
        path; every affected clip then died at pixel verification with
        "'str' object has no attribute 'exists'" (run 37296577217)."""
        alignment = av_sync.align_with_source(str(self.clip), str(self.source), CLIP_START)
        self.assertTrue(alignment.measured, alignment.reason)
        self.assertLessEqual(abs(alignment.drift_ms), av_sync.SOURCE_ALIGN_TOLERANCE_MS)


class TestPathNormalisation(unittest.TestCase):
    def test_string_paths_are_normalised_not_crashed(self) -> None:
        alignment = av_sync.align_with_source("missing_clip.mp4", "missing_source.mp4", 1.0)
        self.assertFalse(alignment.measured)
        self.assertIn("not found", alignment.reason)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
