"""The renderer must certify the stream-duration contract, not hope for it.

Defect evidence (run 37329847616, 2026-10-05)
---------------------------------------------
Four of five rendered clips left the renderer with audio and video within 7ms
of each other. ``clip_2_THE_BABY_WAKE_UP_CALL.mp4`` shipped video ``45.300s``
against audio ``45.400s`` (+100ms) from a source segment that covered
``47.047s`` -- the pixel verdict blocked it on ``av_structural_drift``
(tolerance 80ms). Replaying the exact filtergraph locally from the same
episode did not reproduce the loss, so the renderer treats it as a
frame-boundary fact it cannot prevent: after the first encode, the audio
stream is conformed to the video stream -- trimmed when it overruns,
silence-padded when it falls short. The video stream is never re-encoded, so
freeze/motion evidence is untouched.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from core.video_editor import (
    STREAM_DURATION_TOLERANCE_S,
    _harmonise_stream_durations,
    _probe_stream_durations,
)

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@unittest.skipUnless(FFMPEG and FFPROBE, "ffmpeg and ffprobe required")
class TestStreamHarmonisation(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="stream_harmonise_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_clip(self, name: str, video_s: float, audio_s: float) -> Path:
        path = self.tmp / name
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", f"testsrc2=size=320x180:rate=10:duration={video_s}",
                "-f", "lavfi", "-i", f"sine=frequency=440:duration={audio_s}",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-ar", "48000",
                str(path),
            ],
            capture_output=True, check=True, timeout=180,
        )
        return path

    def test_audio_overrun_is_trimmed_to_the_video(self):
        path = self._make_clip("audio_long.mp4", video_s=2.0, audio_s=2.15)
        video, audio = _probe_stream_durations(path)
        self.assertGreater(audio - video, STREAM_DURATION_TOLERANCE_S)

        _harmonise_stream_durations(path)

        video, audio = _probe_stream_durations(path)
        assert video is not None and audio is not None
        self.assertLessEqual(abs(audio - video), STREAM_DURATION_TOLERANCE_S)

    def test_video_overrun_is_silence_padded(self):
        path = self._make_clip("video_long.mp4", video_s=2.15, audio_s=2.0)
        video, audio = _probe_stream_durations(path)
        self.assertGreater(video - audio, STREAM_DURATION_TOLERANCE_S)

        _harmonise_stream_durations(path)

        video, audio = _probe_stream_durations(path)
        assert video is not None and audio is not None
        self.assertLessEqual(abs(audio - video), STREAM_DURATION_TOLERANCE_S)

    def test_matched_clip_is_left_alone(self):
        path = self._make_clip("matched.mp4", video_s=2.0, audio_s=2.0)
        before = _sha256(path)

        _harmonise_stream_durations(path)

        self.assertEqual(_sha256(path), before)

    def test_video_only_clip_is_left_alone(self):
        path = self.tmp / "silent.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-f", "lavfi",
                "-i", "testsrc2=size=320x180:rate=10:duration=2.0",
                "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(path),
            ],
            capture_output=True, check=True, timeout=180,
        )
        before = _sha256(path)

        _harmonise_stream_durations(path)

        self.assertEqual(_sha256(path), before)


if __name__ == "__main__":
    unittest.main()
