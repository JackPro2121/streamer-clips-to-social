"""P4: the download pre-flight (container -> decode -> luminance variance).

Run 37329847616's dissected segment covered 47.047s cleanly, but the class it
represents -- unusable downloads and flat/dead openings -- should fail before
any render spends CPU. The flat-slate class is from the 2026-10-05 freeze
investigation: a clip's source opened with 3.5s of flat grey/green before the
slide content started, and rendered into freeze blocks.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from core.preflight import PreflightReport, preflight_source

HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


@unittest.skipUnless(HAS_FFMPEG, "ffmpeg/ffprobe required")
class TestPreflight(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _run(self, *args: str) -> Path | None:
        output = self.workspace / f"clip_{len(list(self.workspace.iterdir()))}.mp4"
        result = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", *args, str(output)],
            capture_output=True, text=True, timeout=300,
        )
        return output if result.returncode == 0 and output.exists() else None

    def test_flat_opening_is_rejected_at_the_luminance_stage(self) -> None:
        flat = self._run(
            "-f", "lavfi", "-i", "color=c=gray:s=320x180:d=5:r=10",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
        )
        assert flat is not None
        report = preflight_source(flat)
        self.assertFalse(report.ok)
        self.assertEqual(report.stage, "luminance")
        self.assertLess(report.luminance_stddev_max, 6.0)

    def test_textured_clip_passes_all_stages(self) -> None:
        textured = self._run(
            "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=10:duration=5",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
        )
        assert textured is not None
        report = preflight_source(textured)
        self.assertTrue(report.ok, report.as_dict())
        self.assertGreater(report.luminance_stddev_max, 6.0)

    def test_missing_video_stream_is_rejected_at_the_container_stage(self) -> None:
        audio_only = self._run(
            "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
            "-c:a", "aac", "-vn",
        )
        assert audio_only is not None
        report = preflight_source(audio_only)
        self.assertFalse(report.ok)
        self.assertEqual(report.stage, "container")

    def test_too_short_segment_is_rejected_at_the_container_stage(self) -> None:
        tiny = self._run(
            "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=10:duration=1",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
        )
        assert tiny is not None
        report = preflight_source(tiny)
        self.assertFalse(report.ok)
        self.assertEqual(report.stage, "container")

    def test_stream_length_mismatch_is_reported_but_not_rejected(self) -> None:
        # The renderer harmonises this class; the pre-flight must not lose the
        # candidate (run 37347481817, two of three clips).
        mismatched = self._run(
            "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=10:duration=5",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=5.2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
        )
        assert mismatched is not None
        report = preflight_source(mismatched)
        self.assertTrue(report.ok, report.as_dict())
        self.assertGreater(report.audio_duration, report.video_duration)

    def test_missing_file_is_rejected_not_crashed(self) -> None:
        report = preflight_source(self.workspace / "nope.mp4")
        self.assertFalse(report.ok)
        self.assertIsInstance(report, PreflightReport)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
