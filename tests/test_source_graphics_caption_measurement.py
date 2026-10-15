"""The caption-on-face measurement must measure OUR captions, not source text.

Defect evidence (run 37329847616, 2026-10-05)
---------------------------------------------
``clip_3_STOP_MAKING_EXCUSES_NOW.mp4`` (Dave Ramsey episode) measured **16% of
"caption" pixels on the brow/eyes, worst at 0%**, blocked, re-rendered with the
time-ranged face-safe bands, and measured the same 16% again. Dissection of the
artifact showed every one of our caption pixels sitting safely at the bottom of
the frame (``y[264-269]`` at 360p) while all the overlap came from the show's
own burned-in graphics -- the giant "THE DAVE RAMSEY SHOW" logo and the
"STOP MAKING EXCUSES NOW" banner during the first ~4 seconds -- which
``outlined_text_mask`` cannot distinguish from captions.

The fix: only *persistent* text rows count as captions. Our captions are on
screen in nearly every frame; a programme's own graphics are transient.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.verification.captions import (
    Rect,
    _persistent_caption_rows,
    measure_on_face,
)

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


def _rows_mask(hit_rows: dict[int, float], height: int, frames: int) -> np.ndarray:
    """Build a frame stack mask: row -> fraction of frames where it is set."""
    stack = np.zeros((frames, height, 8), dtype=bool)
    for row, fraction in hit_rows.items():
        hits = max(1, int(round(fraction * frames)))
        stack[:hits, row, :] = True
    return stack


class TestPersistentCaptionRows(unittest.TestCase):
    def test_transient_source_graphics_are_excluded(self):
        # 10 frames; rows 10-13 are captions in every frame, rows 1-3 are the
        # show's intro graphics in 2 frames only. The dominant fractions pick
        # the 0.5 * 1.0 = 0.5 threshold, so the 0.2 graphics rows are dropped.
        masks = _rows_mask({10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 1: 0.2, 2: 0.2, 3: 0.2}, 20, 10)
        keep = _persistent_caption_rows(list(masks), 20)
        assert keep is not None
        self.assertTrue(keep[10] and keep[11] and keep[12] and keep[13])
        self.assertFalse(keep[1] or keep[2] or keep[3])

    def test_persistent_text_on_a_face_survives_the_filter(self):
        # The real defect class (broke_1): captions across the brow in EVERY
        # frame must still be measured, not filtered away.
        masks = _rows_mask({5: 1.0, 6: 1.0, 7: 1.0}, 20, 10)
        keep = _persistent_caption_rows(list(masks), 20)
        assert keep is not None
        self.assertTrue(keep[5] and keep[6] and keep[7])

    def test_short_caption_clip_keeps_its_rows(self):
        # Captions live in only 25% of frames (quiet clip); an even rarer
        # source graphic at 10% must still be dropped.
        masks = _rows_mask({8: 0.25, 9: 0.25, 2: 0.1}, 20, 20)
        keep = _persistent_caption_rows(list(masks), 20)
        assert keep is not None
        self.assertTrue(keep[8] and keep[9])
        self.assertFalse(keep[2])

    def test_no_text_returns_none(self):
        masks = [np.zeros((20, 8), dtype=bool) for _ in range(10)]
        self.assertIsNone(_persistent_caption_rows(masks, 20))


@unittest.skipUnless(FFMPEG and FFPROBE, "ffmpeg and ffprobe required")
class TestMeasureOnFaceIgnoresSourceGraphics(unittest.TestCase):
    """End-to-end: a transient upper-frame graphic must not read as a caption."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="caption_persist_"))
        # A grey 2s 10fps canvas with:
        #  * a black-bordered red box in the "face" zone
        #  * a persistent black-bordered red "caption" box low in the frame
        upper_enable = "enable='between(t,0,0.35)'"
        chain = (
            f"drawbox=x=100:y=20:w=120:h=30:color=black:t=fill:{upper_enable},"
            f"drawbox=x=104:y=23:w=112:h=24:color=0xFFE600:t=fill:{upper_enable},"
            "drawbox=x=100:y=140:w=120:h=25:color=black:t=fill,"
            "drawbox=x=104:y=143:w=112:h=19:color=0xFFE600:t=fill"
        )
        cls.transient = cls.tmp / "transient.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-f", "lavfi",
                "-i", "color=c=gray:s=320x180:d=2:r=10",
                "-vf", chain, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(cls.transient),
            ],
            capture_output=True, check=True, timeout=120,
        )
        persistent_chain = (
            "drawbox=x=100:y=20:w=120:h=30:color=black:t=fill,"
            "drawbox=x=104:y=23:w=112:h=24:color=0xFFE600:t=fill,"
            "drawbox=x=100:y=140:w=120:h=25:color=black:t=fill,"
            "drawbox=x=104:y=143:w=112:h=19:color=0xFFE600:t=fill"
        )
        cls.persistent = cls.tmp / "persistent.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-f", "lavfi",
                "-i", "color=c=gray:s=320x180:d=2:r=10",
                "-vf", persistent_chain, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(cls.persistent),
            ],
            capture_output=True, check=True, timeout=120,
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _avoid(self):
        # The "face" is exactly where the upper box sits.
        return [Rect(x=100, y=20, width=120, height=30)]

    def test_transient_upper_graphic_is_not_measured(self):
        on_face, _at, measured = measure_on_face(
            self.transient, self._avoid(), face_space=(320, 180), max_samples=10
        )
        self.assertEqual(on_face, 0.0, "transient source graphic counted as caption")
        self.assertGreaterEqual(measured, 8)

    def test_persistent_text_on_the_face_still_blocks(self):
        on_face, _at, _measured = measure_on_face(
            self.persistent, self._avoid(), face_space=(320, 180), max_samples=10
        )
        self.assertGreater(on_face, 15.0, "persistent on-face text must still be measured")


if __name__ == "__main__":
    unittest.main()
