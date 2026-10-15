"""The renderer must not shift audio against video.

Why this test exists
--------------------
The pipeline shipped clips whose first 0.25 s was a frozen cover frame while the
audio was already speaking, and a filtergraph ordering bug
(``asetpts=PTS-STARTPTS`` applied before ``atrim``) put audio packets on the
wrong timeline. Both read as "lips do not match the voice".

The rest of the A/V checks in the repo are stream-level (start/duration deltas),
which cannot see a shift that preserves both. This is the render-level proof:
a synthetic 24 fps source puts a white flash and a 1 kHz beep at the same
instants, the clip is rendered through the real ``render_viral_clip`` path, and
the two event trains are compared.

A 24 fps source with a fractional start (0.7 s) is deliberate: that is the
shape of the real downloads, and it exercises the ``fps=30`` conversion, the
10 ms video trim quantisation and both loudnorm passes.

Measured on the fixed code: every flash/beep pair aligns to within one frame
(0.0 ms on a 33 ms grid). Mutation guard: re-introducing the cover frame or
moving ``asetpts`` before ``atrim`` moves a pair by >= 250 ms, which fails the
assertions below.
"""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.face_tracker import FramingDecision, ShotPlan
from core.video_editor import render_viral_clip

FLASH_EVERY_S = 2.0
FIRST_EVENT_S = 1.0
EVENT_LENGTH_S = 0.1
SOURCE_SECONDS = 13.0
CLIP_START_S = 0.7
CLIP_END_S = 11.7
TOLERANCE_S = 0.05


def _build_sync_source(path: Path) -> bool:
    flash = "lt(mod(t-1\\,2)\\,0.1)"
    command = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", "color=c=0x101010:s=1920x1080:r=24:d=%d" % SOURCE_SECONDS,
        "-f", "lavfi", "-i",
        "aevalsrc='if(" + flash + "\\,0.8*sin(2*PI*1000*t)\\,0)':s=48000:d=%d" % SOURCE_SECONDS,
        "-vf", "drawbox=x=0:y=0:w=iw:h=ih:color=white:t=fill:enable='%s'" % flash,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", "-pix_fmt", "yuv420p", "-r", "24",
        "-c:a", "aac", "-b:a", "128k", "-shortest", str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    return result.returncode == 0 and path.exists() and path.stat().st_size > 0


def _flash_times(path: Path, fps: float = 30.0) -> list:
    data = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vf", "fps=30,scale=96:54,format=gray",
         "-f", "rawvideo", "-"],
        capture_output=True, timeout=300,
    ).stdout
    frames = np.frombuffer(data[: len(data) // (96 * 54) * 96 * 54], dtype=np.uint8).reshape(-1, 54, 96)
    bright = frames.mean(axis=(1, 2)) > 128
    times = []
    index = 0
    while index < len(bright):
        if bright[index]:
            end = index
            while end < len(bright) and bright[end]:
                end += 1
            times.append((index + end - 1) / 2.0 / fps)
            index = end
        else:
            index += 1
    return times


def _beep_times(path: Path) -> list:
    data = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
        capture_output=True, timeout=300,
    ).stdout
    samples = np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
    hop = 160
    count = len(samples) // hop
    rms = np.sqrt((samples[: count * hop].reshape(count, hop) ** 2).mean(axis=1))
    loud = rms > 0.05
    times = []
    index = 0
    while index < len(loud):
        if loud[index]:
            end = index
            while end < len(loud) and loud[end]:
                end += 1
            times.append((index + end - 1) / 2.0 * hop / 16000.0)
            index = end
        else:
            index += 1
    return times


class TestRenderPreservesAvSync(unittest.TestCase):
    def test_flash_and_beep_stay_aligned_through_the_render_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            working = Path(directory)
            source = working / "source.mp4"
            output = working / "rendered.mp4"
            self.assertTrue(_build_sync_source(source), "could not build the sync source")

            framing = FramingDecision(
                face_count=1,
                mode="multi_shot_dynamic",
                shots=[ShotPlan(start=0.0, end=11.0, mode="portrait_face",
                                crop_x=657, crop_y=0, crop_w=607, crop_h=1080)],
            )
            render_viral_clip(
                source_video_path=source,
                start_time=CLIP_START_S,
                end_time=CLIP_END_S,
                output_clip_path=output,
                framing=framing,
                ass_subtitle_path=None,
                burn_subtitles=False,
            )

            flashes = _flash_times(output)
            beeps = _beep_times(output)
            self.assertGreaterEqual(len(flashes), 4, f"flashes detected: {flashes}")
            self.assertGreaterEqual(len(beeps), 4, f"beeps detected: {beeps}")

            pairs = min(len(flashes), len(beeps))
            offsets = [beeps[i] - flashes[i] for i in range(pairs)]
            for offset in offsets:
                self.assertLessEqual(
                    abs(offset),
                    TOLERANCE_S,
                    msg=(
                        f"A/V event offset {offset * 1000:+.0f}ms exceeds "
                        f"{TOLERANCE_S * 1000:.0f}ms; flashes={flashes} beeps={beeps}"
                    ),
                )
            mean_offset = float(np.mean(offsets))
            self.assertLessEqual(
                abs(mean_offset),
                TOLERANCE_S,
                msg=f"mean A/V offset {mean_offset * 1000:+.0f}ms over {pairs} events",
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
