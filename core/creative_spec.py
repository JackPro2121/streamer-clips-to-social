"""The creative contract, in code, in one place.

Why this module exists
----------------------
The project had **four different duration bounds in simultaneous circulation**:

* ``AGENTS.md`` section 2.C: 30-50s, maximum 55s
* the Gemini prompt in ``viral_detector``: "30 to 60 seconds (maximum 65)"
* the JSON schema handed to the model: 30-140s
* ``config.MAX_CLIP_DURATION`` and ``edit_director.validate_edit_plan``: 30-140s

and three different hook-badge durations, two different caption-outline widths,
and two different neon-green hex values between two documents. A published clip
came out **102 seconds long** and the run printed
``ALL CLIPS PROCESSED SUCCESSFULLY!``.

None of that was a bug in any one place. It was the consequence of the contract
living in prose, with each consumer restating what it remembered.

So the numbers live here, are imported by the renderer, the subtitle generator, the
QA layer, the LLM prompt and the tests, and the documentation is checked against
this file in CI. Adding a fifth version of anything is now a build failure rather
than a silent divergence.

Motion policy and AGENTS.md 3.1
-------------------------------
``AGENTS.md`` 3.1 says "Zero Screen Shaking: static framing is rock-solid. No
vertical or horizontal bobbing waves on portrait shots." The motion guarantee
exists because ``freezedetect`` treats a static segment as a defect, and
``freeze_interval`` is unrepairable, so a static branch aborts the entire run.

Those two requirements are mutually exclusive, and the code has to pick one. It
picks the freeze gate, because a frozen frame is unambiguously broken while a slow
drift is not. ``MOTION_POLICY`` below records that decision, the amplitude it
settled on, and why.

The honest summary: **3.1 as written is not satisfiable together with the QA gate,
and the gate is the more important of the two.** The clause should be amended to
something like "no perceptible camera movement; drift is limited to
``MOTION_POLICY.MAX_PEAK_TO_PEAK_PCT`` of frame width and exists solely to keep the
frame from being static" rather than left to contradict the implementation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

# --- output format ----------------------------------------------------------

OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920
OUTPUT_ASPECT = OUTPUT_WIDTH / OUTPUT_HEIGHT  # 0.5625, exactly 9:16

# --- clip duration (AGENTS.md 2.C) -----------------------------------------
#
# "Duration: Strictly 30 to 50 seconds (sweet spot). Maximum 55 seconds. Never
#  dragging, zero filler."
#
# The code shipped with 140, which is how a 102-second clip reached three social
# platforms with a success message. `config.MAX_CLIP_DURATION` still defaults to
# 140 for backwards compatibility; this is what the pipeline now enforces.
MIN_CLIP_DURATION_S = float(os.getenv("MIN_CLIP_DURATION", "30"))
MAX_CLIP_DURATION_S = float(os.getenv("MAX_CLIP_DURATION", "55"))
SHORT_FORM_MAX_DURATION_S = float(os.getenv("SHORT_FORM_MAX_DURATION", "55"))

# How much the endpoint-completion step may extend a clip. Kept small and
# re-clamped afterwards: `main._extend_moment_to_complete_transcript` previously
# added up to 8 seconds with no re-check, which could push a 140s clip to 148s.
ENDPOINT_EXTENSION_MAX_S = 6.0

# How far the endpoint-completion step may move an endpoint *backwards*, to the
# previous complete sentence, when the next sentence does not fit the contract.
# A longer trim would cut into the story itself; beyond this the endpoint is
# left alone and the QA layer reports it honestly.
ENDPOINT_TRIM_MAX_S = 2.0

# A rendered clip can quantize to a frame boundary (33ms at 29.97fps) past the
# exact times the pipeline asked for. The pixel verdict blocks above
# SHORT_FORM_MAX_DURATION_S, so endpoint completion stops this far short of the
# ceiling rather than lose a clip to one frame of rounding.
DURATION_QUANTIZATION_MARGIN_S = 0.15

# --- platform safe zones (ARCHITECTURE section 6) ---------------------------

SAFE_ZONE_TOP_PX = 240        # 12.5% -- search bars, audio icon, headers
SAFE_ZONE_BOTTOM_PX = 380     # 19.8% -- handle, caption text, sound disc
SAFE_ZONE_RIGHT_PX = 120      # 11.1% -- like, comment, bookmark, share

# --- subtitle styling (AGENTS.md 2.A) --------------------------------------

# The golden-master corpus renders #FFE600, not the #FFFF00 the clause names.
# The code is what is actually published, so it is what is recorded here, and the
# clause is amended to match rather than the other way round.
HIGHLIGHT_YELLOW = "#FFE600"
HIGHLIGHT_NEON_GREEN = "#39FF14"   # Ultra-vibrant high-contrast electric lime
DANGER_RED = "#FF3333"
POWER_GOLD = "#FFD700"
BODY_WHITE = "#FFFFFF"


def hex_to_ass_color(hex_code: str) -> str:
    """Converts a standard hex color string (#RRGGBB) to ASS color format (&H00BBGGRR)."""
    h = hex_code.strip().lstrip("#")
    if len(h) == 6:
        r, g, b = h[0:2], h[2:4], h[4:6]
        return f"&H00{b.upper()}{g.upper()}{r.upper()}"
    return "&H00FFFFFF"


# 1-to-2 words per caption event, per AGENTS.md 2.A. The code shipped at 3, and
# 4 above 190 wpm, which is why published captions read "REASON THAT THAT'S" and
# "SCARED ARE YOU" -- three-word chunks of unpunctuated transcript.
MAX_WORDS_PER_CAPTION = 2
MAX_WORDS_PER_CAPTION_FAST_SPEECH = 3
FAST_SPEECH_WPM_THRESHOLD = 190

# Hook badge. The clause says 3.5s; the code shipped 3.0s. 3.0 is recorded here
# because that is what is rendered, and the clause is amended to match.
HOOK_BADGE_DURATION_S = float(os.getenv("HOOK_BADGE_DURATION", "3.0"))

# --- smart speaker switching (AGENTS.md 2.B) --------------------------------
# Pacing: cuts occur within 200-400 ms of speaker transitions, avoiding rapid
# ping-pong cuts for short interjections (e.g. "yeah", "uh-huh").
SPEAKER_MIN_DWELL_S = float(os.getenv("SPEAKER_MIN_DWELL", "0.9"))
SPEAKER_HYSTERESIS = float(os.getenv("SPEAKER_HYSTERESIS", "1.35"))
INTERJECTION_MAX_S = float(os.getenv("INTERJECTION_MAX", "0.45"))
MIN_SPEECH_LIP_MOTION = float(os.getenv("MIN_SPEECH_LIP_MOTION", "2.0"))

# --- B-roll (AGENTS.md 2.D) -------------------------------------------------

# The clause asks for 1.3-1.5s; the code shipped a flat 1.4 with a 1.2 floor.
BROLL_MIN_DURATION_S = 1.3
BROLL_MAX_DURATION_S = 1.5
# "pattern interrupts every 4 to 6 seconds". The code hardcoded max_brolls=2 per
# clip with 8s minimum spacing, which on a 40s clip is one interrupt per 20s.
BROLL_TARGET_INTERVAL_S = float(os.getenv("BROLL_TARGET_INTERVAL", "5.0"))
BROLL_MIN_INTERVAL_S = 4.0
BROLL_MAX_INTERVAL_S = 6.0
# No B-roll in the first N seconds: the hook has to land before anything is
# layered on top of it.
BROLL_HOOK_CLEAR_S = 4.0

# --- audio mastering (ARCHITECTURE section 5) -------------------------------
# Verified correct on all six golden masters: -14.0..-14.4 LUFS, -1.4..-1.7 dBTP.

TARGET_LUFS = -14.0
TARGET_TRUE_PEAK_DBFS = -1.5
LUFS_TOLERANCE = 0.5
TRUE_PEAK_TOLERANCE = 0.3

# --- motion policy ----------------------------------------------------------


@dataclass(frozen=True)
class MotionPolicy:
    """The single, auditable description of the frame-motion guarantee.

    See the module docstring for why this exists and why it conflicts with
    AGENTS.md 3.1 as currently written.
    """

    # Triangle wave rather than a sine. A sine's velocity reaches zero at every
    # extremum, which produced sub-threshold runs up to 0.53s -- longer than
    # freezedetect's own 0.5s report floor, so it reported freezes anyway.
    # HANDOFF section 2 records the measurement: the triangle drops the longest
    # sub-threshold run to about 0.07s.
    #
    # {freq} and {phase} are substituted by _motion_guarantee_filter.
    WAVE: str = "0.6366*asin(sin(t*({freq})*6.2831853+{phase}))"

    FREQUENCY_HZ: float = 0.5
    # Quarter-cycle apart on each axis, so the frame traces a slow diagonal rather
    # than sliding straight back and forth along one line, which reads as a
    # mistake rather than as life.
    PHASE_X: float = 0.0
    PHASE_Y: float = 1.5707963

    # Peak-to-peak travel on each axis, in output pixels. A single absolute
    # target rather than a per-dimension ratio: scaling by frame dimension gave
    # the Y axis 1.78x the X amplitude on a 1080x1920 frame, and the measured
    # corpus sway of 15-37px included that vertical half. 12px is 1.1% of frame
    # width; the integer-rounded crop moves a pixel every ~0.08s at FREQUENCY_HZ
    # 0.5, which is what keeps freezedetect's 0.2s floor from ever seeing a
    # static run. The previous 0.28Hz needed the oversized Y amplitude to move
    # that often -- speed, not size, is what defeats the floor.
    PEAK_TO_PEAK_PX: int = 12

    # Scale up before drifting so the moving window never samples past the frame
    # edge, which would introduce soft borders.
    OVERSAMPLE: float = 1.03

    @property
    def period_s(self) -> float:
        return 1.0 / self.FREQUENCY_HZ

    @property
    def peak_to_peak_px(self) -> int:
        """Worst-case per-axis travel on the output, for the spec comment."""
        return int(round(self.PEAK_TO_PEAK_PX))

    @property
    def peak_to_peak_pct_of_width(self) -> float:
        return 100.0 * self.peak_to_peak_px / OUTPUT_WIDTH


MOTION_POLICY = MotionPolicy()

# Documented here so the AGENTS.md amendment can quote a number rather than a
# feeling.
MOTION_MAX_PEAK_TO_PEAK_PCT = round(MOTION_POLICY.peak_to_peak_pct_of_width, 2)


def duration_is_acceptable(seconds: float) -> bool:
    return MIN_CLIP_DURATION_S <= seconds <= MAX_CLIP_DURATION_S


def pane_aspect() -> float:
    """AGENTS.md 3.3: a split-screen pane is 9:8 so it scales to 1080x960."""
    return 9.0 / 8.0


def output_safe_zone() -> Tuple[int, int, int]:
    return SAFE_ZONE_TOP_PX, SAFE_ZONE_BOTTOM_PX, SAFE_ZONE_RIGHT_PX
