# Test fixtures corpus

This directory is the **media corpus** for V2 verification. It was called out in
the second V2 audit: `core.verification.verdict.verify_corpus` existed but was
invoked by nothing, and this directory did not exist — so every gate test ran
against mocks and no test had ever asserted anything about real decoded pixels.

## What belongs here

Real, small, committed media that pins render and gate behaviour:

| File pattern        | Purpose                                                        |
| ------------------- | -------------------------------------------------------------- |
| `*_full_frame.mp4`  | Valid 1080x1920 full-frame clip — must pass the visual audit    |
| `*_letterboxed.mp4` | Clip with black bars — must be blocked (`v2_black_frame_margin`)|
| `*_split.mp4`       | Two-speaker split-screen composition                            |
| `*.json`            | Expected `verdict.json` metrics for the clip beside it          |

## Rules

* **Keep it tiny.** A few seconds at low frame rate is enough — the audits sample
  up to 12 frames, not the whole clip. CI installs ffmpeg, so anything larger is
  pure download cost on every run.
* **No source VODs.** Never commit podcast/stream footage. Fixtures are
  synthesised (`ffmpeg -f lavfi`) or hand-built colour fields.
* **No secrets, no third-party watermarks, no faces from real people.**
* Generated fixtures are preferred over committed binaries. If a case can be
  produced deterministically by ffmpeg at test time, generate it — see
  `tests/test_v2_gate_e2e.py`, which currently synthesises its clips in a
  `TemporaryDirectory` rather than committing them here.

## Current state

`tests/test_v2_gate_e2e.py` closes the immediate gap: it synthesises full-frame
and letterboxed clips with ffmpeg and runs the **real** `verify_clip` path
(real `cv2.VideoCapture`, real `verdict.verify`) over real decoded frames, so a
regression that stops the visual audit from running is caught in CI.

Growing this directory into a checked-in corpus that `verify_corpus` walks is the
follow-up that raises the `--cov-fail-under` ratchet in
`.github/workflows/tests.yml` toward the 80% standard.
