# Streamer Clips to Social (V2)

Autonomous clipper for **US live streamers** — extracts high-retention TikTok /
Shorts / Reels clips from 3–5 hour live streams (Kai Cenat, IShowSpeed, Jynxzi,
CaseOh, xQc, Adin Ross).

> This is the **V2 streamer pipeline only**. The V1 podcast pipeline lives in a
> separate repository (`podcasts-clips-to-social`). This repo is fully
> self-contained: its engine `core/` is a **vendored** copy of
> [`clips-core`](https://github.com/JackPro2121/clips-core) committed directly
> into this repository — no submodule, no shared remote. Nothing is shared with
> V1 except the upstream engine lineage.

---

## What it produces

| Property | Value |
| --- | --- |
| Aspect / size | 9:16, 1080×1920 |
| Duration | **30–45 s** per clip |
| Captions | Kinetic TikTok-style (`MarginV=980` in split-screen to guarantee zero face collision) |
| Framing | SFace identity lock + 3-tier state machine, auto-stacked webcam/gameplay |
| Caption hashtags | **2–4** (never fewer, never more) |
| Publishing | Buffer via **isolated** `STREAMER_BUFFER_*` credentials (`@kaicenatclipsuncut`) |
| Daily Schedule | 2 clips daily (`16:30` & `22:30` UTC), queued >= 1 hour ahead of `02:00 PM` & `08:00 PM` EDT slots |
| Quality Gates | Zero-drift lip-sync, -14.0 LUFS audio, face stability, 3.5% black margin audit |

---

## Architecture

```
streamer-clips-to-social/
├── main_streamer.py               # CLI entrypoint
├── streamers/                     # V2 engine (this repo)
│   ├── pipeline.py                # orchestrator
│   ├── discovery.py               # YouTube RSS + Firecrawl discovery, history/cursor
│   ├── viral_detector.py          # Gemini streamer viral-moment finder
│   ├── face_tracker.py            # SFace identity locking + 3-tier state machine
│   ├── layout_compositor.py       # stacked webcam / multi-person / single-room layouts
│   ├── audio_analyzer.py          # decibel scream & energy peaks
│   ├── chat_analyzer.py           # live-chat velocity (MPS) & emotes
│   ├── editing_enhancer.py        # velocity hook + pattern interrupts
│   ├── censor.py                  # profanity / TOS masking
│   ├── gate.py                    # publish gate (verification verdict)
│   └── spec.py                    # V2 numeric spec overlay & invariant asserts
├── core/                          # vendored engine (committed copy of clips-core)
├── assets/                        # fonts, sfx, audio beds
├── tools/
│   └── build_streamer_profile.py  # [V2] Identity profiles (shared tools live in core/tools/)
├── tests/                         # V2 tests + shared-engine tests
├── data/                          # streamer_history.json, streamer_cursor.json
└── .github/workflows/             # streamer_clipper.yml, tests.yml
```

### The `core/` engine (vendored)

The engine is committed directly into this repository — there is no submodule
and nothing to initialise:

```bash
git clone https://github.com/JackPro2121/streamer-clips-to-social.git   # that is all
```

It is a copy of `clips-core` (tree `fef54c3f`). Large model weights (SFace,
S3FD/SyncNet, TransNetV2) are gitignored inside `core/`. The 232 KB YuNet
detector ships in git; SFace is auto-downloaded on first use. Everything else is
fetched by `core/tools/fetch_models.py` (SHA-256 pinned).

> `core/` is kept byte-identical to the upstream `clips-core` tree. Its
> `core/README.md` still documents the standalone `clips-core` repo (including
> submodule instructions) — that is intentional, so `git diff` against the
> upstream tree stays clean. Ignore the submodule steps there; this repo needs
> none.

To check for upstream drift or pull engine changes in, see [`SYNC.md`](SYNC.md).

---

## Quickstart

```bash
git clone https://github.com/JackPro2121/streamer-clips-to-social.git
cd streamer-clips-to-social
python -m pip install -r requirements.txt     # or: bash core/tools/install_deps.sh
cp .env.example .env                           # then fill in the keys
```

### CLI

```bash
# Auto-discover the latest stream across the default streamer set:
python main_streamer.py --num-clips 3

# Target one streamer:
python main_streamer.py --streamer kai_cenat --num-clips 3
python main_streamer.py --streamer ishowspeed --num-clips 3

# Process a specific VOD:
python main_streamer.py --video-url "https://www.youtube.com/watch?v=VIDEO_ID" --num-clips 3

# Force a framing mode (auto | single_smooth | split_screen | stacked_webcam | fullscreen_room):
python main_streamer.py --video-url "https://www.youtube.com/watch?v=VIDEO_ID" --framing auto

# Publish to the isolated streamer Buffer queue:
python main_streamer.py --video-url "https://www.youtube.com/watch?v=VIDEO_ID" --post-to-buffer
```

---

## Configuration & secrets

Set these locally in `.env` (gitignored) and in **GitHub → Settings → Secrets
and variables → Actions** for CI.

| Secret | Required | Purpose |
| --- | :---: | --- |
| `GEMINI_API_KEY` | **Yes** | Viral-moment extraction (Google AI Studio) |
| `GEMINI_API_KEY_2..4` | Optional | Key rotation on rate limits |
| `APIFY_API_TOKEN` | Optional | Residential-proxy fallback for YouTube |
| `FIRECRAWL_API_KEY` | Optional | Real-time streamer VOD search |
| `GROQ_API_KEY` | Optional | Secondary LLM tier |
| `PEXELS_API_KEY` | Optional | B-roll |
| `STREAMER_BUFFER_ACCESS_TOKEN` | Optional | **Isolated** streamer publish token |
| `STREAMER_BUFFER_CHANNEL_IDS` | Optional | **Isolated** streamer channel list |
| `STREAMER_WATERMARK` | Optional | Watermark text (default `@streamersclips`) |
| `GITHUB_TOKEN` | auto | Release uploads for direct asset URLs |

Engine paths can be overridden with `CLIPS_CORE_ROOT`, `CLIPS_CORE_MODELS_DIR`
and `CLIPS_CORE_ASSETS_DIR` (see `core/README.md`).

### Operational levers (variables, not secrets)

| Variable | Default | Effect |
| --- | --- | --- |
| `STRICT_VISUAL_CHECKS` | `1` | The publish gate **fails closed** when a required visual check cannot run at all (cv2 missing, frames undecodable). Setting this to `0` downgrades that to a warning. Treat `0` as an explicit, temporary, audited override: a clip that cannot be verified must not reach a live account. |

> **Never commit secrets.** They belong in `.env` (gitignored) and GitHub
> Actions secrets only.

---

## Tests

```bash
python -m pytest tests/ -q                                   # unit + integration
python -m pytest tests/ -q --cov=streamers --cov-report=term # with coverage
python -m ruff check --select E9,F63,F7,F82,F401,F811,F841,F501 streamers/ main_streamer.py tests/
```

`tests.yml` runs all three on every push and pull request, with a coverage floor
on the V2-owned `streamers/` surface and a lint gate on V2-owned code.

* `core/` is the **vendored** clips-core engine and must stay byte-identical to
  upstream, so it is deliberately excluded from lint and from the coverage floor.
  Fixes there land upstream and are pulled in (see `SYNC.md`).
* `tests/test_v2_gate_e2e.py` synthesises real 9:16 clips with ffmpeg and runs the
  **real** publish gate over real decoded frames — every other gate test mocks the
  verdict, which is why the fail-closed policy needed a real-media test to pin it.
  See `tests/fixtures/README.md` for the corpus contract.

---

## GitHub Actions

| Workflow | Trigger | Purpose |
| --- | --- | --- |
| `streamer_clipper.yml` | manual dispatch | Discover → clip → render → gate → optionally publish; persists history/cursor |
| `tests.yml` | push / PR | Runs the test suite |

The clipper is manual-dispatch by design, so a run is always an explicit,
budgeted decision. It writes processed VOD IDs and the round-robin cursor back
to `data/` with `[skip ci]`.
