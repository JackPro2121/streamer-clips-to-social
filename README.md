# Streamer Clips to Social (V2)

Autonomous clipper for **US live streamers** — extracts high-retention TikTok /
Shorts / Reels clips from 3–5 hour live streams (Kai Cenat, IShowSpeed, Jynxzi,
CaseOh, xQc, Adin Ross).

> This is the **V2 streamer pipeline only**. The V1 podcast pipeline lives in a
> separate repository (`podcasts-clips-to-social`). Both depend on the same
> shared engine, [`clips-core`](https://github.com/JackPro2121/clips-core),
> mounted here as a git submodule at `core/`. The two pipelines share **no**
> code or credentials through this repo.

---

## What it produces

| Property | Value |
| --- | --- |
| Aspect / size | 9:16, 1080×1920 |
| Duration | **30–45 s** per clip |
| Captions | Kinetic TikTok-style, upper safe-zone meme hook box |
| Framing | SFace identity lock + 3-tier state machine, auto-stacked webcam/gameplay |
| Caption hashtags | **2–4** (never fewer, never more) |
| Publishing | Buffer via **isolated** `STREAMER_BUFFER_*` credentials |

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
├── core/                          # git submodule -> clips-core (shared engine)
├── assets/                        # fonts, sfx, audio beds
├── tools/                         # fetch_models, build_streamer_profile, smoke tests
├── tests/                         # V2 tests + shared-engine tests
├── data/                          # streamer_history.json, streamer_cursor.json
└── .github/workflows/             # streamer_clipper.yml, tests.yml
```

### The `core/` submodule

The shared engine is not vendored here. There is exactly one copy of it:

```bash
git submodule update --init --recursive   # after cloning
```

Large model weights (SFace, S3FD/SyncNet, TransNetV2) are gitignored inside
`clips-core`. The 232 KB YuNet detector and the V2 streamer identity profiles
ship in git; SFace is auto-downloaded on first use. Everything else is fetched
by `tools/fetch_models.py` (SHA-256 pinned).

---

## Quickstart

```bash
git clone --recurse-submodules https://github.com/JackPro2121/streamer-clips-to-social.git
cd streamer-clips-to-social
python -m pip install -r requirements.txt     # or: bash tools/install_deps.sh
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
and `CLIPS_CORE_ASSETS_DIR` (see the clips-core README).

> **Never commit secrets.** They belong in `.env` (gitignored) and GitHub
> Actions secrets only.

---

## Tests

```bash
python -m pytest tests/ -q
```

`tests.yml` runs this on every push and pull request.

---

## GitHub Actions

| Workflow | Trigger | Purpose |
| --- | --- | --- |
| `streamer_clipper.yml` | manual dispatch | Discover → clip → render → gate → optionally publish; persists history/cursor |
| `tests.yml` | push / PR | Runs the test suite |

The clipper is manual-dispatch by design, so a run is always an explicit,
budgeted decision. It writes processed VOD IDs and the round-robin cursor back
to `data/` with `[skip ci]`.
