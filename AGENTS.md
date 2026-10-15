# AGENTS.md — V2 Streamer Pipeline

> **SCOPE:** This repository is the **V2 streamer pipeline only** (US live
> streamers: Kai Cenat, IShowSpeed, Jynxzi, CaseOh, xQc, Adin Ross). The V1
> podcast pipeline lives in a separate repository. Do not add V1 podcast logic
> here.

---

## Architecture

```
main_streamer.py  ->  streamers/pipeline.py  ->  core/*  (git submodule)
```

* `streamers/` — all V2-specific logic. This is where streamer changes belong.
* `core/` — the shared engine, a **git submodule** pointing at
  [`JackPro2121/clips-core`](https://github.com/JackPro2121/clips-core). Never
  edit files inside `core/` from this repo: change them in `clips-core`, then
  bump the submodule pointer here.

```bash
git submodule update --init --recursive      # after cloning
git -C core fetch origin && git -C core checkout <tag-or-sha>
git add core && git commit -m "chore: bump clips-core to <sha>"
```

Any CI checkout of this repo must set `submodules: true` (already done in
`.github/workflows/*.yml`).

---

## Hard rules

1. **Pipeline isolation.** Streamer publishing must read
   `STREAMER_BUFFER_ACCESS_TOKEN` / `STREAMER_BUFFER_CHANNEL_IDS` only. Never
   reuse the podcast (`BUFFER_ACCESS_TOKEN`) credentials.
2. **Hashtags: 2–4 per caption.** Never 0, never 5+.
3. **Safe zones.** Captions must never collide with mobile UI safe-zones
   (bottom 20%, top 15%, right 12%). The TikTok meme hook box stays in the
   upper safe-zone with drop shadow and rounded corners.
4. **No full-VOD downloads.** Fetch the transcript first and range-download
   30–45 s windows with `yt-dlp`. Never download a 15 GB VOD.
5. **Zero secrets in git.** Keys live in `.env` (gitignored) and GitHub Actions
   secrets only. Never log, print, or commit them.
6. **Output contract.** 30–45 s clips, 1080×1920, kinetic captions, verified by
   the publish gate before anything reaches Buffer.

---

## Environment variables

| Secret | Required? | Purpose |
| --- | :---: | --- |
| `GEMINI_API_KEY` | **Yes** | Viral-moment extraction |
| `GEMINI_API_KEY_2..4` | Optional | Key rotation on 429/503 |
| `APIFY_API_TOKEN` | Optional | YouTube proxy fallback |
| `FIRECRAWL_API_KEY` | Optional | Streamer VOD search |
| `GROQ_API_KEY` | Optional | Secondary LLM tier |
| `PEXELS_API_KEY` | Optional | B-roll |
| `STREAMER_BUFFER_ACCESS_TOKEN` | Optional | Isolated Buffer publish |
| `STREAMER_BUFFER_CHANNEL_IDS` | Optional | Isolated Buffer channels |
| `STREAMER_WATERMARK` | Optional | Watermark text |
| `GITHUB_TOKEN` | auto | Release asset uploads |

Engine paths: `CLIPS_CORE_ROOT`, `CLIPS_CORE_MODELS_DIR`, `CLIPS_CORE_ASSETS_DIR`.

---

## Workflow for changes

1. **Plan** — for anything non-trivial, write down the approach and the risks.
2. **Tests first** — add or extend a regression test that fails before the fix.
   The suite is `python -m pytest tests/ -q`.
3. **Implement** — keep changes inside `streamers/` unless the fix belongs in
   the shared engine.
4. **Verify** — run the tests; run `python main_streamer.py --help` and, where
   relevant, a real clip render.
5. **Review** — check the diff for secrets, safe-zone violations and hashtag
   bounds before committing.
6. **Commit** — conventional commits (`feat:`, `fix:`, `refactor:`, `chore:`,
   `test:`, `docs:`), with a short body explaining the why.

---

## Model weights

* Tracked in git: `core/models/face_detection_yunet.onnx` (232 KB) and the V2
  `core/models/streamer_profiles/*.npy` identity embeddings.
* Auto-downloaded on first use: SFace (`face_recognition_sface_2021dec.onnx`).
* Gitignored, fetched via `python tools/fetch_models.py --fetch all`:
  S3FD/SyncNet, TransNetV2.

---

## Directory reference

```
├── main_streamer.py               # CLI entrypoint
├── streamers/                     # V2 engine
│   ├── pipeline.py                # orchestrator
│   ├── discovery.py               # RSS/Firecrawl discovery + history & cursor
│   ├── viral_detector.py          # Gemini streamer viral-moment finder
│   ├── face_tracker.py            # SFace identity locking (3-tier state machine)
│   ├── layout_compositor.py       # stacked / multi-person / single-room layouts
│   ├── audio_analyzer.py          # scream & energy peaks
│   ├── chat_analyzer.py           # live-chat velocity (MPS) & emotes
│   ├── editing_enhancer.py        # velocity hook + pattern interrupts
│   ├── censor.py                  # profanity / TOS masking
│   ├── gate.py                    # publish gate (verification verdict)
│   └── spec.py                    # numeric spec overlay & invariant asserts
├── core/                          # git submodule -> clips-core
├── assets/                        # fonts, sfx, audio beds
├── tools/                         # model fetch, profile build, smoke tests
├── tests/                         # V2 + shared-engine tests
├── data/                          # streamer_history.json, streamer_cursor.json
└── .github/workflows/             # streamer_clipper.yml, tests.yml
```
