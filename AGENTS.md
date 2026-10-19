# AGENTS.md — V2 Streamer Pipeline

> **SCOPE:** This repository is the **V2 streamer pipeline only** (US live
> streamers: Kai Cenat, IShowSpeed, Jynxzi, CaseOh, xQc, Adin Ross). The V1
> podcast pipeline lives in a separate repository. Do not add V1 podcast logic
> here.

---

## Architecture

```
main_streamer.py  ->  streamers/pipeline.py  ->  core/*  (vendored engine)
```

* `streamers/` — all V2-specific logic. This is where streamer changes belong.
* `core/` — the engine, **vendored** into this repository as an ordinary
  directory (no submodule, nothing to initialise). It is a committed copy of
  [`JackPro2121/clips-core`](https://github.com/JackPro2121/clips-core) at tree
  `fef54c3f`. A plain `git clone` is all that is required:

```bash
git clone https://github.com/JackPro2121/streamer-clips-to-social.git   # no --recurse-submodules
```

V2 may edit `core/` directly. When a fix is engine-wide, apply it upstream in
`clips-core` as well so the V1 podcast pipeline receives it. No CI checkout sets
`submodules: true` any more. See [`SYNC.md`](SYNC.md) for the drift check and the
upstream-sync procedure.

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
7. **The gate fails closed.** If a required visual check cannot run at all (cv2
   absent, frames undecodable) the clip is **blocked**, never published with a
   warning. `STRICT_VISUAL_CHECKS=0` is an explicit, temporary, audited override
   — not a fix. A check that silently skipped is indistinguishable from one that
   passed in every log, artifact and metric downstream.
8. **Fail closed on target resolution.** If no Buffer channel resolves, or a
   configured id is not in the account, publishing refuses and names the wrong
   ids. Never fall back to "broadcast to everything connected" — a config typo
   must not post to accounts nobody selected.
9. **Never retry an ambiguous publish.** Buffer's `createPost` has no
   idempotency key, so retry only failures that provably did not create a post.
   A timeout or unparseable response is recorded as `unknown` for a human to
   reconcile; retrying it can double-post.
10. **Bound disk growth.** `streamers/retention.py` expires old run directories,
    download working copies, subtitle stylesheets and derived render artifacts at
    the end of every run. Rendered clips and thumbnails are deliverables and are
    never deleted there.
11. **Untrusted text never becomes a path.** Titles and captions come from an LLM
    and a third party's VOD, so every filename goes through
    `streamers/paths.py` (`slugify_title` / `safe_output_name` / `assert_within`).
    A raw `f"{title}.mp4"` is a path-traversal and an invalid-Windows-name bug.
12. **Gate invariants for streamers.**
    * **Face stability**: continuity breaks on shot cuts and multi-face hops so
      ordinary camera cuts do not register as face jumps; within-shot jumps are
      capped at <= 1.25 face widths.
    * **Visual black margins**: edge bars are measured relative to full frame
      dimensions (1080×1920) with a 3.5% threshold to block real letterbox/pillarbox
      defects while accepting natural webcam bezels and room ceilings.
    * **Camera sway**: disabled (`with_sway=False`) for streamer clips because
      chaotic stream motion and velocity punch zooms are viral features, not defects.
    * **Captions on faces**: split-screen layouts place subtitle baselines at
      `MarginV = 980` (above the `960` divider line in the top speaker's lower chest),
      guaranteeing 0% overlap with both the top and bottom speakers' faces. Top
      `TikTokMemeBox` is suppressed in split-screen/stacked modes.
13. **Daily 2-clip cron schedule.**
    * Trigger 1: `16:30 UTC` (12:30 PM EDT / 09:30 PM PKT) for target queue slot `02:00 PM EDT`.
    * Trigger 2: `22:30 UTC` (06:30 PM EDT / 03:30 AM PKT) for target queue slot `08:00 PM EDT`.
    * Guaranteed >= 1 hour lead time before scheduled social publishing.

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

Policy variables (not secrets): `STRICT_VISUAL_CHECKS` (default `1`) — see hard
rule 7.

---

## Workflow for changes

1. **Plan** — for anything non-trivial, write down the approach and the risks.
2. **Tests first** — add or extend a regression test that fails before the fix.
   The suite is `python -m pytest tests/ -q`.
3. **Implement** — keep changes inside `streamers/` unless the fix belongs in
   the shared engine.
4. **Verify** — run the full gate, not just the tests:
   ```bash
   python -m pytest tests/ -q --cov=streamers --cov-fail-under=45
   python -m ruff check --select E9,F63,F7,F82,F401,F811,F841,F501 streamers/ main_streamer.py tests/
   ```
   Then `python main_streamer.py --help` and, where relevant, a real clip render.
   `core/` is excluded from both: it is the vendored engine and must stay
   byte-identical to upstream (`SYNC.md`).
5. **Review** — check the diff for secrets, safe-zone violations and hashtag
   bounds before committing.
6. **Commit** — conventional commits (`feat:`, `fix:`, `refactor:`, `chore:`,
   `test:`, `docs:`), with a short body explaining the why.

---

## Model weights

* Tracked in git: `core/models/face_detection_yunet.onnx` (232 KB) and the V2
  `core/models/streamer_profiles/*.npy` identity embeddings.
* Auto-downloaded on first use: SFace (`face_recognition_sface_2021dec.onnx`).
* Gitignored, fetched via `python core/tools/fetch_models.py --fetch all`:
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
│   ├── gate.py                    # publish gate (verification verdict, fails closed)
│   ├── paths.py                   # untrusted-title → safe filename (allowlist + traversal guard)
│   ├── retention.py               # bounded artifact retention sweep
│   └── spec.py                    # numeric spec overlay & validate_spec() invariants
├── core/                          # vendored engine (committed copy of clips-core)
├── assets/                        # fonts, sfx, audio beds
├── tools/
│   └── build_streamer_profile.py  # [V2] Streamer identity profiles (shared tools live in core/tools/)
├── tests/                         # V2 + shared-engine tests
├── data/                          # streamer_history.json, streamer_cursor.json
└── .github/workflows/             # streamer_clipper.yml, tests.yml
```
