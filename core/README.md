# clips-core

Shared engine for the vertical-video clip pipelines:

- **V1 — podcast clipper** (`podcasts-clips-to-social`)
- **V2 — streamer clipper** (`streamer-clips-to-social`)

## What this repo is

This repository contains **only** the shared `core` Python package: media
download, transcription, face tracking, layout/composition primitives, subtitle
and thumbnail generation, the FFmpeg video editor, viral-moment detection, and
the `core.verification` quality-gate package. It contains no pipeline-specific
logic — the podcast and streamer pipelines live in their own repos.

## How it is consumed

It is **not** a published package. It is consumed as a **git submodule** mounted
at `core/` in each consumer repo, so `import core.*` works with no install step:

```bash
# in the consumer repo
git submodule update --init --recursive
```

GitHub Actions checkouts must enable submodules:

```yaml
- uses: actions/checkout@v4
  with:
    submodules: true
```

## Configuration

The package resolves its paths at runtime so it works both in a repo checkout
and from an installed location. All are environment-overridable:

| Variable | Default | Purpose |
| --- | --- | --- |
| `CLIPS_CORE_ROOT` | inferred repo root, else CWD | Consumer project root owning `assets/`, `clips/`, `data/`, `downloads/` |
| `CLIPS_CORE_ASSETS_DIR` | `<root>/assets` | Fonts, SFX, audio beds, b-roll |
| `CLIPS_CORE_MODELS_DIR` | `models/` beside the package, else `<root>/models` | ONNX / `.pth` model weights |

## Model weights

Only two things are tracked in `models/`:

- `face_detection_yunet.onnx` (232 KB) — the one model the pipeline cannot run
  without, so it ships with a fresh clone.
- `streamer_profiles/*.npy` — small V2 SFace identity embeddings.

All large weights (SFace, S3FD/SyncNet, TransNetV2) are **gitignored** and
fetched by `core/tools/fetch_models.py` in the consumer repos, pinned by SHA-256.
