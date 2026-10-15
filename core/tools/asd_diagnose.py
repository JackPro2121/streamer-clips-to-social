"""ASD-Lite diagnostic: does the visible mouth move with the voice?

The "video does not match voice" class from run 37347481817 (clip 2,
THE LONELINESS EPIDEMIC OF EARLY RETIREMENT): the show is multi-speaker
(8 faces detected) with zero camera cuts and no face tracking, so the crop
switches people on geometry, not on who is speaking. Voice and captions come
from the transcript, so they agree with each other; the picture can show the
listening face.

This measures it on a rendered clip: per frame, YuNet faces are tracked
greedily; each track's mouth region (lower third) motion is compared with the
audio RMS envelope. A track whose mouth motion does not track the voice while
speech is present is a mismatch window.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

# The pipeline repo root must be importable for ``core.verification``.
_ENGINE_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _ENGINE_ROOT.parent if _ENGINE_ROOT.name == "core" else _ENGINE_ROOT
sys.path.insert(0, str(_REPO_ROOT))

from core.verification import av_sync, faces as faces_mod  # noqa: E402

WIDTH, HEIGHT = 480, 270


def extract_frames(clip: Path, fps: int = 10) -> list[np.ndarray]:
    result = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(clip),
            "-vf", f"fps={fps},scale={WIDTH}:{HEIGHT}",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
        ],
        capture_output=True,
    )
    raw = result.stdout
    size = WIDTH * HEIGHT * 3
    return [
        np.frombuffer(raw[i * size:(i + 1) * size], dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)
        for i in range(len(raw) // size)
    ]


def audio_envelope(clip: Path, hop_s: float) -> np.ndarray:
    samples = av_sync._decode_mono(clip, 16000)
    if samples is None or samples.size == 0:
        return np.zeros(0)
    hop = max(1, int(16000 * hop_s))
    return np.asarray([
        float(np.sqrt(np.mean(samples[i:i + hop] ** 2)))
        for i in range(0, max(1, samples.size - hop), hop)
    ])


def mouth_patch(grey: np.ndarray, box) -> np.ndarray:
    import cv2

    x0 = int(max(0, box.x))
    y0 = int(max(0, box.y + box.height * 0.55))
    x1 = int(min(grey.shape[1], box.x + box.width))
    y1 = int(min(grey.shape[0], box.y + box.height))
    patch = grey[y0:y1, x0:x1]
    if patch.size == 0:
        return np.zeros((10, 20), dtype=np.float32)
    return cv2.resize(patch, (20, 10)).astype(np.float32)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.size, b.size)
    if n < 8:
        return 0.0
    a, b = a[:n].astype(np.float64), b[:n].astype(np.float64)
    a, b = a - a.mean(), b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0


def analyse(clip: Path, fps: int = 10) -> None:
    frames = extract_frames(clip, fps=fps)
    if not frames:
        print("no frames decoded")
        return
    envelope = audio_envelope(clip, 1.0 / fps)
    detector, _ = faces_mod._load_yunet()
    if detector is None:
        print("YuNet unavailable")
        return

    tracks: list[dict] = []
    for index, frame in enumerate(frames):
        grey = frame.mean(axis=2)
        boxes = faces_mod.detect_in_frame(frame, detector) or []
        used = set()
        for track in tracks:
            last = track["last"]
            best, best_dist = None, 1e9
            for box_index, box in enumerate(boxes):
                if box_index in used:
                    continue
                dist = abs((box.x + box.width / 2) - (last.x + last.width / 2)) + abs(
                    (box.y + box.height / 2) - (last.y + last.height / 2)
                )
                if dist < best_dist:
                    best, best_dist = box_index, dist
            if best is not None and best_dist < 0.25 * WIDTH:
                used.add(best)
                box = boxes[best]
                patch = mouth_patch(grey, box)
                track["motion"].append(float(np.mean(np.abs(patch - track["prev"]))))
                track["prev"] = patch
                track["last"] = box
                track["x"].append(box.x + box.width / 2)
        for box_index, box in enumerate(boxes):
            if box_index not in used and len(tracks) < 4:
                tracks.append(
                    {
                        "motion": [],
                        "prev": mouth_patch(grey, box),
                        "last": box,
                        "x": [box.x + box.width / 2],
                        "born": index,
                    }
                )

    print(f"clip: {clip.name}")
    print(f"frames={len(frames)} audio_hop_mean={envelope.mean():.4f}")
    for order, track in enumerate(sorted(tracks, key=lambda t: -len(t["motion"]))[:3]):
        motion = np.asarray(track["motion"])
        corr = pearson(motion, envelope[track["born"] + 1:])
        print(
            f"  track #{order}: frames={len(motion):4d} mean_x={np.mean(track['x']):6.1f} "
            f"mouth_motion={motion.mean():.2f} corr_with_voice={corr:+.2f}"
        )

    # Lag scan on the dominant track: a peak at a non-zero lag means the video
    # content is time-shifted against the audio (segment-cut class); no peak
    # anywhere means the visible face is simply not the speaker (ASD class).
    if tracks:
        top = max(tracks, key=lambda t: len(t["motion"]))
        motion = np.asarray(top["motion"])
        env = envelope[top["born"] + 1:]
        scored = []
        for lag in range(-50, 51):
            if lag >= 0:
                corr = pearson(motion[lag:], env[: max(1, env.size - lag)])
            else:
                corr = pearson(motion[: max(1, motion.size + lag)], env[-lag:])
            scored.append((corr, lag))
        scored.sort(key=lambda pair: -abs(pair[0]))
        for corr, lag in scored[:3]:
            # A peak at lag>0 pairs mouth motion with audio from lag frames
            # earlier: the picture shows older content than the sound.
            direction = "video lags audio" if lag > 0 else "video leads audio"
            print(
                f"  lag {lag * 0.1:+.1f}s corr={corr:+.2f} "
                f"({direction if lag else 'aligned'})"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("clip")
    args = parser.parse_args()
    analyse(Path(args.clip))
