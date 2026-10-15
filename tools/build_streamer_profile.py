"""Streamer SFace Profile Builder (Tool).

Extracts and persists 128-D SFace (SphereFace / ArcFace ONNX) facial identity
embeddings for streamers in the roster.

Usage:
    python tools/build_streamer_profile.py --name "Jynxzi" --image path/to/jynxzi.jpg
    python tools/build_streamer_profile.py --name "CaseOh" --video path/to/caseoh_clip.mp4
    python tools/build_streamer_profile.py --download-defaults
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from streamers.face_tracker import MODELS_DIR, SFACE_PATH, YUNET_PATH, StreamerIdentityTracker

PROFILES_DIR = MODELS_DIR / "streamer_profiles"
PROFILES_DIR.mkdir(parents=True, exist_ok=True)


def extract_embedding_from_image(image_path: Path) -> Optional[np.ndarray]:
    """Extracts 128-D SFace embedding from a face in a static image."""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[-] Could not load image: {image_path}")
        return None

    h, w = img.shape[:2]
    tracker = StreamerIdentityTracker(frame_width=w, frame_height=h)
    faces = tracker.detect_and_scale(img)
    if faces is None or len(faces) == 0:
        print(f"[-] No face detected in image: {image_path}")
        return None

    # Pick highest confidence face
    faces = sorted(faces, key=lambda f: float(f[14]), reverse=True)
    best_face = faces[0]
    conf = float(best_face[14])
    print(f"[+] Detected face in {image_path.name} with confidence {conf:.2f}")

    feat = tracker.extract_face_embedding(img, best_face)
    if feat is None:
        print("[-] Failed to extract SFace feature.")
        return None

    return feat.astype(np.float32)


def extract_embedding_from_video(video_path: Path, max_samples: int = 20) -> Optional[np.ndarray]:
    """Averages high-confidence embeddings across frames in a video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[-] Could not open video: {video_path}")
        return None

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tracker = StreamerIdentityTracker(frame_width=w, frame_height=h)

    embeddings = []
    frame_idx = 0
    while len(embeddings) < max_samples:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame_idx += 1
        if frame_idx % 15 != 0:
            continue

        faces = tracker.detect_and_scale(frame)
        if faces is not None and len(faces) > 0:
            best = max(faces, key=lambda f: float(f[14]))
            if float(best[14]) >= 0.70:
                feat = tracker.extract_face_embedding(frame, best)
                if feat is not None:
                    embeddings.append(feat)

    cap.release()
    if not embeddings:
        print(f"[-] No stable face embeddings found in {video_path}")
        return None

    # Normalize and average embeddings
    stacked = np.concatenate(embeddings, axis=0)
    avg_feat = np.mean(stacked, axis=0, keepdims=True).astype(np.float32)
    print(f"[+] Computed averaged SFace embedding from {len(embeddings)} frames.")
    return avg_feat


def save_streamer_profile(name: str, embedding: np.ndarray) -> Path:
    """Saves embedding as .npy file in streamer_profiles/."""
    slug = name.lower().strip().replace(" ", "_").replace("-", "_")
    out_path = PROFILES_DIR / f"{slug}.npy"
    np.save(str(out_path), embedding)
    print(f"[✓] Saved face profile for {name} -> {out_path} ({embedding.shape}, {embedding.dtype})")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Build SFace streamer face profiles")
    parser.add_argument("--name", type=str, help="Streamer name or slug (e.g. 'Jynxzi', 'CaseOh')")
    parser.add_argument("--image", type=Path, help="Path to reference face image")
    parser.add_argument("--video", type=Path, help="Path to reference video/clip")
    args = parser.parse_args()

    if not args.name:
        parser.print_help()
        sys.exit(1)

    feat = None
    if args.image:
        feat = extract_embedding_from_image(args.image)
    elif args.video:
        feat = extract_embedding_from_video(args.video)
    else:
        print("[-] Please specify either --image or --video")
        sys.exit(1)

    if feat is not None:
        save_streamer_profile(args.name, feat)
    else:
        print("[-] Failed to generate profile.")
        sys.exit(1)


if __name__ == "__main__":
    main()
