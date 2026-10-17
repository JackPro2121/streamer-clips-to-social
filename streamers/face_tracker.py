"""Streamer Identity Tracking & Smart Off-Screen State Machine.

Specialized for high-energy US Streamers (Kai Cenat, IShowSpeed, etc.):
- Face Identity Locking via OpenCV SFace (SphereFace / ArcFace ONNX) + YuNet
- Retains lock on the primary streamer even in multi-guest scenes (e.g. Kai Cenat x MrBeast x Kevin Hart)
- 3-Tier State Machine for handling streamer leaving chair / exiting room:
    1. LOCKED: Streamer centered with smooth spring-damper LERP panning
    2. OCCLUDED: Streamer turned around / fast movement -> Torso anchor hold
    3. OFF-SCREEN (>1.5s): Auto fallback to wide room shot with blurred backdrop (presentation_slide mode)
       Instant snap-back when streamer re-enters frame!
- High-Performance: 640px downscaled detection with sequential frame reading (~7 fps on CPU).
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

from core.config import MODELS_DIR as _MODELS_DIR
from core.face_tracker import ShotPlan, FramingDecision, detect_letterbox_margins
from streamers.spec import (
    STREAMER_IDENTITY_SIMILARITY_FLOOR,
    STREAMER_TIER2_HOLD_S,
)

MODELS_DIR = _MODELS_DIR
YUNET_PATH = MODELS_DIR / "face_detection_yunet.onnx"
SFACE_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"

# Cosine similarity threshold for SFace face recognition from V2 spec
SIMILARITY_THRESHOLD = STREAMER_IDENTITY_SIMILARITY_FLOOR


@dataclass
class StreamerTrackPoint:
    time_s: float
    center_x: int
    center_y: int
    box_w: int
    box_h: int
    confidence: float
    is_streamer: bool
    state: str  # 'LOCKED', 'HOLD', 'OFF_SCREEN'


class StreamerIdentityTracker:
    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        streamer_name: str = "Kai Cenat",
        reference_embedding: Optional[np.ndarray] = None,
        det_max_dim: int = 640
    ):
        self.width = frame_width
        self.height = frame_height
        self.streamer_name = streamer_name
        self.reference_embedding = reference_embedding

        # Auto-load ground-truth face profile if available
        if self.reference_embedding is None and streamer_name:
            slug = streamer_name.lower().strip().replace(" ", "_").replace("-", "_")
            profile_path = MODELS_DIR / "streamer_profiles" / f"{slug}.npy"
            if profile_path.exists():
                try:
                    self.reference_embedding = np.load(str(profile_path))
                    print(f"[+] Loaded ground-truth face profile for {streamer_name} ({profile_path.name})")
                except Exception as e:
                    print(f"[!] Could not load face profile for {streamer_name}: {e}")

        # Determine downscaled detection size preserving aspect ratio
        if frame_width >= frame_height:
            self.det_w = det_max_dim
            self.det_h = int(det_max_dim * (frame_height / frame_width)) // 2 * 2
        else:
            self.det_h = det_max_dim
            self.det_w = int(det_max_dim * (frame_width / frame_height)) // 2 * 2

        self.scale_x = frame_width / max(1, self.det_w)
        self.scale_y = frame_height / max(1, self.det_h)

        self.detector: Optional[cv2.FaceDetectorYN] = None
        self.recognizer: Optional[cv2.FaceRecognizerSF] = None
        self._init_models()

    def _init_models(self):
        if YUNET_PATH.exists():
            self.detector = cv2.FaceDetectorYN.create(
                str(YUNET_PATH), "", (self.det_w, self.det_h), 0.50, 0.3, 5000
            )
            self.detector.setInputSize((self.det_w, self.det_h))

        if not SFACE_PATH.exists():
            try:
                import urllib.request
                sface_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
                print("[*] Downloading SFace face recognition model...")
                urllib.request.urlretrieve(sface_url, str(SFACE_PATH))
            except Exception as e:
                print(f"[!] Could not download SFace model: {e}")

        if SFACE_PATH.exists():
            try:
                self.recognizer = cv2.FaceRecognizerSF.create(str(SFACE_PATH), "")
            except Exception as e:
                print(f"[!] Could not load SFace model: {e}")

    def detect_and_scale(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detects faces on downscaled frame and maps coordinates back to original frame."""
        if self.detector is None or frame is None:
            return None
        small = cv2.resize(frame, (self.det_w, self.det_h), interpolation=cv2.INTER_LINEAR)
        _, faces = self.detector.detect(small)
        if faces is None or len(faces) == 0:
            return None

        scaled_faces = faces.copy()
        scaled_faces[:, 0] *= self.scale_x
        scaled_faces[:, 1] *= self.scale_y
        scaled_faces[:, 2] *= self.scale_x
        scaled_faces[:, 3] *= self.scale_y
        for lm in range(4, 14, 2):
            scaled_faces[:, lm] *= self.scale_x
            scaled_faces[:, lm + 1] *= self.scale_y

        return scaled_faces

    def extract_face_embedding(self, frame: np.ndarray, face_data: np.ndarray) -> Optional[np.ndarray]:
        """Aligns and extracts a 128-D SFace feature vector from a face detection."""
        if self.recognizer is None:
            return None
        try:
            aligned = self.recognizer.alignCrop(frame, face_data)
            feat = self.recognizer.feature(aligned)
            return feat
        except Exception:
            return None

    def match_similarity(self, feat: np.ndarray) -> float:
        """Returns cosine similarity score between given feature and streamer reference embedding."""
        if self.recognizer is None or self.reference_embedding is None or feat is None:
            return 0.0
        try:
            score = self.recognizer.match(self.reference_embedding, feat, cv2.FaceRecognizerSF_FR_COSINE)
            return float(score)
        except Exception:
            return 0.0


def analyze_streamer_clip_framing(
    video_path: Path,
    start_time: float,
    end_time: float,
    streamer_name: str = "Kai Cenat",
    sample_fps: float = 2.5
) -> FramingDecision:
    """Analyzes a streamer video clip and produces a FramingDecision with:

    - Streamer identity locking (rejecting guests/background faces)
    - 3-Tier State Machine (Locked -> Occluded Hold -> Smooth Wide Fallback on exit)
    - Re-entry snap-back when streamer returns
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    clip_duration = max(0.1, end_time - start_time)

    # 1. Letterbox margin check
    start_frame = int(start_time * video_fps)
    end_frame = int(end_time * video_fps)
    active_x, active_y, active_w, active_h = detect_letterbox_margins(cap, start_frame, end_frame)

    target_crop_w = int(active_h * 9 / 16)
    target_crop_w -= target_crop_w % 2
    target_crop_w = min(target_crop_w, active_w)
    target_crop_w -= target_crop_w % 2

    # 2. Initialize Streamer Identity Tracker and Audio Profile
    tracker = StreamerIdentityTracker(w, h, streamer_name=streamer_name)

    try:
        from core.active_speaker import _audio_envelope
        audio_env = _audio_envelope(video_path, fps=int(video_fps))
    except Exception:
        audio_env = []
    med_audio = float(np.median(audio_env)) if audio_env else 0.0

    prev_mouth_patch = None
    recent_mouth_motions: List[float] = []

    # 3. Fast sequential frame sampling
    step_frames = max(1, int(round(video_fps / sample_fps)))
    raw_tracks: List[StreamerTrackPoint] = []
    anchor_candidates: List[Tuple[float, np.ndarray]] = []  # (face_area, feat)

    last_known_cx = active_x + active_w // 2
    last_known_cy = active_y + active_h // 3
    last_known_w = target_crop_w // 2
    last_known_h = active_h // 2
    last_seen_time = -999.0

    # Seek once to start_frame
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_counter = start_frame

    while frame_counter <= end_frame:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if (frame_counter - start_frame) % step_frames == 0:
            rel_t = (frame_counter - start_frame) / video_fps
            scaled_faces = tracker.detect_and_scale(frame)

            best_face = None
            best_score = -1.0
            best_box = None

            if scaled_faces is not None and len(scaled_faces) > 0:
                for face in scaled_faces:
                    conf = float(face[14])
                    if conf < 0.60:
                        continue

                    fw, fh = float(face[2]), float(face[3])
                    if (fw * fh) < 4096.0:
                        continue
                    feat = tracker.extract_face_embedding(frame, face)

                    # Accumulate anchor candidates from early frames if reference not yet set (D20 fix)
                    if tracker.reference_embedding is None and feat is not None:
                        face_cx = float(face[0] + face[2] / 2)
                        dist_to_center = abs(face_cx - (active_x + active_w / 2))
                        norm_dist = dist_to_center / max(1.0, active_w)
                        weight = conf * (fw * fh) / (1.0 + norm_dist)
                        anchor_candidates.append((weight, feat))
                        if len(anchor_candidates) >= 5:
                            anchor_candidates.sort(key=lambda c: c[0], reverse=True)
                            top_feats = [c[1] for c in anchor_candidates[:3]]
                            avg_feat = np.mean(np.concatenate(top_feats, axis=0), axis=0, keepdims=True)
                            tracker.reference_embedding = avg_feat.astype(np.float32)
                            print(f"[+] Locked Streamer Identity Anchor (averaged top 3) for {streamer_name}")

                    score = tracker.match_similarity(feat) if (feat is not None and tracker.reference_embedding is not None) else conf

                    if score > best_score:
                        best_score = score
                        best_face = face
                        best_box = (int(face[0]), int(face[1]), int(face[2]), int(face[3]))

            # Evaluate state
            is_match = (
                best_face is not None and
                (tracker.reference_embedding is None or best_score >= SIMILARITY_THRESHOLD)
            )

            if is_match and best_box is not None:
                # TIER 1: Streamer LOCKED or LISTENING
                cx = best_box[0] + best_box[2] // 2
                cy = best_box[1] + best_box[3] // 2
                last_known_cx, last_known_cy = cx, cy
                last_known_w, last_known_h = best_box[2], best_box[3]
                last_seen_time = rel_t

                # Measure mouth motion to detect whether streamer is talking vs listening
                bx, by, bw, bh = best_box
                x0 = max(0, bx)
                y0 = max(0, int(by + bh * 0.55))
                x1 = min(frame.shape[1], bx + bw)
                y1 = min(frame.shape[0], by + bh)
                mouth_roi = frame[y0:y1, x0:x1]
                diff = 0.0
                if mouth_roi.size > 0:
                    mouth_patch = cv2.resize(cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2GRAY), (24, 12)).astype(np.float32)
                    if prev_mouth_patch is not None and prev_mouth_patch.shape == mouth_patch.shape:
                        diff = float(np.mean(np.abs(mouth_patch - prev_mouth_patch)))
                    prev_mouth_patch = mouth_patch
                recent_mouth_motions.append(diff)

                # Determine whether streamer is listening to external voice/Discord
                curr_audio = audio_env[frame_counter] if frame_counter < len(audio_env) else 0.0
                rolling_mouth = float(np.mean(recent_mouth_motions[-8:])) if len(recent_mouth_motions) >= 4 else 1.0

                is_listening = (
                    len(recent_mouth_motions) >= 6 and
                    rolling_mouth < 0.40 and
                    curr_audio > (med_audio * 1.3)
                )

                pt_state = "LISTENING" if is_listening else "LOCKED"

                raw_tracks.append(StreamerTrackPoint(
                    time_s=rel_t,
                    center_x=cx,
                    center_y=cy,
                    box_w=best_box[2],
                    box_h=best_box[3],
                    confidence=best_score,
                    is_streamer=True,
                    state=pt_state
                ))
            else:
                time_since_seen = rel_t - last_seen_time if last_seen_time >= 0 else 999.0

                if time_since_seen <= STREAMER_TIER2_HOLD_S:
                    # TIER 2: OCCLUDED / RUNNING (Momentarily lost <= STREAMER_TIER2_HOLD_S) -> Hold last known anchor
                    raw_tracks.append(StreamerTrackPoint(
                        time_s=rel_t,
                        center_x=last_known_cx,
                        center_y=last_known_cy,
                        box_w=last_known_w,
                        box_h=last_known_h,
                        confidence=0.0,
                        is_streamer=False,
                        state="HOLD"
                    ))
                else:
                    # TIER 3: OFF_SCREEN (> STREAMER_TIER2_HOLD_S) -> Streamer left chair/room!
                    raw_tracks.append(StreamerTrackPoint(
                        time_s=rel_t,
                        center_x=active_x + active_w // 2,
                        center_y=active_y + active_h // 2,
                        box_w=active_w,
                        box_h=active_h,
                        confidence=0.0,
                        is_streamer=False,
                        state="OFF_SCREEN"
                    ))

        frame_counter += 1

    cap.release()

    if not raw_tracks:
        return FramingDecision(
            mode="single_smooth",
            face_count=1,
            smoothed_center_x=active_x + active_w // 2,
            smoothed_center_y=active_y + active_h // 3,
            video_width=w,
            video_height=h,
            active_x=active_x,
            active_y=active_y,
            active_w=active_w,
            active_h=active_h
        )

    # 4. Cluster tracks into ShotPlans
    shots: List[ShotPlan] = []
    current_state = raw_tracks[0].state
    shot_start_t = 0.0
    shot_pts: List[StreamerTrackPoint] = []

    for pt in raw_tracks:
        is_state_change = (
            (pt.state in ("OFF_SCREEN", "LISTENING") and current_state not in ("OFF_SCREEN", "LISTENING")) or
            (pt.state not in ("OFF_SCREEN", "LISTENING") and current_state in ("OFF_SCREEN", "LISTENING"))
        )

        if is_state_change and (pt.time_s - shot_start_t) >= 2.5:
            shot_end_t = pt.time_s
            shots.append(_build_shot_from_points(
                shot_start_t, shot_end_t, shot_pts, current_state,
                target_crop_w, active_x, active_y, active_w, active_h
            ))
            shot_start_t = shot_end_t
            current_state = pt.state
            shot_pts = [pt]
        else:
            shot_pts.append(pt)

    if shot_pts:
        shots.append(_build_shot_from_points(
            shot_start_t, clip_duration, shot_pts, current_state,
            target_crop_w, active_x, active_y, active_w, active_h
        ))

    if shots:
        shots[0].start = 0.0
        shots[-1].end = clip_duration
        for i in range(len(shots) - 1):
            shots[i].end = shots[i + 1].start

    print(f"[+] Streamer Framing Plan: {len(shots)} shots generated")
    for idx, s in enumerate(shots, 1):
        print(f"    Shot #{idx}: [{s.start:.1f}s - {s.end:.1f}s] Mode: {s.mode} (crop_x: {s.crop_x})")

    if len(shots) == 1 and shots[0].mode == "portrait_face" and len(shots[0].face_centers_timeline) < 2:
        return FramingDecision(
            mode="single_smooth",
            face_count=1,
            smoothed_center_x=shots[0].crop_x + target_crop_w // 2,
            smoothed_center_y=shots[0].center_y,
            video_width=w,
            video_height=h,
            active_x=active_x,
            active_y=active_y,
            active_w=active_w,
            active_h=active_h
        )

    return FramingDecision(
        mode="multi_shot_dynamic",
        face_count=len(shots),
        shots=shots,
        video_width=w,
        video_height=h,
        active_x=active_x,
        active_y=active_y,
        active_w=active_w,
        active_h=active_h
    )


def _build_shot_from_points(
    start_t: float,
    end_t: float,
    points: List[StreamerTrackPoint],
    state: str,
    target_crop_w: int,
    active_x: int,
    active_y: int,
    active_w: int,
    active_h: int
) -> ShotPlan:
    """Creates a ShotPlan tailored for streamer state (LOCKED/HOLD vs OFF_SCREEN/LISTENING)."""
    if state in ("OFF_SCREEN", "LISTENING"):
        return ShotPlan(
            start=start_t,
            end=end_t,
            mode="presentation_slide",
            crop_x=active_x,
            crop_y=active_y,
            crop_w=active_w,
            crop_h=active_h,
            center_y=active_y + active_h // 2,
            zoom_factor=1.0,
            motion="drift"
        )

    cxs = [p.center_x for p in points] if points else [active_x + active_w // 2]
    cys = [p.center_y for p in points] if points else [active_y + active_h // 3]

    avg_cx = int(np.median(cxs))
    avg_cy = int(np.median(cys))

    # Protect headroom: prevent face from being pushed directly beneath top sticker
    min_headroom_y = active_y + int(target_crop_w * 0.42)
    max_headroom_y = active_y + active_h - int(target_crop_w * 0.42)
    avg_cy = max(min_headroom_y, min(avg_cy, max_headroom_y))

    crop_x = max(active_x, min(avg_cx - target_crop_w // 2, active_x + active_w - target_crop_w))

    timeline: List[Tuple[float, int]] = []
    if len(points) >= 2:
        for p in points:
            rel_offset = round(p.time_s - start_t, 2)
            clamped_pt_cx = max(active_x + target_crop_w // 2, min(p.center_x, active_x + active_w - target_crop_w // 2))
            timeline.append((rel_offset, clamped_pt_cx))

    return ShotPlan(
        start=start_t,
        end=end_t,
        mode="portrait_face",
        crop_x=crop_x,
        crop_y=active_y,
        crop_w=target_crop_w,
        crop_h=active_h,
        center_y=avg_cy,
        zoom_factor=1.0,
        face_centers_timeline=timeline,
        motion="drift"
    )
