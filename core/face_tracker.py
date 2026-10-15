import cv2
import math
import numpy as np
import os
import requests
import uuid
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from core.scene_classifier import classify_frame_scene, detect_clip_shots
from core.config import ENABLE_PUNCH_ZOOM, MODELS_DIR
from core.creative_spec import (
    SPEAKER_MIN_DWELL_S,
    SPEAKER_HYSTERESIS,
    INTERJECTION_MAX_S,
    MIN_SPEECH_LIP_MOTION,
)

MODEL_DIR = MODELS_DIR
YUNET_MODEL_PATH = MODEL_DIR / "face_detection_yunet.onnx"
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

@dataclass
class FaceBox:
    x: int
    y: int
    w: int
    h: int
    center_x: int
    center_y: int

@dataclass
class ShotPlan:
    start: float  # Relative start seconds within clip (0.0 to clip duration)
    end: float    # Relative end seconds within clip
    mode: str     # 'presentation_slide', 'portrait_face', 'split_screen'
    crop_x: int = 0
    crop_y: int = 0
    crop_w: int = 0
    crop_h: int = 0
    center_y: int = 0
    zoom_factor: float = 1.0  # 1.15 for viral hook punch-zoom
    # Timeline of center_x for each sample in this shot to enable LERP panning
    face_centers_timeline: List[Tuple[float, int]] = field(default_factory=list)
    speaker1_box: Optional[Tuple[int, int, int, int]] = None
    speaker2_box: Optional[Tuple[int, int, int, int]] = None
    # Faces seen in this shot, in source pixels, for caption placement.
    #
    # This field exists because of a measured defect. Captions render directly
    # across the subject's brow and eyes in 3 of the 6 golden-master clips --
    # 100% of burned-in caption pixels on the upper face in each. The cause is that
    # caption collision avoidance only ever knew about *text* regions, detected by
    # OCR or the edge heuristic. A face is not text, so relocating away from text
    # moved the caption straight onto the face: `upper_center` resolves to roughly
    # 25% of frame height, which clears the 240px platform UI band and lands on
    # the eyes.
    #
    # The renderer already had the faces; it simply never told the caption layer.
    # Tuples of (x, y, w, h) in source pixels, largest first.
    face_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    margin_v: int = 460
    subtitle_placement: str = "lower_third"
    subtitle_alignment: int = 2
    # Per-shot camera move. 'drift' is the motion guarantee; 'push_in'/'pull_out'
    # add an animated zoompan ramp on top of it; 'hold' requests neither (the
    # drift still applies, because a hold is a preference, not a licence to
    # freeze). Set by the director via apply_shot_directives.
    motion: str = "drift"

@dataclass
class FramingDecision:
    mode: str  # 'multi_shot_dynamic', 'single_smooth', 'dynamic_cut', 'split_screen', 'blur_stack'
    face_count: int
    shots: List[ShotPlan] = field(default_factory=list)
    speaker1_box: Optional[Tuple[int, int, int, int]] = None
    speaker2_box: Optional[Tuple[int, int, int, int]] = None
    smoothed_center_x: int = 0
    smoothed_center_y: int = 0
    crop_x_expr: Optional[str] = None
    video_width: int = 1920
    video_height: int = 1080
    active_x: int = 0
    active_y: int = 0
    active_w: int = 1920
    active_h: int = 1080

def detect_letterbox_margins(cap: cv2.VideoCapture, start_frame: int, end_frame: int, max_samples: int = 15) -> Tuple[int, int, int, int]:
    """
    Detects baked-in black letterbox bars (cinematic 2:1 or 2.39:1 aspect ratio inside 16:9).
    Returns (active_x, active_y, active_w, active_h).
    """
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    span = max(1, end_frame - start_frame)
    step = max(1, span // max_samples)

    top_candidates = []
    bot_candidates = []

    saved_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    for fn in range(start_frame, end_frame, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        row_means = np.mean(gray, axis=1)

        # Check top letterbox
        top = 0
        while top < h // 3 and row_means[top] < 12:
            top += 1

        # Check bottom letterbox
        bot = 0
        while bot < h // 3 and row_means[h - 1 - bot] < 12:
            bot += 1

        top_candidates.append(top)
        bot_candidates.append(bot)

    cap.set(cv2.CAP_PROP_POS_FRAMES, saved_pos)

    if not top_candidates:
        return 0, 0, w, h

    med_top = int(np.median(top_candidates))
    med_bot = int(np.median(bot_candidates))

    # Ignore minor padding (< 12px)
    if med_top < 12:
        med_top = 0
    if med_bot < 12:
        med_bot = 0

    active_y = med_top
    active_h = max(100, h - med_top - med_bot)
    return 0, active_y, w, active_h

def _download_model_atomically(url: str, destination: Path, minimum_bytes: int) -> bool:
    temp_path = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.part"
    )
    try:
        response = requests.get(url, allow_redirects=True, timeout=15)
        if response.status_code != 200 or len(response.content) < minimum_bytes:
            return False
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_path, "wb") as file_handle:
            file_handle.write(response.content)
        os.replace(temp_path, destination)
        return True
    except Exception:
        return False
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


_yunet_attempted = False
_yunet_detector = None
_mediapipe_detector = None
_haar_detector = None

# AGENTS.md 3.3. A split-screen pane must be 9:8 so that it scales to 1080x960 with
# no black bars. Exposed as a module constant so tests and the renderer can agree
# on one number rather than each restating 1.125.
PANE_ASPECT = 9.0 / 8.0


def _dominant_face_boxes(
    face_samples: List[List["FaceBox"]], limit: int = 3
) -> List[Tuple[int, int, int, int]]:
    """Median face boxes across a shot, largest first, in source pixels.

    Uses the median rather than the mean because a face box is a geometric
    quantity and an outlier frame -- a head turn at the edge of a detection, or a
    spurious blob -- would drag a mean somewhere no face actually is. The median
    is the position the subject was genuinely in for most of the shot.

    ``face_samples`` is the per-frame list of detected faces. Frames with no
    faces are skipped rather than treated as a zero box.
    """
    if not face_samples:
        return []
    import numpy as np

    flattened = [face for frame in face_samples for face in (frame or [])]
    if not flattened:
        return []
    flattened.sort(key=lambda f: f.w * f.h, reverse=True)
    boxes: List[Tuple[int, int, int, int]] = []
    for face in flattened[:limit]:
        same = [
            candidate
            for candidate in flattened
            if abs(candidate.center_x - face.center_x) < face.w * 1.5
            and abs(candidate.center_y - face.center_y) < face.h * 1.5
        ]
        if not same:
            same = [face]
        median = (
            int(np.median([f.x for f in same])),
            int(np.median([f.y for f in same])),
            int(np.median([f.w for f in same])),
            int(np.median([f.h for f in same])),
        )
        # The top `limit` boxes by area are frequently the same face detected in
        # adjacent samples, which would record one face as several protected
        # regions. Harmless to collision maths, misleading in the plan artifact.
        if any(
            abs(median[0] - kept[0]) <= kept[2] * 0.5
            and abs(median[1] - kept[1]) <= kept[3] * 0.5
            for kept in boxes
        ):
            continue
        boxes.append(median)
    return boxes


def _pane_dimensions(
    target_width: int, max_height: int, max_width: int | None = None
) -> tuple[int, int]:
    """Even width and height at 9:8, within a pixel, inside the available region.

    Both dimensions must be even: ffmpeg's yuv420p output needs chroma-aligned
    dimensions, and an odd value is a class of bug this codebase has already hit
    twice (``main.py:60`` producing 607, and ``face_tracker`` producing 607 for a
    9:16 crop).

    The previous implementation evened the width and the height independently,
    which is what breaks the ratio. 1055 -> 1054 and 937 -> 936 gives 1.1261
    instead of 1.125, and the renderer then padded the difference into a black
    bar. Here only the *width* is the free variable: the height is derived from
    it, and if the derived height is odd it is nudged up and the width recomputed
    from the height, so the ratio stays correct to within one pixel.

    ``max_width`` matters because the odd-height correction can return a width up
    to two pixels *larger* than asked for. An uncapped result can exceed the source
    width, and the position clamp elsewhere then yields a negative right edge, so
    ffmpeg is handed a crop wider than the frame and fails outright.
    """
    ceiling_h = max(2, int(max_height))
    ceiling_h -= ceiling_h % 2
    ceiling_w = None
    if max_width is not None:
        ceiling_w = max(2, int(max_width))
        ceiling_w -= ceiling_w % 2

    width = max(2, int(target_width))
    width -= width % 2
    height = int(round(width * 8.0 / 9.0))
    if height % 2:
        height += 1
        width = int(round(height * 9.0 / 8.0))
        width -= width % 2
    if height > ceiling_h:
        height = ceiling_h
        width = int(round(height * 9.0 / 8.0))
        width -= width % 2
    if ceiling_w is not None and width > ceiling_w:
        width = ceiling_w
        height = int(round(width * 8.0 / 9.0))
        if height % 2:
            height -= 1
        height = max(2, min(height, ceiling_h))
    return max(2, width), max(2, height)


def get_face_detector():
    """Initializes Face Detection with robust multi-tiered fallbacks (YuNet -> MediaPipe -> Haar Cascade)."""
    global _yunet_attempted, _yunet_detector, _mediapipe_detector, _haar_detector

    if _yunet_detector is not None:
        return _yunet_detector

    if not _yunet_attempted:
        _yunet_attempted = True
        try:
            if not YUNET_MODEL_PATH.exists():
                MODEL_DIR.mkdir(parents=True, exist_ok=True)
                print("[*] Downloading high-precision YuNet face detector model...")
                if not _download_model_atomically(YUNET_URL, YUNET_MODEL_PATH, 100000):
                    raise RuntimeError("YuNet model download failed")
                print(f"[+] YuNet model downloaded successfully ({YUNET_MODEL_PATH.stat().st_size / 1024:.1f} KB).")
            if YUNET_MODEL_PATH.exists() and hasattr(cv2, 'FaceDetectorYN'):
                _yunet_detector = cv2.FaceDetectorYN.create(
                    str(YUNET_MODEL_PATH), "", (320, 320), 0.5, 0.3, 5000
                )
                print("[+] Initialized OpenCV YuNet multi-angle face detector.")
        except Exception as e:
            _yunet_attempted = False
            print(f"[-] YuNet initialization warning: {e}. Trying MediaPipe...")

    if _yunet_detector is not None:
        return _yunet_detector

    if _mediapipe_detector is not None:
        return _mediapipe_detector

    try:
        import mediapipe as mp
        if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_detection'):
            _mediapipe_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.55
            )
            return _mediapipe_detector
    except Exception:
        pass

    if _haar_detector is not None:
        return _haar_detector

    try:
        local_cascade = 'haarcascade_frontalface_default.xml'
        if os.path.exists(local_cascade):
            cascade_path = local_cascade
        else:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if os.path.exists(cascade_path):
            _haar_detector = cv2.CascadeClassifier(cascade_path)
            return _haar_detector
    except Exception as e:
        print(f"[-] All face detectors failed: {e}.")
    return None

def detect_faces_in_frame(
    detector,
    frame: np.ndarray,
    width: int,
    height: int,
    active_y: int = 0,
    active_h: int = 0
) -> List[FaceBox]:
    """Performs face detection across different detector backends with strict false-positive filtering."""
    faces: List[FaceBox] = []
    if detector is None or frame is None:
        return faces

    frame_h, frame_w = frame.shape[:2]
    effective_h = active_h if active_h > 0 else frame_h
    effective_top = active_y if active_h > 0 else 0

    def is_plausible_speaker_face(fx: int, fy: int, fw: int, fh: int) -> bool:
        """Filters out lower-third avatar icons, channel watermarks, and noise."""
        if fw < 28 or fh < 28:
            return False
        cx = fx + fw // 2
        cy = fy + fh // 2
        # Reject lower-third region: Speakers' heads are NEVER in the bottom 32% of active video.
        # This rejects social media avatar badges, profile pictures in corners, and lower-third graphics.
        if cy > (effective_top + effective_h * 0.68) or fy > (effective_top + effective_h * 0.62):
            return False
        # Reject extreme bottom corners (small icons at edge)
        if cy > (effective_top + effective_h * 0.50) and (cx < frame_w * 0.15 or cx > frame_w * 0.85) and (fw < frame_w * 0.08):
            return False
        return True

    # Case A: OpenCV FaceDetectorYN (YuNet)
    if hasattr(cv2, 'FaceDetectorYN') and isinstance(detector, cv2.FaceDetectorYN):
        try:
            h, w = frame.shape[:2]
            detector.setInputSize((w, h))
            _, detections = detector.detect(frame)
            if detections is not None:
                for d in detections:
                    fx = int(d[0])
                    fy = int(d[1])
                    fw = int(d[2])
                    fh = int(d[3])
                    conf = float(d[-1])
                    if conf >= 0.32 and is_plausible_speaker_face(fx, fy, fw, fh):
                        faces.append(FaceBox(
                            x=max(0, fx), y=max(0, fy),
                            w=fw, h=fh,
                            center_x=max(0, fx) + fw // 2,
                            center_y=max(0, fy) + fh // 2
                        ))
            return faces
        except Exception:
            pass

    # Case B: MediaPipe Solutions
    if hasattr(detector, 'process'):
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb_frame)
            if results and results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    fx = int(bbox.xmin * width)
                    fy = int(bbox.ymin * height)
                    fw = int(bbox.width * width)
                    fh = int(bbox.height * height)
                    if is_plausible_speaker_face(fx, fy, fw, fh):
                        faces.append(FaceBox(
                            x=max(0, fx), y=max(0, fy),
                            w=fw, h=fh,
                            center_x=max(0, fx) + fw // 2,
                            center_y=max(0, fy) + fh // 2
                        ))
            return faces
        except Exception:
            pass

    # Case C: Haar Cascade
    if hasattr(detector, 'detectMultiScale'):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector.detectMultiScale(gray, 1.1, 4)
            for (x, y, w_box, h_box) in rects:
                if is_plausible_speaker_face(x, y, w_box, h_box):
                    faces.append(FaceBox(
                        x=x, y=y, w=w_box, h=h_box,
                        center_x=x + w_box // 2,
                        center_y=y + h_box // 2
                    ))
        except Exception:
            pass

    return faces


def _skin_tone_center_x(frame: np.ndarray, active_x: int, active_y: int, active_w: int, active_h: int) -> Optional[int]:
    """Fallback: when face detector returns empty, find the dominant skin-tone region in the upper
    55% of the active content area. This catches side-on heads, hats, and profile angles that
    YuNet misses, preventing the crop from locking onto mic stands or lamps."""
    if frame is None:
        return None
    try:
        # Work strictly in the upper 55% where podcast speakers always appear
        top_h = int(active_h * 0.55)
        roi = frame[active_y:active_y + top_h, active_x:active_x + active_w]
        if roi.size == 0:
            return None
        ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
        # Standard skin-tone range in YCrCb
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([235, 173, 127], dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower, upper)
        # Morphological cleanup to merge fragmented skin blobs
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        # Find largest connected skin region
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        roi_area = roi.shape[0] * roi.shape[1]
        # Must be at least 2.5% of the ROI to be a person, not noise
        if area < roi_area * 0.025:
            return None
        # Human head/torso is vertically oriented or square-ish (aspect ratio roughly 0.35 to 1.85)
        # Horizontal wooden shelves, desks, or wall panels have aspect_ratio > 2.0
        bx, by, bw, bh = cv2.boundingRect(largest)
        aspect_ratio = bw / float(max(1, bh))
        if aspect_ratio > 2.0 or aspect_ratio < 0.30:
            return None
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None
        cx_roi = int(M["m10"] / M["m00"])
        return active_x + cx_roi
    except Exception:
        return None

_MIN_SPEECH_LIP_MOTION = MIN_SPEECH_LIP_MOTION
_DOMINANT_SPEAKER_RATIO = SPEAKER_HYSTERESIS

def _estimate_mouth_motion(prev_patch: Optional[np.ndarray], frame: np.ndarray, f: FaceBox) -> Tuple[float, Optional[np.ndarray]]:
    """Calculates normalized pixel displacement in the mouth region (lower 35% of face box) for active speaker detection."""
    try:
        h, w = frame.shape[:2]
        my1 = max(0, min(h - 1, int(f.y + f.h * 0.62)))
        my2 = max(my1 + 1, min(h, int(f.y + f.h * 0.98)))
        mx1 = max(0, min(w - 1, int(f.x + f.w * 0.20)))
        mx2 = max(mx1 + 1, min(w, int(f.x + f.w * 0.80)))
        roi = frame[my1:my2, mx1:mx2]
        if roi.size == 0:
            return 0.0, None
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (32, 24))
        if prev_patch is None or prev_patch.shape != (24, 32):
            return 0.0, resized
        diff = cv2.absdiff(resized, prev_patch)
        score = float(np.mean(diff))
        return score, resized
    except Exception:
        return 0.0, None


def _apply_autoflip_smoothing(timeline: List[Tuple[float, int]], dead_zone_px: int = 40) -> List[Tuple[float, int]]:
    """Applies Google AutoFlip dead-zone & smooth panning to eliminate camera micro-jitter."""
    if len(timeline) <= 1:
        return timeline
    smoothed: List[Tuple[float, int]] = []
    current_anchor = timeline[0][1]
    for t, x in timeline:
        diff = x - current_anchor
        if abs(diff) > dead_zone_px:
            # Shift anchor towards new position with smooth exponential easing
            current_anchor = int(current_anchor + 0.60 * diff)
        smoothed.append((t, current_anchor))
    return smoothed


def _stable_face_timeline(
    timeline: List[Tuple[float, int]],
    active_width: int,
) -> List[Tuple[float, int]]:
    if len(timeline) < 4 or active_width <= 0:
        return []
    ordered = sorted(timeline, key=lambda item: item[0])
    times = [float(item[0]) for item in ordered]
    positions = [int(item[1]) for item in ordered]
    if max(times) - min(times) < 1.0:
        return []
    if max((right - left for left, right in zip(times, times[1:])), default=0.0) > 1.0:
        return []
    if abs(positions[-1] - positions[0]) > active_width * 0.30:
        return []
    direction = 1 if positions[-1] >= positions[0] else -1
    directional_steps = sum(
        1
        for left, right in zip(positions, positions[1:])
        if (right - left) * direction > 0
    )
    if directional_steps / max(1, len(positions) - 1) < 0.75:
        return []
    if max((abs(right - left) for left, right in zip(positions, positions[1:])), default=0.0) > active_width * 0.18:
        return []
    return _apply_autoflip_smoothing(ordered)


def _is_visual_layout_scene(scene_info: dict) -> bool:
    return bool(scene_info.get("is_presentation") or scene_info.get("has_slide_layout"))


# A picture-in-picture host inset must be a small face. Anything larger than this
# share of the frame is a real co-presenter, not a camera inset, so the existing
# split-screen and portrait_face layouts keep their behaviour.
PIP_MAX_HOST_AREA_RATIO = 0.12
# The host face must persist across most of the shot; a face seen in only one
# sampled frame is a transition artefact, not a camera inset.
PIP_MIN_PERSISTENCE = 0.5


def _picture_in_picture_host_box(
    face_samples: List[List["FaceBox"]],
) -> Optional[Tuple[int, int, int, int]]:
    """
    Return a stable (x, y, w, h) host inset box for a slide-with-camera shot.

    Takes the median box across the frames that hold exactly one face, so a
    single noisy detection cannot move the inset. Returns None when no face
    persists, when several faces are present, or when the face is implausibly
    small -- all of which keep the caller on the pre-existing layouts.
    """
    if not face_samples:
        return None
    single = [faces for faces in face_samples if faces and len(faces) == 1]
    if not single:
        return None
    if len(single) < PIP_MIN_PERSISTENCE * len(face_samples):
        return None
    boxes = [(int(f[0].x), int(f[0].y), int(f[0].w), int(f[0].h)) for f in single]
    median = []
    for index in range(4):
        values = sorted(box[index] for box in boxes)
        median.append(values[len(values) // 2])
    x, y, w, h = median
    if w <= 8 or h <= 8:
        return None
    return (x, y, w, h)


def _frame_window(start_time: float, end_time: float, fps: float) -> Tuple[int, int]:
    safe_fps = max(1.0, float(fps))
    start_frame = max(0, int(math.ceil(start_time * safe_fps)))
    end_frame = max(start_frame + 1, int(math.ceil(end_time * safe_fps)))
    return start_frame, end_frame


def _relative_frame_timestamp(
    cap: cv2.VideoCapture,
    start_time: float,
    current_frame: int,
    start_frame: int,
    fps: float,
    clip_duration: float
) -> float:
    frame_time_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
    if math.isfinite(frame_time_ms) and frame_time_ms > 0:
        rel_t = frame_time_ms / 1000.0 - start_time
    else:
        rel_t = (current_frame - start_frame) / max(1.0, float(fps))
    return max(0.0, min(float(clip_duration), rel_t))


def _segment_two_speaker_shots(
    rel_s: float,
    rel_e: float,
    shot_samples: List[Tuple[float, List[FaceBox], bool, List[float]]],
    valid_face_samples: List[List[FaceBox]],
    two_speaker_samples: List[List[FaceBox]],
    target_crop_w: int,
    active_x: int,
    active_y: int,
    active_w: int,
    active_h: int,
) -> List[ShotPlan]:
    """Smart dynamic speaker switching (OpusClip Pro style).

    Analyzes per-sample mouth motion energy of both speakers across the interval.
    Switches camera to active speaker close-up (9:16 portrait) when one speaker is talking.
    Switches to full-bleed 9:8 split-screen when both speak simultaneously (cross-talk/debate).
    Applies dwell-time filtering (SPEAKER_MIN_DWELL_S) and hysteresis (SPEAKER_HYSTERESIS)
    to eliminate rapid ping-pong cuts and ignore quick interjections (INTERJECTION_MAX_S).
    """
    s1_cx = int(np.median([f[0].center_x for f in two_speaker_samples]))
    s2_cx = int(np.median([f[1].center_x for f in two_speaker_samples]))
    s1_cy = int(np.median([f[0].center_y for f in two_speaker_samples]))
    s2_cy = int(np.median([f[1].center_y for f in two_speaker_samples]))
    s1_fw = int(np.median([f[0].w for f in two_speaker_samples]))
    s2_fw = int(np.median([f[1].w for f in two_speaker_samples]))
    s1_fh = int(np.median([f[0].h for f in two_speaker_samples]))
    s2_fh = int(np.median([f[1].h for f in two_speaker_samples]))

    # Split-screen geometry
    min_pane_w = int(active_w * 0.30)
    max_pane_w = int(active_w * 0.55)
    s1_pane_w, s1_pane_h = _pane_dimensions(
        max(min_pane_w, min(int(s1_fw * 3.8), max_pane_w)),
        active_h,
        max_width=active_w,
    )
    s2_pane_w, s2_pane_h = _pane_dimensions(
        max(min_pane_w, min(int(s2_fw * 3.8), max_pane_w)),
        active_h,
        max_width=active_w,
    )
    s1_y_offset = max(0, int(s1_cy - s1_fh * 1.5))
    s2_y_offset = max(0, int(s2_cy - s2_fh * 1.5))
    s1_x = max(active_x, min(s1_cx - s1_pane_w // 2, active_x + active_w - s1_pane_w))
    s2_x = max(active_x, min(s2_cx - s2_pane_w // 2, active_x + active_w - s2_pane_w))
    s1_y = max(active_y, min(active_y + s1_y_offset, active_y + active_h - s1_pane_h))
    s2_y = max(active_y, min(active_y + s2_y_offset, active_y + active_h - s2_pane_h))
    s1_box = (s1_x, s1_y, s1_pane_w, s1_pane_h)
    s2_box = (s2_x, s2_y, s2_pane_w, s2_pane_h)

    # Portrait crop for Speaker 1 and Speaker 2
    crop_x_s1 = max(active_x, min(s1_cx - target_crop_w // 2, active_x + active_w - target_crop_w))
    crop_x_s2 = max(active_x, min(s2_cx - target_crop_w // 2, active_x + active_w - target_crop_w))

    dom_faces = _dominant_face_boxes(valid_face_samples)

    def _create_shot(start_t: float, end_t: float, mode_type: str) -> ShotPlan:
        if mode_type == "s1":
            return ShotPlan(
                start=round(start_t, 2),
                end=round(end_t, 2),
                face_boxes=dom_faces,
                mode="portrait_face",
                crop_x=crop_x_s1,
                crop_y=active_y,
                crop_w=target_crop_w,
                crop_h=active_h,
                center_y=s1_cy,
                margin_v=460,
            )
        elif mode_type == "s2":
            return ShotPlan(
                start=round(start_t, 2),
                end=round(end_t, 2),
                face_boxes=dom_faces,
                mode="portrait_face",
                crop_x=crop_x_s2,
                crop_y=active_y,
                crop_w=target_crop_w,
                crop_h=active_h,
                center_y=s2_cy,
                margin_v=460,
            )
        else:
            return ShotPlan(
                start=round(start_t, 2),
                end=round(end_t, 2),
                face_boxes=dom_faces,
                mode="split_screen",
                speaker1_box=s1_box,
                speaker2_box=s2_box,
                margin_v=0,
                subtitle_placement="divider",
                subtitle_alignment=5,
            )

    duration = rel_e - rel_s
    if duration < 2.0 or not shot_samples:
        s1_motions = [s[3][0] for s in shot_samples if len(s) > 3 and len(s[3]) == 2]
        s2_motions = [s[3][1] for s in shot_samples if len(s) > 3 and len(s[3]) == 2]
        avg_m1 = float(np.mean(s1_motions)) if s1_motions else 0.0
        avg_m2 = float(np.mean(s2_motions)) if s2_motions else 0.0
        if avg_m1 >= _MIN_SPEECH_LIP_MOTION and avg_m1 >= SPEAKER_HYSTERESIS * max(0.5, avg_m2):
            return [_create_shot(rel_s, rel_e, "s1")]
        elif avg_m2 >= _MIN_SPEECH_LIP_MOTION and avg_m2 >= SPEAKER_HYSTERESIS * max(0.5, avg_m1):
            return [_create_shot(rel_s, rel_e, "s2")]
        else:
            return [_create_shot(rel_s, rel_e, "both")]

    raw_states: List[Tuple[float, str]] = []
    for s in shot_samples:
        t = s[0]
        if len(s) > 3 and len(s[3]) == 2:
            m1, m2 = s[3][0], s[3][1]
            if m1 >= _MIN_SPEECH_LIP_MOTION and m1 >= SPEAKER_HYSTERESIS * max(0.5, m2):
                st = "s1"
            elif m2 >= _MIN_SPEECH_LIP_MOTION and m2 >= SPEAKER_HYSTERESIS * max(0.5, m1):
                st = "s2"
            elif m1 >= _MIN_SPEECH_LIP_MOTION and m2 >= _MIN_SPEECH_LIP_MOTION:
                st = "both"
            else:
                st = "silence"
        else:
            st = "silence"
        raw_states.append((t, st))

    if not raw_states:
        return [_create_shot(rel_s, rel_e, "both")]

    initial_st = "both"
    for _, st in raw_states:
        if st != "silence":
            initial_st = st
            break

    segments: List[Tuple[float, float, str]] = []
    seg_start = rel_s
    current_mode = initial_st

    idx = 0
    while idx < len(raw_states):
        t, st = raw_states[idx]
        if st != "silence" and st != current_mode:
            lookahead_end = t
            j = idx
            while j < len(raw_states) and raw_states[j][1] == st:
                lookahead_end = raw_states[j][0]
                j += 1
            st_dur = lookahead_end - t
            if st_dur <= INTERJECTION_MAX_S:
                idx = j
                continue

            if (t - seg_start) >= SPEAKER_MIN_DWELL_S:
                segments.append((seg_start, t, current_mode))
                seg_start = t
                current_mode = st
                idx = j
                continue

        idx += 1

    if rel_e > seg_start:
        final_dur = rel_e - seg_start
        if final_dur < SPEAKER_MIN_DWELL_S and segments:
            prev_s, _, prev_m = segments[-1]
            segments[-1] = (prev_s, rel_e, prev_m)
        else:
            segments.append((seg_start, rel_e, current_mode))

    if not segments:
        return [_create_shot(rel_s, rel_e, "both")]

    return [_create_shot(s_s, s_e, m) for s_s, s_e, m in segments]


def analyze_faces_in_clip(
    video_path: Path,
    start_time: float,
    end_time: float,
    sample_fps: float = 2.5
) -> FramingDecision:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[-] Could not open video {video_path}. Defaulting to blur_stack.")
        return FramingDecision(mode='blur_stack', face_count=0)

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    clip_duration = max(0.0, end_time - start_time)
    start_frame, end_frame = _frame_window(start_time, end_time, fps)
    frame_step = max(1, int(round(fps / sample_fps)))

    # Detect letterbox active content area
    active_x, active_y, active_w, active_h = detect_letterbox_margins(cap, start_frame, end_frame)
    if active_y > 0 or active_h < height:
        print(f"[*] Active Content Detected: Y={active_y}..{active_y + active_h} (Height: {active_h}px, stripped {height - active_h}px black bars)")

    target_crop_w = int(active_h * (9 / 16))
    if target_crop_w > active_w:
        target_crop_w = active_w

    detector = get_face_detector()
    detected_cuts = detect_clip_shots(video_path, start_time, end_time, min_shot_duration=0.4)
    print(f"[*] Visual Shot Segmentation: Detected {len(detected_cuts)} distinct camera cuts.")

    timeline_samples: List[Tuple[float, List[FaceBox], bool, List[float]]] = []
    prev_mouth_patches: List[Optional[np.ndarray]] = [None, None]
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame
    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if (current_frame - start_frame) % frame_step == 0:
            rel_t = _relative_frame_timestamp(
                cap, start_time, current_frame, start_frame, fps, clip_duration
            )
            faces: List[FaceBox] = []
            is_slide = False

            if (current_frame - start_frame) % (frame_step * 2) == 0:
                try:
                    scene_info = classify_frame_scene(frame)
                    if _is_visual_layout_scene(scene_info):
                        is_slide = True
                except Exception:
                    pass

            faces = detect_faces_in_frame(detector, frame, width, height, active_y=active_y, active_h=active_h)
            faces.sort(key=lambda f: f.center_x)

            mouth_scores: List[float] = []
            if len(faces) == 2:
                s1_score, prev_mouth_patches[0] = _estimate_mouth_motion(prev_mouth_patches[0], frame, faces[0])
                s2_score, prev_mouth_patches[1] = _estimate_mouth_motion(prev_mouth_patches[1], frame, faces[1])
                mouth_scores = [s1_score, s2_score]
            else:
                prev_mouth_patches = [None, None]

            timeline_samples.append((rel_t, faces, is_slide, mouth_scores))

        current_frame += 1

    cap.release()

    if not timeline_samples:
        return FramingDecision(
            mode='blur_stack', face_count=0, video_width=width, video_height=height,
            active_x=active_x, active_y=active_y, active_w=active_w, active_h=active_h
        )

    # Compute global anchor across entire clip for robust temporal anchoring
    all_cx = [f.center_x for s in timeline_samples for f in s[1] if active_x <= f.center_x <= active_x + active_w]
    global_anchor_cx = int(np.median(all_cx)) if all_cx else (active_x + active_w // 2)

    def _is_two_speaker_frame(face_pair: List[FaceBox]) -> bool:
        if len(face_pair) != 2:
            return False
        f1, f2 = face_pair[0], face_pair[1]
        x_dist = abs(f1.center_x - f2.center_x)
        y_dist = abs(f1.center_y - f2.center_y)
        area1 = f1.w * f1.h
        area2 = f2.w * f2.h
        if area1 == 0 or area2 == 0:
            return False
        area_ratio = area1 / area2
        # Require faces to be at least 25% of frame width apart (prevents treating
        # a single face detected twice as two speakers)
        # Also allow smaller faces in wide-angle studio podcast sets (down to 0.3% of frame area, ~80x80px)
        min_face_area = (active_w * active_h) * 0.003
        return (
            x_dist >= active_w * 0.25 and
            y_dist <= active_h * 0.35 and
            0.25 <= area_ratio <= 4.0 and
            area1 >= min_face_area and
            area2 >= min_face_area
        )

    shot_plans: List[ShotPlan] = []
    last_known_cx = None

    for s_start, s_end in detected_cuts:
        rel_s = round(max(0.0, s_start - start_time), 2)
        rel_e = round(min(clip_duration, s_end - start_time), 2)
        if rel_e - rel_s < 0.2:
            continue

        shot_samples = [s for s in timeline_samples if rel_s <= s[0] <= rel_e]
        if not shot_samples:
            shot_samples = timeline_samples

        pres_count = sum(1 for s in shot_samples if s[2])
        pres_ratio = pres_count / max(1, len(shot_samples))

        valid_face_samples = [s[1] for s in shot_samples if s[1]]
        two_speaker_samples = [f for f in valid_face_samples if _is_two_speaker_frame(f)]

        # A slide with the host still on screen (screen-share with a camera inset)
        # is the single most common podcast visual. Detect the small persistent
        # host face and keep it as a picture-in-picture inset instead of throwing
        # the face away and rendering a bare slide.
        pip_host_box = _picture_in_picture_host_box(valid_face_samples)
        pip_eligible = pip_host_box is not None and (pip_host_box[2] * pip_host_box[3]) < (
            active_w * active_h * PIP_MAX_HOST_AREA_RATIO
        )

        # 1. Verified Presentation Slide (strictly requiring visual confirmation, not just lack of frontal faces)
        if pres_ratio >= 0.50:
            if pip_eligible:
                shot_plans.append(ShotPlan(
                    start=rel_s,
                    end=rel_e,
                    mode='pip_slide',
                    crop_x=active_x,
                    crop_y=active_y,
                    crop_w=active_w,
                    crop_h=active_h,
                    speaker1_box=pip_host_box,
                    margin_v=400
                ))
            else:
                shot_plans.append(ShotPlan(
                    start=rel_s,
                    end=rel_e,
                    mode='presentation_slide',
                    crop_x=active_x,
                    crop_y=active_y,
                    crop_w=active_w,
                    crop_h=active_h,
                    margin_v=400
                ))
        elif not valid_face_samples:
            shot_plans.append(ShotPlan(
                start=rel_s,
                end=rel_e,
                mode='presentation_slide',
                crop_x=active_x,
                crop_y=active_y,
                crop_w=active_w,
                crop_h=active_h,
                margin_v=400
            ))
        # 2. Dynamic Split-Screen or Active Solo Speaker — smart speaker switching (OpusClip Pro style)
        elif valid_face_samples and (len(two_speaker_samples) / len(valid_face_samples) >= 0.40):
            subshots = _segment_two_speaker_shots(
                rel_s=rel_s,
                rel_e=rel_e,
                shot_samples=shot_samples,
                valid_face_samples=valid_face_samples,
                two_speaker_samples=two_speaker_samples,
                target_crop_w=target_crop_w,
                active_x=active_x,
                active_y=active_y,
                active_w=active_w,
                active_h=active_h,
            )
            shot_plans.extend(subshots)
        # 3. Portrait Solo Face (with temporal anchor memory + skin-tone fallback)
        else:
            eye_level_y = active_y + active_h * 0.35
            timed_single_faces = [(s[0], min(s[1], key=lambda f: abs(f.center_y - eye_level_y))) for s in shot_samples if s[1]]
            
            single_faces = [f[1] for f in timed_single_faces] if timed_single_faces else []
            if not single_faces and valid_face_samples:
                single_faces = [min(faces, key=lambda f: abs(f.center_y - eye_level_y)) for faces in valid_face_samples]

            if single_faces:
                # If detected faces are split across left and right halves of the frame (bimodal),
                # taking np.median produces the empty midpoint between two people (e.g. centering on a car or empty wall).
                # Instead, cluster into left and right, and focus on one speaker's face.
                left_faces = [f for f in single_faces if f.center_x < active_x + active_w * 0.45]
                right_faces = [f for f in single_faces if f.center_x > active_x + active_w * 0.55]
                if left_faces and right_faces and len(left_faces) + len(right_faces) >= len(single_faces) * 0.7:
                    if last_known_cx is not None:
                        left_dist = abs(np.median([f.center_x for f in left_faces]) - last_known_cx)
                        right_dist = abs(np.median([f.center_x for f in right_faces]) - last_known_cx)
                        chosen_faces = left_faces if left_dist <= right_dist else right_faces
                    else:
                        chosen_faces = left_faces if len(left_faces) >= len(right_faces) else right_faces
                    avg_cx = int(np.median([f.center_x for f in chosen_faces]))
                    avg_cy = int(np.median([f.center_y for f in chosen_faces]))
                else:
                    avg_cx = int(np.median([f.center_x for f in single_faces]))
                    avg_cy = int(np.median([f.center_y for f in single_faces]))
                last_known_cx = avg_cx
            else:
                # Skin-tone heatmap fallback: verified against known speaker anchors.
                # Prevents crop landing on mic stand, warm wooden shelves, or lamps.
                skin_cx = None
                cap_inner = cv2.VideoCapture(str(video_path))
                if cap_inner.isOpened():
                    mid_t = (rel_s + rel_e) / 2.0
                    cap_inner.set(cv2.CAP_PROP_POS_MSEC, (start_time + mid_t) * 1000)
                    ret_s, sample_frame = cap_inner.read()
                    cap_inner.release()
                    if ret_s and sample_frame is not None:
                        skin_cx = _skin_tone_center_x(sample_frame, active_x, active_y, active_w, active_h)
                        if skin_cx is not None:
                            anchor_ref = last_known_cx if last_known_cx is not None else global_anchor_cx
                            if anchor_ref is not None and abs(skin_cx - anchor_ref) > active_w * 0.22:
                                print(f"[*] Rejecting skin_cx={skin_cx} as background artifact (too far from anchor {anchor_ref})")
                                skin_cx = None
                            else:
                                print(f"[*] Skin-tone fallback: verified person at x={skin_cx} (no face detected in shot)")
                # Priority: verified skin-tone near anchor → last known position → global anchor
                if last_known_cx is not None:
                    avg_cx = skin_cx if skin_cx is not None else last_known_cx
                    print(f"[*] Temporal anchor fallback: using x={avg_cx}")
                elif skin_cx is not None:
                    avg_cx = skin_cx
                else:
                    avg_cx = global_anchor_cx
                    print(f"[*] Global anchor fallback: using x={avg_cx}")
                avg_cy = int(eye_level_y)

            crop_x = max(active_x, min(avg_cx - target_crop_w // 2, active_x + active_w - target_crop_w))
            # Protect speaker head/ears from border cutoff: ensure at least 35% face width padding
            if single_faces:
                med_fw = int(np.median([f.w for f in single_faces]))
                med_fx = int(np.median([f.x for f in single_faces]))
                min_safe_pad = int(med_fw * 0.35)
                if med_fx - crop_x < min_safe_pad:
                    crop_x = max(active_x, med_fx - min_safe_pad)
                elif (crop_x + target_crop_w) - (med_fx + med_fw) < min_safe_pad:
                    crop_x = min(active_x + active_w - target_crop_w, med_fx + med_fw + min_safe_pad - target_crop_w)

            raw_timeline = [(round(s - rel_s, 2), f.center_x) for s, f in timed_single_faces if isinstance(f, FaceBox)] if timed_single_faces else []
            timeline = _stable_face_timeline(raw_timeline, active_w)

            # High-retention opening hook: Apply 1.15x punch-zoom on the first 3.5 seconds
            zoom = 1.15 if (rel_s < 3.2 and ENABLE_PUNCH_ZOOM) else 1.0

            shot_plans.append(ShotPlan(
                start=rel_s,
                end=rel_e,
                face_boxes=_dominant_face_boxes(valid_face_samples),
                mode='portrait_face',
                crop_x=crop_x,
                crop_y=active_y,
                crop_w=target_crop_w,
                crop_h=active_h,
                center_y=avg_cy,
                zoom_factor=zoom,
                face_centers_timeline=timeline,
                margin_v=460
            ))

    if shot_plans:
        shot_plans[0].start = 0.0
        shot_plans[-1].end = clip_duration
        for idx in range(len(shot_plans) - 1):
            shot_plans[idx].end = shot_plans[idx + 1].start

    modes = set(s.mode for s in shot_plans)
    has_dynamic_shot_data = any(
        s.zoom_factor > 1.01
        or (
            len(s.face_centers_timeline) >= 2
            and abs(s.face_centers_timeline[-1][1] - s.face_centers_timeline[0][1]) > 40
        )
        for s in shot_plans
    )
    # A single picture-in-picture shot must still route through the per-shot
    # renderer, otherwise it falls through to blur_stack and the host inset is
    # silently discarded.
    has_pip_shot = 'pip_slide' in modes
    if len(shot_plans) > 1 or has_dynamic_shot_data or has_pip_shot:
        return FramingDecision(
            mode='multi_shot_dynamic',
            face_count=len(shot_plans),
            shots=shot_plans,
            video_width=width,
            video_height=height,
            active_x=active_x,
            active_y=active_y,
            active_w=active_w,
            active_h=active_h
        )

    if modes == {'presentation_slide'}:
        return FramingDecision(
            mode='blur_stack', face_count=0, video_width=width, video_height=height,
            active_x=active_x, active_y=active_y, active_w=active_w, active_h=active_h
        )

    if modes == {'split_screen'}:
        s = shot_plans[0]
        return FramingDecision(
            mode='split_screen',
            face_count=2,
            speaker1_box=s.speaker1_box,
            speaker2_box=s.speaker2_box,
            video_width=width,
            video_height=height,
            active_x=active_x,
            active_y=active_y,
            active_w=active_w,
            active_h=active_h
        )

    first_shot = shot_plans[0] if shot_plans else None
    crop_x = first_shot.crop_x if first_shot else (active_x + (active_w - target_crop_w) // 2)
    return FramingDecision(
        mode='single_smooth',
        face_count=1,
        smoothed_center_x=crop_x + target_crop_w // 2,
        smoothed_center_y=first_shot.center_y if first_shot else (active_y + active_h // 3),
        video_width=width,
        video_height=height,
        active_x=active_x,
        active_y=active_y,
        active_w=active_w,
        active_h=active_h
    )
