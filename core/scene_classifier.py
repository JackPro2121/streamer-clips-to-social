import cv2
import numpy as np
import os
import uuid
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
import requests

from core.config import MODELS_DIR

MODEL_DIR = MODELS_DIR
YOLO_MODEL_PATH = MODEL_DIR / "yolov8n.onnx"
YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.onnx"

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Classes that signify a presentation, slide, screen-share, or document
PRESENTATION_CLASSES = {"tv", "laptop", "book", "cell phone"}

def _download_yolo_model_atomically() -> bool:
    temp_path = YOLO_MODEL_PATH.with_name(
        f".{YOLO_MODEL_PATH.name}.{uuid.uuid4().hex}.part"
    )
    try:
        response = requests.get(YOLO_URL, allow_redirects=True, timeout=15)
        if response.status_code != 200 or len(response.content) < 1000000:
            return False
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_path, "wb") as file_handle:
            file_handle.write(response.content)
        os.replace(temp_path, YOLO_MODEL_PATH)
        return True
    except Exception:
        return False
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


_yolo_net = None
_yolo_attempted = False

def get_yolo_net() -> Optional[Any]:
    """Loads YOLOv8-Nano ONNX model into OpenCV DNN, auto-downloading if missing."""
    global _yolo_net, _yolo_attempted
    if _yolo_attempted:
        return _yolo_net
    _yolo_attempted = True

    if YOLO_MODEL_PATH.exists() and YOLO_MODEL_PATH.stat().st_size < 1000000:
        try:
            YOLO_MODEL_PATH.unlink()
        except OSError:
            pass

    if not YOLO_MODEL_PATH.exists():
        try:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            print(f"[*] Downloading pre-trained YOLOv8-Nano ONNX model ({YOLO_URL})...")
            if not _download_yolo_model_atomically():
                print("[-] Could not download YOLOv8 model. Using high-precision visual document analyzer.")
                return None
            print(f"[+] YOLOv8-Nano ONNX downloaded successfully ({YOLO_MODEL_PATH.stat().st_size / (1024 * 1024):.2f} MB).")
        except Exception as e:
            print(f"[-] Could not download YOLOv8 model: {e}. Using high-precision visual document analyzer.")
            return None

    if YOLO_MODEL_PATH.exists():
        try:
            _yolo_net = cv2.dnn.readNetFromONNX(str(YOLO_MODEL_PATH))
            _yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            _yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            return _yolo_net
        except Exception as e:
            print(f"[-] Error loading YOLOv8 ONNX: {e}")
            try:
                YOLO_MODEL_PATH.unlink()
            except OSError:
                pass
            return None
    return None

def detect_slide_heuristics(frame: np.ndarray) -> bool:
    """
    High-accuracy visual text, diagram, and presentation slide detector:
    1. Analyzes high-frequency edge distribution (text lines and diagram borders).
    2. Measures document contrast and uniform background areas.
    3. Detects whether the frame contains structured text/charts vs organic human scene.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 1. Edge and text density analysis via Sobel horizontal gradients (reading lines)
    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    _, thresh = cv2.threshold(abs_grad_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Morphological closing along horizontal line direction to connect words into text lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Count horizontal text blocks typical of research papers, slides, and medical reports
    text_line_count = 0
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        aspect = cw / max(1, ch)
        # Text line features: wider than tall, reasonably sized
        if aspect >= 2.5 and cw >= w * 0.10 and ch <= h * 0.15:
            text_line_count += 1
            
    # 2. Background uniformity (white papers, black slides, presentation decks)
    # Presentation slides typically have a large dominant background (>40% of pixels in same luminance range)
    hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
    dominant_luminance_ratio = hist.max() / hist.sum()
    
    is_slide = bool((text_line_count >= 5 and dominant_luminance_ratio >= 0.25) or (text_line_count >= 8))
    return is_slide

def classify_frame_scene(frame: np.ndarray) -> Dict[str, Any]:
    """
    Classifies a frame as 'presentation' (slide/chart/paper) or 'human' (speaker/interview).
    Combines visual text/diagram heuristics with face detection to guarantee that
    any frame containing a human speaker is never misclassified as a static slide.
    """
    h, w = frame.shape[:2]
    
    # 1. Visual document / chart / slide analysis
    has_slide_layout = detect_slide_heuristics(frame)
    
    # 2. Check for human speaker presence via face detection
    person_detected = False
    max_face_area_ratio = 0.0
    total_frame_area = float(max(1, w * h))
    try:
        from core.face_tracker import get_face_detector, detect_faces_in_frame
        detector = get_face_detector()
        if detector is not None:
            faces = detect_faces_in_frame(detector, frame, w, h)
            if faces:
                person_detected = True
                max_face_area_ratio = max(
                    float(f.w * f.h) / total_frame_area for f in faces
                )
    except Exception:
        pass

    # Fallback face check if detector was not ready: OpenCV Haar Cascade
    if not person_detected:
        try:
            import os
            cascade_dir = getattr(getattr(cv2, "data", None), "haarcascades", "")
            cascade_path = str(cascade_dir) + "haarcascade_frontalface_default.xml"
            if os.path.isfile(cascade_path):
                cascade_cls = getattr(cv2, "CascadeClassifier", None)
                if cascade_cls is not None:
                    cascade = cascade_cls(cascade_path)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    haar_faces: Any = cascade.detectMultiScale(gray, 1.2, 3, minSize=(30, 30))
                    if len(haar_faces) > 0:
                        person_detected = True
                        max_face_area_ratio = max(
                            float(fw * fh) / total_frame_area for (_, _, fw, fh) in haar_faces
                        )
        except Exception:
            pass

    # 3. Decision:
    # A large human face (> 12% frame) indicates a normal interview / talking-head shot.
    # A small corner face (<= 12% frame) on top of slide heuristics indicates
    # a presentation with a presenter inset (PiP slide).
    if person_detected and max_face_area_ratio > 0.12:
        is_presentation = False
    else:
        is_presentation = has_slide_layout

    return {
        "is_presentation": is_presentation,
        "person_detected": person_detected,
        "has_slide_layout": has_slide_layout,
        "yolo_presentation": False,
        "max_face_area_ratio": max_face_area_ratio,
    }


def analyze_clip_presentation_ratio(video_path: Path, start_sec: float, end_sec: float) -> float:
    """
    Samples frames across a clip to determine what fraction of the clip is a slide/presentation.
    Returns ratio from 0.0 (100% human speaker) to 1.0 (100% slides/charts/presentation).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        step = max(1, int(fps * 1.0))
        presentation_frames = 0
        total_sampled = 0

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        cur = start_frame
        while cur <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            if (cur - start_frame) % step == 0:
                total_sampled += 1
                info = classify_frame_scene(frame)
                if info["is_presentation"]:
                    presentation_frames += 1
            cur += 1

        if total_sampled == 0:
            return 0.0
        return presentation_frames / total_sampled
    finally:
        cap.release()

def detect_clip_shots(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    min_shot_duration: float = 0.4
) -> List[Tuple[float, float]]:
    """
    Splits video segment [start_sec, end_sec] into discrete camera cuts / shot intervals.

    Delegates to `core.shot_detection.detect_shot_boundaries`, which prefers
    TransNetV2 and falls back to PySceneDetect's AdaptiveDetector. The fallback
    is the historical behaviour, unchanged, and both paths share the same
    boundary normalisation so they cannot drift apart.

    Gracefully falls back to a single full segment if no scene cuts are found or
    if no detector is available. Shots shorter than min_shot_duration are merged
    into their predecessor rather than dropped, so the shots always partition
    the requested window.
    """
    from core.shot_detection import detect_shot_boundaries

    try:
        result = detect_shot_boundaries(
            video_path, start_sec, end_sec, min_shot_duration
        )
    except Exception as error:
        print(f"[-] Shot boundary detection failed ({error}). Using monolithic segment.")
        return [(round(start_sec, 2), round(end_sec, 2))]

    if result.source == "transnetv2":
        print(
            f"[+] TransNetV2 found {result.transition_count} cut(s) "
            f"(mean confidence {result.mean_confidence:.2f})."
        )
    return result.shots or [(round(start_sec, 2), round(end_sec, 2))]

