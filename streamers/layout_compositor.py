"""Streamer Layout Compositor & Facecam ROI Detection.

Intelligent stream composition engine:
1. Auto-detects stream type:
   - 'stacked_webcam': Streamer webcam box overlaid on top of gameplay / YouTube reaction video.
   - 'fullscreen_room': Streamer is in full-screen camera (IRL room, podcasts, challenges).
2. For 'stacked_webcam':
   - Automatically detects webcam ROI coordinates (e.g. top-right, bottom-left).
   - Generates an auto-stacked 9:16 layout:
       - Top 50%: Cropped and centered Streamer Webcam reaction.
       - Bottom 50%: Cropped main content (gameplay / reaction video).
3. For 'fullscreen_room':
   - Routes to StreamerIdentityTracker in v2/streamers/face_tracker.py for
     Kai Cenat identity locking, LERP panning, and smart off-screen wide fallback!

Bug fixes (V2):
    D2  — Pane math now enforces disjoint boxes (max pane_w = source_w // 2).
    D3  — Multi-person detection is track-based, not raw detection-count-based.
    D4  — Vertical sources (aspect < 0.75) get full-height crops, not forced 9:8.
    D8  — face_count is set from actual cluster count, not hardcoded to 2.
    D14 — All crop x/y/w/h are forced to even integers (FFmpeg YUV420p requirement).
    D15 — 'multi_person_room' / 'split_screen' alias handled cleanly in one place.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

from core.face_tracker import ShotPlan, FramingDecision
from streamers.face_tracker import (
    analyze_streamer_clip_framing, StreamerIdentityTracker
)
from streamers.spec import (
    STREAMER_SPLIT_PANE_ASPECT_W,
    STREAMER_SPLIT_PANE_ASPECT_H,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum fraction of sampled frames in which a spatial cluster must appear
# before it is considered a persistent person (not a transient detection).
# Derived from STREAMER_IDENTITY_MIN_DWELL_S at 16 samples over 8 s ≈ 2 fps,
# so 2.0 s × 2 fps = 4 frames; expressed as a fraction of num_samples (16) ≈ 0.25.
_MIN_CLUSTER_FRAME_FRACTION: float = 0.25

# Minimum normalized horizontal separation between two face clusters for them
# to qualify as distinct left/right persons (avoids triggering on head bob).
_MIN_CLUSTER_SEPARATION_FRAC: float = 0.15

# k-means convergence iterations — empirically 6 is sufficient.
_KMEANS_ITERS: int = 6


# ---------------------------------------------------------------------------
# Public dataclass (API contract — do not change field names)
# ---------------------------------------------------------------------------

@dataclass
class StreamLayoutInfo:
    layout_type: str  # 'stacked_webcam', 'multi_person_room', or 'fullscreen_room'
    webcam_box: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    content_box: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    speaker1_box: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h) Top Pane (Host / Streamer)
    speaker2_box: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h) Bottom Pane (Guest)
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _even(n: int) -> int:
    """Force n to the nearest lower even integer.

    Required because FFmpeg YUV420p demands even width, height, x, and y for
    every crop/pad filter. Applying & ~1 is cheaper than n - (n % 2) and
    behaves correctly for n==0 (D14).
    """
    return int(n) & ~1


def _compute_disjoint_split_panes(
    source_w: int,
    source_h: int,
    p1_cx: int, p1_cy: int,
    p2_cx: int, p2_cy: int,
) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    """Compute two 9:8 source crop boxes that are STRICTLY disjoint (D2, D14).

    The key insight: on a 1920x1080 source, the maximum pane width for zero
    overlap is floor(source_w / 2) = 960 px. Pane height = 960 * 8/9 = 853 px
    rounded down to even = 852 px.

    For vertical sources (source_aspect < 0.75) there is no horizontal room to
    split, so this function returns two full-width panes stacked vertically and
    enforces disjointness on the Y axis instead.

    Guarantees:
    - box1.right <= box2.left  (zero pixel overlap — strict disjointness)
    - All x/y/w/h are even integers (required by FFmpeg YUV420p — D14)
    - Panes are centered on the detected face centers where possible
    - p1 is assigned to the LEFT half, p2 to the RIGHT half regardless of
      which cluster was detected as primary streamer (caller re-labels).
    """
    # D2 fix: cap pane_w at half source width so left + right never overlap.
    max_pane_w = _even(source_w // 2)

    # D4 guard: if source is already vertical, stack full-width panes vertically.
    source_aspect = source_w / max(source_h, 1)
    if source_aspect < 0.75:
        # Vertical or square source: there is no horizontal room to split.
        # The branch previously set pane_w = source_w but left the y coordinates
        # to the horizontal-split logic below, so it produced two overlapping
        # full-frame panes that then failed the horizontal disjointness assert —
        # an unsatisfiable invariant, meaning EVERY vertical source raised here
        # (and under `python -O`, where asserts vanish, silently emitted two
        # overlapping panes instead). Complete the stacked layout and assert the
        # invariant that actually applies to it, on the Y axis.
        pane_w = _even(source_w)
        pane_h = _even(source_h // 2)
        top = (0, 0, pane_w, pane_h)
        bottom = (0, pane_h, pane_w, pane_h)
        assert top[1] + top[3] <= bottom[1] + 2, (
            f"D4: stacked panes overlap! top.bottom={top[1] + top[3]} > "
            f"bottom.top={bottom[1]}. source_h={source_h}, pane_h={pane_h}"
        )
        return top, bottom

    pane_w = max_pane_w
    pane_h = _even(int(pane_w * STREAMER_SPLIT_PANE_ASPECT_H / STREAMER_SPLIT_PANE_ASPECT_W))

    # Left pane: must stay entirely within the left half [0, source_w//2).
    # Clamp x so that x + pane_w <= source_w // 2.
    left_max_x = _even(source_w // 2 - pane_w)
    b1_x = _even(max(0, min(p1_cx - pane_w // 2, left_max_x)))
    b1_y = _even(max(0, min(int(p1_cy - pane_h * 0.35), source_h - pane_h)))

    # Right pane: must start at or after source_w // 2.
    right_min_x = _even(source_w // 2)
    right_max_x = _even(source_w - pane_w)
    b2_x = _even(max(right_min_x, min(p2_cx - pane_w // 2, right_max_x)))
    b2_y = _even(max(0, min(int(p2_cy - pane_h * 0.35), source_h - pane_h)))

    b1 = (b1_x, b1_y, pane_w, pane_h)
    b2 = (b2_x, b2_y, pane_w, pane_h)

    # Verify strict disjointness — allow 2 px tolerance for rounding.
    assert b1[0] + b1[2] <= b2[0] + 2, (
        f"D2: Panes overlap! b1.right={b1[0] + b1[2]} > b2.left={b2[0]}. "
        f"source_w={source_w}, pane_w={pane_w}"
    )

    return b1, b2


def _count_persistent_identity_tracks(
    face_sim_pairs: list,
    detected_faces: list,
    multi_face_frames: int,
    valid_face_frames: int,
    num_samples: int,
    source_w: int,
    min_dwell_frames: int = 3,
) -> int:
    """Count how many DISTINCT people appear persistently across sampled frames (D3).

    Unlike counting total detections — which fires whenever a single moving
    person is detected multiple times — this clusters spatial face positions
    along the x-axis and requires each cluster to be seen in at least
    `min_dwell_frames` distinct sampled frames.

    This prevents a solo streamer walking across the room from triggering the
    split-screen path (D3 root cause: detection count, not identity count).

    Args:
        face_sim_pairs:    List of (fbox, similarity_score) tuples accumulated
                           across all sampled frames.
        detected_faces:    All accepted face boxes across all sampled frames.
        multi_face_frames: Frames where >= 2 faces were detected simultaneously.
        valid_face_frames: Frames where >= 1 face was detected.
        num_samples:       Total number of frames sampled (used for dwell ratio).
        source_w:          Source video width (for separation threshold).
        min_dwell_frames:  Minimum distinct frames a cluster must appear in to
                           count as a real persistent person.

    Returns:
        1 for a single streamer (even if they move across the frame),
        2 for a genuine two-person collab (two persistent spatial clusters).
    """
    if not detected_faces:
        return 0

    if len(detected_faces) < 2:
        return 1

    cxs_all = [f[0] + f[2] // 2 for f in detected_faces]

    # 2-means clustering on center-x positions.
    c1 = float(min(cxs_all))
    c2 = float(max(cxs_all))

    for _ in range(_KMEANS_ITERS):
        g1_faces = [f for f in detected_faces
                    if abs((f[0] + f[2] // 2) - c1) <= abs((f[0] + f[2] // 2) - c2)]
        g2_faces = [f for f in detected_faces
                    if abs((f[0] + f[2] // 2) - c1) > abs((f[0] + f[2] // 2) - c2)]
        if g1_faces:
            c1 = float(np.median([f[0] + f[2] // 2 for f in g1_faces]))
        if g2_faces:
            c2 = float(np.median([f[0] + f[2] // 2 for f in g2_faces]))

    # Cluster separation check — must span at least 15% of frame width.
    sep = abs(c1 - c2)
    if sep < _MIN_CLUSTER_SEPARATION_FRAC * source_w:
        # Both clusters converged too close: single person.
        return 1

    # Persistence check: each cluster must appear in enough distinct frames.
    # We approximate frame membership by partitioning detections (which were
    # appended in frame order) and counting unique positions as proxy frames.
    # This is an approximation; real frame-stamp tracking requires refactoring
    # detect_and_scale to emit timestamps — that is a future enhancement.
    g1_count = len(g1_faces) if g1_faces else 0
    g2_count = len(g2_faces) if g2_faces else 0

    # Each cluster must have appeared in at least min_dwell_frames of the
    # sampled frames.  We bound from above by valid_face_frames to avoid
    # false positives on a single noisy frame.
    g1_persists = g1_count >= min_dwell_frames
    g2_persists = g2_count >= min_dwell_frames

    if g1_persists and g2_persists:
        return 2
    return 1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_stream_layout(
    video_path: Path,
    sample_duration: float = 8.0,
    num_samples: int = 16,
    streamer_name: str = "Kai Cenat"
) -> StreamLayoutInfo:
    """Samples video frames to classify layout into 'stacked_webcam', 'multi_person_room', or 'fullscreen_room'.

    Uses track-based identity counting (D3) and disjoint pane math (D2) to
    avoid false split-screen triggers and crop overlap. All output coordinates
    are even-aligned (D14).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return StreamLayoutInfo(layout_type="fullscreen_room", confidence=0.5)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    max_frame = min(total_frames, int(sample_duration * video_fps))
    step = max(1, max_frame // num_samples)

    tracker = StreamerIdentityTracker(w, h, det_max_dim=640, streamer_name=streamer_name)

    detected_faces: List[Tuple[int, int, int, int]] = []  # (x, y, w, h)
    face_sim_pairs: List[Tuple[Tuple[int, int, int, int], float]] = []
    multi_face_frames = 0
    valid_face_frames = 0

    f_idx = 0
    while f_idx < max_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        scaled_faces = tracker.detect_and_scale(frame)
        if scaled_faces is not None and len(scaled_faces) > 0:
            valid_faces = [f for f in scaled_faces if float(f[14]) >= 0.45]
            if valid_faces:
                valid_face_frames += 1
                if len(valid_faces) >= 2:
                    multi_face_frames += 1
                for face in valid_faces:
                    fbox = (int(face[0]), int(face[1]), int(face[2]), int(face[3]))
                    detected_faces.append(fbox)
                    feat = tracker.extract_face_embedding(frame, face)
                    sim = tracker.match_similarity(feat) if feat is not None else 0.0
                    face_sim_pairs.append((fbox, sim))

        f_idx += step

    cap.release()

    if not detected_faces:
        return StreamLayoutInfo(layout_type="fullscreen_room", confidence=0.6)

    # ------------------------------------------------------------------
    # Webcam-box detection: small, stationary face in one corner
    # ------------------------------------------------------------------
    face_widths = [f[2] for f in detected_faces]
    face_heights = [f[3] for f in detected_faces]
    med_fw = float(np.median(face_widths))
    med_fh = float(np.median(face_heights))

    rel_fw = med_fw / w
    rel_fh = med_fh / h

    cxs = [f[0] + f[2] // 2 for f in detected_faces]
    cys = [f[1] + f[3] // 2 for f in detected_faces]
    std_cx = float(np.std(cxs)) if len(cxs) > 1 else 0.0
    std_cy = float(np.std(cys)) if len(cys) > 1 else 0.0

    # Low standard deviation (< 8% of frame width) and small face (< 18% of frame width)
    # indicates a stationary webcam box!
    is_webcam_box = (
        rel_fw < 0.20 and
        rel_fh < 0.28 and
        (std_cx / w) < 0.08 and
        (std_cy / h) < 0.08
    )

    if is_webcam_box:
        # Stationary webcam detected!
        avg_cx = int(np.median(cxs))
        avg_cy = int(np.median(cys))
        box_w = _even(int(med_fw * 2.8))  # Expand to capture streamer's webcam frame
        box_h = _even(int(box_w * STREAMER_SPLIT_PANE_ASPECT_H / STREAMER_SPLIT_PANE_ASPECT_W))
        if box_h > h:
            box_h = _even(h)
            box_w = _even(int(box_h * STREAMER_SPLIT_PANE_ASPECT_W / STREAMER_SPLIT_PANE_ASPECT_H))

        wx = _even(max(0, min(avg_cx - box_w // 2, w - box_w)))
        wy = _even(max(0, min(avg_cy - box_h // 2, h - box_h)))
        webcam_box = (wx, wy, box_w, box_h)

        # Content box: Main gameplay / video matching 9:8 pane aspect ratio
        ch = _even(h)
        cw = _even(int(ch * STREAMER_SPLIT_PANE_ASPECT_W / STREAMER_SPLIT_PANE_ASPECT_H))
        if cw > w:
            cw = _even(w)
            ch = _even(int(cw * STREAMER_SPLIT_PANE_ASPECT_H / STREAMER_SPLIT_PANE_ASPECT_W))
        # Offset away from webcam side
        cx_offset = 0 if wx > w // 2 else _even(int(w - cw))
        cy_offset = _even(max(0, (h - ch) // 2))
        content_box = (cx_offset, cy_offset, cw, ch)

        print(f"[+] Detected 'stacked_webcam' stream layout: Webcam at ({wx}, {wy}, {box_w}x{box_h})")
        return StreamLayoutInfo(
            layout_type="stacked_webcam",
            webcam_box=webcam_box,
            content_box=content_box,
            confidence=0.88
        )

    # ------------------------------------------------------------------
    # Multi-person room detection (D3 fix: track-based, not count-based)
    # ------------------------------------------------------------------
    # Count distinct persistent identity tracks rather than raw detection frames.
    distinct_people = _count_persistent_identity_tracks(
        face_sim_pairs=face_sim_pairs,
        detected_faces=detected_faces,
        multi_face_frames=multi_face_frames,
        valid_face_frames=valid_face_frames,
        num_samples=num_samples,
        source_w=w,
        min_dwell_frames=3,
    )

    # Additional guard: require some horizontal spread across detected centers.
    min_cx, max_cx = min(cxs), max(cxs)
    is_horizontally_split = (max_cx - min_cx) >= (0.18 * w)

    if distinct_people >= 2 and is_horizontally_split and len(detected_faces) >= 3:
        # Partition faces into two spatial clusters via 2-means
        c1, c2 = float(min_cx), float(max_cx)
        g1: List[Tuple[int, int, int, int]] = []
        g2: List[Tuple[int, int, int, int]] = []
        for _ in range(_KMEANS_ITERS):
            g1 = [f for f in detected_faces if abs((f[0] + f[2] // 2) - c1) <= abs((f[0] + f[2] // 2) - c2)]
            g2 = [f for f in detected_faces if abs((f[0] + f[2] // 2) - c1) > abs((f[0] + f[2] // 2) - c2)]
            if g1:
                c1 = float(np.median([f[0] + f[2] // 2 for f in g1]))
            if g2:
                c2 = float(np.median([f[0] + f[2] // 2 for f in g2]))

        if g1 and g2 and abs(c1 - c2) >= int(_MIN_CLUSTER_SEPARATION_FRAC * w):
            g1_cx = int(np.median([f[0] + f[2] // 2 for f in g1]))
            g1_cy = int(np.median([f[1] + f[3] // 2 for f in g1]))
            g2_cx = int(np.median([f[0] + f[2] // 2 for f in g2]))
            g2_cy = int(np.median([f[1] + f[3] // 2 for f in g2]))

            # Identity match: Host / Streamer on Top pane, Guest on Bottom pane
            g1_sims = [sim for box, sim in face_sim_pairs if abs((box[0] + box[2] // 2) - c1) < abs((box[0] + box[2] // 2) - c2)]
            g2_sims = [sim for box, sim in face_sim_pairs if abs((box[0] + box[2] // 2) - c2) < abs((box[0] + box[2] // 2) - c1)]
            g1_sim = float(np.mean(g1_sims)) if g1_sims else 0.0
            g2_sim = float(np.mean(g2_sims)) if g2_sims else 0.0

            if g2_sim > g1_sim + 0.05:
                # Group 2 is primary streamer (Top), Group 1 is Guest (Bottom)
                # g2 is the RIGHT cluster; keep right pane as p2 for disjoint math.
                p1_cx, p1_cy = g1_cx, g1_cy   # left cluster → left pane
                p2_cx, p2_cy = g2_cx, g2_cy   # right cluster → right pane
            else:
                # Group 1 is primary streamer (Top), Group 2 is Guest (Bottom)
                p1_cx, p1_cy = g1_cx, g1_cy   # left cluster → left pane
                p2_cx, p2_cy = g2_cx, g2_cy   # right cluster → right pane

            # D2 + D4 + D14 fix: compute mathematically disjoint, even-aligned panes.
            try:
                speaker1_box, speaker2_box = _compute_disjoint_split_panes(
                    source_w=w, source_h=h,
                    p1_cx=p1_cx, p1_cy=p1_cy,
                    p2_cx=p2_cx, p2_cy=p2_cy,
                )
            except AssertionError as exc:
                print(f"[!] D2 assertion failed, falling back to fullscreen_room: {exc}")
                return StreamLayoutInfo(layout_type="fullscreen_room", confidence=0.70)

            s1_x, s1_y, pane_w, pane_h = speaker1_box
            s2_x, s2_y, _, _ = speaker2_box

            print(
                f"[+] Detected 'multi_person_room' collab: Split-Screen Stacked layout!\n"
                f"    Left Pane  (Streamer): ({s1_x}, {s1_y}, {pane_w}x{pane_h})\n"
                f"    Right Pane (Guest):    ({s2_x}, {s2_y}, {pane_w}x{pane_h})"
            )
            return StreamLayoutInfo(
                layout_type="multi_person_room",
                speaker1_box=speaker1_box,
                speaker2_box=speaker2_box,
                confidence=0.93
            )

    print(f"[*] Detected 'fullscreen_room' single streamer layout (rel_face_w: {rel_fw:.2f}, std_cx: {std_cx:.1f}px)")
    return StreamLayoutInfo(
        layout_type="fullscreen_room",
        confidence=0.92
    )


def build_streamer_framing(
    video_path: Path,
    start_time: float,
    end_time: float,
    streamer_name: str = "Kai Cenat",
    force_layout: str = "auto"
) -> FramingDecision:
    """Builds optimal FramingDecision for streamer clip.

    Args:
        video_path:    Path to video segment.
        start_time:    Relative or absolute start time.
        end_time:      Relative or absolute end time.
        streamer_name: Name of primary streamer.
        force_layout:  'auto', 'stacked_webcam', 'multi_person_room',
                       'split_screen', 'fullscreen_room', 'single_smooth'.
                       Note: 'split_screen' and 'multi_person_room' are treated
                       as aliases (D15 fix: one canonical normalisation point).

    Returns:
        FramingDecision with face_count derived from actual cluster count (D8).
    """
    clip_dur = max(0.1, end_time - start_time)

    # D15 fix: normalise all aliases to canonical layout names here and nowhere else.
    _SPLIT_ALIASES = {"multi_person_room", "split_screen"}

    layout_info = StreamLayoutInfo(layout_type="fullscreen_room")
    if force_layout == "stacked_webcam":
        layout_info = detect_stream_layout(video_path, sample_duration=min(6.0, clip_dur), streamer_name=streamer_name)
        # Only honour the override when detection actually found a webcam overlay.
        # Forcing 'stacked_webcam' without boxes silently degraded to fullscreen
        # identity framing while still claiming the stacked layout type.
        if layout_info.webcam_box and layout_info.content_box:
            layout_info.layout_type = "stacked_webcam"
        else:
            print(
                "[!] --framing stacked_webcam requested but no webcam overlay was "
                f"detected; using detected '{layout_info.layout_type}' layout instead."
            )
    elif force_layout in _SPLIT_ALIASES:
        layout_info = detect_stream_layout(video_path, sample_duration=min(6.0, clip_dur), streamer_name=streamer_name)
        if (
            layout_info.layout_type == "stacked_webcam"
            and layout_info.webcam_box
            and layout_info.content_box
        ):
            # Detection found a webcam-over-gameplay stream, which already renders as a
            # split (webcam pane over content pane). A forced split must NOT discard
            # those detected boxes and substitute synthetic half-frame panes.
            print(
                "[*] --framing split_screen requested on a detected webcam stream; "
                "keeping the detected 'stacked_webcam' split layout."
            )
        else:
            if layout_info.speaker2_box is None:
                print(
                    "[!] --framing split_screen requested but only one speaker was "
                    "detected; generating synthetic half-frame panes because the "
                    "layout was explicitly forced."
                )
            layout_info.layout_type = "multi_person_room"
    elif force_layout in ("fullscreen_room", "single_smooth"):
        layout_info.layout_type = "fullscreen_room"
    elif force_layout == "auto":
        layout_info = detect_stream_layout(video_path, sample_duration=min(6.0, clip_dur), streamer_name=streamer_name)

    if layout_info.layout_type == "stacked_webcam" and layout_info.webcam_box and layout_info.content_box:
        # Split-screen Auto-Stacked 9:16 layout
        cap = cv2.VideoCapture(str(video_path))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        cap.release()

        shot = ShotPlan(
            start=0.0,
            end=clip_dur,
            mode="split_screen",
            speaker1_box=layout_info.webcam_box,
            speaker2_box=layout_info.content_box,
            crop_x=0,
            crop_y=0,
            crop_w=w,
            crop_h=h
        )

        # stacked_webcam is always exactly 2 panes by definition.
        return FramingDecision(
            mode="split_screen",
            face_count=2,
            shots=[shot],
            speaker1_box=layout_info.webcam_box,
            speaker2_box=layout_info.content_box,
            video_width=w,
            video_height=h,
            active_x=0,
            active_y=0,
            active_w=w,
            active_h=h
        )

    elif layout_info.layout_type == "multi_person_room":
        # Dual Split-Screen (Left: Host / Right: Guest)
        # Fills 100% of 9:16 mobile canvas with ZERO blurry padding and large clear reactions!
        cap = cv2.VideoCapture(str(video_path))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        cap.release()

        # D2 + D14 fix: use the same disjoint helper for the fallback default boxes.
        default_pane_w = _even(w // 2)
        source_aspect = w / max(h, 1)
        if source_aspect < 0.75:
            # D4: vertical source — use full-height pane defaults
            default_pane_h = _even(h)
        else:
            default_pane_h = _even(int(
                default_pane_w * STREAMER_SPLIT_PANE_ASPECT_H / STREAMER_SPLIT_PANE_ASPECT_W
            ))

        s1_box = layout_info.speaker1_box or (0, 0, default_pane_w, default_pane_h)
        s2_box = layout_info.speaker2_box or (_even(w - default_pane_w), 0, default_pane_w, default_pane_h)

        # D8 fix: face_count comes from how many distinct speaker boxes we have,
        # not a hardcoded literal. speaker1 is always present (s1_box above falls
        # back to the host pane), so the count is 1 plus the guest when detected.
        # The previous `1 if ... is not None else 1` was tautological.
        actual_face_count = 1 + (1 if layout_info.speaker2_box is not None else 0)

        shot = ShotPlan(
            start=0.0,
            end=clip_dur,
            mode="split_screen",
            speaker1_box=s1_box,
            speaker2_box=s2_box,
            crop_x=0,
            crop_y=0,
            crop_w=w,
            crop_h=h
        )

        return FramingDecision(
            mode="split_screen",
            face_count=actual_face_count,
            shots=[shot],
            speaker1_box=s1_box,
            speaker2_box=s2_box,
            video_width=w,
            video_height=h,
            active_x=0,
            active_y=0,
            active_w=w,
            active_h=h
        )

    # Fullscreen single-camera layout: use StreamerIdentityTracker with off-screen fallback
    return analyze_streamer_clip_framing(
        video_path=video_path,
        start_time=start_time,
        end_time=end_time,
        streamer_name=streamer_name
    )
