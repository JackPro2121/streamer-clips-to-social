import os
import json
import subprocess
import numpy as np
from pathlib import Path
import cv2

CLIPS_DIR = Path(r"D:\Workspace\PODCASTS-CLIPS-TO-SOCIAL\downloaded_clips\run_37903527595")
OUT_DIR = Path(r"D:\Workspace\PODCASTS-CLIPS-TO-SOCIAL\audit_inspection\run_37903527595\deep_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

clips = sorted(list(CLIPS_DIR.glob("*.mp4")))
print(f"Analyzing {len(clips)} clips...")

analysis = {}

for clip in clips:
    name = clip.stem
    print("\n==========================================")
    print(f"Deep Analyzing: {name}")
    print("==========================================")
    clip_out = OUT_DIR / name
    clip_out.mkdir(parents=True, exist_ok=True)
    
    # 1. FFprobe streams inspection
    probe_cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", str(clip)
    ]
    probe = json.loads(subprocess.check_output(probe_cmd).decode("utf-8"))
    v_stream = next((s for s in probe["streams"] if s["codec_type"] == "video"), None)
    a_stream = next((s for s in probe["streams"] if s["codec_type"] == "audio"), None)
    
    v_dur = float(v_stream.get("duration", probe["format"]["duration"]))
    a_dur = float(a_stream.get("duration", probe["format"]["duration"])) if a_stream else 0.0
    
    # 2. Extract Audio WAV for numpy analysis
    wav_path = clip_out / "audio.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", str(clip), "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(wav_path)
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    
    import wave
    with wave.open(str(wav_path), "rb") as wf:
        n_frames = wf.getnframes()
        data = wf.readframes(n_frames)
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        sr = wf.getframerate()
    
    rms = np.sqrt(np.mean(samples**2))
    rms_db = 20 * np.log10(rms + 1e-9)
    peak = np.max(np.abs(samples))
    peak_db = 20 * np.log10(peak + 1e-9)
    
    # Silence detection at -35dB threshold
    frame_len = int(sr * 0.05) # 50ms frames
    n_chunks = len(samples) // frame_len
    chunk_energies = [np.sqrt(np.mean(samples[i*frame_len:(i+1)*frame_len]**2)) for i in range(n_chunks)]
    chunk_dbs = [20 * np.log10(e + 1e-9) for e in chunk_energies]
    silence_chunks = sum(1 for d in chunk_dbs if d < -35)
    silence_ratio = silence_chunks / max(1, n_chunks)
    
    # 3. Transcribe with faster-whisper to inspect words and timing
    print("Transcribing spoken audio with Faster-Whisper...")
    from faster_whisper import WhisperModel
    model = WhisperModel("base.en", device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(wav_path), word_timestamps=True)
    
    seg_list = []
    all_words = []
    for s in segments:
        seg_data = {
            "start": round(s.start, 2),
            "end": round(s.end, 2),
            "text": s.text.strip(),
            "words": [{"word": w.word, "start": round(w.start, 2), "end": round(w.end, 2), "prob": round(w.probability, 2)} for w in (s.words or [])]
        }
        seg_list.append(seg_data)
        for w in (s.words or []):
            all_words.append(w)
            
    # Calculate speech pacing (words per minute)
    duration_min = v_dur / 60.0
    wpm = len(all_words) / duration_min if duration_min > 0 else 0
    
    # 4. Face tracking & Headroom stability via YuNet
    print("Analyzing face framing stability across time...")
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    yunet = cv2.FaceDetectorYN.create(
        model=r"src\assets\face_detection_yunet_2023mar.onnx" if os.path.exists(r"src\assets\face_detection_yunet_2023mar.onnx") else "face_detection_yunet_2023mar.onnx",
        config="",
        input_size=(320, 320),
        score_threshold=0.6
    ) if os.path.exists(r"src\assets\face_detection_yunet_2023mar.onnx") else None
    
    face_samples = []
    sample_interval = int(fps) # 1 sample per second
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_interval == 0:
            h, w = frame.shape[:2]
            t = frame_idx / fps
            
            # Save visual inspection frame
            if t in [0.5, 1.5, 3.0, 10.0, 20.0, 30.0, round(v_dur/2, 1), round(v_dur - 1.5, 1)]:
                cv2.imwrite(str(clip_out / f"frame_{t:05.1f}s.jpg"), frame)
                
            if yunet:
                yunet.setInputSize((w, h))
                _, faces = yunet.detect(frame)
                if faces is not None and len(faces) > 0:
                    best = faces[0]
                    fx, fy, fw, fh = best[0], best[1], best[2], best[3]
                    center_x_norm = (fx + fw/2) / w
                    center_y_norm = (fy + fh/2) / h
                    headroom_norm = fy / h
                    face_samples.append({
                        "t": round(t, 2),
                        "center_x": round(center_x_norm, 3),
                        "center_y": round(center_y_norm, 3),
                        "headroom": round(headroom_norm, 3),
                        "width_norm": round(fw / w, 3),
                        "height_norm": round(fh / h, 3)
                    })
        frame_idx += 1
    cap.release()
    
    avg_center_x = np.mean([f["center_x"] for f in face_samples]) if face_samples else 0.5
    avg_headroom = np.mean([f["headroom"] for f in face_samples]) if face_samples else 0.0
    
    analysis[name] = {
        "duration": round(v_dur, 3),
        "video_codec": v_stream.get("codec_name"),
        "audio_codec": a_stream.get("codec_name"),
        "resolution": f"{v_stream.get('width')}x{v_stream.get('height')}",
        "av_drift_ms": round(abs(v_dur - a_dur) * 1000, 2),
        "audio_metrics": {
            "rms_db": round(float(rms_db), 2),
            "peak_db": round(float(peak_db), 2),
            "silence_ratio": round(float(silence_ratio), 3),
            "wpm": round(float(wpm), 1),
            "total_words": len(all_words)
        },
        "hook_analysis": {
            "first_3_words": " ".join([w.word.strip() for w in all_words[:6]]),
            "hook_lead_in": seg_list[0]["text"] if seg_list else "",
            "hook_duration_s": seg_list[0]["end"] if seg_list else 0.0
        },
        "face_framing": {
            "avg_center_x": round(float(avg_center_x), 3),
            "avg_headroom": round(float(avg_headroom), 3),
            "faces_detected_count": len(face_samples)
        },
        "segments": seg_list
    }
    print(f"Clip {name}: WPM={wpm:.1f}, RMS={rms_db:.1f}dB, Peak={peak_db:.1f}dB, Hook='{seg_list[0]['text'] if seg_list else ''}'")

with open(OUT_DIR / "deep_analysis_report.json", "w", encoding="utf-8") as f:
    json.dump(analysis, f, indent=2)

print("\nDeep analysis complete! Saved to deep_analysis_report.json")
