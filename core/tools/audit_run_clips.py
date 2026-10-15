import os
import sys
import json
import re
import subprocess
from pathlib import Path

# Resolve the pipeline repo root — the directory that contains the engine's
# ``core/`` — because the audit artefacts live beside it, not inside the engine.
_ENGINE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = _ENGINE_ROOT.parent if _ENGINE_ROOT.name == "core" else _ENGINE_ROOT
CLIPS_DIR = REPO_ROOT / "downloaded_clips" / "run_37778680165" / "viral-podcast-clips"
OUTPUT_DIR = REPO_ROOT / "audit_inspection" / "run_37778680165"

sys.path.insert(0, str(REPO_ROOT))
from core.verification import verdict

def get_ffprobe_info(clip_path: Path):
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(clip_path)
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    return json.loads(res.stdout)

def get_ebur128_loudness(clip_path: Path):
    cmd = [
        "ffmpeg",
        "-i", str(clip_path),
        "-filter:a", "ebur128=peak=true",
        "-f", "null",
        "-"
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stderr = res.stderr
    
    # Parse ebur128 summary:
    # Integrated loudness:
    #   I:         -14.2 LUFS
    #   Threshold: -24.3 LUFS
    # Loudness range:
    #   LRA:         5.1 LU
    # True peak:
    #   Peak:       -1.8 dBFS
    i_match = re.search(r"Integrated loudness:\s+I:\s+([-\d.]+)\s+LUFS", stderr)
    lra_match = re.search(r"Loudness range:\s+LRA:\s+([-\d.]+)\s+LU", stderr)
    peak_match = re.search(r"True peak:\s+Peak:\s+([-\d.]+)\s+dBFS", stderr)
    
    return {
        "integrated_lufs": float(i_match.group(1)) if i_match else None,
        "lra": float(lra_match.group(1)) if lra_match else None,
        "true_peak_dbfs": float(peak_match.group(1)) if peak_match else None
    }

def get_silence_intervals(clip_path: Path):
    cmd = [
        "ffmpeg",
        "-i", str(clip_path),
        "-filter:a", "silencedetect=noise=-32dB:d=0.4",
        "-f", "null",
        "-"
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stderr = res.stderr
    
    starts = re.findall(r"silence_start:\s+([\d.]+)", stderr)
    ends = re.findall(r"silence_end:\s+([\d.]+)", stderr)
    durs = re.findall(r"silence_duration:\s+([\d.]+)", stderr)
    
    intervals = []
    for s, e, d in zip(starts, ends, durs):
        intervals.append({
            "start": float(s),
            "end": float(e),
            "duration": float(d)
        })
    return intervals

def get_scene_cuts(clip_path: Path):
    cmd = [
        "ffmpeg",
        "-i", str(clip_path),
        "-filter:v", "select='gt(scene,0.3)',metadata=print",
        "-f", "null",
        "-"
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stderr = res.stderr
    
    # pts_time values
    pts = re.findall(r"pts_time:([\d.]+)", stderr)
    return [float(p) for p in pts]

def extract_frames(clip_path: Path, out_dir: Path, timestamps: list):
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted = []
    for t in timestamps:
        frame_name = f"frame_{t:05.2f}s.jpg"
        target = out_dir / frame_name
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(t),
            "-i", str(clip_path),
            "-vframes", "1",
            "-q:v", "2",
            str(target)
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if target.exists():
            extracted.append(str(target))
    return extracted

def audit_all():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clips = sorted(list(CLIPS_DIR.glob("*.mp4")))
    print(f"Starting audit for {len(clips)} clips in {CLIPS_DIR}...")
    
    results = {}
    out_file = OUTPUT_DIR / "audit_report.json"
    if out_file.exists():
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                results = json.load(f)
        except Exception:
            results = {}

    for clip in clips:
        clip_name = clip.name
        print(f"\n--- Processing: {clip_name} ---")
        clip_frames_dir = OUTPUT_DIR / clip.stem / "frames"
        
        probe = get_ffprobe_info(clip)
        v_stream = next((s for s in probe["streams"] if s["codec_type"] == "video"), None)
        a_stream = next((s for s in probe["streams"] if s["codec_type"] == "audio"), None)
        
        v_dur = float(v_stream.get("duration", probe["format"]["duration"]))
        a_dur = float(a_stream.get("duration", probe["format"]["duration"])) if a_stream else 0.0
        av_drift_ms = abs(v_dur - a_dur) * 1000.0
        
        print("  Extracting loudness (EBU R128)...")
        loudness = get_ebur128_loudness(clip)
        print(f"  Loudness: {loudness}")
        
        print("  Detecting silence intervals...")
        silence = get_silence_intervals(clip)
        print(f"  Silences (>0.4s): {len(silence)}")
        
        print("  Detecting scene cuts...")
        cuts = get_scene_cuts(clip)
        print(f"  Scene cuts count: {len(cuts)}")
        
        # Determine timestamps for frame extraction
        sample_ts = [0.25, 0.75, 1.5, 3.0, 7.0, 15.0, 25.0, 35.0, max(0.5, v_dur - 1.0)]
        for c in cuts[:6]:
            sample_ts.append(round(c, 2))
            sample_ts.append(round(c + 0.3, 2))
        sample_ts = sorted(list(set([t for t in sample_ts if 0 <= t < v_dur])))
        
        print(f"  Extracting {len(sample_ts)} frames...")
        extracted_frames = extract_frames(clip, clip_frames_dir, sample_ts)
        
        results[clip_name] = {
            "path": str(clip),
            "file_size_mb": round(clip.stat().st_size / (1024 * 1024), 2),
            "video_stream": {
                "codec": v_stream.get("codec_name"),
                "width": v_stream.get("width"),
                "height": v_stream.get("height"),
                "fps": eval(v_stream.get("r_frame_rate", "30/1")),
                "duration_s": round(v_dur, 3),
                "bitrate_kbps": round(int(v_stream.get("bit_rate", 0)) / 1000, 1) if v_stream.get("bit_rate") else None
            },
            "audio_stream": {
                "codec": a_stream.get("codec_name") if a_stream else None,
                "sample_rate": a_stream.get("sample_rate") if a_stream else None,
                "channels": a_stream.get("channels") if a_stream else None,
                "duration_s": round(a_dur, 3),
                "bitrate_kbps": round(int(a_stream.get("bit_rate", 0)) / 1000, 1) if a_stream and a_stream.get("bit_rate") else None
            },
            "av_drift_ms": round(av_drift_ms, 2),
            "loudness": loudness,
            "silence_intervals_count": len(silence),
            "silence_intervals": silence,
            "scene_cuts_count": len(cuts),
            "scene_cut_timestamps": cuts,
            "extracted_frames": extracted_frames
        }
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    print("\nPhase 1 (Stream analysis & Frame extraction) COMPLETE for all clips!")
    print("\nPhase 2: Running formal verdict checks...")
    for clip in clips:
        clip_name = clip.name
        if "verdict" in results[clip_name]:
            continue
        print(f"Running verdict on {clip_name}...")
        try:
            verd = verdict.verify(clip)
            results[clip_name]["verdict"] = verd.as_dict()
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"  Verdict passed: {verd.passed}, blocking: {verd.blocking_codes}, warnings: {[f.code for f in verd.warnings]}")
        except Exception as e:
            print(f"  Verdict error: {e}")
            results[clip_name]["verdict_error"] = str(e)

    print(f"\nAll audits completed successfully! Output: {out_file}")

if __name__ == "__main__":
    audit_all()
