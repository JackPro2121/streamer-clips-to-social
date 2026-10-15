from faster_whisper import WhisperModel
from pathlib import Path

CLIPS = [
    "clip_1_HOW_TO_KEEP_FIVE_MILLION_DOLLA.mp4",
    "clip_2_TEAM_SO_DID_WANT_STRONG.mp4",
    "clip_3_HOW_MRBEAST_BUILDS_AN_EMPIRE.mp4"
]

model = WhisperModel("base", device="cpu", compute_type="int8")

for name in CLIPS:
    path = Path("downloaded_clips/run_37778680165/viral-podcast-clips") / name
    print(f"\n=================== {name} ===================")
    segments, info = model.transcribe(str(path), word_timestamps=True)
    for s in segments:
        words_preview = " ".join([f"{w.word}({w.start:.2f}-{w.end:.2f})" for w in s.words[:6]])
        print(f"[{s.start:05.2f}s -> {s.end:05.2f}s] {s.text}")
        print(f"   Words: {words_preview} ...")
