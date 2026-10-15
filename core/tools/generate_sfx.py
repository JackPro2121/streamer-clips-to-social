import math
import struct
import wave
from pathlib import Path
import numpy as np

SFX_DIR = Path(__file__).resolve().parent.parent / "assets" / "sfx"
SFX_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_RATE = 44100

def write_wav(filename: Path, samples: np.ndarray):
    samples = np.clip(samples, -1.0, 1.0)
    int_samples = (samples * 32767).astype(np.int16)
    with wave.open(str(filename), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(int_samples.tobytes())
    print(f"[+] Generated: {filename} ({len(samples)/SAMPLE_RATE:.2f}s)")

def generate_pop():
    duration = 0.08
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    # Frequency drops from 950Hz to 150Hz
    freq = 950 * np.exp(-30 * t) + 150
    phase = 2 * np.pi * np.cumsum(freq) / SAMPLE_RATE
    envelope = np.exp(-45 * t)
    samples = np.sin(phase) * envelope * 0.9
    write_wav(SFX_DIR / "pop.wav", samples)

def generate_whoosh():
    duration = 0.35
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    # White noise
    noise = np.random.uniform(-1.0, 1.0, len(t))
    # Gaussian-like envelope centered at 0.18s
    center = 0.18
    width = 0.09
    envelope = np.exp(-0.5 * ((t - center) / width) ** 2)
    # Simple lowpass / bandpass filtering
    window_len = 15
    kernel = np.ones(window_len) / window_len
    smoothed_noise = np.convolve(noise, kernel, mode="same")
    # Add subtle low frequency sweep
    sweep_freq = 120 + 400 * np.sin(np.pi * t / duration)
    sweep_phase = 2 * np.pi * np.cumsum(sweep_freq) / SAMPLE_RATE
    sweep = np.sin(sweep_phase) * envelope * 0.4
    samples = (smoothed_noise * envelope * 0.6) + sweep
    write_wav(SFX_DIR / "whoosh.wav", samples)

def generate_ding():
    duration = 0.65
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    # Bell chime: 1760 Hz (A6), 3520 Hz (octave), 4978 Hz (harmonic)
    fund = 1760.0
    s1 = 0.6 * np.sin(2 * np.pi * fund * t) * np.exp(-8 * t)
    s2 = 0.3 * np.sin(2 * np.pi * (fund * 2) * t) * np.exp(-12 * t)
    s3 = 0.15 * np.sin(2 * np.pi * (fund * 2.83) * t) * np.exp(-18 * t)
    samples = (s1 + s2 + s3) * 0.85
    write_wav(SFX_DIR / "ding.wav", samples)

if __name__ == "__main__":
    generate_pop()
    generate_whoosh()
    generate_ding()
