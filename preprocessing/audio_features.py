import logging
import os
import time
import shutil
import subprocess
from functools import lru_cache
import numpy as np
import librosa
import soundfile as sf
from config import (
    DURATION,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    RMS_NORMALIZE,
    RMS_TARGET,
    SAMPLE_RATE,
)

# Call load_audio_full, pad/trim audio to target duration, and optionally normalize by RMS
def load_audio(audio_path, target_sr=SAMPLE_RATE, duration=DURATION):
    y, sr = load_audio_full(audio_path, target_sr=target_sr)
    target_len = int(target_sr * duration)
    if len(y) < target_len:
        pad_width = target_len - len(y)
        y = np.pad(y, (0, pad_width), mode="constant")
    else:
        y = y[:target_len]
    if RMS_NORMALIZE:
        rms = np.sqrt(np.mean(y**2)) if y.size else 0.0
        if rms > 0:
            y = y * (RMS_TARGET / rms)
    return y, target_sr

# Load full audio waveform and resample to target sample rate
def load_audio_full(audio_path, target_sr=SAMPLE_RATE):
    logger = logging.getLogger("uvicorn.error")
    logger.info("Audio load start: %s", audio_path)
    try:
        loader_pref = os.getenv("AUDIO_LOADER", "ffmpeg").lower()
        if loader_pref == "librosa":
            y, sr = librosa.load(audio_path, sr=target_sr, mono=True, res_type="kaiser_fast")
            logger.info("Audio load done via librosa (samples=%d, sr=%d).", y.size, sr)
            return y, sr

        if loader_pref == "soundfile":
            y, sr = sf.read(audio_path, dtype="float32", always_2d=False)
            if y.ndim > 1:
                y = np.mean(y, axis=1)
            if sr != target_sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
            logger.info("Audio load done via soundfile (samples=%d, sr=%d).", y.size, sr)
            return y, sr

        if shutil.which("ffmpeg"):
            cmd = [
                "ffmpeg",
                "-nostdin",
                "-i",
                audio_path,
                "-f",
                "f32le",
                "-ac",
                "1",
                "-ar",
                str(target_sr),
                "pipe:1",
            ]
            timeout_seconds = float(os.getenv("FFMPEG_TIMEOUT_SECONDS", "20"))
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_seconds,
            )
            y = np.frombuffer(result.stdout, dtype=np.float32)
            logger.info("Audio load done via ffmpeg (samples=%d, sr=%d).", y.size, target_sr)
            return y, target_sr

        y, sr = sf.read(audio_path, dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        logger.info("Audio load done via soundfile (samples=%d, sr=%d).", y.size, sr)
        return y, sr
    except Exception:
        # Fall back to librosa's loader if soundfile can't read the file
        logger.info("Audio load fallback to librosa for %s", audio_path)
        y, sr = librosa.load(audio_path, sr=target_sr, mono=True, res_type="kaiser_fast")
        logger.info("Audio load done via librosa (samples=%d, sr=%d).", y.size, sr)
        return y, sr

# Compute log-mel spectrogram from waveform (manual STFT + mel filterbank)
def compute_spectrogram(y, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
    logger = logging.getLogger("uvicorn.error")
    step_start = time.perf_counter()

    # 1) Short Time Fourier Transform (STFT) -> magnitude^2 (power spectrogram)
    logger.info("Spectrogram: STFT start.")
    fft = _stft(y, n_fft=n_fft, hop_length=hop_length)
    power_spec = (np.abs(fft) ** 2).T
    logger.info("Spectrogram: STFT done in %.2fs.", time.perf_counter() - step_start)
    step_start = time.perf_counter()

    # 2) Build mel filterbank and apply
    logger.info("Spectrogram: mel filter start.")
    mel_filter = _get_mel_filter(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_spec = np.dot(mel_filter, power_spec)
    logger.info("Spectrogram: mel filter done in %.2fs.", time.perf_counter() - step_start)
    step_start = time.perf_counter()

    # 3) Convert to log scale (dB) and normalize
    logger.info("Spectrogram: power_to_db start.")
    S_db = librosa.power_to_db(mel_spec, ref=np.max)
    std = S_db.std() or 1.0
    S_db_norm = (S_db - S_db.mean()) / std
    logger.info("Spectrogram: power_to_db done in %.2fs.", time.perf_counter() - step_start)
    return S_db_norm


@lru_cache(maxsize=8)
def _get_mel_filter(sr, n_fft, n_mels):
    return librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)

# Short Time Fourier Transform (STFT)
def _stft(y, n_fft, hop_length):
    y = np.ascontiguousarray(y, dtype=np.float32)
    pad = n_fft // 2
    if y.size == 0:
        y = np.zeros(n_fft, dtype=np.float32)
    y = np.pad(y, (pad, pad), mode="reflect")
    if y.size < n_fft:
        y = np.pad(y, (0, n_fft - y.size), mode="constant")

    n_frames = 1 + (y.size - n_fft) // hop_length
    if n_frames <= 0:
        n_frames = 1
    stride = y.strides[0]
    frames = np.lib.stride_tricks.as_strided(
        y,
        shape=(n_frames, n_fft),
        strides=(hop_length * stride, stride),
        writeable=False,
    )
    window = np.hanning(n_fft).astype(np.float32)
    frames = frames * window[None, :]
    fft = np.fft.rfft(frames, n=n_fft, axis=1)
    return fft
