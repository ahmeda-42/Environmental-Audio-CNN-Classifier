import logging
import os
import wave
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
        ext = os.path.splitext(audio_path)[1].lower()
        if ext == ".wav":
            y, sr = _read_wav_pcm(audio_path)
        else:
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
        y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        logger.info("Audio load done via librosa (samples=%d, sr=%d).", y.size, sr)
        return y, sr


def _read_wav_pcm(path):
    with wave.open(path, "rb") as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sr = wf.getframerate()
        num_frames = wf.getnframes()
        data = wf.readframes(num_frames)

    if sample_width == 1:
        y = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
        y = (y - 128.0) / 128.0
    elif sample_width == 2:
        y = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        raw = np.frombuffer(data, dtype=np.uint8)
        raw = raw.reshape(-1, 3)
        y = (
            raw[:, 0].astype(np.int32)
            | (raw[:, 1].astype(np.int32) << 8)
            | (raw[:, 2].astype(np.int32) << 16)
        )
        y = (y.astype(np.float32) / 8388608.0)
    elif sample_width == 4:
        y = np.frombuffer(data, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    if num_channels > 1:
        y = y.reshape(-1, num_channels)
        y = y.mean(axis=1)
    return y, sr

# Compute log-mel spectrogram from waveform (manual STFT + mel filterbank)
def compute_spectrogram(y, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
    # 1) Short Time Fourier Transform (STFT) -> magnitude^2 (power spectrogram)
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    power_spec = np.abs(D) ** 2

    # 2) Build mel filterbank and apply
    mel_filter = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_spec = np.dot(mel_filter, power_spec)

    # 3) Convert to log scale (dB) and normalize
    S_db = librosa.power_to_db(mel_spec, ref=np.max)
    std = S_db.std() or 1.0
    S_db_norm = (S_db - S_db.mean()) / std
    return S_db_norm
