import numpy as np
import librosa
from config import (
    DURATION,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    RMS_NORMALIZE,
    RMS_TARGET,
    SAMPLE_RATE,
)

# Load audio file as waveform and resample to target sample rate and duration
def load_audio(audio_path, target_sr=SAMPLE_RATE, duration=DURATION):
    y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
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
