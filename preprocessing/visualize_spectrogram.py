import base64
import io
import numpy as np
import librosa
from PIL import Image

from config import HOP_LENGTH, N_MELS, SAMPLE_RATE


def spectrogram_to_base64(spec):
    # Normalize safely to [0, 1] and encode as grayscale PNG
    spec = spec.astype(np.float32)
    spec_min = float(np.min(spec))
    spec_max = float(np.max(spec))
    spec_range = spec_max - spec_min or 1.0
    normalized = (spec - spec_min) / spec_range
    normalized = np.clip(normalized, 0.0, 1.0)

    pixels = (normalized * 255).astype(np.uint8)
    image = Image.fromarray(pixels, mode="L").convert("RGB")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def build_spectrogram_metadata(
    spec,
    sample_rate=SAMPLE_RATE,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    num_ticks=5,
):
    num_mels, num_frames = spec.shape
    duration = ((num_frames - 1) * hop_length) / sample_rate if num_frames > 1 else 0.0

    time_ticks = np.linspace(0, duration, num_ticks)
    mel_freqs = librosa.mel_frequencies(
        n_mels=n_mels,
        fmin=0,
        fmax=sample_rate / 2,
    )
    mel_indices = np.linspace(0, num_mels - 1, num_ticks).astype(int)
    freq_ticks = mel_freqs[mel_indices][::-1]

    db_min = float(np.min(spec))
    db_max = float(np.max(spec))
    db_mid = (db_min + db_max) / 2

    return {
        "time_ticks": [float(x) for x in time_ticks],
        "freq_ticks": [float(x) for x in freq_ticks],
        "db_ticks": [float(db_max), float(db_mid), float(db_min)],
        "sample_rate": int(sample_rate),
        "hop_length": int(hop_length),
        "n_mels": int(n_mels),
    }