import numpy as np

from preprocessing.audio_features import compute_spectrogram
from preprocessing.visualize_spectrogram import (
    build_spectrogram_metadata,
    spectrogram_to_base64,
)


def test_compute_spectrogram_shape():
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440 * t)

    spec = compute_spectrogram(y, sr=sr, n_mels=32)
    assert spec.shape[0] == 32
    assert np.isfinite(spec).all()


def test_spectrogram_to_base64():
    spec = np.random.rand(16, 10).astype(np.float32)
    encoded = spectrogram_to_base64(spec)
    assert isinstance(encoded, str)
    assert len(encoded) > 0


def test_build_spectrogram_metadata_ticks():
    spec = np.random.randn(32, 11).astype(np.float32)
    metadata = build_spectrogram_metadata(
        spec,
        sample_rate=10,
        hop_length=2,
        n_mels=32,
        num_ticks=5,
    )

    time_ticks = metadata["time_ticks"]
    freq_ticks = metadata["freq_ticks"]
    db_ticks = metadata["db_ticks"]

    assert len(time_ticks) == 5
    assert time_ticks[0] == 0.0
    assert time_ticks[-1] == 2.0
    assert len(freq_ticks) == 5
    assert all(0.0 <= f <= 5.0 for f in freq_ticks)
    assert freq_ticks == sorted(freq_ticks, reverse=True)
    assert len(db_ticks) == 3
    assert db_ticks[0] >= db_ticks[1] >= db_ticks[2]
