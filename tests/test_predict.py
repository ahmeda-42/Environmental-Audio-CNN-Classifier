import numpy as np
import torch

from model import predict as predict_module


class DummyModel(torch.nn.Module):
    def forward(self, x):
        return torch.tensor([[1.0, 2.0]], device=x.device)


def _stub_compute_spectrogram_item(
    sample_rate,
    duration,
    hop_length,
    n_mels,
    y=None,
    sr=None,
):
    features = np.zeros((n_mels, 5), dtype=np.float32)
    item = {
        "image": "",
        "features": features.tolist(),
        "shape": list(features.shape),
        "time_ticks": [],
        "freq_ticks": [],
        "db_ticks": [],
        "sample_rate": int(sample_rate),
        "hop_length": int(hop_length),
        "n_mels": int(n_mels),
    }
    return features, item


def test_predict_windowing_multiple_windows(monkeypatch):
    def fake_load(path, sr, mono):
        return np.zeros(100, dtype=np.float32), sr

    monkeypatch.setattr(predict_module.librosa, "load", fake_load)
    monkeypatch.setattr(predict_module, "compute_spectrogram_item", _stub_compute_spectrogram_item)
    monkeypatch.setattr(
        predict_module,
        "labels",
        lambda: ({"a": 0, "b": 1}, {0: "a", 1: "b"}),
    )
    monkeypatch.setattr(
        predict_module,
        "load_model",
        lambda num_classes: (DummyModel(), torch.device("cpu")),
    )

    result = predict_module.predict(
        "dummy.wav",
        sample_rate=10,
        duration=4.0,
        n_mels=8,
        hop_length=2,
        top_k=2,
    )

    assert result["top_prediction"]["label"] in {"a", "b"}
    assert len(result["spectrograms"]) == 4
    assert result["spectrograms"][-1]["window_end"] == 10.0


def test_predict_short_clip(monkeypatch):
    def fake_load(path, sr, mono):
        return np.zeros(10, dtype=np.float32), sr

    monkeypatch.setattr(predict_module.librosa, "load", fake_load)
    monkeypatch.setattr(predict_module, "compute_spectrogram_item", _stub_compute_spectrogram_item)
    monkeypatch.setattr(
        predict_module,
        "labels",
        lambda: ({"a": 0, "b": 1}, {0: "a", 1: "b"}),
    )
    monkeypatch.setattr(
        predict_module,
        "load_model",
        lambda num_classes: (DummyModel(), torch.device("cpu")),
    )

    result = predict_module.predict(
        "dummy.wav",
        sample_rate=10,
        duration=4.0,
        n_mels=8,
        hop_length=2,
        top_k=2,
    )

    assert len(result["spectrograms"]) == 1
