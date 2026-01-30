import numpy as np
import torch
from fastapi.testclient import TestClient

from app import main as app_main
from app import websocket_handler as ws_handler


class DummyModel(torch.nn.Module):
    def forward(self, x):
        return torch.tensor([[1.0, 2.0, 3.0]], device=x.device)


def _stub_compute_spectrogram_item(
    sample_rate,
    duration,
    n_mels,
    y=None,
    sr=None,
):
    features = np.zeros((n_mels, 4), dtype=np.float32)
    item = {
        "image": "",
        "features": features.tolist(),
        "shape": list(features.shape),
        "time_ticks": [0.0, duration],
        "freq_ticks": [0.0],
        "db_ticks": [0.0],
        "sample_rate": int(sample_rate),
        "hop_length": 1,
        "n_mels": int(n_mels),
    }
    return features, item


def test_websocket_predict(monkeypatch):
    monkeypatch.setattr(
        ws_handler,
        "load_label_mapping",
        lambda _path: {"a": 0, "b": 1, "c": 2},
    )
    monkeypatch.setattr(
        ws_handler,
        "load_model",
        lambda num_classes: (DummyModel(), torch.device("cpu")),
    )
    monkeypatch.setattr(
        ws_handler,
        "compute_spectrogram_item",
        _stub_compute_spectrogram_item,
    )

    client = TestClient(app_main.app)
    with client.websocket_connect("/ws/predict") as ws:
        ws.send_json({"sample_rate": 8000, "duration": 1.0, "n_mels": 16})
        ws.send_bytes(np.zeros(8000, dtype=np.float32).tobytes())
        data = ws.receive_json()

    assert "top_prediction" in data
    assert len(data["top_k"]) == 3
    assert data["spectrograms"][0]["window_end"] == 1.0
