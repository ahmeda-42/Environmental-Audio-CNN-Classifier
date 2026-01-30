from fastapi.testclient import TestClient

from app import main as app_main


def test_health():
    client = TestClient(app_main.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_config():
    client = TestClient(app_main.app)
    resp = client.get("/config")
    assert resp.status_code == 200
    data = resp.json()
    assert "duration" in data
    assert "n_mels" in data


def test_labels(monkeypatch):
    def fake_labels():
        return {"a": 0, "b": 1}, {0: "a", 1: "b"}

    monkeypatch.setattr(app_main, "get_labels", fake_labels)
    client = TestClient(app_main.app)
    resp = client.get("/labels")
    assert resp.status_code == 200
    assert resp.json()["labels"] == ["a", "b"]


def test_predict_endpoint(monkeypatch):
    def fake_predict(*_args, **_kwargs):
        return {
            "top_prediction": {"label": "a", "confidence": 0.9},
            "top_k": [{"label": "a", "confidence": 0.9}],
            "spectrogram": {
                "image": "",
                "features": [[0.0]],
                "shape": [1, 1],
                "time_ticks": [0.0],
                "freq_ticks": [0.0],
                "db_ticks": [0.0],
                "sample_rate": 22050,
                "hop_length": 512,
                "n_mels": 128,
            },
            "spectrograms": [
                {
                    "image": "",
                    "features": [[0.0]],
                    "shape": [1, 1],
                    "time_ticks": [0.0],
                    "freq_ticks": [0.0],
                    "db_ticks": [0.0],
                    "sample_rate": 22050,
                    "hop_length": 512,
                    "n_mels": 128,
                    "window_start": 0.0,
                    "window_end": 4.0,
                }
            ],
        }

    monkeypatch.setattr(app_main, "run_predict", fake_predict)
    client = TestClient(app_main.app)
    resp = client.post(
        "/predict?top_k=4",
        files={"file": ("test.wav", b"fake", "audio/wav")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "top_prediction" in data
    assert "spectrograms" in data


def test_spectrogram_endpoint(monkeypatch):
    def fake_spectrogram(*_args, **_kwargs):
        return None, {
            "image": "",
            "features": [[0.0]],
            "shape": [1, 1],
            "time_ticks": [0.0],
            "freq_ticks": [0.0],
            "db_ticks": [0.0],
            "sample_rate": 22050,
            "hop_length": 512,
            "n_mels": 128,
        }

    monkeypatch.setattr(app_main, "compute_spectrogram_item", fake_spectrogram)
    client = TestClient(app_main.app)
    resp = client.post(
        "/spectrogram",
        files={"file": ("test.wav", b"fake", "audio/wav")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "features" in data
    assert "shape" in data
