import torch

from model.cnn import AudioCNN
from model import load_model as load_model_module


def test_model_loads(tmp_path, monkeypatch):
    model_path = tmp_path / "cnn.pt"
    model = AudioCNN(num_classes=10)
    torch.save(model.state_dict(), model_path)

    monkeypatch.setattr(load_model_module, "MODEL_PATH", str(model_path))
    loaded_model, device = load_model_module.load_model(num_classes=10)

    assert loaded_model is not None
    assert device is not None
    assert not loaded_model.training
