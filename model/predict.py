import os
import sys

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dataset import load_label_mapping
from preprocessing.audio_features import load_audio, compute_spectrogram
from cnn import AudioCNN

MODEL_PATH = "artifacts/cnn.pt"
AUDIO_PATH = "data/UrbanSound8K/audio/fold1/21684-9-0-39.wav"
SAMPLE_RATE = 22050
DURATION = 4.0
N_MELS = 64


def load_model(model_path, num_classes, device):
    model = AudioCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_logits(model, x, device):
    with torch.no_grad():
        return model(x.to(device))


def predict_label(model, x, device):
    logits = predict_logits(model, x, device)
    return torch.argmax(logits, dim=1).item()


def main():
    label_to_index = load_label_mapping(MODEL_PATH + ".labels.json")
    index_to_label = {v: k for k, v in label_to_index.items()}

    y, sr = load_audio(AUDIO_PATH, SAMPLE_RATE, DURATION)
    feat = compute_spectrogram(y, sr, n_mels=N_MELS)
    x = torch.tensor(feat).unsqueeze(0).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(MODEL_PATH, num_classes=len(label_to_index), device=device)
    pred = predict_label(model, x, device)
    print(index_to_label[pred])


if __name__ == "__main__":
    main()
