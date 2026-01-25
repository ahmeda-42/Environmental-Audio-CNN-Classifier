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

def load_model(model_path, num_classes, device):
    model = AudioCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict(audio_path, sample_rate=22050, duration=4.0, n_mels=64):
    label_to_index = load_label_mapping(MODEL_PATH + ".labels.json")
    index_to_label = {v: k for k, v in label_to_index.items()}

    y, sr = load_audio(audio_path, sample_rate, duration)
    feat = compute_spectrogram(y, sr, n_mels=n_mels)
    x = torch.tensor(feat).unsqueeze(0).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(MODEL_PATH, num_classes=len(label_to_index), device=device)
    with torch.no_grad():
        logits = model(x.to(device))
        pred = torch.argmax(logits, dim=1).item()
    print(index_to_label[pred])