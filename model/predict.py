import os
import sys
import torch
import torch.nn.functional as F
import numpy as np

# Make local modules importable when running from the repo root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT_DIR, "model")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from model.load_model import MODEL_PATH, load_model
from config import DURATION, N_MELS, SAMPLE_RATE
from model.dataset import load_label_mapping
from preprocessing.audio_features import load_audio, compute_spectrogram
from preprocessing.visualize_spectrogram import spectrogram_to_base64


def labels():
    # Load label mapping to translate indices -> class names
    label_to_index = load_label_mapping(MODEL_PATH + ".labels.json")
    index_to_label = {v: k for k, v in label_to_index.items()}
    return label_to_index, index_to_label


def compute_spectrogram_item(audio_path, sample_rate=SAMPLE_RATE, duration=DURATION, n_mels=N_MELS):
    # Load audio from audio path and compute spectrogram
    y, sr = load_audio(audio_path, sample_rate, duration)
    features = compute_spectrogram(y, sr, n_mels=n_mels)

    # Convert spectrogram to PNG base64 image
    spectrogram_item = {
        "image": spectrogram_to_base64(features),
        "features": features.tolist(),
        "shape": list(features.shape),
    }
    return features, spectrogram_item


def predict(audio_path, sample_rate=SAMPLE_RATE, duration=DURATION, n_mels=N_MELS, top_k=3):
    # Load label mapping to translate indices -> class names
    label_to_index, index_to_label = labels()

    # Convert raw audio to a spectrogram tensor
    features, spectrogram_item = compute_spectrogram_item(
        audio_path, sample_rate, duration, n_mels
    )
    x = torch.tensor(features).unsqueeze(0).unsqueeze(0)

    # Load the model and run inference
    model, device = load_model(num_classes=len(label_to_index))
    with torch.no_grad():
        logits = model(x.to(device))
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    
    # Get the top k predictions
    top_k = max(1, min(top_k, len(probs)))
    top_indices = np.argsort(probs)[-top_k:][::-1]
    top_predictions = [
        {"label": index_to_label[i], "confidence": float(probs[i])}
        for i in top_indices
    ]

    return {
        "top_prediction": top_predictions[0],
        "top_k": top_predictions,
        "spectrogram": spectrogram_item,
    }