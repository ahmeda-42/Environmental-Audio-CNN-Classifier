import torch
import torch.nn.functional as F
import numpy as np
from fastapi import HTTPException
from app.main import spectrogram_to_png_base64
from cnn import AudioCNN
from dataset import load_label_mapping
from preprocessing.audio_features import load_audio_from_upload, compute_spectrogram

MODEL_PATH = "artifacts/cnn.pt"

def load_model(model_path, num_classes, device):
    # Build the model and load trained weights
    model = AudioCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict(file, sample_rate=22050, duration=4.0, n_mels=64, top_k=3):
    # Check if a file was provided
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    # Load label mapping to translate indices -> class names
    label_to_index = load_label_mapping(MODEL_PATH + ".labels.json")
    index_to_label = {v: k for k, v in label_to_index.items()}

    # Convert raw audio to a log-mel spectrogram tensor
    y, sr = load_audio_from_upload(file, sample_rate, duration)
    spectogram = compute_spectrogram(y, sr, n_mels=n_mels)
    x = torch.tensor(spectogram).unsqueeze(0).unsqueeze(0)

    # Run the model on a single example
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(MODEL_PATH, num_classes=len(label_to_index), device=device)
    with torch.no_grad():
        logits = model(x.to(device))
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    spec_payload = {
        "image": spectrogram_to_png_base64(spectogram),
        "features": spectogram.tolist(),
        "shape": list(spectogram.shape),
    }

    top_k = max(1, min(top_k, len(probs)))
    top_indices = np.argsort(probs)[-top_k:][::-1]
    top_predictions = [
        {"label": index_to_label[i], "confidence": float(probs[i])}
        for i in top_indices
    ]

    return {
        "top_prediction": top_predictions[0],
        "top_k": top_predictions,
        "spectrogram": spec_payload,
    }