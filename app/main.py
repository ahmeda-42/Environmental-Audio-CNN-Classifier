import base64
import io
import json
import os
import sys
import tempfile
from functools import lru_cache

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware

# Make local modules importable when running from the repo root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT_DIR, "model")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from cnn import AudioCNN
from dataset import load_label_mapping
from preprocessing.audio_features import load_audio, compute_spectrogram
from app.schemas import (
    HealthResponse,
    LabelsResponse,
    PredictRequest,
    PredictResponse,
    Prediction,
    SpectrogramRequest,
    SpectrogramResponse,
    StreamConfig,
)

app = FastAPI(title="Environmental Audio CNN API")

# Allow local frontend dev servers to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inference defaults (keep in sync with training)
MODEL_PATH = "artifacts/cnn.pt"
LABELS_PATH = MODEL_PATH + ".labels.json"
SAMPLE_RATE = 22050
DURATION = 4.0
N_MELS = 64


def _load_audio_from_upload(upload: UploadFile, target_sr, duration):
    # Save to a temp file so librosa can read it reliably
    suffix = os.path.splitext(upload.filename or "")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(upload.file.read())
        tmp_path = tmp.name
    try:
        y, sr = load_audio(tmp_path, target_sr=target_sr, duration=duration)
    finally:
        os.remove(tmp_path)
    return y, sr


@lru_cache(maxsize=1)
def get_label_mapping():
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(f"labels not found: {LABELS_PATH}")
    return load_label_mapping(LABELS_PATH)


@lru_cache(maxsize=1)
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"model not found: {MODEL_PATH}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioCNN(num_classes=len(get_label_mapping()))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model, device


def predict_from_waveform(y, sr, n_mels):
    # Convert waveform -> spectrogram -> logits -> probabilities
    feat = compute_spectrogram(y, sr, n_mels=n_mels)
    x = torch.tensor(feat).unsqueeze(0).unsqueeze(0)
    model, device = load_model()
    with torch.no_grad():
        logits = model(x.to(device))
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    return probs, feat


def spectrogram_to_png_base64(spec):
    spec_min = float(np.min(spec))
    spec_max = float(np.max(spec))
    spec_range = spec_max - spec_min or 1.0
    normalized = (spec - spec_min) / spec_range
    pixels = (normalized * 255).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(pixels, mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}


@app.get("/labels", response_model=LabelsResponse)
def labels():
    label_to_index = get_label_mapping()
    index_to_label = {v: k for k, v in label_to_index.items()}
    return {"labels": [index_to_label[i] for i in range(len(index_to_label))]}


@app.post("/predict", response_model=PredictResponse)
def predict_audio(
    params: PredictRequest = Depends(),
    file: UploadFile = File(...),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    y, sr = _load_audio_from_upload(
        file,
        target_sr=params.sample_rate,
        duration=params.duration,
    )
    probs, spec = predict_from_waveform(y, sr, params.n_mels)
    spec_payload = {
        "image": spectrogram_to_png_base64(spec),
        "features": spec.tolist(),
        "shape": list(spec.shape),
    }

    label_to_index = get_label_mapping()
    index_to_label = {v: k for k, v in label_to_index.items()}

    top_k = max(1, min(params.top_k, len(probs)))
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


@app.post("/spectrogram", response_model=SpectrogramResponse)
def spectrogram(
    params: SpectrogramRequest = Depends(),
    file: UploadFile = File(...),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    y, sr = _load_audio_from_upload(
        file,
        target_sr=params.sample_rate,
        duration=params.duration,
    )
    _, spec = predict_from_waveform(y, sr, params.n_mels)
    return {
        "image": spectrogram_to_png_base64(spec),
        "features": spec.tolist(),
        "shape": list(spec.shape),
    }


@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    await websocket.accept()

    # First message should be JSON config for the stream
    config_raw = await websocket.receive_text()
    try:
        config = StreamConfig.model_validate_json(config_raw)
        sample_rate = config.sample_rate
        duration = config.duration
        n_mels = config.n_mels
    except Exception as exc:
        await websocket.send_json({"error": f"invalid config: {exc}"})
        await websocket.close()
        return

    target_len = int(sample_rate * duration)
    buffer = np.zeros(0, dtype=np.float32)

    # Client sends raw float32 mono PCM in binary messages
    while True:
        message = await websocket.receive()
        if "bytes" in message:
            chunk = np.frombuffer(message["bytes"], dtype=np.float32)
            if chunk.size == 0:
                continue
            buffer = np.concatenate([buffer, chunk])

            # Keep only the most recent window
            if buffer.size > target_len * 2:
                buffer = buffer[-target_len * 2 :]

            if buffer.size >= target_len:
                window = buffer[-target_len:]
                probs, _ = predict_from_waveform(window, sample_rate, n_mels)
                label_to_index = get_label_mapping()
                index_to_label = {v: k for k, v in label_to_index.items()}
                pred_idx = int(np.argmax(probs))
                await websocket.send_json(
                    {
                        "label": index_to_label[pred_idx],
                        "confidence": float(probs[pred_idx]),
                    }
                )
        elif "text" in message:
            # Allow client to reset the buffer
            if message["text"].strip().lower() == "reset":
                buffer = np.zeros(0, dtype=np.float32)
