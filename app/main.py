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

from app.predict import labels, spectogram, predict
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


SAMPLE_RATE = 22050
DURATION = 4.0
N_MELS = 64


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}


@app.get("/labels", response_model=LabelsResponse)
def labels():
    _, index_to_label = labels()
    return {"labels": [index_to_label[i] for i in range(len(index_to_label))]}


@app.post("/spectrogram", response_model=SpectrogramResponse)
def spectrogram(params: SpectrogramRequest = Depends(), file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")
    _, spectogram_response = spectogram(file, params.sample_rate, params.duration, params.n_mels)
    return spectogram_response


@app.post("/predict", response_model=PredictResponse)
def predict_audio(params: PredictRequest = Depends(), file: UploadFile = File(...),):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")
    predict_response = predict(file, params.sample_rate, params.duration, params.n_mels, params.top_k)
    return predict_response


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
                feat = compute_spectrogram(window, sample_rate, n_mels=n_mels)
                x = torch.tensor(feat).unsqueeze(0).unsqueeze(0)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = load_model(MODEL_PATH, num_classes=idk, device=device)
                with torch.no_grad():
                    logits = model(x.to(device))
                    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                label_to_index = load_label_mapping(MODEL_PATH + ".labels.json")
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
