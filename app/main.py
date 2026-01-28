import os
import tempfile
from functools import lru_cache
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from model.predict import (
    labels as get_labels,
    compute_spectrogram_item,
    predict as run_predict,
)
from config import DURATION, N_MELS, SAMPLE_RATE
from app.websocket_handler import handle_websocket_predict
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

app = FastAPI(title="Environmental Audio CNN Classifier API")

# Allow local frontend dev servers to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}


@app.get("/labels", response_model=LabelsResponse)
def labels_endpoint():
    _, index_to_label = get_labels()
    return {"labels": [index_to_label[i] for i in range(len(index_to_label))]}


@app.post("/spectrogram", response_model=SpectrogramResponse)
def spectrogram_endpoint(params: SpectrogramRequest = Depends(), upload: UploadFile = File(...)):
    if not upload.filename:
        raise HTTPException(status_code=400, detail="No file provided.")
    suffix = os.path.splitext(upload.filename or "")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(upload.file.read())
        tmp_path = tmp.name
    try:
        _, spectrogram_response = compute_spectrogram_item(
            tmp_path, params.sample_rate, params.duration, params.n_mels
        )
    finally:
        os.remove(tmp_path)
    return spectrogram_response


@app.post("/predict", response_model=PredictResponse)
def predict_audio(params: PredictRequest = Depends(), upload: UploadFile = File(...)):
    if not upload.filename:
        raise HTTPException(status_code=400, detail="No file provided.")
    suffix = os.path.splitext(upload.filename or "")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(upload.file.read())
        tmp_path = tmp.name
    try:
        predict_response = run_predict(tmp_path, params.sample_rate, params.duration, params.n_mels, params.top_k)
    finally:
        os.remove(tmp_path)
    return predict_response


@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    await handle_websocket_predict(websocket)
