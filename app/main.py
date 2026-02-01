import logging
import os
import tempfile
import time
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from model.predict import (
    labels as get_labels,
    compute_spectrogram_item,
    predict as run_predict,
)
from model.load_model import MODEL_PATH, load_model
from model.dataset import load_label_mapping
from config import DURATION, N_MELS, SAMPLE_RATE
from app.websocket_handler import handle_websocket_predict
from app.schemas import (
    HealthResponse,
    LabelsResponse,
    ConfigResponse,
    PredictRequest,
    PredictResponse,
    Prediction,
    SpectrogramRequest,
    SpectrogramResponse,
    StreamConfig,
)

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Environmental Audio CNN Classifier API")
logger = logging.getLogger("uvicorn.error")

# Allow the deployed frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://environmental-audio-cnn-classifier-ce92.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def warm_start():
    try:
        label_to_index = load_label_mapping(MODEL_PATH + ".labels.json")
        load_model(num_classes=len(label_to_index))
        logger.info("Model warmup complete.")
    except Exception as exc:
        logger.warning("Model warmup skipped: %s", exc)


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}


@app.get("/labels", response_model=LabelsResponse)
def labels_endpoint():
    _, index_to_label = get_labels()
    return {"labels": [index_to_label[i] for i in range(len(index_to_label))]}


@app.get("/config", response_model=ConfigResponse)
def config_endpoint():
    return {"duration": DURATION, "n_mels": N_MELS}


@app.post("/spectrogram", response_model=SpectrogramResponse)
def spectrogram_endpoint(params: SpectrogramRequest = Depends(), file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")
    suffix = os.path.splitext(file.filename or "")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name
    try:
        file_size = os.path.getsize(tmp_path)
    except OSError:
        file_size = -1
    logger.info("Upload saved to %s (%d bytes).", tmp_path, file_size)
    try:
        _, spectrogram_response = compute_spectrogram_item(
            tmp_path,
            sample_rate=params.sample_rate,
            duration=params.duration,
            n_mels=params.n_mels,
        )
    finally:
        os.remove(tmp_path)
    return spectrogram_response


@app.post("/predict", response_model=PredictResponse)
def predict_audio(params: PredictRequest = Depends(), file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")
    start_time = time.perf_counter()
    logger.info("Predict request received for %s", file.filename)
    suffix = os.path.splitext(file.filename or "")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name
    try:
        predict_response = run_predict(
            tmp_path,
            sample_rate=params.sample_rate,
            duration=params.duration,
            n_mels=params.n_mels,
            top_k=params.top_k,
        )
    finally:
        os.remove(tmp_path)
    logger.info("Predict finished in %.2fs", time.perf_counter() - start_time)
    return predict_response


@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    await handle_websocket_predict(websocket)
