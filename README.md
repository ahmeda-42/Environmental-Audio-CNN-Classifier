# Environmental Audio CNN Classifier

A full-stack ML project that detects and classifies environmental sounds using deep learning. The system extracts log-mel spectrogram features from audio through signal processing and Fourier Transforms, runs a convolutional neural network (CNN) for inference, and provides a React-based web interface for file upload and real-time visualization. Includes a FastAPI backend, real-time WebSocket streaming, and Docker support.

## What This Project Does
- Converts audio to log-mel spectrograms with optional RMS normalization.
- Trains a 2D CNN on spectrograms with SpecAugment.
- Supports long-audio inference via 50% overlap windowing + averaged predictions.
- Provides a FastAPI API for predictions, spectrograms, and real-time streaming.
- Ships a React UI for upload + microphone streaming with live spectrograms.

## Project Structure
```bash
Environmental-Audio-CNN-Classifier/
├── app/                         # FastAPI app + WebSocket handler
│   ├── main.py                  # API entry point + routes
│   ├── schemas.py               # Pydantic request/response models
│   └── websocket_handler.py     # Real-time streaming logic
├── model/                       # CNN, training, evaluation, prediction
│   ├── cnn.py                   # AudioCNN architecture
│   ├── dataset.py               # Dataset + SpecAugment
│   ├── train.py                 # Training script
│   ├── evaluate.py              # Evaluation script
│   ├── predict.py               # Inference + windowing
│   ├── load_model.py            # Model loading helper
│   └── try_predict.py           # Example prediction script
├── preprocessing/               # Audio features + utilities
│   ├── audio_features.py        # Load audio + compute spectrogram
│   ├── prepare_urbansound8k.py  # Build CSV from dataset
│   └── visualize_spectrogram.py # Spectrogram encoding + metadata
├── frontend/                    # React UI (Vite)
│   ├── src/
│   ├── Dockerfile
│   └── nginx.conf
├── tests/                       # Pytest suite
│   ├── test_api.py
│   ├── test_model.py
│   ├── test_predict.py
│   ├── test_preprocessing.py
│   └── test_websocket.py
├── artifacts/                   # Saved model + labels (not committed)
├── data/                        # Dataset + CSV (not committed)
├── config.py                    # Centralized settings
├── Dockerfile                   # Backend container
├── docker-compose.yml           # Run API + frontend
├── requirements.txt
└── README.md
```

## Try It Yourself!!

### 1. Prepare the dataset (UrbanSound8K)
Download and extract UrbanSound8K to `data/` from https://urbansounddataset.weebly.com/download-urbansound8k.html

### 2. Setup virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Train the model
```bash
python model/train.py
```
Trains with folds 1–8, validates on fold 9, and saves:
- `artifacts/cnn.pt`
- `artifacts/cnn.pt.labels.json`

### 4. Evaluate the model
```bash
python model/evaluate.py
```
Evaluates on fold 10 and prints the classification report and confusion matrix.

### 5. Run the FastAPI backend
```bash
uvicorn app.main:app --reload
```
Base URL: `http://127.0.0.1:8000`

Endpoints:
- `GET /health`
- `GET /labels`
- `GET /config`
- `POST /predict` (multipart file upload)
- `POST /spectrogram` (multipart file upload)
- `WS /ws/predict` (float32 PCM streaming)

### 6. Run the React frontend
```bash
cd frontend
npm install
npm run dev
```
Default API base: `http://127.0.0.1:8000`  
Override with: `VITE_API_BASE`

## Docker (API + Frontend)
```bash
docker compose up --build
```
- Frontend: `http://localhost:5173`
- API: `http://localhost:8000`

## Testing
Run the full test suite:
```bash
pytest
```
Tests cover:
- REST endpoints and responses
- Model loading
- Prediction windowing logic
- Preprocessing + spectrogram metadata
- WebSocket streaming

## Key Configuration
Edit `config.py` to change:
- Audio: `SAMPLE_RATE`, `DURATION`, `N_MELS`, `N_FFT`, `HOP_LENGTH`
- Normalization: `RMS_NORMALIZE`, `RMS_TARGET`
- Training: `BATCH_SIZE`, `LEARNING_RATE`, `EPOCHS`, `SEED`
- Augmentation: `SPEC_AUGMENT`, `TIME_MASK_PARAM`, `FREQ_MASK_PARAM`, `NUM_TIME_MASKS`, `NUM_FREQ_MASKS`
