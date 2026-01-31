# Environmental Audio CNN Classifier

Full-stack ML project that detects and classifies environmental sounds using a CNN on log-mel spectrograms. Includes a FastAPI backend, a React frontend, real-time WebSocket streaming, and Docker support.

## What This Project Does
- Converts audio to log-mel spectrograms with optional RMS normalization.
- Trains a 2D CNN on spectrograms with SpecAugment.
- Supports long-audio inference via 50% overlap windowing + averaged predictions.
- Provides a FastAPI API for predictions, spectrograms, and real-time streaming.
- Ships a React UI for upload + microphone streaming with live spectrograms.

## Project Structure
```
app/                      # FastAPI app + WebSocket handler
model/                    # CNN, training, evaluation, prediction
preprocessing/            # Audio feature + spectrogram utilities
frontend/                 # React UI (Vite)
tests/                    # Pytest suite
config.py                 # Centralized settings
artifacts/                # Saved model + label mapping (not committed)
data/                     # Dataset + CSV (not committed)
```

## Setup (Python)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset (UrbanSound8K)
1. Download and extract UrbanSound8K to `data/UrbanSound8K`.
2. Generate the CSV (optional; `train.py` will auto-create if missing):
```bash
python -c "from preprocessing.prepare_urbansound8k import build_csv; build_csv('data/UrbanSound8K', 'data/urbansound8k.csv')"
```

## Train
```bash
python model/train.py
```
Trains with folds 1–8, validates on fold 9, and saves:
- `artifacts/cnn.pt`
- `artifacts/cnn.pt.labels.json`

## Evaluate
```bash
python model/evaluate.py
```
Evaluates on fold 10 and prints accuracy + confusion matrix.

## Predict (Python)
```bash
python model/try_predict.py
```
Edits to `model/try_predict.py` are the fastest way to test local files.

## FastAPI Backend
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

## React Frontend
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
```bash
pytest
```

## Key Configuration
Edit `config.py` to change:
- Audio: `SAMPLE_RATE`, `DURATION`, `N_MELS`, `N_FFT`, `HOP_LENGTH`
- Normalization: `RMS_NORMALIZE`, `RMS_TARGET`
- Training: `BATCH_SIZE`, `LEARNING_RATE`, `EPOCHS`, `SEED`
- Augmentation: `SPEC_AUGMENT`, `TIME_MASK_PARAM`, `FREQ_MASK_PARAM`, `NUM_TIME_MASKS`, `NUM_FREQ_MASKS`
