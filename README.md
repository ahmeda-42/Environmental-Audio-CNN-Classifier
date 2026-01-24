# Environmental Audio CNN Classifier

End-to-end ML project that classifies environmental sounds (e.g., dog bark, siren, drilling) using signal processing features and a convolutional neural network (CNN).

## Highlights
- Signal processing pipeline: audio loading, resampling, normalization, and log-mel spectrogram extraction
- CNN classifier trained on spectrograms
- Reproducible training with configurable hyperparameters
- Scripts to prepare the UrbanSound8K dataset

## Project Structure
- `src/` - core training, model, and feature code
- `scripts/` - dataset preparation utilities
- `data/` - place datasets here (not committed)
- `artifacts/` - saved checkpoints
- `notebooks/` - exploration notebooks

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset (UrbanSound8K)
1. Download UrbanSound8K and extract to `data/UrbanSound8K`.
2. Create a CSV for training:
```bash
python scripts/prepare_urbansound8k.py \
  --dataset-root data/UrbanSound8K \
  --output-csv data/urbansound8k.csv
```

## Train
```bash
python src/train.py \
  --csv-path data/urbansound8k.csv \
  --model-out artifacts/cnn.pt \
  --epochs 20
```

## Evaluate
```bash
python src/evaluate.py \
  --csv-path data/urbansound8k.csv \
  --model-path artifacts/cnn.pt
```

## Predict a Single File
```bash
python src/predict.py \
  --audio-path path/to/example.wav \
  --model-path artifacts/cnn.pt
```
