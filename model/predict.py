import logging
import os
import sys
import time
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
from config import DURATION, HOP_LENGTH, N_MELS, RMS_NORMALIZE, RMS_TARGET, SAMPLE_RATE
from model.dataset import load_label_mapping
from preprocessing.audio_features import load_audio, load_audio_full, compute_spectrogram
from preprocessing.visualize_spectrogram import build_spectrogram_metadata, spectrogram_to_base64

logger = logging.getLogger("uvicorn.error")


def labels():
    # Load label mapping to translate indices -> class names
    label_to_index = load_label_mapping(MODEL_PATH + ".labels.json")
    index_to_label = {v: k for k, v in label_to_index.items()}
    return label_to_index, index_to_label


def _rms_normalize(y):
    if not RMS_NORMALIZE:
        return y
    rms = np.sqrt(np.mean(y**2)) if y.size else 0.0
    if rms > 0:
        return y * (RMS_TARGET / rms)
    return y


def _pad_to_length(y, target_len):
    if len(y) < target_len:
        return np.pad(y, (0, target_len - len(y)), mode="constant")
    return y[:target_len]


def compute_spectrogram_item(
    audio_path=None,
    sample_rate=SAMPLE_RATE,
    duration=DURATION,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    y=None,
    sr=None,
    include_image=True,
):
    # Load or use provided audio and compute spectrogram
    if y is None:
        y, sr = load_audio(audio_path, sample_rate, duration)
    else:
        sr = sr or sample_rate
        target_len = int(sample_rate * duration)
        y = _pad_to_length(y, target_len)
        y = _rms_normalize(y)
    features = compute_spectrogram(y, sr, n_mels=n_mels, hop_length=hop_length)
    metadata = build_spectrogram_metadata(
        features,
        sample_rate=sample_rate,
        hop_length=hop_length,
        n_mels=n_mels,
    )

    # Convert spectrogram to PNG base64 image
    spectrogram_item = {
        "image": spectrogram_to_base64(features) if include_image else "",
        "features": features.tolist(),
        "shape": list(features.shape),
        **metadata,
    }
    return features, spectrogram_item


def predict(
    audio_path,
    sample_rate=SAMPLE_RATE,
    duration=DURATION,
    n_mels=N_MELS,
    hop_length=HOP_LENGTH,
    top_k=3,
):
    start_time = time.perf_counter()
    logger.info("Predict pipeline start: %s", audio_path)
    # Load label mapping to translate indices -> class names
    label_to_index, index_to_label = labels()
    logger.info("Labels loaded (%d classes).", len(label_to_index))

    # Load full audio for windowed prediction
    y, sr = load_audio_full(audio_path, target_sr=sample_rate)
    logger.info("Audio loaded (samples=%d, sr=%d).", y.size, sr)
    target_len = int(sample_rate * duration)
    if y.size == 0:
        y = np.zeros(target_len, dtype=np.float32)

    # Split into fixed windows with 50% overlap, padding the last one
    step = max(1, target_len // 2)
    max_start = max(0, len(y) - target_len)
    starts = list(range(0, max_start + 1, step))
    windows = [
        _pad_to_length(y[start : start + target_len], target_len)
        for start in starts
    ]
    if not windows:
        windows = [np.zeros(target_len, dtype=np.float32)]
        starts = [0]
    logger.info("Windows prepared (count=%d).", len(windows))

    # Load the model once and run inference per window
    model, device = load_model(num_classes=len(label_to_index))
    logger.info("Model loaded on %s.", device)
    summed_probs = None
    spectrogram_item = None
    spectrograms = []
    with torch.no_grad():
        for idx, (start, window) in enumerate(zip(starts, windows)):
            window_start = time.perf_counter()
            logger.info("Window %d: spectrogram start.", idx)
            features, window_item = compute_spectrogram_item(
                sample_rate=sample_rate,
                duration=duration,
                hop_length=hop_length,
                n_mels=n_mels,
                y=window,
                sr=sr,
            )
            logger.info("Window %d: spectrogram done in %.2fs.", idx, time.perf_counter() - window_start)
            window_item = {
                **window_item,
                "window_start": float(start / sample_rate),
                "window_end": float(
                    min(start + target_len, len(y)) / sample_rate
                ),
            }
            spectrograms.append(window_item)
            if idx == 0:
                spectrogram_item = window_item

            infer_start = time.perf_counter()
            x = torch.tensor(features).unsqueeze(0).unsqueeze(0)
            logits = model(x.to(device))
            logger.info("Window %d: model forward done in %.2fs.", idx, time.perf_counter() - infer_start)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            if summed_probs is None:
                summed_probs = probs
            else:
                summed_probs += probs

    probs = summed_probs / max(1, len(windows))

    # Get the top k predictions
    top_k = max(1, min(top_k, len(probs)))
    top_indices = np.argsort(probs)[-top_k:][::-1]
    top_predictions = [
        {"label": index_to_label[i], "confidence": float(probs[i])}
        for i in top_indices
    ]

    logger.info("Predict pipeline complete in %.2fs.", time.perf_counter() - start_time)
    return {
        "top_prediction": top_predictions[0],
        "top_k": top_predictions,
        "spectrogram": spectrogram_item,
        "spectrograms": spectrograms,
    }