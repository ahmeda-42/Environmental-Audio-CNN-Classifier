import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure repo root is on sys.path for local imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from model.dataset import AudioDataset, build_label_mapping, load_label_mapping
from model.load_model import MODEL_PATH, load_model
from config import (
    BATCH_SIZE,
    CSV_PATH,
    DURATION,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
)


def ensure_fold_column(df):
    # Use explicit fold column if present; otherwise infer from file path
    if "fold" in df.columns:
        return df
    extracted = df["file_path"].str.extract(r"[/\\\\]fold(\d+)[/\\\\]", expand=False)
    if extracted.isnull().any():
        raise ValueError(
            "Missing 'fold' column and could not infer fold from file_path. "
            "Re-run scripts/prepare_urbansound8k.py to regenerate the CSV."
        )
    df = df.copy()
    df["fold"] = extracted.astype(int)
    return df


def evaluate(model, loader, device):
    # Run inference on all batches and collect predictions/labels
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="eval", leave=False):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(y.cpu().numpy().tolist())
    acc = correct / total if total else 0.0
    return acc, np.array(all_labels), np.array(all_preds)


def main():
    # Load metadata and label mapping
    df = pd.read_csv(CSV_PATH)
    df = ensure_fold_column(df)
    labels_path = MODEL_PATH + ".labels.json"
    try:
        label_to_index = load_label_mapping(labels_path)
    except FileNotFoundError:
        label_to_index = build_label_mapping(df["label"].tolist())

    # Fixed test split: fold 10 only
    test_df = df[df["fold"] == 10]

    # Dataset handles audio loading + spectrogram extraction
    test_ds = AudioDataset(
        test_df,
        label_to_index,
        sample_rate=SAMPLE_RATE,
        duration=DURATION,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        augment=False,
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Load model weights for evaluation
    model, device = load_model(num_classes=len(label_to_index))

    # Overall accuracy and per-sample predictions
    acc, y_true, y_pred = evaluate(model, test_loader, device)
    print(f"\ntest_acc = {acc:.3f}")

    # Confusion matrix and class-wise accuracy
    labels_sorted = [label for label, _ in sorted(label_to_index.items(), key=lambda x: x[1])]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels_sorted))))
    denom = cm.sum(axis=1)
    per_class_acc = np.divide(
        cm.diagonal(),
        denom,
        out=np.zeros_like(denom, dtype=float),
        where=denom != 0,
    )

    print("\nconfusion_matrix =")
    print(cm)
    print("\nclass_wise_accuracy =")
    for label, value in zip(labels_sorted, per_class_acc):
        print(f"{label}: {value:.3f}")


if __name__ == "__main__":
    main()
