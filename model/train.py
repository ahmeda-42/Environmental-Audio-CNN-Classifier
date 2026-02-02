import os
import random
import sys
import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure repo root is on sys.path for local imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from model.dataset import AudioDataset, build_label_mapping, save_label_mapping
from model.cnn import AudioCNN
from preprocessing.prepare_urbansound8k import build_csv
from config import (
    BATCH_SIZE,
    CSV_PATH,
    DATASET_ROOT,
    DURATION,
    EPOCHS,
    LEARNING_RATE,
    MODEL_PATH,
    N_MELS,
    NUM_FREQ_MASKS,
    NUM_TIME_MASKS,
    SPEC_AUGMENT,
    SPEC_AUGMENT_STRENGTH,
    FREQ_MASK_PARAM,
    TIME_MASK_PARAM,
    SAMPLE_RATE,
    SEED,
    WEIGHT_DECAY,
    USE_CLASS_WEIGHTS,
    EARLY_STOPPING_PATIENCE,
    SCHEDULER_PATIENCE,
    SCHEDULER_FACTOR,
    AUG_TIME_STRETCH,
    TIME_STRETCH_RANGE,
    AUG_PITCH_SHIFT,
    PITCH_SHIFT_STEPS,
    AUG_NOISE,
    NOISE_STD,
    AUG_TIME_SHIFT,
    TIME_SHIFT_MAX_FRACTION,
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


def set_seed(seed):
    # Make results more reproducible across runs
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_epoch(model, loader, optimizer, criterion, device):
    # One full pass over the training data
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion, device):
    # One full pass over the validation data
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc="val", leave=False):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total


def main():
    set_seed(SEED)

    if not os.path.exists(CSV_PATH):
        build_csv(DATASET_ROOT, CSV_PATH)

    # Load metadata and build label mapping
    df = pd.read_csv(CSV_PATH)
    df = ensure_fold_column(df)
    label_to_index = build_label_mapping(df["label"].tolist())

    # Fixed fold split: train on folds 1-8, validate on fold 9
    train_df = df[df["fold"].isin([1, 2, 3, 4, 5, 6, 7, 8])]
    val_df = df[df["fold"] == 9]

    # Dataset objects perform audio loading + spectrogram extraction
    train_ds = AudioDataset(
        train_df,
        label_to_index,
        sample_rate=SAMPLE_RATE,
        duration=DURATION,
        n_mels=N_MELS,
        augment=SPEC_AUGMENT,
        spec_strength=SPEC_AUGMENT_STRENGTH,
        time_mask_param=TIME_MASK_PARAM,
        freq_mask_param=FREQ_MASK_PARAM,
        num_time_masks=NUM_TIME_MASKS,
        num_freq_masks=NUM_FREQ_MASKS,
        aug_time_stretch=AUG_TIME_STRETCH,
        time_stretch_range=TIME_STRETCH_RANGE,
        aug_pitch_shift=AUG_PITCH_SHIFT,
        pitch_shift_steps=PITCH_SHIFT_STEPS,
        aug_noise=AUG_NOISE,
        noise_std=NOISE_STD,
        aug_time_shift=AUG_TIME_SHIFT,
        time_shift_max_fraction=TIME_SHIFT_MAX_FRACTION,
    )
    val_ds = AudioDataset(
        val_df,
        label_to_index,
        sample_rate=SAMPLE_RATE,
        duration=DURATION,
        n_mels=N_MELS,
        augment=False,
        spec_strength=SPEC_AUGMENT_STRENGTH,
        time_mask_param=TIME_MASK_PARAM,
        freq_mask_param=FREQ_MASK_PARAM,
        num_time_masks=NUM_TIME_MASKS,
        num_freq_masks=NUM_FREQ_MASKS,
    )
    
    # Batch loaders for efficient training
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Model and device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioCNN(num_classes=len(label_to_index)).to(device)

    # Loss function (optionally class-balanced) and optimizer
    class_weights = None
    if USE_CLASS_WEIGHTS:
        counts = (
            train_df["label"]
            .map(label_to_index)
            .value_counts()
            .reindex(range(len(label_to_index)), fill_value=0)
        )
        safe_counts = counts.replace(0, 1)
        weights = counts.max() / safe_counts
        class_weights = torch.tensor(weights.values, dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", patience=SCHEDULER_PATIENCE, factor=SCHEDULER_FACTOR
    )

    # Train and keep the best checkpoint by validation accuracy
    best_acc = 0.0
    best_loss = float("inf")
    patience = 0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.3f} val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.3f} lr={current_lr:.2e}"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            # Save weights and label mapping together
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            save_label_mapping(MODEL_PATH + ".labels.json", label_to_index)

        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOPPING_PATIENCE:
                print("early stopping triggered")
                break

    # Final summary of the best validation accuracy
    print(f"best_val_acc={best_acc:.3f}")


if __name__ == "__main__":
    main()
