import json
import os
import torch
from torch.utils.data import Dataset
from preprocessing.audio_features import load_audio, compute_spectrogram
import pandas as pd
import numpy as np

def build_label_mapping(labels):
    # Map string labels to stable integer indices
    unique = sorted(set(labels))
    return {label: idx for idx, label in enumerate(unique)}


class AudioDataset(Dataset):
    def __init__(self, dataframe, label_to_index, sample_rate=22050, duration=4.0, n_mels=64, n_fft=1024, hop_length=512):
        # Store metadata and feature parameters
        self.df = dataframe.reset_index(drop=True)
        self.label_to_index = label_to_index
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __len__(self):
        # Number of audio samples in the split
        return len(self.df)

    def __getitem__(self, idx):
        # Load one file, compute spectrogram, and return (tensor, label_id)
        row = self.df.iloc[idx]
        y, sr = load_audio(
            row["file_path"],
            target_sr=self.sample_rate,
            duration=self.duration,
        )
        feat = compute_spectrogram(
            y,
            sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        
        # Add a channel dimension for CNN input: (1, F, T)
        x = np.expand_dims(feat, axis=0)
        x = torch.tensor(x, dtype=torch.float32)

        # Convert string label to class index
        y_label = self.label_to_index[row["label"]]
        return x, y_label


def save_label_mapping(path, label_to_index):
    # Persist label mapping alongside the model weights
    with open(path, "w", encoding="utf-8") as f:
        json.dump(label_to_index, f, indent=2, sort_keys=True)


def load_label_mapping(path):
    # Load label mapping for inference/evaluation
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: int(v) for k, v in data.items()}
