import json
from functools import lru_cache
import os
import librosa
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
    def __init__(
        self,
        dataframe,
        label_to_index,
        sample_rate=22050,
        duration=4.0,
        n_mels=64,
        n_fft=1024,
        hop_length=512,
        augment=True,
        spec_strength=1.0,
        time_mask_param=20,
        freq_mask_param=8,
        num_time_masks=2,
        num_freq_masks=2,
        aug_time_stretch=True,
        time_stretch_range=(0.9, 1.1),
        aug_pitch_shift=True,
        pitch_shift_steps=(-2.0, 2.0),
        aug_noise=True,
        noise_std=0.005,
        aug_time_shift=True,
        time_shift_max_fraction=0.1,
    ):
        # Store metadata and feature parameters
        self.df = dataframe.reset_index(drop=True)
        self.label_to_index = label_to_index
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.augment = augment
        self.spec_strength = spec_strength
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks
        self.aug_time_stretch = aug_time_stretch
        self.time_stretch_range = time_stretch_range
        self.aug_pitch_shift = aug_pitch_shift
        self.pitch_shift_steps = pitch_shift_steps
        self.aug_noise = aug_noise
        self.noise_std = noise_std
        self.aug_time_shift = aug_time_shift
        self.time_shift_max_fraction = time_shift_max_fraction

    def _apply_spec_augment(self, feat):
        # SpecAugment: random time/frequency masking
        augmented = feat.copy()
        num_mels, num_frames = augmented.shape

        freq_param = max(1, int(self.freq_mask_param * self.spec_strength))
        time_param = max(1, int(self.time_mask_param * self.spec_strength))
        num_freq_masks = max(1, int(self.num_freq_masks * self.spec_strength))
        num_time_masks = max(1, int(self.num_time_masks * self.spec_strength))

        for _ in range(num_freq_masks):
            f = np.random.randint(0, freq_param + 1)
            if f == 0 or f >= num_mels:
                continue
            f0 = np.random.randint(0, num_mels - f)
            augmented[f0 : f0 + f, :] = 0.0

        for _ in range(num_time_masks):
            t = np.random.randint(0, time_param + 1)
            if t == 0 or t >= num_frames:
                continue
            t0 = np.random.randint(0, num_frames - t)
            augmented[:, t0 : t0 + t] = 0.0

        return augmented

    def _pad_or_trim(self, y):
        target_len = int(self.sample_rate * self.duration)
        if y.size < target_len:
            return np.pad(y, (0, target_len - y.size), mode="constant")
        return y[:target_len]

    def _apply_audio_augment(self, y):
        # Apply waveform-level augmentation before spectrogram
        if self.aug_time_shift:
            max_shift = int(self.time_shift_max_fraction * y.size)
            if max_shift > 0:
                shift = np.random.randint(-max_shift, max_shift + 1)
                y = np.roll(y, shift)

        if self.aug_time_stretch:
            rate = np.random.uniform(*self.time_stretch_range)
            if rate > 0:
                y = librosa.effects.time_stretch(y, rate=rate)
                y = self._pad_or_trim(y)

        if self.aug_pitch_shift:
            steps = np.random.uniform(*self.pitch_shift_steps)
            y = librosa.effects.pitch_shift(y, sr=self.sample_rate, n_steps=steps)
            y = self._pad_or_trim(y)

        if self.aug_noise:
            noise = np.random.normal(0.0, self.noise_std, size=y.shape)
            y = y + noise

        return y

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
        if self.augment:
            y = self._apply_audio_augment(y)

        feat = compute_spectrogram(
            y,
            sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        if self.augment and self.spec_strength > 0:
            feat = self._apply_spec_augment(feat)

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


@lru_cache(maxsize=4)
def load_label_mapping(path):
    # Load label mapping for inference/evaluation
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: int(v) for k, v in data.items()}
