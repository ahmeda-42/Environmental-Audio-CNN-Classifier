import argparse
import os
import sys

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dataset import load_label_mapping
from preprocessing.audio_features import load_audio, compute_spectrogram
from cnn import AudioCNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--n-mels", type=int, default=64)
    args = parser.parse_args()

    label_to_index = load_label_mapping(args.model_path + ".labels.json")
    index_to_label = {v: k for k, v in label_to_index.items()}

    y, sr = load_audio(args.audio_path, args.sample_rate, args.duration)
    feat = compute_spectrogram(y, sr, n_mels=args.n_mels)
    x = torch.tensor(feat).unsqueeze(0).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioCNN(num_classes=len(label_to_index))
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(x.to(device))
        pred = torch.argmax(logits, dim=1).item()
    print(index_to_label[pred])


if __name__ == "__main__":
    main()
