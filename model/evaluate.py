import argparse

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import AudioDataset, build_label_mapping, load_label_mapping
from model import AudioCNN


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc="eval", leave=False):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return correct / total if total else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    labels_path = args.model_path + ".labels.json"
    try:
        label_to_index = load_label_mapping(labels_path)
    except FileNotFoundError:
        label_to_index = build_label_mapping(df["label"].tolist())

    _, test_df = train_test_split(
        df,
        test_size=args.split,
        random_state=args.seed,
        stratify=df["label"],
    )

    test_ds = AudioDataset(test_df, label_to_index)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioCNN(num_classes=len(label_to_index))
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    acc = evaluate(model, test_loader, device)
    print(f"test_acc={acc:.3f}")


if __name__ == "__main__":
    main()
