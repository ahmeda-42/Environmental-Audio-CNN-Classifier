import os
from functools import lru_cache
import torch
from model.cnn import AudioCNN
from config import MODEL_PATH


@lru_cache(maxsize=1)
def load_model(num_classes=10):
    # Build the model and load trained weights
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. Train the model first."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model, device