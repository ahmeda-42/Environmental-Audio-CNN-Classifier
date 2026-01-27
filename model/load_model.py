import torch
from model.cnn import AudioCNN
from model.train import MODEL_OUT

MODEL_PATH = MODEL_OUT

def load_model(num_classes = 10):
    # Build the model and load trained weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model, device