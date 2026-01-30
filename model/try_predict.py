import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from model.predict import predict

AUDIO_PATHS = [
    "data/UrbanSound8K/audio/fold1/21684-9-0-39.wav",
    "data/UrbanSound8K/audio/fold1/7061-6-0-0.wav",
    "data/UrbanSound8K/audio/fold1/9031-3-1-0.wav",
]

for audio_path in AUDIO_PATHS:
    data = predict(audio_path)
    top_k = data["top_k"]
    print("\nProcessing audio: " + audio_path)
    print("Spectrogram's shape: " + str(data["spectrogram"]["shape"]))
    print("Top predictions:")
    for prediction in top_k:
        print(f'• Prediction: {prediction["label"]} ({prediction["confidence"]*100:.1f}%)')
    print("\n" + "="*50)