from model.predict import predict

AUDIO_PATHS = [
    "data/UrbanSound8K/audio/fold1/21684-9-0-39.wav",
    "data/UrbanSound8K/audio/fold1/21684-9-0-39.wav",
    "data/UrbanSound8K/audio/fold1/21684-9-0-39.wav",
]

for audio_path in AUDIO_PATHS:
    data = predict(audio_path, sample_rate=22050, duration=4.0, n_mels=64)
    print(data)