import matplotlib.pyplot as plt
from preprocessing.audio_features import load_audio, compute_spectrogram

y, sr = load_audio("audio/fold1/101415-3-0-2.wav")
mel = compute_spectrogram(y, sr)

plt.figure(figsize=(10, 4))
plt.imshow(mel, origin='lower', aspect='auto', cmap='magma')
plt.title("Mel Spectrogram")
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar(format="%+2.0f dB")
plt.show()