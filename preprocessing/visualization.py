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

def spectrogram_to_png_base64(spec):
    spec_min = float(np.min(spec))
    spec_max = float(np.max(spec))
    spec_range = spec_max - spec_min or 1.0
    normalized = (spec - spec_min) / spec_range
    pixels = (normalized * 255).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(pixels, mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")