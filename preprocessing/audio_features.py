import numpy as np
import librosa

# Load audio file as waveform and resample to target sample rate and duration
def load_audio(audio_path, target_sr=22050, duration=4.0):
    y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    target_len = int(target_sr * duration)
    if len(y) < target_len:
        pad_width = target_len - len(y)
        y = np.pad(y, (0, pad_width), mode="constant")
    else:
        y = y[:target_len]
    return y, target_sr


# Load audio from upload file
def load_audio_from_upload(upload: UploadFile, target_sr, duration):
    # Save to a temp file so librosa can read it reliably
    suffix = os.path.splitext(upload.filename or "")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(upload.file.read())
        tmp_path = tmp.name
    try:
        y, sr = load_audio(tmp_path, target_sr=target_sr, duration=duration)
    finally:
        os.remove(tmp_path)
    return y, sr


# Compute spectrogram from waveform (maybe change 64 to 128 and 1024 to 2048?)
def compute_spectrogram(y, sr=22050, n_mels=64, n_fft=1024, hop_length=512):
    # Short Time Fourier Transform (STFT) to get complex spectrogram
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    
    # magnitude
    S, _ = librosa.magphase(D)
    
    # convert to log scale (dB)
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    
    # normalize
    S_db_norm = (S_db - S_db.mean()) / S_db.std()
    return S_db_norm
