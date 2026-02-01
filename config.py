MODEL_PATH = "artifacts/cnn.pt"
LABELS_PATH = MODEL_PATH + ".labels.json"
CSV_PATH = "data/urbansound8k.csv"
DATASET_ROOT = "data/UrbanSound8K"

SAMPLE_RATE = 22050
DURATION = 4.0
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512

# SpecAugment defaults (optional)
SPEC_AUGMENT = True
TIME_MASK_PARAM = 20
FREQ_MASK_PARAM = 8
NUM_TIME_MASKS = 2
NUM_FREQ_MASKS = 2

# RMS normalization (optional)
RMS_NORMALIZE = True
RMS_TARGET = 0.1

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 30
SEED = 67
