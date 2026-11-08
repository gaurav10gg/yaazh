"""
config.py - every path and hyperparameter for Yaazh lives here.

Why one file? So an experiment (different voice, subset size, batch size...)
is changed in ONE place, and every script - prepare_data, train, inference,
the Gradio demo and the dashboard - automatically agrees on the same values.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"          # processed dataset + speaker embedding
OUTPUT_DIR = PROJECT_ROOT / "outputs"     # checkpoints, final model, progress log

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATASET_NAME = "SPRINGLab/IndicTTS_Tamil"

# The dataset has exactly two speakers: one female, one male.
# Training on one keeps the voice consistent.
SPEAKER_GENDER = "male"   # "male" or "female"

# Start small to check the whole pipeline works, then set to None for all clips.
SUBSET_SIZE = None        # None = every clip of the chosen speaker (5,022); 2000 = quick test run
SEED = 42                 # makes shuffling / splitting reproducible

SAMPLING_RATE = 16_000    # SpeechT5 and HiFi-GAN expect 16 kHz (the dataset is 48 kHz)
MAX_INPUT_TOKENS = 200    # drop sentences longer than this (after romanization)
MIN_AUDIO_SECONDS = 1.0   # drop clips that are too short to learn from
MAX_AUDIO_SECONDS = 20.0  # drop very long clips - they blow up GPU memory
TEST_SIZE = 0.1           # 90% train / 10% test

# Name the processed folder after the settings, so a subset run and a full run
# never overwrite each other.
_subset_tag = "full" if SUBSET_SIZE is None else f"subset{SUBSET_SIZE}"
PROCESSED_DATASET_DIR = DATA_DIR / f"processed_{SPEAKER_GENDER}_{_subset_tag}"
SPEAKER_EMBEDDING_PATH = DATA_DIR / f"speaker_embedding_{SPEAKER_GENDER}.npy"
SHARD_INDEX_PATH = DATA_DIR / "shard_index.json"   # which parquet files hold which speaker

# ---------------------------------------------------------------------------
# Pretrained models (downloaded from the Hugging Face Hub)
# ---------------------------------------------------------------------------
BASE_MODEL = "microsoft/speecht5_tts"              # the TTS model we fine-tune
VOCODER_MODEL = "microsoft/speecht5_hifigan"       # turns spectrograms into audio
SPEAKER_MODEL = "speechbrain/spkrec-xvect-voxceleb"  # makes 512-dim voice "fingerprints"
SPEAKER_MODEL_DIR = PROJECT_ROOT / "pretrained_models" / "spkrec-xvect-voxceleb"

# ---------------------------------------------------------------------------
# Training (sized for a 6 GB RTX 4050)
# ---------------------------------------------------------------------------
# Trainer saves checkpoint-1000, -2000, ... here. One folder per data setting, so a
# full-data run never accidentally "resumes" from the subset run's checkpoints.
RUN_NAME = f"{SPEAKER_GENDER}_{_subset_tag}"
CHECKPOINT_DIR = OUTPUT_DIR / f"checkpoints_{RUN_NAME}"
FINAL_MODEL_DIR = OUTPUT_DIR / "yaazh-final"  # the model used by inference + demo
PROGRESS_FILE = OUTPUT_DIR / "progress.json"  # written during training, read by dashboard.py
PAUSE_FILE = OUTPUT_DIR / "PAUSE_REQUESTED"   # dashboard creates it -> training saves + stops
SMOKE_TEST_DIR = OUTPUT_DIR / "smoke-test"

PER_DEVICE_TRAIN_BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8   # effective batch size = 4 x 8 = 32
LEARNING_RATE = 1e-5
WARMUP_STEPS = 500
MAX_STEPS = 4000
EVAL_STEPS = 1000
SAVE_STEPS = 1000
LOGGING_STEPS = 25                # how often loss is logged (and the dashboard updates)
SAVE_TOTAL_LIMIT = 2              # keep only the newest checkpoints (each is ~1.7 GB)
FP16 = True                       # half-precision math: less memory, faster on RTX GPUs
GRADIENT_CHECKPOINTING = True     # recompute activations instead of storing them: saves memory
SMOKE_TEST_STEPS = 20

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
# Clarity boost (src/clarity.py): 0 = off, 1 = match real speech's spectrogram
# contrast. 1.2 was picked by ear: slightly sharper than real speech, which
# offsets some of SpeechT5's blur. Needs `python -m src.clarity` once per model.
DEFAULT_CLARITY = 1.2

# ---------------------------------------------------------------------------
# Hugging Face Hub (optional)
# ---------------------------------------------------------------------------
# Read the IndicTTS license before publishing (see README).
PUSH_TO_HUB = False
HUB_MODEL_ID = "your-hf-username/yaazh-tamil-tts"
