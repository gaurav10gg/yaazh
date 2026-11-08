"""
clarity.py - reduce Yaazh's "muffled" sound by restoring spectrogram contrast.

WHY IT SOUNDS MUFFLED
---------------------
SpeechT5 is trained to minimise the average error of its spectrogram. When it is
unsure, the safest guess is the AVERAGE of the possibilities - and averages are
smooth. Each of the 80 frequency bands then swings up and down less than in real
speech, consonants lose their edges, and the voice sounds like it's speaking
"with something in its mouth". (We checked: the real recording through the same
vocoder sounds clear, so the blur comes from the predicted spectrogram.)

THE FIX: variance compensation (a classic text-to-speech post-filter)
---------------------------------------------------------------------
  1. Measure once, on held-out sentences, how much each mel band varies within a
     sentence in REAL speech vs in YAAZH's prediction of the same sentence.
         ratio[band] = real_std[band] / predicted_std[band]      (> 1 = too smooth)
  2. At synthesis time, stretch each band's ups and downs around its average:
         sharpened = mean + (mel - mean) * ratio ** strength
     strength 0 = off, 1 = match real speech's contrast, > 1 = exaggerate (harsh).

    python -m src.clarity        # measure once; saves contrast_stats.npz next to the model
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

import config

STATS_FILE = "contrast_stats.npz"


def sharpen(mel: torch.Tensor, stats: dict, strength: float) -> torch.Tensor:
    """mel: (frames, 80) log-mel spectrogram from SpeechT5. Returns a sharper copy."""
    mean = mel.mean(dim=0, keepdim=True)                 # each band's average in this sentence
    scale = stats["ratio"] ** strength                   # how much to stretch each band
    sharpened = mean + (mel - mean) * scale
    return torch.clamp(sharpened, min=stats["mel_min"], max=stats["mel_max"])  # stay in a realistic range


def load_stats(model_dir: Path, device) -> dict | None:
    path = Path(model_dir) / STATS_FILE
    if not path.exists():
        return None
    data = np.load(path)
    return {key: torch.tensor(data[key], dtype=torch.float32, device=device)
            for key in ("ratio", "mel_min", "mel_max")}


def measure(model_dir: Path, num_sentences: int):
    from datasets import load_from_disk
    import librosa

    from src.inference import YaazhTTS  # imported here to avoid a circular import

    tts = YaazhTTS(model_dir)
    test_set = load_from_disk(str(config.PROCESSED_DATASET_DIR))["test"]
    real_stds, predicted_stds, real_frames = [], [], []

    print(f"Comparing real vs predicted spectrograms on {num_sentences} held-out sentences...")
    for i, example in enumerate(test_set.select(range(min(num_sentences, len(test_set))))):
        real = np.array(example["labels"], dtype=np.float32)                       # (frames, 80)
        ids = tts.processor(text=example["normalized_text"], return_tensors="pt")["input_ids"].to(tts.device)
        with torch.no_grad():
            predicted = tts.model.generate_speech(ids, tts.speaker_embedding).cpu().numpy()
        real_stds.append(real.std(axis=0))
        predicted_stds.append(predicted.std(axis=0))
        real_frames.append(real)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{num_sentences}", flush=True)

    real_std, predicted_std = np.mean(real_stds, axis=0), np.mean(predicted_stds, axis=0)
    ratio = real_std / np.maximum(predicted_std, 1e-4)
    all_real = np.concatenate(real_frames)
    np.savez(Path(model_dir) / STATS_FILE, ratio=ratio.astype(np.float32),
             mel_min=all_real.min(axis=0), mel_max=all_real.max(axis=0))

    # Report per frequency band (SpeechT5's 80 mel bands span 80-7600 Hz).
    freqs = librosa.mel_frequencies(n_mels=80, fmin=80, fmax=7600)
    print(f"\nYaazh's spectrogram contrast vs real speech (1.00 = identical):")
    for name, lo, hi in [("low   (80 Hz - 1 kHz, vowels/voice)", 0, 1000),
                         ("mid   (1 - 3 kHz, vowel colour)", 1000, 3000),
                         ("high  (3 - 7.6 kHz, consonants/clarity)", 3000, 8000)]:
        band = (freqs >= lo) & (freqs < hi)
        print(f"  {name:42s} Yaazh has {1 / ratio[band].mean():.0%} of real contrast")
    print(f"\nSaved {Path(model_dir) / STATS_FILE}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Measure contrast stats for the clarity boost.")
    parser.add_argument("--model", default=str(config.FINAL_MODEL_DIR))
    parser.add_argument("--sentences", type=int, default=60)
    args = parser.parse_args()
    measure(Path(args.model), args.sentences)


if __name__ == "__main__":
    main()
