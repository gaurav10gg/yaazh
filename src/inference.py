"""
inference.py - Tamil text in, 16 kHz .wav file out.

    python -m src.inference --text "தமிழ் ஒரு அழகான மொழி." --out outputs/hello.wav

How speech is produced:
    Tamil text --normalize_text--> romanized text --tokenizer--> input_ids
    input_ids + speaker embedding --SpeechT5--> log-mel spectrogram (a "picture" of sound)
    spectrogram --HiFi-GAN vocoder--> waveform (actual audio samples)
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

import config
from src.clarity import load_stats, sharpen
from src.text_normalizer import normalize_text

SENTENCE_END = re.compile(r"[.!?।\n]+")  # split long input into sentences here
PAUSE_SECONDS = 0.3                      # silence inserted between sentences


class YaazhTTS:
    def __init__(self, model=config.FINAL_MODEL_DIR, device=None):
        """`model` is a local folder (final model or a checkpoint) or a Hub model id."""
        model = str(model)
        is_local = Path(model).exists()
        if not is_local and Path(model).is_absolute():
            raise FileNotFoundError(f"No model at {model}. Train one first: python -m src.train")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = SpeechT5Processor.from_pretrained(model)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(model).to(self.device).eval()
        self.vocoder = SpeechT5HifiGan.from_pretrained(config.VOCODER_MODEL).to(self.device).eval()

        embedding = np.load(self._find_speaker_embedding(model, is_local))
        self.speaker_embedding = torch.tensor(embedding).unsqueeze(0).to(self.device)  # (1, 512)

        # Contrast stats for the clarity boost (created by `python -m src.clarity`).
        self.contrast_stats = load_stats(model, self.device) if is_local else None

    @staticmethod
    def _find_speaker_embedding(model: str, is_local: bool) -> Path:
        """The averaged voice embedding: saved next to the model, or from prepare_data."""
        if is_local and (Path(model) / "speaker_embedding.npy").exists():
            return Path(model) / "speaker_embedding.npy"
        if not is_local:
            try:  # a pushed Yaazh model carries its own embedding
                return Path(hf_hub_download(model, "speaker_embedding.npy"))
            except Exception:
                pass  # e.g. the base microsoft/speecht5_tts - fall back to ours
        if config.SPEAKER_EMBEDDING_PATH.exists():
            return config.SPEAKER_EMBEDDING_PATH
        raise FileNotFoundError("No speaker_embedding.npy found - run `python -m src.prepare_data` first.")

    def _num_tokens(self, normalized: str) -> int:
        return len(self.processor.tokenizer(normalized).input_ids)

    def split_into_chunks(self, text: str) -> list[str]:
        """
        Normalize the text sentence by sentence, keeping each piece within
        MAX_INPUT_TOKENS - the longest input the model saw during training.
        """
        chunks = []
        for sentence in SENTENCE_END.split(text):
            normalized = normalize_text(sentence)
            if not normalized:
                continue
            if self._num_tokens(normalized) <= config.MAX_INPUT_TOKENS:
                chunks.append(normalized)
                continue
            # Still too long: break it at word boundaries.
            current = ""
            for word in normalized.split():
                candidate = f"{current} {word}".strip()
                if current and self._num_tokens(candidate) > config.MAX_INPUT_TOKENS:
                    chunks.append(current)
                    current = word
                else:
                    current = candidate
            if current:
                chunks.append(current)
        return chunks

    @torch.no_grad()
    def synthesize(self, text: str, clarity: float = 0.0) -> tuple[np.ndarray, list[str]]:
        """
        Returns (16 kHz float32 audio, the romanized chunks the model read).
        clarity: 0 = off, 1 = restore real speech's spectrogram contrast (see src/clarity.py).
        """
        chunks = self.split_into_chunks(text)
        if not chunks:
            raise ValueError("Nothing to say: the text is empty after normalization.")
        if clarity > 0 and self.contrast_stats is None:
            raise ValueError("Clarity boost needs contrast stats: run `python -m src.clarity` once.")

        pause = np.zeros(int(PAUSE_SECONDS * config.SAMPLING_RATE), dtype=np.float32)
        pieces = []
        for chunk in chunks:
            input_ids = self.processor(text=chunk, return_tensors="pt")["input_ids"].to(self.device)
            spectrogram = self.model.generate_speech(input_ids, self.speaker_embedding)  # (frames, 80)
            if clarity > 0:
                spectrogram = sharpen(spectrogram, self.contrast_stats, clarity)
            speech = self.vocoder(spectrogram)                                        # spectrogram -> audio
            pieces += [speech.cpu().numpy().astype(np.float32), pause]
        return np.concatenate(pieces[:-1]), chunks


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Tamil text -> .wav")
    parser.add_argument("--text", required=True, help="Tamil text to speak")
    parser.add_argument("--out", default=str(config.OUTPUT_DIR / "yaazh.wav"), help="output .wav path")
    parser.add_argument("--model", default=str(config.FINAL_MODEL_DIR),
                        help="model folder (or a checkpoint folder, or a Hub model id)")
    parser.add_argument("--clarity", type=float, default=config.DEFAULT_CLARITY,
                        help=f"0 = off, 1 = real speech contrast (default {config.DEFAULT_CLARITY})")
    args = parser.parse_args()

    tts = YaazhTTS(args.model)
    clarity = args.clarity
    if clarity > 0 and tts.contrast_stats is None:
        print("No contrast stats for this model, so the clarity boost is off "
              "(run `python -m src.clarity --model ...` once to enable it).")
        clarity = 0.0
    audio, chunks = tts.synthesize(args.text, clarity=clarity)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, audio, config.SAMPLING_RATE)
    print("Model read   :", " | ".join(chunks))
    print(f"Saved {len(audio) / config.SAMPLING_RATE:.1f} s of audio -> {args.out}")


if __name__ == "__main__":
    main()
