"""
prepare_data.py - turn the raw IndicTTS Tamil dataset into training-ready features.

Two modes:

    python -m src.prepare_data --inspect
        QUICK (no audio download). Reads only the small `text` and `gender`
        columns from the dataset's parquet files and reports:
          - exact clip counts per speaker
          - which characters survive normalization (vocabulary check)
          - how many sentences are too long (> MAX_INPUT_TOKENS)

    python -m src.prepare_data
        FULL. Downloads the audio for the chosen speaker (several GB), then:
          1. keeps only that speaker            (one consistent voice)
          2. takes a random subset              (config.SUBSET_SIZE; None = all)
          3. normalizes the text                (Tamil -> Latin, same function as inference)
          4. drops sentences > 200 tokens       (SpeechT5 struggles with very long inputs)
          5. resamples audio 48 kHz -> 16 kHz   (what SpeechT5 + HiFi-GAN expect)
          6. per clip: input_ids + log-mel spectrogram (SpeechT5Processor)
                       + 512-dim x-vector speaker embedding (SpeechBrain)
          7. drops very short / very long clips
          8. 90/10 train/test split, saves everything to disk
          9. saves one AVERAGED speaker embedding = "the voice" used at inference
        Runs once: if the output folder already exists, it stops (use --force to redo).
"""

import argparse
import json
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from datasets import Audio, load_dataset
from huggingface_hub import HfFileSystem, hf_hub_download
from transformers import SpeechT5Processor

import config
from src.text_normalizer import find_unsupported_chars, normalize_text

# From the dataset card: `gender` is a ClassLabel with 0 = female, 1 = male.
GENDER_IDS = {"female": 0, "male": 1}


# ---------------------------------------------------------------------------
# Reading just the small columns (no audio) straight from the Hub
# ---------------------------------------------------------------------------

def read_text_and_gender_columns():
    """
    The dataset is 17 parquet files (~500 MB each, almost all of it audio).
    Parquet stores each column separately, so we can fetch only `text` and
    `gender` - a few MB in total - without downloading any audio.
    Returns one dict per file: {"file", "text": [...], "gender": [...]}.
    """
    fs = HfFileSystem()
    repo_path = f"datasets/{config.DATASET_NAME}"
    files = sorted(fs.glob(f"{repo_path}/data/*.parquet"))
    shards = []
    for i, path in enumerate(files, 1):
        # Small block size: fetch only the bytes of these two columns, not 10 MB chunks.
        with fs.open(path, "rb", block_size=256 * 1024) as f:
            table = pq.read_table(f, columns=["text", "gender"])
        shards.append({
            "file": path.removeprefix(repo_path + "/"),
            "text": table.column("text").to_pylist(),
            "gender": table.column("gender").to_pylist(),
        })
        print(f"  read columns of {i}/{len(files)}: {shards[-1]['file']}", flush=True)
    return shards


def build_shard_index(shards):
    """How many clips of each speaker are in each parquet file (saved as JSON)."""
    index = [
        {
            "file": s["file"],
            "female": s["gender"].count(GENDER_IDS["female"]),
            "male": s["gender"].count(GENDER_IDS["male"]),
        }
        for s in shards
    ]
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.SHARD_INDEX_PATH.write_text(json.dumps(index, indent=2))
    return index


def choose_shards(index, gender, subset_size):
    """Pick only the parquet files we need, so we don't download all 8.4 GB."""
    chosen, total = [], 0
    for shard in index:
        if shard[gender] == 0:
            continue
        chosen.append(shard["file"])
        total += shard[gender]
        if subset_size is not None and total >= subset_size:
            break
    return chosen, total


# ---------------------------------------------------------------------------
# --inspect mode
# ---------------------------------------------------------------------------

def inspect(tokenizer):
    print("Reading the text + gender columns from the Hub (no audio)...")
    shards = read_text_and_gender_columns()
    index = build_shard_index(shards)

    female = sum(s["female"] for s in index)
    male = sum(s["male"] for s in index)
    print(f"\nExact clip counts: female = {female}, male = {male}, total = {female + male}")
    print("Clips per file (female / male):")
    for s in index:
        print(f"  {s['file']}: {s['female']:4d} / {s['male']:4d}")

    gender_id = GENDER_IDS[config.SPEAKER_GENDER]
    texts = [t for s in shards for t, g in zip(s["text"], s["gender"]) if g == gender_id]
    print(f"\nNormalizing all {len(texts)} {config.SPEAKER_GENDER} sentences...")
    normalized = [normalize_text(t) for t in texts]

    print("\nExamples:")
    for raw, norm in list(zip(texts, normalized))[:3]:
        print(f"  {raw}\n  -> {norm}\n")

    leftovers = find_unsupported_chars(normalized, tokenizer)
    if leftovers:
        print("UNSUPPORTED characters left after normalization (add them to "
              "CHAR_REPLACEMENTS in src/text_normalizer.py):")
        for ch, count in leftovers.most_common():
            print(f"  {ch!r} (U+{ord(ch):04X}) x {count}")
    else:
        print("Vocabulary check: every character is in the SpeechT5 vocabulary.")

    lengths = np.array([len(tokenizer(t).input_ids) for t in normalized])
    too_long = int((lengths > config.MAX_INPUT_TOKENS).sum())
    print(f"\nToken lengths: median {int(np.median(lengths))}, max {lengths.max()}; "
          f"{too_long} of {len(lengths)} ({too_long / len(lengths):.0%}) exceed "
          f"{config.MAX_INPUT_TOKENS} and would be dropped.")

    chosen, available = choose_shards(index, config.SPEAKER_GENDER, config.SUBSET_SIZE)
    print(f"\nWith SUBSET_SIZE = {config.SUBSET_SIZE}, the full run will download "
          f"{len(chosen)} of {len(index)} parquet files ({available} {config.SPEAKER_GENDER} clips, "
          f"~{len(chosen) * 0.5:.1f} GB).")


# ---------------------------------------------------------------------------
# Full preparation
# ---------------------------------------------------------------------------

def load_speaker_model(device):
    """SpeechBrain x-vector model: turns a clip into a 512-number voice 'fingerprint'."""
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    return EncoderClassifier.from_hparams(
        source=config.SPEAKER_MODEL,
        savedir=str(config.SPEAKER_MODEL_DIR),
        run_opts={"device": device},
        # Copy files instead of symlinking: symlinks need admin rights on Windows.
        local_strategy=LocalStrategy.COPY,
    )


def compute_speaker_embedding(speaker_model, waveform) -> np.ndarray:
    with torch.no_grad():
        wav = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)    # (1, samples)
        embedding = speaker_model.encode_batch(wav)                        # (1, 1, 512)
        embedding = torch.nn.functional.normalize(embedding, dim=2)        # length 1, like the tutorial
    return embedding.squeeze().cpu().numpy()


def prepare(processor, force):
    out_dir = config.PROCESSED_DATASET_DIR
    if out_dir.exists() and not force:
        print(f"{out_dir} already exists - nothing to do. Use --force to rebuild it.")
        return

    tokenizer = processor.tokenizer
    counts = {}  # how many clips survive each step (saved to a summary file)
    t0 = time.time()

    # --- which parquet files contain our speaker? -------------------------
    if config.SHARD_INDEX_PATH.exists():
        index = json.loads(config.SHARD_INDEX_PATH.read_text())
    else:
        print("Building the shard index (reads only text + gender columns)...")
        index = build_shard_index(read_text_and_gender_columns())
    files, available = choose_shards(index, config.SPEAKER_GENDER, config.SUBSET_SIZE)

    # --- download only those files (cached: a re-run won't download again) --
    print(f"\nDownloading {len(files)} parquet files (~{len(files) * 0.5:.1f} GB, "
          f"{available} {config.SPEAKER_GENDER} clips)...")
    local_files = [
        hf_hub_download(config.DATASET_NAME, filename=f, repo_type="dataset") for f in files
    ]
    dataset = load_dataset("parquet", data_files=local_files, split="train")
    counts["downloaded_rows"] = len(dataset)

    # 1. one speaker ---------------------------------------------------------
    gender_id = GENDER_IDS[config.SPEAKER_GENDER]
    dataset = dataset.filter(lambda g: g == gender_id, input_columns=["gender"])
    counts["speaker_rows"] = len(dataset)

    # 2. subset --------------------------------------------------------------
    dataset = dataset.shuffle(seed=config.SEED)
    if config.SUBSET_SIZE is not None:
        dataset = dataset.select(range(min(config.SUBSET_SIZE, len(dataset))))
    counts["subset_rows"] = len(dataset)

    # 3. normalize text + vocabulary check -----------------------------------
    dataset = dataset.map(lambda t: {"normalized_text": normalize_text(t)}, input_columns=["text"])
    leftovers = find_unsupported_chars(dataset["normalized_text"], tokenizer)
    if leftovers:
        sys.exit(f"Unsupported characters after normalization: {dict(leftovers)}\n"
                 "Add them to CHAR_REPLACEMENTS in src/text_normalizer.py and re-run.")

    # 4. drop sentences that are too long ------------------------------------
    dataset = dataset.filter(
        lambda t: len(tokenizer(t).input_ids) <= config.MAX_INPUT_TOKENS,
        input_columns=["normalized_text"],
    )
    counts["after_token_filter"] = len(dataset)

    # 5. resample: decoding now returns 16 kHz arrays instead of 48 kHz -------
    dataset = dataset.cast_column("audio", Audio(sampling_rate=config.SAMPLING_RATE))

    # 6. the heavy step: spectrograms + x-vectors -----------------------------
    device = "cuda:0" if torch.cuda.is_available() else "cpu"  # SpeechBrain wants an explicit GPU index
    print(f"\nLoading the speaker-embedding model on {device}...")
    speaker_model = load_speaker_model(device)

    def featurize(example):
        audio = example["audio"]
        features = processor(
            text=example["normalized_text"],
            audio_target=audio["array"],             # -> log-mel spectrogram "labels"
            sampling_rate=audio["sampling_rate"],
            return_attention_mask=False,
        )
        labels = features["labels"][0]               # remove the batch dimension
        return {
            "input_ids": features["input_ids"],
            "labels": labels,                        # (frames, 80) log-mel spectrogram
            "speaker_embeddings": compute_speaker_embedding(speaker_model, audio["array"]),
            "num_frames": len(labels),
            "duration_seconds": len(audio["array"]) / audio["sampling_rate"],
        }

    print("Computing spectrograms and x-vectors (this is the slow part)...")
    dataset = dataset.map(featurize, remove_columns=["audio", "text", "gender"])

    # 7. drop very short / very long clips -----------------------------------
    dataset = dataset.filter(
        lambda d: config.MIN_AUDIO_SECONDS <= d <= config.MAX_AUDIO_SECONDS,
        input_columns=["duration_seconds"],
    )
    counts["after_duration_filter"] = len(dataset)

    # 8. split + save --------------------------------------------------------
    splits = dataset.train_test_split(test_size=config.TEST_SIZE, seed=config.SEED)
    counts["train"], counts["test"] = len(splits["train"]), len(splits["test"])
    splits.save_to_disk(str(out_dir))

    # 9. the averaged speaker embedding = the voice used at inference ---------
    embeddings = np.array(splits["train"]["speaker_embeddings"], dtype=np.float32)
    average = embeddings.mean(axis=0)
    average /= np.linalg.norm(average)
    np.save(config.SPEAKER_EMBEDDING_PATH, average.astype(np.float32))

    hours = sum(splits["train"]["duration_seconds"]) / 3600
    counts["train_hours"] = round(hours, 2)
    counts["minutes_taken"] = round((time.time() - t0) / 60, 1)
    summary_path = config.DATA_DIR / f"{out_dir.name}_summary.json"
    summary_path.write_text(json.dumps(counts, indent=2))

    print("\nDone. Clips surviving each step:")
    for step, value in counts.items():
        print(f"  {step:24s} {value}")
    print(f"\nSaved dataset   -> {out_dir}")
    print(f"Saved embedding -> {config.SPEAKER_EMBEDDING_PATH}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles can't print Tamil otherwise
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inspect", action="store_true", help="quick report, no audio download")
    parser.add_argument("--force", action="store_true", help="rebuild even if the output exists")
    args = parser.parse_args()

    print(f"Speaker: {config.SPEAKER_GENDER} | subset: {config.SUBSET_SIZE} | "
          f"output: {config.PROCESSED_DATASET_DIR}")
    processor = SpeechT5Processor.from_pretrained(config.BASE_MODEL)
    if args.inspect:
        inspect(processor.tokenizer)
    else:
        prepare(processor, args.force)


if __name__ == "__main__":
    main()
