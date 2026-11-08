"""
benchmark/run_benchmark.py - how does Yaazh compare with facebook/mms-tts-tam?

Every system reads the SAME sentences from Yaazh's held-out test set (Yaazh
never trained on them; MMS was trained on different data). Then we score:

  CER    Whisper (a speech-recognition model) writes down what it hears. CER is
         the % of characters it gets wrong vs the original Tamil text.
         LOWER = clearer. Whisper also makes mistakes on REAL speech, so the real
         speaker's CER is the floor: compare models to that, not to 0%.
  UTMOS  An AI that predicts the 1-5 rating human listeners would give.
         HIGHER = more natural. Trained mostly on English: use it to rank.
  Rate   Tamil characters spoken per second. The real speaker = natural pace.

    python -m benchmark.run_benchmark            (run from the project root)
    python -m benchmark.run_benchmark --utmos    (also the UTMOS score: this downloads
                                                  and runs code from GitHub via torch.hub)

Writes benchmark/results.md (+ results.json) and keeps all audio in
benchmark/audio/<system>/ so you can listen. Re-running skips audio that exists.
"""

import gc
import json
import sys
import time
import traceback
import unicodedata

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset, load_from_disk
from huggingface_hub import hf_hub_download

import config
from src.clarity import sharpen
from src.text_normalizer import normalize_text

NUM_SENTENCES = 50
SEED = 0
SR = config.SAMPLING_RATE
BENCH_DIR = config.PROJECT_ROOT / "benchmark"
AUDIO_DIR = BENCH_DIR / "audio"
LOG_FILE = BENCH_DIR / "run.log"
WHISPER_MODEL = "openai/whisper-large-v3-turbo"
MMS_MODEL = "facebook/mms-tts-tam"

YAAZH_PLAIN = "yaazh_clarity_0"
YAAZH_CLEAR = f"yaazh_clarity_{config.DEFAULT_CLARITY}"
SYSTEMS = ["real_speaker", YAAZH_PLAIN, YAAZH_CLEAR, "mms_tts_tam"]


def log(message: str):
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def free_gpu():
    gc.collect()
    torch.cuda.empty_cache()


def wav_path(system: str, sentence_id: str):
    return AUDIO_DIR / system / f"{sentence_id}.wav"


def save(system: str, sentence_id: str, audio: np.ndarray):
    path = wav_path(system, sentence_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, SR)


def already_done(system: str, sentences) -> bool:
    return all(wav_path(system, s["id"]).exists() for s in sentences)


# ---------------------------------------------------------------------------
# 1. The sentences (+ the real recordings, as the reference)
# ---------------------------------------------------------------------------

def pick_sentences():
    """Held-out test sentences with their ORIGINAL Tamil text and real recording."""
    test_texts = load_from_disk(str(config.DATA_DIR / "processed_male_full"))["test"]["normalized_text"]
    position = {text: i for i, text in enumerate(test_texts)}

    # The raw parquet files are already in the Hugging Face cache from data prep.
    shard_index = json.loads(config.SHARD_INDEX_PATH.read_text())
    files = [hf_hub_download(config.DATASET_NAME, s["file"], repo_type="dataset")
             for s in shard_index if s["male"] > 0]
    raw = load_dataset("parquet", data_files=files, split="train")
    raw = raw.filter(lambda g: g == 1, input_columns=["gender"])  # 1 = male

    # Match raw rows to the test set by their normalized text.
    matches = {}
    for row, text in enumerate(raw["text"]):
        normalized = normalize_text(text)
        if normalized in position and normalized not in matches:
            matches[normalized] = row
    chosen = sorted(matches, key=position.get)[:NUM_SENTENCES]

    rows = raw.select([matches[t] for t in chosen]).cast_column("audio", Audio(sampling_rate=SR))
    return [{"id": f"s{i:02d}", "text": r["text"], "normalized": normalize_text(r["text"]),
             "audio": np.asarray(r["audio"]["array"], dtype=np.float32)}
            for i, r in enumerate(rows)]


# ---------------------------------------------------------------------------
# 2. Generate audio with each system
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_yaazh(sentences):
    from src.inference import YaazhTTS

    tts = YaazhTTS(config.FINAL_MODEL_DIR)
    for s in sentences:
        torch.manual_seed(SEED)  # SpeechT5 adds a little randomness: fix it
        ids = tts.processor(text=s["normalized"], return_tensors="pt")["input_ids"].to(tts.device)
        mel = tts.model.generate_speech(ids, tts.speaker_embedding)   # one spectrogram...
        save(YAAZH_PLAIN, s["id"], tts.vocoder(mel).cpu().numpy())     # ...without and with clarity
        sharp = sharpen(mel, tts.contrast_stats, config.DEFAULT_CLARITY)
        save(YAAZH_CLEAR, s["id"], tts.vocoder(sharp).cpu().numpy())


@torch.no_grad()
def generate_mms(sentences):
    from transformers import AutoTokenizer, VitsModel

    tokenizer = AutoTokenizer.from_pretrained(MMS_MODEL)
    model = VitsModel.from_pretrained(MMS_MODEL).to("cuda").eval()
    assert model.config.sampling_rate == SR, "MMS should output 16 kHz"
    for s in sentences:
        torch.manual_seed(SEED)  # VITS also samples random noise when it speaks
        inputs = tokenizer(s["text"], return_tensors="pt").to("cuda")  # MMS reads Tamil script directly
        save("mms_tts_tam", s["id"], model(**inputs).waveform[0].cpu().numpy())


# ---------------------------------------------------------------------------
# 3. Score
# ---------------------------------------------------------------------------

def comparable(text: str) -> str:
    """Tamil text for CER: drop punctuation, symbols and spaces - but KEEP vowel signs."""
    text = unicodedata.normalize("NFC", text)
    return "".join(ch for ch in text if unicodedata.category(ch)[0] not in "PSZ" and not ch.isspace())


def edit_distance(a: str, b: str) -> int:
    """Minimum single-character insertions/deletions/substitutions to turn a into b."""
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (char_a != char_b)))
        previous = current
    return previous[-1]


def load_wav(system, sentence_id):
    return sf.read(wav_path(system, sentence_id), dtype="float32")[0]


@torch.no_grad()
def transcribe(sentences):
    from transformers import pipeline

    asr = pipeline("automatic-speech-recognition", model=WHISPER_MODEL,
                   torch_dtype=torch.float16, device="cuda:0")
    transcripts = {}
    for system in SYSTEMS:
        clips = [{"raw": load_wav(system, s["id"]), "sampling_rate": SR} for s in sentences]
        results = asr(clips, batch_size=8, generate_kwargs={"language": "tamil", "task": "transcribe"})
        transcripts[system] = [r["text"] for r in results]
        log(f"  transcribed {system}")
    return transcripts


@torch.no_grad()
def utmos(sentences):
    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    predictor = predictor.to("cuda").eval()
    scores = {}
    for system in SYSTEMS:
        scores[system] = [float(predictor(torch.tensor(load_wav(system, s["id"])).unsqueeze(0).cuda(), SR))
                          for s in sentences]
        log(f"  UTMOS scored {system}")
    return scores


def report(sentences, transcripts, mos):
    rows, details = [], {}
    for system in SYSTEMS:
        edits = chars = 0
        seconds = 0.0
        for s, heard in zip(sentences, transcripts[system]):
            reference = comparable(s["text"])
            edits += edit_distance(comparable(heard), reference)
            chars += len(reference)
            seconds += len(load_wav(system, s["id"])) / SR
        rows.append({
            "system": system,
            "cer": edits / chars,
            "utmos": float(np.mean(mos[system])) if mos else None,
            "utmos_std": float(np.std(mos[system])) if mos else None,
            "chars_per_second": chars / seconds,
        })
        details[system] = [{"id": s["id"], "text": s["text"], "heard": h,
                            "utmos": mos[system][i] if mos else None}
                           for i, (s, h) in enumerate(zip(sentences, transcripts[system]))]

    real = rows[0]
    lines = [
        f"# Yaazh benchmark ({len(sentences)} held-out sentences)\n",
        "| System | CER ↓ (clarity) | UTMOS ↑ (naturalness, 1-5) | Speaking rate (chars/s) |",
        "|---|---|---|---|",
    ]
    for r in rows:
        mos_text = f"{r['utmos']:.2f} ± {r['utmos_std']:.2f}" if r["utmos"] is not None else "n/a"
        pace = r["chars_per_second"] / real["chars_per_second"]
        lines.append(f"| {r['system']} | {r['cer']:.1%} | {mos_text} | "
                     f"{r['chars_per_second']:.1f} ({pace:.0%} of real) |")
    lines += [
        "",
        "- **CER**: % of characters Whisper got wrong vs the original text. The real speaker's CER is "
        "Whisper's own error floor; a model is as clear as real speech when it matches that.",
        "- **UTMOS**: predicted listener rating. Mostly English-trained: use it to rank, not as an absolute.",
        f"- Whisper: `{WHISPER_MODEL}`. UTMOS: `utmos22_strong` (SpeechMOS). Audio: `benchmark/audio/`.",
    ]
    (BENCH_DIR / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (BENCH_DIR / "results.json").write_text(
        json.dumps({"summary": rows, "per_sentence": details}, ensure_ascii=False, indent=1), encoding="utf-8")
    log("RESULTS\n" + "\n".join(lines[:len(rows) + 3]))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    BENCH_DIR.mkdir(exist_ok=True)
    log("Picking held-out sentences and their real recordings...")
    sentences = pick_sentences()
    log(f"{len(sentences)} sentences picked")
    (BENCH_DIR / "sentences.json").write_text(json.dumps(
        [{"id": s["id"], "text": s["text"], "romanized_for_yaazh": s["normalized"]} for s in sentences],
        ensure_ascii=False, indent=1), encoding="utf-8")

    if not already_done("real_speaker", sentences):
        for s in sentences:
            save("real_speaker", s["id"], s["audio"])
    for system, generate in [(YAAZH_CLEAR, generate_yaazh), ("mms_tts_tam", generate_mms)]:
        if already_done(system, sentences):
            log(f"{system}: audio already generated, skipping")
            continue
        log(f"Generating {system}...")
        generate(sentences)
        free_gpu()

    log(f"Transcribing everything with Whisper ({WHISPER_MODEL})...")
    transcripts = transcribe(sentences)
    free_gpu()

    mos = None
    if "--utmos" in sys.argv:  # opt-in: downloads and runs code from GitHub via torch.hub
        log("Scoring naturalness with UTMOS...")
        try:
            mos = utmos(sentences)
        except Exception as error:  # the CER results are still useful without it
            log(f"UTMOS failed, continuing without it: {error!r}")
        free_gpu()
    else:
        log("UTMOS skipped (add --utmos to include the naturalness score)")

    report(sentences, transcripts, mos)
    log("DONE - see benchmark/results.md")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("FAILED\n" + traceback.format_exc())
        raise
