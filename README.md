# Yaazh (யாழ்) - Tamil text-to-speech

Yaazh fine-tunes [microsoft/speecht5_tts](https://huggingface.co/microsoft/speecht5_tts) on the
male speaker of [SPRINGLab/IndicTTS_Tamil](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Tamil)
so it can read Tamil text aloud. It trains locally on a 6 GB laptop GPU (RTX 4050).

**Version 1.0.0.** This repo contains the code only. The dataset and trained weights are not
included (see [License](#license)); the scripts below rebuild everything.

```
Tamil text ──normalize_text──► "thamizh oru azhakaana mozhi"
           ──SpeechT5──────► log-mel spectrogram (a picture of the sound)
           ──HiFi-GAN──────► 16 kHz audio (.wav)
```

## Project layout

| File | What it does |
|---|---|
| `config.py` | Every path and hyperparameter in one place |
| `src/text_normalizer.py` | Tamil → Latin romanization (numbers → Tamil words first). Used by training, inference and the demo |
| `src/prepare_data.py` | Downloads the data and builds training features: text tokens, spectrograms, speaker embeddings |
| `src/train.py` | Fine-tunes SpeechT5 (with a smoke-test mode and automatic resume) |
| `src/progress_callback.py` | Writes training progress to `outputs/progress.json`; handles safe pausing |
| `dashboard.py` | Live training dashboard: progress, time left, loss curve, GPU memory, Pause button |
| `src/inference.py` | Text in, `.wav` out (optional `--clarity` boost) |
| `src/clarity.py` | Measures and fixes the over-smoothed ("muffled") spectrograms |
| `app.py` | Gradio demo |
| `tests/test_text_normalizer.py` | Unit tests for the normalizer |

## Setup (Windows, PowerShell)

```powershell
py -3.11 -m venv .venv                 # Python 3.11: best library support
.venv\Scripts\Activate.ps1             # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt        # includes the CUDA 12.6 build of PyTorch (~2.5 GB)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"   # expect: True ...
```

Why these versions are pinned is explained at the top of `requirements.txt`. In short:
- speechbrain needs torchaudio, and the newest torchaudio (2.11) pins torch to 2.11.
- datasets 3.6 decodes audio without FFmpeg.

**Hugging Face login** is only needed to push the model. The dataset is public and not gated.
Run `huggingface-cli login`, or set the `HF_TOKEN` environment variable. Never put the token in code.

## Usage

Run everything from the project folder with the venv active. Use `python -m src.xxx`, not
`python src/xxx.py`, so the scripts can find `config.py`.

```powershell
pytest -v                                   # 1. check the normalizer (seconds)
python -m src.text_normalizer "தமிழ் 24"     #    try it: -> thamizh irupaththu naanku

python -m src.prepare_data --inspect        # 2. quick report: exact clip counts, vocabulary
                                            #    check, too-long sentences (no audio download)
python -m src.prepare_data                  # 3. download + build features (~1 GB per 1,000 clips)

python -m src.train --smoke-test            # 4. 20 steps on the longest clips: peak GPU memory
python -m src.train                         # 5. full training (resumes if interrupted)
python -m src.train --max-steps 1000 --save-every 250   #  shorter run / more frequent checkpoints
python dashboard.py                         #    in a 2nd terminal: http://127.0.0.1:7861

python -m src.inference --text "வணக்கம்" --out outputs/hello.wav   # 6. synthesize
python app.py                               #    demo: http://127.0.0.1:7860

python -m src.clarity                       # 7. (once) measure contrast for the clarity boost
python -m src.inference --text "வணக்கம்" --clarity 0      #    boost off (default: config.DEFAULT_CLARITY = 1.2)
```

**Subset first, then full data.** `config.SUBSET_SIZE = 2000` builds a small dataset to check the
whole pipeline. When that works, set it to `None` and re-run steps 3–5. Each setting gets its own
processed-data and checkpoint folder, so nothing is overwritten.

**Training in several sessions.** Press **⏸ Pause training safely** on the dashboard. Training
saves a checkpoint at the current step and exits. Run the *same* train command again to resume:
the weights, optimizer, learning-rate schedule and data position all continue exactly.

Avoid Ctrl+C: it loses everything since the last checkpoint. Keep the laptop plugged in and awake
while training.

**Out of memory?** In `config.py`, halve `PER_DEVICE_TRAIN_BATCH_SIZE` and double
`GRADIENT_ACCUMULATION_STEPS`. This keeps the same effective batch of 32. Also close other apps
that use the GPU.

**Push to the Hub (optional):** set `PUSH_TO_HUB = True` and `HUB_MODEL_ID` in `config.py`. Read
the license section first.

## Results (v1.0.0)

Training data: male speaker only, 3,637 clips (7.5 h). 4,000 steps on an RTX 4050, about 2.5 h
with free RAM. Eval loss: 0.461 → 0.440 → 0.434 → **0.430**.

Benchmark on 50 held-out sentences (`python -m benchmark.run_benchmark`). CER is the share of
characters Whisper (`whisper-large-v3-turbo`) mishears, measured against the original Tamil text.
Lower is clearer. Whisper also makes mistakes on real speech, so the real speaker's score is the
floor.

| System | CER ↓ | Extra errors vs real speech | Speaking pace |
|---|---|---|---|
| Real speaker (reference) | 11.6% | n/a | 100% |
| **Yaazh v1.0.0** (clarity 1.2) | **20.4%** | +8.8 points | 98% |
| Yaazh v1.0.0 (clarity 0) | 20.5% | +8.9 points | 98% |
| `facebook/mms-tts-tam` (VITS) | 14.2% | +2.6 points | 94% |

- **MMS is easier to understand.** It is Tamil-native and adversarially trained (VITS).
- **Yaazh matches the speaker's pace and voice.** MMS uses its own voice.
- **The clarity boost makes the voice sound crisper but doesn't change intelligibility.**

## How the romanization works

SpeechT5's tokenizer only knows Latin characters, so Tamil must be romanized. The steps are:
- Digits become Tamil words first (24 → இருபத்து நான்கு).
- Tamil script is transliterated with `indic-transliteration` (Tamil → ITRANS).
- The ITRANS output is rewritten into one fixed spelling per sound:

| Tamil | Latin | | Tamil | Latin | | Tamil | Latin |
|---|---|---|---|---|---|---|---|
| ஆ ஈ ஊ ஏ ஓ | aa ii uu ee oo | | க ச ட த ப | k ch t th p | | ழ | zh |
| அ இ உ எ ஒ | a i u e o | | ங ஞ ண ந/ன | ng nj nh n | | ள / ல | lh / l |
| ஐ ஔ | ai au | | ம ய ர வ | m y r v | | ற / ர | rr / r |

Note: `indic-transliteration`'s Tamil table is built for writing *Sanskrit* in Tamil script. That
causes three quirks, and `ITRANS_TO_YAAZH` in `src/text_normalizer.py` corrects all of them:
- It outputs க as `gh` and த as `dh`.
- ன is missing from the table.
- ம் comes out as `M`.

The unit tests check every consonant + vowel-sign combination.

## Known limitations

- **Romanization is lossy and context-free.** We keep ழ/ள/ல, ற/ர and ண/ந apart with distinct
  spellings. But other details are lost:
  - ந and ன are merged.
  - Tamil's context-dependent pronunciation (e.g. க sounding like *k*, *g* or *h* depending on
    position) must be learned from the audio, not the spelling.
  - The model starts from English pretraining, so some sounds will be imperfect, especially with
    limited data.
- **Slightly muffled ("over-smoothed") speech.** SpeechT5 minimizes average error, so when unsure
  it predicts an average, and averages are smooth. We measured Yaazh's spectrograms at 82–88% of
  real speech's contrast, flattest at 3–7.6 kHz where consonants live. The real recordings sound
  clear through the same vocoder, so the blur is in the model's predictions, not the vocoder.
  `src/clarity.py` applies *variance compensation*: it stretches each mel band's ups and downs back
  toward real speech (`--clarity 1.0`, or the slider in the demo). This helps but doesn't fully
  undo the averaging. An end-to-end adversarial model such as VITS is the real fix.
- **Stress and intonation will be weak.** Punctuation is removed, so the model gets no prosody
  cues. Sentence breaks only add a pause at inference time.
- **One voice.** The model is trained on a single speaker (male).
- **Future upgrades:**
  - Add Tamil characters directly to the tokenizer, which would remove romanization entirely.
  - Compare against [facebook/mms-tts-tam](https://huggingface.co/facebook/mms-tts-tam).

## License

**Code:** MIT (see [LICENSE](LICENSE)). This covers only the code in this repository.

**Data and trained models:** the dataset is tagged **CC-BY-4.0** on Hugging Face. However, its card says it is *subject to the
original Indic TTS license terms* from IIT Madras:
<https://www.iitm.ac.in/donlab/indictts/downloads/license.pdf>.

Read that license before publishing the model or using it commercially.

Cite the Indic TTS project (BibTeX on the dataset card) if you use this work.
