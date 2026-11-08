# Yaazh benchmark (50 held-out sentences)

| System | CER ↓ (clarity) | UTMOS ↑ (naturalness, 1-5) | Speaking rate (chars/s) |
|---|---|---|---|
| real_speaker | 11.6% | n/a | 11.3 (100% of real) |
| yaazh_clarity_0 | 20.5% | n/a | 11.1 (98% of real) |
| yaazh_clarity_1.2 | 20.4% | n/a | 11.1 (98% of real) |
| mms_tts_tam | 14.2% | n/a | 10.6 (94% of real) |

- **CER**: % of characters Whisper got wrong vs the original text. The real speaker's CER is Whisper's own error floor; a model is as clear as real speech when it matches that.
- **UTMOS**: predicted listener rating. Mostly English-trained: use it to rank, not as an absolute.
- Whisper: `openai/whisper-large-v3-turbo`. UTMOS: `utmos22_strong` (SpeechMOS). Audio: `benchmark/audio/`.
