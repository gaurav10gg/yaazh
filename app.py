"""
app.py - Gradio demo: type Tamil, hear Yaazh speak.

    python app.py                                   -> http://127.0.0.1:7860
    python app.py --model outputs/yaazh-subset-1000steps

The "Clarity boost" slider sharpens the predicted spectrogram before the vocoder
(see src/clarity.py). It needs `python -m src.clarity` to have been run once.
"""

import argparse

import gradio as gr

import config
from src.inference import YaazhTTS

EXAMPLES = [
    "வணக்கம், எப்படி இருக்கிறீர்கள்?",
    "தமிழ் ஒரு அழகான மொழி.",
    "இன்று 24 பேர் கூட்டத்திற்கு வந்தனர்.",
]


def build_demo(tts: YaazhTTS) -> gr.Interface:
    has_stats = tts.contrast_stats is not None
    default_clarity = config.DEFAULT_CLARITY if has_stats else 0.0

    def speak(text: str, clarity: float):
        if not text or not text.strip():
            raise gr.Error("Please type some Tamil text.")
        audio, chunks = tts.synthesize(text, clarity=clarity if has_stats else 0.0)
        return (config.SAMPLING_RATE, audio), "\n".join(chunks)

    return gr.Interface(
        fn=speak,
        inputs=[
            gr.Textbox(label="Tamil text", lines=4, placeholder="தமிழில் எழுதுங்கள்..."),
            gr.Slider(0.0, 1.5, value=default_clarity, step=0.1, label="Clarity boost",
                      interactive=has_stats,
                      info="0 = off · 1 = match real speech's sharpness · higher can sound harsh"
                      if has_stats else "Run `python -m src.clarity` once to enable"),
        ],
        outputs=[
            gr.Audio(label="Speech", type="numpy"),
            gr.Textbox(label="What the model actually reads (romanized)"),
        ],
        title="Yaazh - Tamil text-to-speech",
        description="SpeechT5 fine-tuned on IndicTTS Tamil. Numbers are spoken in Tamil. "
                    "Try the same sentence at different clarity settings.",
        examples=[[sentence, default_clarity] for sentence in EXAMPLES],
        flagging_mode="never",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(config.FINAL_MODEL_DIR),
                        help="model folder, checkpoint folder, or Hub model id")
    args = parser.parse_args()
    build_demo(YaazhTTS(args.model)).launch(server_port=7860)
