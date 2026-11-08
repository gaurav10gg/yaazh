"""
train.py - fine-tune microsoft/speecht5_tts on the processed Tamil data.

    python -m src.train --smoke-test
        20 steps on the LONGEST clips, then reports peak GPU memory and speed.
        Longest clips = worst case for memory, so if this fits, the real run fits.

    python -m src.train
        The real run (config.MAX_STEPS steps). If checkpoints exist it resumes
        from the latest one, so you can pause and continue later.

    python -m src.train --max-steps 1000 --save-every 250
        Override the run length / checkpoint interval without editing config.py
        (eval runs at every checkpoint too).

Watch progress live in a second terminal:   python dashboard.py
Pause safely: the dashboard's Pause button (saves a checkpoint first, nothing lost).
Run the SAME command again to resume.
"""

import argparse
import shutil
import sys
from dataclasses import dataclass
from typing import Any

import torch
from datasets import load_from_disk
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    SpeechT5ForTextToSpeech,
    SpeechT5Processor,
)
from transformers.trainer_utils import get_last_checkpoint

import config
from src.progress_callback import ProgressCallback


@dataclass
class TTSDataCollatorWithPadding:
    """
    Turns a list of examples (all different lengths) into one padded batch.
    Follows the Hugging Face SpeechT5 fine-tuning tutorial.
    """
    processor: Any
    reduction_factor: int

    def __call__(self, features):
        input_ids = [{"input_ids": f["input_ids"]} for f in features]
        label_features = [{"input_values": f["labels"]} for f in features]
        speaker_features = [f["speaker_embeddings"] for f in features]

        # Pad text and spectrograms to the longest example in the batch.
        batch = self.processor.pad(input_ids=input_ids, labels=label_features, return_tensors="pt")

        # Padded spectrogram frames -> -100, which the loss function ignores.
        batch["labels"] = batch["labels"].masked_fill(
            batch.decoder_attention_mask.unsqueeze(-1).ne(1), -100
        )
        del batch["decoder_attention_mask"]  # not used during fine-tuning

        # SpeechT5's decoder predicts `reduction_factor` (=2) frames per step,
        # so the label length must be a multiple of it.
        if self.reduction_factor > 1:
            lengths = [len(f["input_values"]) for f in label_features]
            lengths = [n - n % self.reduction_factor for n in lengths]
            batch["labels"] = batch["labels"][:, : max(lengths)]

        # One 512-dim voice embedding per example -> tensor of shape (batch, 512).
        batch["speaker_embeddings"] = torch.tensor(speaker_features)
        return batch


def build_training_args(smoke_test: bool, max_steps: int, save_every: int) -> Seq2SeqTrainingArguments:
    common = dict(
        per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        warmup_steps=config.WARMUP_STEPS,
        fp16=config.FP16,
        gradient_checkpointing=config.GRADIENT_CHECKPOINTING,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # the recommended, safer variant
        label_names=["labels"],
        report_to="none",          # our own dashboard instead of TensorBoard/W&B
        dataloader_num_workers=0,  # most reliable on Windows
        seed=config.SEED,
    )
    if smoke_test:
        return Seq2SeqTrainingArguments(
            output_dir=str(config.SMOKE_TEST_DIR),
            max_steps=config.SMOKE_TEST_STEPS,
            logging_steps=5,
            eval_strategy="no",
            save_strategy="no",
            **common,
        )
    return Seq2SeqTrainingArguments(
        output_dir=str(config.CHECKPOINT_DIR),
        max_steps=max_steps,
        logging_steps=config.LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=save_every,   # must line up with saving, so the best checkpoint has an eval score
        save_strategy="steps",
        save_steps=save_every,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,       # at the end, keep the checkpoint with the lowest eval loss
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        **common,
    )


def report_smoke_test(result):
    peak = torch.cuda.max_memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    seconds_per_step = result.metrics["train_runtime"] / config.SMOKE_TEST_STEPS
    full_run_hours = seconds_per_step * config.MAX_STEPS / 3600
    print("\n=========== SMOKE TEST RESULT ===========")
    print(f"Peak GPU memory : {peak:.2f} GB of {total:.2f} GB ({peak / total:.0%})")
    print(f"Speed           : {seconds_per_step:.1f} s per step "
          f"({config.PER_DEVICE_TRAIN_BATCH_SIZE} x {config.GRADIENT_ACCUMULATION_STEPS} clips per step)")
    print(f"Full run        : ~{full_run_hours:.1f} h for {config.MAX_STEPS} steps "
          "(upper bound - the smoke test used the longest clips)")
    print(f"Final loss      : {result.metrics['train_loss']:.4f}")
    print("==========================================")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke-test", action="store_true", help=f"{config.SMOKE_TEST_STEPS} steps, report memory")
    parser.add_argument("--max-steps", type=int, default=config.MAX_STEPS, help="total training steps")
    parser.add_argument("--save-every", type=int, default=config.SAVE_STEPS,
                        help="checkpoint + eval interval in steps")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("No CUDA GPU visible to PyTorch - see README 'Setup'. (fp16 training needs a GPU.)")
    if not config.PROCESSED_DATASET_DIR.exists():
        sys.exit(f"{config.PROCESSED_DATASET_DIR} not found - run `python -m src.prepare_data` first.")

    # --- data ----------------------------------------------------------------
    dataset = load_from_disk(str(config.PROCESSED_DATASET_DIR))
    train_set, eval_set = dataset["train"], dataset["test"]
    clips_per_step = config.PER_DEVICE_TRAIN_BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS
    if args.smoke_test:
        # Longest clips first = the most memory any real batch could need.
        n = config.SMOKE_TEST_STEPS * clips_per_step
        train_set = train_set.sort("num_frames", reverse=True).select(range(min(n, len(train_set))))
        eval_set = None

    # --- model -----------------------------------------------------------------
    processor = SpeechT5Processor.from_pretrained(config.BASE_MODEL)
    model = SpeechT5ForTextToSpeech.from_pretrained(config.BASE_MODEL)
    model.config.use_cache = False  # caching is for generation; it conflicts with gradient checkpointing

    run_name = "smoke-test" if args.smoke_test else config.RUN_NAME
    epochs = args.max_steps * clips_per_step / len(dataset["train"])
    print(f"Run: {run_name} | train clips: {len(dataset['train'])} | test clips: {len(dataset['test'])}")
    if not args.smoke_test:
        print(f"{args.max_steps} steps x {clips_per_step} clips = ~{epochs:.0f} passes over the training set; "
              f"checkpoint + eval every {args.save_every} steps")

    trainer = Seq2SeqTrainer(
        model=model,
        args=build_training_args(args.smoke_test, args.max_steps, args.save_every),
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=TTSDataCollatorWithPadding(processor, model.config.reduction_factor),
        processing_class=processor,  # saved alongside each checkpoint
        callbacks=[ProgressCallback(config.PROGRESS_FILE, run_name, pause_file=config.PAUSE_FILE, extra_info={
            "train_clips": len(dataset["train"]),
            "batch_size": config.PER_DEVICE_TRAIN_BATCH_SIZE,
            "gradient_accumulation_steps": config.GRADIENT_ACCUMULATION_STEPS,
        })],
    )

    # --- resume if a checkpoint exists ---------------------------------------------
    last_checkpoint = None
    if not args.smoke_test and config.CHECKPOINT_DIR.exists():
        last_checkpoint = get_last_checkpoint(str(config.CHECKPOINT_DIR))
        if last_checkpoint:
            print(f"Resuming from {last_checkpoint}")

    torch.cuda.reset_peak_memory_stats()
    try:
        result = trainer.train(resume_from_checkpoint=last_checkpoint)
    except torch.cuda.OutOfMemoryError:
        b, a = config.PER_DEVICE_TRAIN_BATCH_SIZE, config.GRADIENT_ACCUMULATION_STEPS
        sys.exit(
            "\nOUT OF GPU MEMORY. In config.py, halve the batch and double accumulation "
            f"(same effective batch of {b * a}):\n"
            f"    PER_DEVICE_TRAIN_BATCH_SIZE = {max(b // 2, 1)}\n"
            f"    GRADIENT_ACCUMULATION_STEPS = {a * 2}\n"
            "Also close other GPU-heavy apps (browsers, games), then run the smoke test again."
        )

    if args.smoke_test:
        report_smoke_test(result)
        return

    step = trainer.state.global_step
    if step < args.max_steps:  # stopped by the dashboard's Pause button
        print(f"\nPaused at step {step}/{args.max_steps}. Checkpoint saved in {config.CHECKPOINT_DIR}.\n"
              f"Resume with the same command:  python -m src.train {' '.join(sys.argv[1:])}")
        return

    # --- save the final (best) model + the voice it should speak with ---------------
    trainer.save_model(str(config.FINAL_MODEL_DIR))
    processor.save_pretrained(str(config.FINAL_MODEL_DIR))
    shutil.copy(config.SPEAKER_EMBEDDING_PATH, config.FINAL_MODEL_DIR / "speaker_embedding.npy")
    print(f"\nSaved final model -> {config.FINAL_MODEL_DIR}")

    if config.PUSH_TO_HUB:
        push_to_hub(model, processor)


def push_to_hub(model, processor):
    """Optional. Needs `huggingface-cli login` (or HF_TOKEN). Check the IndicTTS license first!"""
    from huggingface_hub import HfApi

    print(f"Pushing to https://huggingface.co/{config.HUB_MODEL_ID} ...")
    model.push_to_hub(config.HUB_MODEL_ID)
    processor.push_to_hub(config.HUB_MODEL_ID)
    HfApi().upload_file(
        path_or_fileobj=str(config.FINAL_MODEL_DIR / "speaker_embedding.npy"),
        path_in_repo="speaker_embedding.npy",
        repo_id=config.HUB_MODEL_ID,
    )


if __name__ == "__main__":
    main()
