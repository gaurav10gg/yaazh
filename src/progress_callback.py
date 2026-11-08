"""
progress_callback.py - lets the dashboard see (and safely pause) training.

The Hugging Face Trainer calls "callback" methods at fixed moments
(training started, a step finished, loss was logged, a checkpoint was saved...).
This callback writes the latest numbers to a small JSON file (config.PROGRESS_FILE).
dashboard.py only READS that file, so watching training can never slow it down
or crash it.

Pausing: the dashboard's Pause button creates config.PAUSE_FILE. After every
step we check for it; if it's there we ask the Trainer to save a checkpoint
right now and stop. Running the train command again resumes from that step.
"""

import json
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import torch
from transformers import TrainerCallback


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class ProgressCallback(TrainerCallback):
    def __init__(self, progress_file: Path, run_name: str, pause_file: Path | None = None,
                 extra_info: dict | None = None):
        self.path = Path(progress_file)
        self.run_name = run_name
        self.pause_file = Path(pause_file) if pause_file else None
        self.extra_info = extra_info or {}   # e.g. batch size, dataset size - shown on the dashboard
        self.data: dict = {}

    # --- file helpers -------------------------------------------------------

    def _read_previous(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _write(self):
        """Write to a temp file, then swap it in, so the dashboard never reads half a file."""
        self.data["updated_at"] = _now()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1), encoding="utf-8")
        for _ in range(5):  # on Windows the swap fails if the dashboard is reading at that instant
            try:
                os.replace(tmp, self.path)
                return
            except PermissionError:
                time.sleep(0.1)

    def _update_gpu_memory(self):
        if torch.cuda.is_available():
            self.data["peak_gpu_memory_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 2)
            self.data["gpu_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)

    # --- Trainer hooks --------------------------------------------------------

    def on_train_begin(self, args, state, control, **kwargs):
        if self.pause_file:
            self.pause_file.unlink(missing_ok=True)  # ignore a stale request from an earlier run
        previous = self._read_previous()
        resuming = previous.get("run_name") == self.run_name and state.global_step > 0
        history = previous.get("history", []) if resuming else []
        # After resuming from checkpoint-N, forget anything logged after step N.
        history = [entry for entry in history if entry["step"] <= state.global_step]

        self._session_start_time = time.time()
        # (time, step) of the last 20 steps: speed/ETA use only recent steps, so a
        # laptop sleep or a slow patch doesn't distort the estimate for the rest of the run.
        self._recent_steps = deque([(self._session_start_time, state.global_step)], maxlen=21)
        self._elapsed_before = previous.get("elapsed_seconds", 0) if resuming else 0

        self.data = {
            "run_name": self.run_name,
            "status": "running",
            "step": state.global_step,
            "max_steps": state.max_steps,
            "started_at": previous.get("started_at", _now()) if resuming else _now(),
            "resumed_from_step": state.global_step if resuming else None,
            "elapsed_seconds": self._elapsed_before,
            "eta_seconds": None,
            "seconds_per_step": None,
            "latest_checkpoint": previous.get("latest_checkpoint") if resuming else None,
            "best_eval_loss": previous.get("best_eval_loss") if resuming else None,
            "history": history,
            **self.extra_info,
        }
        self._update_gpu_memory()
        self._write()

    def on_step_end(self, args, state, control, **kwargs):
        now = time.time()
        self._recent_steps.append((now, state.global_step))
        self.data["step"] = state.global_step
        self.data["elapsed_seconds"] = round(self._elapsed_before + now - self._session_start_time)

        (first_time, first_step), (last_time, last_step) = self._recent_steps[0], self._recent_steps[-1]
        if last_step > first_step:
            per_step = (last_time - first_time) / (last_step - first_step)
            self.data["seconds_per_step"] = round(per_step, 2)
            self.data["eta_seconds"] = round(per_step * (state.max_steps - state.global_step))
        self._update_gpu_memory()

        if self.pause_file and self.pause_file.exists():
            self.pause_file.unlink(missing_ok=True)
            control.should_save = True            # write checkpoint-<this step> now...
            control.should_training_stop = True   # ...then leave the training loop
            self.data["status"] = "pausing"
        self._write()

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        entry = {"step": state.global_step}
        if "loss" in logs:
            entry["loss"] = round(logs["loss"], 4)
            entry["learning_rate"] = logs.get("learning_rate")
        if "eval_loss" in logs:
            entry["eval_loss"] = round(logs["eval_loss"], 4)
            best = self.data.get("best_eval_loss")
            if best is None or logs["eval_loss"] < best:
                self.data["best_eval_loss"] = round(logs["eval_loss"], 4)
        if len(entry) > 1:
            self.data["history"].append(entry)
            self._write()

    def on_save(self, args, state, control, **kwargs):
        self.data["latest_checkpoint"] = f"checkpoint-{state.global_step}"
        self._write()

    def on_train_end(self, args, state, control, **kwargs):
        self.data["status"] = "finished" if state.global_step >= state.max_steps else "paused"
        self.data["eta_seconds"] = 0 if self.data["status"] == "finished" else None
        self._write()
