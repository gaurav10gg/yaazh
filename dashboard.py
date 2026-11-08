"""
dashboard.py - a live view of training progress.

Run it in a SECOND terminal while training runs:

    python dashboard.py          -> open http://127.0.0.1:7861

It reads outputs/progress.json (written by src/progress_callback.py) every
5 seconds. It never touches the training process, so you can open, close or
restart it at any time.

The "Pause training safely" button only creates a small marker file
(config.PAUSE_FILE). Training notices it after the current step, saves a
checkpoint and exits. Run the same train command again to resume.
"""

import json
from datetime import datetime, timedelta

import gradio as gr
import pandas as pd

import config

REFRESH_SECONDS = 5
STALE_AFTER_MINUTES = 10   # no update for this long while "running" = probably crashed/stopped

# Validated categorical palette, slots 1 and 2.
SERIES_COLORS = {"train loss": "#2a78d6", "eval loss": "#eb6834"}

CSS = """
.yz-bar { height: 14px; border-radius: 7px; background: var(--border-color-primary); overflow: hidden; }
.yz-bar > div { height: 100%; border-radius: 7px; background: #2a78d6; }
.yz-head { display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap;
           gap: 8px; margin-bottom: 8px; }
.yz-big { font-size: 1.6rem; font-weight: 600; color: var(--body-text-color); }
.yz-muted { color: var(--body-text-color-subdued); font-size: 0.9rem; }
.yz-tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px;
            margin-top: 16px; }
.yz-tile { border: 1px solid var(--border-color-primary); border-radius: 8px; padding: 10px 12px;
           background: var(--block-background-fill); }
.yz-tile .label { color: var(--body-text-color-subdued); font-size: 0.8rem; }
.yz-tile .value { color: var(--body-text-color); font-size: 1.25rem; font-weight: 600; margin-top: 2px; }
.yz-note { margin-top: 12px; padding: 8px 12px; border-radius: 8px;
           border: 1px solid var(--border-color-primary); color: var(--body-text-color); }
"""


def read_progress() -> dict | None:
    try:
        return json.loads(config.PROGRESS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, PermissionError):
        return None  # not started yet, or caught mid-write (next tick will work)


def format_duration(seconds) -> str:
    if seconds is None:
        return "-"
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes = rest // 60
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m {seconds % 60:02d}s"


def status_text(progress: dict) -> tuple[str, str | None]:
    """(status label, optional warning) - status is always spelled out, never colour alone."""
    status = progress.get("status", "unknown")
    updated = datetime.fromisoformat(progress["updated_at"])
    minutes_quiet = (datetime.now() - updated).total_seconds() / 60
    if status in ("running", "pausing") and minutes_quiet > STALE_AFTER_MINUTES:
        return ("⚠ Not responding",
                f"No update for {minutes_quiet:.0f} minutes - training may have crashed or been "
                "stopped. Run the train command again to resume from the latest checkpoint.")
    if status == "paused":
        return ("⏸ Paused",
                f"Paused safely at step {progress.get('step')} ({progress.get('latest_checkpoint')} saved). "
                "Run the same train command again to resume - nothing is lost.")
    labels = {"running": "● Running", "pausing": "⏸ Pausing (saving checkpoint)…",
              "finished": "✓ Finished", "stopped": "■ Stopped early"}
    return labels.get(status, status), None


def request_pause() -> str:
    """Create the marker file that tells training to save a checkpoint and stop."""
    progress = read_progress()
    if not progress or progress.get("status") != "running":
        return "No training is running right now."
    config.PAUSE_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.PAUSE_FILE.touch()
    return ("⏸ Pause requested. Training will save a checkpoint after the current step "
            "(a few seconds) and stop. Resume any time with the same train command.")


def tile(label: str, value: str) -> str:
    return f'<div class="yz-tile"><div class="label">{label}</div><div class="value">{value}</div></div>'


def summary_html(progress: dict | None) -> str:
    if progress is None:
        return ('<div class="yz-note">No training run yet. Start one with '
                '<code>python -m src.train --smoke-test</code> (quick) or '
                '<code>python -m src.train</code>.</div>')

    step, max_steps = progress.get("step", 0), progress.get("max_steps") or 1
    percent = 100 * step / max_steps
    label, warning = status_text(progress)

    eta = progress.get("eta_seconds")
    finish_at = (datetime.now() + timedelta(seconds=eta)).strftime("%a %H:%M") if eta else "-"
    history = progress.get("history", [])
    train_losses = [h["loss"] for h in history if "loss" in h]
    peak, total = progress.get("peak_gpu_memory_gb"), progress.get("gpu_total_gb")

    html = f"""
    <div class="yz-head">
      <div><span class="yz-big">Step {step:,} / {max_steps:,}</span>
           <span class="yz-muted">&nbsp;({percent:.1f}%)</span></div>
      <div class="yz-muted">{label} · run <b>{progress.get('run_name')}</b> · updated {progress['updated_at'][11:]}</div>
    </div>
    <div class="yz-bar" role="progressbar" aria-valuenow="{percent:.0f}" aria-valuemin="0" aria-valuemax="100">
      <div style="width:{percent:.2f}%"></div>
    </div>
    <div class="yz-tiles">
      {tile("Elapsed", format_duration(progress.get("elapsed_seconds")))}
      {tile("Time left", format_duration(eta))}
      {tile("Expected finish", finish_at)}
      {tile("Latest train loss", f"{train_losses[-1]:.4f}" if train_losses else "-")}
      {tile("Best eval loss", f"{progress['best_eval_loss']:.4f}" if progress.get("best_eval_loss") else "-")}
      {tile("Peak GPU memory", f"{peak:.1f} / {total:.1f} GB" if peak else "-")}
      {tile("Speed", f"{progress['seconds_per_step']:.1f} s/step" if progress.get("seconds_per_step") else "-")}
      {tile("Latest checkpoint", progress.get("latest_checkpoint") or "-")}
    </div>
    """
    if warning:
        html += f'<div class="yz-note">{warning}</div>'
    return html


def loss_frame(progress: dict | None) -> pd.DataFrame:
    """Long format (step, loss, series) - one row per point, as the line plot expects."""
    rows = []
    for entry in (progress or {}).get("history", []):
        if "loss" in entry:
            rows.append({"step": entry["step"], "loss": entry["loss"], "series": "train loss"})
        if "eval_loss" in entry:
            rows.append({"step": entry["step"], "loss": entry["eval_loss"], "series": "eval loss"})
    return pd.DataFrame(rows, columns=["step", "loss", "series"])


def recent_table(progress: dict | None) -> pd.DataFrame:
    history = (progress or {}).get("history", [])[-12:][::-1]  # newest first
    return pd.DataFrame(
        [{"step": h["step"], "train loss": h.get("loss"), "eval loss": h.get("eval_loss"),
          "learning rate": f"{h['learning_rate']:.2e}" if h.get("learning_rate") else None}
         for h in history],
        columns=["step", "train loss", "eval loss", "learning rate"],
    )


def refresh():
    progress = read_progress()
    return summary_html(progress), loss_frame(progress), recent_table(progress)


with gr.Blocks(title="Yaazh training", css=CSS) as dashboard:
    gr.Markdown("# Yaazh training dashboard\nRefreshes every 5 seconds. "
                "Loss going **down** = the model is learning.")
    summary = gr.HTML()
    with gr.Row():
        pause_button = gr.Button("⏸ Pause training safely", variant="secondary", scale=0, min_width=240)
        pause_message = gr.Markdown()
    pause_button.click(request_pause, outputs=pause_message)
    loss_plot = gr.LinePlot(
        x="step", y="loss", color="series", color_map=SERIES_COLORS,
        title="Loss (lower is better)", x_title="Training step", y_title="Loss",
        tooltip="all", height=320,
    )
    gr.Markdown("**Recent log entries** (newest first)")
    table = gr.Dataframe(interactive=False)

    timer = gr.Timer(REFRESH_SECONDS)
    timer.tick(refresh, outputs=[summary, loss_plot, table])
    dashboard.load(refresh, outputs=[summary, loss_plot, table])


if __name__ == "__main__":
    dashboard.launch(server_port=7861)
