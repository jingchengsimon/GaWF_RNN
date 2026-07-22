#!/usr/bin/env python3
"""Generate learning curves for completed MiniGrid paper-aligned PPO experiments.

Generates separate plots for RedBlueDoors and MemoryS7, comparing all
completed models (paper_lstm, lstm_core, s5, mamba) with rolling mean over 100 logs.
Reference style: results/train_figs/rl/minigrid/redblue_l1.png

Data source: amarel remote at /cache/home/js3269/projects/FAW_RNN-minigrid-paper-20260715/results/train_data/
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import subprocess
import os

# Configuration
RESULTS_PATH = Path(__file__).parent.parent / "results/train_data"
OUTPUT_DIR = Path(__file__).parent.parent / "results/train_figs/rl/minigrid"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Remote amarel path
REMOTE_HOST = "amarel"
REMOTE_BASE = "/cache/home/js3269/projects/FAW_RNN-minigrid-paper-20260715/results/train_data"

# Models in order
MODEL_ORDER = ["paper_lstm", "lstm_core", "rnn", "gru", "gawf", "s5", "mamba"]
MODEL_COLORS = {
    "paper_lstm": "#1f77b4",    # blue
    "lstm_core": "#4C72B0",     # medium blue
    "rnn": "#FF9999",           # light red
    "gru": "#FF6666",           # medium red
    "gawf": "#DD0000",          # bright red
    "s5": "#FFA500",            # orange
    "mamba": "#00AA00",         # green
}

def load_metrics_history(path: Path, remote_path: str = None) -> dict:
    """Load metrics from metrics_history.jsonl (local or remote via SSH)."""
    data = defaultdict(list)

    # Try remote first if provided
    if remote_path:
        try:
            cmd = f"ssh -o BatchMode=yes {REMOTE_HOST} 'cat {remote_path}'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        try:
                            record = json.loads(line)
                            for key, val in record.items():
                                data[key].append(val)
                        except json.JSONDecodeError:
                            continue
                return dict(data)
        except Exception as e:
            print(f"Remote fetch failed ({remote_path}): {e}")

    # Fall back to local
    if path.exists():
        try:
            with open(path) as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        for key, val in record.items():
                            data[key].append(val)
        except (json.JSONDecodeError, IOError):
            pass

    return dict(data)

def rolling_mean(arr, window=100):
    """Compute rolling mean with padding."""
    if len(arr) < window:
        return np.array(arr)

    smoothed = np.convolve(arr, np.ones(window) / window, mode='valid')
    # Pad to original length by repeating endpoints
    pad_left = np.full(window // 2, smoothed[0])
    pad_right = np.full(len(arr) - len(smoothed) - window // 2, smoothed[-1])
    return np.concatenate([pad_left, smoothed, pad_right])

def plot_environment(env_name: str, env_prefix: str):
    """Generate learning curve plot for one environment."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"MiniGrid {env_name} learning curves (rolling mean: 100 logs)",
        fontsize=13, fontweight="normal"
    )

    completed_models = []

    # Collect data for completed models (from remote amarel)
    for model in MODEL_ORDER:
        result_dir = RESULTS_PATH / f"mg_ppo_paper_{env_prefix}_fov3_{model}_seed42_100m"
        metrics_file = result_dir / "metrics_history.jsonl"
        remote_metrics = f"{REMOTE_BASE}/mg_ppo_paper_{env_prefix}_fov3_{model}_seed42_100m/metrics_history.jsonl"
        remote_json = f"{REMOTE_BASE}/mg_ppo_paper_{env_prefix}_fov3_{model}_seed42_100m/metrics.json"

        # Check if exists on remote
        check_cmd = f"ssh -o BatchMode=yes {REMOTE_HOST} 'test -f {remote_json}'"
        if subprocess.run(check_cmd, shell=True, capture_output=True).returncode != 0:
            continue  # Skip incomplete runs

        data = load_metrics_history(metrics_file, remote_path=remote_metrics)
        if not data or 'global_step' not in data:
            continue

        completed_models.append((model, data))

    if not completed_models:
        print(f"No completed models found for {env_name}")
        return

    # Plot success rate and episodic return
    for idx, ax in enumerate(axes):
        ax.set_xlabel("Environment steps (millions)", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 200)

        if idx == 0:
            ax.set_ylabel("Success rate", fontsize=11)
            metric_key = "success_rate"
            ax.set_ylim(0, 1.0)
        else:
            ax.set_ylabel("Episodic return (last 100)", fontsize=11)
            metric_key = "episodic_return_100"
            ax.set_ylim(0, 1.0)

        for model, data in completed_models:
            if metric_key not in data:
                continue

            steps = np.array(data['global_step']) / 1e6
            values = np.array(data[metric_key])

            # Rolling mean
            smoothed = rolling_mean(values, window=100)

            # Confidence band (std over 100-log window)
            std_values = []
            for i in range(len(values)):
                window_start = max(0, i - 50)
                window_end = min(len(values), i + 50)
                std_values.append(np.std(values[window_start:window_end]))
            std_smoothed = rolling_mean(std_values, window=100)

            color = MODEL_COLORS.get(model, "#888888")
            ax.plot(steps, smoothed, color=color, label=model, linewidth=1.5, alpha=0.85)
            ax.fill_between(
                steps,
                smoothed - std_smoothed,
                smoothed + std_smoothed,
                color=color,
                alpha=0.15
            )

    # Create unified legend
    handles = []
    labels = []
    for model, data in completed_models:
        color = MODEL_COLORS.get(model, "#888888")
        handles.append(plt.Line2D([0], [0], color=color, lw=1.5))
        labels.append(model)

    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(labels),
        frameon=False,
        fontsize=10
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save
    output_file = OUTPUT_DIR / f"minigrid_ppo_paper_{env_name.lower().replace('-', '_').replace(' ', '_')}_learning_curves.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()

if __name__ == "__main__":
    plot_environment("RedBlueDoors-8x8", "RedBlueDoors-8x8")
    plot_environment("MemoryS7", "MemoryS7")
    print("\nNote: GaWF, GRU, paper_lstm, and RNN runs are still in progress.")
    print("Re-run this script after recovery jobs (58873264/65/66) complete.")
