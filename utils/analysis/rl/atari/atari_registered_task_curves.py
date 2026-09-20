"""Plot registered Seaquest/Skiing histories from a local, timestamped result snapshot.

Reads exact manifest units and JSONL histories; writes PNG, numeric NPZ, summary CSV and
provenance JSON. Missing histories remain explicitly listed as not started, never as zero returns.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.analysis.rl.atari.atari_5task_raw_learning_curves import (
    MODEL_COLORS, MODEL_LABELS, curve_for_task, load_run_histories,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", action="append", required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", choices=["Seaquest", "Skiing"], required=True)
    parser.add_argument("--skiing-acdef", action="store_true",
                        help="Plot Skiing A/C/D/E/F variants in three model panels.")
    parser.add_argument("--seaquest-ac-multiseed", action="store_true",
                        help="Plot A/C in four panels with seed curves and common-support mean.")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[4]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = ["lstm", "gawf"] if args.task == "Seaquest" else ["lstm", "gru", "gawf"]
    assert not args.skiing_acdef or args.task == "Skiing"
    panel_models = args.task == "Seaquest" or args.skiing_acdef
    multiseed = args.seaquest_ac_multiseed
    assert not multiseed or args.task == "Seaquest"
    groups = {}
    panels = 4 if multiseed else len(models) if panel_models else 1
    fig, axes = plt.subplots(2 if multiseed else 1, 2 if multiseed else panels,
                             figsize=(12, 9) if multiseed else (6 * panels, 5) if panel_models else (8, 4.8),
                             squeeze=False, sharey=True)
    axes = axes.reshape(1, -1)
    variants = {"a": "#1f77b4", "b": "#ff7f0e", "c": "#2ca02c",
                "f995": "#9467bd", "f999": "#8c564b",
                "d": "#ff7f0e", "e": "#9467bd", "f": "#8c564b"}
    rows, sources, arrays = [], [], {}
    for experiment_id in args.experiment_id:
        manifest_path = project / "experiments/monitoring/jobs" / (experiment_id + ".json")
        manifest = json.loads(manifest_path.read_text())
        for unit in manifest["tracking"]["units"]:
            remote = unit["result_dir"]
            relative = remote.split("/results/", 1)[1]
            local = args.snapshot_root / manifest["host"] / relative
            model = unit["expected"]["model_type"]
            target = int(unit["expected"]["global_step"])
            variant = local.name.split("_")[1] if panel_models else "unclipped"
            history = local / "metrics_history.jsonl"
            metrics = local / "metrics.json"
            final = json.loads(metrics.read_text()) if metrics.exists() else {}
            completed = int(final.get("global_step", 0)) >= target
            row = dict(experiment_id=experiment_id, unit=unit["id"], model=model,
                       variant=variant, target_steps=target, latest_steps=0,
                       status="not_started", return_100=None)
            if history.exists():
                run = load_run_histories([local])[0]
                x, y = curve_for_task(run, args.task, "environment_steps")
                if not len(x):
                    raise RuntimeError(f"No finite {args.task} returns: {history}")
                assert np.all(np.diff(x) >= 0) and x[-1] <= target
                row.update(latest_steps=int(run.history[-1]["global_step"]),
                           status="completed" if completed else "in_progress",
                           return_100=float(y[-1]))
                key = unit["id"].replace("-", "_")
                arrays[key + "_steps"] = x.astype(np.int64)
                arrays[key + "_return_100"] = y.astype(np.float32)
                panel = models.index(model) if panel_models else 0
                seed = int(local.name.rsplit("seed", 1)[1]) if multiseed else None
                if multiseed:
                    panel = models.index(model) * 2 + ["a", "c"].index(variant)
                    groups.setdefault((model, variant), []).append((x, y))
                axis = axes[0, panel]
                name = variant.upper() if panel_models else MODEL_LABELS[model]
                color = variants[variant] if panel_models else MODEL_COLORS[model]
                if multiseed:
                    name, color = f"seed {seed}", f"C{seed - 1}"
                suffix = "done" if completed else f"{x[-1] / 1e6:.3f}M / running"
                axis.plot(x / 1e6, y, color=color, linewidth=1.4, label=f"{name} ({suffix})")
                if not completed:
                    axis.scatter(x[-1] / 1e6, y[-1], color=color, s=20, zorder=3)
                for source in [history, metrics]:
                    if source.exists():
                        sources.append(dict(path=str(source.resolve()),
                                            sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
            rows.append(row)
    if multiseed:
        for (model, variant), curves in groups.items():
            if len(curves) != 4:
                continue
            lower = max(x[0] for x, _ in curves)
            upper = min(x[-1] for x, _ in curves)
            if lower > upper:
                continue
            grid = np.unique(np.concatenate([x[(x >= lower) & (x <= upper)]
                                             for x, _ in curves]))
            values = np.stack([np.interp(grid, x, y) for x, y in curves])
            mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
            prefix = f"{model}_{variant}_4seed"
            arrays.update({prefix + "_steps": grid.astype(np.int64),
                           prefix + "_mean": mean.astype(np.float32),
                           prefix + "_sample_sd": sd.astype(np.float32)})
            axis = axes[0, models.index(model) * 2 + ["a", "c"].index(variant)]
            axis.plot(grid / 1e6, mean, color="black", linewidth=2.2, label="4-seed mean")
    for i, axis in enumerate(axes[0]):
        axis.set_title(
            f"{MODEL_LABELS[models[i // 2]]} · { ['A', 'C'][i % 2]}" if multiseed
            else MODEL_LABELS[models[i]] if panel_models else "LSTM / GRU / GaWF")
        axis.set_xlim(0, 3 if args.task == "Seaquest" else 4)
        axis.set_xlabel("Environment steps (M)")
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(fontsize=8, loc="best", framealpha=0.85)
        if panel_models and not multiseed:
            missing = [r["variant"].upper() for r in rows
                       if r["model"] == models[i] and r["status"] == "not_started"]
            if missing:
                axis.text(0.02, 0.98, "Not started: " + ", ".join(missing),
                          transform=axis.transAxes, va="top", fontsize=9)
    axes[0, 0].set_ylabel("Episode return (last 100 episodes)")
    protocol = "seed 2 · A/B/C/F995/F999" if args.task == "Seaquest" else (
        "seed 1 · BF16 · unclipped reward · gamma=0.999")
    if multiseed:
        protocol = "seeds 1–4 · A/C · BF16 · mean on common seed coverage"
        axes[0, 2].set_ylabel("Episode return (last 100 episodes)")
    if args.skiing_acdef:
        protocol += " · A/C/D/E/F"
    fig.suptitle(f"{args.task} · {protocol}\nSnapshot {args.snapshot_root.name}", fontsize=11)
    fig.tight_layout()
    png = f"01_{args.task.lower()}_learning_curves_environment_steps.png"
    fig.savefig(args.output_dir / png, dpi=160)
    plt.close(fig)
    np.savez_compressed(args.output_dir / "curves.npz", **arrays)
    with (args.output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    provenance = dict(script_path=str(Path(__file__).relative_to(project)),
                      experiment_ids=args.experiment_id, snapshot=str(args.snapshot_root.resolve()),
                      rows=rows, sources=sources, files_written=[png, "curves.npz", "summary.csv"])
    (args.output_dir / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
