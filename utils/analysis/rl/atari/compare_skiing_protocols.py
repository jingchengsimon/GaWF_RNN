"""Compare old staged Skiing histories with a completed 4M run and optional BF16 audits.

Explicit figure manifests locate histories; optional audit roots contain model summary/NPZ pairs.
Outputs are comparison PNGs, numeric curves NPZ, metrics JSON and provenance metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.analysis.rl.atari.atari_5task_raw_learning_curves import (
    MODEL_LABELS, curve_for_task, load_run_histories,
)


def audit_metrics(root: Path) -> dict:
    """Measure exact top ties both across 18 heads and distinct legal-action groups."""
    result = {}
    for model in ("lstm", "gru", "gawf"):
        summary = json.loads((root / model / "summary.json").read_text())
        trace = root / model / "step_trace.npz"
        assert hashlib.sha256(trace.read_bytes()).hexdigest() == summary["trace_sha256"]
        # The original evaluator snapshot omits AMP metadata; its launcher uses BF16.
        assert summary.get("amp_dtype") in (None, "bfloat16")
        with np.load(trace) as d:
            stored = d["q_values"]
            assert stored.dtype == np.float32
            assert ((stored.view(np.uint32) & 65535) == 0).all()
            q = d["q_values"].astype(np.float64)
            assert q.shape[1] == 18 and np.isfinite(q).all()
            mapping = np.asarray(summary["canonical_action_mapping"])
            legal_q = np.stack([q[:, mapping == a].max(1) for a in range(9)], axis=1)
            top = q.max(1, keepdims=True)
            gaps = np.sort(q, axis=1)[:, -1] - np.sort(q, axis=1)[:, -2]
            legal_gaps = np.sort(legal_q, axis=1)[:, -1] - np.sort(legal_q, axis=1)[:, -2]
            result[model] = dict(
                steps=len(q), top_tie_rate=float(((q == top).sum(1) > 1).mean()),
                all18_equal_rate=float((q.max(1) == q.min(1)).mean()),
                legal_top_tie_rate=float((legal_gaps == 0).mean()),
                median_top_margin=float(np.median(gaps)),
                median_legal_margin=float(np.median(legal_gaps)),
                median_abs_q=float(np.median(np.abs(q))),
                min_q=float(q.min()), max_q=float(q.max()),
                trace_path=str(trace), trace_sha256=summary["trace_sha256"],
                summary_amp_dtype=summary.get("amp_dtype"),
                bf16_representable_fraction=1.0,
                canonical_action_counts=np.bincount(d["action"], minlength=18).tolist(),
                legal_action_counts=np.bincount(d["legal_action"], minlength=9).tolist(),
                stall_rate=summary["stall_rate"], mean_return=summary["mean_return"],
                eval_seed=summary["eval_seed"], num_episodes=summary["num_episodes"],
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-inputs", type=Path, required=True)
    parser.add_argument("--new-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--old-audit", type=Path)
    parser.add_argument("--new-audit", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    old = json.loads(args.old_inputs.read_text())
    new = json.loads(args.new_manifest.read_text())
    models = ["lstm", "gru", "gawf"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
    arrays, stats, source_paths = {}, {}, []
    for model, axis in zip(models, axes):
        stats[model] = {}
        for protocol, color in [("old", "#777777"), ("new", "#0072B2")]:
            if protocol == "old":
                selected = [(Path(d["run_directory"]), d.get("step_offset", 0))
                            for d in old if d["model"] == model]
            else:
                selected = [(Path(d["path"]).parent, 0) for d in new["sources"]
                            if Path(d["path"]).name == "metrics_history.jsonl"
                            and f"_{model}_seed1" in d["path"]]
            xs, ys = [], []
            for i, (path, offset) in enumerate(selected):
                run = load_run_histories([path], step_offsets=[offset])[0]
                x, y = curve_for_task(run, "Skiing", "environment_steps")
                xs.extend(x); ys.extend(y)
                axis.plot(x / 1e6, y, color=color, linewidth=1.2,
                          label=("old: clipped, gamma=.99" if protocol == "old" else
                                 "new: unclipped, gamma=.999") if i == 0 else None)
                source_paths.append(str(path / "metrics_history.jsonl"))
            assert xs
            arrays[f"{model}_{protocol}_steps"] = np.asarray(xs, dtype=np.int64)
            arrays[f"{model}_{protocol}_return100"] = np.asarray(ys, dtype=np.float32)
            stats[model][protocol] = dict(last_return100=ys[-1],
                                         mean_return100_last_500k=float(np.mean(
                                             np.asarray(ys)[np.asarray(xs) >= 3500000])))
        axis.set_title(MODEL_LABELS[model])
        axis.set_xlim(0, 4); axis.set_xlabel("Environment steps (M)")
        axis.grid(alpha=.2); axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Episode return (last 100 episodes)")
    axes[1].legend(fontsize=9, loc="lower right")
    fig.suptitle("Skiing seed 1 · training curves (not greedy evaluation)\n"
                 "Old: staged weights-only extensions; new: one continuous 4M phase", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.output_dir / "01_skiing_protocol_comparison.png", dpi=160)
    plt.close(fig)
    np.savez_compressed(args.output_dir / "curves.npz", **arrays)
    report = dict(training=stats, source_histories=source_paths,
                  caveat="Not a strict clip/gamma-only ablation: training phase/resume schedules differ.")
    if args.old_audit and args.new_audit:
        report["bf16_audits"] = dict(old=audit_metrics(args.old_audit),
                                     new=audit_metrics(args.new_audit))
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True)
        fields = [("top_tie_rate", "Top-Q tied (18 heads)"),
                  ("legal_top_tie_rate", "Top-Q tied across legal groups"),
                  ("all18_equal_rate", "All 18 Q-values equal")]
        for axis, (field, title) in zip(axes, fields):
            x = np.arange(3)
            for protocol, shift, color in [("old", -.18, "#777777"),
                                            ("new", .18, "#0072B2")]:
                values = [100 * report["bf16_audits"][protocol][m][field] for m in models]
                bars = axis.bar(x + shift, values, width=.36, color=color, label=protocol)
                axis.bar_label(bars, fmt="%.1f", fontsize=8)
            axis.set_xticks(x, [MODEL_LABELS[m] for m in models])
            axis.set_title(title, fontsize=10)
            axis.set_ylim(0, 112)
            axis.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("Fraction of greedy steps (%)")
        axes[2].legend()
        fig.suptitle("Final 4M checkpoints · BF16 · 20 greedy episodes · eval seed 20260904")
        fig.tight_layout()
        fig.savefig(args.output_dir / "02_skiing_bf16_q_ties.png", dpi=160)
        plt.close(fig)
    (args.output_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
