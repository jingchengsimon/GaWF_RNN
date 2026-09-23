"""Plan and run the GaWF legacy no-tanh analysis refresh without overwriting formal outputs.

The launcher reuses existing analysis module CLIs.  Per-seed collection is GPU-capable and may
be dispatched independently; aggregation and the Markdown record run only after all ten seed
markers exist.  Outputs are isolated below caller-supplied data and figure roots.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


MODEL = "gawf_legacy_notanh"
SEEDS = tuple(range(1, 11))
EXPECTED_METRICS = {
    "model_type": MODEL,
    "actual_epochs": 150,
    "core_rnn_activation": "identity",
    "core_output_wrap": "ln_relu_dropout",
    "gawf_core_semantics": "legacy",
}
DEFERRED = (
    "six-model behavior and target-switch figures",
    "six-model activation ANOVA figure",
    "GaWF/LSTM/GRU unit-gate comparison row",
    "cross-scale and H128 comparison figures",
)
STATIC = ("GaWF/CM-MNIST schematic", "dataset-generation protocol")


@dataclass(frozen=True)
class Config:
    """Resolved paths and execution settings for one refresh."""

    checkpoint_root: Path
    data_dir: Path
    output_root: Path
    figure_root: Path
    record_output: Path
    python: str
    device: str


@dataclass(frozen=True)
class Step:
    """One existing module invocation and its primary expected output."""

    name: str
    command: tuple[str, ...]
    expected: Path


def _module(config: Config, name: str, *arguments: object) -> tuple[str, ...]:
    """Return one unambiguous Python module command."""

    return (config.python, "-B", "-m", name, *(str(value) for value in arguments))


def _seed_tag(seed: int) -> str:
    """Return a validated two-digit training seed."""

    if seed not in SEEDS:
        raise ValueError(f"seed must be in {SEEDS}, got {seed}")
    return f"{seed:02d}"


def _seed_artifacts(config: Config, seed: int) -> dict[str, Path]:
    """Resolve exactly one model, metrics, and history file for one completed seed."""

    tag = _seed_tag(seed)
    leaf = config.checkpoint_root / f"{MODEL}-seed{tag}"
    patterns = {"model": "*_model.pth", "metrics": "*_metrics.json", "history": "*.pkl"}
    artifacts: dict[str, Path] = {}
    for name, pattern in patterns.items():
        matches = sorted(leaf.glob(pattern))
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one {name} for seed {tag} in {leaf}, found {len(matches)}"
            )
        artifacts[name] = matches[0]
    metrics = json.loads(artifacts["metrics"].read_text(encoding="utf-8"))
    mismatches = {
        key: (metrics.get(key), expected)
        for key, expected in EXPECTED_METRICS.items()
        if metrics.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"Seed {tag} metrics protocol mismatch: {mismatches}")
    return artifacts


def _collect_steps(
    config: Config,
    seed: int,
    checkpoint: Path | None = None,
) -> list[Step]:
    """Return the ordered GaWF-only per-seed collection commands."""

    tag = _seed_tag(seed)
    if checkpoint is None:
        checkpoint = (
            config.checkpoint_root
            / f"{MODEL}-seed{tag}"
            / "<resolved-model-checkpoint>.pth"
        )
    root = config.output_root
    trajectory = root / "fig3" / f"seed{tag}" / "gawf_gate_trajectory.npz"
    selectivity = root / "fig7" / f"seed{tag}" / "selectivity" / "part1_selectivity.npz"
    compact = (
        root / "fig7" / f"seed{tag}" / "compact" / "recurrent_gate_condition_means.npz"
    )
    behavior = root / "behavior" / f"{MODEL}-seed{tag}"
    steps = [
        Step(
            "reset-excluded test accuracy",
            _module(
                config,
                "utils.analysis.clutter.fig1_reset_excluded_test_accuracy",
                "collect",
                "--ckpt",
                checkpoint,
                "--model",
                MODEL,
                "--seed",
                seed,
                "--output_dir",
                behavior,
                "--data_dir",
                config.data_dir,
                "--data_suffix",
                "40h-uint8",
                "--sequence_length",
                32,
                "--batch_size",
                256,
                "--num_workers",
                2,
                "--device",
                config.device,
            ),
            behavior / "reset_excluded_test_accuracy.json",
        ),
        Step(
            "Figure 3 gate trajectory and distributions",
            _module(
                config,
                "utils.analysis.clutter.fig3_gate_distribution",
                "--ckpt",
                checkpoint,
                "--data_dir",
                config.data_dir,
                "--data_suffix",
                "40h-uint8",
                "--device",
                config.device,
                "--batch_size",
                16,
                "--gate_chunk_size",
                32,
                "--save_dir",
                root / "fig3" / f"seed{tag}",
            ),
            trajectory,
        ),
    ]
    for condition in ("sector", "digit"):
        destination = root / "fig6_encoder" / condition / f"seed{tag}"
        steps.append(
            Step(
                f"Figure 6 encoder {condition} patterns",
                _module(
                    config,
                    "utils.analysis.clutter.fig6_encoder_sector_patterns",
                    "collect",
                    "--ckpt",
                    checkpoint,
                    "--data_dir",
                    config.data_dir,
                    "--output_dir",
                    destination,
                    "--condition",
                    condition,
                    "--device",
                    config.device,
                    "--batch_size",
                    16,
                    "--num_workers",
                    2,
                ),
                destination / f"encoder_{condition}_patterns.npz",
            )
        )
    steps.extend(
        [
            Step(
                "Figure 6 sequential input-gate maps",
                _module(
                    config,
                    "utils.analysis.clutter.fig6_sector_gate_sequential",
                    "--trajectory",
                    trajectory,
                    "--save_dir",
                    root / "fig6_gate" / f"seed{tag}",
                    "--device",
                    config.device,
                    "--seed",
                    seed,
                ),
                root
                / "fig6_gate"
                / f"seed{tag}"
                / "sector_gate_mean_sequential_equal_n.npz",
            ),
            Step(
                "Figure 7 hidden selectivity",
                _module(
                    config,
                    "utils.analysis.clutter.fig7_hidden_selectivity_collect",
                    "--ckpt",
                    checkpoint,
                    "--data_dir",
                    config.data_dir,
                    "--output_dir",
                    root / "fig7" / f"seed{tag}" / "selectivity",
                    "--device",
                    config.device,
                    "--batch_size",
                    16,
                    "--num_workers",
                    2,
                    "--seed",
                    seed,
                ),
                selectivity,
            ),
            Step(
                "Figure 7 recurrent-gate compact cache",
                _module(
                    config,
                    "utils.analysis.clutter.fig7_recurrent_gate_multiseed",
                    "collect",
                    "--trajectory",
                    trajectory,
                    "--selectivity",
                    selectivity,
                    "--output_dir",
                    root / "fig7" / f"seed{tag}" / "compact",
                    "--seed",
                    seed,
                    "--device",
                    config.device,
                ),
                compact,
            ),
            Step(
                "Supplementary 2 input-gate sign/magnitude",
                _module(
                    config,
                    "utils.analysis.clutter.supple2_input_gate_sign_magnitude_sector",
                    "collect",
                    "--trajectory",
                    trajectory,
                    "--output_dir",
                    root / "supple2" / f"seed{tag}",
                    "--seed",
                    seed,
                    "--device",
                    config.device,
                    "--all_sectors",
                ),
                root / "supple2" / f"seed{tag}" / "input_gate_sign_magnitude_9sector.npz",
            ),
        ]
    )
    for condition in ("digit", "sector"):
        destination = root / "net_current" / condition / f"seed{tag}"
        steps.append(
            Step(
                f"Net recurrent current ({condition})",
                _module(
                    config,
                    "utils.analysis.clutter.fig6_net_recurrent_current",
                    "collect",
                    "--ckpt",
                    checkpoint,
                    "--data_dir",
                    config.data_dir,
                    "--compact",
                    compact,
                    "--output_dir",
                    destination,
                    "--seed",
                    seed,
                    "--condition",
                    condition,
                    "--device",
                    config.device,
                    "--batch_size",
                    16,
                    "--num_workers",
                    2,
                ),
                destination / "net_recurrent_current.npz",
            )
        )
    return steps


def _aggregate_steps(config: Config) -> list[Step]:
    """Return the ordered ten-seed aggregation and plotting commands."""

    root = config.output_root
    final = root / "final"
    figures = config.figure_root
    seed_dirs = [root / "fig3" / f"seed{seed:02d}" for seed in SEEDS]
    gate_seed_data = [
        root / "fig6_gate" / f"seed{seed:02d}" / "sector_gate_mean_sequential_equal_n.npz"
        for seed in SEEDS
    ]
    behavior_csv = final / "behavior" / "reset_excluded_test_accuracy_10seed_notanh.csv"
    steps = [
        Step(
            "Aggregate reset-excluded behavior data",
            _module(
                config,
                "utils.analysis.clutter.fig1_reset_excluded_test_accuracy",
                "aggregate",
                "--data_root",
                root / "behavior",
                "--models",
                MODEL,
                "--output_csv",
                behavior_csv,
            ),
            behavior_csv,
        ),
        Step(
            "Render Figure 3 distributions",
            _module(
                config,
                "utils.analysis.clutter.fig3_gate_distribution_plot",
                "--seed_dirs",
                *seed_dirs,
                "--raw_dir",
                final / "fig3",
                "--only_gate_weight_2x2",
                "--gate_weight_layout",
                "1x4",
                "--gate_weight_stem",
                "gate_and_weight_distributions_1x4_10seed_notanh",
                "--metadata_path",
                final / "fig3" / "gate_and_weight_distributions_1x4_10seed_notanh.json",
            ),
            final / "fig3" / "gate_and_weight_distributions_1x4_10seed_notanh.pdf",
        ),
        Step(
            "Audit Figure 3 half mass",
            _module(
                config,
                "utils.analysis.clutter.fig3_gate_half_mass",
                "--trajectory_root",
                root / "fig3",
                "--output",
                final / "fig3" / "fig3_gate_half_mass_notanh.json",
                "--device",
                config.device,
            ),
            final / "fig3" / "fig3_gate_half_mass_notanh.json",
        ),
    ]
    for condition in ("sector", "digit"):
        destination = final / "fig6_encoder" / condition
        steps.append(
            Step(
                f"Aggregate Figure 6 encoder {condition} patterns",
                _module(
                    config,
                    "utils.analysis.clutter.fig6_encoder_sector_patterns",
                    "plot",
                    "--data_root",
                    root / "fig6_encoder" / condition,
                    "--figure_dir",
                    destination,
                    "--condition",
                    condition,
                ),
                destination / f"encoder_{condition}_patterns_10seed_summary.npz",
            )
        )
    steps.extend(
        [
            Step(
                "Aggregate Figure 6 sequential input-gate maps",
                _module(
                    config,
                    "utils.analysis.clutter.fig6_sector_gate_sequential_plot",
                    "--seed_data",
                    *gate_seed_data,
                    "--fig_dir",
                    final / "fig6_gate",
                    "--stem",
                    "sector_gate_mean_sequential_equal_n_10seed_notanh",
                ),
                final
                / "fig6_gate"
                / "sector_gate_mean_sequential_equal_n_10seed_notanh_point_excluded.pdf",
            ),
            Step(
                "Aggregate Supplementary 2",
                _module(
                    config,
                    "utils.analysis.clutter.supple2_input_gate_sign_magnitude_sector",
                    "plot",
                    "--data_root",
                    root / "supple2",
                    "--figure_dir",
                    final / "supple2",
                    "--all_sectors",
                ),
                final
                / "supple2"
                / "Supple2_input_gate_sign_vs_mag_9sector_10seed_stats.json",
            ),
            Step(
                "Render combined Figure 6",
                _module(
                    config,
                    "utils.analysis.clutter.fig6_overall_sector_input_gate",
                    "--encoder_summary",
                    final
                    / "fig6_encoder"
                    / "sector"
                    / "encoder_sector_patterns_10seed_summary.npz",
                    "--gate_root",
                    root / "fig6_gate",
                    "--supple2_root",
                    root / "supple2",
                    "--figure",
                    figures / "overall_sector_input_gate_1x3_10seed.pdf",
                    "--output_data_dir",
                    final / "fig6_overall_data",
                    "--output_figure_dir",
                    final / "fig6_overall_figs",
                ),
                figures / "overall_sector_input_gate_1x3_10seed.pdf",
            ),
            Step(
                "Aggregate Figure 7 and Supplementary 3",
                _module(
                    config,
                    "utils.analysis.clutter.fig7_recurrent_gate_multiseed",
                    "plot",
                    "--data_root",
                    root / "fig7",
                    "--figure_dir",
                    final / "fig7",
                    "--summary_dir",
                    final / "fig7",
                ),
                final / "fig7" / "fig7_seed_level_summary.npz",
            ),
            Step(
                "Write Supplementary 3 seed statistics",
                _module(
                    config,
                    "utils.analysis.clutter.fig7_recurrent_gate_multiseed",
                    "plot",
                    "--data_root",
                    root / "fig7",
                    "--figure_dir",
                    final / "fig7",
                    "--summary_dir",
                    final / "fig7",
                    "--supple3_stats_only",
                ),
                final / "fig7" / "supple3_seed_level_sign_magnitude_stats.json",
            ),
        ]
    )
    for condition in ("digit", "sector"):
        unit_final = final / "net_current" / condition
        connection_final = final / "net_current" / "connection" / condition
        steps.extend(
            [
                Step(
                    f"Summarize net current ({condition})",
                    _module(
                        config,
                        "utils.analysis.clutter.fig6_net_recurrent_current",
                        "summarize",
                        "--data_root",
                        root / "net_current" / condition,
                        "--output_dir",
                        unit_final,
                        "--condition",
                        condition,
                    ),
                    unit_final / "net_recurrent_current_10seed_long.csv",
                ),
                Step(
                    f"Connection-normalize net current ({condition})",
                    _module(
                        config,
                        "utils.analysis.clutter.fig6_net_recurrent_current",
                        "connection",
                        "--data_root",
                        root / "net_current" / condition,
                        "--compact_root",
                        root / "fig7",
                        "--output_dir",
                        connection_final,
                        "--condition",
                        condition,
                    ),
                    connection_final / "net_recurrent_current_connection_10seed_long.csv",
                ),
            ]
        )
    digit_unit = final / "net_current" / "digit" / "net_recurrent_current_10seed_long.csv"
    sector_unit = final / "net_current" / "sector" / "net_recurrent_current_10seed_long.csv"
    digit_conn = (
        final
        / "net_current"
        / "connection"
        / "digit"
        / "net_recurrent_current_connection_10seed_long.csv"
    )
    sector_conn = (
        final
        / "net_current"
        / "connection"
        / "sector"
        / "net_recurrent_current_connection_10seed_long.csv"
    )
    steps.extend(
        [
            Step(
                "Render destination-unit recurrent current",
                _module(
                    config,
                    "utils.analysis.clutter.fig6_net_recurrent_current",
                    "fig8",
                    "--digit_long",
                    digit_unit,
                    "--sector_long",
                    sector_unit,
                    "--figure_dir",
                    final / "current_figures",
                    "--output_dir",
                    final / "current_records",
                    "--normalization",
                    "destination",
                ),
                final / "current_figures" / "Supple4_recurrent_current_unit.pdf",
            ),
            Step(
                "Render connection-normalized recurrent current",
                _module(
                    config,
                    "utils.analysis.clutter.fig6_net_recurrent_current",
                    "fig8",
                    "--digit_long",
                    digit_conn,
                    "--sector_long",
                    sector_conn,
                    "--figure_dir",
                    final / "current_figures",
                    "--output_dir",
                    final / "current_records",
                    "--normalization",
                    "connection",
                ),
                final / "current_figures" / "Fig8_recurrent_current_connection.pdf",
            ),
            Step(
                "Render combined Figure 7",
                _module(
                    config,
                    "utils.analysis.clutter.fig7_combined_recurrent_gate_and_current",
                    "--pdf_only",
                    "--fig7_data_root",
                    root / "fig7",
                    "--digit_long",
                    digit_conn,
                    "--sector_long",
                    sector_conn,
                    "--output_pdf",
                    figures / "rec_gate_disinhibit_and_current_2x3_10seed.pdf",
                ),
                figures / "rec_gate_disinhibit_and_current_2x3_10seed.pdf",
            ),
            Step(
                "Render Appendix sign/magnitude figures",
                _module(
                    config,
                    "utils.analysis.clutter.appendix_sign_magnitude_figures",
                    "--input-root",
                    root / "supple2",
                    "--recurrent-root",
                    root / "fig7",
                    "--output-dir",
                    final / "appendix_sign_magnitude",
                ),
                final
                / "appendix_sign_magnitude"
                / "Supple3_rec_gate_sign_vs_mag_digit_sector_delta_2x4_10seed.pdf",
            ),
        ]
    )
    return steps


def _run_steps(steps: Iterable[Step]) -> None:
    """Run existing analysis entry points and verify each primary output."""

    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
    for step in steps:
        print(f"[{step.name}] {' '.join(step.command)}", flush=True)
        subprocess.run(step.command, check=True, env=environment)
        if not step.expected.is_file():
            raise RuntimeError(f"Missing expected output after {step.name}: {step.expected}")


def _publish(config: Config) -> list[Path]:
    """Copy regenerated vector figures into the isolated notanh figure directory."""

    final = config.output_root / "final"
    mappings = {
        final / "fig3" / "gate_and_weight_distributions_1x4_10seed_notanh.pdf": (
            config.figure_root / "gate_and_weight_distributions_1x4_10seed.pdf"
        ),
        final
        / "appendix_sign_magnitude"
        / "Supple2_input_gate_sign_vs_mag_sector_delta_zoom_10seed_9sector.pdf": (
            config.figure_root
            / "input_gate_sign_vs_mag_sector_delta_zoom_10seed_9sector.pdf"
        ),
        final
        / "appendix_sign_magnitude"
        / "Supple3_rec_gate_sign_vs_mag_digit_sector_delta_2x4_10seed.pdf": (
            config.figure_root
            / "rec_gate_sign_vs_mag_digit_sector_delta_2x4_10seed.pdf"
        ),
        final / "current_figures" / "Supple4_recurrent_current_unit.pdf": (
            config.figure_root / "recurrent_current_unit.pdf"
        ),
    }
    config.figure_root.mkdir(parents=True, exist_ok=True)
    outputs = []
    for source, destination in mappings.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        if destination.exists():
            raise FileExistsError(destination)
        shutil.copy2(source, destination)
        outputs.append(destination)
    return sorted(config.figure_root.glob("*.pdf"))


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one structured result."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _behavior_summary(path: Path) -> str:
    """Return GaWF no-tanh test accuracy as seed-level mean and SEM."""

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 10:
        raise RuntimeError(f"Expected ten behavior rows, found {len(rows)}")
    pieces = []
    for key, label in (("char_acc", "Digit"), ("sector_acc", "Sector")):
        values = [float(row[key]) for row in rows]
        mean = sum(values) / len(values)
        sem = math.sqrt(sum((value - mean) ** 2 for value in values) / 9) / math.sqrt(10)
        pieces.append(f"- {label}: {mean:.4f}% ± {sem:.4f}% seed-level SEM")
    return "\n".join(pieces)


def _write_record(config: Config) -> None:
    """Write a non-overwriting notanh delta record from the regenerated structured outputs."""

    if config.record_output.exists():
        raise FileExistsError(config.record_output)
    final = config.output_root / "final"
    behavior = final / "behavior" / "reset_excluded_test_accuracy_10seed_notanh.csv"
    sources = (
        final / "fig3" / "fig3_gate_half_mass_notanh.json",
        final
        / "fig6_overall_data"
        / "fig6_overall_sector_input_gate_1x3_10seed.json",
        final / "supple2" / "Supple2_input_gate_sign_vs_mag_9sector_10seed_stats.json",
        final / "fig7" / "supple3_seed_level_sign_magnitude_stats.json",
        final / "current_records" / "Supple4_recurrent_current_unit_caption_stats.json",
        final / "current_records" / "Fig8_recurrent_current_connection_caption_stats.json",
    )
    for path in (behavior, *sources):
        if not path.is_file():
            raise FileNotFoundError(path)
    figure_paths = sorted(config.figure_root.glob("*.pdf"))
    if not figure_paths:
        raise RuntimeError(f"No notanh figures found in {config.figure_root}")
    source_rows = "\n".join(
        f"| `{path}` | `{_sha256(path)}` |" for path in (behavior, *sources)
    )
    figure_rows = "\n".join(f"- `{path}`" for path in figure_paths)
    deferred_rows = "\n".join(f"- {item}" for item in DEFERRED)
    static_rows = "\n".join(f"- {item}" for item in STATIC)
    content = f"""# GaWF no-tanh 统计记录

本文件由 `gawf_notanh_refresh.py` 从 `gawf_legacy_notanh` 十个 seed 的独立结构化结果
生成。它是 `docs/GaWF_STATS_RECORD.md` 的 **notanh delta 版本**，不复制旧 GaWF
数值。未列出的
dataset、sampling、reset exclusion 与 seed-level aggregation 口径沿用原记录。

## 模型定义

`gawf_legacy_notanh` 保留 legacy in-loop `LayerNorm → ReLU → Dropout` recurrence，但将
内部
`tanh(preactivation)` 替换为 identity。训练协议要求 `actual_epochs=150`、
`core_rnn_activation=identity`、`core_output_wrap=ln_relu_dropout`、
`gawf_core_semantics=legacy`。

## Reset-excluded test accuracy

{_behavior_summary(behavior)}

结构化十 seed 表：`{behavior}`。

    ## 结构化事实源

| Path | SHA-256 |
|---|---|
{source_rows}

## 已生成 GaWF-only figures

{figure_rows}

## Deferred until baseline no-wrap completion

{deferred_rows}

## Static; no checkpoint-dependent refresh

{static_rows}
"""
    config.record_output.parent.mkdir(parents=True, exist_ok=True)
    config.record_output.write_text(content, encoding="utf-8")


def _plan(config: Config, seed: int | None) -> dict[str, object]:
    """Return a machine-readable no-write execution plan."""

    example_seed = seed or 1
    collect = _collect_steps(config, example_seed)
    aggregate = _aggregate_steps(config)
    return {
        "model": MODEL,
        "expected_seeds": list(SEEDS),
        "checkpoint_root": str(config.checkpoint_root),
        "data_dir": str(config.data_dir),
        "output_root": str(config.output_root),
        "figure_root": str(config.figure_root),
        "record_output": str(config.record_output),
        "collect_example_seed": example_seed,
        "collect": [step.__dict__ | {"expected": str(step.expected)} for step in collect],
        "aggregate": [step.__dict__ | {"expected": str(step.expected)} for step in aggregate],
        "deferred": list(DEFERRED),
        "static": list(STATIC),
    }


def parse_args() -> argparse.Namespace:
    """Parse planner, validation, collection, aggregation, and record actions."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "validate", "collect", "aggregate", "record"))
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--figure-root", required=True, type=Path)
    parser.add_argument(
        "--record-output",
        type=Path,
        default=Path("docs/GaWF_STATS_RECORD_notanh.md"),
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Dispatch the requested non-overwriting refresh stage."""

    args = parse_args()
    config = Config(
        checkpoint_root=args.checkpoint_root.resolve(),
        data_dir=args.data_dir.resolve(),
        output_root=args.output_root.resolve(),
        figure_root=args.figure_root.resolve(),
        record_output=args.record_output.resolve(),
        python=args.python,
        device=args.device,
    )
    if args.action == "plan":
        print(json.dumps(_plan(config, args.seed), indent=2, default=str))
        return
    if args.action == "validate":
        seeds = (args.seed,) if args.seed else SEEDS
        result = {f"seed{seed:02d}": _seed_artifacts(config, seed) for seed in seeds}
        print(json.dumps(result, indent=2, default=str))
        return
    if not args.execute:
        raise SystemExit(f"{args.action} requires --execute")
    if args.action == "collect":
        if args.seed is None:
            raise SystemExit("collect requires --seed")
        marker = config.output_root / "status" / f"seed{args.seed:02d}.complete.json"
        if marker.exists():
            raise FileExistsError(marker)
        checkpoint = _seed_artifacts(config, args.seed)["model"]
        steps = _collect_steps(config, args.seed, checkpoint)
        _run_steps(steps)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps(
                {
                    "model": MODEL,
                    "seed": args.seed,
                    "outputs": [str(step.expected) for step in steps],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return
    if args.action == "aggregate":
        missing = [
            config.output_root / "status" / f"seed{seed:02d}.complete.json"
            for seed in SEEDS
            if not (config.output_root / "status" / f"seed{seed:02d}.complete.json").is_file()
        ]
        if missing:
            raise RuntimeError("Missing seed completion markers: " + ", ".join(map(str, missing)))
        final = config.output_root / "final"
        if final.exists() or config.figure_root.exists():
            raise FileExistsError(
                f"Refusing to overwrite aggregate outputs: {final} or {config.figure_root}"
            )
        _run_steps(_aggregate_steps(config))
        published = _publish(config)
        (final / ".complete").write_text(
            json.dumps({"model": MODEL, "figures": [str(path) for path in published]}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        return
    complete = config.output_root / "final" / ".complete"
    if not complete.is_file():
        raise RuntimeError(f"Aggregate completion marker is absent: {complete}")
    _write_record(config)


if __name__ == "__main__":
    main()
