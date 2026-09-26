"""Plot the first three best6 panels for three Clutter model definitions.

Inputs are historical seed-level test accuracy, historical mean/SEM curves, 25 current
training histories, and optional matching current test/recovery exports. Outputs are one
2x3 PDF, seed-level test CSV, epoch/offset curve CSVs, and a coverage manifest. All plotted
means use one independent training seed per observation; numeric arrays are float64 in memory.
"""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from utils.analysis.anal_paths import output_dir
from utils.analysis.clutter.fg_switch_offset_acc import MODEL_COLORS


PROJECT_ROOT = Path(__file__).resolve().parents[3]
NAME = "best18_outputonly_partial25"
DATA_DIR = output_dir("G_behaviour", NAME, "data")
FIG_DIR = output_dir("G_behaviour", NAME, "figs")
HISTORY_DIR = DATA_DIR / "inputs/current25"
EVAL_DIR = DATA_DIR / "inputs/current25_eval"
OLD_DIR = PROJECT_ROOT / "results/data/analysis/G_behaviour/best14_behavior_2x4_current"
FAMILIES = ("gawf", "rnn", "lstm", "gru", "mamba", "s5")
DISPLAY = {"gawf": "GaWF", "rnn": "RNN", "lstm": "LSTM", "gru": "GRU",
           "mamba": "Mamba", "s5": "S5"}
VERSIONS = ("v1", "v2", "v3")
CURRENT_SEEDS = {"rnn": 10, "lstm": 10, "gru": 5}
HISTORICAL = {
    ("v1", family): family for family in FAMILIES
} | {
    ("v2", "gawf"): "gawf_legacy_notanh",
    ("v2", "rnn"): "rnn_inloop_notanh",
} | {("v2", family): f"{family}_nowrap" for family in FAMILIES[2:]}
OFFSETS = np.asarray([*range(-10, 0), *range(1, 11)], dtype=np.int64)
LINE_STYLE = {"v1": (":", 0.65, 1.0), "v2": ("--", 0.8, 1.25), "v3": ("-", 1.0, 2.0)}

Series = tuple[str, str]
TestData = dict[Series, dict[int, tuple[float, float]]]
CurveData = dict[tuple[Series, str], tuple[np.ndarray, np.ndarray, int]]


def _mean_sem(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return seed mean and SEM along axis zero; one-seed SEM is zero."""

    mean = values.mean(axis=0)
    sem = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0]) if len(values) > 1 else mean * 0
    return mean, sem


def _read_historical_tests() -> TestData:
    """Read all 120 historical seed-level standard-test values."""

    inverse = {name: key for key, name in HISTORICAL.items()}
    tests: TestData = defaultdict(dict)
    path = OLD_DIR / "test_accuracy_14model.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["model"] not in inverse:
                continue
            series = inverse[row["model"]]
            seed = int(row["seed"])
            if seed in tests[series]:
                raise ValueError(f"Duplicate historical test {series} seed {seed}")
            tests[series][seed] = (float(row["digit_acc"]), float(row["sector_acc"]))
    for series in HISTORICAL:
        if set(tests[series]) != set(range(1, 11)):
            raise ValueError(f"Historical test coverage must be 10 seeds: {series}")
    return tests


def _read_historical_curves(filename: str, x_field: str) -> CurveData:
    """Read one complete historical 12-model mean/SEM curve table."""

    inverse = {name: key for key, name in HISTORICAL.items()}
    grouped: dict[tuple[Series, str], list[tuple[int, float, float, int]]] = defaultdict(list)
    with (OLD_DIR / filename).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["model"] in inverse:
                grouped[(inverse[row["model"]], row["readout"])].append(
                    (int(row[x_field]), float(row["mean"]), float(row["sem"]),
                     int(row["n_seeds"]))
                )
    expected_x = list(range(1, 151)) if x_field == "epoch" else OFFSETS.tolist()
    curves: CurveData = {}
    for series in HISTORICAL:
        for readout in ("sector", "digit"):
            rows = sorted(grouped[(series, readout)])
            if [item[0] for item in rows] != expected_x or {item[3] for item in rows} != {10}:
                raise ValueError(f"Incomplete historical {filename}: {series} {readout}")
            curves[(series, readout)] = (
                np.asarray([item[1] for item in rows], dtype=np.float64),
                np.asarray([item[2] for item in rows], dtype=np.float64), 10,
            )
    return curves


def _read_current_histories() -> CurveData:
    """Read the fixed 25 completed training histories and verify their protocol tags."""

    curves: CurveData = {}
    for family, count in CURRENT_SEEDS.items():
        arrays: dict[str, list[np.ndarray]] = {"digit": [], "sector": []}
        for seed in range(1, count + 1):
            leaf = HISTORY_DIR / f"{family}-seed{seed:02d}"
            metric_files = list(leaf.glob("*_metrics.json"))
            history_files = list(leaf.glob("*.pkl"))
            if len(metric_files) != 1 or len(history_files) != 1:
                raise ValueError(f"Expected one metrics/history pair in {leaf}")
            metadata = json.loads(metric_files[0].read_text(encoding="utf-8"))
            expected = {
                "model_type": family, "seed": seed, "actual_epochs": 150,
                "dataset_suffix": "40h-uint8", "eval_dataset_suffix": "40h-uint8",
                "dropout_protocol": "per_layer_output_only_clean_state_v1",
            }
            if any(metadata.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Protocol mismatch in {metric_files[0]}")
            with history_files[0].open("rb") as handle:
                history = pickle.load(handle)
            for readout, field in (("digit", "val_loss_char"), ("sector", "val_loss_pos")):
                values = np.asarray(history[field], dtype=np.float64)
                if values.shape != (150,) or not np.isfinite(values).all():
                    raise ValueError(f"Invalid {field} trajectory in {history_files[0]}")
                arrays[readout].append(values)
        for readout, blocks in arrays.items():
            mean, sem = _mean_sem(np.stack(blocks))
            curves[(("v3", family), readout)] = mean, sem, count
    return curves


def _read_current_tests(tests: TestData) -> None:
    """Add matching completed current test JSONs, if available."""

    for path in sorted((EVAL_DIR / "test").glob("*/reset_excluded_test_accuracy.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        family, sep, seed_text = path.parent.name.rpartition("-seed")
        if not sep or family not in CURRENT_SEEDS:
            raise ValueError(f"Unexpected test leaf {path}")
        seed = int(seed_text)
        expected = {
            "model": family, "seed": seed, "sequence_length": 32,
            "excluded_timestep": 0, "n_frames": 55769,
        }
        if seed > CURRENT_SEEDS[family] or any(
            payload.get(key) != value for key, value in expected.items()
        ):
            raise ValueError(f"Current test protocol mismatch in {path}")
        tests[("v3", family)][seed] = float(payload["char_acc"]), float(payload["sector_acc"])


def _read_current_recovery(curves: CurveData) -> dict[str, set[int]]:
    """Add matching completed current recovery NPZs and return seed coverage."""

    grouped: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    coverage: dict[str, set[int]] = defaultdict(set)
    for path in sorted((EVAL_DIR / "recovery").glob("*/fg_switch_offset_acc_*.npz")):
        family, sep, seed_text = path.parent.name.rpartition("-seed")
        if not sep or family not in CURRENT_SEEDS:
            raise ValueError(f"Unexpected recovery leaf {path}")
        seed = int(seed_text)
        meta_path = path.with_name(path.name.replace("_acc_", "_meta_", 1)).with_suffix(
            ".json"
        )
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if seed > CURRENT_SEEDS[family] or metadata.get("exclude_window_initial_frame") is not True:
            raise ValueError(f"Current recovery protocol mismatch in {path}")
        with np.load(path) as payload:
            if not np.array_equal(payload["offset_order"], OFFSETS):
                raise ValueError(f"Current recovery offsets mismatch in {path}")
            for readout, field in (("digit", "char_acc"), ("sector", "sector_acc")):
                values = np.asarray(payload[field], dtype=np.float64)
                if values.shape != (20,) or not np.isfinite(values).all():
                    raise ValueError(f"Invalid recovery values in {path}")
                grouped[(family, readout)].append(values)
        if seed in coverage[family]:
            raise ValueError(f"Duplicate recovery seed {family} {seed}")
        coverage[family].add(seed)
    for family, seeds in coverage.items():
        for readout in ("sector", "digit"):
            mean, sem = _mean_sem(np.stack(grouped[(family, readout)]))
            curves[(("v3", family), readout)] = mean, sem, len(seeds)
    return coverage


def _write_tables(tests: TestData, losses: CurveData, recovery: CurveData) -> list[str]:
    """Write all plotted seed values and curve statistics to structured CSV files."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    files = ["test_accuracy_available.csv", "validation_loss_available.csv",
             "target_switch_available.csv"]
    with (DATA_DIR / files[0]).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("version", "model", "seed", "digit_acc", "sector_acc"))
        for version in VERSIONS:
            for family in FAMILIES:
                for seed, (digit, sector) in sorted(tests.get((version, family), {}).items()):
                    writer.writerow((version, family, seed, digit, sector))
    for filename, curves, x_name, x_values in (
        (files[1], losses, "epoch", np.arange(1, 151)),
        (files[2], recovery, "offset", OFFSETS),
    ):
        with (DATA_DIR / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("version", "model", "readout", x_name, "mean", "sem", "n_seeds"))
            for version in VERSIONS:
                for family in FAMILIES:
                    for readout in ("sector", "digit"):
                        if ((version, family), readout) not in curves:
                            continue
                        mean, sem, n_seeds = curves[((version, family), readout)]
                        for x, value, error in zip(x_values, mean, sem):
                            writer.writerow((version, family, readout, x, value, error, n_seeds))
    return files


def _plot_test(axis: plt.Axes, tests: TestData, readout: str) -> None:
    """Plot version-grouped accuracy bars, SEM, and one point per seed."""

    selected = 1 if readout == "sector" else 0
    rng = np.random.default_rng(18)
    for family_index, family in enumerate(FAMILIES):
        for version_index, version in enumerate(VERSIONS):
            seed_values = tests.get((version, family), {})
            if not seed_values:
                continue
            values = np.asarray([pair[selected] for _, pair in sorted(seed_values.items())])
            mean, sem = _mean_sem(values)
            x = family_index + (version_index - 1) * 0.25
            axis.bar(x, mean, width=0.22, color=MODEL_COLORS[family],
                     alpha=(0.4, 0.7, 1.0)[version_index], edgecolor="black" if version == "v3"
                     else "none", linewidth=0.45, yerr=sem, capsize=1.4,
                     error_kw={"elinewidth": 0.7, "ecolor": "#333333"})
            jitter = rng.uniform(-0.058, 0.058, size=len(values))
            axis.scatter(x + jitter, values, s=3.8, color="#242424", alpha=0.55,
                         linewidths=0, zorder=3)
    axis.set_xlim(-0.55, 5.55)
    axis.set_ylim((38, 100) if readout == "sector" else (25, 95))
    axis.set_ylabel("Accuracy (%)")
    axis.set_xticks(np.arange(len(FAMILIES)), [DISPLAY[name] for name in FAMILIES])
    axis.grid(axis="y", alpha=0.18, linewidth=0.5)


def _plot_curve(axis: plt.Axes, curves: CurveData, readout: str, kind: str) -> None:
    """Plot family-coloured curves with line style encoding the definition version."""

    x = np.arange(1, 151) if kind == "loss" else np.arange(20)
    for version in VERSIONS:
        for family in FAMILIES:
            key = ((version, family), readout)
            if key not in curves:
                continue
            mean, sem, _n = curves[key]
            style, alpha, width = LINE_STYLE[version]
            axis.plot(x, mean, color=MODEL_COLORS[family], linestyle=style,
                      linewidth=width, alpha=alpha)
            axis.fill_between(x, mean - sem, mean + sem, color=MODEL_COLORS[family],
                              alpha=0.035 if version != "v3" else 0.10, linewidth=0)
    if kind == "loss":
        axis.set_xlim(1, 150)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Cross-entropy")
    else:
        axis.set_xlim(0, 19)
        axis.set_ylim(0, 100)
        axis.axvline(9.5, color="#777777", linewidth=0.55, alpha=0.5)
        axis.set_xticks((0, 9.5, 13, 19), ("pre10", "switch", "post4", "post10"))
        axis.set_xlabel("Frame relative to target switch")
        axis.set_ylabel("Accuracy (%)")
    axis.grid(axis="y", alpha=0.18, linewidth=0.5)


def _render(tests: TestData, losses: CurveData, recovery: CurveData, path: Path) -> None:
    """Render the best6-like two-readout, three-column comparison PDF."""

    with plt.rc_context({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6,
                         "ytick.labelsize": 6, "pdf.fonttype": 42}):
        figure, axes = plt.subplots(2, 3, figsize=(11.7, 5.0),
                                   gridspec_kw={"width_ratios": [2.5, 1.3, 1.5]})
        for row, readout in enumerate(("sector", "digit")):
            _plot_test(axes[row, 0], tests, readout)
            _plot_curve(axes[row, 1], losses, readout, "loss")
            _plot_curve(axes[row, 2], recovery, readout, "recovery")
            if row == 0:
                for axis in axes[row]:
                    axis.tick_params(labelbottom=False)
        figure.subplots_adjust(left=0.085, right=0.985, bottom=0.15, top=0.82,
                               wspace=0.34, hspace=0.32)
        handles = [Line2D([0], [0], color=MODEL_COLORS[family], lw=2,
                          label=DISPLAY[family])
                   for family in FAMILIES]
        handles.extend([
            Patch(facecolor="#777777", alpha=0.4, label="v1: early wrap"),
            Patch(facecolor="#777777", alpha=0.7, label="v2: no-tanh / no-wrap"),
            Patch(facecolor="#777777", edgecolor="black", label="v3: output-only"),
        ])
        figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.55, 0.995),
                      ncol=5, frameon=False, fontsize=7, handlelength=1.8)
        titles = ("A  Test accuracy", "B  Validation loss", "C  Target-switch recovery")
        for index, title in enumerate(titles):
            box = axes[0, index].get_position()
            figure.text(box.x0 + box.width / 2, 0.845, title, ha="center", va="bottom",
                        fontsize=8.5, fontweight="bold")
        for row, label in enumerate(("Location", "Identity")):
            box = axes[row, 0].get_position()
            figure.text(0.021, box.y0 + box.height / 2, label, rotation=90,
                        ha="center", va="center", fontsize=8.5, fontweight="bold")
        figure.text(0.5, 0.035,
                    "v1/v2: historical 10 seeds each. v3: RNN/LSTM 10 seeds, GRU 5 seeds; "
                    "GaWF/Mamba/S5 pending. Curves: dotted/dashed/solid by version; "
                    "shading: SEM. Test/recovery use completed evals only.",
                    ha="center", va="bottom", fontsize=6.7)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path)
        plt.close(figure)


def main() -> None:
    """Validate sources, write three panel tables and coverage, then render the PDF."""

    tests = _read_historical_tests()
    losses = _read_historical_curves("validation_loss_12model_10seed.csv", "epoch")
    recovery = _read_historical_curves("target_switch_12model_10seed.csv", "offset")
    losses.update(_read_current_histories())
    _read_current_tests(tests)
    recovery_seeds = _read_current_recovery(recovery)
    files = _write_tables(tests, losses, recovery)
    pdf_path = FIG_DIR / f"{NAME}_2x3.pdf"
    _render(tests, losses, recovery, pdf_path)
    coverage = {
        f"{version}-{family}": {
            "test_seeds": sorted(tests.get((version, family), {})),
            "validation_loss_n": losses.get(((version, family), "digit"), (None, None, 0))[2],
            "recovery_seeds": (
                sorted(recovery_seeds.get(family, set())) if version == "v3" else
                list(range(1, 11))
            ),
        }
        for version in VERSIONS for family in FAMILIES
    }
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()
    source_files = [
        OLD_DIR / "test_accuracy_14model.csv",
        OLD_DIR / "validation_loss_12model_10seed.csv",
        OLD_DIR / "target_switch_12model_10seed.csv",
        *sorted(HISTORY_DIR.glob("*/*_metrics.json")),
        *sorted(HISTORY_DIR.glob("*/*.pkl")),
        *sorted(EVAL_DIR.glob("*/*/*")),
    ]
    digest = hashlib.sha256()
    for source in source_files:
        digest.update(str(source.relative_to(PROJECT_ROOT)).encode("utf-8"))
        digest.update(hashlib.sha256(source.read_bytes()).digest())
    provenance = {
        "schema_version": 1,
        "script": str(Path(__file__).resolve()), "git_commit": commit,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_bundle_sha256": digest.hexdigest(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "category": "G_behaviour", "data_root": str(DATA_DIR.resolve()),
        "figure_root": str(FIG_DIR.resolve()),
        "files_written": [*files, "coverage.json", "provenance.json", pdf_path.name],
        "inputs": {
            "historical_tables": str(OLD_DIR), "current_histories": str(HISTORY_DIR),
            "current_evaluation": str(EVAL_DIR),
            "current_training_commit": "a671eeac912008a4d39bddce7dd390e21da1bdb8",
            "current_evaluation_host": "sjc-remote",
            "current_evaluation_id": (
                "clutter-outputonly-partial25-retry1-eval40h-rnn-lstm-gru-s1to10-s1to5"
            ),
            "current_evaluation_run_id": "sjc_clutter_partial25_eval_retry1_20260926",
            "current_evaluation_result_root": (
                "/G/MIMOlab/Codes/aim3_gawf_rnn/results/data/analysis/"
                "clutter_output_only_330_partial25_eval_sjc_20260926_retry1"
            ),
        },
        "key_results": {
            "historical_models": 12, "current_training_units": 25,
            "current_test_units": sum(len(tests.get(("v3", family), {})) for family in FAMILIES),
            "current_recovery_units": sum(len(seeds) for seeds in recovery_seeds.values()),
        },
        "panel_coverage": coverage,
    }
    (DATA_DIR / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n",
                                                encoding="utf-8")
    (DATA_DIR / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n",
                                               encoding="utf-8")
    print(json.dumps(provenance["key_results"], sort_keys=True))
    print(pdf_path)


if __name__ == "__main__":
    main()
