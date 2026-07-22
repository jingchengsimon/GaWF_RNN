"""Move legacy analysis outputs into the sole category-indexed result tree.

The command is a dry run unless ``--apply`` is passed.  It moves files rather than copying,
never overwrites a destination, removes only symlinks from a previous index view, and reports
ambiguous files without assigning them.  Empty legacy roots receive a README that points to
``results/anal_index``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils_anal.anal_paths import CATEGORIES


@dataclass(frozen=True)
class Classification:
    """One unambiguous migration destination."""

    category: str
    script_name: str


FIGURE_RULES: tuple[tuple[str, Classification], ...] = (
    (
        r"gawf_gate_audit/(01_pooled|02_weight_sign|07_effective)",
        Classification("A_raw_gate", "gawf_gate_distribution"),
    ),
    (
        r"gawf_gate_audit/(04_per_|06_sparsity)",
        Classification("B_gate_by_context", "gawf_gate_distribution"),
    ),
    (
        r"gawf_gate_audit/(03_sector|03_sector_digit)",
        Classification("C_delta_gate", "gawf_gate_distribution"),
    ),
    (
        r"gawf_gate_audit/05_task_relevance",
        Classification("E_relevance_alignment", "gawf_gate_distribution"),
    ),
    (
        r"gawf_gate_audit_digit/03_digit_centered",
        Classification("C_delta_gate", "gawf_gate_digit_distribution"),
    ),
    (
        r"gawf_gate_audit_digit/",
        Classification("B_gate_by_context", "gawf_gate_digit_distribution"),
    ),
    (r"1_sector_v_modulation/", Classification("B_gate_by_context", "1_sector_v_modulation")),
    (r"2_sector_sigmoid_gate/", Classification("B_gate_by_context", "2_sector_sigmoid_gate")),
    (
        r"gawf_gate_context_specificity/0[12]_",
        Classification("C_delta_gate", "gawf_gate_context_specificity"),
    ),
    (
        r"gawf_gate_context_specificity/0[34]_",
        Classification("D_variance_decomposition", "gawf_gate_context_specificity"),
    ),
    (
        r"gawf_gate_context_specificity/0[5-8]_",
        Classification("H_controls", "gawf_gate_context_specificity"),
    ),
    (
        r"gawf_gate_robustness/01_delta_survival",
        Classification("C_delta_gate", "gawf_gate_robustness"),
    ),
    (r"gawf_gate_robustness/", Classification("H_controls", "gawf_gate_robustness")),
    (
        r"gawf_symmetric_relevance_timing/part1_",
        Classification("D_variance_decomposition", "gawf_symmetric_relevance_timing"),
    ),
    (
        r"gawf_symmetric_relevance_timing/part2_",
        Classification("E_relevance_alignment", "gawf_symmetric_relevance_timing"),
    ),
    (
        r"gawf_symmetric_relevance_timing/part3_",
        Classification("F_timing", "gawf_symmetric_relevance_timing"),
    ),
    (
        r"3_digit_v_vs_activation/",
        Classification("E_relevance_alignment", "3_digit_v_vs_activation"),
    ),
    (
        r"crossdecode_gawf_identity",
        Classification("E_relevance_alignment", "crossdecode_gawf_identity"),
    ),
    (
        r"rnn_unit_gate_context_specificity/",
        Classification("D_variance_decomposition", "rnn_unit_gate_context_specificity"),
    ),
    (
        r"dpca_marginalized_variance",
        Classification("D_variance_decomposition", "5_dpca_marginalized_variance"),
    ),
    (
        r"feedback_ablation|4_feedback_ablation",
        Classification("G_behaviour", "viz_feedback_ablation"),
    ),
    (r"fg_switch_offset_acc/", Classification("G_behaviour", "fg_switch_offset_acc")),
    (r"5_pop_act_switch_trajectory/", Classification("F_timing", "pop_act_switch_trajectory")),
    (r"5_pop_act_umap", Classification("D_variance_decomposition", "pop_act_umap")),
    (r"cnn_channel/", Classification("E_relevance_alignment", "cnn_channel")),
    (r"hidden_activation/", Classification("E_relevance_alignment", "hidden_activation")),
    (r"gate_avg_allsector/", Classification("B_gate_by_context", "gate_avg_allsector")),
    (r"gate_avg/", Classification("B_gate_by_context", "gate_avg")),
    (r"gate_sample/", Classification("B_gate_by_context", "gate_sample")),
    (r"V_basis/", Classification("B_gate_by_context", "V_basis")),
    (r"whh/", Classification("H_controls", "whh")),
)

DATA_DIRECTORY_RULES: dict[str, Classification] = {
    "1_sector_v_modulation": Classification("B_gate_by_context", "1_sector_v_modulation"),
    "2_sector_sigmoid_gate": Classification("B_gate_by_context", "2_sector_sigmoid_gate"),
    "3_digit_v_vs_activation": Classification("E_relevance_alignment", "3_digit_v_vs_activation"),
    "cnn_channel": Classification("E_relevance_alignment", "cnn_channel_stats"),
    "crossdecode_gawf_identity": Classification(
        "E_relevance_alignment", "crossdecode_gawf_identity"
    ),
    "feedback_ablation": Classification("G_behaviour", "feedback_ablation"),
    "fg_switch_offset_acc": Classification("G_behaviour", "export_fg_switch_offset_acc"),
    "gate_avg": Classification("B_gate_by_context", "export_gate_avg"),
    "gate_avg_allsector": Classification("B_gate_by_context", "export_gate_avg_allsector"),
    "gate_sample": Classification("B_gate_by_context", "export_gate_sample"),
    "gawf_gate_audit_digit": Classification("B_gate_by_context", "gawf_gate_digit_distribution"),
    "hidden_activation": Classification("E_relevance_alignment", "hidden_unit_tuning"),
    "pop_act": Classification("D_variance_decomposition", "export_pop_act"),
    "pop_act_multiseg": Classification("D_variance_decomposition", "export_pop_act"),
    "5_pop_act_umap": Classification(
        "D_variance_decomposition", "pop_act_representation_similarity"
    ),
    "5_pop_act_umap_multiseg": Classification(
        "D_variance_decomposition", "5_dpca_marginalized_variance"
    ),
    "5_pop_act_switch_trajectory": Classification("F_timing", "pop_act_switch_trajectory"),
    "rnn_unit_gate_context_specificity": Classification(
        "D_variance_decomposition", "rnn_unit_gate_context_specificity"
    ),
    "V_basis_exports": Classification("B_gate_by_context", "export_V_basis"),
    "V_basis_exports_0317": Classification("B_gate_by_context", "export_V_basis"),
    "whh": Classification("H_controls", "export_whh"),
}

DATA_FILE_RULES: tuple[tuple[str, Classification], ...] = (
    (
        r"gawf_gate_audit/gawf_gate_trajectory\.npz$",
        Classification("A_raw_gate", "gawf_gate_distribution"),
    ),
    (
        r"gawf_gate_audit/gawf_gate_interventions\.json$",
        Classification("H_controls", "gawf_gate_distribution"),
    ),
    (
        r"gawf_symmetric_relevance_timing/part1_",
        Classification("D_variance_decomposition", "gawf_symmetric_relevance_timing"),
    ),
    (
        r"gawf_symmetric_relevance_timing/part2_",
        Classification("E_relevance_alignment", "gawf_symmetric_relevance_timing"),
    ),
    (
        r"gawf_symmetric_relevance_timing/part3_",
        Classification("F_timing", "gawf_symmetric_relevance_timing"),
    ),
    (
        r"gawf_symmetric_relevance_timing/(part0_splits|run_metadata)\.json$",
        Classification("H_controls", "gawf_symmetric_relevance_timing"),
    ),
    (
        r"gawf_gate_context_specificity/(part0_prerequisites|delta_group_mean_verification)\.json$",
        Classification("H_controls", "gawf_gate_context_specificity"),
    ),
    (
        r"gawf_gate_context_specificity/part2_headline\.csv$",
        Classification("D_variance_decomposition", "gawf_gate_context_parts123"),
    ),
    (
        r"gawf_gate_robustness/part[2-5]_.*\.csv$",
        Classification("H_controls", "gawf_gate_robustness"),
    ),
)

AMBIGUOUS_DATA_ROOTS = {
    "gawf_gate_audit": "mixed raw/context/delta statistics in one artifact",
    "gawf_gate_context_specificity": "mixed C/D/H arrays and reports",
    "gawf_gate_robustness": "compact file mixes delta survival and robustness controls",
    "gawf_symmetric_relevance_timing": "run metadata spans D/E/F parts",
}

CANONICAL_LEGACY_ROOTS = set(DATA_DIRECTORY_RULES) | set(AMBIGUOUS_DATA_ROOTS) | {
    "4_feedback_ablation",
    "V_basis",
}


def parse_args() -> argparse.Namespace:
    """Parse migration paths and the explicit apply switch."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy_data", type=Path, default=PROJECT_ROOT / "results/anal_data")
    parser.add_argument("--legacy_figs", type=Path, default=PROJECT_ROOT / "results/anal_figs")
    parser.add_argument("--index_root", type=Path, default=PROJECT_ROOT / "results/anal_index")
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def classify_figure(relative: Path) -> Classification | None:
    """Classify one figure path, returning None when human judgement is required."""

    value = relative.as_posix()
    for pattern, classification in FIGURE_RULES:
        if re.search(pattern, value):
            return classification
    return None


def classify_data(relative: Path) -> Classification | None:
    """Classify one data path without hiding mixed-artifact ambiguity."""

    value = relative.as_posix()
    for pattern, classification in DATA_FILE_RULES:
        if re.search(pattern, value):
            return classification
    root = relative.parts[0]
    if root in AMBIGUOUS_DATA_ROOTS:
        return None
    prefix_rules = (
        ("fg_switch_offset_acc", Classification("G_behaviour", "export_fg_switch_offset_acc")),
        ("feedback_ablation", Classification("G_behaviour", "feedback_ablation")),
        (
            "crossdecode_gawf_identity",
            Classification("E_relevance_alignment", "crossdecode_gawf_identity"),
        ),
        (
            "clutter_best6_multiseed",
            Classification("G_behaviour", "clutter_multiseed_best_acc_bars"),
        ),
    )
    for prefix, classification in prefix_rules:
        if root.startswith(prefix):
            return classification
    return DATA_DIRECTORY_RULES.get(root)


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _destination(
    index_root: Path,
    classification: Classification,
    kind: str,
    relative: Path,
) -> Path:
    tail = (
        Path(*relative.parts[1:])
        if relative.parts[0] in CANONICAL_LEGACY_ROOTS and len(relative.parts) > 1
        else relative
    )
    return index_root / classification.category / classification.script_name / kind / tail


def _symlink_view(index_root: Path) -> list[Path]:
    return sorted(path for path in index_root.rglob("*") if path.is_symlink())


def _legacy_readme(root: Path, *, apply: bool) -> None:
    if not apply:
        return
    readme = root / "README.md"
    if readme.is_file():
        return
    readme.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text(
        "# Legacy analysis output path\n\n"
        "Unambiguously classified outputs moved to "
        "`results/anal_index/<CATEGORY>/<script_name>/`. Only files listed as ambiguous in "
        "`results/anal_index/MIGRATION_REPORT.json` may remain here.\n",
        encoding="utf-8",
    )


def migrate(
    legacy_data: Path,
    legacy_figs: Path,
    index_root: Path,
    *,
    apply: bool,
) -> dict[str, object]:
    """Plan or apply a non-overwriting move of every unambiguously classified file."""

    for category in CATEGORIES:
        if not category or Path(category).name != category:
            raise RuntimeError(f"Unsafe category declaration: {category!r}")
    removed_links = _symlink_view(index_root) if index_root.exists() else []
    moves: list[tuple[Path, Path, Classification]] = []
    ambiguous: list[dict[str, str]] = []
    for kind, root, classifier in (
        ("data", legacy_data, classify_data),
        ("figs", legacy_figs, classify_figure),
    ):
        if not root.exists():
            continue
        for source in sorted(path for path in root.rglob("*") if path.is_file()):
            if source.name in {"README.md", ".DS_Store"}:
                continue
            relative = source.relative_to(root)
            classification = classifier(relative)
            if classification is None:
                reason = AMBIGUOUS_DATA_ROOTS.get(relative.parts[0], "no explicit category rule")
                ambiguous.append({"path": f"{kind}/{relative.as_posix()}", "reason": reason})
                continue
            destination = _destination(index_root, classification, kind, relative)
            if (
                destination.exists() or destination.is_symlink()
            ) and destination not in removed_links:
                raise FileExistsError(f"Refusing to overwrite migration destination: {destination}")
            moves.append((source, destination, classification))

    destinations: dict[Path, list[Path]] = {}
    for source, destination, _classification in moves:
        destinations.setdefault(destination, []).append(source)
    duplicates = {
        destination: sources for destination, sources in destinations.items() if len(sources) > 1
    }
    if duplicates:
        details = "; ".join(
            f"{destination} <- {[str(source) for source in sources]}"
            for destination, sources in sorted(duplicates.items())
        )
        raise RuntimeError(f"Duplicate planned migration destinations: {details}")

    if apply:
        for link in removed_links:
            link.unlink()
        for source, destination, _classification in moves:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
        for root in (legacy_data, legacy_figs):
            for directory, _subdirs, _files in os.walk(root, topdown=False):
                path = Path(directory)
                if path != root and not any(path.iterdir()):
                    path.rmdir()
            _legacy_readme(root, apply=True)

        commit = _git_commit()
        timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
        grouped: dict[Classification, list[Path]] = {}
        for _source, destination, classification in moves:
            grouped.setdefault(classification, []).append(destination)
        for classification, destinations in grouped.items():
            run_root = index_root / classification.category / classification.script_name
            manifest = {
                "script_path": f"migration:legacy/{classification.script_name}",
                "git_commit": commit,
                "timestamp": timestamp,
                "category": classification.category,
                "files_written": sorted(
                    path.relative_to(run_root).as_posix() for path in destinations
                ),
                "key_numerical_results": {},
            }
            (run_root / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

    report = {
        "mode": "apply" if apply else "dry-run",
        "symlinks_removed": [str(path) for path in removed_links],
        "moves": [
            {"from": str(source), "to": str(destination)} for source, destination, _ in moves
        ],
        "ambiguous": ambiguous,
    }
    if apply:
        index_root.mkdir(parents=True, exist_ok=True)
        (index_root / "MIGRATION_REPORT.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return report


def main() -> None:
    """Print the complete plan or apply it after all collision checks pass."""

    args = parse_args()
    report = migrate(
        args.legacy_data.resolve(),
        args.legacy_figs.resolve(),
        args.index_root.resolve(),
        apply=args.apply,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["ambiguous"]:
        print(
            f"Reported {len(report['ambiguous'])} ambiguous files without moving them.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
