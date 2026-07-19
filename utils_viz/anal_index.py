"""Build a category-indexed view of the GaWF analysis figures.

The analysis and plotting scripts write to hard-coded output paths, so the figures
cannot be reorganized in place -- a rerun would rebuild the old layout and leave two
inconsistent trees. This tool therefore builds a *view*: ``results/anal_index/``
holds one subdirectory per analysis category, populated with relative symlinks back
to the originals, which are never moved, modified, or deleted.

Categories A-D form a ladder over the same gate values, each level stripping one more
layer of structure; E-H are orthogonal to that ladder. See ``CATEGORIES``.

Facts that can go stale (modification times, missing files, figures older than the
script that makes them) are recomputed on every run. Human judgement that cannot be
derived from the filesystem lives in ``NOTES.md`` and is embedded verbatim, so
rerunning this tool never overwrites an analysis conclusion.

Usage:
    python utils_viz/anal_index.py
    python utils_viz/anal_index.py --check      # report only, write nothing
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

FIGURE_SUFFIXES = (".png", ".pdf", ".svg")
DATA_SUFFIXES = (".json", ".csv", ".npz", ".npy")
SCRIPT_DIRS = ("utils_viz", "utils_anal")
SELF_FILENAME = os.path.basename(os.path.abspath(__file__))

CATEGORIES: dict[str, tuple[str, str, str]] = {
    "A": (
        "A_raw_gate",
        "Raw gate distribution",
        "Gate values only. No condition labels enter the computation.",
    ),
    "B": (
        "B_gate_by_context",
        "Gate distribution split by context",
        "Condition labels partition trials only. Plotted values are raw gate values, "
        "not baseline-subtracted.",
    ),
    "C": (
        "C_delta_gate",
        "Delta-g distribution",
        "Per-synapse grand mean subtracted; plotted values are deviations.",
    ),
    "D": (
        "D_variance_decomposition",
        "Variance decomposition",
        "Apportions variance to sector / digit / interaction. Output is variance "
        "fractions or eta-squared. Classified by output type, not by whether the input "
        "array was pre-centered: decomposing raw g and delta-g give identical results "
        "because the ANOVA subtracts the per-synapse grand mean first.",
    ),
    "E": (
        "E_relevance_alignment",
        "Relevance / alignment",
        "Requires an external label derived from activations, not gates. Output is a "
        "two-group contrast (Cohen's d) or an alignment matrix.",
    ),
    "F": (
        "F_timing",
        "Timing / causality",
        "Indexed by post-switch frame or per-event latency.",
    ),
    "G": (
        "G_behaviour",
        "Behaviour",
        "Task performance only; no gate internals.",
    ),
    "H": (
        "H_controls",
        "Methodological control / robustness",
        "Supports the validity of another analysis rather than a scientific claim in "
        "its own right.",
    ),
}

# (regex on the repo-relative figure path, category, description, recommended use).
# First match wins, so order matters: put specific patterns before general ones.
FIGURE_SPECS: tuple[tuple[str, str, str, str], ...] = (
    (r"gawf_gate_audit/01_pooled_histogram", "A",
     "Pooled input/recurrent gate-value histogram over all synapses and trials.", "main"),
    (r"gawf_gate_audit/02_weight_sign_histogram", "A",
     "Gate histogram split by the sign of the underlying weight.", "supplementary"),
    (r"gawf_gate_audit/07_effective_weight", "A",
     "W vs G*W effective-weight distribution.", "supplementary"),
    (r"gawf_gate_audit/04_per_context_histogram", "B",
     "Raw gate histogram per sector (9 levels).", "main"),
    (r"gawf_gate_audit/04_per_digit_histogram", "B",
     "Raw gate histogram per digit (10 levels).", "supplementary"),
    (r"gawf_gate_audit/06_sparsity_by_context", "B",
     "Gini / normalized PR (Treves-Rolls) per sector.", "supplementary"),
    (r"gawf_gate_audit/06_sparsity_by_digit", "B",
     "Gini / normalized PR per digit.", "supplementary"),
    (r"2_sector_sigmoid_gate/fig2_sector_gate_mean", "B",
     "Raw mean-gate spatial map per sector.", "supplementary"),
    (r"2_sector_sigmoid_gate/fig2_sector_gate_max", "B",
     "Raw max-gate spatial map per sector.", "superseded"),
    (r"1_sector_v_modulation/fig1_sector_v_maps", "B",
     "Per-sector pre-sigmoid V modulation maps (ambiguous A/B: not gate values proper).",
     "supplementary"),
    (r"gawf_gate_audit/03_sector_digit_group_mean_delta_histogram", "C",
     "Combined sector/digit group-mean delta-g histogram on shared bin edges.", "superseded"),
    (r"gawf_gate_audit/03_sector_centered_gate_histogram", "C",
     "Sector-centered gate histogram.", "supplementary"),
    (r"gawf_gate_audit_digit/03_digit_centered_gate_histogram", "C",
     "Digit-centered gate histogram.", "supplementary"),
    (r"gawf_gate_context_specificity/01_delta_histograms_point_excluded", "C",
     "Group-mean delta-g histograms, 0.5 point mass excluded.", "main"),
    (r"gawf_gate_context_specificity/01_delta_histograms_point_included", "C",
     "Group-mean delta-g histograms, 0.5 point mass included.", "supplementary"),
    (r"gawf_gate_context_specificity/02_input_spatial_maps", "C",
     "Input-gate spatial delta-g maps.", "supplementary"),
    (r"gawf_gate_robustness/01_delta_survival", "C",
     "Survival functions P(|delta-g| > t), log-y (ambiguous C/H).", "main"),
    (r"gawf_gate_context_specificity/03_variance_decomposition", "D",
     "Gate variance apportioned to sector / digit / interaction / residual.", "main"),
    (r"gawf_gate_context_specificity/04_encoder_control_decomposition", "D",
     "Same decomposition applied to encoder activations as a control.", "supplementary"),
    (r"gawf_symmetric_relevance_timing/part1_selectivity", "D",
     "Per-unit sector/digit selectivity (eta-squared).", "supplementary"),
    (r"gawf_symmetric_relevance_timing/part1_architecture_axis", "D",
     "Spatial vs channel architecture-axis variance test.", "supplementary"),
    (r"rnn_unit_gate_context_specificity/03_.*_unit_gate_variance_decomposition", "D",
     "LSTM/GRU unit-level gate variance decomposition.", "supplementary"),
    (r"dpca_marginalized_variance_compare", "D",
     "Cross-model dPCA marginalized variance comparison.", "supplementary"),
    (r"gawf_gate_audit/05_task_relevance_histogram", "E",
     "Gate histogram split by activation-derived task-relevance proxy.", "supplementary"),
    (r"gawf_symmetric_relevance_timing/part2_relevance_effects", "E",
     "Symmetric 2x2 relevance effects, Cohen's d.", "main"),
    (r"gawf_symmetric_relevance_timing/part2_continuous_alignment", "E",
     "Continuous activation/gate alignment.", "supplementary"),
    (r"3_digit_v_vs_activation/.*align_matrix", "E",
     "Digit gate-tuning x activation-tuning cosine alignment matrix.", "supplementary"),
    (r"3_digit_v_vs_activation/.*_zscore", "E",
     "Z-scored per-digit gate mean vs activation.", "supplementary"),
    (r"3_digit_v_vs_activation/", "E",
     "Per-digit gate mean vs activation scatter.", "superseded"),
    (r"crossdecode_gawf_identity.*/fig_align_matrix", "E",
     "Cross-decode alignment matrix.", "supplementary"),
    (r"crossdecode_gawf_identity.*/fig_crossdecode_confusion", "E",
     "Cross-decode confusion matrix.", "supplementary"),
    (r"gawf_symmetric_relevance_timing/part3_switch_timing", "F",
     "Gate reconfiguration vs readout recovery by post-switch frame.", "main"),
    (r"gawf_symmetric_relevance_timing/part3_per_event_lead", "F",
     "Paired per-event gate-lead latency distribution.", "supplementary"),
    (r"4_feedback_ablation/[^/]+/fig_ablation_switch_recovery", "G",
     "Post-switch recovery curve under feedback ablation.", "main"),
    (r"4_feedback_ablation/[^/]+/fig_ablation_2x2", "G",
     "2x2 feedback-ablation accuracy.", "supplementary"),
    (r"4_feedback_ablation/fig_ablation", "G",
     "Feedback ablation (duplicate of the subdirectory run).", "superseded"),
    (r"fg_switch_offset_acc/fg/clutter_best_jointswitch_balanced_prepost10", "G",
     "Foreground switch-offset accuracy across models, primary condition.", "main"),
    (r"fg_switch_offset_acc/fg/", "G",
     "Foreground switch-offset accuracy, dataset-config variant.", "supplementary"),
    (r"fg_switch_offset_acc/bg/", "G",
     "Background switch-offset accuracy control.", "supplementary"),
    (r"gawf_gate_robustness/02_ci_width_convergence", "H",
     "95% bootstrap-CI width vs sampled-synapse count.", "supplementary"),
    (r"gawf_gate_context_specificity/06_digit_gini_ink_regression", "H",
     "Decisive ink-area confound regression on per-digit Gini.", "supplementary"),
    (r"gawf_gate_context_specificity/05_digit_metric_gini_scatter", "H",
     "Per-digit Gini vs 5 low-level image metrics, 2x5 grid.", "supplementary"),
    (r"gawf_gate_context_specificity/07_sector_symmetric_control", "H",
     "Sector-factor version of the metric scatter grid, as symmetry control.", "supplementary"),
    (r"gawf_gate_context_specificity/08_digit_variance_contribution_scatter", "H",
     "Per-digit variance contribution vs the same 5 metrics.", "superseded"),
)

# Figure trees deliberately excluded: they are not GaWF gate analyses and forcing them
# into A-H would require a ninth category to stay honest.
EXCLUDED_TREES: tuple[tuple[str, str], ...] = (
    ("presentation_context_demo", "Stimulus illustrations, not analysis output."),
    ("5_pop_act_umap/", "Population-activity geometry; decomposes activations, not gates."),
    ("5_pop_act_umap_multiseg/dpca_scatter", "Population-activity geometry, not gates."),
    ("archive/", "Superseded layout kept for provenance."),
)


def parse_args() -> argparse.Namespace:
    """Parse index-building arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fig_root", default="./results/anal_figs")
    parser.add_argument("--data_root", default="./results/anal_data")
    parser.add_argument("--index_dir", default="./results/anal_index")
    parser.add_argument(
        "--notes",
        default="./results/anal_index/NOTES.md",
        help="Markdown file embedded verbatim; never written by this tool.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report classification and problems without writing anything.",
    )
    return parser.parse_args()


def _mtime(path: str) -> dt.datetime:
    return dt.datetime.fromtimestamp(os.path.getmtime(path))


def _stamp(path: str) -> str:
    return _mtime(path).strftime("%Y-%m-%d %H:%M")


def discover_figures(fig_root: str) -> list[str]:
    """Return every figure under ``fig_root`` that is not in an excluded tree."""

    found = []
    for dirpath, _dirnames, filenames in os.walk(fig_root):
        for filename in sorted(filenames):
            if not filename.lower().endswith(FIGURE_SUFFIXES):
                continue
            path = os.path.join(dirpath, filename)
            relative = os.path.relpath(path, fig_root)
            if any(marker in relative for marker, _why in EXCLUDED_TREES):
                continue
            found.append(path)
    return sorted(found)


def classify(fig_root: str, path: str) -> tuple[str | None, str, str]:
    """Return ``(category, description, recommendation)`` for one figure."""

    relative = os.path.relpath(path, fig_root)
    for pattern, category, description, recommendation in FIGURE_SPECS:
        if re.search(pattern, relative):
            return category, description, recommendation
    return None, "", ""


MIN_CANDIDATE_CHARS = 12
# A line that builds or writes a path identifies the generator; a line that merely
# mentions a filename is usually the module docstring listing its outputs, which
# names the right file but the wrong line.
WRITE_MARKERS = ("os.path.join", "savefig", "_save(", "_finish(", "_save_figure(")
MENTION_MARKERS = (".png", ".pdf", ".svg")


def _name_candidates(stem: str) -> list[str]:
    """Return literal fragments to search for, longest first.

    Figure names are frequently built by interpolation -- a prefix
    (``f"{prefix}_switch_offset_acc_models.png"``), a suffix
    (``f"01_pooled_histogram.{suffix}"``), or a middle token
    (``f"fig3_digit_sigmoid_{label}_vs_activation.png"``). Only part of the name is
    ever a literal, so try progressively shorter prefixes and suffixes of the stem
    and keep the longest fragment that actually appears in a source line.
    """

    tokens = stem.split("_")
    candidates = [stem]
    for size in range(len(tokens) - 1, 1, -1):
        candidates.append("_".join(tokens[:size]))
        candidates.append("_".join(tokens[-size:]))
    seen: set[str] = set()
    unique = []
    for candidate in candidates:
        if len(candidate) >= MIN_CANDIDATE_CHARS and candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return sorted(unique, key=len, reverse=True)


def find_generating_script(path: str) -> str:
    """Locate the script that writes this figure, by searching for its basename stem.

    Returns ``file:line`` when a literal or f-string reference is found, otherwise
    ``UNKNOWN`` -- guessing here would be worse than admitting the gap. Lines that
    look like path construction win over prose, so a filename mentioned in a module
    docstring never masks the line that actually writes it.
    """

    stem = os.path.splitext(os.path.basename(path))[0]
    sources = _load_sources()
    fallback = ""
    # Candidates are ordered longest first and must be the OUTER loop: a short
    # fragment matching an alphabetically earlier script would otherwise beat the
    # long, specific fragment that identifies the real generator.
    for candidate in _name_candidates(stem):
        hits = [(location, line) for location, line in sources if candidate in line]
        if not hits:
            continue
        for location, line in hits:
            if any(marker in line for marker in WRITE_MARKERS):
                return location
        # An exact full-stem hit is decisive even without a write marker: no other
        # figure shares the name. Names assembled as `combined_name = f"...{suffix}"`
        # carry neither a marker nor a literal extension on the line that builds
        # them, so requiring one would misattribute them to another script. A write
        # site elsewhere still wins, which keeps a mere `np.load` of the matching
        # data file from being reported as the generator.
        if candidate == stem:
            return hits[0][0]
        if not fallback:
            for location, line in hits:
                if any(marker in line for marker in MENTION_MARKERS):
                    fallback = location
                    break
    return fallback or "UNKNOWN"


def _load_sources() -> list[tuple[str, str]]:
    """Return ``(file:line, text)`` for every scannable analysis and plotting line."""

    if _load_sources.cache is not None:
        return _load_sources.cache
    collected: list[tuple[str, str]] = []
    for script_dir in SCRIPT_DIRS:
        directory = os.path.join(PROJECT_ROOT, script_dir)
        if not os.path.isdir(directory):
            continue
        for filename in sorted(os.listdir(directory)):
            if not filename.endswith(".py") or filename == SELF_FILENAME:
                # FIGURE_SPECS lists every figure name, so this file matches all of
                # them; skipping it stops the index from naming itself as the source.
                continue
            script = os.path.join(directory, filename)
            try:
                with open(script, "r", encoding="utf-8") as handle:
                    lines = handle.readlines()
            except (OSError, UnicodeDecodeError):
                continue
            collected += [
                (f"{script_dir}/{filename}:{number}", line)
                for number, line in enumerate(lines, start=1)
            ]
    _load_sources.cache = collected
    return collected


_load_sources.cache = None


def pair_data_files(data_root: str, fig_root: str, path: str) -> list[str]:
    """Return data files sitting in the directory matching this figure's directory."""

    relative_dir = os.path.relpath(os.path.dirname(path), fig_root)
    data_dir = os.path.join(data_root, relative_dir)
    if not os.path.isdir(data_dir):
        return []
    return sorted(
        os.path.join(data_dir, name)
        for name in os.listdir(data_dir)
        if name.lower().endswith(DATA_SUFFIXES)
    )


def find_stale(records: list[dict]) -> list[dict]:
    """Flag figures older than the script that generates them.

    This is the only automatic way to catch a figure that predates a bug fix: the
    filename is identical either way, so modification time is the sole signal.
    """

    stale = []
    for record in records:
        script = record["script"]
        if script == "UNKNOWN":
            continue
        script_path = os.path.join(PROJECT_ROOT, script.split(":")[0])
        if not os.path.isfile(script_path):
            continue
        if _mtime(record["path"]) < _mtime(script_path):
            stale.append(
                {
                    **record,
                    "script_mtime": _stamp(script_path),
                    "lag_minutes": int(
                        (
                            _mtime(script_path) - _mtime(record["path"])
                        ).total_seconds()
                        // 60
                    ),
                }
            )
    return stale


def find_orphan_data(data_root: str, records: list[dict], fig_root: str) -> list[str]:
    """Return data files with no figure, restricted to indexed analyses.

    Only directories that an indexed figure already points at are searched. Walking
    all of ``anal_data`` would report every unrelated export as an orphan, which
    buries the cases that matter: a saved array inside an analysis that *does*
    produce figures, but which no figure draws on.
    """

    relevant_dirs = {
        os.path.join(data_root, os.path.relpath(os.path.dirname(record["path"]), fig_root))
        for record in records
    }
    paired = {
        data
        for record in records
        for data in _figure_specific_data(record)
    }
    orphans = []
    for directory in sorted(relevant_dirs):
        if not os.path.isdir(directory):
            continue
        for filename in sorted(os.listdir(directory)):
            if not filename.lower().endswith(DATA_SUFFIXES):
                continue
            path = os.path.join(directory, filename)
            if path not in paired:
                orphans.append(path)
    return sorted(orphans)


def _figure_specific_data(record: dict) -> list[str]:
    """Return the data files a figure plausibly reads, by stem overlap.

    A figure and its data share naming tokens far more often than not, so a token
    overlap is a better signal than 'sits in the same directory' -- the latter marks
    every file in a directory as paired and makes orphan detection vacuous.
    """

    stem = os.path.splitext(os.path.basename(record["path"]))[0]
    tokens = {token for token in stem.split("_") if len(token) > 3}
    matched = []
    for data in record["data"]:
        data_tokens = {
            token
            for token in os.path.splitext(os.path.basename(data))[0].split("_")
            if len(token) > 3
        }
        if tokens & data_tokens:
            matched.append(data)
    return matched or record["data"]


def find_missing_declared(fig_root: str) -> list[str]:
    """Return figures that FIGURE_SPECS names but that are absent from disk."""

    present = {
        os.path.relpath(path, fig_root) for path in discover_figures(fig_root)
    }
    missing = []
    for pattern, _category, _description, _recommendation in FIGURE_SPECS:
        if re.search(r"[\[\](){}*+?|\\]", pattern):
            continue
        if not any(re.search(pattern, name) for name in present):
            missing.append(pattern)
    return missing


def build_links(index_dir: str, records: list[dict], fig_root: str) -> int:
    """Populate the category tree with relative symlinks, replacing any stale links."""

    for _letter, (dirname, _title, _definition) in sorted(CATEGORIES.items()):
        os.makedirs(os.path.join(index_dir, dirname), exist_ok=True)
    made = 0
    for record in records:
        dirname = CATEGORIES[record["category"]][0]
        relative = os.path.relpath(record["path"], fig_root)
        provenance = os.path.dirname(relative).replace(os.sep, "__")
        link_name = f"{provenance}__{os.path.basename(relative)}"
        link_path = os.path.join(index_dir, dirname, link_name)
        if os.path.islink(link_path) or os.path.exists(link_path):
            os.remove(link_path)
        target = os.path.relpath(record["path"], os.path.dirname(link_path))
        try:
            os.symlink(target, link_path)
        except OSError:
            import shutil

            shutil.copy2(record["path"], link_path)
        made += 1
    return made


def _table(records: list[dict], fig_root: str, data_root: str) -> list[str]:
    lines = [
        "| Figure | Generating script | Paired data | Modified | Description |",
        "|---|---|---|---|---|",
    ]
    for record in sorted(records, key=lambda item: item["path"]):
        data = record["data"]
        data_cell = (
            os.path.relpath(data[0], data_root) if len(data) == 1
            else f"{len(data)} files in `{os.path.relpath(os.path.dirname(data[0]), data_root)}/`"
            if data
            else "**none**"
        )
        lines.append(
            f"| `{os.path.relpath(record['path'], fig_root)}` "
            f"| `{record['script']}` | {data_cell} "
            f"| {_stamp(record['path'])} | {record['description']} |"
        )
    return lines


def write_index(
    index_dir: str,
    records: list[dict],
    unclassified: list[str],
    stale: list[dict],
    orphans: list[str],
    missing: list[str],
    fig_root: str,
    data_root: str,
    notes_path: str,
) -> str:
    """Write INDEX.md from live filesystem facts plus the verbatim notes file."""

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# GaWF Analysis Output Index",
        "",
        f"Generated {now} by `utils_viz/anal_index.py`. **This directory is a view, "
        "not a store.** Every entry is a relative symlink into `results/anal_figs/`; "
        "originals are never moved, modified, or deleted, because the generating "
        "scripts have hard-coded output paths.",
        "",
        f"{len(records)} figures indexed across {len(CATEGORIES)} categories.",
        "",
        "## Category definitions",
        "",
        "A-D form a ladder over the same gate values, each level stripping one more "
        "layer of structure. E-H are orthogonal to that ladder.",
        "",
        "| | Category | Operational criterion |",
        "|---|---|---|",
    ]
    for letter, (_dirname, title, definition) in sorted(CATEGORIES.items()):
        lines.append(f"| **{letter}** | {title.upper()} | {definition} |")
    lines.append("")

    for letter, (dirname, title, _definition) in sorted(CATEGORIES.items()):
        subset = [r for r in records if r["category"] == letter]
        lines += [f"## {letter} — {title} ({len(subset)} figures)", ""]
        lines += _table(subset, fig_root, data_root) if subset else ["_No figures._"]
        lines += ["", f"Linked in `{dirname}/`.", ""]

    lines += ["---", "", "# Problem lists", ""]

    lines += [f"## 1. Unclassifiable ({len(unclassified)})", ""]
    if unclassified:
        lines.append(
            "These figures matched no rule in `FIGURE_SPECS` and were **not** linked. "
            "Add a rule or record them as deliberately excluded — do not force-fit."
        )
        lines += [f"- `{os.path.relpath(p, fig_root)}`" for p in unclassified]
    else:
        lines.append("None. Trees deliberately excluded from the index:")
        lines += [f"- `{marker}` — {why}" for marker, why in EXCLUDED_TREES]
    lines.append("")

    lines += [f"## 2. Declared but missing from disk ({len(missing)})", ""]
    lines += (
        [f"- `{pattern}`" for pattern in missing]
        if missing
        else ["None."]
    )
    lines.append("")

    lines += [f"## 3. Orphan data files — no paired figure ({len(orphans)})", ""]
    if orphans:
        lines.append(
            "A saved array with no figure is not itself a problem, but a conclusion "
            "that rests on one is invisible in the figure set."
        )
        lines += [f"- `{os.path.relpath(p, data_root)}`" for p in orphans]
    else:
        lines.append("None.")
    lines.append("")

    lines += [f"## 4. Stale — figure older than its generating script ({len(stale)})", ""]
    if stale:
        lines.append(
            "A figure produced before its script was fixed is indistinguishable from a "
            "current one by filename alone. Modification time is the only automatic signal."
        )
        lines += [
            "",
            "| Figure | Figure mtime | Script | Script mtime | Lag (min) |",
            "|---|---|---|---|---|",
        ]
        lines += [
            f"| `{os.path.relpath(r['path'], fig_root)}` | {_stamp(r['path'])} "
            f"| `{r['script']}` | {r['script_mtime']} | {r['lag_minutes']} |"
            for r in stale
        ]
    else:
        lines.append("None — every figure is at least as new as its generating script.")
    lines.append("")

    lines += ["---", "", "# Recommended use", ""]
    lines += [
        "Within each of A-D only one representative figure is normally needed.",
        "",
        "| Figure | Category | Recommended use |",
        "|---|---|---|",
    ]
    label = {
        "main": "**Main figure**",
        "supplementary": "Supplementary",
        "superseded": "**Superseded**",
    }
    for record in sorted(records, key=lambda item: (item["category"], item["path"])):
        lines.append(
            f"| `{os.path.relpath(record['path'], fig_root)}` | {record['category']} "
            f"| {label.get(record['recommendation'], record['recommendation'])} |"
        )
    lines.append("")

    if os.path.isfile(notes_path):
        with open(notes_path, "r", encoding="utf-8") as handle:
            notes = handle.read().strip()
        lines += [
            "---",
            "",
            f"# Notes (verbatim from `{os.path.basename(notes_path)}`)",
            "",
            notes,
            "",
        ]

    os.makedirs(index_dir, exist_ok=True)
    output_path = os.path.join(index_dir, "INDEX.md")
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return output_path


def main() -> None:
    """Classify every figure, build the symlink view, and write INDEX.md."""

    args = parse_args()
    fig_root = os.path.abspath(args.fig_root)
    data_root = os.path.abspath(args.data_root)
    index_dir = os.path.abspath(args.index_dir)
    if not os.path.isdir(fig_root):
        raise SystemExit(f"Figure root not found: {fig_root}")

    records: list[dict] = []
    unclassified: list[str] = []
    for path in discover_figures(fig_root):
        category, description, recommendation = classify(fig_root, path)
        if category is None:
            unclassified.append(path)
            continue
        records.append(
            {
                "path": path,
                "category": category,
                "description": description,
                "recommendation": recommendation,
                "script": find_generating_script(path),
                "data": pair_data_files(data_root, fig_root, path),
            }
        )

    stale = find_stale(records)
    orphans = find_orphan_data(data_root, records, fig_root)
    missing = find_missing_declared(fig_root)
    unknown = [r for r in records if r["script"] == "UNKNOWN"]

    print(f"Indexed {len(records)} figures; {len(unclassified)} unclassified.")
    for letter, (dirname, _title, _definition) in sorted(CATEGORIES.items()):
        count = sum(1 for r in records if r["category"] == letter)
        print(f"  {letter} {dirname:<26} {count}")
    print(f"Generating script UNKNOWN: {len(unknown)}")
    print(f"Stale figures: {len(stale)}")
    print(f"Orphan data files: {len(orphans)}")
    print(f"Declared but missing: {len(missing)}")
    for path in unclassified:
        print(f"  UNCLASSIFIED {os.path.relpath(path, fig_root)}")

    if args.check:
        print("--check: nothing written.")
        return

    made = build_links(index_dir, records, fig_root)
    print(f"Wrote {made} symlinks under {index_dir}")
    output_path = write_index(
        index_dir,
        records,
        unclassified,
        stale,
        orphans,
        missing,
        fig_root,
        data_root,
        os.path.abspath(args.notes),
    )
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
