"""Tests for the category-indexed view of the GaWF analysis figures."""

from __future__ import annotations

import os

from utils_viz.anal_index import (
    CATEGORIES,
    FIGURE_SPECS,
    _name_candidates,
    build_links,
    classify,
    discover_figures,
    find_orphan_data,
    find_stale,
)


def _write(path: str, content: str = "x") -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)
    return path


def test_every_spec_names_a_declared_category() -> None:
    """A typo in a category letter would silently drop figures from the index."""

    for _pattern, category, _description, recommendation in FIGURE_SPECS:
        assert category in CATEGORIES
        assert recommendation in ("main", "supplementary", "superseded")


def test_category_directories_are_unique() -> None:
    """Two categories sharing a directory would merge their links."""

    dirnames = [dirname for dirname, _title, _definition in CATEGORIES.values()]
    assert len(dirnames) == len(set(dirnames))


def test_name_candidates_cover_prefix_and_suffix_interpolation() -> None:
    """Figure names are built by interpolation, so both ends must be searchable."""

    candidates = _name_candidates("clutter_best_prepost10_fg_switch_offset_acc_models")
    assert "clutter_best_prepost10_fg_switch_offset_acc_models" == candidates[0]
    assert any(c.endswith("switch_offset_acc_models") for c in candidates)
    assert any(c.startswith("clutter_best") for c in candidates)
    assert all(len(c) >= 12 for c in candidates)


def test_candidates_are_ordered_longest_first() -> None:
    """A short fragment must never outrank the specific one that identifies a script."""

    candidates = _name_candidates("01_delta_histograms_point_excluded")
    assert candidates == sorted(candidates, key=len, reverse=True)


def test_classify_is_deterministic_and_first_match_wins() -> None:
    """Ordering in FIGURE_SPECS is load-bearing; specific rules precede general ones."""

    root = "/figs"
    align, _description, _use = classify(
        root, "/figs/3_digit_v_vs_activation/fig3_x_align_matrix_zscore.png"
    )
    plain, _description, use = classify(
        root, "/figs/3_digit_v_vs_activation/fig3_x.png"
    )
    assert align == "E" and plain == "E"
    assert use == "superseded"


def test_unclassified_figures_are_reported_not_forced(tmp_path) -> None:
    """An unmatched figure must return None rather than land in an arbitrary bin."""

    category, _description, _use = classify(
        str(tmp_path), str(tmp_path / "brand_new_analysis" / "99_unknown.png")
    )
    assert category is None


def test_discover_skips_excluded_trees(tmp_path) -> None:
    """Non-gate figure trees stay out of the index instead of being force-fitted."""

    fig_root = tmp_path / "anal_figs"
    _write(str(fig_root / "gawf_gate_audit" / "01_pooled_histogram.png"))
    _write(str(fig_root / "presentation_context_demo" / "frame_00.png"))
    found = discover_figures(str(fig_root))
    assert len(found) == 1
    assert "01_pooled_histogram" in found[0]


def test_build_links_creates_relative_symlinks_without_touching_originals(
    tmp_path,
) -> None:
    """The index is a view: originals must be untouched and links must be relative."""

    fig_root = tmp_path / "anal_figs"
    original = _write(
        str(fig_root / "gawf_gate_audit" / "01_pooled_histogram.png"), "original"
    )
    before = os.stat(original)
    index_dir = tmp_path / "anal_index"
    records = [
        {
            "path": original,
            "category": "A",
            "description": "",
            "recommendation": "main",
            "script": "utils_viz/gawf_gate_distribution.py:82",
            "data": [],
        }
    ]
    assert build_links(str(index_dir), records, str(fig_root)) == 1
    link = index_dir / "A_raw_gate" / "gawf_gate_audit__01_pooled_histogram.png"
    assert os.path.islink(link)
    assert not os.path.isabs(os.readlink(link))
    assert os.path.exists(link), "symlink must resolve"
    after = os.stat(original)
    assert (before.st_mtime, before.st_size) == (after.st_mtime, after.st_size)
    with open(link, "r", encoding="utf-8") as handle:
        assert handle.read() == "original"


def test_build_links_is_idempotent(tmp_path) -> None:
    """Rerunning must replace stale links rather than fail on an existing name."""

    fig_root = tmp_path / "anal_figs"
    original = _write(str(fig_root / "gawf_gate_audit" / "01_pooled_histogram.png"))
    index_dir = tmp_path / "anal_index"
    records = [
        {
            "path": original,
            "category": "A",
            "description": "",
            "recommendation": "main",
            "script": "x",
            "data": [],
        }
    ]
    build_links(str(index_dir), records, str(fig_root))
    assert build_links(str(index_dir), records, str(fig_root)) == 1
    assert len(os.listdir(index_dir / "A_raw_gate")) == 1


def test_find_stale_flags_a_figure_older_than_its_script(tmp_path) -> None:
    """The only automatic signal for a pre-bug-fix figure is its modification time."""

    figure = _write(str(tmp_path / "fig.png"))
    script = _write(os.path.join(os.path.dirname(__file__), "_tmp_stale_probe.py"))
    try:
        os.utime(figure, (1_000_000, 1_000_000))
        os.utime(script, (2_000_000, 2_000_000))
        record = {
            "path": figure,
            "script": f"tests/{os.path.basename(script)}:1",
            "data": [],
        }
        stale = find_stale([record])
        assert len(stale) == 1
        assert stale[0]["lag_minutes"] > 0

        os.utime(figure, (3_000_000, 3_000_000))
        assert find_stale([record]) == []
    finally:
        os.remove(script)


def test_find_stale_ignores_unknown_scripts(tmp_path) -> None:
    """An unresolved script cannot support a staleness claim either way."""

    figure = _write(str(tmp_path / "fig.png"))
    assert find_stale([{"path": figure, "script": "UNKNOWN", "data": []}]) == []


def test_orphan_data_reports_unpaired_files_in_indexed_directories(tmp_path) -> None:
    """A saved array no figure draws on is a reproducibility gap worth naming."""

    fig_root = tmp_path / "anal_figs"
    data_root = tmp_path / "anal_data"
    figure = _write(str(fig_root / "gawf_gate_robustness" / "01_delta_survival.png"))
    paired = _write(str(data_root / "gawf_gate_robustness" / "delta_survival.npz"))
    orphan = _write(str(data_root / "gawf_gate_robustness" / "part4_leave_one_out.csv"))
    _write(str(data_root / "unrelated_export" / "ignore_me.csv"))
    records = [
        {
            "path": figure,
            "category": "C",
            "description": "",
            "recommendation": "main",
            "script": "x",
            "data": sorted([paired, orphan]),
        }
    ]
    orphans = find_orphan_data(str(data_root), records, str(fig_root))
    assert orphan in orphans
    assert paired not in orphans
    assert not any("unrelated_export" in path for path in orphans), (
        "directories with no indexed figure must not be searched"
    )
