"""Tests for safe entity migration into the analysis category tree."""

from __future__ import annotations

from pathlib import Path

from utils_anal.migrate_analysis_outputs import migrate


def _write(path: Path, content: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_dry_run_does_not_move_or_remove_symlinks(tmp_path: Path) -> None:
    data = tmp_path / "anal_data"
    figs = tmp_path / "anal_figs"
    index = tmp_path / "anal_index"
    source = _write(data / "cnn_channel" / "stats.npz")
    target = _write(figs / "cnn_channel" / "matrix.png")
    index.mkdir()
    link = index / "old-link.png"
    link.symlink_to(target)
    report = migrate(data, figs, index, apply=False)
    assert source.exists() and target.exists() and link.is_symlink()
    assert len(report["moves"]) == 2
    assert len(report["symlinks_removed"]) == 1


def test_apply_moves_known_files_and_reports_mixed_data(tmp_path: Path) -> None:
    data = tmp_path / "anal_data"
    figs = tmp_path / "anal_figs"
    index = tmp_path / "anal_index"
    known = _write(data / "cnn_channel" / "stats.npz")
    ambiguous = _write(data / "gawf_gate_audit" / "gawf_gate_distribution_stats.npz")
    figure = _write(figs / "gawf_gate_audit" / "01_pooled_histogram.png")
    report = migrate(data, figs, index, apply=True)
    assert not known.exists() and not figure.exists()
    assert ambiguous.exists(), "mixed artifact must wait for human classification"
    assert (
        index / "E_relevance_alignment" / "cnn_channel_stats" / "data" / "stats.npz"
    ).is_file()
    assert (
        index / "A_raw_gate" / "gawf_gate_distribution" / "figs" / "01_pooled_histogram.png"
    ).is_file()
    assert len(report["ambiguous"]) == 1
    assert (data / "README.md").is_file() and (figs / "README.md").is_file()


def test_apply_refuses_to_overwrite_destination(tmp_path: Path) -> None:
    data = tmp_path / "anal_data"
    figs = tmp_path / "anal_figs"
    index = tmp_path / "anal_index"
    _write(data / "cnn_channel" / "stats.npz", "old")
    _write(
        index / "E_relevance_alignment" / "cnn_channel_stats" / "data" / "stats.npz",
        "new",
    )
    try:
        migrate(data, figs, index, apply=True)
    except FileExistsError:
        pass
    else:
        raise AssertionError("migration must not overwrite an existing destination")


def test_campaign_roots_remain_distinct(tmp_path: Path) -> None:
    data = tmp_path / "anal_data"
    figs = tmp_path / "anal_figs"
    index = tmp_path / "anal_index"
    first = _write(data / "feedback_ablation" / "ablation_metrics.csv", "first")
    second = _write(
        data / "feedback_ablation_jointswitch_balanced" / "ablation_metrics.csv", "second"
    )
    report = migrate(data, figs, index, apply=False)
    destinations = {Path(row["to"]) for row in report["moves"]}
    assert len(destinations) == 2
    assert all(source.exists() for source in (first, second))


def test_apply_preserves_existing_legacy_readme(tmp_path: Path) -> None:
    data = tmp_path / "anal_data"
    figs = tmp_path / "anal_figs"
    index = tmp_path / "anal_index"
    readme = _write(data / "README.md", "tracked guidance\n")
    _write(data / "cnn_channel" / "stats.npz")
    migrate(data, figs, index, apply=True)
    assert readme.read_text(encoding="utf-8") == "tracked guidance\n"
