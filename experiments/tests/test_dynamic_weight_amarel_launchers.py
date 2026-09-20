"""Regression checks for dynamic-baseline Amarel compute runners."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNERS = (
    "run_clutter_dynamic_baselines_preflight.sh",
    "run_clutter_dynamic_baselines_preflight_aggregate.sh",
    "run_clutter_dynamic_baselines_formal.sh",
    "run_clutter_dynamic_baselines_aggregate.sh",
)


def test_compute_runners_use_absolute_git_fallback() -> None:
    """Compute-node source guards must not depend on the batch job's minimal PATH."""
    directory = ROOT / "experiments/clutter/amarel"
    for name in RUNNERS:
        text = (directory / name).read_text(encoding="utf-8")
        assert 'GIT_BIN="${AIM3_GIT_BIN:-/usr/bin/git}"' in text
        assert '"$GIT_BIN" -C "$ROOT" rev-parse HEAD' in text
        assert "$(git rev-parse HEAD)" not in text
