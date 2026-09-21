"""Regression checks for Amarel compute-node source guards.

Amarel compute nodes are not guaranteed to provide `git` at all: several GPU nodes expose neither
`git` on PATH nor `/usr/bin/git`. The guard therefore resolves the checked-out commit from git
metadata with shell builtins and falls back to the submit-time stamp written by `submit_*.sh`.
"""

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AMAREL_DIR = ROOT / "experiments/clutter/amarel"
RUNNERS = (
    "run_clutter_dynamic_baselines_preflight.sh",
    "run_clutter_dynamic_baselines_preflight_aggregate.sh",
    "run_clutter_dynamic_baselines_formal.sh",
    "run_clutter_dynamic_baselines_aggregate.sh",
    "run_clutter_feedback_controls_preflight.sh",
    "run_clutter_feedback_controls_formal.sh",
    "run_clutter_feedback_controls_aggregate.sh",
)
STAMPING_SUBMITTERS = (
    "submit_clutter_dynamic_baselines_preflight.sh",
    "submit_clutter_dynamic_baselines_formal.sh",
    "submit_clutter_feedback_controls_formal.sh",
)


def test_compute_runners_do_not_require_a_login_node_git() -> None:
    """Compute-node guards must not call git or hard-code a login-node path."""
    for name in RUNNERS:
        text = (AMAREL_DIR / name).read_text(encoding="utf-8")
        assert "/usr/bin/git" not in text, name
        assert "$(git rev-parse HEAD)" not in text, name
        assert "amarel_source_guard.sh" in text, name
        assert "amarel_require_source_commit" in text, name


def test_compute_runners_write_a_fail_marker_when_the_guard_fails() -> None:
    """A bare `exit` bypasses the ERR trap, so the guard body must write the marker itself."""
    for name in RUNNERS:
        text = (AMAREL_DIR / name).read_text(encoding="utf-8")
        guard = text.split("amarel_require_source_commit", 1)[1]
        block = guard[: guard.index("\nfi")] if "\nfi" in guard else guard
        assert "fail" in block.lower(), name


def test_compute_runners_resolve_helpers_from_the_submitted_root() -> None:
    """sbatch executes a copy under /var/lib/slurm/slurmd, so BASH_SOURCE leaves the checkout."""
    for name in RUNNERS:
        text = (AMAREL_DIR / name).read_text(encoding="utf-8")
        assert "BASH_SOURCE" not in text, name
        assert 'AMAREL_SOURCE_GUARD="$ROOT/experiments/remote/amarel_source_guard.sh"' in text, name


def test_submitters_stamp_the_submitted_commit() -> None:
    """The run-side fallback needs the submit-time commit stamp to exist."""
    for name in STAMPING_SUBMITTERS:
        text = (AMAREL_DIR / name).read_text(encoding="utf-8")
        assert "source_commit.txt" in text, name


def test_compute_runners_are_valid_bash() -> None:
    """A guard edit that breaks quoting or leaves a stray block must fail before submission."""
    for name in RUNNERS:
        completed = subprocess.run(
            ["bash", "-n", str(AMAREL_DIR / name)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, f"{name}: {completed.stderr}"


def test_formal_submitter_supports_the_preflight_dependency_chain() -> None:
    """The whole preflight -> aggregate -> formal chain must be expressible with afterok."""
    text = (AMAREL_DIR / "submit_clutter_dynamic_baselines_formal.sh").read_text(encoding="utf-8")
    assert "--afterok-preflight-aggregate" in text
    assert '--dependency="afterok:$AFTEROK_PREFLIGHT"' in text
    assert "AIM3_DYNAMIC_BASELINE_PREFLIGHT_SUMMARY" in text


def test_formal_runner_rechecks_the_preflight_gate_on_the_compute_node() -> None:
    """A chained array must re-verify the preflight summary for its own source commit."""
    text = (AMAREL_DIR / "run_clutter_dynamic_baselines_formal.sh").read_text(encoding="utf-8")
    assert "AIM3_DYNAMIC_BASELINE_PREFLIGHT_SUMMARY" in text
    assert '"status": "passed"' in text
    assert '\\"source_commit\\": \\"$SOURCE_COMMIT\\"' in text
