import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBMITTERS = (
    ROOT / "experiments/amarel/submit_atari_pong_fs1_stack1.sh",
    ROOT / "experiments/amarel/submit_atari_pong_depth2.sh",
)

# Reviewed, bounded standard-library metadata commands that predate the global guard. New
# submitters must not be added here: move their Python work to an sbatch preflight runner.
REVIEWED_METADATA_PYTHON = {
    "submit_atari_pong_depth2.sh": ('/usr/bin/python3 - "$MATCH_JSON"',),
    "submit_atari_pong_fs1_stack1.sh": ('/usr/bin/python3 - "$MATCH_JSON"',),
    "submit_gawf_single_feedback_lr_grid.sh": ('python "$GRID_UTIL" list-task-ids',),
    "submit_hparam_param_match.sh": ('python "$GRID_UTIL" list-task-ids',),
    "submit_imdb_5model_full50_grid.sh": ('GRID_UTIL_PATH="$GRID_UTIL" python -',),
    "submit_imdb_gawf_depth_grid.sh": ('python "$GRID_UTIL" list-task-ids',),
    "submit_imdb_hparam_grid_batches.sh": ("python -c",),
    "submit_ssm_mamba_hparam_grid_batches.sh": (
        'python "$GRID_UTIL" emit-task',
        "python -c",
    ),
}

BANNED_LOGIN_NODE_PATTERNS = {
    "Conda activation": r"\bconda\s+activate\b",
    "Python module execution": r"\bpython(?:3)?\s+-m\b",
    "distributed launcher": r"\b(?:torchrun|accelerate\s+launch)\b",
    "direct Slurm execution": r"\bsrun\b",
    "workload Python selection": r"\b(?:AIM3_)?PYTHON\s*=|\$\{?PYTHON\}?",
    "ML/scientific import": (
        r"(?:^|[;\s])(?:from|import)\s+"
        r"(?:torch|numpy|jax|tensorflow|cupy|sklearn|mamba_ssm|s5)(?:\b|\.)"
    ),
    "training entry point": r"\bpython(?:3)?\b[^#\n]*(?:train_|benchmark_|visuali[sz]e)[^\s]*\.py\b",
}


def _active_lines(text: str) -> list[str]:
    """Return non-comment, non-message lines from a shell submitter."""

    active = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if re.match(r"^(?:echo|printf|log)\b", stripped):
            continue
        active.append(line)
    return active


def _python_invocation_lines(lines: list[str]) -> list[str]:
    pattern = re.compile(r"(?:^|[;&|()\s])(?:/usr/bin/)?python(?:3)?(?:\s|$)")
    return [line.strip() for line in lines if pattern.search(line)]


def test_pytorch_parameter_matching_is_not_run_by_login_node_submitters() -> None:
    unsafe_command = "python -m experiments.atari.atari_ssm_param_match"
    for path in SUBMITTERS:
        text = path.read_text(encoding="utf-8")
        assert unsafe_command not in text
        assert "run_atari_param_match.sh" in text
        assert "afterok:" in text


def test_parameter_matching_runner_requests_a_compute_node() -> None:
    path = ROOT / "experiments/amarel/run_atari_param_match.sh"
    text = path.read_text(encoding="utf-8")
    assert "#SBATCH --partition=gpu-redhat" in text
    assert "#SBATCH --gres=gpu:1" in text
    assert "python -m experiments.atari.atari_ssm_param_match" in text


def test_all_amarel_submitters_keep_compute_off_login_nodes() -> None:
    submitters = sorted((ROOT / "experiments/amarel").glob("submit_*.sh"))
    assert submitters

    for path in submitters:
        active_lines = _active_lines(path.read_text(encoding="utf-8"))
        active_text = "\n".join(active_lines)
        for label, pattern in BANNED_LOGIN_NODE_PATTERNS.items():
            assert not re.search(pattern, active_text, flags=re.MULTILINE), (
                f"{path.name} contains forbidden login-node {label}; move it to an sbatch "
                "run_*.sh/preflight job"
            )

        allowed = REVIEWED_METADATA_PYTHON.get(path.name, ())
        for invocation in _python_invocation_lines(active_lines):
            assert any(fragment in invocation for fragment in allowed), (
                f"{path.name} has an unreviewed Python command: {invocation}. "
                "Do not extend the allowlist; move it to an sbatch preflight job."
            )


def test_agent_constraints_require_the_amarel_safety_gate() -> None:
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    runbook = (ROOT / "docs/operations/REMOTE_EXECUTION.md").read_text(encoding="utf-8")
    command = "python -m pytest -q tests/test_amarel_submit_safety.py"
    assert "control-plane only" in agents
    assert command in agents
    assert "Login-node safety boundary" in runbook
    assert command in runbook
