"""Static and dry-run coverage for the SJC Skiing weights-only launcher."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments/remote/run_sjc_atari_skiing_warmstart_l3.sh"
ENV_IDS = [
    "ALE/Pong-v5",
    "ALE/Breakout-v5",
    "ALE/Assault-v5",
    "ALE/Seaquest-v5",
    "ALE/Skiing-v5",
]
MODEL_SIZES = {"lstm": 373, "gru": 458, "gawf": 604}


def _source_files(
    tmp_path: Path,
    model: str,
    source_step: int = 20_000_000,
    skiing_extension: bool = False,
    cumulative_2m: bool = False,
) -> tuple[Path, Path]:
    checkpoint = tmp_path / f"source_{model}.pth"
    checkpoint.touch()
    metrics = tmp_path / f"source_{model}.json"
    metrics.write_text(
        json.dumps(
            {
                "model_type": model,
                "hidden_size": MODEL_SIZES[model],
                "num_layers": 3,
                "action_space_mode": "full18",
                "num_actions": 18,
                "global_step": source_step,
                "env_ids": ["ALE/Skiing-v5"] if skiing_extension else ENV_IDS,
                "env_id": "ALE/Skiing-v5" if skiing_extension else None,
                "atari_env_protocol": (
                    "skiing-stall-actionfix-v1" if skiing_extension else "baseline"
                ),
                "action_mapping_protocol": (
                    "single_canonical_full18" if skiing_extension else "baseline"
                ),
                "initialization": {"mode": "weights_only"} if cumulative_2m else None,
                "extended_from_total_timesteps": (
                    1_000_000 if cumulative_2m and model == "gawf" else None
                ),
            }
        ),
        encoding="utf-8",
    )
    return checkpoint, metrics


def _dry_run(
    tmp_path: Path,
    model: str,
    *extra: str,
    source_step: int = 20_000_000,
) -> str:
    extension_1m = "--extend-from-skiing-1m" in extra
    extension_2m = "--extend-from-skiing-2m" in extra
    skiing_extension = extension_1m or extension_2m
    source_dir = tmp_path
    if extension_2m:
        source_dir /= (
            "atari_dqn_skiing_fs4_stack4_l3_full18_stallactionfix_v1_"
            + (
                f"extend1mto2m_1m_{model}_seed1"
                if model in {"gru", "lstm"}
                else "warmstart19450000_1m_gawf_seed1"
            )
        )
        source_dir.mkdir()
    checkpoint, metrics = _source_files(
        source_dir,
        model,
        2_000_000 if extension_2m and model == "gawf" else 1_000_000
        if skiing_extension
        else source_step,
        skiing_extension,
        extension_2m,
    )
    result = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "--model",
            model,
            "--cuda-device",
            "1",
            "--source-checkpoint",
            str(checkpoint),
            "--source-metrics",
            str(metrics),
            "--results-root",
            str(tmp_path / "results"),
            *extra,
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


@pytest.mark.parametrize(
    ("model", "hidden", "feedback"),
    [("lstm", 373, "none"), ("gru", 458, "none"), ("gawf", 604, "qvalues")],
)
def test_formal_dry_run_locks_weights_only_single_skiing_protocol(
    tmp_path: Path, model: str, hidden: int, feedback: str
) -> None:
    output = _dry_run(tmp_path, model)
    assert "formal_20m_4mpertask_raw_seeds" in output
    assert "--env_id ALE/Skiing-v5" in output
    assert "--action_space_mode full18" in output
    assert "--atari_env_protocol skiing-stall-actionfix-v1" in output
    assert f"--model_type {model}" in output
    assert f"--hidden_size {hidden}" in output
    assert f"--feedback_mode {feedback}" in output
    assert "--init_weights_from" in output and "--resume_from" not in output
    assert "--total_timesteps 1000000" in output
    assert "--buffer_size 500000" in output
    assert "--learning_starts 20000" in output
    assert "--exploration_steps 500000" in output
    assert "--seed 1" in output
    assert "CUDA_VISIBLE_DEVICES=1" in output


def test_smoke_is_25k_and_keeps_same_result_parent(tmp_path: Path) -> None:
    output = _dry_run(tmp_path, "lstm", "--smoke")
    assert "formal_20m_4mpertask_raw_seeds" in output
    assert "smoke25k_lstm_seed1" in output
    assert "--total_timesteps 25000" in output
    assert "--buffer_size 25000" in output
    assert "--learning_starts 2000" in output


def test_incomplete_source_requires_explicit_opt_in(tmp_path: Path) -> None:
    with pytest.raises(subprocess.CalledProcessError):
        _dry_run(tmp_path, "gawf", source_step=19_469_000)

    output = _dry_run(
        tmp_path,
        "gawf",
        "--allow-incomplete-source",
        "--run-tag",
        "gawf_source19469000",
        source_step=19_469_000,
    )
    assert "--init_weights_from" in output
    assert "gawf_source19469000" in output


def test_skiing_1m_extension_uses_terminal_schedules_and_fresh_state(tmp_path: Path) -> None:
    output = _dry_run(tmp_path, "gru", "--extend-from-skiing-1m")
    assert "extend1mto2m_1m_gru_seed1" in output
    assert "--init_weights_from" in output and "--resume_from" not in output
    assert "--total_timesteps 1000000" in output
    assert "--learning_rate 1e-5" in output
    assert "--learning_rate_decay_step 0" in output
    assert "--learning_rate_decay_scale 1.0" in output
    assert "--start_epsilon 0.01" in output
    assert "--end_epsilon 0.01" in output


@pytest.mark.parametrize("model", ["gru", "lstm", "gawf"])
def test_skiing_2m_extension_adds_fresh_two_million_steps(
    tmp_path: Path, model: str
) -> None:
    output = _dry_run(tmp_path, model, "--extend-from-skiing-2m")
    assert f"extend2mto4m_2m_{model}_seed1" in output
    assert "--init_weights_from" in output and "--resume_from" not in output
    assert "--total_timesteps 2000000" in output
    assert "--learning_rate 1e-5" in output
    assert "--learning_rate_decay_step 0" in output
    assert "--learning_rate_decay_scale 1.0" in output
    assert "--start_epsilon 0.01" in output
    assert "--end_epsilon 0.01" in output


def test_existing_run_can_explicitly_extend_its_budget(tmp_path: Path) -> None:
    source_checkpoint, source_metrics = _source_files(tmp_path, "gawf")
    results_root = tmp_path / "results"
    run_tag = "gawf_continuous"
    result_dir = (
        results_root
        / "data/rl/atari/5task_18action/formal_20m_4mpertask_raw_seeds"
        / run_tag
    )
    result_dir.mkdir(parents=True)
    (result_dir / "checkpoint.pth").touch()
    output = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "--model",
            "gawf",
            "--cuda-device",
            "1",
            "--source-checkpoint",
            str(source_checkpoint),
            "--source-metrics",
            str(source_metrics),
            "--results-root",
            str(results_root),
            "--run-tag",
            run_tag,
            "--total-timesteps",
            "2000000",
            "--allow-total-timesteps-extension",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "--resume_from" in output
    assert "--allow_total_timesteps_extension" in output
    assert "--total_timesteps 2000000" in output


def test_history_without_checkpoint_is_refused(tmp_path: Path) -> None:
    checkpoint, metrics = _source_files(tmp_path, "lstm")
    results_root = tmp_path / "results"
    run_tag = "existing"
    result_dir = (
        results_root
        / "data/rl/atari/5task_18action/formal_20m_4mpertask_raw_seeds"
        / run_tag
    )
    result_dir.mkdir(parents=True)
    (result_dir / "metrics_history.jsonl").touch()
    result = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "--model",
            "lstm",
            "--cuda-device",
            "0",
            "--source-checkpoint",
            str(checkpoint),
            "--source-metrics",
            str(metrics),
            "--results-root",
            str(results_root),
            "--run-tag",
            run_tag,
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 3
    assert "without resumable checkpoint" in result.stderr


def test_unclipped_gamma_protocol_has_separate_destination(tmp_path: Path) -> None:
    parent = tmp_path / "single_skiing"
    output = _dry_run(
        tmp_path, "lstm", "--total-timesteps", "4000000", "--gamma", "0.999",
        "--no-reward-clip", "--result-parent", str(parent), "--skip-smoke-video",
        "--keep-replay-on-success", "--requeue-on-pause", "--run-tag", "unclipped_4m",
    )
    assert f"RESULT_PARENT={parent}" in output
    assert "--total_timesteps 4000000" in output
    assert "--gamma 0.999 --no_reward_clip" in output
    assert "--keep_replay_on_success" in output
    assert "--amp_dtype bfloat16" in output
    assert "--init_weights_from" in output
