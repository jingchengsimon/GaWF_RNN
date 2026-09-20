"""Check the independent Skiing sweep command and legal-action tie metric."""
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.rl.atari.amarel.skiing_acdef_diagnostic import (
    HIDDEN, VARIANTS, audit_statistics, training_command,
)


def test_variants_preserve_protocol(tmp_path: Path) -> None:
    for model in HIDDEN:
        for variant, changes in VARIANTS.items():
            args = training_command(model, variant, tmp_path / "source",
                                    tmp_path / model / variant, False)
            for key, value in changes.items():
                assert args[args.index(f"--{key}") + 1] == str(value)
            assert args[args.index("--seed") + 1] == "1"
            assert args[args.index("--total_timesteps") + 1] == "4000000"
            assert "--no_reward_clip" in args and "--keep_replay_on_success" in args
            assert args[args.index("--amp_dtype") + 1] == "bfloat16"
            assert args[args.index("--init_weights_from") + 1].endswith(f"{model}/model.pth")
            assert args[args.index("--diagnostic_checkpoint_steps") + 1:][:8] == [
                str(i) for i in range(500000, 4000001, 500000)]


def test_legal_ties_exclude_alias_only_ties(tmp_path: Path) -> None:
    mapping = [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8]
    q = np.zeros((2, 18), dtype=np.float32)
    q[0, :2] = 1  # Same effective NOOP action: not a legal-group tie.
    q[1, [0, 2]] = 1  # Distinct legal actions: a legal-group tie.
    trace = tmp_path / "step_trace.npz"
    np.savez(trace, q_values=q, action=np.array([0, 0]), legal_action=np.array([0, 0]))
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps(dict(trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
        amp_dtype="bfloat16", canonical_action_mapping=mapping, mean_return=-10,
        stall_rate=0, source_checkpoint_sha256="test", num_episodes=20, eval_seed=20260904)))
    result = audit_statistics(summary, trace)
    assert result["legal_top_q_tie_rate"] == .5
    assert result["canonical_top_q_tie_rate"] == 1
    assert result["legal_action_fraction"][0] == 1
