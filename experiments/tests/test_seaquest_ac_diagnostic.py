"""Check the sixteen independent cells preserve the source Seaquest training protocol."""
from pathlib import Path

from experiments.remote.run_sjc_seaquest_diagnostics import command
from experiments.rl.atari.amarel.seaquest_ac_diagnostic import cell_identity, training_command


def test_seed_cells_preserve_source_protocol() -> None:
    identities = {cell_identity(cell) for cell in range(16)}
    assert identities == {(m, v, s) for m in ("lstm", "gawf")
                          for v in ("a", "c") for s in range(1, 5)}
    for cell in range(16):
        model, variant, seed = cell_identity(cell)
        for smoke in (True, False):
            result = Path("/tmp/seaquest") / f"{model}_{variant}_seed{seed}"
            expected = command(model, variant, result, smoke)
            expected[expected.index("--seed") + 1] = str(seed)
            assert training_command(cell, result, smoke) == expected + ["--keep_replay_on_success"]
