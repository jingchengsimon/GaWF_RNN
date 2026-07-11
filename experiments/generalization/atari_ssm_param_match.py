"""Parameter-match S5 and Mamba readout cores to the LSTM anchor for Atari DQN.

The Atari Q-networks (``utils/atari_dqn_models.AtariQNetwork``) share the
Nature-DQN conv stack and a linear Q-head across all model types, so only the
recurrent "readout core" differs. To compare S5/Mamba fairly against the
RNN/GRU/LSTM/GaWF variants we size their cores to match the **LSTM core param
count** at ``hidden_size=512`` (the anchor, consistent with the IMDB param-match
convention).

The LSTM target is pure-torch and computable anywhere. The S5/Mamba search
instantiates the real cores and therefore requires ``s5-pytorch`` / ``mamba-ssm``
(GPU box, i.e. Amarel). Run:

    python -m experiments.generalization.atari_ssm_param_match \
        --conv_out 3136 --hidden_size 512 --ssm_state_size 128 --num_layers 1

It writes ``atari_ssm_param_match.json`` with the matched ``ssm_d_model`` per
core, which the launch scripts feed to ``train_atari_dqn.py --ssm_d_model``.
"""

from __future__ import annotations

import argparse
import json
import os

# Guard against the macOS/conda duplicate-libomp crash when numpy and torch pull
# in separate OpenMP runtimes (harmless on the Linux GPU boxes this runs on).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from utils.recurrent_cores.rnn import LSTMCore


def count_params(module: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def lstm_target(conv_out: int, hidden_size: int) -> int:
    return count_params(LSTMCore(conv_out, hidden_size))


def _build_s5(conv_out: int, d_model: int, state_size: int, num_layers: int):
    from utils.recurrent_cores.s5 import S5Core

    return S5Core(
        input_size=conv_out,
        d_model=d_model,
        state_size=state_size,
        num_layers=num_layers,
    )


def _build_mamba(conv_out: int, d_model: int, state_size: int, num_layers: int):
    from utils.recurrent_cores.mamba import MambaCore

    return MambaCore(
        input_size=conv_out,
        d_model=d_model,
        num_layers=num_layers,
        d_state=state_size,
    )


def search_d_model(
    build_fn,
    conv_out: int,
    target: int,
    state_size: int,
    num_layers: int,
    d_min: int = 16,
    d_max: int = 2048,
) -> dict:
    """Pick the d_model whose core param count is closest to ``target``."""
    best = None
    for d_model in range(d_min, d_max + 1, 2):
        try:
            params = count_params(build_fn(conv_out, d_model, state_size, num_layers))
        except Exception as exc:  # noqa: BLE001 - surface first hard failure
            raise RuntimeError(f"Failed to build core at d_model={d_model}: {exc}") from exc
        diff = abs(params - target)
        if best is None or diff < best["abs_diff"]:
            best = {
                "d_model": d_model,
                "params": params,
                "abs_diff": diff,
                "rel_diff_pct": 100.0 * (params - target) / target,
            }
        # params grow monotonically in d_model; stop once we pass the target.
        if params > target and best["d_model"] != d_model:
            break
    return best


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Param-match S5/Mamba cores to LSTM anchor")
    p.add_argument("--conv_out", type=int, default=3136,
                   help="Flattened Nature-DQN conv features (64*7*7 for 84x84).")
    p.add_argument("--hidden_size", type=int, default=512, help="LSTM anchor hidden size.")
    p.add_argument("--ssm_state_size", type=int, default=128,
                   help="S5 state_size / Mamba d_state held fixed while searching d_model.")
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--cores", nargs="+", default=["s5", "mamba"], choices=["s5", "mamba"])
    p.add_argument("--out_dir", type=str, default="results/atari_param_match")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    target = lstm_target(args.conv_out, args.hidden_size)
    print(f"LSTM anchor core params @ hidden={args.hidden_size}, conv_out={args.conv_out}: "
          f"{target:,}")

    builders = {"s5": _build_s5, "mamba": _build_mamba}
    matched: dict[str, dict] = {}
    for core in args.cores:
        try:
            best = search_d_model(
                builders[core], args.conv_out, target, args.ssm_state_size, args.num_layers
            )
        except ImportError as exc:
            print(f"[skip {core}] optional dependency missing: {exc}")
            continue
        matched[core] = best
        print(f"{core:6s} -> ssm_d_model={best['d_model']:4d} "
              f"state_size={args.ssm_state_size} layers={args.num_layers} | "
              f"params={best['params']:,} ({best['rel_diff_pct']:+.2f}% vs anchor)")

    os.makedirs(args.out_dir, exist_ok=True)
    out = {
        "anchor": "lstm",
        "conv_out": args.conv_out,
        "hidden_size": args.hidden_size,
        "target_core_params": target,
        "ssm_state_size": args.ssm_state_size,
        "num_layers": args.num_layers,
        "matched": matched,
    }
    out_path = os.path.join(args.out_dir, "atari_ssm_param_match.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
