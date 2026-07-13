"""Parameter-match every Atari DQN readout core to the LSTM anchor.

The Atari Q-networks (``utils/atari_dqn_models.AtariQNetwork``) share the
Nature-DQN conv stack and a linear Q-head across all model types, so only the
recurrent "readout core" differs. To compare the variants fairly we size each
core to match the **LSTM core param count** at ``hidden_size=512`` (the anchor):

    RNN/GRU/GaWF -> search ``hidden_size``
    S5/Mamba     -> search ``d_model`` (state_size / d_state held fixed)
    LSTM         -> search hidden_size when candidate depth differs from the anchor
    ANN          -> feedforward control; NOT param-matched, width 512 per layer

RNN/GRU/GaWF/S5 are pure-torch or use the locally-available s5-pytorch, so they
match anywhere. Mamba needs ``mamba-ssm`` (GPU box, i.e. Amarel). Run:

    python -m experiments.generalization.atari_ssm_param_match \
        --conv_out 3136 --hidden_size 512 --ssm_state_size 128 \
        --num_actions 6 --num_layers 1

It writes ``atari_param_match.json`` mapping each model to the sizing args the
launch scripts feed to ``train_atari_dqn.py``.
"""

from __future__ import annotations

import argparse
import json
import os

# Guard against the macOS/conda duplicate-libomp crash when numpy and torch pull
# in separate OpenMP runtimes (harmless on the Linux GPU boxes this runs on).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from utils.recurrent_cores.gawf import GaWFCore
from utils.recurrent_cores.rnn import GRUCore, LSTMCore, RNNCore


def count_params(module: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def lstm_target(conv_out: int, hidden_size: int) -> int:
    """Return the fixed legacy single-layer LSTM anchor budget."""
    return count_params(LSTMCore(conv_out, hidden_size, num_layers=1))


# ---- hidden-size search for the torch recurrent cores ----------------------
def _build_rnn(conv_out, hidden, num_layers, **kw):
    return RNNCore(conv_out, hidden, num_layers=num_layers)


def _build_gru(conv_out, hidden, num_layers, **kw):
    return GRUCore(conv_out, hidden, num_layers=num_layers)


def _build_lstm(conv_out, hidden, num_layers, **kw):
    return LSTMCore(conv_out, hidden, num_layers=num_layers)


def _build_gawf(conv_out, hidden, num_actions, num_layers, **kw):
    # Matches how AtariQNetwork builds GaWF: feedback_dim = num_actions (qvalues).
    feedback_dim = max(1, num_actions)
    return GaWFCore(
        conv_out,
        hidden,
        feedback_dim=feedback_dim,
        num_layers=num_layers,
        layer_feedback_dims=(
            [hidden] * (num_layers - 1) + [feedback_dim] if num_layers > 1 else None
        ),
    )


def search_hidden(build_fn, conv_out, target, num_actions, num_layers, h_min=8, h_max=8192) -> dict:
    best = None
    for hidden in range(h_min, h_max + 1):
        params = count_params(
            build_fn(conv_out, hidden, num_actions=num_actions, num_layers=num_layers)
        )
        diff = abs(params - target)
        if best is None or diff < best["abs_diff"]:
            best = {
                "hidden_size": hidden,
                "params": params,
                "abs_diff": diff,
                "rel_diff_pct": 100.0 * (params - target) / target,
            }
        if params > target and best["hidden_size"] != hidden:
            break
    return best


# ---- d_model search for the SSM cores --------------------------------------
def _build_s5(conv_out, d_model, state_size, num_layers):
    from utils.recurrent_cores.s5 import S5Core

    return S5Core(
        input_size=conv_out, d_model=d_model, state_size=state_size, num_layers=num_layers
    )


def _build_mamba(conv_out, d_model, state_size, num_layers):
    from utils.recurrent_cores.mamba import MambaCore

    return MambaCore(
        input_size=conv_out, d_model=d_model, num_layers=num_layers, d_state=state_size
    )


def search_d_model(
    build_fn, conv_out, target, state_size, num_layers, d_min=16, d_max=4096
) -> dict:
    best = None
    for d_model in range(d_min, d_max + 1, 2):
        params = count_params(build_fn(conv_out, d_model, state_size, num_layers))
        diff = abs(params - target)
        if best is None or diff < best["abs_diff"]:
            best = {
                "d_model": d_model,
                "params": params,
                "abs_diff": diff,
                "rel_diff_pct": 100.0 * (params - target) / target,
            }
        if params > target and best["d_model"] != d_model:
            break
    return best


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Param-match Atari DQN cores to the LSTM anchor")
    p.add_argument(
        "--conv_out",
        type=int,
        default=3136,
        help="Flattened Nature-DQN conv features (64*7*7 for 84x84).",
    )
    p.add_argument("--hidden_size", type=int, default=512, help="LSTM anchor hidden size.")
    p.add_argument(
        "--num_actions", type=int, default=6, help="Action count (GaWF feedback_dim). Pong=6."
    )
    p.add_argument(
        "--ssm_state_size",
        type=int,
        default=128,
        help="S5 state_size / Mamba d_state held fixed while searching d_model.",
    )
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument(
        "--models",
        nargs="+",
        default=["rnn", "gru", "lstm", "gawf", "s5", "mamba"],
        choices=["rnn", "gru", "lstm", "gawf", "s5", "mamba"],
    )
    p.add_argument("--out_dir", type=str, default="results/atari_param_match")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    target = lstm_target(args.conv_out, args.hidden_size)
    print(
        f"LSTM anchor core params @ hidden={args.hidden_size}, conv_out={args.conv_out}: "
        f"{target:,}\n"
    )

    matched: dict[str, dict] = {
        "ann": {
            "hidden_size": 512,
            "num_layers": args.num_layers,
            "note": "feedforward control; not param-matched",
        },
    }

    hidden_builders = {
        "rnn": _build_rnn,
        "gru": _build_gru,
        "lstm": _build_lstm,
        "gawf": _build_gawf,
    }
    ssm_builders = {"s5": _build_s5, "mamba": _build_mamba}

    for model in args.models:
        try:
            if model in hidden_builders:
                best = search_hidden(
                    hidden_builders[model], args.conv_out, target, args.num_actions, args.num_layers
                )
                best["num_layers"] = args.num_layers
                matched[model] = best
                print(
                    f"{model:6s} -> hidden_size={best['hidden_size']:5d} | "
                    f"params={best['params']:,} ({best['rel_diff_pct']:+.2f}%)"
                )
            else:
                best = search_d_model(
                    ssm_builders[model], args.conv_out, target, args.ssm_state_size, args.num_layers
                )
                best["state_size"] = args.ssm_state_size
                matched[model] = best
                print(
                    f"{model:6s} -> d_model={best['d_model']:5d} "
                    f"state_size={args.ssm_state_size} | "
                    f"params={best['params']:,} ({best['rel_diff_pct']:+.2f}%)"
                )
        except ImportError as exc:
            print(f"[skip {model}] optional dependency missing: {exc}")

    os.makedirs(args.out_dir, exist_ok=True)
    out = {
        "anchor": "lstm",
        "anchor_num_layers": 1,
        "conv_out": args.conv_out,
        "hidden_size": args.hidden_size,
        "num_actions": args.num_actions,
        "target_core_params": target,
        "ssm_state_size": args.ssm_state_size,
        "candidate_num_layers": args.num_layers,
        "matched": matched,
    }
    out_path = os.path.join(args.out_dir, "atari_param_match.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
