import argparse
import importlib
import os
import sys

# Import NumPy before torch on macOS/conda to avoid duplicate libomp initialization.
importlib.import_module("numpy")
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.clutter_task_models import GRUConv, LSTMConv, RNNConv, S5Conv, GaWFRNNConv

try:
    from utils.clutter_task_models import MambaConv
except ImportError:
    MambaConv = None


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_models(
    hidden_rnn: int,
    hidden_lstm: int,
    hidden_gru: int,
    hidden_gawf: int,
    mamba_d_model: int,
    s5_d_model: int,
    mamba_num_layers: int = 1,
    mamba_d_state: int = 16,
    mamba_d_conv: int = 4,
    mamba_expand: int = 2,
    s5_num_layers: int = 1,
    s5_state_size: int | None = None,
    num_classes: int = 10,
    num_pos: int = 9,
    device: str = "cpu",
    kernel_size: int = 5,
    cnn_dropout: float = 0.0,
    rnn_dropout: float = 0.5,
    feedback_dim: int | None = None,
):
    common = dict(
        num_classes=num_classes,
        num_pos=num_pos,
        kernel_size=kernel_size,
        device=device,
        cnn_dropout=cnn_dropout,
        rnn_dropout=rnn_dropout,
        max_chars=15,
        predict_all_chars=False,
    )

    models = {
        "rnn": RNNConv(hidden_size=hidden_rnn, **common),
        "lstm": LSTMConv(hidden_size=hidden_lstm, **common),
        "gru": GRUConv(hidden_size=hidden_gru, **common),
        "gawf": GaWFRNNConv(hidden_size=hidden_gawf, feedback_dim=feedback_dim, **common),
    }
    try:
        models["s5"] = S5Conv(
            s5_d_model=s5_d_model,
            s5_num_layers=s5_num_layers,
            s5_state_size=s5_state_size,
            **common,
        )
    except ImportError:
        models["s5"] = None
    if MambaConv is not None:
        try:
            models["mamba"] = MambaConv(
                mamba_d_model=mamba_d_model,
                mamba_num_layers=mamba_num_layers,
                mamba_d_state=mamba_d_state,
                mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand,
                **common,
            )
        except ImportError:
            models["mamba"] = None
    return models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print parameter counts for RNN/LSTM/GRU/GaWF models."
    )
    parser.add_argument(
        "--hidden_rnn",
        type=int,
        default=275,
        help="Hidden size for RNNConv (default: 256).",
    )
    parser.add_argument(
        "--hidden_lstm",
        type=int,
        default=80,
        help="Hidden size for LSTMConv (default: 256).",
    )
    parser.add_argument(
        "--hidden_gru",
        type=int,
        default=105,
        help="Hidden size for GRUConv (default: 256).",
    )
    parser.add_argument(
        "--hidden_gawf",
        type=int,
        default=256,
        help="Hidden size for GaWFRNNConv (default: 256).",
    )
    parser.add_argument(
        "--mamba_d_model",
        type=int,
        default=170,
        help="Mamba sequence width d_model (default: 170).",
    )
    parser.add_argument(
        "--s5_d_model",
        type=int,
        default=256,
        help="S5 sequence feature width d_model (default: 256).",
    )
    parser.add_argument(
        "--mamba_num_layers",
        type=int,
        default=1,
        help="Number of Mamba blocks (default: 1).",
    )
    parser.add_argument(
        "--mamba_d_state",
        type=int,
        default=16,
        help="Mamba SSM state dimension per channel (default: 16).",
    )
    parser.add_argument(
        "--mamba_d_conv",
        type=int,
        default=4,
        help="Mamba local convolution width (default: 4).",
    )
    parser.add_argument(
        "--mamba_expand",
        type=int,
        default=2,
        help="Mamba inner expansion factor (default: 2).",
    )
    parser.add_argument(
        "--s5_num_layers",
        type=int,
        default=1,
        help="Number of S5 layers (default: 1).",
    )
    parser.add_argument(
        "--s5_state_size",
        type=int,
        default=128,
        help="S5 latent state size (default: 128, param-matched to GaWF h=256).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device on which to instantiate models (default: cpu).",
    )
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=5,
        help="Kernel size for conv layers (default: 5).",
    )
    parser.add_argument(
        "--cnn_dropout",
        type=float,
        default=0.0,
        help="CNN encoder dropout p (default: 0).",
    )
    parser.add_argument(
        "--rnn_dropout",
        type=float,
        default=0.5,
        help="Middle-path dropout p after ReLU (default: 0.5).",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=10,
        help="Number of character classes (default: 10).",
    )
    parser.add_argument(
        "--num_pos",
        type=int,
        default=9,
        help="Number of position outputs (default: 9, sector mode).",
    )
    parser.add_argument(
        "--feedback_dim",
        "--dz",
        type=int,
        default=None,
        help="GaWFRNN feedback context dimension dz (default: legacy num_classes + num_pos).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    models = build_models(
        hidden_rnn=args.hidden_rnn,
        hidden_lstm=args.hidden_lstm,
        hidden_gru=args.hidden_gru,
        hidden_gawf=args.hidden_gawf,
        mamba_d_model=args.mamba_d_model,
        s5_d_model=args.s5_d_model,
        mamba_num_layers=args.mamba_num_layers,
        mamba_d_state=args.mamba_d_state,
        mamba_d_conv=args.mamba_d_conv,
        mamba_expand=args.mamba_expand,
        s5_num_layers=args.s5_num_layers,
        s5_state_size=args.s5_state_size,
        num_classes=args.num_classes,
        num_pos=args.num_pos,
        feedback_dim=args.feedback_dim,
        device=args.device,
        kernel_size=args.kernel_size,
        cnn_dropout=args.cnn_dropout,
        rnn_dropout=args.rnn_dropout,
    )

    print(
        "Config:"
        f" hidden_rnn={args.hidden_rnn},"
        f" hidden_lstm={args.hidden_lstm},"
        f" hidden_gru={args.hidden_gru},"
        f" hidden_gawf={args.hidden_gawf},"
        f" mamba_d_model={args.mamba_d_model},"
        f" s5_d_model={args.s5_d_model},"
        f" mamba_num_layers={args.mamba_num_layers},"
        f" mamba_d_state={args.mamba_d_state},"
        f" mamba_d_conv={args.mamba_d_conv},"
        f" mamba_expand={args.mamba_expand},"
        f" s5_num_layers={args.s5_num_layers},"
        f" s5_state_size={args.s5_state_size},"
        f" num_classes={args.num_classes},"
        f" num_pos={args.num_pos},"
        f" feedback_dim={args.feedback_dim},"
        f" kernel_size={args.kernel_size},"
        f" device={args.device},"
        f" cnn_dropout={args.cnn_dropout},"
        f" rnn_dropout={args.rnn_dropout}"
    )
    print("-" * 72)

    for name, model in models.items():
        if model is None:
            if name == "mamba":
                msg = "install mamba-ssm and causal-conv1d to instantiate MambaConv."
            else:
                msg = "install s5-pytorch to instantiate S5Conv."
            print(f"{name.upper():7s}  skipped: {msg}")
            continue
        total, trainable = count_parameters(model)
        print(
            f"{name.upper():7s}  total_params={total:,}  "
            f"trainable_params={trainable:,}"
        )


if __name__ == "__main__":
    main()
