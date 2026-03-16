import argparse

import torch

from utils.train_rnn_core import GRUConv, LSTMConv, RNNConv
from utils.train_gawf_core import GaWFRNNConv


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_models(
    hidden_rnn: int,
    hidden_lstm: int,
    hidden_gru: int,
    hidden_gawf: int,
    num_classes: int = 10,
    num_pos: int = 9,
    device: str = "cpu",
    kernel_size: int = 5,
    dropout_rate: float = 0.3,
):
    common = dict(
        num_classes=num_classes,
        num_pos=num_pos,
        kernel_size=kernel_size,
        device=device,
        dropout_rate=dropout_rate,
        max_chars=15,
        predict_all_chars=False,
    )

    models = {
        "rnn": RNNConv(hidden_size=hidden_rnn, **common),
        "lstm": LSTMConv(hidden_size=hidden_lstm, **common),
        "gru": GRUConv(hidden_size=hidden_gru, **common),
        "gawf": GaWFRNNConv(hidden_size=hidden_gawf, **common),
    }
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
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate used when constructing models (default: 0.3).",
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
    return parser.parse_args()


def main():
    args = parse_args()

    models = build_models(
        hidden_rnn=args.hidden_rnn,
        hidden_lstm=args.hidden_lstm,
        hidden_gru=args.hidden_gru,
        hidden_gawf=args.hidden_gawf,
        num_classes=args.num_classes,
        num_pos=args.num_pos,
        device=args.device,
        kernel_size=args.kernel_size,
        dropout_rate=args.dropout,
    )

    print(
        "Config:"
        f" hidden_rnn={args.hidden_rnn},"
        f" hidden_lstm={args.hidden_lstm},"
        f" hidden_gru={args.hidden_gru},"
        f" hidden_gawf={args.hidden_gawf},"
        f" num_classes={args.num_classes},"
        f" num_pos={args.num_pos},"
        f" kernel_size={args.kernel_size},"
        f" device={args.device},"
        f" dropout={args.dropout}"
    )
    print("-" * 72)

    for name, model in models.items():
        total, trainable = count_parameters(model)
        print(
            f"{name.upper():5s}  total_params={total:,}  "
            f"trainable_params={trainable:,}"
        )


if __name__ == "__main__":
    main()

