"""Reproduce parameter matching for mLSTM, HyperLSTM, and clean-room BRIMs.

The historical Clutter procedure minimizes the absolute difference in complete
trainable parameters relative to GaWF H=256, including the shared encoder and
readout heads. HyperLSTM candidates keep ``hyper_hidden/main_hidden`` in the
paper-scale interval [0.10, 0.16]. BRIMs varies only the shared ``nhid`` and
keeps the official MNIST structure fixed at blocks ``6 3`` and top-k ``4 2``.
"""

from __future__ import annotations

from dataclasses import dataclass

from utils.training.clutter.clutter_task_models import (
    BRIMsConv,
    GaWFRNNConv,
    HyperLSTMConv,
    MLSTMConv,
)


@dataclass(frozen=True)
class Match:
    """One parameter-count candidate."""

    model: str
    architecture: str
    core_parameters: int
    total_parameters: int
    target_difference: int


def _counts(model) -> tuple[int, int]:
    return (
        sum(parameter.numel() for parameter in model.core.parameters()),
        sum(parameter.numel() for parameter in model.parameters()),
    )


def find_matches() -> list[Match]:
    """Return the closest formal configuration for each added baseline."""
    common = {"num_classes": 10, "num_pos": 9, "kernel_size": 5, "device": "cpu"}
    target_model = GaWFRNNConv(hidden_size=256, **common)
    _target_core, target_total = _counts(target_model)

    mlstm_candidates = []
    for hidden_size in range(16, 160):
        model = MLSTMConv(hidden_size=hidden_size, **common)
        core, total = _counts(model)
        mlstm_candidates.append((abs(total - target_total), hidden_size, core, total))
    mlstm_diff, mlstm_hidden, mlstm_core, mlstm_total = min(mlstm_candidates)

    hyper_candidates = []
    for hidden_size in range(32, 160):
        min_hyper = max(4, round(0.10 * hidden_size))
        max_hyper = max(min_hyper, round(0.16 * hidden_size))
        for hyper_hidden_size in range(min_hyper, max_hyper + 1):
            model = HyperLSTMConv(
                hidden_size=hidden_size,
                hyper_hidden_size=hyper_hidden_size,
                hyper_embedding_size=4,
                **common,
            )
            core, total = _counts(model)
            hyper_candidates.append(
                (
                    abs(total - target_total),
                    abs(hyper_hidden_size / hidden_size - 0.128),
                    hidden_size,
                    hyper_hidden_size,
                    core,
                    total,
                )
            )
    (
        hyper_diff,
        _ratio_diff,
        hyper_hidden,
        hyper_network_hidden,
        hyper_core,
        hyper_total,
    ) = min(hyper_candidates)

    brims_candidates = []
    for hidden_size in range(6, 241, 6):
        model = BRIMsConv(hidden_size=hidden_size, **common)
        core, total = _counts(model)
        brims_candidates.append((abs(total - target_total), hidden_size, core, total))
    brims_diff, brims_hidden, brims_core, brims_total = min(brims_candidates)

    return [
        Match("GaWF", "H=256", _target_core, target_total, 0),
        Match(
            "mLSTM",
            f"H={mlstm_hidden}",
            mlstm_core,
            mlstm_total,
            mlstm_diff,
        ),
        Match(
            "HyperLSTM",
            f"H={hyper_hidden}, hyper_H={hyper_network_hidden}, n_z=4",
            hyper_core,
            hyper_total,
            hyper_diff,
        ),
        Match(
            "BRIMs",
            f"H={brims_hidden}, blocks=(6,3), topk=(4,2)",
            brims_core,
            brims_total,
            brims_diff,
        ),
    ]


def main() -> None:
    """Print the deterministic parameter-match table."""
    print("model\tarchitecture\tcore_parameters\ttotal_parameters\tabsolute_difference")
    for match in find_matches():
        print(
            f"{match.model}\t{match.architecture}\t{match.core_parameters}\t"
            f"{match.total_parameters}\t{match.target_difference}"
        )


if __name__ == "__main__":
    main()
