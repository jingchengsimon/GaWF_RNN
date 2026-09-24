"""Cross-seed driver for the Figure-03 unit-level gate context-variance analysis.

For every ``(model, seed)`` checkpoint in a best-6 multi-seed campaign this recomputes the exact
balanced 9-sector x 10-digit condition-mean variance fractions of that model's unit-level gates,
reusing the single-seed code path from ``rnn_unit_gate_context_specificity``. GaWF afferent gates
are reconstructed from a freshly collected reset-feedback trajectory plus that seed's U/V/W;
LSTM/GRU gates are extracted by manual recurrence. The per-seed fractions are pooled into one
compact JSON so the visualization can draw cross-seed mean +/- SEM, the shared spread
convention as the best-model-accuracy figure.

Inputs: a campaign checkpoint root laid out as ``{model}-seedNN/*_model.pth`` and the Clutter test
stimulus/label pair.
Outputs: ``unit_gate_context_variance_multiseed.json`` (written atomically, resumable).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.analysis.anal_helpers import build_model_from_ckpt, build_test_dataset
from utils.analysis.anal_paths import output_dir
from utils.analysis.clutter.supple2_gate_context import _balanced_masks
from utils.analysis.clutter.fig3_gate_distribution import collect_trajectory
from utils.analysis.clutter.fig5_unit_gate_context import (
    GATE_NAMES,
    analyze_gawf,
    analyze_model,
)


MODEL_ORDER = ("gawf", "lstm", "gru")
FACTORS = ("sector", "digit", "interaction")
RESIDUAL = "residual"


def parse_args() -> argparse.Namespace:
    """Parse campaign, data, and output settings."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint_root",
        default="/scratch/js3269/results/data/clutter/runs/clutter_best6_multiseed_40h_ep150",
        help="Directory holding one {model}-seedNN subdirectory per completed unit.",
    )
    parser.add_argument("--models", nargs="+", default=list(MODEL_ORDER), choices=MODEL_ORDER)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--data_dir", default="/scratch/js3269/stimuli")
    parser.add_argument("--data_suffix", default="40h-uint8")
    parser.add_argument(
        "--save_json",
        default=str(
            output_dir(
                "D_variance_decomposition",
                "rnn_unit_gate_context_specificity_multiseed",
                "data",
            )
            / "unit_gate_context_variance_multiseed.json"
        ),
    )
    parser.add_argument("--balance_seed", type=int, default=260718)
    parser.add_argument("--gawf_gate_tau", type=float, default=0.5)
    parser.add_argument("--rnn_batch_size", type=int, default=16)
    parser.add_argument("--gawf_trajectory_batch_size", type=int, default=16)
    parser.add_argument("--gawf_frame_batch_size", type=int, default=64)
    parser.add_argument(
        "--gawf_trajectory_root",
        type=Path,
        default=None,
        help="Optionally retain each GaWF seed trajectory for downstream gate analyses.",
    )
    parser.add_argument("--chan_num", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--use_mmap", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _resolve_checkpoint(checkpoint_root: str, model: str, seed: int) -> str:
    """Return the single ``*_model.pth`` for one campaign unit, failing loudly if ambiguous."""

    unit_dir = Path(checkpoint_root) / f"{model}-seed{seed:02d}"
    if not unit_dir.is_dir():
        raise FileNotFoundError(f"Missing seed unit directory: {unit_dir}")
    checkpoints = sorted(unit_dir.glob("*_model.pth"))
    if len(checkpoints) != 1:
        raise RuntimeError(
            f"Expected exactly one *_model.pth in {unit_dir}, found {len(checkpoints)}"
        )
    return str(checkpoints[0])


def _collect_reference_labels(dataset: torch.utils.data.Dataset, batch_size: int) -> np.ndarray:
    """Return the ordered ``(n_frames, 2)`` digit/sector labels shared by every seed."""

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    batches = [np.asarray(batch[1], dtype=np.int64) for batch in loader]
    return np.concatenate(batches, axis=0).reshape(-1, 2)


def _fractions_from_report(report: dict[str, Any], model_type: str) -> dict[str, dict[str, float]]:
    """Extract condition-mean factors plus trial-level residual as plain fractions."""

    return {
        gate: {
            **{
                factor: float(
                    report["gates"][gate]["equal_cell_condition_mean"]["fractions"][factor]
                )
                for factor in FACTORS
            },
            RESIDUAL: float(
                report["gates"][gate]["equal_cell_trial_total"]["percent"][RESIDUAL]
            )
            / 100.0,
        }
        for gate in GATE_NAMES[model_type]
    }


def _gawf_fractions(
    checkpoint: str,
    seed: int,
    dataset: torch.utils.data.Dataset,
    num_pos: int,
    reference_labels: np.ndarray,
    equal_joint_mask: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    work_dir: Path,
) -> dict[str, dict[str, float]]:
    """Collect one seed's reset-feedback trajectory and decompose its destination-unit gates."""

    model = build_model_from_ckpt(checkpoint, num_pos, device, chan_num=args.chan_num)
    if not getattr(model, "is_gawf_model", False) or getattr(model, "is_gawf_multi_model", False):
        raise RuntimeError("Figure-03 multi-seed analysis requires single-layer GaWF checkpoints")
    weight_ih = model.rnn.weight_ih_l0.detach().cpu().numpy().astype(np.float32)
    weight_hh = model.rnn.weight_hh_l0.detach().cpu().numpy().astype(np.float32)
    u = model.U.detach().cpu().numpy().astype(np.float32)
    v = model.V.detach().cpu().numpy().astype(np.float32)
    gate_tau = float(model.gate_tau)
    trajectory, _performance = collect_trajectory(
        dataset, model, device, args.gawf_trajectory_batch_size
    )
    labels = np.asarray(trajectory["labels"], dtype=np.int64).reshape(-1, 2)
    if not np.array_equal(labels, reference_labels):
        raise RuntimeError("GaWF trajectory labels diverge from the reference labels")
    trajectory.update({"U": u, "V": v, "weight_ih": weight_ih, "weight_hh": weight_hh})
    trajectory_path = (
        args.gawf_trajectory_root / f"seed{seed:02d}" / "gawf_gate_trajectory.npz"
        if args.gawf_trajectory_root is not None
        else work_dir / "gawf_trajectory_tmp.npz"
    )
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    if trajectory_path.exists():
        raise FileExistsError(f"Refusing to overwrite GaWF trajectory: {trajectory_path}")
    np.savez_compressed(trajectory_path, **trajectory)
    del model, trajectory
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    # A manifest path that does not exist forces the exact float32 feedback/U/V reconstruction.
    manifest_path = work_dir / "no_saved_manifest.json"
    report, _arrays = analyze_gawf(
        manifest_path,
        trajectory_path,
        reference_labels,
        equal_joint_mask,
        args.gawf_frame_batch_size,
        args.gawf_gate_tau,
        device,
    )
    if args.gawf_trajectory_root is None:
        trajectory_path.unlink(missing_ok=True)
    return _fractions_from_report(report, "gawf")


def _rnn_fractions(
    model_type: str,
    checkpoint: str,
    dataset: torch.utils.data.Dataset,
    num_pos: int,
    reference_labels: np.ndarray,
    equal_joint_mask: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """Extract one LSTM/GRU seed's unit gates and reduce them to condition-mean fractions."""

    model_args = argparse.Namespace(batch_size=args.rnn_batch_size, chan_num=args.chan_num)
    report, _arrays = analyze_model(
        model_type,
        checkpoint,
        dataset,
        num_pos,
        reference_labels,
        equal_joint_mask,
        device,
        model_args,
    )
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return _fractions_from_report(report, model_type)


def _load_progress(save_json: Path) -> dict[str, dict[int, dict[str, dict[str, float]]]]:
    """Rebuild per-(model, seed) fractions from an existing output JSON for resumption."""

    if not save_json.is_file():
        return {}
    payload = json.loads(save_json.read_text(encoding="utf-8"))
    collected: dict[str, dict[int, dict[str, dict[str, float]]]] = {}
    for model, model_block in payload.get("models", {}).items():
        seeds = [int(seed) for seed in model_block.get("seeds", [])]
        per_seed: dict[int, dict[str, dict[str, float]]] = {seed: {} for seed in seeds}
        for gate, gate_block in model_block.get("gates", {}).items():
            fractions = gate_block["equal_cell_condition_mean"]["fractions"]
            residuals = gate_block.get("equal_cell_trial_total", {}).get("fractions", {}).get(
                RESIDUAL, []
            )
            for factor, values in {**fractions, RESIDUAL: residuals}.items():
                for seed, value in zip(seeds, values):
                    per_seed[seed].setdefault(gate, {})[factor] = float(value)
        collected[model] = per_seed
    return collected


def _save_progress(
    save_json: Path,
    collected: dict[str, dict[int, dict[str, dict[str, float]]]],
    metadata: dict[str, Any],
) -> None:
    """Atomically write the pooled fractions, one array entry per seed."""

    models_block: dict[str, Any] = {}
    for model in MODEL_ORDER:
        if model not in collected or not collected[model]:
            continue
        seeds = sorted(collected[model])
        gates_block: dict[str, Any] = {}
        for gate in GATE_NAMES[model]:
            fractions = {
                factor: [collected[model][seed][gate][factor] for seed in seeds]
                for factor in FACTORS
            }
            residuals = [collected[model][seed][gate][RESIDUAL] for seed in seeds]
            gates_block[gate] = {
                "equal_cell_condition_mean": {"fractions": fractions},
                "equal_cell_trial_total": {"fractions": {RESIDUAL: residuals}},
            }
        models_block[model] = {"seeds": seeds, "gates": gates_block}
    payload = {**metadata, "models": models_block}
    partial = save_json.with_name(f"{save_json.stem}.partial{save_json.suffix}")
    partial.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    partial.replace(save_json)


def main() -> None:
    """Compute and pool per-seed gate fractions for every requested model and seed."""

    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    # cuDNN/cuBLAS TF32 (default on Ampere+) perturbs the native LSTM/GRU kernels by ~1e-2,
    # which breaks the exact-recurrence parity check in analyze_model; force full float32 so the
    # manual gate recurrence and the native kernels agree to ~1e-5 as they do on CPU.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    save_json = Path(args.save_json)
    save_json.parent.mkdir(parents=True, exist_ok=True)

    dataset, num_pos = build_test_dataset(args)
    reference_labels = _collect_reference_labels(dataset, args.rnn_batch_size)
    sequence_length = reference_labels.shape[0] // len(dataset)
    frame_in_sequence = np.arange(reference_labels.shape[0]) % sequence_length
    valid_frame_mask = frame_in_sequence != 0
    valid_labels = reference_labels[valid_frame_mask]
    digits, sectors = valid_labels[:, 0], valid_labels[:, 1]
    _marginal_masks, valid_equal_joint_mask = _balanced_masks(digits, sectors, args.balance_seed)
    equal_joint_mask = np.zeros(reference_labels.shape[0], dtype=bool)
    equal_joint_mask[valid_frame_mask] = valid_equal_joint_mask
    equal_n = int(np.count_nonzero(equal_joint_mask)) // 90
    print(
        f"Reference labels: n_frames={reference_labels.shape[0]} | "
        f"equal_joint_cell_n={equal_n} | balance_seed={args.balance_seed}",
        flush=True,
    )

    metadata: dict[str, Any] = {
        "analysis": (
            "Cross-seed GaWF/LSTM/GRU unit-level gate context-variance fractions (Figure 03)"
        ),
        "dataset": args.data_suffix,
        "n_frames": int(reference_labels.shape[0]),
        "reset_frames_excluded": int(np.count_nonzero(~valid_frame_mask)),
        "analysis_n_frames": int(np.count_nonzero(valid_frame_mask)),
        "balance_seed": int(args.balance_seed),
        "equal_joint_cell_n": equal_n,
        "labels": {"digit_levels": 10, "sector_levels": 9},
        "spread_convention": "cross-seed mean +/- SEM (ddof=1)",
        "checkpoint_root": str(Path(args.checkpoint_root).resolve()),
        "data_dir": str(Path(args.data_dir).resolve()),
        "gate_convention": {
            "gawf": (
                "destination-unit arithmetic mean across raw sigmoid input or recurrent "
                "synapse gates"
            ),
            "lstm": "PyTorch i/f/g/o order; sigmoid i/f/o reported, candidate g excluded",
            "gru": "PyTorch r/z/n order; sigmoid reset/update reported, candidate n excluded",
        },
    }

    collected = _load_progress(save_json)
    units = [(model, seed) for model in args.models for seed in args.seeds]
    for index, (model, seed) in enumerate(units, start=1):
        if model in collected and seed in collected[model]:
            print(f"[{index}/{len(units)}] skip {model}-seed{seed:02d} (already computed)")
            continue
        checkpoint = _resolve_checkpoint(args.checkpoint_root, model, seed)
        print(f"[{index}/{len(units)}] {model}-seed{seed:02d} <- {checkpoint}", flush=True)
        if model == "gawf":
            fractions = _gawf_fractions(
                checkpoint,
                seed,
                dataset,
                num_pos,
                reference_labels,
                equal_joint_mask,
                args,
                device,
                save_json.parent,
            )
        else:
            fractions = _rnn_fractions(
                model,
                checkpoint,
                dataset,
                num_pos,
                reference_labels,
                equal_joint_mask,
                args,
                device,
            )
        collected.setdefault(model, {})[seed] = fractions
        _save_progress(save_json, collected, metadata)
    _save_progress(save_json, collected, metadata)
    print(f"Saved {save_json}")


if __name__ == "__main__":
    main()
