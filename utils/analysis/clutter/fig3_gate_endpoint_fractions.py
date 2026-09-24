"""Audit strict GaWF gate endpoint fractions from saved feedback trajectories.

Inputs are ten ``seedNN/gawf_gate_trajectory.npz`` files produced by the Figure 3 collector.
The script reconstructs float32 input and recurrent gates, excludes zero-feedback reset frames,
and writes one JSON containing per-seed and aggregate fractions for ``g < 0.1`` and ``g > 0.9``.
Exact equality counts at both thresholds are retained so strict and closed interval conventions
cannot be confused.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from utils.analysis.clutter.fig3_gate_distribution import _gate_tensors


def parse_args() -> argparse.Namespace:
    """Parse trajectory root, output path, device, and streaming chunk size."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--chunk-size", type=int, default=256)
    return parser.parse_args()


def count_gate_endpoints(
    feedback: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    input_size: int,
    tau: float,
    chunk_size: int,
    device: torch.device,
) -> dict[str, dict[str, int]]:
    """Return exact strict endpoint and equality counts for one trajectory."""

    flat = np.asarray(feedback, dtype=np.float32).reshape(-1, feedback.shape[-1])
    keep = ~np.all(flat == 0.0, axis=1)
    flat = flat[keep]
    if flat.size == 0:
        raise RuntimeError("No non-reset feedback frames remain")

    u_tensor = torch.as_tensor(u, dtype=torch.float32, device=device)
    v_tensor = torch.as_tensor(v, dtype=torch.float32, device=device)
    counts = {
        kind: {"total": 0, "below_0_1": 0, "above_0_9": 0, "equal_0_1": 0, "equal_0_9": 0}
        for kind in ("input", "recurrent")
    }
    with torch.no_grad():
        for start in range(0, flat.shape[0], chunk_size):
            chunk = torch.as_tensor(
                flat[start : start + chunk_size], dtype=torch.float32, device=device
            )
            input_gate, recurrent_gate = _gate_tensors(
                chunk, u_tensor, v_tensor, input_size, tau
            )
            for kind, gate in (("input", input_gate), ("recurrent", recurrent_gate)):
                counts[kind]["total"] += gate.numel()
                counts[kind]["below_0_1"] += int(torch.count_nonzero(gate < 0.1).item())
                counts[kind]["above_0_9"] += int(torch.count_nonzero(gate > 0.9).item())
                counts[kind]["equal_0_1"] += int(torch.count_nonzero(gate == 0.1).item())
                counts[kind]["equal_0_9"] += int(torch.count_nonzero(gate == 0.9).item())
    return counts


def _mean_sem(values: list[float]) -> dict[str, float | list[float]]:
    """Return seed-level mean, SEM, and the retained seed values."""

    array = np.asarray(values, dtype=np.float64)
    if array.shape != (10,):
        raise RuntimeError(f"Expected ten seed values, got {array.shape}")
    return {
        "mean": float(array.mean()),
        "sem": float(array.std(ddof=1) / np.sqrt(array.size)),
        "seed_values": array.tolist(),
    }


def main() -> None:
    """Audit ten trajectories and write strict endpoint fractions."""

    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("chunk-size must be positive")
    paths = [
        args.trajectory_root / f"seed{seed:02d}" / "gawf_gate_trajectory.npz"
        for seed in range(1, 11)
    ]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing trajectories: " + ", ".join(map(str, missing)))

    device = torch.device(args.device)
    seeds: list[dict[str, object]] = []
    fractions = {
        kind: {"below_0_1": [], "above_0_9": [], "equal_0_1": [], "equal_0_9": []}
        for kind in ("input", "recurrent")
    }
    for seed, path in enumerate(paths, start=1):
        with np.load(path, allow_pickle=False) as arrays:
            counts = count_gate_endpoints(
                arrays["feedback"],
                arrays["U"],
                arrays["V"],
                int(arrays["weight_ih"].shape[1]),
                0.5,
                args.chunk_size,
                device,
            )
        seed_record: dict[str, object] = {"seed": seed, "trajectory": str(path)}
        for kind in ("input", "recurrent"):
            total = counts[kind]["total"]
            summary = counts[kind] | {
                key + "_fraction": counts[kind][key] / total
                for key in ("below_0_1", "above_0_9", "equal_0_1", "equal_0_9")
            }
            seed_record[kind] = summary
            for key in fractions[kind]:
                fractions[kind][key].append(summary[key + "_fraction"])
        seeds.append(seed_record)
        print(f"seed {seed:02d}/10 complete", flush=True)

    aggregate = {
        kind: {key: _mean_sem(values) for key, values in fractions[kind].items()}
        for kind in ("input", "recurrent")
    }
    payload = {
        "definition": {
            "closed": "g < 0.1",
            "open": "g > 0.9",
            "threshold_equality": "counted separately",
            "reset": "all-zero pre-step feedback frames excluded",
            "aggregation": "fraction within seed, then mean and SEM across seeds 1-10",
        },
        "seeds": seeds,
        "aggregate": aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
