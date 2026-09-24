"""Find RNN fixed points with absent feedforward drive or fixed real movie inputs.

Inputs: canonical checkpoint, saved continuous event bank, mmap movie. Outputs: all candidate
states/residuals, unique numerical roots, local eigenspectra, perturbation checks and provenance.
Reconstructs the historical tanh nn.RNN from checkpoint-compatible affine weights; autonomous
analysis overrides only bias_ih and supplies zero features, removing the complete input pathway
while preserving recurrent bias. No training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.func import functional_call

from utils.analysis.anal_helpers import build_model_from_ckpt
from utils.analysis.clutter.continuous_switch_pca import _stacked_frames


def step(
    rnn: torch.nn.RNN, h: torch.Tensor, inputs: torch.Tensor, autonomous: bool = False
) -> torch.Tensor:
    """Advance the canonical RNN with the selected feedforward intervention."""
    if autonomous:
        inputs = torch.zeros_like(inputs)
        _, next_h = functional_call(
            rnn, {"bias_ih_l0": torch.zeros_like(rnn.bias_ih_l0)}, (inputs[:, None], h[None])
        )
    else:
        _, next_h = rnn(inputs[:, None], h[None])
    return next_h[0]


def search(
    rnn: torch.nn.RNN,
    initial: torch.Tensor,
    inputs: torch.Tensor,
    autonomous: bool,
    iterations: int,
    lr: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Minimize squared update residual with independent batched initializations."""
    h = initial.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([h], lr=lr)
    best = h.detach().clone()
    best_q = torch.full((len(h),), float("inf"), device=h.device, dtype=h.dtype)
    for iteration in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        delta = step(rnn, h, inputs, autonomous) - h
        q = 0.5 * delta.square().sum(dim=1)
        with torch.no_grad():
            improved = q < best_q
            best[improved] = h[improved]
            best_q[improved] = q[improved]
        q.sum().backward()
        optimizer.step()
        if iteration == iterations // 2:
            optimizer.param_groups[0]["lr"] = lr / 10
    with torch.no_grad():
        q = 0.5 * (step(rnn, h, inputs, autonomous) - h).square().sum(dim=1)
        improved = q < best_q
        best[improved], best_q[improved] = h[improved], q[improved]
    if not torch.isfinite(best).all() or not torch.isfinite(best_q).all():
        raise RuntimeError("Non-finite fixed-point candidates")
    return best.cpu().numpy().astype(np.float32), best_q.cpu().numpy().astype(np.float32)


def characterize(
    rnn: torch.nn.RNN,
    candidates: np.ndarray,
    inputs: torch.Tensor,
    autonomous: bool,
    tolerance: float,
    merge_rms: float,
) -> dict[str, np.ndarray]:
    """Refine in float64; deduplicate within input; characterize numerical roots only."""
    roots, residuals, spectra, jacobians, errors = [], [], [], [], []
    candidates_t = torch.as_tensor(candidates, device=inputs.device, dtype=torch.float64)
    rnn = rnn.double()
    input64 = inputs[:1].double()
    dimension = candidates.shape[-1]
    eye = torch.eye(dimension, device=inputs.device, dtype=torch.float64)
    for h0 in candidates_t:
        h = h0.detach().clone()
        # Damped Newton refinement of Adam candidates; accept only residual-reducing steps.
        for _ in range(12):
            f = step(rnn, h[None], input64, autonomous)[0]
            residual = f - h
            if float(residual.abs().max()) <= tolerance:
                break
            jac = torch.autograd.functional.jacobian(
                lambda state: step(rnn, state[None], input64, autonomous)[0], h
            )
            try:
                update = torch.linalg.solve(jac - eye, -residual)
            except torch.linalg.LinAlgError:
                break
            accepted = False
            for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
                trial = (h + scale * update).detach()
                delta = step(rnn, trial[None], input64, autonomous)[0] - trial
                if torch.isfinite(delta).all() and delta.square().sum() < residual.square().sum():
                    h = trial
                    accepted = True
                    break
            if not accepted:
                break
        residual = step(rnn, h[None], input64, autonomous)[0] - h
        err = float(residual.abs().max())
        if err > tolerance or not np.isfinite(err):
            continue
        point = h.detach().cpu().numpy()
        if any(np.sqrt(np.mean((point - old) ** 2)) < merge_rms for old in roots):
            continue
        jac = torch.autograd.functional.jacobian(
            lambda state: step(rnn, state[None], input64, autonomous)[0], h
        )
        rng = np.random.default_rng(0)
        perturb = torch.as_tensor(
            rng.normal(size=(8, dimension)), device=h.device, dtype=torch.float64
        )
        perturb = perturb / perturb.norm(dim=1, keepdim=True) * 1e-4
        actual = step(rnn, h[None] + perturb, input64.expand(8, -1), autonomous)
        base = step(rnn, h[None], input64, autonomous)
        predicted_delta = perturb @ jac.T
        rel_error = (
            (actual - base - predicted_delta).norm(dim=1)
            / (actual - base).norm(dim=1).clamp_min(1e-15)
        ).mean()
        j = jac.detach().cpu().numpy()
        roots.append(point)
        residuals.append(err)
        spectra.append(np.linalg.eigvals(j))
        jacobians.append(j)
        errors.append(float(rel_error))
    rnn.float()
    eigen = np.asarray(spectra, dtype=np.complex128).reshape(-1, dimension)
    return {
        "roots": np.asarray(roots, dtype=np.float32).reshape(-1, dimension),
        "residual_max": np.asarray(residuals, dtype=np.float32),
        "jacobians": np.asarray(jacobians, dtype=np.float32).reshape(-1, dimension, dimension),
        "eigenvalues_real": eigen.real.astype(np.float32),
        "eigenvalues_imag": eigen.imag.astype(np.float32),
        "spectral_radius": np.max(np.abs(eigen), axis=1).astype(np.float32),
        "linearization_relative_error": np.asarray(errors, dtype=np.float32),
    }


def run(args: argparse.Namespace) -> None:
    """Analyze all requested input conditions for one independently trained RNN seed."""
    if args.save_dir.exists():
        raise FileExistsError(f"Refusing overwrite: {args.save_dir}")
    device = torch.device(args.device)
    torch.set_num_threads(2)
    torch.backends.cudnn.enabled = False
    torch.manual_seed(args.seed)
    model = build_model_from_ckpt(str(args.ckpt), 9, device)
    weights = model.core.rnn
    rnn = torch.nn.RNN(weights.input_size, weights.hidden_size, batch_first=True).to(device)
    rnn.load_state_dict(weights.state_dict(), strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in rnn.parameters():
        parameter.requires_grad_(False)
    with np.load(args.event_dir / "event_trajectories.npz") as data:
        trajectories, events = data["raw_hidden"], data["event_frames"]
        relative = data["relative_frames"]
    movie = np.load(args.movie, mmap_mode="r")
    anchors = np.stack((events - 1, events), axis=1).reshape(-1)
    frames = np.concatenate([_stacked_frames(movie, int(t), int(t) + 1, 2) for t in anchors])
    with torch.no_grad():
        features = model.encode_frames(torch.tensor(frames, device=device)[None])[0]
    inputs = torch.cat((torch.zeros_like(features[:1]), features))
    rng = np.random.default_rng(args.seed)
    pool = trajectories.reshape(-1, trajectories.shape[-1])
    sampled = rng.choice(len(pool), args.starts, replace=len(pool) < args.starts)
    initial = pool[sampled].copy()
    initial[args.starts // 2 :] += rng.normal(0, 0.01, initial[args.starts // 2 :].shape)
    initial_t = torch.tensor(initial, device=device)
    # Real-input one-step reproduction checks the original saved temporal indexing.
    zero = int(np.flatnonzero(relative == 0)[0])
    previous = torch.tensor(
        trajectories[:, [zero - 2, zero - 1]].reshape(-1, rnn.hidden_size), device=device
    )
    expected = trajectories[:, [zero - 1, zero]].reshape(-1, rnn.hidden_size)
    with torch.no_grad():
        reproduced = step(rnn, previous, features).cpu().numpy()
    np.testing.assert_allclose(reproduced, expected, rtol=1e-4, atol=2e-5)
    args.save_dir.mkdir(parents=True)
    np.savez_compressed(
        args.save_dir / "inputs_and_initializations.npz",
        anchor_frames=anchors.astype(np.int64),
        features=features.cpu().numpy().astype(np.float32),
        initial=initial.astype(np.float32),
        pool_indices=sampled.astype(np.int64),
    )
    summaries = []
    for index, feature in enumerate(inputs):
        autonomous = index == 0
        name = "autonomous" if autonomous else f"real_frame{anchors[index - 1]:05d}"
        condition_input = feature[None].expand(args.starts, -1)
        candidates, q = search(
            rnn, initial_t, condition_input, autonomous, args.iterations, args.lr
        )
        result = characterize(
            rnn, candidates, condition_input, autonomous, args.residual_tolerance, args.merge_rms
        )
        np.savez_compressed(args.save_dir / f"{name}.npz", candidates=candidates, q=q, **result)
        spectral = result["spectral_radius"]
        summaries.append(
            {
                "condition": name,
                "roots": len(spectral),
                "stable": int((spectral < 1 - 1e-5).sum()),
                "unstable": int((spectral > 1 + 1e-5).sum()),
                "near_unit": int((np.abs(spectral - 1) <= 1e-5).sum()),
                "minimum_candidate_q": float(q.min()),
            }
        )
        print(json.dumps(summaries[-1]), flush=True)
    metadata = {
        "seed": args.seed,
        "checkpoint": str(args.ckpt),
        "event_dir": str(args.event_dir),
        "movie": str(args.movie),
        "autonomous": "W_ih*u + b_ih removed; W_hh*h + b_hh retained",
        "real": "fixed CNN features from actual stacked frames at switch-1 and switch",
        "starts": args.starts,
        "iterations": args.iterations,
        "residual_tolerance_max_absolute": args.residual_tolerance,
        "merge_rms": args.merge_rms,
        "conditions": summaries,
        "limitations": "Multistart numerical search is not an exhaustive root census; failed candidates retained.",
    }
    (args.save_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.save_dir / ".complete").touch()


def main() -> None:
    """Parse the single-seed analysis command."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--event_dir", type=Path, required=True)
    p.add_argument("--movie", type=Path, required=True)
    p.add_argument("--save_dir", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--starts", type=int, default=64)
    p.add_argument("--iterations", type=int, default=2000)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--residual_tolerance", type=float, default=1e-7)
    p.add_argument("--merge_rms", type=float, default=1e-4)
    run(p.parse_args())


if __name__ == "__main__":
    main()
