"""Verify the portable aim3_rnn runtime contract and optional CUDA acceleration.

The command checks the active Conda environment, Python and package versions, optional sequence
model dependencies, CUDA availability, and an executable ``torch.compile`` smoke test.

Outputs:
- JSON on stdout containing versions, capabilities, warnings, and failures.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
from typing import Any


def parse_args() -> argparse.Namespace:
    """Parse environment verification arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("macos", "linux-login", "linux-cuda"),
        required=True,
    )
    parser.add_argument(
        "--compile-smoke",
        action="store_true",
        help="Execute a tiny compiled tensor function; recommended on CUDA workers.",
    )
    parser.add_argument(
        "--project-smoke",
        action="store_true",
        help="Exercise compiled ANN/GaWF plus Mamba and S5 on a CUDA worker.",
    )
    return parser.parse_args()


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _run_project_cuda_smoke() -> dict[str, str]:
    """Exercise project-specific accelerated paths on one CUDA device."""

    import torch

    from utils.atari_dqn_models import AtariQNetwork
    from utils.atari_train_acceleration import AtariAcceleration
    from utils.recurrent_cores import (
        GaWFCore,
        MambaCore,
        S5Core,
        configure_gawf_feedback_acceleration,
    )

    device = torch.device("cuda")
    results: dict[str, str] = {}

    ann = AtariQNetwork(
        num_actions=6,
        input_channels=1,
        model_type="ann",
        encoder_feature_dim=32,
    ).to(device)
    ann_forward = AtariAcceleration(device=device, compile_model=True).compile_callable(
        ann.forward_sequence
    )
    obs = torch.randint(0, 256, (1, 2, 1, 84, 84), dtype=torch.uint8, device=device)
    dones = torch.zeros(1, 2, device=device)
    ann_q, _ = ann_forward(obs, dones)
    ann_q.square().mean().backward()
    results["ann_full_compile"] = "passed"

    gawf = GaWFCore(8, 16, feedback_dim=6).to(device)
    configured = configure_gawf_feedback_acceleration(gawf, enabled=True)
    if configured != 1:
        raise RuntimeError(f"expected one compiled GaWF core, configured={configured}")
    gawf_out = gawf.step(
        torch.randn(2, 8, device=device),
        torch.randn(2, 16, device=device),
        torch.randn(2, 6, device=device),
    )
    gawf_out.square().mean().backward()
    results["gawf_feedback_compile"] = "passed"

    mamba = MambaCore(input_size=8, d_model=16, d_state=8).to(device)
    mamba_out, _ = mamba(torch.randn(2, 4, 8, device=device))
    mamba_out.square().mean().backward()
    results["mamba_cuda"] = "passed"

    s5 = S5Core(input_size=8, d_model=16, state_size=8).to(device)
    s5_out, _ = s5(torch.randn(2, 4, 8, device=device))
    s5_out.square().mean().backward()
    results["s5_cuda"] = "passed"
    torch.cuda.synchronize()
    return results


def verify_environment(
    profile: str,
    compile_smoke: bool,
    project_smoke: bool = False,
) -> dict[str, Any]:
    """Return a machine-readable environment report for ``profile``."""

    import torch

    versions = {
        name: _package_version(name)
        for name in (
            "torch",
            "numpy",
            "gymnasium",
            "ale-py",
            "opencv-python",
            "s5-pytorch",
            "mamba-ssm",
            "causal-conv1d",
        )
    }
    failures: list[str] = []
    warnings: list[str] = []
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    if conda_env not in {"aim3_rnn", "aim3_rnn_next"}:
        failures.append(f"expected aim3_rnn environment, active={conda_env!r}")
    if sys.version_info[:2] != (3, 11):
        failures.append(f"Python 3.11 required, active={platform.python_version()}")

    required = ["torch", "numpy", "gymnasium", "ale-py", "opencv-python", "s5-pytorch"]
    if profile.startswith("linux"):
        required.extend(("mamba-ssm", "causal-conv1d"))
    for name in required:
        if versions[name] is None:
            failures.append(f"missing required distribution: {name}")

    cuda_available = torch.cuda.is_available()
    if profile == "linux-cuda" and not cuda_available:
        failures.append("CUDA worker profile requires torch.cuda.is_available()")
    if profile == "linux-login" and not cuda_available:
        warnings.append("CUDA unavailable on login node; run linux-cuda verification in a job")
    if profile == "macos" and versions["mamba-ssm"] is None:
        warnings.append("Mamba CUDA kernels are intentionally absent from the macOS profile")

    compile_result = "not_requested"
    if compile_smoke:
        device = torch.device("cuda" if cuda_available else "cpu")
        try:
            compiled = torch.compile(lambda tensor: tensor.square() + 1, fullgraph=True)
            value = compiled(torch.ones(8, device=device))
            if cuda_available:
                torch.cuda.synchronize()
            if not torch.equal(value.cpu(), torch.full((8,), 2.0)):
                raise RuntimeError("compiled function returned an unexpected value")
            compile_result = "passed"
        except Exception as exc:  # Runtime/compiler failures vary by PyTorch release.
            compile_result = f"failed: {type(exc).__name__}: {exc}"
            failures.append("torch.compile smoke test failed")

    project_result: dict[str, str] | str = "not_requested"
    if project_smoke:
        if profile != "linux-cuda" or not cuda_available:
            project_result = "failed: project smoke requires the linux-cuda profile"
            failures.append("project CUDA smoke test requires a CUDA worker")
        else:
            try:
                project_result = _run_project_cuda_smoke()
            except Exception as exc:
                project_result = f"failed: {type(exc).__name__}: {exc}"
                failures.append("project CUDA smoke test failed")

    return {
        "ok": not failures,
        "profile": profile,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "conda_env": conda_env,
        "python": platform.python_version(),
        "versions": versions,
        "cuda": {
            "available": cuda_available,
            "build": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        },
        "torch_compile": compile_result,
        "project_cuda_smoke": project_result,
        "warnings": warnings,
        "failures": failures,
    }


def main() -> None:
    """Run the verifier and exit nonzero when the profile contract is not met."""

    args = parse_args()
    report = verify_environment(args.profile, args.compile_smoke, args.project_smoke)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
