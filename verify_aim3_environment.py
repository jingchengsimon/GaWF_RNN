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
    return parser.parse_args()


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def verify_environment(profile: str, compile_smoke: bool) -> dict[str, Any]:
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
        "warnings": warnings,
        "failures": failures,
    }


def main() -> None:
    """Run the verifier and exit nonzero when the profile contract is not met."""

    args = parse_args()
    report = verify_environment(args.profile, args.compile_smoke)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
