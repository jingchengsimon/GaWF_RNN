# Reproducible `aim3_rnn` environments

Code synchronization and environment synchronization are separate. A run is reproducible only
when its Git commit and verified environment profile are both recorded.

## Canonical profiles

- `aim3_rnn-linux-cuda.yml`: Amarel and sjc-remote training. Python 3.11 is required because
  `torch.compile` is unsupported on Python 3.14. The CUDA, Mamba, and S5 versions match the
  validated Amarel stack.
- `aim3_rnn-macos.yml`: Mac and Mac mini development/tests. Mamba is intentionally omitted because
  its CUDA kernels are Linux-only; Mamba training must be verified on a Linux GPU host.
- `environment_full.yml` at the repository root is a historical machine export, not a portable
  environment definition.

Create a new environment without changing an existing one:

```bash
conda env create -n aim3_rnn_next -f environments/aim3_rnn-linux-cuda.yml
conda activate aim3_rnn_next
bash environments/install_linux_cuda_extras.sh
python verify_aim3_environment.py --profile linux-cuda --compile-smoke --project-smoke
```

The CUDA extensions are intentionally installed in a second phase. Their build scripts import
PyTorch, so pip build isolation would otherwise fail even though PyTorch is already present in
the Conda environment.

On macOS, use `environments/aim3_rnn-macos.yml` and profile `macos`. After the new environment
passes, keep the old environment as a rollback copy and rename the verified environment to
`aim3_rnn`. Do not repair an active training environment in place.

## Version contract

The portable contract is Python 3.11 plus the versions/ranges in the profile YAML. Exact package
build strings may differ between macOS, Amarel, and sjc-remote. For each formal run, save these
alongside the source commit:

```bash
git rev-parse HEAD
python verify_aim3_environment.py --profile linux-cuda --compile-smoke --project-smoke
conda env export --from-history
python -m pip freeze
```

The verifier reports capabilities, not only package names. In particular it catches a Python
version that imports PyTorch but cannot execute `torch.compile`.
