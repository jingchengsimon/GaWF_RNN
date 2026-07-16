#!/usr/bin/env bash
""":"
Install CUDA extensions after PyTorch exists in the active Linux environment.

This second phase is required because causal-conv1d and mamba-ssm inspect PyTorch during their
build. PEP 517 build isolation hides the already installed PyTorch and fails before compilation.
:"""

set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "aim3_rnn" && "${CONDA_DEFAULT_ENV:-}" != "aim3_rnn_next" ]]; then
  echo "Activate aim3_rnn or aim3_rnn_next before installing CUDA extras" >&2
  exit 2
fi

python -c 'import torch; print(f"torch={torch.__version__} cuda_build={torch.version.cuda}")'
export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
export MAX_JOBS="${MAX_JOBS:-8}"
python -m pip install --no-build-isolation \
  causal-conv1d==1.5.0.post8 \
  mamba-ssm==2.2.4 \
  transformers==4.40.2
