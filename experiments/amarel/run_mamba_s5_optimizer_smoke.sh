#!/usr/bin/env bash
#SBATCH --job-name=aim3-mamba-s5-smoke
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=experiments/amarel/artifacts/mamba_s5_optimizer_smoke/%j.out
#SBATCH --error=experiments/amarel/artifacts/mamba_s5_optimizer_smoke/%j.err

# Run required smoke checks before the full Mamba/S5 grid.
# Inspect the output logs before bulk submission:
#   - Mamba A_log/D should appear in the no_decay log line.
#   - S5 Lambda/B/log_step-like parameters should appear in ssm_core.
#   - S5 C/readout/input projections should not appear in ssm_core.
#   - S5 synthetic AMP forward/backward should complete on CUDA.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$ROOT" || ! -f "$ROOT/train_model.py" ]]; then
  ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$ROOT"

ART_ROOT="$ROOT/experiments/amarel/artifacts/mamba_s5_optimizer_smoke"
MARKER="$ART_ROOT/optimizer_grouping_smoke.done"
mkdir -p "$ART_ROOT"
rm -f "$MARKER"

if [[ -n "${AIM3_SETUP_CMD:-}" ]]; then
  eval "$AIM3_SETUP_CMD"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  if [[ -n "$CONDA_BASE" && -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "${AIM3_CONDA_ENV:-aim3_rnn}" || true
  fi
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ -n "${AIM3_DATA_DIR:-}" ]]; then
  DATA_DIR="$AIM3_DATA_DIR"
elif [[ -d "/scratch/${USER}/stimuli" ]]; then
  DATA_DIR="/scratch/${USER}/stimuli"
elif [[ -d "/cache/${USER}/stimuli" ]]; then
  DATA_DIR="/cache/${USER}/stimuli"
elif [[ -d "$ROOT/stimuli" ]]; then
  DATA_DIR="$ROOT/stimuli"
else
  echo "Data directory not found. Set AIM3_DATA_DIR or create stimuli under scratch/cache/repo." >&2
  exit 2
fi

SMOKE_DATA_SUFFIX="${SMOKE_DATA_SUFFIX:-4h-float32}"
SMOKE_EVAL_SUFFIX="${SMOKE_EVAL_SUFFIX:-40h-float32}"
SMOKE_NUM_EPOCHS="${SMOKE_NUM_EPOCHS:-1}"
SMOKE_LR="${SMOKE_LR:-0.001}"
SMOKE_WD="${SMOKE_WD:-0.0001}"
SMOKE_SEED="${SMOKE_SEED:-42}"
MAMBA_D_MODEL="${MAMBA_D_MODEL:-170}"
S5_D_MODEL="${S5_D_MODEL:-256}"
S5_STATE_SIZE="${S5_STATE_SIZE:-128}"
S5_SSM_LR_SCALE="${S5_SSM_LR_SCALE:-0.1}"

run_smoke_model() {
  local model="$1"
  local log_path="$ART_ROOT/${model}_optimizer_grouping.log"
  local result_suffix="gen_mamba_s5_optimizer_smoke/${model}"
  local model_args=()

  case "$model" in
    mamba)
      model_args=(--model_types mamba --mamba_d_models "$MAMBA_D_MODEL")
      ;;
    s5)
      model_args=(
        --model_types s5
        --s5_d_models "$S5_D_MODEL"
        --s5_state_sizes "$S5_STATE_SIZE"
        --s5_ssm_lr_scale "$S5_SSM_LR_SCALE"
      )
      ;;
    *)
      echo "Unsupported smoke model: $model" >&2
      exit 2
      ;;
  esac

  echo "[$(date -Is)] smoke model=$model lr=$SMOKE_LR wd=$SMOKE_WD" | tee "$log_path"
  DISABLE_TQDM=1 python train_model.py \
    "${model_args[@]}" \
    --data_suffix "$SMOKE_DATA_SUFFIX" \
    --eval_data_suffix "$SMOKE_EVAL_SUFFIX" \
    --data_dir "$DATA_DIR" \
    --lrs "$SMOKE_LR" \
    --wds "$SMOKE_WD" \
    --cnn_dropout 0.0 \
    --rnn_dropout 0.5 \
    --num_epochs "$SMOKE_NUM_EPOCHS" \
    --patience 0 \
    --seed "$SMOKE_SEED" \
    --use_acceleration \
    --use_sector_mode \
    --result_suffix "$result_suffix" 2>&1 | tee -a "$log_path"
}

run_smoke_model mamba
run_smoke_model s5

S5_AMP_LOG="$ART_ROOT/s5_amp_synthetic_smoke.log"
echo "[$(date -Is)] synthetic AMP smoke model=s5" | tee "$S5_AMP_LOG"
python - <<'PY' 2>&1 | tee -a "$S5_AMP_LOG"
import importlib
importlib.import_module("numpy")

import torch

from utils.train_s5_core import S5Conv


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("S5 AMP smoke requires CUDA")

    torch.manual_seed(7)
    device = torch.device("cuda")
    model = S5Conv(
        num_classes=10,
        num_pos=9,
        device="cuda",
        cnn_dropout=0.0,
        rnn_dropout=0.5,
        s5_d_model=64,
        s5_state_size=64,
    ).to(device)
    model.train()

    bsz, tlen = 2, 8
    frames = torch.randn(bsz, tlen, 2, 96, 96, device=device)
    labels_char = torch.randint(0, 10, (bsz, tlen), device=device)
    labels_pos = torch.randint(0, 9, (bsz, tlen), device=device)
    ce = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    has_complex_params = any(
        param.requires_grad and torch.is_complex(param)
        for param in model.parameters()
    )
    scaler = None if has_complex_params else torch.amp.GradScaler("cuda")
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        out_char, out_pos = model(frames)
        loss = (
            ce(out_char.reshape(-1, 10), labels_char.reshape(-1))
            + ce(out_pos.reshape(-1, 9), labels_pos.reshape(-1))
        )

    if scaler is None:
        loss.backward()
        optimizer.step()
    else:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    print(f"S5_AMP_SMOKE_OK loss={float(loss.detach().cpu()):.6f}")


if __name__ == "__main__":
    main()
PY

MAMBA_LOG="$ART_ROOT/mamba_optimizer_grouping.log"
S5_LOG="$ART_ROOT/s5_optimizer_grouping.log"

grep -q "no_decay:" "$MAMBA_LOG"
grep -q "A_log" "$MAMBA_LOG"
grep -Eq "([^[:alnum:]_]|^)D([^[:alnum:]_]|$|[.])" "$MAMBA_LOG"

grep -q "S5 ssm_core:" "$S5_LOG"
grep -q "S5 decay" "$S5_LOG"
grep -Eq "S5 ssm_core: .*Lambda" "$S5_LOG"
grep -Eq "S5 ssm_core: .*B" "$S5_LOG"
grep -Eq "S5 ssm_core: .*(log_step|log_dt|inv_dt)" "$S5_LOG"
grep -q "S5_AMP_SMOKE_OK" "$S5_AMP_LOG"
if grep -Eq "S5 ssm_core: .*([.]C|[.]C_|C[.])" "$S5_LOG"; then
  echo "S5 C/readout-like parameter appears in ssm_core; inspect grouping before full grid." >&2
  exit 1
fi

{
  echo "status=done"
  echo "timestamp=$(date -Is)"
  echo "mamba_log=$MAMBA_LOG"
  echo "s5_log=$S5_LOG"
  echo "s5_amp_log=$S5_AMP_LOG"
  echo "checked=Mamba A_log/D no_decay; S5 Lambda/B/dt ssm_core; S5 C not in ssm_core; S5 AMP synthetic step"
} > "$MARKER"

echo "Optimizer grouping smoke passed: $MARKER"
