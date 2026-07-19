#!/usr/bin/env bash
#SBATCH --job-name=clutter-dl-gpu-smoke
#SBATCH --partition=cgpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=volta
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=experiments/amarel/artifacts/clutter_dataloader_gpu_synthetic_smoke/%j.out
#SBATCH --error=experiments/amarel/artifacts/clutter_dataloader_gpu_synthetic_smoke/%j.err

set -euo pipefail

ROOT="${AIM3_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$ROOT" || ! -f "$ROOT/train_model.py" ]]; then
  echo "AIM3_ROOT or SLURM_SUBMIT_DIR must identify the project root." >&2
  exit 2
fi
if [[ -z "${AIM3_CONDA_INIT:-}" || ! -f "$AIM3_CONDA_INIT" ]]; then
  echo "AIM3_CONDA_INIT must identify the Amarel Conda initialization script." >&2
  exit 2
fi

cd "$ROOT"
source "$AIM3_CONDA_INIT"
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"

RESULTS_ROOT="${AIM3_RESULTS_PATH:?AIM3_RESULTS_PATH is required}"
OUTPUT_DIR="$RESULTS_ROOT/benchmarks/clutter_dataloader/amarel_gpu_synthetic_${SLURM_JOB_ID}"
LOCAL_TMP="${SLURM_TMPDIR:-/tmp/${USER}/aim3_${SLURM_JOB_ID}}"
SMOKE_DIR="$LOCAL_TMP/clutter_uint8_smoke"
mkdir -p "$OUTPUT_DIR" "$SMOKE_DIR"

SMOKE_DIR="$SMOKE_DIR" python -c 'import os; from pathlib import Path; import numpy as np; import pandas as pd; p=Path(os.environ["SMOKE_DIR"]); n=4096; frames=(np.arange(n,dtype=np.uint16)%256).astype(np.uint8)[:,None,None]; frames=np.broadcast_to(frames,(n,96,96)).copy(); labels=pd.DataFrame({"fg_char_id":np.arange(n)%10,"fg_char_x":np.arange(n)%96,"fg_char_y":(np.arange(n)*3)%96}); [(np.save(p/f"stimulus_reg-{split}-40h-uint8.npy",frames),labels.to_csv(p/f"stimulus_reg-{split}-40h-uint8.tsv",sep="\t",index=True)) for split in ("train","validation")]'

python experiments/clutter/benchmark_dataloader_pipeline.py \
  --data-dir "$SMOKE_DIR" \
  --output "$OUTPUT_DIR/e2e_amp.json" \
  --device cuda \
  --batch-size 8 \
  --num-workers 2 \
  --pin-memory \
  --warmup-batches 1 \
  --num-batches 2 \
  --mode e2e \
  --variants uint8_sample_stacked_global uint8_device_compact_block256

AIM3_BATCH_SIZE=8 AIM3_NUM_WORKERS=2 AIM3_PIN_MEMORY=1 DISABLE_TQDM=1 \
python train_model.py \
  --model_types rnn \
  --hidden_sizes 8 \
  --num_layers 1 \
  --chan_num 2 \
  --data_dir "$SMOKE_DIR" \
  --results_dir "$OUTPUT_DIR/results" \
  --lrs 0.001 \
  --wds 0.0 \
  --cnn_dropout 0.0 \
  --rnn_dropout 0.0 \
  --num_epochs 1 \
  --patience 0 \
  --seed 42 \
  --use_acceleration \
  --use_sector_mode \
  --use_mmap \
  --result_suffix synthetic_default

OUTPUT_DIR="$OUTPUT_DIR" python -c 'import json,os; from pathlib import Path; root=Path(os.environ["OUTPUT_DIR"]); paths=list((root/"results").rglob("*_metrics.json")); assert len(paths)==1,paths; d=json.loads(paths[0].read_text()); expected={"dataset_suffix":"40h-uint8","input_cast_mode":"device","frame_layout":"compact","shuffle_block_size":-1,"actual_epochs":1}; assert all(d.get(k)==v for k,v in expected.items()),{k:d.get(k) for k in expected}; (root/"production_smoke_summary.json").write_text(json.dumps({"metrics_path":str(paths[0]),**expected},indent=2)+"\n")'

echo "status=complete"
echo "output_dir=$OUTPUT_DIR"
