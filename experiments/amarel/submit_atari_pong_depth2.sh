#!/usr/bin/env bash
# Submit or dry-run the parameter-matched Pong depth-2 pilot.
# Default protocol: frame_skip=1, frame_stack=1, BF16 AMP, TF32, torch.compile.

set -euo pipefail

SEEDS_CSV=42
DRY_RUN=0
CONCURRENCY=10
FRAME_SKIP=1
AMP_DTYPE=bfloat16
while (( $# )); do
  case "$1" in
    --seeds) SEEDS_CSV="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --frame-skip) FRAME_SKIP="$2"; shift 2 ;;
    --amp-dtype) AMP_DTYPE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"
MODELS=(ann rnn gru lstm gawf)
N_TASKS=$((${#MODELS[@]} * ${#SEEDS[@]} * 2))

if (( DRY_RUN )); then
  for ((task=0; task<N_TASKS; task++)); do
    model="${MODELS[$((task % ${#MODELS[@]}))]}"
    rest=$((task / ${#MODELS[@]}))
    setting=$((rest / ${#SEEDS[@]}))
    seed="${SEEDS[$((rest % ${#SEEDS[@]}))]}"
    compile=0
    [[ "$model" == "ann" ]] && compile=1
    printf 'task=%d model=%s setting=%d seed=%s layers=2 frame_skip=%s amp=%s compile=%s\n' \
      "$task" "$model" "$setting" "$seed" "$FRAME_SKIP" "$AMP_DTYPE" "$compile"
  done
  exit 0
fi

source /home/js3269/enter/etc/profile.d/conda.sh
conda activate aim3_rnn
MATCH_DIR=results/atari_param_match_depth2
python -m experiments.generalization.atari_ssm_param_match \
  --num_layers 2 \
  --models rnn gru lstm gawf \
  --out_dir "$MATCH_DIR"
python - "$MATCH_DIR/atari_param_match.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["anchor_num_layers"] == 1
assert d["candidate_num_layers"] == 2
assert set(("ann", "rnn", "gru", "lstm", "gawf")) <= set(d["matched"])
assert all(d["matched"][m].get("num_layers") == 2 for m in ("ann", "rnn", "gru", "lstm", "gawf"))
PY

mkdir -p experiments/amarel/artifacts/atari_pong_depth2
sbatch \
  --array="0-$((N_TASKS - 1))%${CONCURRENCY}" \
  --export="ALL,SEEDS_CSV=${SEEDS_CSV},FRAME_SKIP=${FRAME_SKIP},AMP_DTYPE=${AMP_DTYPE},ALLOW_TF32=1,COMPILE_MODEL=1,AIM3_NUM_WORKERS=12,AIM3_PIN_MEMORY=1" \
  experiments/amarel/run_atari_pong_depth2_array.sh
