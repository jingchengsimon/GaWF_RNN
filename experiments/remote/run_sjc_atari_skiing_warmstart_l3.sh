#!/usr/bin/env bash
# Run one recoverable seed-1 Skiing weights-only adaptation on sjc-remote.

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

usage() {
  cat <<'EOF'
Usage: run_sjc_atari_skiing_warmstart_l3.sh --model lstm|gru|gawf \
  --cuda-device ID --source-checkpoint PATH --source-metrics PATH [options]

Options:
  --smoke              Run the fixed 25,000-step smoke and render its evaluation video.
  --results-root PATH  Root containing data/rl/atari (AIM3_RESULTS_PATH or ROOT/results).
  --run-tag TAG        Explicit unique result leaf under formal_20m_4mpertask_raw_seeds.
  --allow-incomplete-source
                       Allow a positive-step source from the 20M campaign.
  --extend-from-skiing-1m
                       Start a fresh 1M phase from a completed 1M Skiing model.
  --extend-from-skiing-2m
                       Start a fresh 2M phase from the completed cumulative-2M model.
  --total-timesteps N  Override the default 1M target for a resumable run.
  --gamma VALUE       Discount factor (default 0.99).
  --no-reward-clip    Preserve reward magnitudes in TD targets.
  --result-parent PATH  Explicit absolute parent for a separate protocol.
  --skip-smoke-video  Validate training artifacts without an evaluation video.
  --keep-replay-on-success  Retain this run's replay after completion.
  --requeue-on-pause   Requeue this Slurm task after a resumable interruption.
  --allow-total-timesteps-extension
                       Permit only an existing resumable run's target to increase.
  --dry-run            Validate inputs and print the resolved command without writing.
  -h, --help           Show this help.
EOF
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL=""
CUDA_DEVICE=""
SOURCE_CHECKPOINT=""
SOURCE_METRICS=""
RESULTS_ROOT="${AIM3_RESULTS_PATH:-${ROOT}/results}"
RUN_TAG=""
PHASE="formal"
ALLOW_INCOMPLETE_SOURCE=false
EXTEND_FROM_SKIING_1M=false
EXTEND_FROM_SKIING_2M=false
REQUESTED_TOTAL_TIMESTEPS=""
ALLOW_TOTAL_TIMESTEPS_EXTENSION=false
DRY_RUN=false
GAMMA=0.99
REWARD_ARGS=()
REWARD_CLIP=true
EXPLICIT_RESULT_PARENT=""
SKIP_SMOKE_VIDEO=false
REQUEUE_ON_PAUSE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="${2:-}"; shift 2 ;;
    --cuda-device) CUDA_DEVICE="${2:-}"; shift 2 ;;
    --source-checkpoint) SOURCE_CHECKPOINT="${2:-}"; shift 2 ;;
    --source-metrics) SOURCE_METRICS="${2:-}"; shift 2 ;;
    --results-root) RESULTS_ROOT="${2:-}"; shift 2 ;;
    --run-tag) RUN_TAG="${2:-}"; shift 2 ;;
    --smoke) PHASE="smoke"; shift ;;
    --allow-incomplete-source) ALLOW_INCOMPLETE_SOURCE=true; shift ;;
    --extend-from-skiing-1m) EXTEND_FROM_SKIING_1M=true; shift ;;
    --extend-from-skiing-2m) EXTEND_FROM_SKIING_2M=true; shift ;;
    --total-timesteps) REQUESTED_TOTAL_TIMESTEPS="${2:-}"; shift 2 ;;
    --gamma) GAMMA="${2:-}"; shift 2 ;;
    --no-reward-clip) REWARD_ARGS+=(--no_reward_clip); REWARD_CLIP=false; shift ;;
    --result-parent) EXPLICIT_RESULT_PARENT="${2:-}"; shift 2 ;;
    --skip-smoke-video) SKIP_SMOKE_VIDEO=true; shift ;;
    --keep-replay-on-success) REWARD_ARGS+=(--keep_replay_on_success); shift ;;
    --requeue-on-pause) REQUEUE_ON_PAUSE=true; shift ;;
    --allow-total-timesteps-extension) ALLOW_TOTAL_TIMESTEPS_EXTENSION=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$MODEL" in
  lstm) HIDDEN=373; FEEDBACK_MODE=none ;;
  gru) HIDDEN=458; FEEDBACK_MODE=none ;;
  gawf) HIDDEN=604; FEEDBACK_MODE=qvalues ;;
  *) echo "--model must be lstm, gru, or gawf" >&2; exit 2 ;;
esac
[[ "$CUDA_DEVICE" =~ ^[0-9]+$ ]] || {
  echo "--cuda-device must be a non-negative integer" >&2
  exit 2
}
[[ -f "$SOURCE_CHECKPOINT" ]] || {
  echo "Missing source checkpoint: $SOURCE_CHECKPOINT" >&2
  exit 2
}
[[ -f "$SOURCE_METRICS" ]] || {
  echo "Missing source metrics: $SOURCE_METRICS" >&2
  exit 2
}
[[ -n "$RESULTS_ROOT" ]] || { echo "--results-root must not be empty" >&2; exit 2; }
if [[ -n "$REQUESTED_TOTAL_TIMESTEPS" ]] \
  && ! [[ "$REQUESTED_TOTAL_TIMESTEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "--total-timesteps must be a positive integer" >&2
  exit 2
fi
if [[ "$EXTEND_FROM_SKIING_1M" == true && "$EXTEND_FROM_SKIING_2M" == true ]]; then
  echo "Skiing extension modes are mutually exclusive" >&2
  exit 2
fi

python - "$SOURCE_METRICS" "$MODEL" "$HIDDEN" "$ALLOW_INCOMPLETE_SOURCE" \
  "$EXTEND_FROM_SKIING_1M" "$EXTEND_FROM_SKIING_2M" <<'PY'
import json
import os
import sys

path, model, hidden, allow_incomplete, extension_1m, extension_2m = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    metrics = json.load(handle)
expected_envs = {
    "ALE/Pong-v5",
    "ALE/Breakout-v5",
    "ALE/Assault-v5",
    "ALE/Seaquest-v5",
    "ALE/Skiing-v5",
}
checks = {
    "model_type": (metrics.get("model_type"), model),
    "hidden_size": (int(metrics.get("hidden_size", -1)), int(hidden)),
    "num_layers": (int(metrics.get("num_layers", -1)), 3),
    "action_space_mode": (metrics.get("action_space_mode"), "full18"),
    "num_actions": (int(metrics.get("num_actions", -1)), 18),
}
mismatches = {key: value for key, value in checks.items() if value[0] != value[1]}
if mismatches:
    raise RuntimeError(f"Source metrics do not match the requested model: {mismatches}")
source_step = int(metrics.get("global_step", -1))
if extension_1m == "true" or extension_2m == "true":
    expected_step = 1_000_000
    if extension_2m == "true" and model == "gawf":
        expected_step = 2_000_000
    skiing_checks = {
        "global_step": (source_step, expected_step),
        "env_id": (metrics.get("env_id"), "ALE/Skiing-v5"),
        "atari_env_protocol": (
            metrics.get("atari_env_protocol"),
            "skiing-stall-actionfix-v1",
        ),
        "action_mapping_protocol": (
            metrics.get("action_mapping_protocol"),
            "single_canonical_full18",
        ),
    }
    skiing_mismatches = {
        key: value for key, value in skiing_checks.items() if value[0] != value[1]
    }
    if skiing_mismatches:
        raise RuntimeError(f"Unexpected completed Skiing source: {skiing_mismatches}")
    if extension_2m == "true":
        expected_leaf = (
            "atari_dqn_skiing_fs4_stack4_l3_full18_stallactionfix_v1_"
            + (
                f"extend1mto2m_1m_{model}_seed1"
                if model in {"gru", "lstm"}
                else "warmstart19450000_1m_gawf_seed1"
            )
        )
        actual_leaf = os.path.basename(os.path.dirname(os.path.abspath(path)))
        if actual_leaf != expected_leaf:
            raise RuntimeError(
                f"Unexpected cumulative-2M source leaf: {actual_leaf!r}; "
                f"expected {expected_leaf!r}"
            )
        if metrics.get("initialization", {}).get("mode") != "weights_only":
            raise RuntimeError("Cumulative-2M source lacks weights-only provenance")
        if model == "gawf" and metrics.get("extended_from_total_timesteps") != 1_000_000:
            raise RuntimeError("GaWF cumulative-2M source lacks continuous-extension provenance")
elif source_step != 20_000_000 and not (
    allow_incomplete == "true" and source_step > 0
):
    raise RuntimeError(
        f"Source global_step={source_step}; expected 20000000 or explicit "
        "--allow-incomplete-source"
    )
elif set(metrics.get("env_ids", [])) != expected_envs:
    raise RuntimeError(f"Unexpected source task set: {metrics.get('env_ids')}")
PY

TOTAL_TIMESTEPS="${REQUESTED_TOTAL_TIMESTEPS:-1000000}"
BUFFER_SIZE=500000
LEARNING_STARTS=20000
CHECKPOINT_INTERVAL=50000
PHASE_TAG="1m"
LEARNING_RATE=1e-4
LEARNING_RATE_DECAY_STEP=1000000
LEARNING_RATE_DECAY_SCALE=0.1
START_EPSILON=1.0
END_EPSILON=0.01
if [[ "$EXTEND_FROM_SKIING_1M" == true ]]; then
  [[ -z "$REQUESTED_TOTAL_TIMESTEPS" || "$TOTAL_TIMESTEPS" == 1000000 ]] || {
    echo "--extend-from-skiing-1m is a fixed 1M extension phase" >&2
    exit 2
  }
  LEARNING_RATE=1e-5
  LEARNING_RATE_DECAY_STEP=0
  LEARNING_RATE_DECAY_SCALE=1.0
  START_EPSILON=0.01
  PHASE_TAG="extend1mto2m_1m"
fi
if [[ "$EXTEND_FROM_SKIING_2M" == true ]]; then
  [[ -z "$REQUESTED_TOTAL_TIMESTEPS" || "$TOTAL_TIMESTEPS" == 1000000 ]] || {
    echo "--extend-from-skiing-2m has a fixed additional 2M budget" >&2
    exit 2
  }
  TOTAL_TIMESTEPS=2000000
  LEARNING_RATE=1e-5
  LEARNING_RATE_DECAY_STEP=0
  LEARNING_RATE_DECAY_SCALE=1.0
  START_EPSILON=0.01
  PHASE_TAG="extend2mto4m_2m"
fi
if [[ "$PHASE" == "smoke" ]]; then
  TOTAL_TIMESTEPS=25000
  BUFFER_SIZE=25000
  LEARNING_STARTS=2000
  CHECKPOINT_INTERVAL=10000
  PHASE_TAG="smoke25k"
  [[ "$EXTEND_FROM_SKIING_1M" == false ]] || PHASE_TAG="extend1mto2m_smoke25k"
  [[ "$EXTEND_FROM_SKIING_2M" == false ]] || PHASE_TAG="extend2mto4m_smoke25k"
fi

BASE_TAG="atari_dqn_skiing_fs4_stack4_l3_full18_stallactionfix_v1_"
if [[ "$EXTEND_FROM_SKIING_1M" == true || "$EXTEND_FROM_SKIING_2M" == true ]]; then
  BASE_TAG+="${PHASE_TAG}_${MODEL}_seed1"
else
  BASE_TAG+="warmstart20m_${PHASE_TAG}_${MODEL}_seed1"
fi
RUN_TAG="${RUN_TAG:-$BASE_TAG}"
[[ "$RUN_TAG" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "--run-tag contains unsafe characters" >&2
  exit 2
}

RESULT_PARENT="$RESULTS_ROOT/data/rl/atari/5task_18action/formal_20m_4mpertask_raw_seeds"
if [[ -n "$EXPLICIT_RESULT_PARENT" ]]; then
  [[ "$EXPLICIT_RESULT_PARENT" == /* ]] || { echo "Result parent must be absolute" >&2; exit 2; }
  RESULT_PARENT="$EXPLICIT_RESULT_PARENT"
fi
RESULT_DIR="$RESULT_PARENT/$RUN_TAG"
STATUS_DIR="$RESULT_DIR/status"
CHECKPOINT="$RESULT_DIR/checkpoint.pth"
HISTORY="$RESULT_DIR/metrics_history.jsonl"
METRICS="$RESULT_DIR/metrics.json"

RESUME_ARGS=()
if [[ -f "$METRICS" ]]; then
  echo "Refusing to overwrite completed result: $RESULT_DIR" >&2
  exit 3
elif [[ -f "$CHECKPOINT" ]]; then
  RESUME_ARGS=(--resume_from "$CHECKPOINT")
elif [[ -f "$HISTORY" ]]; then
  echo "Refusing existing history without resumable checkpoint: $RESULT_DIR" >&2
  exit 3
else
  RESUME_ARGS=(--init_weights_from "$SOURCE_CHECKPOINT")
fi
if [[ "$ALLOW_TOTAL_TIMESTEPS_EXTENSION" == true ]]; then
  [[ "${RESUME_ARGS[0]}" == --resume_from ]] || {
    echo "--allow-total-timesteps-extension requires an existing checkpoint.pth" >&2
    exit 3
  }
  RESUME_ARGS+=(--allow_total_timesteps_extension)
fi

COMMAND=(
  python run_task.py atari-dqn
  --env_id ALE/Skiing-v5 --action_space_mode full18
  --atari_env_protocol skiing-stall-actionfix-v1
  --model_type "$MODEL" --feedback_mode "$FEEDBACK_MODE"
  --num_layers 3 --hidden_size "$HIDDEN" --gawf_feedback_lr_scale 1.0
  --frame_skip 4 --frame_stack 4 --flicker_prob 0.0
  --total_timesteps "$TOTAL_TIMESTEPS"
  --learning_rate "$LEARNING_RATE" --learning_rate_decay_step "$LEARNING_RATE_DECAY_STEP"
  --learning_rate_decay_per_task_steps 0 --learning_rate_decay_scale "$LEARNING_RATE_DECAY_SCALE"
  --learning_starts "$LEARNING_STARTS" --learning_starts_per_task 0
  --exploration_steps 500000 --start_epsilon "$START_EPSILON" --end_epsilon "$END_EPSILON"
  --replay_backing mmap --buffer_size "$BUFFER_SIZE"
  --checkpoint_interval_steps "$CHECKPOINT_INTERVAL"
  --batch_size 32 --seq_len 16 --sequences_per_batch 8
  --amp_dtype bfloat16 --allow_tf32 --cudnn_benchmark --fused_optimizer
  --seed 1 --device cuda --result_suffix "$RUN_TAG" --save_dir "$RESULT_DIR"
  --gamma "$GAMMA"
)
if (( ${#REWARD_ARGS[@]} > 0 )); then
  COMMAND+=("${REWARD_ARGS[@]}")
fi
COMMAND+=("${RESUME_ARGS[@]}")

if [[ "$DRY_RUN" == true ]]; then
  printf 'RESULT_PARENT=%q\n' "$RESULT_PARENT"
  printf 'RESULT_DIR=%q\n' "$RESULT_DIR"
  printf 'SOURCE_CHECKPOINT=%q\n' "$SOURCE_CHECKPOINT"
  printf 'SOURCE_METRICS=%q\n' "$SOURCE_METRICS"
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$CUDA_DEVICE"
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "$STATUS_DIR"
printf 'status=started phase=%s model=%s result_dir=%s timestamp=%s\n' \
  "$PHASE" "$MODEL" "$RESULT_DIR" "$(date -Is)" > "$STATUS_DIR/started"

set +e
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" DISABLE_TQDM=1 "${COMMAND[@]}" &
TRAIN_PID=$!
trap 'kill -USR1 "$TRAIN_PID" 2>/dev/null || true' USR1 TERM
wait "$TRAIN_PID"
TRAIN_RC=$?
while (( TRAIN_RC >= 128 )) && kill -0 "$TRAIN_PID" 2>/dev/null; do
  wait "$TRAIN_PID"
  TRAIN_RC=$?
done
trap - USR1 TERM
set -e
if (( TRAIN_RC != 0 )); then
  printf 'status=train_failed model=%s exit_code=%s timestamp=%s\n' \
    "$MODEL" "$TRAIN_RC" "$(date -Is)" > "$STATUS_DIR/fail"
  exit "$TRAIN_RC"
fi
if [[ ! -f "$METRICS" ]]; then
  [[ -s "$CHECKPOINT" ]] || { echo "Missing resumable checkpoint" >&2; exit 4; }
  printf 'status=paused checkpoint=%s timestamp=%s\n' "$CHECKPOINT" "$(date -Is)" \
    > "$STATUS_DIR/paused"
  if [[ "$REQUEUE_ON_PAUSE" == true ]]; then
    scontrol requeue "${SLURM_JOB_ID:?Slurm job ID required for requeue}"
  fi
  exit 0
fi

python - "$METRICS" "$TOTAL_TIMESTEPS" "$MODEL" "$HIDDEN" "$GAMMA" "$REWARD_CLIP" <<'PY'
import glob
import json
import math
import os
import sys

path, total_steps, model, hidden, gamma, reward_clip = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    metrics = json.load(handle)
expected = {
    "global_step": int(total_steps),
    "model_type": model,
    "hidden_size": int(hidden),
    "num_layers": 3,
    "env_id": "ALE/Skiing-v5",
    "action_space_mode": "full18",
    "num_actions": 18,
    "atari_env_protocol": "skiing-stall-actionfix-v1",
    "action_mapping_protocol": "single_canonical_full18",
    "stalled_truncation_bootstrap": True,
    "gamma": float(gamma),
    "reward_clip": reward_clip == "true",
}
mismatches = {
    key: (metrics.get(key), value)
    for key, value in expected.items()
    if metrics.get(key) != value
}
if mismatches:
    raise RuntimeError(f"Completed Skiing metrics mismatch: {mismatches}")
if not math.isfinite(float(metrics.get("loss", float("nan")))):
    raise RuntimeError("Final loss is not finite")
if metrics.get("initialization", {}).get("mode") != "weights_only":
    raise RuntimeError("Missing weights-only initialization provenance")
result_dir = os.path.dirname(path)
if not os.path.isfile(os.path.join(result_dir, "metrics_history.jsonl")):
    raise RuntimeError("Missing metrics history")
if len(glob.glob(os.path.join(result_dir, "*.pth"))) != 1:
    raise RuntimeError("Expected exactly one completed model state_dict")
PY

if [[ "$PHASE" == "smoke" && "$SKIP_SMOKE_VIDEO" == false ]]; then
  VIDEO="$RESULT_DIR/smoke_skiing_${MODEL}_seed1.mp4"
  VIDEO_METADATA="$RESULT_DIR/smoke_skiing_${MODEL}_seed1.json"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python -m utils.analysis.rl.atari.evaluate_dqn_video \
    --metrics_path "$METRICS" --output_path "$VIDEO" --metadata_path "$VIDEO_METADATA" \
    --num_episodes 3 --eval_seed 20260826 --device cuda --amp_dtype bfloat16 \
    --video_title "Skiing stall-actionfix smoke | ${MODEL} seed1"
  [[ -s "$VIDEO" && -s "$VIDEO_METADATA" ]] || {
    echo "Smoke evaluation did not produce video and metadata" >&2
    exit 4
  }
fi

printf 'status=done phase=%s model=%s result_dir=%s timestamp=%s\n' \
  "$PHASE" "$MODEL" "$RESULT_DIR" "$(date -Is)" > "$STATUS_DIR/done"
rm -f "$STATUS_DIR/fail" "$STATUS_DIR/paused"
