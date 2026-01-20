#!/bin/zsh
# 或 #!/bin/bash 也可以

# 切到脚本所在目录，确保相对路径正确
# SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# cd "$SCRIPT_DIR" || exit 1

# 日志目录（如不需要日志，可按需重定向到 /dev/null）
LOG_DIR="logs_hparam"
mkdir -p "$LOG_DIR"

# 基本固定参数（按需修改）
NUM_EPOCHS=200
RESULT_SUFFIX="hparam_search"
MODEL_TYPES=("rnn")          # 或加上 "rnn" / "lstm" / "gru"
HIDDEN_SIZES=(128 256)       # 你要扫的 hidden size

# 超参搜索表（在这里控制哪些组合用“外层并行”）
LRS=(0.0003 0.0004) # learning rates
WDS=(0.0003 0.001) # weight decays
DROPS=(0.5 0.6) # dropout rates

# 如果有多张卡，在这里列出可用 GPU；当前仅用 GPU1，避免占用 GPU0 的大任务
GPUS=(1)   # 多卡示例: (0 1 2 3)

# 防止碎片化的可选环境变量（若不需要可注释）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

job_id=0

for lr in "${LRS[@]}"; do
  for wd in "${WDS[@]}"; do
    for drop in "${DROPS[@]}"; do
      # 轮询分配 GPU
      # zsh 数组索引从 1 开始，所以需要 +1
      gpu_idx=$(( (job_id % ${#GPUS[@]}) + 1 ))
      gpu="${GPUS[$gpu_idx]}"

      job_id=$((job_id + 1))

      # 构造日志文件名（方便之后查对应的曲线）
      LOG_FILE="$LOG_DIR/job${job_id}_lr${lr}_wd${wd}_do${drop}.log"

      echo "Launching job $job_id on GPU $gpu: lr=$lr, wd=$wd, drop=$drop"
      # 验证 GPU 变量不为空
      if [ -z "$gpu" ]; then
        echo "ERROR: GPU variable is empty! Check GPUS array indexing."
        continue
      fi
      
      CUDA_VISIBLE_DEVICES=$gpu python train_rnn_updated.py \
        --model_types "${MODEL_TYPES[@]}" \
        --hidden_sizes "${HIDDEN_SIZES[@]}" \
        --lrs "$lr" \
        --weight_decays "$wd" \
        --dropout_rates "$drop" \
        --num_epochs "$NUM_EPOCHS" \
        --result_suffix "$RESULT_SUFFIX" \
        > "$LOG_FILE" 2>&1
      exit_code=$?
      echo "  → Job $job_id exit code: $exit_code"
      if [ "$exit_code" -ne 0 ]; then
        echo "ERROR: Job $job_id failed, stopping the script."
        exit "$exit_code"
      fi

      # 稍微错开发，避免同时抢资源
      sleep 3
    done
  done
done

echo "All jobs launched. Use 'jobs -l' 或 'ps | grep train_rnn_updated.py' 查看运行状态。"