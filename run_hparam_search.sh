#!/bin/zsh
# 或 #!/bin/bash 也可以

# ============================================================
# 逐个调优超参数搜索脚本
# ============================================================
# 策略：固定其他参数，逐个调优 lr, wd, dropout
# 
# Stage 1: 调优 learning rate (固定 wd 和 drop)
# Stage 2: 调优 weight decay (固定 lr 和 drop)
# Stage 3: 调优 dropout (固定 lr 和 wd)
#
# 使用方法：
# 1. 运行脚本，完成 Stage 1-3 的所有实验
# 2. 查看结果，找出每个 stage 的最佳值
# 3. 可选：更新 DEFAULT_* 值，重新运行以细化搜索
# ============================================================

# 切到脚本所在目录，确保相对路径正确
# SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# cd "$SCRIPT_DIR" || exit 1

# 激活 conda 环境
# source /G/anaconda3/etc/profile.d/conda.sh
# conda activate aim3_rnn

# 日志目录（如不需要日志，可按需重定向到 /dev/null）
LOG_DIR="logs_hparam"
mkdir -p "$LOG_DIR"

# 基本固定参数（按需修改）
NUM_EPOCHS=200
RESULT_SUFFIX="default_sector"

# 超参搜索表（逐个调优策略）
MODEL_TYPES=("rnn")          # 或加上 "rnn" / "lstm" / "gru"
HIDDEN_SIZES=(128 256)       # 你要扫的 hidden size

# 定义搜索范围
LRS=(0.001) #(0.0008 0.001 0.0012)        # learning rates
WDS=(0.0001) #(0.00008 0.0001 0.00012)     # weight decays
DROPS=(0.3) #(0.25 0.3 0.35)            # dropout rates

# 默认值（用于固定其他参数）
DEFAULT_LR=0.001
DEFAULT_WD=0.0001
DEFAULT_DROP=0.3

# 每张 GPU 最多同时跑几个进程（并行时建议 1-2）
MAX_JOBS_PER_GPU=1

# 如果有多张卡，在这里列出可用 GPU
GPUS=(0 1)   # 多卡示例: (0 1 2 3)

# 防止碎片化的可选环境变量（若不需要可注释）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

job_id=0
PIDS=()

get_gpu_running_jobs() {
  local gpu_id="$1"
  if command -v nvidia-smi > /dev/null 2>&1; then
    nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$gpu_id" 2>/dev/null | wc -l | tr -d ' '
  else
    # fallback: count all training processes
    pgrep -f "train_rnn_updated.py" | wc -l | tr -d ' '
  fi
}

# 逐个调优策略：生成超参数组合
# Stage 1: 固定 wd 和 drop，只调 lr
# Stage 2: 固定 lr 和 drop，只调 wd
# Stage 3: 固定 lr 和 wd，只调 drop

COMBINATIONS=()

echo "=== 逐个调优策略 ==="
echo "Stage 1: 调优 learning rate (固定 wd=$DEFAULT_WD, drop=$DEFAULT_DROP)"
echo "Stage 2: 调优 weight decay (固定 lr=$DEFAULT_LR, drop=$DEFAULT_DROP)"
echo "Stage 3: 调优 dropout (固定 lr=$DEFAULT_LR, wd=$DEFAULT_WD)"
echo ""

for model_type in "${MODEL_TYPES[@]}"; do
  for hidden_size in "${HIDDEN_SIZES[@]}"; do
    # Stage 1: 只调 lr，固定 wd 和 drop
    echo "Stage 1 - Model: $model_type, Hidden: $hidden_size"
    for lr in "${LRS[@]}"; do
      COMBINATIONS+=("$model_type,$hidden_size,$lr,$DEFAULT_WD,$DEFAULT_DROP,stage1_lr${lr}")
      echo "  添加: lr=$lr, wd=$DEFAULT_WD, drop=$DEFAULT_DROP"
    done
    
    # Stage 2: 只调 wd，固定 lr 和 drop
    echo "Stage 2 - Model: $model_type, Hidden: $hidden_size"
    for wd in "${WDS[@]}"; do
      COMBINATIONS+=("$model_type,$hidden_size,$DEFAULT_LR,$wd,$DEFAULT_DROP,stage2_wd${wd}")
      echo "  添加: lr=$DEFAULT_LR, wd=$wd, drop=$DEFAULT_DROP"
    done
    
    # Stage 3: 只调 drop，固定 lr 和 wd
    echo "Stage 3 - Model: $model_type, Hidden: $hidden_size"
    for drop in "${DROPS[@]}"; do
      COMBINATIONS+=("$model_type,$hidden_size,$DEFAULT_LR,$DEFAULT_WD,$drop,stage3_drop${drop}")
      echo "  添加: lr=$DEFAULT_LR, wd=$DEFAULT_WD, drop=$drop"
    done
  done
done

echo ""
echo "总共 ${#COMBINATIONS[@]} 个超参数组合 (逐个调优策略)"
echo "预计实验数: ${#MODEL_TYPES[@]} models × ${#HIDDEN_SIZES[@]} hidden_sizes × (${#LRS[@]} + ${#WDS[@]} + ${#DROPS[@]}) params"
echo ""

# 遍历所有组合
for combo in "${COMBINATIONS[@]}"; do
  # 解析组合参数（包含 stage_label）
  IFS=',' read -r model_type hidden_size lr wd drop stage_label <<< "$combo"
  
  # 轮询分配 GPU
  gpu_idx=$(( (job_id % ${#GPUS[@]}) + 1 ))
  gpu="${GPUS[$gpu_idx]}"
  
  job_id=$((job_id + 1))
  
  # 构造日志文件名（包含 stage 信息）
  LOG_FILE="$LOG_DIR/job${job_id}_${model_type}_h${hidden_size}_${stage_label}.log"
  
  echo "Launching job $job_id on GPU $gpu [$stage_label]: model=$model_type, h=$hidden_size, lr=$lr, wd=$wd, drop=$drop"
  
  # 验证 GPU 变量不为空
  if [ -z "$gpu" ]; then
    echo "ERROR: GPU variable is empty! Check GPUS array indexing."
    continue
  fi
  
  # 控制并发数量：当前 GPU 上任务数达到上限则等待
  while true; do
    running_jobs=$(get_gpu_running_jobs "$gpu")
    if [ "$running_jobs" -lt "$MAX_JOBS_PER_GPU" ]; then
      break
    fi
    sleep 10
  done
  
  CUDA_VISIBLE_DEVICES=$gpu nohup python train_rnn_updated.py \
    --model_types "$model_type" \
    --hidden_sizes "$hidden_size" \
    --lrs "$lr" \
    --weight_decays "$wd" \
    --dropout_rates "$drop" \
    --num_epochs "$NUM_EPOCHS" \
    --result_suffix "$RESULT_SUFFIX" \
    --use_sector_mode \
    > "$LOG_FILE" 2>&1 &
  pid=$!
  PIDS+=("$pid")
  echo "  → Job $job_id PID: $pid"
  
  # 稍微错开发，避免同时抢资源
  sleep 3
done

echo "All jobs launched. Use 'ps | grep train_rnn_updated.py' 查看运行状态。"

# 若希望脚本等待所有任务完成，取消注释以下三行
# for pid in "${PIDS[@]}"; do
#   wait "$pid"
# done
