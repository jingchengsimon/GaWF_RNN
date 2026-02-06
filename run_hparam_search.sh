#!/bin/zsh

# 启用数组支持
setopt KSH_ARRAYS

# 切到脚本所在目录，确保相对路径正确
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo "Working directory: $(pwd)"
echo "Python location: $(which python)"
echo ""

# 激活 conda 环境
# 注意：必须在 nohup 之前激活，并通过 bash -c 传递给后台进程
source /G/anaconda3/etc/profile.d/conda.sh
conda activate aim3_rnn

echo "Activated conda environment: $CONDA_DEFAULT_ENV"
echo ""

# 日志目录（如不需要日志，可按需重定向到 /dev/null）
LOG_DIR="logs_hparam"
mkdir -p "$LOG_DIR"
echo "Log directory: $LOG_DIR"
echo ""



# 运行模式
DRY_RUN=false   # 可通过命令行参数 --dry-run 启用（仅打印，不实际launch）
# 解析简单的命令行开关
if [ "$1" = "--dry-run" ]; then
  DRY_RUN=true
  echo "DRY_RUN mode enabled: will not launch jobs, only print planned allocations"
fi

# ============================================================
# 搜索模式说明
# ============================================================
# 本脚本采用分阶段小规模搜索（Stage A → Stage B → Stage C），默认只运行Stage A（WD sweep）。
# 如需扩大搜索，可修改 WD_GRID / DO_GRID / LR_GRID 变量并重启脚本。
# ============================================================
# （注）旧的 USE_GRID_SEARCH 参数已弃用，使用分阶段逻辑替代。

# 每张 GPU 最多同时跑几个进程（并行时建议 1-2）
MAX_JOBS_PER_GPU=1

# 如果有多张卡，在这里列出可用 GPU
GPUS=(0 1)   # 多卡示例: (0 1 2 3)

# 防止碎片化的可选环境变量（若不需要可注释）
export PYTORCH_ALLOC_CONF=expandable_segments:True

job_id=0
PIDS=()

# 清理函数：在脚本退出或被中断时终止后台任务并清理共享内存
cleanup() {
  echo "Cleaning up: killing child jobs..."
  if [ ${#PIDS[@]} -gt 0 ]; then
    echo "Killing PIDs: ${PIDS[@]}"
    kill "${PIDS[@]}" 2>/dev/null || true
  fi
  # 额外保证：杀死本脚本的所有子进程
  pkill -P $$ 2>/dev/null || true
  # 清除 IPC 共享内存段（如果有 ipcs/ipcrm 工具）
  if command -v ipcs >/dev/null 2>&1; then
    echo "Removing IPC shared memory segments owned by $(whoami)"
    ipcs -m | awk -v user="$(whoami)" '$3==user {print $2}' | xargs -r -n1 ipcrm -m || true
  fi
  echo "Cleanup done."
}

# 在接收到 EXIT/INT/TERM 时调用 cleanup
trap 'cleanup; exit' EXIT INT TERM

get_gpu_running_jobs() {
  local gpu_id="$1"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$gpu_id" 2>/dev/null \
      | awk 'NF{print $1}' | wc -l | tr -d ' '
  else
    pgrep -f "train_rnn_updated.py" | wc -l | tr -d ' '
  fi
}

# ============================================================
# 分阶段小规模超参数搜索（按顺序执行 Stage A → Stage B → Stage C）
# 设计原则：先保守微调正则化（WD），再加入Dropout，最后微调LR；默认只运行Stage A（小规模）
# ============================================================

COMBINATIONS=()

# 基本固定参数（按需修改）
NUM_EPOCHS=200
RESULT_SUFFIX="lr_search_sector"

# 超参搜索表（逐个调优策略）
MODEL_TYPES=("rnn")          # 或加上 "rnn" / "lstm" / "gru"
HIDDEN_SIZES=(256)       # 你要扫的 hidden size

# 默认值（用于固定其他参数）
DEFAULT_LR=1e-3
DEFAULT_WD=0 
DEFAULT_DROP=0 

# --- 小规模试验配置（可按需扩展）
LR_GRID=(1e-3 1e-4)    # 学习率搜索范围          
WD_GRID=()      
DO_GRID=()               
SEEDS=(42)                  # 每个配置的随机种子（小规模先用单一seed）

# 说明给用户
echo "=== 分阶段小规模超参数搜索 ==="
echo "Stage A (LR): ${LR_GRID[@]}"
echo "Stage B (WD sweep): ${WD_GRID[@]:-<disabled>}"
echo "Stage C (DO): ${DO_GRID[@]:-<disabled>}"
echo "Seeds: ${SEEDS[@]}"
echo "Models: ${MODEL_TYPES[@]} | Hidden sizes: ${HIDDEN_SIZES[@]}"
echo ""

# Stage A: learning rate sweep
for model_type in "${MODEL_TYPES[@]}"; do
  for hidden_size in "${HIDDEN_SIZES[@]}"; do
    for lr in "${LR_GRID[@]}"; do
      for seed in "${SEEDS[@]}"; do
        COMBINATIONS+=("$model_type,$hidden_size,$lr,$DEFAULT_WD,$DEFAULT_DROP,stageA_40h_lr${lr},${seed}")
        echo "StageA 40h add: model=$model_type h=$hidden_size lr=$lr seed=$seed"
      done
    done
  done
done

echo ""
echo "总共 ${#COMBINATIONS[@]} 个超参数组合 (分阶段小规模搜索)"
echo ""


# 遍历所有组合
for combo in "${COMBINATIONS[@]}"; do
  # 解析组合参数（包含 stage_label 和 seed）
  # 使用 subshell 读取，避免修改全局 IFS
  (
    IFS=',' read -r model_type hidden_size lr wd drop stage_label seed_field <<< "$combo"
  )

  # 恢复并重新解析为防止特殊字符问题
  model_type=$(echo "$combo" | cut -d',' -f1)
  hidden_size=$(echo "$combo" | cut -d',' -f2)
  lr=$(echo "$combo" | cut -d',' -f3)
  wd=$(echo "$combo" | cut -d',' -f4)
  drop=$(echo "$combo" | cut -d',' -f5)
  stage_label=$(echo "$combo" | cut -d',' -f6)
  seed_field=$(echo "$combo" | cut -d',' -f7)
  # 如果 seed_field 为空，默认使用 1
  seed=${seed_field:-1}
  
  # 轮询分配 GPU（修正为 0-based 索引，避免越界）
  gpu_idx=$(( job_id % ${#GPUS[@]} ))
  gpu="${GPUS[$gpu_idx]}"
  
  job_id=$((job_id + 1))
  
  # 构造日志文件名（根据搜索模式调整）
  if [ "$stage_label" = "default" ] || [ "$stage_label" = "grid" ]; then
    # 全组合模式：使用完整超参数作为文件名（包含 seed）
    LOG_FILE="$LOG_DIR/job${job_id}_${model_type}_h${hidden_size}_lr${lr}_wd${wd}_do${drop}_s${seed}.log"
  else
    # 分段模式：使用 stage 信息（包含 seed）
    LOG_FILE="$LOG_DIR/job${job_id}_${model_type}_h${hidden_size}_${stage_label}_s${seed}.log"
  fi
  
  echo "Launching job $job_id on GPU $gpu [$stage_label]: model=$model_type, h=$hidden_size, lr=$lr, wd=$wd, drop=$drop"
  
  # 验证 GPU 变量不为空；若为空则回退到第一个 GPU 并打印警告
  if [ -z "$gpu" ]; then
    echo "Warning: gpu empty; defaulting to ${GPUS[0]}"
    gpu="${GPUS[0]}"
  fi
  
  # 控制并发数量：当前 GPU 上任务数达到上限则等待
  while true; do
    running_jobs=$(get_gpu_running_jobs "$gpu")
    if [ "$running_jobs" -lt "$MAX_JOBS_PER_GPU" ]; then
      break
    fi
    sleep 10
  done

  # 在这里打印 launch 信息（关键）
  echo "[LAUNCH] job=$job_id gpu=$gpu CUDA_VISIBLE_DEVICES=$gpu"

  # 如果是 dry-run，只打印将要执行的命令，不实际launch
  if [ "$DRY_RUN" = "true" ]; then
    echo "[DRY-RUN] Would launch on GPU $gpu: python train_rnn_updated.py --model_types $model_type --hidden_sizes $hidden_size --lrs $lr --weight_decays $wd --dropout_rates $drop --num_epochs $NUM_EPOCHS --result_suffix $RESULT_SUFFIX --seed ${seed} --use_acceleration True > $LOG_FILE 2>&1"
  else
    (
      export CUDA_VISIBLE_DEVICES="$gpu"

      source /G/anaconda3/etc/profile.d/conda.sh
      conda activate aim3_rnn

      exec python -u train_rnn_updated.py \
        --model_types "$model_type" \
        --hidden_sizes "$hidden_size" \
        --num_epochs "$NUM_EPOCHS" \
        --lrs "$lr" \
        --weight_decays "$wd" \
        --dropout_rates "$drop" \
        --seed "$seed" \
        --use_acceleration \
        --use_sector_mode \
        --result_suffix "$RESULT_SUFFIX" 
    ) > "$LOG_FILE" 2>&1 &
    pid=$!
    PIDS+=("$pid")

    echo "  → Job $job_id PID: $pid, Log: $LOG_FILE"
  fi
  
  # 稍微错开发，避免同时抢资源
  sleep 3
done

echo ""
echo "============================================================"
echo "All jobs launched!"
echo "============================================================"
if [ "$DRY_RUN" = "true" ]; then
  echo "Note: DRY_RUN enabled; no jobs were actually launched."
else
  if [ ${#PIDS[@]} -gt 0 ]; then
    echo "Launched PIDs: ${PIDS[@]}"
    echo "To kill all launched jobs: pkill -P $$   # or use 'kill ${PIDS[@]}'"
  else
    echo "No PIDs recorded. Either no jobs launched or they finished quickly."
  fi
fi

echo "Monitor with: ps aux | grep train_rnn_updated.py"
echo "Check logs: ls -lart logs_hparam/"
echo "View log: tail -f logs_hparam/job*.log"
echo ""


