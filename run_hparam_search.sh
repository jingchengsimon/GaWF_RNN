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

# 基本固定参数（按需修改）
NUM_EPOCHS=200
RESULT_SUFFIX="default_sector"

# 超参搜索表（逐个调优策略）
MODEL_TYPES=("rnn")          # 或加上 "rnn" / "lstm" / "gru"
HIDDEN_SIZES=(128 256)       # 你要扫的 hidden size

# 定义搜索范围
LRS=(0.001) #(0.0008 0.001 0.0012)        # learning rates
WDS=(0) #(0.00008 0.0001 0.00012)     # weight decays
DROPS=(0) #(0.25 0.3 0.35)            # dropout rates

# 默认值（用于固定其他参数，仅在分段搜索时使用）
DEFAULT_LR=0.001
DEFAULT_WD=0.0001
DEFAULT_DROP=0.3

# ============================================================
# 搜索模式说明
# ============================================================
# 本脚本采用分阶段小规模搜索（Stage A → Stage B → Stage C），默认只运行Stage A（WD sweep）。
# 如需扩大搜索，可修改 WD_GRID / DROPOUT_GRID_STAGEB / LR_GRID_STAGEC 变量并重启脚本。
# ============================================================
# （注）旧的 USE_GRID_SEARCH 参数已弃用，使用分阶段逻辑替代。

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

# ============================================================
# 分阶段小规模超参数搜索（按顺序执行 Stage A → Stage B → Stage C）
# 设计原则：先保守微调正则化（WD），再加入Dropout，最后微调LR；默认只运行Stage A（小规模）
# ============================================================

COMBINATIONS=()

# --- 小规模试验配置（可按需扩展）
WD_GRID=(0 1e-6 1e-5 1e-4)             # Stage A: weight decay 网格（从弱到强）
DROPOUT_GRID_STAGEB=()                # Stage B: dropout 网格（默认空，表示不运行Stage B）
LR_GRID_STAGEC=()                     # Stage C: lr 网格（默认空，表示不运行Stage C）
SEEDS=(42)                             # 每个配置的随机种子（小规模先用单一seed）

# 说明给用户
echo "=== 分阶段小规模超参数搜索 ==="
echo "Stage A (WD sweep): ${WD_GRID[@]}"
echo "Stage B (Dropout): ${DROPOUT_GRID_STAGEB[@]:-<disabled>}"
echo "Stage C (LR): ${LR_GRID_STAGEC[@]:-<disabled>}"
echo "Seeds: ${SEEDS[@]}"
echo "Models: ${MODEL_TYPES[@]} | Hidden sizes: ${HIDDEN_SIZES[@]}"
echo ""

# Stage A: weight decay sweep (dropout fixed 0, lr = DEFAULT_LR)
for model_type in "${MODEL_TYPES[@]}"; do
  for hidden_size in "${HIDDEN_SIZES[@]}"; do
    for wd in "${WD_GRID[@]}"; do
      for seed in "${SEEDS[@]}"; do
        COMBINATIONS+=("$model_type,$hidden_size,$DEFAULT_LR,$wd,0,stageA_wd${wd}_s${seed},${seed}")
        echo "StageA add: model=$model_type h=$hidden_size wd=$wd seed=$seed"
      done
    done
  done
done

# Stage B: dropout sweep (if DROPOUT_GRID_STAGEB not empty)
for model_type in "${MODEL_TYPES[@]}"; do
  for hidden_size in "${HIDDEN_SIZES[@]}"; do
    for wd in "${WD_GRID[@]}"; do
      for drop in "${DROPOUT_GRID_STAGEB[@]}"; do
        for seed in "${SEEDS[@]}"; do
          COMBINATIONS+=("$model_type,$hidden_size,$DEFAULT_LR,$wd,$drop,stageB_wd${wd}_do${drop}_s${seed},${seed}")
          echo "StageB add: model=$model_type h=$hidden_size wd=$wd drop=$drop seed=$seed"
        done
      done
    done
  done
done

# Stage C: lr sweep (if LR_GRID_STAGEC not empty)
for model_type in "${MODEL_TYPES[@]}"; do
  for hidden_size in "${HIDDEN_SIZES[@]}"; do
    for lr in "${LR_GRID_STAGEC[@]}"; do
      for wd in "${WD_GRID[@]}"; do
        for seed in "${SEEDS[@]}"; do
          COMBINATIONS+=("$model_type,$hidden_size,$lr,$wd,0,stageC_lr${lr}_wd${wd}_s${seed},${seed}")
          echo "StageC add: model=$model_type h=$hidden_size lr=$lr wd=$wd seed=$seed"
        done
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
  
  # 轮询分配 GPU
  gpu_idx=$(( (job_id % ${#GPUS[@]}) + 1 ))
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
  
  # 重要：通过 bash -c 执行，确保后台进程继承 conda 环境
  # 注意：使用双引号确保变量被展开
  CUDA_VISIBLE_DEVICES=$gpu nohup bash -c "
    source /G/anaconda3/etc/profile.d/conda.sh
    conda activate aim3_rnn
    python train_rnn_updated.py \
      --model_types $model_type \
      --hidden_sizes $hidden_size \
      --lrs $lr \
      --weight_decays $wd \
      --dropout_rates $drop \
      --num_epochs $NUM_EPOCHS \
      --result_suffix $RESULT_SUFFIX \
      --use_sector_mode \
      --seed ${seed} \
      --use_acceleration True
  " > "$LOG_FILE" 2>&1 &
  pid=$!
  PIDS+=("$pid")
  echo "  → Job $job_id PID: $pid, Log: $LOG_FILE"
  
  # 稍微错开发，避免同时抢资源
  sleep 3
done

echo ""
echo "============================================================"
echo "All jobs launched!"
echo "============================================================"
echo "Monitor with: ps aux | grep train_rnn_updated.py"
echo "Check logs: ls -lart logs_hparam/"
echo "View log: tail -f logs_hparam/job*.log"
echo ""


