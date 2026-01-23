#!/bin/zsh
# 或 #!/bin/bash 也可以

# ============================================================
# 超参数搜索脚本 (支持两种搜索策略)
# ============================================================
#
# 【策略 1】全组合搜索 (Grid Search)
#   - 设置 USE_GRID_SEARCH=true
#   - 测试所有 LRS × WDS × DROPS 的组合
#   - 适合：全面探索超参数空间
#   - 实验数：|LRS| × |WDS| × |DROPS| × |MODEL_TYPES| × |HIDDEN_SIZES|
#   - 示例：3个lr × 3个wd × 3个drop = 27种组合/模型
#
# 【策略 2】分段搜索 (Sequential Tuning)
#   - 设置 USE_GRID_SEARCH=false
#   - Stage 1: 调优 lr (固定 wd, drop)
#   - Stage 2: 调优 wd (固定 lr, drop)
#   - Stage 3: 调优 drop (固定 lr, wd)
#   - 适合：快速找到较优配置
#   - 实验数：(|LRS| + |WDS| + |DROPS|) × |MODEL_TYPES| × |HIDDEN_SIZES|
#   - 示例：3个lr + 3个wd + 3个drop = 9种组合/模型
#
# 使用示例：
#
# 【快速测试单一配置】
#   LRS=(0.001)
#   WDS=(0.0001)
#   DROPS=(0.3)
#   USE_GRID_SEARCH=false
#   → 结果：只训练 2 次 (h128, h256)，不重复
#
# 【分段搜索】
#   LRS=(0.0008 0.001 0.0012)
#   WDS=(0.00008 0.0001 0.00012)
#   DROPS=(0.25 0.3 0.35)
#   USE_GRID_SEARCH=false
#   → 结果：9 × 2 = 18 次训练
#
# 【全组合搜索】
#   LRS=(0.0008 0.001 0.0012)
#   WDS=(0.00008 0.0001 0.00012)
#   DROPS=(0.25 0.3 0.35)
#   USE_GRID_SEARCH=true
#   → 结果：27 × 2 = 54 次训练
#
# ============================================================

# 切到脚本所在目录，确保相对路径正确
# SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# cd "$SCRIPT_DIR" || exit 1

# 激活 conda 环境
# source /G/anaconda3/etc/profile.d/conda.sh
# conda activate aim3_rnn

# 日志目录（如不需要日志，可按需重定向到 /dev/null）
LOG_DIR="logs_hparam"
# mkdir -p "$LOG_DIR"

# 基本固定参数（按需修改）
NUM_EPOCHS=200
RESULT_SUFFIX="default_sector"

# 超参搜索表（逐个调优策略）
MODEL_TYPES=("rnn")          # 或加上 "rnn" / "lstm" / "gru"
HIDDEN_SIZES=(128 256)       # 你要扫的 hidden size

# 定义搜索范围
LRS=(0.001) #(0.0008 0.001 0.0012)        # learning rates
WDS=(0.0001) #(0.00008 0.0001 0.00012)     # weight decays
DROPS=(0) #(0.25 0.3 0.35)            # dropout rates

# 默认值（用于固定其他参数，仅在分段搜索时使用）
DEFAULT_LR=0.001
DEFAULT_WD=0.0001
DEFAULT_DROP=0.3

# ============================================================
# 超参数搜索策略选择
# ============================================================
# USE_GRID_SEARCH=true:  全组合搜索 (Grid Search)
#   - 生成 LRS × WDS × DROPS 的全部组合
#   - 适合全面探索超参数空间
#   - 实验数量：|LRS| × |WDS| × |DROPS| × |MODEL_TYPES| × |HIDDEN_SIZES|
#
# USE_GRID_SEARCH=false: 分段搜索 (Sequential Tuning)
#   - Stage 1: 只调 lr，固定 wd 和 drop
#   - Stage 2: 只调 wd，固定 lr 和 drop
#   - Stage 3: 只调 drop，固定 lr 和 wd
#   - 适合快速找到较优配置
#   - 实验数量：(|LRS| + |WDS| + |DROPS|) × |MODEL_TYPES| × |HIDDEN_SIZES|
# ============================================================
USE_GRID_SEARCH=true  # 改为 true 启用全组合搜索

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
# 生成超参数组合
# ============================================================

COMBINATIONS=()

if [ "$USE_GRID_SEARCH" = true ]; then
  # ========== 全组合搜索模式 ==========
  echo "=== 全组合搜索 (Grid Search) ==="
  echo "搜索空间: LRS × WDS × DROPS"
  echo "  LRS: ${LRS[@]}"
  echo "  WDS: ${WDS[@]}"
  echo "  DROPS: ${DROPS[@]}"
  echo ""
  
  # 检查是否所有数组长度都为1（避免重复）
  if [ ${#LRS[@]} -eq 1 ] && [ ${#WDS[@]} -eq 1 ] && [ ${#DROPS[@]} -eq 1 ]; then
    echo "检测到所有超参数数组长度为1，只生成一次组合（避免重复）"
    echo ""
    for model_type in "${MODEL_TYPES[@]}"; do
      for hidden_size in "${HIDDEN_SIZES[@]}"; do
        lr="${LRS[0]}"
        wd="${WDS[0]}"
        drop="${DROPS[0]}"
        COMBINATIONS+=("$model_type,$hidden_size,$lr,$wd,$drop,default")
        echo "添加: model=$model_type, h=$hidden_size, lr=$lr, wd=$wd, drop=$drop"
      done
    done
  else
    # 正常的全组合搜索
    for model_type in "${MODEL_TYPES[@]}"; do
      for hidden_size in "${HIDDEN_SIZES[@]}"; do
        for lr in "${LRS[@]}"; do
          for wd in "${WDS[@]}"; do
            for drop in "${DROPS[@]}"; do
              COMBINATIONS+=("$model_type,$hidden_size,$lr,$wd,$drop,grid")
              echo "添加: model=$model_type, h=$hidden_size, lr=$lr, wd=$wd, drop=$drop"
            done
          done
        done
      done
    done
  fi
  
  echo ""
  echo "总共 ${#COMBINATIONS[@]} 个超参数组合 (全组合搜索)"
  expected_count=$((${#MODEL_TYPES[@]} * ${#HIDDEN_SIZES[@]} * ${#LRS[@]} * ${#WDS[@]} * ${#DROPS[@]}))
  echo "预计实验数: ${#MODEL_TYPES[@]} models × ${#HIDDEN_SIZES[@]} hidden_sizes × ${#LRS[@]} lrs × ${#WDS[@]} wds × ${#DROPS[@]} drops = $expected_count"
  echo ""
  
else
  # ========== 分段搜索模式 ==========
  echo "=== 分段搜索 (Sequential Tuning) ==="
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
  echo "总共 ${#COMBINATIONS[@]} 个超参数组合 (分段搜索)"
  expected_count=$((${#MODEL_TYPES[@]} * ${#HIDDEN_SIZES[@]} * (${#LRS[@]} + ${#WDS[@]} + ${#DROPS[@]})))
  echo "预计实验数: ${#MODEL_TYPES[@]} models × ${#HIDDEN_SIZES[@]} hidden_sizes × (${#LRS[@]} + ${#WDS[@]} + ${#DROPS[@]}) params = $expected_count"
  echo ""
fi

# 遍历所有组合
for combo in "${COMBINATIONS[@]}"; do
  # 解析组合参数（包含 stage_label）
  IFS=',' read -r model_type hidden_size lr wd drop stage_label <<< "$combo"
  
  # 轮询分配 GPU
  gpu_idx=$(( (job_id % ${#GPUS[@]}) + 1 ))
  gpu="${GPUS[$gpu_idx]}"
  
  job_id=$((job_id + 1))
  
  # 构造日志文件名（根据搜索模式调整）
  if [ "$stage_label" = "default" ] || [ "$stage_label" = "grid" ]; then
    # 全组合模式：使用完整超参数作为文件名
    LOG_FILE="$LOG_DIR/job${job_id}_${model_type}_h${hidden_size}_lr${lr}_wd${wd}_do${drop}.log"
  else
    # 分段模式：使用 stage 信息
    LOG_FILE="$LOG_DIR/job${job_id}_${model_type}_h${hidden_size}_${stage_label}.log"
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
