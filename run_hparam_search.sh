#!/bin/zsh
# 或 #!/bin/bash 也可以

# 切到工程目录
# cd /Users/jingchengshi/Desktop/Vscode/FAW_RNN || exit 1

# 日志目录
LOG_DIR="logs_hparam"
mkdir -p "$LOG_DIR"

# 基本固定参数（按需修改）
NUM_EPOCHS=200
RESULT_SUFFIX="hparam_search"
MODEL_TYPES=("rnn")          # 或加上 "rnn"
HIDDEN_SIZES=(128 256)              # 你要扫的 hidden size

# 超参搜索表（在这里控制哪些组合用“外层并行”）
LRS=(0.0003 0.0004) # learning rates
WDS=(0.0003 0.001) # weight decays
DROPS=(0.5 0.6) # dropout rates

# 每张 GPU 最多同时跑几个进程（单卡情况可以当作总并发上限）
MAX_JOBS_PER_GPU=1

# 如果有多张卡，在这里列出可用 GPU
GPUS=(1)   # 多卡示例: (0 1 2 3)

job_id=0

for lr in "${LRS[@]}"; do
  for wd in "${WDS[@]}"; do
    for drop in "${DROPS[@]}"; do
      # 轮询分配 GPU
      gpu_idx=$(( job_id % ${#GPUS[@]} ))
      gpu="${GPUS[$gpu_idx]}"

      # 控制并发数量：如果当前后台 job 数 >= 总卡数 * 每卡上限，则等待
      while true; do
        running_jobs=$(jobs -rp | wc -l | tr -d ' ')
        max_jobs=$(( ${#GPUS[@]} * MAX_JOBS_PER_GPU ))
        if [ "$running_jobs" -lt "$max_jobs" ]; then
          break
        fi
        sleep 5
      done

      job_id=$((job_id + 1))

      # 构造日志文件名（方便之后查对应的曲线）
      LOG_FILE="$LOG_DIR/job${job_id}_lr${lr}_wd${wd}_do${drop}.log"

      echo "Launching job $job_id on GPU $gpu: lr=$lr, wd=$wd, drop=$drop"
      CUDA_VISIBLE_DEVICES=$gpu nohup python train_rnn_updated.py \
        --model_types "${MODEL_TYPES[@]}" \
        --hidden_sizes "${HIDDEN_SIZES[@]}" \
        --lrs "$lr" \
        --weight_decays "$wd" \
        --dropout_rates "$drop" \
        --num_epochs "$NUM_EPOCHS" \
        --result_suffix "$RESULT_SUFFIX" \
        > "$LOG_FILE" 2>&1 &

      # 稍微错开发，避免同时抢资源
      sleep 3
    done
  done
done

echo "All jobs launched. Use 'jobs -l' 或 'ps | grep train_rnn_updated.py' 查看运行状态。"