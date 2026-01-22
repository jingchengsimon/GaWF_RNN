# 逐个调优超参数搜索指南

## 策略说明

采用**逐个调优**策略，而不是全网格搜索（Grid Search），大幅减少实验次数。

### 实验设计

| Stage | 调优参数 | 固定参数 | 实验次数 |
|-------|---------|---------|---------|
| Stage 1 | Learning Rate | wd=0.0001, drop=0.3 | 3 × 2 = 6 |
| Stage 2 | Weight Decay | lr=0.001, drop=0.3 | 3 × 2 = 6 |
| Stage 3 | Dropout | lr=0.001, wd=0.0001 | 3 × 2 = 6 |
| **总计** | - | - | **18 个实验** |

对比全网格搜索：3 × 3 × 3 × 2 = 54 个实验，节省了 **67%** 的时间！

---

## 搜索空间

```bash
# 定义搜索范围
LRS=(0.0008 0.001 0.0012)        # 3个值
WDS=(0.00008 0.0001 0.00012)     # 3个值
DROPS=(0.25 0.3 0.35)            # 3个值
HIDDEN_SIZES=(128 256)           # 2个值

# 默认值（用于固定其他参数）
DEFAULT_LR=0.001
DEFAULT_WD=0.0001
DEFAULT_DROP=0.3
```

---

## 使用步骤

### 1. 运行脚本

```bash
cd /path/to/FAW_RNN
./run_hparam_search.sh
```

脚本会自动：
- **Stage 1**: 测试 3 个 lr 值（固定 wd=0.0001, drop=0.3）
- **Stage 2**: 测试 3 个 wd 值（固定 lr=0.001, drop=0.3）
- **Stage 3**: 测试 3 个 dropout 值（固定 lr=0.001, wd=0.0001）

每个 stage 对 2 个 hidden_size（128, 256）各执行一次。

---

### 2. 查看结果

结果保存在 `results/hparam_search_2/` 目录：

```bash
# 查看所有结果文件
ls -lh results/hparam_search_2/

# 文件命名格式：
# rnn_coord_acc_h128_lr0.001_wd0.0001_do0.3_hparam_search_2.pkl
```

---

### 3. 分析最佳超参数

使用 notebook 分析：

```python
import pickle
import glob
from pathlib import Path

# 读取所有结果
results_dir = Path("results/hparam_search_2/")
all_results = {}

for pkl_file in results_dir.glob("*.pkl"):
    with open(pkl_file, 'rb') as f:
        results = pickle.load(f)
        key = pkl_file.stem  # 文件名（不含扩展名）
        all_results[key] = results['val_acc_char'][-1]  # 最后一个epoch的验证准确率

# 按 stage 分组
stage1_results = {k: v for k, v in all_results.items() if 'stage1' in k}
stage2_results = {k: v for k, v in all_results.items() if 'stage2' in k}
stage3_results = {k: v for k, v in all_results.items() if 'stage3' in k}

# 找出最佳值
print("Stage 1 (LR):")
best_lr = max(stage1_results, key=stage1_results.get)
print(f"  最佳: {best_lr}, val_acc: {stage1_results[best_lr]:.2f}%")

print("\nStage 2 (WD):")
best_wd = max(stage2_results, key=stage2_results.get)
print(f"  最佳: {best_wd}, val_acc: {stage2_results[best_wd]:.2f}%")

print("\nStage 3 (Dropout):")
best_drop = max(stage3_results, key=stage3_results.get)
print(f"  最佳: {best_drop}, val_acc: {stage3_results[best_drop]:.2f}%")
```

---

### 4. （可选）细化搜索

如果 Stage 1 发现 `lr=0.0012` 最好，可以在此基础上细化：

修改 `run_hparam_search.sh`：

```bash
# 在 0.0012 附近细搜
LRS=(0.0010 0.0012 0.0014 0.0016)
DEFAULT_LR=0.0012  # 更新默认值

# 更新结果目录，避免覆盖
RESULT_SUFFIX="hparam_search_3_refined"
```

---

## 并行执行

脚本自动利用两张 GPU 并行执行：

```bash
MAX_JOBS_PER_GPU=1    # 每张卡同时跑 1 个任务
GPUS=(0 1)            # 使用 GPU 0 和 GPU 1
```

**预计总时间**：
- 18 个实验 ÷ 2 卡 = 9 轮
- 每轮约 1 小时（200 epochs）
- **总计约 9 小时**

---

## 日志文件

每个实验的日志保存在 `logs_hparam/` 目录：

```bash
# 查看某个实验的日志
tail -f logs_hparam/job1_rnn_h128_stage1_lr0.0008.log

# 查看所有正在运行的任务
ps aux | grep train_rnn_updated.py

# 监控 GPU 使用情况
watch -n 1 nvidia-smi
```

---

## 注意事项

1. **默认值选择**：
   - 当前默认值 `lr=0.001, wd=0.0001, drop=0.3` 基于你之前的训练经验
   - 如果这些值在 Stage 1-3 中都不是最优，说明搜索范围可能需要调整

2. **hidden_size**：
   - 当前对每个 hidden_size 都执行完整的 3 个 stage
   - 如果发现 `h=128` 和 `h=256` 的最优超参数差异很大，可能需要分别调优

3. **交互作用**：
   - 这种逐个调优方法**无法捕捉参数间的交互作用**
   - 例如：最佳 lr 可能在不同的 wd 下有所不同
   - 如果发现结果不理想，可以考虑用 Bayesian Optimization

---

## 下一步建议

完成逐个调优后，如果想进一步提升：

1. **组合最佳值测试**：
   - 取 Stage 1-3 的最佳值组合
   - 单独训练一次验证性能

2. **局部网格搜索**：
   - 在最佳值附近进行小范围的网格搜索
   - 例如：`best_lr ± 0.0002`，`best_wd ± 0.00002`

3. **Bayesian Optimization**：
   - 如果预算充足，可以用 Optuna 进行更智能的搜索
   - 参考之前提供的 Optuna 代码示例

---

## 问题排查

**Q: 脚本启动后立即退出？**
```bash
# 检查是否激活了正确的 conda 环境
conda activate aim3_rnn

# 查看日志文件确认错误
cat logs_hparam/job1_*.log
```

**Q: GPU 显存不足？**
```bash
# 减少并行任务数
MAX_JOBS_PER_GPU=1  # 或改为 1

# 或者在 train_rnn_updated.py 中减少 batch_size
```

**Q: 如何停止所有训练任务？**
```bash
# 查找所有 python 训练进程
ps aux | grep train_rnn_updated.py

# 杀死所有相关进程
pkill -f train_rnn_updated.py
```

