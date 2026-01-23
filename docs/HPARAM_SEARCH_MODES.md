# 超参数搜索模式说明

## 脚本优化

`run_hparam_search.sh` 现在支持两种超参数组合生成策略，通过 `USE_GRID_SEARCH` 变量控制。

## 两种搜索模式

### 1. 全组合搜索 (Grid Search)

**配置：**
```bash
USE_GRID_SEARCH=true
```

**行为：**
- 生成所有 `LRS × WDS × DROPS` 的笛卡尔积
- 每种组合都会被测试

**适用场景：**
- 全面探索超参数空间
- 不确定哪个参数最重要
- 有充足的计算资源

**实验数量：**
```
总训练次数 = |MODEL_TYPES| × |HIDDEN_SIZES| × |LRS| × |WDS| × |DROPS|
```

**示例：**
```bash
MODEL_TYPES=("rnn")
HIDDEN_SIZES=(128 256)
LRS=(0.0008 0.001 0.0012)
WDS=(0.00008 0.0001 0.00012)
DROPS=(0.25 0.3 0.35)
USE_GRID_SEARCH=true

# 结果：1 × 2 × 3 × 3 × 3 = 54 次训练
```

### 2. 分段搜索 (Sequential Tuning)

**配置：**
```bash
USE_GRID_SEARCH=false
```

**行为：**
- **Stage 1**: 调优 lr，固定 wd=DEFAULT_WD, drop=DEFAULT_DROP
- **Stage 2**: 调优 wd，固定 lr=DEFAULT_LR, drop=DEFAULT_DROP
- **Stage 3**: 调优 drop，固定 lr=DEFAULT_LR, wd=DEFAULT_WD

**适用场景：**
- 快速找到较优配置
- 计算资源有限
- 了解参数的大致影响范围

**实验数量：**
```
总训练次数 = |MODEL_TYPES| × |HIDDEN_SIZES| × (|LRS| + |WDS| + |DROPS|)
```

**示例：**
```bash
MODEL_TYPES=("rnn")
HIDDEN_SIZES=(128 256)
LRS=(0.0008 0.001 0.0012)
WDS=(0.00008 0.0001 0.00012)
DROPS=(0.25 0.3 0.35)
USE_GRID_SEARCH=false

DEFAULT_LR=0.001
DEFAULT_WD=0.0001
DEFAULT_DROP=0.3

# 结果：1 × 2 × (3 + 3 + 3) = 18 次训练
```

## 特殊优化：单一配置检测

当所有超参数数组长度都为 1 且使用全组合模式时，脚本会自动检测并**只生成一次组合**，避免重复训练。

**示例：**
```bash
LRS=(0.001)          # 长度 = 1
WDS=(0.0001)         # 长度 = 1
DROPS=(0.3)          # 长度 = 1
USE_GRID_SEARCH=true

# 检测到单一配置
# 结果：1 × 2 × 1 = 2 次训练 (h128, h256)
# 而不是：1 × 2 × (1 + 1 + 1) = 6 次训练
```

## 文件命名规则

### 模型文件命名

所有模式下，模型文件都使用相同的命名规则：

```
{model_type}_{mode}_acc_h{hidden_size}_lr{lr}_wd{wd}_do{dropout}.pkl
{model_type}_{mode}_acc_h{hidden_size}_lr{lr}_wd{wd}_do{dropout}_model.pth
```

**示例：**
```
rnn_sector_acc_h256_lr0.001_wd0.0001_do0.3.pkl
rnn_sector_acc_h256_lr0.001_wd0.0001_do0.3_model.pth
```

### 日志文件命名

**全组合模式：**
```
job{N}_{model}\_h{size}_lr{lr}_wd{wd}_do{drop}.log
```

**分段模式：**
```
job{N}_{model}_h{size}_stage{X}_{param}{value}.log
```

**示例：**
```bash
# 全组合模式
job1_rnn_h256_lr0.001_wd0.0001_do0.3.log

# 分段模式
job1_rnn_h256_stage1_lr0.001.log
job2_rnn_h256_stage2_wd0.0001.log
job3_rnn_h256_stage3_drop0.3.log
```

## 对比表

| 特性 | 全组合搜索 | 分段搜索 |
|------|-----------|---------|
| **实验数** | N × M × L × W × D | N × M × (L + W + D) |
| **覆盖范围** | 完整空间 | 三条坐标轴 |
| **时间成本** | 高 (指数增长) | 低 (线性增长) |
| **最优保证** | 一定找到全局最优 | 可能错过组合效应 |
| **推荐用途** | 精细调优 | 粗略探索 |

## 实际使用建议

### 第一阶段：粗略探索（分段搜索）

```bash
MODEL_TYPES=("rnn")
HIDDEN_SIZES=(128 256)
LRS=(0.0005 0.001 0.002)
WDS=(0.00005 0.0001 0.0002)
DROPS=(0.2 0.3 0.4)
USE_GRID_SEARCH=false

# 18 次训练，快速找到大致最优范围
```

### 第二阶段：精细调优（全组合搜索）

基于第一阶段结果，缩小搜索范围：

```bash
MODEL_TYPES=("rnn")
HIDDEN_SIZES=(256)  # 选择最佳 size
LRS=(0.0008 0.001 0.0012)      # 在最佳值周围
WDS=(0.00008 0.0001 0.00012)   # 在最佳值周围
DROPS=(0.25 0.3 0.35)          # 在最佳值周围
USE_GRID_SEARCH=true

# 27 次训练，找到最优组合
```

### 第三阶段：验证（单一配置）

```bash
MODEL_TYPES=("rnn")
HIDDEN_SIZES=(256)
LRS=(0.001)      # 最优值
WDS=(0.0001)     # 最优值
DROPS=(0.3)      # 最优值
USE_GRID_SEARCH=false

# 1 次训练，多次运行验证稳定性
```

## 常见问题

### Q1: 为什么分段搜索有时会生成重复配置？

**A**: 当 `LRS`, `WDS`, `DROPS` 都只有一个元素时，会生成3个完全相同的配置。

**解决方案**: 
- 方案1：使用全组合模式（会自动检测并只生成一次）
- 方案2：确保每个数组至少有2个不同的值

### Q2: 如何避免文件覆盖？

**A**: 确保不同的超参数组合产生不同的文件名。

**检查方法**：
```bash
# 查看会生成的组合（不实际运行）
./run_hparam_search.sh 2>&1 | grep "添加:"
```

### Q3: 全组合搜索太慢怎么办？

**A**: 使用以下策略减少实验数：
1. 先用分段搜索找大致范围
2. 减少搜索空间（fewer values per parameter）
3. 固定一些参数（reduce dimensionality）
4. 增加并行度（MAX_JOBS_PER_GPU=2）

## 总结

- **USE_GRID_SEARCH=true**: 全面但耗时，适合精细调优
- **USE_GRID_SEARCH=false**: 快速但粗略，适合探索阶段
- **单一配置自动优化**: 避免不必要的重复训练
- **推荐策略**: 分段探索 → 全组合精调 → 验证稳定性

