# 模型文件说明文档

## PKL 和 PTH 文件的区别

在训练过程中，我们保存了两种类型的文件：

### 1. PKL 文件 (`.pkl`)

**保存内容：** 训练过程的统计数据和元信息

```python
# 保存的数据结构
{
    'train_acc_char': np.array([...]),  # 训练集字符准确率曲线
    'val_acc_char': np.array([...]),    # 验证集字符准确率曲线
    'train_acc_pos': np.array([...]),   # 训练集位置准确率（sector）或 MSE（coord）
    'val_acc_pos': np.array([...]),     # 验证集位置准确率（sector）或 MSE（coord）
    'actual_epochs': 200,               # 实际训练的轮数
    # 注意：不包含模型结构或权重
}
```

**用途：**
- 绘制训练曲线
- 分析训练过程
- 比较不同超参数配置的性能
- **无法**恢复模型进行推理

### 2. PTH 文件 (`_model.pth`)

**保存内容：** 模型的所有参数（权重和偏置）

```python
# 保存的 state_dict 结构（示例）
{
    'conv1.weight': Tensor(shape=[32, 2, 5, 5]),
    'conv1.bias': Tensor(shape=[32]),
    'conv2.weight': Tensor(shape=[64, 32, 3, 3]),
    'conv2.bias': Tensor(shape=[64]),
    'rnn.weight_ih_l0': Tensor(shape=[256, 9216]),  # input-to-hidden
    'rnn.weight_hh_l0': Tensor(shape=[256, 256]),   # hidden-to-hidden
    'rnn.bias_ih_l0': Tensor(shape=[256]),
    'rnn.bias_hh_l0': Tensor(shape=[256]),
    'fcchar.weight': Tensor(shape=[10, 256]),       # 10 classes
    'fcpos.weight': Tensor(shape=[9, 256]),         # 9 sectors
    # ... 更多层的参数
}
```

**用途：**
- 加载模型进行推理
- 继续训练（需要配合正确的模型定义）
- 分析模型权重
- 从参数形状**推断**模型配置（hidden_size, num_classes 等）

**局限性：**
- **不包含**超参数配置（learning rate, dropout rate, weight decay 等）
- **不包含**训练过程信息
- 需要与正确的模型定义代码配合使用

## 从 PTH 文件推断模型配置

虽然 PTH 文件不直接保存超参数，但我们可以从权重矩阵的形状推断出部分配置：

| 参数 | 推断方法 | 示例 |
|------|---------|------|
| `hidden_size` | `rnn.weight_hh_l0.shape[0]` | 256 |
| `num_classes` | `fcchar.weight.shape[0]` | 10 |
| `num_pos` | `fcpos.weight.shape[0]` | 9 (sector) 或 2 (coord) |
| `mode` | 根据 `num_pos` 判断 | 9→sector, 2→coord |
| `conv1_out_channels` | `conv1.weight.shape[0]` | 32 |
| `kernel_size` | `conv1.weight.shape[2]` | 5 |

**无法推断的参数：**
- `dropout_rate` - dropout 不是模型参数
- `learning_rate` - 优化器参数，不保存在模型中
- `weight_decay` - 优化器参数
- `batch_size` - 训练配置，不属于模型

## 使用诊断脚本

### 基本用法

```bash
# 分析单个模型
python diagnose_training_config.py results/models/xxx.pkl

# 对比两个模型
python diagnose_training_config.py \
    results/models/model1.pkl \
    results/models/model2.pkl \
    --compare
```

### 输出示例

```
================================================================================
Analysis: rnn_sector_h256.pkl
================================================================================

从 PTH 文件提取模型配置:
  RNN type: rnn
  Hidden size: 256
  Num classes: 10
  Num positions: 9 (sector mode)
  Conv1: in=2, out=32, kernel=5
  Conv2: in=32, out=64
  Total parameters: 2,567,978
  RNN weight_ih shape: torch.Size([256, 9216])
  RNN weight_hh shape: torch.Size([256, 256])

PKL 文件中的训练曲线:
  - train_acc_char: <class 'numpy.ndarray'>
  - val_acc_char: <class 'numpy.ndarray'>
  ...

性能指标:
  Train char acc: max=99.61%, final=99.60%
  Val char acc: max=55.40%, final=53.49%
  ...
```

## 性能差异诊断流程

当发现两个模型性能差异较大时：

### 1. 对比模型架构

```bash
python diagnose_training_config.py model1.pkl model2.pkl --compare
```

检查：
- ✅ `total_params` 是否一致？
- ✅ `hidden_size` 是否一致？
- ✅ `rnn_type` 是否一致？

### 2. 检查训练日志

查看实际使用的超参数：

```bash
# 查看训练日志
tail -100 logs_hparam/train_xxx.log

# 关键信息：
# - "Using batch_size = XXX"
# - "Learning rate: XXX"
# - "Weight decay: XXX"
# - "Dropout rate: XXX"
```

### 3. 常见问题排查

| 现象 | 可能原因 | 解决方案 |
|------|---------|---------|
| Train acc 下降 10%+ | batch_size 不一致 | 固定 batch_size=256 |
| Val acc 意外上升 | 模型欠拟合 | 检查是否 dropout 过大 |
| 参数数量不匹配 | hidden_size 不同 | 确认模型配置 |
| Loss 曲线震荡 | learning rate 过大 | 降低 lr |

## 文件命名规范

```
{model_type}_{mode}_{acc_flag}_h{hidden_size}_lr{lr}_wd{wd}_do{dropout}.pkl
例如：rnn_sector_acc_h256_lr0.001_wd0.0001_do0.3.pkl

对应的 PTH 文件：
rnn_sector_acc_h256_lr0.001_wd0.0001_do0.3_model.pth
```

命名中包含的信息：
- `model_type`: rnn, lstm, gru, gawf
- `mode`: sector, coord, allchars
- `acc_flag`: acc（加速训练）或无
- `h{size}`: hidden size
- `lr{value}`: learning rate
- `wd{value}`: weight decay
- `do{value}`: dropout rate

## 总结

- **PKL**: 保存训练曲线和性能指标，用于分析和可视化
- **PTH**: 保存模型权重，用于推理和继续训练
- **诊断脚本**: 从 PTH 推断配置，从 PKL 获取性能，综合分析
- **关键**: 超参数（lr, wd, dropout, batch_size）不在 PTH 中，需要从文件名或日志中获取

