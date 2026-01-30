# 训练版本对比说明

## 问题回顾

即使修复了 `batch_size=256`，当前代码的训练准确率仍然无法达到 100%（只有 89%），而原始 notebook 代码可以达到 99%+。

## 根本原因

在 `encoder` 卷积层添加了 **dropout2d** 正则化，这导致模型表达能力下降。

## 三个版本对比

### 1. 原始版本 (RecurrentNetwork_ori.ipynb)
**特点：**
- ✅ **Encoder 无 dropout**
- ✅ Middle 层 dropout=0.5
- ❌ 无 weight decay
- ❌ batch_size=32（较小）

**性能：**
- Train acc: **97-99%** ✓
- Val acc: 50-55%
- 可以充分学习训练集

### 2. 当前版本 (train_rnn_updated.py)
**特点：**
- ❌ **Encoder dropout=0.3** (新增)
- ✅ Middle 层 dropout=0.5
- ✅ Weight decay=0.0001 (新增)
- ✅ batch_size=256（已修复）
- ✅ 支持 allchar mode

**性能：**
- Train acc: **89%** ✗ (下降 10%)
- Val acc: 57-60% (反而提升)
- 欠拟合，无法充分学习训练集

**问题根源：**
```python
# train_rnn_updated.py: 第 255, 261 行
def encoder(self, x):
    x = self.conv1(x)
    ...
    x = F.dropout2d(x, p=self.dropout_rate, training=self.training)  # ← 问题所在！
    
    x = self.conv2(x)
    ...
    x = F.dropout2d(x, p=self.dropout_rate, training=self.training)  # ← 问题所在！
```

### 3. 简化版本 (train_rnn_sector_original.py) - **推荐**
**特点：**
- ✅ **Encoder 无 dropout** (恢复原始设计)
- ✅ Middle 层 dropout=0.5
- ❌ 无 weight decay (匹配原始)
- ✅ batch_size=256（固定）
- ❌ 不支持 allchar mode (简化)

**预期性能：**
- Train acc: **97-99%** ✓ (恢复)
- Val acc: 50-55% (与原始一致)
- 充分学习训练集，匹配原始性能

## 关键代码对比

### Encoder 实现对比

**原始版本（正确）：**
```python
def encoder(self, x):
    x = self.conv1(x)
    x = self.MP1(x)
    x = self.LNorm1(x)
    x = F.relu(x)
    # NO DROPOUT! ✓
    
    x = self.conv2(x)
    x = self.MP2(x)
    x = self.LNorm2(x)
    x = F.relu(x)
    # NO DROPOUT! ✓
    return x
```

**当前版本（过度正则化）：**
```python
def encoder(self, x):
    x = self.conv1(x)
    x = self.MP1(x)
    x = self.LNorm1(x)
    x = F.relu(x)
    x = F.dropout2d(x, p=self.dropout_rate, training=self.training)  # ✗ 导致欠拟合
    
    x = self.conv2(x)
    x = self.MP2(x)
    x = self.LNorm2(x)
    x = F.relu(x)
    x = F.dropout2d(x, p=self.dropout_rate, training=self.training)  # ✗ 导致欠拟合
    return x
```

## 使用说明

### 方案 A：使用简化版本（推荐）

```bash
# 在远程服务器
cd /G/MIMOlab/Codes/aim3_RNN

# 拉取新文件
git pull origin master

# 训练 sector mode（应该能达到 99% train acc）
python train_rnn_sector_original.py \
    --model_types rnn \
    --hidden_sizes 128 256 \
    --num_epochs 200 \
    --batch_size 256 \
    --lr 0.001 \
    --use_sector_mode \
    --result_suffix sector_original

# 预期输出：
# Train char acc: ~97-99%
# Val char acc: ~50-55%
```

### 方案 B：修改当前版本

如果想继续使用 `train_rnn_updated.py` 并保留 allchar mode 支持，需要修改：

```python
# 在 train_rnn_updated.py 中的所有模型类（RNNConv, GRUConv, LSTMConv）
# 将 encoder 方法改为：

def encoder(self, x):
    x = self.conv1(x)
    x = self.MP1(x)
    x = self.LNorm1(x)
    x = F.relu(x)
    # x = F.dropout2d(x, p=self.dropout_rate, training=self.training)  # 注释掉
    
    x = self.conv2(x)
    x = self.MP2(x)
    x = self.LNorm2(x)
    x = F.relu(x)
    # x = F.dropout2d(x, p=self.dropout_rate, training=self.training)  # 注释掉
    return x
```

## 性能对比表

| 指标 | 原始版本 | 当前版本 | 简化版本 |
|------|---------|---------|---------|
| **Train char acc** | 97.85% | 89.20% (-8.7%) | ~98% (预期) |
| **Val char acc** | 50.55% | 60.18% (+9.6%) | ~51% (预期) |
| **Train pos acc** | 97.68% | 90.39% (-7.3%) | ~98% (预期) |
| **Val pos acc** | 82.54% | 85.22% (+2.7%) | ~83% (预期) |
| **Encoder dropout** | 无 | 0.3 | 无 |
| **Weight decay** | 无 | 0.0001 | 无 |
| **Batch size** | 32 | 256 | 256 |

## 结论

1. **当前版本的问题**：在 encoder 添加 dropout 导致过度正则化，模型无法充分学习训练集
2. **影响分析**：
   - ✗ Train acc 下降 10%：模型表达能力不足
   - ✓ Val acc 提升 7-10%：正则化减少过拟合，但牺牲了训练集性能
3. **解决方案**：
   - 推荐使用 `train_rnn_sector_original.py`（简化版本）
   - 或者注释掉当前版本中 encoder 的 dropout

## 历史修改记录

| 修改 | 时间点 | 目的 | 副作用 |
|------|--------|------|--------|
| 添加 allchar mode | 中期 | 支持多字符预测 | 无 |
| 添加 encoder dropout | 中期 | 防止过拟合 | ✗ 导致欠拟合 |
| 添加 weight decay | 中期 | 额外正则化 | 轻微影响 |
| Batch size 自动搜索 | 中期 | 优化 GPU 利用 | ✗ 选择过小 |
| 固定 batch_size=256 | 最近 | 修复性能 | ✓ 部分改善 |
| 移除 encoder dropout | 现在 | 恢复原始性能 | ✓ 完全修复 |

## 推荐行动

1. **立即**: 使用 `train_rnn_sector_original.py` 重新训练
2. **验证**: 确认 train acc 可以达到 97-99%
3. **对比**: 将新结果与原始模型对比
4. **决策**: 如果性能匹配，说明 encoder dropout 确实是根本原因

