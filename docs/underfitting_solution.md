# 欠拟合问题分析与解决方案

## 当前情况分析

从新的训练曲线可以看出：
- **Character accuracy**: Train ~25%, Validation ~25-28% (差距很小，但准确率很低)
- **Sector accuracy**: Train ~78%, Validation ~80-83% (差距很小，val甚至略高于train)

### 问题诊断

1. ✅ **过拟合问题已解决**：train和val差距很小
2. ❌ **出现欠拟合问题**：整体准确率偏低，模型学习能力不足
3. ⚠️ **正则化过度**：dropout=0.5和weight_decay=5e-4可能太强

## 解决方案：找到正则化与模型能力的平衡点

### 策略1：适度降低正则化强度（推荐先试）

**目标**：在防止过拟合的同时，允许模型学习更多特征

```python
# 方案A：温和调整（推荐）
weight_decay=2e-4,  # 从5e-4降低到2e-4
dropout_rate=0.4,   # 从0.5降低到0.4
# RNN层dropout: 0.5 → 0.4

# 方案B：更激进的调整
weight_decay=1e-4,  # 回到原始值
dropout_rate=0.35,  # 略高于原始0.3
# RNN层dropout: 0.5 → 0.4
```

### 策略2：启用Early Stopping + 学习率调度器

**目标**：在最佳泛化点停止训练，避免过拟合

```python
use_early_stopping=True,
early_stopping_patience=10,
min_delta=0.001

# 添加学习率调度器
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optim, mode='max', factor=0.5, patience=5
)
```

### 策略3：增加模型容量（如果策略1和2不够）

**目标**：提高模型学习能力

```python
# 增加RNN hidden_size
hidden_size = 384  # 从256增加到384

# 或增加CNN通道数
self.conv1 = nn.Conv2d(2, 48, ...)  # 从32增加到48
self.conv2 = nn.Conv2d(48, 96, ...)  # 从64增加到96
```

### 策略4：调整学习率

**目标**：找到最佳学习率，平衡收敛速度和稳定性

```python
lr=0.0005,  # 从0.001降低到0.0005，更稳定的训练
# 或使用学习率调度器动态调整
```

### 策略5：分层正则化策略

**目标**：不同层使用不同的正则化强度

```python
# CNN层：较低dropout（0.3-0.4），因为需要学习特征
dropout_rate=0.35

# RNN层：中等dropout（0.4-0.5），防止序列过拟合
x = F.dropout(x, p=0.45, training=self.training)

# 分类器层：添加dropout（0.2-0.3），防止输出层过拟合
def classifier(self, x):
    x = F.dropout(x, p=0.25, training=self.training)
    return self.fcchar(x), self.fcpos(x)
```

## 推荐的渐进式调整方案

### 第一步：适度降低正则化 + 启用Early Stopping
```python
weight_decay=2e-4,  # 降低weight decay
dropout_rate=0.4,   # 降低CNN dropout
# RNN dropout: 0.5 → 0.4
use_early_stopping=True,
early_stopping_patience=10
```

### 第二步：如果准确率提升但仍有过拟合
```python
weight_decay=1.5e-4,
dropout_rate=0.35,
# RNN dropout: 0.4 → 0.35
```

### 第三步：如果仍欠拟合，增加模型容量
```python
# 增加hidden_size或CNN通道数
```

## 预期效果

- **短期目标**：Character准确率从25%提升到40-50%
- **中期目标**：Sector准确率从78%提升到85-90%
- **长期目标**：找到最佳平衡点，train和val差距在5-10%以内，整体准确率最大化

## 注意事项

1. **逐步调整**：不要一次性改变太多参数
2. **监控两个指标**：
   - Train-Val差距（过拟合指标）
   - 整体准确率（欠拟合指标）
3. **Character任务更难**：可能需要单独调整character相关的正则化
4. **保存最佳模型**：确保early stopping保存的是验证集上表现最好的模型

