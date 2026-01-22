# 进一步优化建议 - Character任务过拟合

## 当前状态分析

### Sector任务 ✅
- Train: 89%, Val: 85%, 差距: 4%
- **表现良好，过拟合已基本解决**

### Character任务 ⚠️
- Train: 80%, Val: 55%, 差距: 25%
- **仍存在明显过拟合，需要进一步优化**

## 优化策略

### 策略1：针对Character任务增强正则化（推荐）

由于Character任务比Sector任务更难，可能需要更强的正则化：

```python
# 方案A：适度增强正则化
weight_decay=3e-4,  # 从2e-4增加到3e-4
dropout_rate=0.45,  # 从0.4增加到0.45
# RNN dropout: 0.4 → 0.45
```

### 策略2：任务特定的Dropout策略

在分类器层为Character和Sector使用不同的dropout率：

```python
def classifier(self, x):
    x = F.dropout(x, p=0.3, training=self.training)
    char_out = self.fcchar(x)
    # Character分类器使用更强的dropout
    char_out = F.dropout(char_out, p=0.2, training=self.training) if self.training else char_out
    pos_out = self.fcpos(x)
    return char_out, pos_out
```

### 策略3：调整Loss权重

增加Character loss的权重，让模型更关注Character任务：

```python
# 在训练函数中
if use_sector:
    loss_weights = [1.5, 1.0]  # Character权重从1.0增加到1.5
else:
    loss_weights = [1, 0.001]
```

### 策略4：更早的Early Stopping

Character验证准确率在epoch 60-80达到峰值后开始下降，可以：

```python
early_stopping_patience=5,  # 从10减少到5，更早停止
```

### 策略5：添加Label Smoothing

防止模型对Character预测过度自信：

```python
criterion_char = nn.CrossEntropyLoss(label_smoothing=0.1)
```

### 策略6：在分类器层添加Dropout

```python
def classifier(self, x):
    x = F.dropout(x, p=0.3, training=self.training)  # 添加dropout
    return self.fcchar(x), self.fcpos(x)
```

## 推荐的渐进式调整

### 第一步：增强正则化 + 分类器Dropout
```python
weight_decay=3e-4,
dropout_rate=0.45,
# RNN dropout: 0.4 → 0.45
# 在classifier中添加dropout: 0.3
```

### 第二步：如果仍过拟合，添加Label Smoothing
```python
criterion_char = nn.CrossEntropyLoss(label_smoothing=0.1)
```

### 第三步：调整Early Stopping
```python
early_stopping_patience=5,
```

## 预期效果

- **Character准确率**：Val从55%提升到60-65%
- **Train-Val差距**：从25%缩小到15-20%
- **Sector准确率**：保持85%左右（不要降低）

## 注意事项

1. **不要过度正则化**：避免Character任务出现欠拟合
2. **监控两个任务**：确保Sector任务不会因为增强正则化而变差
3. **逐步调整**：一次只改变一个参数，观察效果

