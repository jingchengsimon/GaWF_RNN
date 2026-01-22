# 过拟合问题分析与改进方案

## 当前过拟合情况

从训练曲线可以看出：
- **Character accuracy**: Train ~100%, Validation ~55% (差距45%)
- **Sector accuracy**: Train ~99%, Validation ~85% (差距14%)

这是典型的**严重过拟合**现象。

## 当前已有的防过拟合措施

1. ✅ **Dropout**: CNN层0.3，RNN层0.5
2. ✅ **Weight Decay (L2正则化)**: 1e-4
3. ✅ **LayerNorm**: 在CNN和RNN层都有
4. ✅ **Gradient Clipping**: max_norm=2.0
5. ❌ **Early Stopping**: 当前设置为`False`（未启用）

## 改进方案（按优先级排序）

### 🔴 高优先级（立即实施）

#### 1. **启用Early Stopping**
```python
use_early_stopping=True,  # 当前为False
early_stopping_patience=10,  # 可以适当减小
min_delta=0.001
```
**原因**: 训练200个epoch，但验证准确率在50-60 epoch后就不再提升，继续训练只会加剧过拟合。

#### 2. **增加Dropout率**
```python
# CNN encoder dropout: 0.3 → 0.5
dropout_rate=0.5  # 当前为0.3

# RNN层dropout: 0.5 → 0.6-0.7
# 在模型定义中修改
x = F.dropout(x, p=0.6, training=self.training)  # 当前为0.5
```
**原因**: 当前dropout率可能不足以防止过拟合，特别是RNN层。

#### 3. **增加Weight Decay**
```python
weight_decay=1e-3  # 当前为1e-4，增加到1e-3或5e-4
```
**原因**: 1e-4的weight decay可能太弱，无法有效约束模型参数。

#### 4. **添加学习率调度器（Learning Rate Scheduler）**
```python
# 使用ReduceLROnPlateau，当验证准确率不再提升时降低学习率
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optim, mode='max', factor=0.5, patience=5, verbose=True
)
# 在每个epoch后调用
scheduler.step(val_acc_char[epoch])
```
**原因**: 固定学习率可能导致后期训练不稳定，降低学习率有助于模型收敛到更好的泛化点。

### 🟡 中优先级（建议实施）

#### 5. **添加Label Smoothing**
```python
# 修改损失函数
criterion_char = nn.CrossEntropyLoss(label_smoothing=0.1)
criterion_pos = nn.CrossEntropyLoss(label_smoothing=0.1)  # sector mode
```
**原因**: Label smoothing可以防止模型过度自信，提高泛化能力。

#### 6. **在分类器层添加Dropout**
```python
# 在classifier方法中添加dropout
def classifier(self, x):
    x = F.dropout(x, p=0.3, training=self.training)
    return self.fcchar(x), self.fcpos(x)
```
**原因**: 分类器层直接输出最终预测，添加dropout可以进一步防止过拟合。

#### 7. **减小模型容量（如果可能）**
```python
# 减小RNN hidden_size: 256 → 128或192
hidden_size = 128  # 当前为256

# 减小CNN通道数: 32→64 → 16→32
self.conv1 = nn.Conv2d(2, 16, ...)  # 当前为32
self.conv2 = nn.Conv2d(16, 32, ...)  # 当前为64
```
**原因**: 模型可能过于复杂，减小容量可以减少过拟合风险。

### 🟢 低优先级（可选实施）

#### 8. **数据增强（Data Augmentation）**
```python
# 在Dataset类中添加数据增强
# 例如：随机翻转、轻微旋转、噪声添加等
# 注意：需要根据任务特点选择合适的增强方式
```

#### 9. **使用Batch Normalization替代或补充LayerNorm**
```python
# 在某些层添加BatchNorm
self.BN1 = nn.BatchNorm2d(32)
```

#### 10. **增加训练数据量**
如果可能，增加训练样本数量。

## 推荐的改进配置

### 方案A：保守改进（推荐先试）
```python
results_rnn = network_train(
    mdl_rnn, 
    train_ds, 
    val_ds, 
    num_epochs=200, 
    use_acceleration=use_acceleration,
    weight_decay=5e-4,  # 增加到5e-4
    dropout_rate=0.5,   # 增加到0.5
    use_early_stopping=True,  # 启用
    early_stopping_patience=10,
    min_delta=0.001
)
```

### 方案B：激进改进（如果方案A效果不佳）
```python
results_rnn = network_train(
    mdl_rnn, 
    train_ds, 
    val_ds, 
    num_epochs=200, 
    use_acceleration=use_acceleration,
    weight_decay=1e-3,  # 增加到1e-3
    dropout_rate=0.6,   # 增加到0.6
    use_early_stopping=True,
    early_stopping_patience=8,
    min_delta=0.001
)
# 同时需要修改模型中的RNN dropout: 0.5 → 0.7
```

## 实施步骤

1. **第一步**: 启用early stopping（最简单，立即见效）
2. **第二步**: 增加dropout率和weight decay
3. **第三步**: 添加学习率调度器
4. **第四步**: 如果仍有过拟合，考虑减小模型容量
5. **第五步**: 添加label smoothing和其他高级技术

## 预期效果

- **短期目标**: 将train-val准确率差距从45%缩小到20-30%
- **中期目标**: 验证准确率提升5-10%
- **长期目标**: 达到更好的泛化性能，train和val准确率差距在10%以内

## 注意事项

1. **不要同时应用所有改进**：建议逐步添加，观察每个改进的效果
2. **监控训练过程**：关注train和val准确率的差距变化
3. **保存最佳模型**：确保early stopping保存的是验证集上表现最好的模型
4. **记录实验配置**：记录每次实验的参数配置，便于对比分析

