# main_pytorch.py 脚本结构分析

## 1. 脚本概述

`main_pytorch.py` 是另一个项目（dendritic-ANN）的Python脚本，使用Keras backend设置为PyTorch。

## 2. 脚本结构

### 2.1 依赖和导入
- 使用 `os.environ["KERAS_BACKEND"] = "torch"` 设置Keras后端为PyTorch
- 从 `opt` 模块导入：
  - `make_masks`: 创建掩码
  - `custom_train_loop_torch`: 自定义训练循环
  - `get_model`: 获取模型
  - `get_data`: 获取数据
  - `get_model_name`: 获取模型名称

### 2.2 命令行参数（共15个）
1. `sys.argv[1]`: GPU设备ID
2. `sys.argv[2]`: 序列标志（0=False, 1=True）
3. `sys.argv[3]`: 早停标志（0=False, 1=True）
4. `sys.argv[4]`: 试验编号（用于随机种子）
5. `sys.argv[5]`: 模型类型（0-11，不同ANN架构）
6. `sys.argv[6]`: 噪声sigma值
7. `sys.argv[7]`: 数据类型（'mnist', 'fmnist', 'kmnist', 'emnist', 'cifar10'）
8. `sys.argv[8]`: 树突数量（num_dends）
9. `sys.argv[9]`: 胞体数量（num_soma）
10. `sys.argv[10]`: 层数（num_layers）
11. `sys.argv[11]`: 突触数量（synapses）
12. `sys.argv[12]`: Dropout标志（0=False, 1=True）
13. `sys.argv[13]`: Dropout率
14. `sys.argv[14]`: 学习率（lr）
15. `sys.argv[15]`: 保存路径

### 2.3 数据加载
```python
data, labels, img_height, img_width, channels = get_data(
    validation_split=0.1,
    dtype=datatype,
    normalize=True,
    add_noise=noise,
    sigma=sigma,
    sequential=seq_flag,
    batch_size=batch_size,
    seed=trial,
)
```

**输入格式：**
- `get_data` 函数返回：
  - `data`: 字典，包含 `'train'`, `'val'`, `'test'` 键
  - `labels`: 字典，包含 `'train'`, `'val'`, `'test'` 键
  - `img_height`, `img_width`, `channels`: 图像尺寸信息

**数据提取：**
```python
x_train, x_val, x_test = data['train'], data['val'], data['test']
y_train, y_val, y_test = labels['train'], labels['val'], labels['test']
```

### 2.4 模型创建
- 使用 `get_model` 创建Keras模型
- 模型类型取决于 `model_type` 参数（0-11）
- 支持多种架构：dendritic ANN (dANN), vanilla ANN (vANN), sparse ANN (sANN)
- 使用 `make_masks` 创建掩码来限制连接

### 2.5 训练过程
```python
model, out = custom_train_loop_torch(
    model,
    loss_fn,
    optimizer,
    Masks,
    batch_size,
    num_epochs,
    x_train, y_train,
    x_val, y_val,
    x_test, y_test,
    shuffle=False if seq_flag else True,
    early_stop=early_stop,
    patience=10,
)
```

**输入：**
- `model`: Keras模型
- `loss_fn`: Keras损失函数（`SparseCategoricalCrossentropy`）
- `optimizer`: Keras优化器（`Adam`）
- `Masks`: 掩码列表
- `batch_size`: 批次大小
- `num_epochs`: 训练轮数
- `x_train, y_train`: 训练数据
- `x_val, y_val`: 验证数据
- `x_test, y_test`: 测试数据

**输出：**
- `model`: 训练后的模型
- `out`: 输出字典（包含训练结果）

### 2.6 输出格式
- 模型保存为 `.keras` 文件
- 结果保存为 `.pkl` 文件（包含 `out` 字典和 `Masks`）

---

# train_rnn_updated.py 脚本结构分析

## 1. 脚本概述

`train_rnn_updated.py` 是用户的RNN训练脚本，使用纯PyTorch实现。

## 2. 脚本结构

### 2.1 数据加载
```python
# 从numpy文件加载stimuli
stims_train = np.load(stim_train_path, allow_pickle=True)
stims_val = np.load(stim_val_path, allow_pickle=True)

# 从TSV文件加载labels
lbls_train = pd.read_csv(label_train_path, sep="\t", index_col=0)
lbls_val = pd.read_csv(label_val_path, sep="\t", index_col=0)
```

**输入格式：**
- **Stimuli**: numpy数组，形状为 `(num_samples, num_frames, height, width)`
- **Labels**: pandas DataFrame，包含列：
  - `fg_char_id`: 前景字符ID
  - `fg_char_x`: 前景字符x坐标
  - `fg_char_y`: 前景字符y坐标
  - `bg_char_ids`: 背景字符ID（逗号分隔的字符串）

### 2.2 数据集类
```python
class MC_RNN_Dataset(Dataset):
    def __getitem__(self, idx):
        # 返回:
        # stacked_frames: (frame_num, chan_num, height, width)
        # labels: 根据模式不同
```

**输出格式：**
- **stacked_frames**: `(frame_num, chan_num, height, width)` - 堆叠的多通道图像
- **labels**: 
  - Sector模式: `(frame_num, 2)` - `[char_id, sector_id]`
  - Coordinate模式: `(frame_num, 3)` - `[char_id, x, y]`
  - All-chars模式: `(frame_num, max_chars)` - 每个位置是char_id或-1（填充）

### 2.3 模型类
- `RNNConv`: 基础RNN模型
- `GRUConv`: GRU模型
- `LSTMConv`: LSTM模型
- `GaWFRNNConv`: GaWF RNN模型（带反馈机制）

所有模型继承自 `nn.Module`，输出：
- `out_char`: 字符预测 `(B, T, num_classes)` 或 `(B, T, max_chars, num_classes)`
- `out_pos`: 位置预测 `(B, T, num_pos)` 或 `None`

### 2.4 训练函数
```python
results = network_train(
    mdl,
    train_ds,
    val_ds,
    num_epochs=args.num_epochs,
    lr=lr,
    ...
)
```

**输入：**
- `mdl`: PyTorch模型（nn.Module）
- `train_ds`: `MC_RNN_Dataset` 训练数据集
- `val_ds`: `MC_RNN_Dataset` 验证数据集
- 其他超参数

**输出：**
```python
# Sector模式:
{
    "train_acc_char": np.array,  # 训练字符准确率
    "val_acc_char": np.array,    # 验证字符准确率
    "train_acc_pos": np.array,   # 训练位置准确率（sector）
    "val_acc_pos": np.array,     # 验证位置准确率（sector）
    "model": nn.Module,          # 训练后的模型
    "actual_epochs": int         # 实际训练的轮数
}

# Coordinate模式:
{
    "train_acc_char": np.array,
    "val_acc_char": np.array,
    "train_err_pos": np.array,   # 训练位置误差（MSE）
    "val_err_pos": np.array,     # 验证位置误差（MSE）
    "model": nn.Module,
    "actual_epochs": int
}

# All-chars模式:
{
    "train_acc_char": np.array,  # 精确匹配准确率
    "val_acc_char": np.array,
    "model": nn.Module,
    "actual_epochs": int
}
```

### 2.5 Loss函数
```python
def loss_fn(out_char, out_pos, labels):
    # 字符损失
    loss_char = criterion_char(outputs_char, labels_char)
    
    # 位置损失（根据模式不同）
    if use_sector:
        loss_pos = criterion_pos(outputs_pos, labels_pos)  # CrossEntropyLoss
    else:
        loss_pos = criterion_pos(outputs_pos, labels_pos)  # MSELoss
    
    # RNN正则化项
    rnn_hh_diag = torch.diagonal(rnn_hh).abs().mean()
    
    # 总损失
    loss = (loss_weights[0] * loss_char) + (loss_weights[1] * loss_pos) + rnn_diag_lambda * rnn_hh_diag
    return loss
```

---

# 兼容性分析

## 1. 输入格式差异

| 项目 | main_pytorch.py | train_rnn_updated.py |
|------|----------------|---------------------|
| **数据格式** | 通过 `get_data` 函数返回字典 | 直接从numpy/TSV文件加载 |
| **数据形状** | 未知（取决于 `get_data` 实现） | `(num_samples, num_frames, H, W)` |
| **标签格式** | 字典中的数组 | DataFrame with columns |
| **数据集类** | 无（直接使用数组） | `MC_RNN_Dataset` (PyTorch Dataset) |

## 2. 模型差异

| 项目 | main_pytorch.py | train_rnn_updated.py |
|------|----------------|---------------------|
| **框架** | Keras (backend=torch) | 纯PyTorch |
| **模型类型** | dendritic-ANN架构 | RNN/GRU/LSTM/GaWF |
| **模型接口** | Keras模型（`get_weights()`, `set_weights()`） | PyTorch nn.Module |

## 3. 训练循环差异

| 项目 | main_pytorch.py | train_rnn_updated.py |
|------|----------------|---------------------|
| **训练函数** | `custom_train_loop_torch` | `network_train` |
| **损失函数** | `SparseCategoricalCrossentropy` | 组合损失（字符+位置+正则化） |
| **优化器** | Keras `Adam` | PyTorch `torch.optim.Adam` |
| **数据加载** | 直接传入数组 | PyTorch `DataLoader` |

## 4. 输出格式差异

| 项目 | main_pytorch.py | train_rnn_updated.py |
|------|----------------|---------------------|
| **输出内容** | `out` 字典（内容未知） | 明确的字典结构（acc_char, acc_pos等） |
| **Loss输出** | 未知（可能在 `out` 中） | 在训练过程中计算，不直接返回 |

---

# 修改可行性评估

## ✅ 可以修改

**是的，可以在 `main_pytorch.py` 基础上修改**，使其接受用户的输入格式并返回相同的loss输出。

## 🔧 需要修改的部分

### 1. 数据加载部分
- **当前**: 使用 `get_data` 函数（来自 `opt` 模块）
- **需要**: 替换为用户的数据加载逻辑：
  ```python
  # 加载numpy和TSV文件
  stims_train = np.load(stim_train_path)
  lbls_train = pd.read_csv(label_train_path, sep="\t", index_col=0)
  # 创建 MC_RNN_Dataset
  train_ds = MC_RNN_Dataset(stims_train, lbls_train, ...)
  ```

### 2. 模型部分
- **当前**: 使用 `get_model` 创建Keras模型
- **需要**: 替换为PyTorch模型类（RNNConv, GRUConv等）
- **注意**: 需要移除Keras相关的代码（`get_weights()`, `set_weights()`等）

### 3. 训练循环部分
- **当前**: 使用 `custom_train_loop_torch`
- **需要**: 替换为 `network_train` 函数
- **注意**: 需要适配PyTorch的DataLoader和训练循环

### 4. Loss函数部分
- **当前**: `SparseCategoricalCrossentropy`（仅分类）
- **需要**: 实现组合损失函数（字符+位置+正则化）
- **注意**: 需要根据模式（sector/coordinate/all-chars）选择不同的损失计算

### 5. 输出格式部分
- **当前**: `out` 字典（内容未知）
- **需要**: 返回与 `train_rnn_updated.py` 相同的字典结构

## ⚠️ 注意事项

1. **Keras vs PyTorch**: `main_pytorch.py` 使用Keras API，需要完全转换为PyTorch
2. **依赖模块**: `opt` 模块中的函数（`get_data`, `get_model`等）在当前项目中不存在
3. **数据格式**: 需要确保数据格式完全匹配用户的输入格式
4. **模型架构**: dendritic-ANN架构与RNN架构不同，可能需要重新设计模型部分

## 📝 建议的修改步骤

1. **移除Keras依赖**: 删除所有Keras相关代码
2. **添加数据加载**: 实现用户的数据加载逻辑
3. **替换模型**: 使用PyTorch模型类替换Keras模型
4. **替换训练循环**: 使用 `network_train` 函数
5. **适配Loss**: 实现组合损失函数
6. **统一输出**: 确保输出格式与 `train_rnn_updated.py` 一致

---

# 总结

**结论**: ✅ **可以修改**

`main_pytorch.py` 的结构相对简单，主要是一个配置和调用脚本。虽然它使用Keras API，但可以完全重写为PyTorch版本，使其：
1. 接受用户的输入格式（numpy数组 + DataFrame）
2. 使用用户的模型架构（RNN/GRU/LSTM/GaWF）
3. 返回与 `train_rnn_updated.py` 相同的loss输出格式

主要工作是将Keras代码转换为PyTorch，并集成用户现有的数据加载和训练逻辑。
