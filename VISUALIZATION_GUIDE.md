# 批量可视化使用指南

## 概述

在远程服务器上完成训练结果的可视化，生成PNG图片，避免传输大量pkl和pth文件。

---

## 文件说明

### 1. `visualize_single_result.py`
- **功能**: 单个模型结果的可视化脚本
- **输入**: pkl 文件路径
- **输出**: PNG 图片（训练曲线）
- **特性**:
  - 自动从文件名解析超参数
  - 支持 sector/coord/allchars 三种模式
  - 显示训练和验证的准确率/误差曲线

### 2. `batch_visualize.sh`
- **功能**: 批量可视化脚本
- **输入**: 结果目录后缀（如 `hparam_search_2`）
- **输出**: 批量生成的PNG图片
- **特性**:
  - 自动遍历指定目录下的所有pkl文件
  - 生成统计报告
  - 容错处理（单个文件失败不影响其他）

---

## 使用步骤

### 步骤1：在远程服务器上赋予执行权限

```bash
cd /G/MIMOlab/Codes/aim3_RNN  # 或你的项目路径
chmod +x batch_visualize.sh
```

### 步骤2：运行批量可视化

```bash
# 基本用法
./batch_visualize.sh hparam_search_2

# 或指定其他结果目录
./batch_visualize.sh hparam_search_3_refined
```

### 步骤3：查看生成的图片

```bash
# 查看生成的图片列表
ls -lh results/visualization/hparam_search_2/

# 查看总大小
du -sh results/visualization/hparam_search_2/
```

### 步骤4：传输图片到本地

```bash
# 在本地Mac上执行
scp -r sjc@172.26.48.213:/G/MIMOlab/Codes/aim3_RNN/results/visualization/hparam_search_2 \
    ~/Desktop/Vscode/FAW_RNN/results/visualization/

# 或使用rsync（更高效）
rsync -avP sjc@172.26.48.213:/G/MIMOlab/Codes/aim3_RNN/results/visualization/hparam_search_2/ \
    ~/Desktop/Vscode/FAW_RNN/results/visualization/hparam_search_2/
```

---

## 输出示例

### 目录结构

```
results/
├── models/
│   └── hparam_search_2/
│       ├── rnn_coord_acc_h256_lr0.0008_wd0.0001_do0.3.pkl
│       ├── rnn_coord_acc_h256_lr0.0008_wd0.0001_do0.3_model.pth
│       ├── rnn_coord_acc_h256_lr0.001_wd0.0001_do0.3.pkl
│       └── ...
└── visualization/
    └── hparam_search_2/
        ├── rnn_coord_acc_h256_lr0.0008_wd0.0001_do0.3.png  ← 生成的图片
        ├── rnn_coord_acc_h256_lr0.001_wd0.0001_do0.3.png
        └── ...
```

### 图片内容

每张图片包含：
- **总标题**: 显示超参数（model, h, lr, wd, dropout）
- **左图**: Character Accuracy（训练集 vs 验证集）
- **右图**: Position Error/Accuracy（训练集 vs 验证集）
  - Sector 模式：显示准确率
  - Coord 模式：显示 MSE
  - All-chars 模式：不显示位置指标

---

## 单个文件可视化

如果只想可视化某个特定结果：

```bash
# 直接调用 Python 脚本
python visualize_single_result.py \
    results/models/hparam_search_2/rnn_coord_acc_h256_lr0.001_wd0.0001_do0.3.pkl \
    --output results/visualization/my_model.png

# 或使用默认输出路径
python visualize_single_result.py \
    results/models/hparam_search_2/rnn_coord_acc_h256_lr0.001_wd0.0001_do0.3.pkl
```

---

## 脚本输出示例

```bash
$ ./batch_visualize.sh hparam_search_2

============================================================
批量可视化训练结果
============================================================
输入目录: results/models/hparam_search_2
输出目录: results/visualization/hparam_search_2
============================================================

找到 9 个结果文件

处理: rnn_coord_acc_h256_lr0.0008_wd0.0001_do0.3.pkl
============================================================
可视化: rnn_coord_acc_h256_lr0.0008_wd0.0001_do0.3.pkl
超参数: {'model_type': 'RNN', 'hidden_size': 256, 'lr': 0.0008, 'wd': 0.0001, 'dropout': 0.3}
============================================================

实际训练的epoch数: 200
Train char acc max: 75.23%
Val char acc max: 45.67%
Train pos MSE min: 234.56 pixel^2
Val pos MSE min: 456.78 pixel^2
图片已保存: results/visualization/hparam_search_2/rnn_coord_acc_h256_lr0.0008_wd0.0001_do0.3.png

============================================================
完成！
============================================================

  ✓ 成功

处理: rnn_coord_acc_h256_lr0.001_wd0.0001_do0.3.pkl
...

============================================================
批量可视化完成
============================================================
成功: 9 个文件
失败: 0 个文件
输出目录: results/visualization/hparam_search_2
============================================================

生成的图片:
-rw-r--r-- 1 user group 156K Jan 22 10:30 rnn_coord_acc_h256_lr0.0008_wd0.0001_do0.3.png
-rw-r--r-- 1 user group 158K Jan 22 10:31 rnn_coord_acc_h256_lr0.001_wd0.0001_do0.3.png
...

总大小: 1.4M
```

---

## 高级用法

### 1. 在训练完成后自动可视化

在 `run_hparam_search.sh` 末尾添加：

```bash
echo "All jobs launched. Use 'ps | grep train_rnn_updated.py' 查看运行状态。"

# 等待所有任务完成
for pid in "${PIDS[@]}"; do
  wait "$pid"
done

# 自动运行可视化
echo "所有训练任务完成，开始批量可视化..."
./batch_visualize.sh "$RESULT_SUFFIX"
```

### 2. 批量比较不同实验

```bash
# 可视化多个实验目录
for suffix in hparam_search_1 hparam_search_2 hparam_search_3; do
    echo "可视化: $suffix"
    ./batch_visualize.sh "$suffix"
done
```

### 3. 只传输特定图片

```bash
# 只传输 Stage 1 的结果
scp sjc@172.26.48.213:/G/MIMOlab/Codes/aim3_RNN/results/visualization/hparam_search_2/*stage1*.png \
    ~/Desktop/

# 只传输 hidden_size=256 的结果
scp sjc@172.26.48.213:/G/MIMOlab/Codes/aim3_RNN/results/visualization/hparam_search_2/*h256*.png \
    ~/Desktop/
```

---

## 优势对比

### 传统方式（传输pkl+pth）：
```
pkl 文件: ~200KB × 9 = 1.8MB
pth 文件: ~10MB × 9 = 90MB
总计: ~92MB
传输时间: ~30-60秒（取决于网速）
```

### 新方式（只传输png）：
```
png 文件: ~150KB × 9 = 1.4MB
总计: ~1.4MB
传输时间: ~3-5秒
节省: ~98% 的传输量！
```

---

## 故障排查

### Q1: 脚本提示"未找到 .pkl 文件"

**原因**: 结果目录路径不正确

**解决**:
```bash
# 查看可用的结果目录
ls -d results/models/*/

# 确认目录名
./batch_visualize.sh <正确的目录名>
```

### Q2: 可视化失败（某些文件报错）

**原因**: pkl 文件损坏或不完整

**解决**:
```bash
# 查看具体错误信息（脚本会继续处理其他文件）
# 手动测试单个文件
python visualize_single_result.py results/models/hparam_search_2/<problematic_file>.pkl
```

### Q3: 图片显示不正常

**原因**: matplotlib 后端问题

**解决**:
```bash
# 确认使用了 Agg 后端（已在脚本中设置）
# 如果仍有问题，检查 matplotlib 版本
python -c "import matplotlib; print(matplotlib.__version__)"
```

### Q4: 权限问题

**原因**: 脚本没有执行权限

**解决**:
```bash
chmod +x batch_visualize.sh
```

---

## 集成到工作流

推荐的完整工作流：

```bash
# 1. 运行超参数搜索
./run_hparam_search.sh

# 2. 等待训练完成（监控进度）
watch -n 60 nvidia-smi

# 3. 批量可视化
./batch_visualize.sh hparam_search_2

# 4. 在本地传输图片
# (在本地Mac执行)
rsync -avP sjc@172.26.48.213:/G/MIMOlab/Codes/aim3_RNN/results/visualization/hparam_search_2/ \
    ~/Desktop/Vscode/FAW_RNN/results/visualization/hparam_search_2/

# 5. 在本地查看和分析图片
open ~/Desktop/Vscode/FAW_RNN/results/visualization/hparam_search_2/
```

---

## 注意事项

1. **图片质量**: 默认 DPI=150，如需更高质量，修改 `visualize_single_result.py` 中的 `dpi` 参数

2. **并行生成**: 当前是串行处理，如需加速可使用 GNU parallel：
   ```bash
   find results/models/hparam_search_2 -name "*.pkl" | \
       parallel python visualize_single_result.py {} --output_dir results/visualization/hparam_search_2
   ```

3. **磁盘空间**: 每张图片约 150KB，9 个结果约 1.4MB，确保有足够空间

4. **conda 环境**: 确保激活了正确的环境（包含 matplotlib, numpy 等）

---

## 扩展功能

如需添加更多可视化内容（如测试集结果），可修改 `visualize_single_result.py`:

```python
# 在 visualize_training_curves 函数中添加第三个子图
plt.subplot(1, 3, 1)  # 改为 3 列
# ... 现有代码 ...

plt.subplot(1, 3, 3)  # 新增测试集结果
# 绘制测试集曲线
```

需要进一步定制的话，请告知具体需求！

