"""
单个模型结果的可视化脚本
用于在远程服务器批量生成训练曲线图
"""
import os
import sys
import pickle
import argparse
import numpy as np

# Fix numpy version compatibility for pickle loading
# 修复numpy版本兼容性问题
import numpy.core.numeric as _num
try:
    import numpy._core.numeric
except ImportError:
    import sys
    sys.modules['numpy._core.numeric'] = _num
    sys.modules['numpy._core'] = np.core

import matplotlib
matplotlib.use('Agg')  # 无图形界面模式
import matplotlib.pyplot as plt
from pathlib import Path


def format_number(value, is_lr_or_wd=False):
    """
    智能格式化数值显示
    - 对于很小的数值（< 0.001）使用科学计数法
    - 对于lr/wd，如果是常见值则用普通格式，否则用科学计数法
    - 其他情况显示合适的小数位数
    """
    if value == 0:
        return "0"
    
    abs_val = abs(value)
    
    # 对于 lr 和 wd，特殊处理
    if is_lr_or_wd:
        # 如果小于 0.001 或大于等于 1，使用科学计数法
        if abs_val < 0.001 or abs_val >= 1:
            return f"{value:.1e}"
        # 否则保留 4 位小数
        else:
            return f"{value:.4f}".rstrip('0').rstrip('.')
    
    # 对于其他参数（如 dropout）
    if abs_val >= 100:
        return f"{value:.0f}"
    elif abs_val >= 1:
        return f"{value:.2f}".rstrip('0').rstrip('.')
    else:
        # 保留足够的有效数字
        return f"{value:.3f}".rstrip('0').rstrip('.')


def visualize_training_curves(pkl_path, output_path, hparams=None):
    """
    可视化训练曲线并保存为PNG
    
    Args:
        pkl_path: 结果pkl文件路径
        output_path: 输出图片路径
        hparams: 超参数字典（用于标题）
    """
    # 读取结果
    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)
    
    # 检测模式
    is_sector = 'train_acc_pos' in results
    is_allchars = 'train_acc_char' in results and 'train_acc_pos' not in results and 'train_err_pos' not in results
    
    # 获取实际训练的epoch数
    if "actual_epochs" in results:
        actual_epochs = results["actual_epochs"]
    else:
        train_acc = results["train_acc_char"]
        non_zero_mask = train_acc > 1.0
        if np.any(non_zero_mask):
            actual_epochs = np.where(non_zero_mask)[0][-1] + 1
        else:
            actual_epochs = len(train_acc)
    
    # 打印统计信息
    print(f"实际训练的epoch数: {actual_epochs}")
    
    def _safe_max(arr, epochs):
        if arr is None or len(arr) == 0:
            return None
        arr = np.asarray(arr)[:epochs]
        if arr.size == 0:
            return None
        return float(np.nanmax(arr))
    
    max_train_char = _safe_max(results.get("train_acc_char"), actual_epochs)
    max_val_char = _safe_max(results.get("val_acc_char"), actual_epochs)
    
    if max_train_char is not None:
        print(f"Train char acc max: {max_train_char:.2f}%")
    if max_val_char is not None:
        print(f"Val char acc max: {max_val_char:.2f}%")
    
    # 创建图形
    fig = plt.figure(figsize=(12, 5))
    
    # 添加总标题显示超参数信息
    if hparams is not None:
        title_parts = []
        if 'model_type' in hparams:
            title_parts.append(f"model={hparams['model_type']}")
        if 'hidden_size' in hparams:
            title_parts.append(f"h={hparams['hidden_size']}")
        if 'lr' in hparams:
            title_parts.append(f"lr={format_number(hparams['lr'], is_lr_or_wd=True)}")
        if 'wd' in hparams:
            title_parts.append(f"wd={format_number(hparams['wd'], is_lr_or_wd=True)}")
        if 'dropout' in hparams:
            title_parts.append(f"dropout={format_number(hparams['dropout'])}")
        if title_parts:
            plt.suptitle(' | '.join(title_parts), fontsize=14, fontweight='bold')
    
    # 左图：字符识别准确率
    plt.subplot(1, 2, 1)
    plt.plot(results["train_acc_char"][:actual_epochs], label="train char acc", linewidth=2)
    plt.plot(results["val_acc_char"][:actual_epochs], label="val char acc", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title("Character accuracy", fontsize=13)
    plt.ylim(-5, 105)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    
    # 右图：位置相关指标
    plt.subplot(1, 2, 2)
    if is_allchars:
        # all-chars 模式：不绘制位置曲线
        plt.axis("off")
        plt.title("No position metrics (all-chars mode)", fontsize=13)
    elif is_sector:
        # sector 模式：显示准确率
        if "train_acc_pos" in results and "val_acc_pos" in results:
            plt.plot(results["train_acc_pos"][:actual_epochs], label="train sector acc", linewidth=2)
            plt.plot(results["val_acc_pos"][:actual_epochs], label="val sector acc", linewidth=2)
            plt.xlabel("Epoch", fontsize=12)
            plt.ylabel("Accuracy (%)", fontsize=12)
            plt.title("Sector accuracy", fontsize=13)
            plt.ylim(40, 105)
            plt.legend(fontsize=10)
            plt.grid(alpha=0.3)
            
            max_train_pos = _safe_max(results.get("train_acc_pos"), actual_epochs)
            max_val_pos = _safe_max(results.get("val_acc_pos"), actual_epochs)
            if max_train_pos is not None:
                print(f"Train pos acc max: {max_train_pos:.2f}%")
            if max_val_pos is not None:
                print(f"Val pos acc max: {max_val_pos:.2f}%")
    else:
        # coordinate 模式：显示MSE
        if "train_err_pos" in results and "val_err_pos" in results:
            plt.plot(results["train_err_pos"][:actual_epochs], label="train pos MSE", linewidth=2)
            plt.plot(results["val_err_pos"][:actual_epochs], label="val pos MSE", linewidth=2)
            plt.xlabel("Epoch", fontsize=12)
            plt.ylabel("MSE (pixel^2)", fontsize=12)
            plt.title("Position error", fontsize=13)
            plt.legend(fontsize=10)
            plt.grid(alpha=0.3)
            
            # 打印最小MSE
            train_err = np.asarray(results["train_err_pos"][:actual_epochs])
            val_err = np.asarray(results["val_err_pos"][:actual_epochs])
            if train_err.size > 0:
                print(f"Train pos MSE min: {np.nanmin(train_err):.2f} pixel^2")
            if val_err.size > 0:
                print(f"Val pos MSE min: {np.nanmin(val_err):.2f} pixel^2")
    
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"图片已保存: {output_path}")


def parse_hparams_from_filename(filename):
    """
    从文件名解析超参数
    例如: rnn_coord_acc_h256_lr0.001_wd0.0001_do0.3.pkl
    """
    hparams = {}
    
    # 提取 model_type
    if filename.startswith('rnn_'):
        hparams['model_type'] = 'RNN'
    elif filename.startswith('lstm_'):
        hparams['model_type'] = 'LSTM'
    elif filename.startswith('gru_'):
        hparams['model_type'] = 'GRU'
    elif filename.startswith('gawf_'):
        hparams['model_type'] = 'GaWF'
    
    # 提取 hidden_size
    import re
    h_match = re.search(r'_h(\d+)', filename)
    if h_match:
        hparams['hidden_size'] = int(h_match.group(1))
    
    # 提取 lr (支持科学计数法，如 1e-4)
    # 使用贪婪匹配 + 只在下划线处停止
    lr_match = re.search(r'_lr([\deE.+-]+)_', filename)
    if lr_match:
        hparams['lr'] = float(lr_match.group(1))
    
    # 提取 wd (支持科学计数法)
    wd_match = re.search(r'_wd([\deE.+-]+)_', filename)
    if wd_match:
        hparams['wd'] = float(wd_match.group(1))
    
    # 提取 dropout (在文件扩展名前停止)
    do_match = re.search(r'_do([\d.]+)(?:_|\.)', filename)
    if do_match:
        hparams['dropout'] = float(do_match.group(1))
    
    return hparams


def main():
    parser = argparse.ArgumentParser(description='可视化单个模型的训练曲线')
    parser.add_argument('pkl_path', type=str, help='结果pkl文件路径')
    parser.add_argument('--output', type=str, default=None, 
                       help='输出图片路径（默认：results/visualization/<basename>.png）')
    parser.add_argument('--output_dir', type=str, default='results/visualization',
                       help='输出目录（默认：results/visualization）')
    
    args = parser.parse_args()
    
    # 检查输入文件
    pkl_path = Path(args.pkl_path)
    if not pkl_path.exists():
        print(f"错误：文件不存在 - {pkl_path}")
        sys.exit(1)
    
    # 确定输出路径
    if args.output is None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_filename = pkl_path.stem + '.png'
        output_path = output_dir / output_filename
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 从文件名解析超参数
    hparams = parse_hparams_from_filename(pkl_path.name)
    
    print(f"\n{'=' * 60}")
    print(f"可视化: {pkl_path.name}")
    print(f"超参数: {hparams}")
    print(f"{'=' * 60}\n")
    
    # 生成可视化
    visualize_training_curves(str(pkl_path), str(output_path), hparams)
    
    print(f"\n{'=' * 60}")
    print(f"完成！")
    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    main()


