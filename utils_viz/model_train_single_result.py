"""
单个模型结果的可视化脚本
用于在远程服务器批量生成训练曲线图
"""
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


def visualize_training_curves(pkl_path, output_path, hparams=None, epoch_start=0, epoch_end=None):
    """
    可视化训练曲线并保存为PNG
    
    Args:
        pkl_path: 结果pkl文件路径
        output_path: 输出图片路径
        hparams: 超参数字典（用于标题）
        epoch_start: 绘制的起始 epoch（含，0-based）
        epoch_end: 绘制的结束 epoch（不含）；None 表示画到最后一个 epoch
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
    
    # 绘制范围：仅 [epoch_start, epoch_end)
    plot_end = actual_epochs if epoch_end is None else min(epoch_end, actual_epochs)
    plot_start = max(0, min(epoch_start, plot_end))
    plot_epochs = plot_end - plot_start
    # 使用 1-based 显示 epoch（从 1 开始）
    epoch_indices = np.arange(plot_start, plot_end) + 1
    
    # 打印统计信息
    print(f"实际训练的epoch数: {actual_epochs}, 绘制范围: [{plot_start}, {plot_end}) 共 {plot_epochs} 个 epoch")
    
    def _safe_max(arr, start, end):
        if arr is None or len(arr) == 0:
            return None
        arr = np.asarray(arr)[start:end]
        if arr.size == 0:
            return None
        return float(np.nanmax(arr))
    
    max_train_char = _safe_max(results.get("train_acc_char"), plot_start, plot_end)
    max_val_char = _safe_max(results.get("val_acc_char"), plot_start, plot_end)
    
    if max_train_char is not None:
        print(f"Train char acc max: {max_train_char:.2f}%")
    if max_val_char is not None:
        print(f"Val char acc max: {max_val_char:.2f}%")

    has_transition_metrics = (
        is_sector
        and results.get("glob_train_acc_char") is not None
        and len(np.asarray(results["glob_train_acc_char"])) > 0
    )

    # 创建图形：sector 模式为 2 行（acc + loss）；若含 glob / fg_switch 窗口指标则为 4 行
    nrows = 4 if has_transition_metrics else (2 if is_sector else 1)
    ncols = 2
    fig = plt.figure(figsize=(12, 5 * nrows))
    
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
        if 'rnn_dropout' in hparams:
            title_parts.append(f"rnn_dropout={format_number(hparams['rnn_dropout'])}")
        elif 'dropout' in hparams:
            title_parts.append(f"dropout={format_number(hparams['dropout'])}")
        if 'cnn_dropout' in hparams:
            title_parts.append(f"cnn_dropout={format_number(hparams['cnn_dropout'])}")
        if title_parts:
            plt.suptitle(' | '.join(title_parts), fontsize=14, fontweight='bold')
    
    # 第一行左：字符识别准确率
    plt.subplot(nrows, ncols, 1)
    plt.plot(epoch_indices, results["train_acc_char"][plot_start:plot_end], label="train accuracy", linewidth=2)
    plt.plot(epoch_indices, results["val_acc_char"][plot_start:plot_end], label="validation accuracy", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title("Character accuracy", fontsize=13)
    plt.ylim(-5, 105)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    
    # 第一行右：位置相关指标（acc 或 MSE）
    plt.subplot(nrows, ncols, 2)
    if is_allchars:
        # all-chars 模式：不绘制位置曲线
        plt.axis("off")
        plt.title("No position metrics (all-chars mode)", fontsize=13)
    elif is_sector:
        # sector 模式：第一行右 = sector 准确率
        if "train_acc_pos" in results and "val_acc_pos" in results:
            plt.plot(epoch_indices, results["train_acc_pos"][plot_start:plot_end], label="train sector acc", linewidth=2)
            plt.plot(epoch_indices, results["val_acc_pos"][plot_start:plot_end], label="val sector acc", linewidth=2)
            plt.xlabel("Epoch", fontsize=12)
            plt.ylabel("Accuracy (%)", fontsize=12)
            plt.title("Sector accuracy", fontsize=13)
            plt.ylim(40, 105)
            plt.legend(fontsize=10)
            plt.grid(alpha=0.3)
            
            max_train_pos = _safe_max(results.get("train_acc_pos"), plot_start, plot_end)
            max_val_pos = _safe_max(results.get("val_acc_pos"), plot_start, plot_end)
            if max_train_pos is not None:
                print(f"Train pos acc max: {max_train_pos:.2f}%")
            if max_val_pos is not None:
                print(f"Val pos acc max: {max_val_pos:.2f}%")
    else:
        # coordinate 模式：显示MSE
        if "train_err_pos" in results and "val_err_pos" in results:
            plt.plot(epoch_indices, results["train_err_pos"][plot_start:plot_end], label="train pos MSE", linewidth=2)
            plt.plot(epoch_indices, results["val_err_pos"][plot_start:plot_end], label="val pos MSE", linewidth=2)
            plt.xlabel("Epoch", fontsize=12)
            plt.ylabel("MSE (pixel^2)", fontsize=12)
            plt.title("Position error", fontsize=13)
            plt.legend(fontsize=10)
            plt.grid(alpha=0.3)
            
            # 打印最小MSE
            train_err = np.asarray(results["train_err_pos"][plot_start:plot_end])
            val_err = np.asarray(results["val_err_pos"][plot_start:plot_end])
            if train_err.size > 0:
                print(f"Train pos MSE min: {np.nanmin(train_err):.2f} pixel^2")
            if val_err.size > 0:
                print(f"Val pos MSE min: {np.nanmin(val_err):.2f} pixel^2")
    
    # sector 模式：第二行 = char loss（若未保存则占位）+ pos loss
    if is_sector:
        # 第二行左：Character loss（当前结果中未保存则显示占位）
        plt.subplot(nrows, ncols, 3)
        if "train_loss_char" in results and "val_loss_char" in results:
            plt.plot(epoch_indices, results["train_loss_char"][plot_start:plot_end], label="train char loss", linewidth=2)
            plt.plot(epoch_indices, results["val_loss_char"][plot_start:plot_end], label="val char loss", linewidth=2)
            plt.xlabel("Epoch", fontsize=12)
            plt.ylabel("Loss", fontsize=12)
            plt.title("Character loss", fontsize=13)
            plt.legend(fontsize=10)
            plt.grid(alpha=0.3)
        else:
            plt.axis("off")
            plt.title("Character loss (not saved)", fontsize=13)
        
        # 第二行右：Sector position CE loss
        plt.subplot(nrows, ncols, 4)
        if "train_loss_pos" in results and "val_loss_pos" in results:
            plt.plot(epoch_indices, results["train_loss_pos"][plot_start:plot_end], label="train sector loss", linewidth=2)
            plt.plot(epoch_indices, results["val_loss_pos"][plot_start:plot_end], label="val sector loss", linewidth=2)
            plt.xlabel("Epoch", fontsize=12)
            plt.ylabel("Loss (CE)", fontsize=12)
            plt.title("Sector position loss", fontsize=13)
            plt.legend(fontsize=10)
            plt.grid(alpha=0.3)
        else:
            plt.axis("off")
            plt.title("Sector position loss (not in this run)", fontsize=13)

    # sector + 严格全局 acc + fg_switch 前后窗 acc（第三、四行）
    if has_transition_metrics:
        plt.subplot(nrows, ncols, 5)
        plt.plot(
            epoch_indices,
            results["glob_train_acc_char"][plot_start:plot_end],
            label="train glob (char)",
            linewidth=2,
        )
        plt.plot(
            epoch_indices,
            results["glob_val_acc_char"][plot_start:plot_end],
            label="val glob (char)",
            linewidth=2,
        )
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Accuracy (%)", fontsize=12)
        plt.title("Character accuracy (global frame)", fontsize=13)
        plt.ylim(-5, 105)
        plt.legend(fontsize=9)
        plt.grid(alpha=0.3)

        plt.subplot(nrows, ncols, 6)
        plt.plot(
            epoch_indices,
            results["glob_train_acc_pos"][plot_start:plot_end],
            label="train glob (sector)",
            linewidth=2,
        )
        plt.plot(
            epoch_indices,
            results["glob_val_acc_pos"][plot_start:plot_end],
            label="val glob (sector)",
            linewidth=2,
        )
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Accuracy (%)", fontsize=12)
        plt.title("Sector accuracy (global frame)", fontsize=13)
        plt.ylim(40, 105)
        plt.legend(fontsize=9)
        plt.grid(alpha=0.3)

        plt.subplot(nrows, ncols, 7)
        plt.plot(
            epoch_indices,
            results["fg_switch_pre5_train_acc_char"][plot_start:plot_end],
            label="train pre5",
            linewidth=2,
        )
        plt.plot(
            epoch_indices,
            results["fg_switch_pre5_val_acc_char"][plot_start:plot_end],
            label="val pre5",
            linewidth=2,
            linestyle="--",
        )
        plt.plot(
            epoch_indices,
            results["fg_switch_post5_train_acc_char"][plot_start:plot_end],
            label="train post5",
            linewidth=2,
        )
        plt.plot(
            epoch_indices,
            results["fg_switch_post5_val_acc_char"][plot_start:plot_end],
            label="val post5",
            linewidth=2,
            linestyle="--",
        )
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Accuracy (%)", fontsize=12)
        plt.title("Fg switch windows — character", fontsize=13)
        plt.ylim(-5, 105)
        plt.legend(fontsize=8, loc="best")
        plt.grid(alpha=0.3)

        plt.subplot(nrows, ncols, 8)
        plt.plot(
            epoch_indices,
            results["fg_switch_pre5_train_acc_pos"][plot_start:plot_end],
            label="train pre5",
            linewidth=2,
        )
        plt.plot(
            epoch_indices,
            results["fg_switch_pre5_val_acc_pos"][plot_start:plot_end],
            label="val pre5",
            linewidth=2,
            linestyle="--",
        )
        plt.plot(
            epoch_indices,
            results["fg_switch_post5_train_acc_pos"][plot_start:plot_end],
            label="train post5",
            linewidth=2,
        )
        plt.plot(
            epoch_indices,
            results["fg_switch_post5_val_acc_pos"][plot_start:plot_end],
            label="val post5",
            linewidth=2,
            linestyle="--",
        )
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Accuracy (%)", fontsize=12)
        plt.title("Fg switch windows — sector", fontsize=13)
        plt.ylim(40, 105)
        plt.legend(fontsize=8, loc="best")
        plt.grid(alpha=0.3)

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
    
    # cnn / rnn dropout (checkpoint stem); legacy _do = single shared p
    cdo_match = re.search(r'_cdo([\d.]+)(?:_|\.|$)', filename)
    if cdo_match:
        hparams['cnn_dropout'] = float(cdo_match.group(1))
    rdo_match = re.search(r'_rdo([\d.]+)(?:_|\.|$)', filename)
    if rdo_match:
        hparams['rnn_dropout'] = float(rdo_match.group(1))
    do_match = re.search(r'_do([\d.]+)(?:_|\.|$)', filename)
    if do_match and 'rnn_dropout' not in hparams:
        v = float(do_match.group(1))
        hparams['dropout'] = v
        hparams['rnn_dropout'] = v
        if 'cnn_dropout' not in hparams:
            hparams['cnn_dropout'] = v

    return hparams


def main():
    parser = argparse.ArgumentParser(description='可视化单个模型的训练曲线')
    parser.add_argument('pkl_path', type=str, help='结果pkl文件路径')
    parser.add_argument('--output', type=str, default=None, 
                       help='输出图片路径（默认：results/train_figs/<basename>.png）')
    parser.add_argument('--output_dir', type=str, default='results/train_figs',
                       help='输出目录（默认：results/train_figs）')
    parser.add_argument('--epoch_start', type=int, default=0,
                       help='绘制的起始 epoch（0-based，含）。默认 0')
    parser.add_argument('--epoch_end', type=int, default=None,
                       help='绘制的结束 epoch（0-based，不含）；不指定则画到最后一个 epoch。例如 100 表示只画 0~99')
    
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
    visualize_training_curves(
        str(pkl_path), str(output_path), hparams,
        epoch_start=args.epoch_start, epoch_end=args.epoch_end,
    )
    
    print(f"\n{'=' * 60}")
    print(f"完成！")
    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    main()


