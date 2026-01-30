"""
超参数搜索结果分析脚本
用于横向比较不同超参数组合的性能
"""
import os
import sys
import pickle
import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def parse_hparams_from_filename(filename):
    """从文件名解析超参数"""
    hparams = {}
    
    # 提取 model_type
    if filename.startswith('rnn_'):
        hparams['model_type'] = 'rnn'
    elif filename.startswith('lstm_'):
        hparams['model_type'] = 'lstm'
    elif filename.startswith('gru_'):
        hparams['model_type'] = 'gru'
    elif filename.startswith('gawf_'):
        hparams['model_type'] = 'gawf'
    
    # 提取 mode
    if 'sector' in filename:
        hparams['mode'] = 'sector'
    elif 'coord' in filename:
        hparams['mode'] = 'coord'
    elif 'allchar' in filename:
        hparams['mode'] = 'allchar'
    
    # 提取 hidden_size
    h_match = re.search(r'_h(\d+)', filename)
    if h_match:
        hparams['hidden_size'] = int(h_match.group(1))
    
    # 提取 lr (贪婪匹配)
    lr_match = re.search(r'_lr([\deE.+-]+)_', filename)
    if lr_match:
        hparams['lr'] = float(lr_match.group(1))
    
    # 提取 wd
    wd_match = re.search(r'_wd([\deE.+-]+)_', filename)
    if wd_match:
        hparams['wd'] = float(wd_match.group(1))
    
    # 提取 dropout
    do_match = re.search(r'_do([\d.]+)(?:_|\.)', filename)
    if do_match:
        hparams['dropout'] = float(do_match.group(1))
    
    return hparams


def load_results(results_dir):
    """加载所有pkl文件并提取关键指标"""
    results_dir = Path(results_dir)
    pkl_files = list(results_dir.glob('*.pkl'))
    
    all_results = []
    
    for pkl_file in pkl_files:
        # 跳过临时文件
        if pkl_file.name.startswith('.') or pkl_file.name.endswith('~'):
            continue
        
        try:
            # 加载pkl文件
            with open(pkl_file, 'rb') as f:
                results = pickle.load(f)
            
            # 解析超参数
            hparams = parse_hparams_from_filename(pkl_file.name)
            
            # 获取实际训练的epoch数
            if "actual_epochs" in results:
                actual_epochs = results["actual_epochs"]
            else:
                train_acc = results.get("train_acc_char", [])
                actual_epochs = len(train_acc)
            
            # 提取指标
            train_acc_char = np.array(results.get("train_acc_char", []))[:actual_epochs]
            val_acc_char = np.array(results.get("val_acc_char", []))[:actual_epochs]
            
            if len(train_acc_char) == 0 or len(val_acc_char) == 0:
                print(f"Warning: {pkl_file.name} has no data, skipping")
                continue
            
            # 计算关键指标
            record = {
                'filename': pkl_file.name,
                'model_type': hparams.get('model_type', 'unknown'),
                'mode': hparams.get('mode', 'unknown'),
                'hidden_size': hparams.get('hidden_size', 0),
                'lr': hparams.get('lr', 0),
                'wd': hparams.get('wd', 0),
                'dropout': hparams.get('dropout', 0),
                'actual_epochs': actual_epochs,
                'train_acc_max': float(np.max(train_acc_char)),
                'train_acc_final': float(train_acc_char[-1]),
                'val_acc_max': float(np.max(val_acc_char)),
                'val_acc_final': float(val_acc_char[-1]),
            }
            
            # 计算过拟合gap（train - val）
            record['overfitting_gap_max'] = record['train_acc_max'] - record['val_acc_max']
            record['overfitting_gap_final'] = record['train_acc_final'] - record['val_acc_final']
            
            all_results.append(record)
            
        except Exception as e:
            print(f"Error loading {pkl_file.name}: {e}")
            continue
    
    return pd.DataFrame(all_results)


def generate_summary_table(df, output_path):
    """生成汇总表格"""
    # 按 val_acc_max 降序排序
    df_sorted = df.sort_values('val_acc_max', ascending=False)
    
    # 选择关键列
    columns = ['model_type', 'hidden_size', 'lr', 'wd', 'dropout', 
               'train_acc_max', 'val_acc_max', 'train_acc_final', 'val_acc_final',
               'overfitting_gap_final', 'actual_epochs']
    
    df_output = df_sorted[columns].copy()
    
    # 格式化数值
    df_output['lr'] = df_output['lr'].apply(lambda x: f'{x:.4f}')
    df_output['wd'] = df_output['wd'].apply(lambda x: f'{x:.6f}')
    df_output['dropout'] = df_output['dropout'].apply(lambda x: f'{x:.2f}')
    
    for col in ['train_acc_max', 'val_acc_max', 'train_acc_final', 'val_acc_final', 'overfitting_gap_final']:
        df_output[col] = df_output[col].apply(lambda x: f'{x:.2f}')
    
    # 保存CSV
    df_output.to_csv(output_path, index=False)
    print(f"\n✓ Summary table saved to: {output_path}")
    
    # 打印到控制台
    print("\n" + "="*80)
    print("HYPERPARAMETER SEARCH SUMMARY (Top 10 by val_acc_max)")
    print("="*80)
    print(df_output.head(10).to_string(index=False))
    print("="*80 + "\n")


def plot_hparam_comparison(df, output_dir):
    """生成超参数对比图"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 只分析coord模式的RNN模型
    df_filtered = df[(df['model_type'] == 'rnn') & (df['mode'] == 'coord')].copy()
    
    if len(df_filtered) == 0:
        print("Warning: No RNN coord models found for comparison")
        return
    
    # 1. 固定 wd 和 dropout，比较不同 lr 的影响
    # 找到最常见的 wd 和 dropout 组合
    common_wd = df_filtered['wd'].mode().iloc[0] if len(df_filtered['wd'].mode()) > 0 else 0.0001
    common_dropout = df_filtered['dropout'].mode().iloc[0] if len(df_filtered['dropout'].mode()) > 0 else 0.3
    
    df_lr_compare = df_filtered[
        (df_filtered['wd'] == common_wd) & 
        (df_filtered['dropout'] == common_dropout)
    ].sort_values('lr')
    
    if len(df_lr_compare) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(df_lr_compare))
        width = 0.35
        
        ax.bar(x - width/2, df_lr_compare['train_acc_final'], width, label='Train Acc', alpha=0.8)
        ax.bar(x + width/2, df_lr_compare['val_acc_final'], width, label='Val Acc', alpha=0.8)
        
        ax.set_xlabel('Learning Rate', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title(f'Learning Rate Comparison (wd={common_wd:.6f}, dropout={common_dropout:.2f})', 
                     fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{lr:.4f}' for lr in df_lr_compare['lr']], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'lr_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ LR comparison plot saved to: {output_dir / 'lr_comparison.png'}")
    
    # 2. 固定 lr 和 dropout，比较不同 wd 的影响
    common_lr = df_filtered['lr'].mode().iloc[0] if len(df_filtered['lr'].mode()) > 0 else 0.001
    
    df_wd_compare = df_filtered[
        (df_filtered['lr'] == common_lr) & 
        (df_filtered['dropout'] == common_dropout)
    ].sort_values('wd')
    
    if len(df_wd_compare) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(df_wd_compare))
        width = 0.35
        
        ax.bar(x - width/2, df_wd_compare['train_acc_final'], width, label='Train Acc', alpha=0.8)
        ax.bar(x + width/2, df_wd_compare['val_acc_final'], width, label='Val Acc', alpha=0.8)
        
        ax.set_xlabel('Weight Decay', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title(f'Weight Decay Comparison (lr={common_lr:.4f}, dropout={common_dropout:.2f})', 
                     fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{wd:.1e}' for wd in df_wd_compare['wd']], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'wd_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ WD comparison plot saved to: {output_dir / 'wd_comparison.png'}")
    
    # 3. 固定 lr 和 wd，比较不同 dropout 的影响
    df_dropout_compare = df_filtered[
        (df_filtered['lr'] == common_lr) & 
        (df_filtered['wd'] == common_wd)
    ].sort_values('dropout')
    
    if len(df_dropout_compare) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(df_dropout_compare))
        width = 0.35
        
        ax.bar(x - width/2, df_dropout_compare['train_acc_final'], width, label='Train Acc', alpha=0.8)
        ax.bar(x + width/2, df_dropout_compare['val_acc_final'], width, label='Val Acc', alpha=0.8)
        
        ax.set_xlabel('Dropout Rate', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title(f'Dropout Comparison (lr={common_lr:.4f}, wd={common_wd:.6f})', 
                     fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{do:.2f}' for do in df_dropout_compare['dropout']], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'dropout_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Dropout comparison plot saved to: {output_dir / 'dropout_comparison.png'}")


def plot_training_curves_overlay(results_dir, df, output_dir):
    """叠加绘制训练曲线，便于直接对比"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(results_dir)
    
    # 只分析coord模式的RNN模型
    df_filtered = df[(df['model_type'] == 'rnn') & (df['mode'] == 'coord')].copy()
    
    if len(df_filtered) == 0:
        return
    
    # 检测是 sector 还是 coord 模式
    is_sector = df_filtered['mode'].iloc[0] == 'sector' if len(df_filtered) > 0 else False
    
    # 找到baseline参数
    common_lr = df_filtered['lr'].mode().iloc[0] if len(df_filtered['lr'].mode()) > 0 else 0.001
    common_wd = df_filtered['wd'].mode().iloc[0] if len(df_filtered['wd'].mode()) > 0 else 0.0001
    common_dropout = df_filtered['dropout'].mode().iloc[0] if len(df_filtered['dropout'].mode()) > 0 else 0.3
    
    # 定义颜色列表
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # 1. 比较不同lr的训练曲线
    df_lr_vary = df_filtered[
        (df_filtered['wd'] == common_wd) & 
        (df_filtered['dropout'] == common_dropout)
    ].sort_values('lr')
    
    if len(df_lr_vary) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        for idx, (_, row) in enumerate(df_lr_vary.iterrows()):
            pkl_file = results_dir / row['filename']
            with open(pkl_file, 'rb') as f:
                results = pickle.load(f)
            
            epochs = row['actual_epochs']
            train_acc = np.array(results['train_acc_char'])[:epochs]
            val_acc = np.array(results['val_acc_char'])[:epochs]
            
            color = colors[idx % len(colors)]
            label = f"lr={row['lr']:.4f}, h={int(row['hidden_size'])}"
            
            # 左图：字符准确率（train实线，val虚线）
            ax1.plot(train_acc, label=f'{label} (train)', linewidth=2, color=color, linestyle='-')
            ax1.plot(val_acc, label=f'{label} (val)', linewidth=2, color=color, linestyle='--', alpha=0.8)
            
            # 右图：位置指标
            if is_sector:
                # Sector mode: 绘制位置准确率
                if 'train_acc_pos' in results and 'val_acc_pos' in results:
                    train_pos = np.array(results['train_acc_pos'])[:epochs]
                    val_pos = np.array(results['val_acc_pos'])[:epochs]
                    ax2.plot(train_pos, linewidth=2, color=color, linestyle='-')
                    ax2.plot(val_pos, linewidth=2, color=color, linestyle='--', alpha=0.8)
            else:
                # Coord mode: 绘制位置MSE
                if 'train_err_pos' in results and 'val_err_pos' in results:
                    train_mse = np.array(results['train_err_pos'])[:epochs]
                    val_mse = np.array(results['val_err_pos'])[:epochs]
                    ax2.plot(train_mse, linewidth=2, color=color, linestyle='-')
                    ax2.plot(val_mse, linewidth=2, color=color, linestyle='--', alpha=0.8)
        
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy (%)', fontsize=12)
        ax1.set_title('Character Accuracy', fontsize=13)
        ax1.legend(fontsize=8, loc='best')
        ax1.grid(alpha=0.3)
        
        ax2.set_xlabel('Epoch', fontsize=12)
        if is_sector:
            ax2.set_ylabel('Accuracy (%)', fontsize=12)
            ax2.set_title('Sector Accuracy', fontsize=13)
            ax2.set_ylim(40, 105)
        else:
            ax2.set_ylabel('MSE (pixel²)', fontsize=12)
            ax2.set_title('Position Error (MSE)', fontsize=13)
        ax2.grid(alpha=0.3)
        
        plt.suptitle(f'Learning Rate Comparison (wd={common_wd:.6f}, dropout={common_dropout:.2f})', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'lr_curves_overlay.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ LR curves overlay saved to: {output_dir / 'lr_curves_overlay.png'}")
    
    # 2. 比较不同 weight decay 的训练曲线
    df_wd_vary = df_filtered[
        (df_filtered['lr'] == common_lr) & 
        (df_filtered['dropout'] == common_dropout)
    ].sort_values('wd')
    
    if len(df_wd_vary) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        for idx, (_, row) in enumerate(df_wd_vary.iterrows()):
            pkl_file = results_dir / row['filename']
            with open(pkl_file, 'rb') as f:
                results = pickle.load(f)
            
            epochs = row['actual_epochs']
            train_acc = np.array(results['train_acc_char'])[:epochs]
            val_acc = np.array(results['val_acc_char'])[:epochs]
            
            color = colors[idx % len(colors)]
            label = f"wd={row['wd']:.1e}, h={int(row['hidden_size'])}"
            
            # 左图：字符准确率
            ax1.plot(train_acc, label=f'{label} (train)', linewidth=2, color=color, linestyle='-')
            ax1.plot(val_acc, label=f'{label} (val)', linewidth=2, color=color, linestyle='--', alpha=0.8)
            
            # 右图：位置指标
            if is_sector:
                if 'train_acc_pos' in results and 'val_acc_pos' in results:
                    train_pos = np.array(results['train_acc_pos'])[:epochs]
                    val_pos = np.array(results['val_acc_pos'])[:epochs]
                    ax2.plot(train_pos, linewidth=2, color=color, linestyle='-')
                    ax2.plot(val_pos, linewidth=2, color=color, linestyle='--', alpha=0.8)
            else:
                if 'train_err_pos' in results and 'val_err_pos' in results:
                    train_mse = np.array(results['train_err_pos'])[:epochs]
                    val_mse = np.array(results['val_err_pos'])[:epochs]
                    ax2.plot(train_mse, linewidth=2, color=color, linestyle='-')
                    ax2.plot(val_mse, linewidth=2, color=color, linestyle='--', alpha=0.8)
        
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy (%)', fontsize=12)
        ax1.set_title('Character Accuracy', fontsize=13)
        ax1.legend(fontsize=8, loc='best')
        ax1.grid(alpha=0.3)
        
        ax2.set_xlabel('Epoch', fontsize=12)
        if is_sector:
            ax2.set_ylabel('Accuracy (%)', fontsize=12)
            ax2.set_title('Sector Accuracy', fontsize=13)
            ax2.set_ylim(40, 105)
        else:
            ax2.set_ylabel('MSE (pixel²)', fontsize=12)
            ax2.set_title('Position Error (MSE)', fontsize=13)
        ax2.grid(alpha=0.3)
        
        plt.suptitle(f'Weight Decay Comparison (lr={common_lr:.4f}, dropout={common_dropout:.2f})', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'wd_curves_overlay.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ WD curves overlay saved to: {output_dir / 'wd_curves_overlay.png'}")
    
    # 3. 比较不同 dropout 的训练曲线
    df_dropout_vary = df_filtered[
        (df_filtered['lr'] == common_lr) & 
        (df_filtered['wd'] == common_wd)
    ].sort_values('dropout')
    
    if len(df_dropout_vary) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        for idx, (_, row) in enumerate(df_dropout_vary.iterrows()):
            pkl_file = results_dir / row['filename']
            with open(pkl_file, 'rb') as f:
                results = pickle.load(f)
            
            epochs = row['actual_epochs']
            train_acc = np.array(results['train_acc_char'])[:epochs]
            val_acc = np.array(results['val_acc_char'])[:epochs]
            
            color = colors[idx % len(colors)]
            label = f"dropout={row['dropout']:.2f}, h={int(row['hidden_size'])}"
            
            # 左图：字符准确率
            ax1.plot(train_acc, label=f'{label} (train)', linewidth=2, color=color, linestyle='-')
            ax1.plot(val_acc, label=f'{label} (val)', linewidth=2, color=color, linestyle='--', alpha=0.8)
            
            # 右图：位置指标
            if is_sector:
                if 'train_acc_pos' in results and 'val_acc_pos' in results:
                    train_pos = np.array(results['train_acc_pos'])[:epochs]
                    val_pos = np.array(results['val_acc_pos'])[:epochs]
                    ax2.plot(train_pos, linewidth=2, color=color, linestyle='-')
                    ax2.plot(val_pos, linewidth=2, color=color, linestyle='--', alpha=0.8)
            else:
                if 'train_err_pos' in results and 'val_err_pos' in results:
                    train_mse = np.array(results['train_err_pos'])[:epochs]
                    val_mse = np.array(results['val_err_pos'])[:epochs]
                    ax2.plot(train_mse, linewidth=2, color=color, linestyle='-')
                    ax2.plot(val_mse, linewidth=2, color=color, linestyle='--', alpha=0.8)
        
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy (%)', fontsize=12)
        ax1.set_title('Character Accuracy', fontsize=13)
        ax1.legend(fontsize=8, loc='best')
        ax1.grid(alpha=0.3)
        
        ax2.set_xlabel('Epoch', fontsize=12)
        if is_sector:
            ax2.set_ylabel('Accuracy (%)', fontsize=12)
            ax2.set_title('Sector Accuracy', fontsize=13)
            ax2.set_ylim(40, 105)
        else:
            ax2.set_ylabel('MSE (pixel²)', fontsize=12)
            ax2.set_title('Position Error (MSE)', fontsize=13)
        ax2.grid(alpha=0.3)
        
        plt.suptitle(f'Dropout Comparison (lr={common_lr:.4f}, wd={common_wd:.6f})', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'dropout_curves_overlay.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Dropout curves overlay saved to: {output_dir / 'dropout_curves_overlay.png'}")


def main():
    parser = argparse.ArgumentParser(description='分析超参数搜索结果')
    parser.add_argument('results_dir', type=str, help='结果目录路径（包含pkl文件）')
    parser.add_argument('--output_dir', type=str, default='results/analysis',
                       help='输出目录（默认：results/analysis）')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"Analyzing hyperparameter search results")
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")
    
    # 加载所有结果
    print("Loading results...")
    df = load_results(results_dir)
    
    if len(df) == 0:
        print("Error: No valid results found")
        sys.exit(1)
    
    print(f"✓ Loaded {len(df)} models\n")
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成汇总表格
    print("Generating summary table...")
    generate_summary_table(df, output_dir / 'summary.csv')
    
    # 生成对比图
    print("\nGenerating comparison plots...")
    plot_hparam_comparison(df, output_dir)
    
    # 生成叠加曲线图
    print("\nGenerating overlay curves...")
    plot_training_curves_overlay(results_dir, df, output_dir)
    
    print(f"\n{'='*80}")
    print(f"Analysis complete!")
    print(f"All results saved to: {output_dir}")
    print(f"{'='*80}\n")
    
    # 推荐最佳超参数
    best_model = df.loc[df['val_acc_max'].idxmax()]
    print("\n🏆 BEST MODEL (by val_acc_max):")
    print(f"  Model: {best_model['model_type'].upper()}")
    print(f"  Hidden size: {best_model['hidden_size']}")
    print(f"  Learning rate: {best_model['lr']:.4f}")
    print(f"  Weight decay: {best_model['wd']:.6f}")
    print(f"  Dropout: {best_model['dropout']:.2f}")
    print(f"  Val accuracy (max): {best_model['val_acc_max']:.2f}%")
    print(f"  Val accuracy (final): {best_model['val_acc_final']:.2f}%")
    print(f"  Overfitting gap: {best_model['overfitting_gap_final']:.2f}%\n")


if __name__ == '__main__':
    main()

