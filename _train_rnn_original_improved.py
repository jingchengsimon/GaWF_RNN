"""
Improved RNN Sector training script - 改进版训练脚本
基于 train_rnn_sector_original.py，加入以下改进来修复 accuracy 周期性下降问题：

主要改进：
1. 启用 Early Stopping - 防止过拟合
2. 增加 Weight Decay - 正则化权重
3. 添加学习率调度器 - 自动降低学习率
4. 添加 Label Smoothing - 减少过度自信
5. 在分类器层添加 Dropout - 增强泛化能力
6. 增加 middle 层的 Dropout 率 - 更强的正则化
"""
import os
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ==================== Dataset Class ====================
class MC_RNN_Dataset(Dataset):
    def __init__(self, data, labels, frame_num=32, chan_num=2, use_sector=False, num_sectors=9):
        """
        简化版数据集，只支持 sector 和 coordinate 模式
        """
        self.data = data
        self.frame_num = frame_num
        self.chan_num = chan_num
        self.use_sector = use_sector
        self.num_sectors = num_sectors
        self.labels = labels[['fg_char_id', 'fg_char_x', 'fg_char_y']].values

    def __len__(self):
        return (self.data.shape[0]-self.chan_num) // self.frame_num

    def __getitem__(self, idx):
        start_idx = (idx * self.frame_num) + self.chan_num
        end_idx = start_idx + self.frame_num

        # Stack frames
        for i in range(-(self.chan_num-1), 1):
            if i == -(self.chan_num-1):
                stacked_frames = np.expand_dims(self.data[(start_idx + i):(end_idx + i)], axis=1)
            else:
                stacked_frames = np.concatenate((stacked_frames,
                                                 np.expand_dims(self.data[(start_idx + i):(end_idx + i)],
                                                                axis=1)), axis=1)
        stacked_frames = stacked_frames.astype(np.float32)

        labels = self.labels[start_idx:end_idx].copy()

        if self.use_sector:
            # Map (x, y) to sector
            height = self.data.shape[-2]
            width = self.data.shape[-1]
            grid_size = int(np.sqrt(self.num_sectors))
            
            x = labels[:, 1].astype(np.float32)
            y = labels[:, 2].astype(np.float32)
            
            col = (x / max(width - 1, 1) * grid_size).astype(np.int64)
            row = (y / max(height - 1, 1) * grid_size).astype(np.int64)
            
            col = np.clip(col, 0, grid_size - 1)
            row = np.clip(row, 0, grid_size - 1)
            
            sector = row * grid_size + col
            labels = np.stack([labels[:, 0].astype(np.int64), sector], axis=1)

        return stacked_frames, labels


# ==================== Model Classes ====================
class RNNConv(nn.Module):
    def __init__(self, num_classes, num_pos, kernel_size=3, device='cuda', hidden_size=256, 
                 classifier_dropout=0.3, middle_dropout=0.6):
        """
        改进版 RNN 模型
        
        新增参数：
            classifier_dropout: 分类器层的 dropout 率（默认 0.3）
            middle_dropout: middle 层的 dropout 率（默认 0.6，原始为 0.5）
        """
        super(RNNConv, self).__init__()
        self.device = device
        self.classifier_dropout = classifier_dropout
        self.middle_dropout = middle_dropout
        
        self.conv1 = nn.Conv2d(2, 32, kernel_size=kernel_size, padding='same')
        self.MP1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.LNorm1 = nn.LayerNorm([32, 48, 48])
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.MP2 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.LNorm2 = nn.LayerNorm([64, 12, 12])
        self.rnn = nn.RNN(input_size=64 * 12 * 12, hidden_size=hidden_size,
                          num_layers=1, batch_first=True)
        self.LNormRNN = nn.LayerNorm(hidden_size)
        self.fcchar = nn.Linear(hidden_size, num_classes)
        self.fcpos = nn.Linear(hidden_size, num_pos)
        self.to(self.device)

    def encoder(self, x):
        """
        Encoder 保持原样，不添加 dropout
        """
        x = self.conv1(x)
        x = self.MP1(x)
        x = self.LNorm1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.MP2(x)
        x = self.LNorm2(x)
        x = F.relu(x)
        return x

    def middle(self, x):
        """改进：增加 dropout 率从 0.5 到 0.6"""
        x = self.rnn(x)[0]
        x = self.LNormRNN(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.middle_dropout, training=self.training)
        return x

    def classifier(self, x):
        """改进：在分类器前添加 dropout"""
        x = F.dropout(x, p=self.classifier_dropout, training=self.training)
        return self.fcchar(x), self.fcpos(x)

    def forward(self, x):
        x = x.to(self.device)
        batch_size, frame_num, channels, height, width = x.size()
        x = x.view(batch_size * frame_num, channels, height, width)
        x = self.encoder(x)
        x = x.view(batch_size, frame_num, -1)
        x = self.middle(x)
        char_out, pos_out = self.classifier(x)
        return char_out, pos_out


# ==================== Training Function ====================
def network_train(mdl, train_data, val_data, num_epochs=50, loss_weights=None, lr=0.001, 
                  batch_size=256, weight_decay=5e-4, use_early_stopping=True, 
                  early_stopping_patience=10, min_delta=0.001, label_smoothing=0.1, 
                  use_lr_scheduler=True):
    """
    改进的训练函数
    
    新增参数：
        weight_decay: L2 正则化系数（默认 5e-4）
        use_early_stopping: 是否使用 early stopping（默认 True）
        early_stopping_patience: early stopping 的耐心值（默认 10）
        min_delta: 最小改进阈值（默认 0.001）
        label_smoothing: label smoothing 系数（默认 0.1）
        use_lr_scheduler: 是否使用学习率调度器（默认 True）
    """
    use_sector = train_data.use_sector
    device = mdl.device
    mdl.to(device)
    
    # 设置默认 loss weights
    if loss_weights is None:
        if use_sector:
            loss_weights = [1, 1]  # sector mode
        else:
            loss_weights = [1, 0.001]  # coordinate mode
    
    # 改进1：添加 weight_decay
    optim = torch.optim.Adam(mdl.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 改进2：添加 label smoothing
    criterion_char = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    if use_sector:
        criterion_pos = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    else:
        criterion_pos = nn.MSELoss()
    
    # 改进3：添加学习率调度器
    scheduler = None
    if use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, 
            mode='max',  # 监控验证准确率（越大越好）
            factor=0.5,  # 学习率衰减因子
            patience=5,  # 5个epoch无改善则降低学习率
            min_lr=1e-6  # 最小学习率
        )
    
    def loss_fn(out_char, out_pos, labels):
        labels_char = labels[:, :, 0].long().view(-1)
        outputs_char = out_char.view(-1, out_char.shape[-1])
        loss_char = criterion_char(outputs_char, labels_char)
        
        if use_sector:
            labels_pos = labels[:, :, 1].long().view(-1)
            outputs_pos = out_pos.view(-1, out_pos.shape[-1])
            loss_pos = criterion_pos(outputs_pos, labels_pos)
        else:
            labels_pos = labels[:, :, 1:].float()
            outputs_pos = out_pos
            loss_pos = criterion_pos(outputs_pos, labels_pos)

        rnn_hh = mdl.rnn.weight_hh_l0
        rnn_hh_diag = torch.diagonal(rnn_hh).abs().sum()
        
        loss = (loss_weights[0] * loss_char) + (loss_weights[1] * loss_pos) + rnn_hh_diag
        return loss

    def evaluate(mdl, data_loader):
        mdl.eval()
        total_acc_char = 0
        total_metric_pos = 0
        with torch.no_grad():
            for batch in data_loader:
                inputs, labels = batch
                labels = labels.to(device)
                out_char, out_pos = mdl(inputs)
                total_acc_char += (torch.argmax(out_char, dim=2) == labels[:, :, 0].long()).float().mean().item()
                
                if use_sector:
                    total_metric_pos += (torch.argmax(out_pos, dim=2) == labels[:, :, 1].long()).float().mean().item()
                else:
                    labels_pos = labels[:, :, 1:].float()
                    total_metric_pos += F.mse_loss(out_pos, labels_pos, reduction='mean').item()
        
        acc_char = total_acc_char * 100 / len(data_loader)
        if use_sector:
            metric_pos = total_metric_pos * 100 / len(data_loader)
        else:
            metric_pos = total_metric_pos / len(data_loader)
        return acc_char, metric_pos

    # Data loaders
    train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    print(f"\nTraining Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Label smoothing: {label_smoothing}")
    print(f"  Classifier dropout: {mdl.classifier_dropout}")
    print(f"  Middle dropout: {mdl.middle_dropout}")
    print(f"  Early stopping: {use_early_stopping} (patience={early_stopping_patience})")
    print(f"  LR scheduler: {use_lr_scheduler}")
    print(f"  Mode: {'sector' if use_sector else 'coordinate'}")
    print()
    
    train_acc_char = []
    val_acc_char = []
    train_metric_pos = []
    val_metric_pos = []
    
    # 改进4：Early stopping 变量
    best_val_acc = 0.0
    epochs_no_improve = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        mdl.train()
        epoch_train_acc = 0
        epoch_train_pos = 0
        num_batches = 0
        
        for batch in train_dl:
            inputs, labels = batch
            labels = labels.to(device)
            
            optim.zero_grad()
            out_char, out_pos = mdl(inputs)
            loss = loss_fn(out_char, out_pos, labels)
            loss.backward()
            # Gradient clipping (standard training)
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), max_norm=2.0)
            optim.step()
            
            epoch_train_acc += (torch.argmax(out_char, dim=2) == labels[:, :, 0].long()).float().mean().item()
            
            if use_sector:
                epoch_train_pos += (torch.argmax(out_pos, dim=2) == labels[:, :, 1].long()).float().mean().item()
            else:
                labels_pos = labels[:, :, 1:].float()
                epoch_train_pos += F.mse_loss(out_pos, labels_pos, reduction='mean').item()
            
            num_batches += 1
        
        train_acc_char.append((epoch_train_acc / num_batches) * 100)
        if use_sector:
            train_metric_pos.append((epoch_train_pos / num_batches) * 100)
        else:
            train_metric_pos.append(epoch_train_pos / num_batches)
        
        # Validation
        with torch.no_grad():
            val_acc, val_metric = evaluate(mdl, val_dl)
            val_acc_char.append(val_acc)
            val_metric_pos.append(val_metric)
            
            if use_sector:
                train_str = f"Epoch {epoch + 1}/{num_epochs} - Train (char, sector): ({train_acc_char[-1]:.2f}%, {train_metric_pos[-1]:.2f}%)"
                val_str = f" Val (char, sector): ({val_acc_char[-1]:.2f}%, {val_metric_pos[-1]:.2f}%)"
            else:
                train_str = f"Epoch {epoch + 1}/{num_epochs} - Train (char, pos): ({train_acc_char[-1]:.2f}%, {train_metric_pos[-1]:.2f} pix^2)"
                val_str = f" Val (char, pos): ({val_acc_char[-1]:.2f}%, {val_metric_pos[-1]:.2f} pix^2)"
            print(train_str, val_str)
        
        # 改进5：学习率调度器
        if scheduler is not None:
            # 使用验证准确率作为监控指标
            if use_sector:
                monitor_metric = (val_acc_char[-1] + val_metric_pos[-1]) / 2.0
            else:
                monitor_metric = val_acc_char[-1]
            
            old_lr = optim.param_groups[0]['lr']
            scheduler.step(monitor_metric)
            new_lr = optim.param_groups[0]['lr']
            
            if new_lr < old_lr:
                print(f"  → Learning rate reduced: {old_lr:.2e} → {new_lr:.2e}")
        
        # 改进6：Early stopping 逻辑
        if use_early_stopping:
            current_val_acc = val_acc_char[-1]
            
            if current_val_acc > best_val_acc + min_delta:
                best_val_acc = current_val_acc
                epochs_no_improve = 0
                # 保存最佳模型状态
                best_model_state = {k: v.cpu().clone() for k, v in mdl.state_dict().items()}
                print(f"  → New best validation accuracy: {best_val_acc:.2f}%")
            else:
                epochs_no_improve += 1
                print(f"  → No improvement for {epochs_no_improve} epoch(s)")
            
            if epochs_no_improve >= early_stopping_patience:
                print(f"\n🛑 Early stopping triggered at epoch {epoch + 1}")
                print(f"   Best validation accuracy: {best_val_acc:.2f}%")
                # 恢复最佳模型
                if best_model_state is not None:
                    mdl.load_state_dict(best_model_state)
                    print(f"   Restored best model from epoch {epoch + 1 - epochs_no_improve}")
                break
    
    torch.cuda.empty_cache()
    
    # 转换为 numpy 数组
    train_acc_char = np.array(train_acc_char)
    val_acc_char = np.array(val_acc_char)
    train_metric_pos = np.array(train_metric_pos)
    val_metric_pos = np.array(val_metric_pos)
    
    if use_sector:
        return {
            "train_acc_char": train_acc_char,
            "val_acc_char": val_acc_char,
            "train_acc_pos": train_metric_pos,
            "val_acc_pos": val_metric_pos,
            "model": mdl.to("cpu"),
            "actual_epochs": len(train_acc_char),
            "best_val_acc": best_val_acc
        }
    else:
        return {
            "train_acc_char": train_acc_char,
            "val_acc_char": val_acc_char,
            "train_err_pos": train_metric_pos,
            "val_err_pos": val_metric_pos,
            "model": mdl.to("cpu"),
            "actual_epochs": len(train_acc_char),
            "best_val_acc": best_val_acc
        }


# ==================== Helper Functions ====================
def save_results(results, filepath):
    """Save training results"""
    import os
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    results_path = filepath + '.pkl'
    save_dict = {k: v for k, v in results.items() if k != "model"}
    
    model_path = results_path.replace('.pkl', '_model.pth')
    if "model" in results:
        torch.save(results["model"].state_dict(), model_path)
        print(f"Model saved to: {model_path}")
    
    with open(results_path, 'wb') as f:
        pickle.dump(save_dict, f)
    print(f"Results saved to: {results_path}")


def get_base_path():
    """Detect environment and return data path"""
    is_wsl = os.name == "posix" and os.path.exists("/mnt/c")
    is_linux = os.name == "posix" and not os.path.exists("/mnt/c")
    
    if is_linux:
        base_path = "/G/MIMOlab/Codes/aim3_RNN/stimuli"
        print(f"Linux environment: {base_path}")
    else:
        base_path = r"C:\Users\12265\Desktop\SJC\archive\Aim3\stimuli"
        if is_wsl:
            base_path = "/mnt/c/Users/12265/Desktop/SJC/archive/Aim3/stimuli"
        print(f"WSL/Windows environment: {base_path}")
    
    return base_path


# ==================== Main ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="改进版 RNN Sector 训练脚本 - 修复 accuracy 周期性下降")
    parser.add_argument("--model_types", nargs="+", default=["rnn"], choices=["rnn"])
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[128])
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--use_sector_mode", action="store_true", default=False)
    parser.add_argument("--result_suffix", type=str, default="original_improved")
    
    # 新增改进参数
    parser.add_argument("--weight_decay", type=float, default=0, 
                       help="Weight decay (L2 regularization)")
    parser.add_argument("--label_smoothing", type=float, default=0,
                       help="Label smoothing coefficient")
    parser.add_argument("--classifier_dropout", type=float, default=0,
                       help="Dropout rate for classifier layer")
    parser.add_argument("--middle_dropout", type=float, default=0.5,
                       help="Dropout rate for middle layer")
    parser.add_argument("--early_stopping_patience", type=int, default=10,
                       help="Early stopping patience")
    parser.add_argument("--no_early_stopping", action="store_true",
                       help="Disable early stopping")
    parser.add_argument("--no_lr_scheduler", action="store_true",
                       help="Disable learning rate scheduler")
    parser.set_defaults(no_early_stopping=True)
    parser.set_defaults(no_lr_scheduler=True)

    
    args = parser.parse_args()
    
    # Load data
    base_path = get_base_path()
    stim_train_path = os.path.join(base_path, "stimulus_reg-train.npy")
    label_train_path = os.path.join(base_path, "stimulus_reg-train.tsv")
    stim_val_path = os.path.join(base_path, "stimulus_reg-validation.npy")
    label_val_path = os.path.join(base_path, "stimulus_reg-validation.tsv")
    
    print("\nLoading data...")
    stims_train = np.load(stim_train_path, allow_pickle=True)
    lbls_train = pd.read_csv(label_train_path, sep="\t", index_col=0)
    stims_val = np.load(stim_val_path, allow_pickle=True)
    lbls_val = pd.read_csv(label_val_path, sep="\t", index_col=0)
    print(f"  Train: {stims_train.shape}, Val: {stims_val.shape}")
    
    # Create datasets
    use_sector_mode = args.use_sector_mode
    if use_sector_mode:
        num_pos = 9
        train_ds = MC_RNN_Dataset(stims_train, lbls_train, use_sector=True, num_sectors=9)
        val_ds = MC_RNN_Dataset(stims_val, lbls_val, use_sector=True, num_sectors=9)
        print("Mode: Sector (3x3 grid)")
    else:
        num_pos = 2
        train_ds = MC_RNN_Dataset(stims_train, lbls_train, use_sector=False)
        val_ds = MC_RNN_Dataset(stims_val, lbls_val, use_sector=False)
        print("Mode: Coordinate")
    
    # Create results directory
    results_dir = f"results/models/{args.result_suffix}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Training loop
    for hidden_size in args.hidden_sizes:
        print(f"\n{'='*60}")
        print(f"Training RNN with hidden_size={hidden_size}")
        print(f"{'='*60}\n")
        
        mdl = RNNConv(
            num_classes=10, 
            num_pos=num_pos, 
            kernel_size=5, 
            hidden_size=hidden_size,
            classifier_dropout=args.classifier_dropout,
            middle_dropout=args.middle_dropout
        )
        
        results = network_train(
            mdl, train_ds, val_ds,
            num_epochs=args.num_epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            weight_decay=args.weight_decay,
            use_early_stopping=not args.no_early_stopping,
            early_stopping_patience=args.early_stopping_patience,
            label_smoothing=args.label_smoothing,
            use_lr_scheduler=not args.no_lr_scheduler
        )
        
        mode_suffix = "sector" if use_sector_mode else "coord"
        results_path = os.path.join(
            results_dir,
            f"rnn_{mode_suffix}_improved_h{hidden_size}_lr{args.lr}_bs{args.batch_size}"
        )
        
        save_results(results, results_path)
        print(f"\n✓ Training completed for hidden_size={hidden_size}")
        print(f"  Best validation accuracy: {results['best_val_acc']:.2f}%")
        print(f"  Actual epochs: {results['actual_epochs']}\n")
    
    print(f"\n{'='*60}")
    print(f"All experiments completed!")
    print(f"Results saved to: {results_dir}/")
    print(f"{'='*60}\n")

