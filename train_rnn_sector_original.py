"""
Simplified RNN Sector training script - Matches original notebook performance
移除了所有 allchar mode 和过度正则化的修改，只保留核心功能
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
    def __init__(self, num_classes, num_pos, kernel_size=3, device='cuda', hidden_size=256):
        """
        原始版本的 RNN 模型 - 匹配 notebook 性能
        关键：encoder 中没有 dropout！
        """
        super(RNNConv, self).__init__()
        self.device = device
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
        关键修改：移除了 dropout2d
        这是原始 notebook 的实现方式
        """
        x = self.conv1(x)
        x = self.MP1(x)
        x = self.LNorm1(x)
        x = F.relu(x)
        # NO DROPOUT HERE! 这是关键！
        
        x = self.conv2(x)
        x = self.MP2(x)
        x = self.LNorm2(x)
        x = F.relu(x)
        # NO DROPOUT HERE! 这是关键！
        return x

    def middle(self, x):
        """保持 middle 层的 dropout (0.5)，这是原始设计"""
        x = self.rnn(x)[0]
        x = self.LNormRNN(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        return x

    def classifier(self, x):
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
                  batch_size=256):
    """
    简化的训练函数 - 匹配原始 notebook
    """
    use_sector = train_data.use_sector
    device = mdl.device
    mdl.to(device)
    
    # 设置默认loss weights
    if loss_weights is None:
        if use_sector:
            loss_weights = [1, 1]  # sector mode
        else:
            loss_weights = [1, 0.001]  # coordinate mode
    
    # 优化器：无 weight_decay！
    optim = torch.optim.Adam(mdl.parameters(), lr=lr)
    criterion_char = nn.CrossEntropyLoss()
    
    if use_sector:
        criterion_pos = nn.CrossEntropyLoss()
    else:
        criterion_pos = nn.MSELoss()
    
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
    print(f"  Weight decay: 0 (disabled)")
    print(f"  Encoder dropout: 0 (disabled)")
    print(f"  Middle dropout: 0.5 (original)")
    print(f"  Mode: {'sector' if use_sector else 'coordinate'}")
    print()
    
    train_acc_char = np.zeros(num_epochs)
    val_acc_char = np.zeros(num_epochs)
    train_metric_pos = np.zeros(num_epochs)
    val_metric_pos = np.zeros(num_epochs)
    
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
            optim.step()
            
            epoch_train_acc += (torch.argmax(out_char, dim=2) == labels[:, :, 0].long()).float().mean().item()
            
            if use_sector:
                epoch_train_pos += (torch.argmax(out_pos, dim=2) == labels[:, :, 1].long()).float().mean().item()
            else:
                labels_pos = labels[:, :, 1:].float()
                epoch_train_pos += F.mse_loss(out_pos, labels_pos, reduction='mean').item()
            
            num_batches += 1
        
        train_acc_char[epoch] = (epoch_train_acc / num_batches) * 100
        if use_sector:
            train_metric_pos[epoch] = (epoch_train_pos / num_batches) * 100
        else:
            train_metric_pos[epoch] = epoch_train_pos / num_batches
        
        # Validation
        with torch.no_grad():
            val_acc_char[epoch], val_metric_pos[epoch] = evaluate(mdl, val_dl)
            
            if use_sector:
                train_str = f"Epoch {epoch + 1}/{num_epochs} - Train (char, sector): ({train_acc_char[epoch]:.2f}%, {train_metric_pos[epoch]:.2f}%)"
                val_str = f" Val (char, sector): ({val_acc_char[epoch]:.2f}%, {val_metric_pos[epoch]:.2f}%)"
            else:
                train_str = f"Epoch {epoch + 1}/{num_epochs} - Train (char, pos): ({train_acc_char[epoch]:.2f}%, {train_metric_pos[epoch]:.2f} pix^2)"
                val_str = f" Val (char, pos): ({val_acc_char[epoch]:.2f}%, {val_metric_pos[epoch]:.2f} pix^2)"
            print(train_str, val_str)
    
    torch.cuda.empty_cache()
    
    if use_sector:
        return {
            "train_acc_char": train_acc_char,
            "val_acc_char": val_acc_char,
            "train_acc_pos": train_metric_pos,
            "val_acc_pos": val_metric_pos,
            "model": mdl.to("cpu"),
            "actual_epochs": num_epochs
        }
    else:
        return {
            "train_acc_char": train_acc_char,
            "val_acc_char": val_acc_char,
            "train_err_pos": train_metric_pos,
            "val_err_pos": val_metric_pos,
            "model": mdl.to("cpu"),
            "actual_epochs": num_epochs
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
    parser = argparse.ArgumentParser(description="原始版本 RNN Sector 训练脚本")
    parser.add_argument("--model_types", nargs="+", default=["rnn"], choices=["rnn"])
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[256])
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--use_sector_mode", action="store_true", default=False)
    parser.add_argument("--result_suffix", type=str, default="original")
    
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
        
        mdl = RNNConv(num_classes=10, num_pos=num_pos, kernel_size=5, hidden_size=hidden_size)
        
        results = network_train(
            mdl, train_ds, val_ds,
            num_epochs=args.num_epochs,
            lr=args.lr,
            batch_size=args.batch_size
        )
        
        mode_suffix = "sector" if use_sector_mode else "coord"
        results_path = os.path.join(
            results_dir,
            f"rnn_{mode_suffix}_original_h{hidden_size}_lr{args.lr}_bs{args.batch_size}"
        )
        
        save_results(results, results_path)
        print(f"\n✓ Training completed for hidden_size={hidden_size}\n")
    
    print(f"\n{'='*60}")
    print(f"All experiments completed!")
    print(f"Results saved to: {results_dir}/")
    print(f"{'='*60}\n")

