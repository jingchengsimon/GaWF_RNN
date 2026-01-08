"""
Standalone RNN Sector training script
Used to train RNN models and save results
"""

import os
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

try:
    import psutil
except ImportError:
    psutil = None


# ==================== Dataset Classes ====================
class MC_RNN_Dataset(Dataset):
    def __init__(self, data, labels, frame_num=32, chan_num=2, use_sector=False, num_sectors=9):
        """
        Args:
            data (np.ndarray): Array of shape (num_samples, num_frames, height, width)
            labels (np.ndarray): DataFrame with columns ['fg_char_id', 'fg_char_x', 'fg_char_y']
            frame_num (int): Number of frames to stack for input as multichannel image
            chan_num (int): Number of channels in the input images. Each channel is a previous frame.
            use_sector (bool): If True, map (x, y) position to sector id 0-(num_sectors-1)
            num_sectors (int): Number of sectors, e.g., 9 means 0-8 sectors (3x3 grid)
        """
        self.data = data
        self.labels = labels[['fg_char_id', 'fg_char_x', 'fg_char_y']].values
        self.frame_num = frame_num
        self.chan_num = chan_num
        self.use_sector = use_sector
        self.num_sectors = num_sectors

    def __len__(self):
        return (self.data.shape[0]-self.chan_num) // self.frame_num

    def __getitem__(self, idx):
        start_idx = (idx * self.frame_num) + self.chan_num
        end_idx = start_idx + self.frame_num

        # Stack frames to create a multichannel image
        for i in range(-(self.chan_num-1), 1):
            if i == -(self.chan_num-1):
                stacked_frames = np.expand_dims(self.data[(start_idx + i):(end_idx + i)], axis=1)
            else:
                stacked_frames = np.concatenate((stacked_frames,
                                                 np.expand_dims(self.data[(start_idx + i):(end_idx + i)],
                                                                axis=1)), axis=1)
        stacked_frames = stacked_frames.astype(np.float32)

        # labels: (frame_num, 3) -> [char_id, x, y]
        labels = self.labels[start_idx:end_idx].copy()

        if self.use_sector:
            # Use image width and height to map (x, y) to a grid_size x grid_size grid,
            # obtaining sector id 0-(num_sectors-1) (e.g., num_sectors=9 -> 3x3 grid)
            height = self.data.shape[-2]
            width = self.data.shape[-1]

            # Derive grid_size for each dimension from num_sectors (assuming num_sectors is a perfect square, e.g., 9, 16)
            grid_size = int(np.sqrt(self.num_sectors))
            if grid_size * grid_size != self.num_sectors:
                raise ValueError(f"num_sectors={self.num_sectors} is not a perfect square, cannot form grid_size x grid_size grid")

            x = labels[:, 1].astype(np.float32)
            y = labels[:, 2].astype(np.float32)

            # Normalize coordinates to [0, grid_size) then round, using (width-1)/(height-1) to avoid out-of-bounds
            col = (x / max(width - 1, 1) * grid_size).astype(np.int64)
            row = (y / max(height - 1, 1) * grid_size).astype(np.int64)

            # Prevent out-of-bounds due to numerical or boundary issues
            col = np.clip(col, 0, grid_size - 1)
            row = np.clip(row, 0, grid_size - 1)

            # Encode sector id in row-major order: row * grid_size + col, range 0-(num_sectors-1)
            sector = row * grid_size + col

            # New label: [char_id, sector_id]
            labels = np.stack([labels[:, 0].astype(np.int64), sector], axis=1)

        # Convert to torch tensor to avoid pin_memory issues
        stacked_frames = torch.from_numpy(stacked_frames).contiguous()
        labels = torch.from_numpy(labels).contiguous()
        
        return stacked_frames, labels


# ==================== Model Classes ====================
class RNNConv(nn.Module):
    def __init__(self, num_classes, num_pos, kernel_size=3, device='cuda', hidden_size=256):
        super(RNNConv, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(2, 32, kernel_size=kernel_size, padding='same')
        self.MP1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.LNorm1 = nn.LayerNorm([32, 48, 48])
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.MP2 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.LNorm2 = nn.LayerNorm([64, 12, 12])
        # Original RNN version
        self.rnn = nn.RNN(input_size=64 * 12 * 12, hidden_size=hidden_size,
                          num_layers=1, batch_first=True)
        self.LNormRNN = nn.LayerNorm(hidden_size)
        self.fcchar = nn.Linear(hidden_size, num_classes)
        self.fcpos = nn.Linear(hidden_size, num_pos)
        self.to(self.device)

    def encoder(self, x):
        return nn.Sequential(
            self.conv1,
            self.MP1,
            self.LNorm1,
            nn.ReLU(),
            self.conv2,
            self.MP2,
            self.LNorm2,
            nn.ReLU()
        )(x)

    def middle(self, x):
        x = self.rnn(x)[0]
        x = self.LNormRNN(x)
        x = nn.Dropout(0.5)(nn.ReLU()(x))
        return x

    def classifier(self, x):
        return self.fcchar(x), self.fcpos(x)

    def forward(self, x):
        x = x.to(self.device)

        batch_size, frame_num, channels, height, width = x.size()

        # resize to process each frame individually
        x = x.view(batch_size * frame_num, channels, height, width)

        # apply CNN encoder
        x = self.encoder(x)
        
        # reshape back to batches of stacks of frames and flatten each image
        x = x.view(batch_size, frame_num, -1)

        # apply RNN
        x = self.middle(x)

        # apply classification heads
        char_out, pos_out = self.classifier(x)
        return char_out, pos_out


class GRUConv(nn.Module):
    def __init__(self, num_classes, num_pos, kernel_size=3, device='cuda', hidden_size=256):
        super(GRUConv, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(2, 32, kernel_size=kernel_size, padding='same')
        self.MP1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.LNorm1 = nn.LayerNorm([32, 48, 48])
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.MP2 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.LNorm2 = nn.LayerNorm([64, 12, 12])
        self.rnn = nn.GRU(input_size=64 * 12 * 12, hidden_size=hidden_size,
                          num_layers=1, batch_first=True)
        self.LNormRNN = nn.LayerNorm(hidden_size)
        self.fcchar = nn.Linear(hidden_size, num_classes)
        self.fcpos = nn.Linear(hidden_size, num_pos)
        self.to(self.device)

    def encoder(self, x):
        return nn.Sequential(
            self.conv1,
            self.MP1,
            self.LNorm1,
            nn.ReLU(),
            self.conv2,
            self.MP2,
            self.LNorm2,
            nn.ReLU()
        )(x)

    def middle(self, x):
        x = self.rnn(x)[0]
        x = self.LNormRNN(x)
        x = nn.Dropout(0.5)(nn.ReLU()(x))
        return x

    def classifier(self, x):
        return self.fcchar(x), self.fcpos(x)

    def forward(self, x):
        x = x.to(self.device)

        batch_size, frame_num, channels, height, width = x.size()

        # resize to process each frame individually
        x = x.view(batch_size * frame_num, channels, height, width)

        # apply CNN encoder
        x = self.encoder(x)

        # reshape back to batches of stacks of frames and flatten each image
        x = x.view(batch_size, frame_num, -1)

        # apply RNN
        x = self.middle(x)

        # apply classification heads
        char_out, pos_out = self.classifier(x)
        return char_out, pos_out


class LSTMConv(nn.Module):
    def __init__(self, num_classes, num_pos, kernel_size=3, device='cuda', hidden_size=256):
        super(LSTMConv, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(2, 32, kernel_size=kernel_size, padding='same')
        self.MP1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.LNorm1 = nn.LayerNorm([32, 48, 48])
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.MP2 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.LNorm2 = nn.LayerNorm([64, 12, 12])
        self.rnn = nn.LSTM(input_size=64 * 12 * 12, hidden_size=hidden_size,
                           num_layers=1, batch_first=True)
        self.LNormRNN = nn.LayerNorm(hidden_size)
        self.fcchar = nn.Linear(hidden_size, num_classes)
        self.fcpos = nn.Linear(hidden_size, num_pos)
        self.to(self.device)

    def encoder(self, x):
        return nn.Sequential(
            self.conv1,
            self.MP1,
            self.LNorm1,
            nn.ReLU(),
            self.conv2,
            self.MP2,
            self.LNorm2,
            nn.ReLU()
        )(x)

    def middle(self, x):
        x = self.rnn(x)[0]
        x = self.LNormRNN(x)
        x = nn.Dropout(0.5)(nn.ReLU()(x))
        return x

    def classifier(self, x):
        return self.fcchar(x), self.fcpos(x)

    def forward(self, x):
        x = x.to(self.device)

        batch_size, frame_num, channels, height, width = x.size()

        # resize to process each frame individually
        x = x.view(batch_size * frame_num, channels, height, width)

        # apply CNN encoder
        x = self.encoder(x)

        # reshape back to batches of stacks of frames and flatten each image
        x = x.view(batch_size, frame_num, -1)

        # apply RNN
        x = self.middle(x)

        # apply classification heads
        char_out, pos_out = self.classifier(x)
        return char_out, pos_out


class GaWFRNNConv(nn.Module):
    """
    GaWF (Gated with Feedback) RNN Model
    
    Main improvements:
    1. Use classifier output as feedback to RNN input
    2. Feedback is transformed by U @ diag(concat) @ V, then Hadamard product with RNN weights
    """
    def __init__(self, num_classes, num_pos, kernel_size=3, device='cuda', dropout_rate=0.3, hidden_size=256):
        super(GaWFRNNConv, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.num_pos = num_pos
        self.dropout_rate = dropout_rate
        
        # CNN encoder (same as RNNConv)
        self.conv1 = nn.Conv2d(2, 32, kernel_size=kernel_size, padding='same')
        self.MP1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.LNorm1 = nn.LayerNorm([32, 48, 48])
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.MP2 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.LNorm2 = nn.LayerNorm([64, 12, 12])
        
        # RNN parameters
        input_size = 64 * 12 * 12  # 9216
        
        # Create RNN (but not using built-in, manually implement to support feedback)
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                          num_layers=1, batch_first=True)
        
        # Feedback transformation matrices
        # Dimension after concatenating classifier outputs
        feedback_dim = num_classes + num_pos  # e.g., 10 + 9 = 19
        
        # RNN weight matrix shapes
        # weight_ih: (hidden_size, input_size) = (hidden_size, 9216)
        # weight_hh: (hidden_size, hidden_size) = (hidden_size, hidden_size)
        # Concatenated shape: (hidden_size, input_size + hidden_size)
        combined_weight_size = input_size + hidden_size
        
        # U: (hidden_size, feedback_dim) = (hidden_size, 19)
        # V: (feedback_dim, combined_weight_size) = (19, input_size + hidden_size)
        # diag(concat): (feedback_dim, feedback_dim) = (19, 19)
        # U @ diag @ V: (hidden_size, 19) @ (19, 19) @ (19, combined_weight_size) = (hidden_size, combined_weight_size)
        self.U = nn.Parameter(torch.randn(hidden_size, feedback_dim) * 0.01)
        self.V = nn.Parameter(torch.randn(feedback_dim, combined_weight_size) * 0.01)
        
        # LayerNorm and Dropout
        self.LNormRNN = nn.LayerNorm(hidden_size)
        
        # Classifier heads
        self.fcchar = nn.Linear(hidden_size, num_classes)
        self.fcpos = nn.Linear(hidden_size, num_pos)
        
        # Store previous forward's classifier output as feedback for next time
        self.register_buffer('prev_feedback', None)
        
        self.to(self.device)

    def encoder(self, x):
        x = self.conv1(x)
        x = self.MP1(x)
        x = self.LNorm1(x)
        x = F.relu(x)
        # x = F.dropout2d(x, p=self.dropout_rate, training=self.training)
        
        x = self.conv2(x)
        x = self.MP2(x)
        x = self.LNorm2(x)
        x = F.relu(x)
        # x = F.dropout2d(x, p=self.dropout_rate, training=self.training)
        return x

    def middle(self, x, feedback=None):
        """
        GaWF RNN middle layer with feedback mechanism
        
        According to the diagram (Panel A):
        1. Concatenate Input and Feedback
        2. Matrix multiplication with weight matrix (standard RNN computation)
        3. Feedback transformed by U @ diag(concat) @ V, then sigmoid (gating signal)
        4. Hadamard product of the two results (element-wise multiplication)
        5. LayerNorm
        6. ReLU
        
        Args:
            x: Input sequence (B, T, input_size)
            feedback: Feedback from classifier (B, T, feedback_dim) or None
        """
        batch_size, seq_len, input_size = x.size()
        hidden_size = self.rnn.hidden_size
        
        if feedback is not None:
            # Get RNN weights
            weight_ih = self.rnn.weight_ih_l0  # (hidden_size, input_size)
            weight_hh = self.rnn.weight_hh_l0  # (hidden_size, hidden_size)
            bias_ih = self.rnn.bias_ih_l0 if self.rnn.bias_ih_l0 is not None else None
            bias_hh = self.rnn.bias_hh_l0 if self.rnn.bias_hh_l0 is not None else None
            
            # Process each time step
            outputs = []
            h = torch.zeros(batch_size, hidden_size, device=x.device)  # (B, hidden_size)
            
            # Concatenate RNN input weight matrix and recurrent weight matrix (compute once)
            combined_weight = torch.cat([weight_ih, weight_hh], dim=1)  # (hidden_size, input_size + hidden_size)
            
            for t in range(seq_len):
                # Current time step input and feedback
                x_t = x[:, t, :]  # (B, input_size)
                fb_t = feedback[:, t, :]  # (B, feedback_dim)
                
                # Feedback transformation: U @ diag(fb_t) @ V, then sigmoid
                gated_weights = []
                for b in range(batch_size):
                    # Build diagonal matrix: diag(fb_t[b])
                    diag_matrix = torch.diag(fb_t[b])  # (feedback_dim, feedback_dim)
                    
                    # U @ diag @ V: (hidden_size, combined_weight_size)
                    transformed = self.U @ diag_matrix @ self.V  # (hidden_size, combined_weight_size)
                    
                    # Apply sigmoid
                    transformed = torch.sigmoid(transformed)  # (hidden_size, combined_weight_size)
                    
                    # Hadamard product: transformed * combined_weight
                    gated_weight = transformed * combined_weight  # (hidden_size, combined_weight_size)
                    gated_weights.append(gated_weight)
                
                # Stack gated weights: (B, hidden_size, combined_weight_size)
                gated_weights = torch.stack(gated_weights, dim=0)  # (B, hidden_size, combined_weight_size)
                
                # Separate back into weight_ih and weight_hh parts
                gated_weight_ih = gated_weights[:, :, :input_size]  # (B, hidden_size, input_size)
                gated_weight_hh = gated_weights[:, :, input_size:]  # (B, hidden_size, hidden_size)
                
                # Compute RNN output using gated weights
                h_t_list = []
                for b in range(batch_size):
                    # Input to hidden: (1, hidden_size)
                    ih = F.linear(x_t[b:b+1], gated_weight_ih[b], bias_ih)  # (1, hidden_size)
                    # Hidden to hidden: (1, hidden_size)
                    hh = F.linear(h[b:b+1], gated_weight_hh[b], bias_hh)  # (1, hidden_size)
                    # Combine and apply activation
                    h_t = torch.tanh(ih + hh)  # (1, hidden_size)
                    h_t_list.append(h_t)
                
                # Stack: (B, hidden_size)
                h_t = torch.cat(h_t_list, dim=0)  # (B, hidden_size)
                
                # LayerNorm and ReLU
                gated_output = self.LNormRNN(h_t)  # (B, hidden_size)
                gated_output = F.relu(gated_output)  # (B, hidden_size)
                
                outputs.append(gated_output)
                h = gated_output  # Update hidden state
            
            # Stack outputs: (B, T, hidden_size)
            x = torch.stack(outputs, dim=1)  # (B, T, hidden_size)
        else:
            # When no feedback, use standard RNN
            x = self.rnn(x)[0]
            x = self.LNormRNN(x)
            x = nn.Dropout(0.5)(nn.ReLU()(x))
        
        # Dropout
        # x = F.dropout(x, p=0.5, training=self.training)
        return x

    def classifier(self, x):
        return self.fcchar(x), self.fcpos(x)

    def forward(self, x, use_feedback=True, reset_feedback=False):
        """
        Args:
            x: Input sequence (B, T, C, H, W)
            use_feedback: Whether to use feedback mechanism
            reset_feedback: Whether to reset feedback (for new sequence start, e.g., new epoch or new batch)
        """
        x = x.to(self.device)

        batch_size, frame_num, channels, height, width = x.size()

        # resize to process each frame individually
        x = x.view(batch_size * frame_num, channels, height, width)

        # apply CNN encoder
        x = self.encoder(x)
        
        # reshape back to batches of stacks of frames and flatten each image
        x = x.view(batch_size, frame_num, -1)

        # Determine whether to use feedback
        if use_feedback:
            # If reset_feedback is True, or prev_feedback is None (first forward)
            if reset_feedback or self.prev_feedback is None:
                # First forward: don't use feedback
                feedback = None
            else:
                # Use previous forward's saved classifier output as feedback
                feedback = self.prev_feedback  # (B, T, feedback_dim)
        else:
            feedback = None
        
        # apply RNN
        x = self.middle(x, feedback=feedback) #feedback)

        # apply classification heads
        char_out, pos_out = self.classifier(x)
        
        # If using feedback, save current output as feedback for next time
        if use_feedback:
            # Concatenate two classifier outputs
            # char_out: (B, T, num_classes)
            # pos_out: (B, T, num_pos)
            # prev_feedback: (B, T, num_classes + num_pos)
            # Use .detach() to break gradient connection, avoid computation graph issues
            self.prev_feedback = torch.cat([char_out, pos_out], dim=-1).detach()  # (B, T, feedback_dim)
            
        return char_out, pos_out


class FeedForwardConv(nn.Module):
    """FeedForward model, adapted to RNN data format (using frame_num as input_channels)"""
    def __init__(self, input_channels, num_classes, num_pos, kernel_size=3, device='cuda', hidden_size=512):
        super(FeedForwardConv, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=kernel_size, padding='same')
        self.MP1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.LNorm1 = nn.LayerNorm([32, 48, 48])
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.MP2 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.LNorm2 = nn.LayerNorm([64, 12, 12])
        self.fc1 = nn.Linear(64 * 12 * 12, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.fcchar = nn.Linear(hidden_size, num_classes)
        self.fcpos = nn.Linear(hidden_size, num_pos)
        self.to(self.device)

    def encoder(self, x):
        return nn.Sequential(
            self.conv1,
            self.MP1,
            self.LNorm1,
            nn.ReLU(),
            self.conv2,
            self.MP2,
            self.LNorm2,
            nn.ReLU()
        )(x)

    def middle(self, x):
        return nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.dropout
        )(x)

    def classifier(self, x):
        return self.fcchar(x), self.fcpos(x)

    def forward(self, x):
        x = x.to(self.device)
        batch_size, frame_num, channels, height, width = x.size()
        
        # Merge frame_num and channels as input_channels
        # reshape: (B, T, C, H, W) -> (B, T*C, H, W)
        x = x.view(batch_size, frame_num * channels, height, width)
        
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.middle(x)
        char_out, pos_out = self.classifier(x)
        
        # FFN output is (B, num_classes) and (B, num_pos), need to expand to (B, T, ...) to match RNN format
        # Here we repeat T times to match RNN output format
        char_out = char_out.unsqueeze(1).expand(-1, frame_num, -1)
        pos_out = pos_out.unsqueeze(1).expand(-1, frame_num, -1)
        
        return char_out, pos_out


# ==================== GPU Utility Functions ====================
def get_gpu_memory_usage():
    """Get current GPU memory usage (percentage)"""
    if not torch.cuda.is_available():
        return 0.0
    # Use reserved memory to calculate usage more accurately
    allocated = torch.cuda.memory_allocated() / 1024**3  # GB
    reserved = torch.cuda.memory_reserved() / 1024**3    # GB
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
    return (reserved / total) * 100.0 if total > 0 else 0.0

def find_optimal_batch_size(model, train_data, device='cuda', start_batch_size=32, max_batch_size=256):
    """
    Automatically find optimal batch_size without overloading GPU
    
    Args:
        model: Model
        train_data: Training dataset
        device: Device
        start_batch_size: Starting batch_size
        max_batch_size: Maximum batch_size
    
    Returns:
        Optimal batch_size
    """
    if device == 'cpu':
        return start_batch_size
    
    model.eval()
    optimal_batch_size = start_batch_size
    
    # Test different batch sizes
    for batch_size in [start_batch_size]:#, 64, 128, 256]:
        if batch_size > max_batch_size:
            break
        
        try:
            # Clear cache
            torch.cuda.empty_cache()
            
            # Create test data
            test_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=0)
            test_batch = next(iter(test_loader))
            inputs, labels = test_batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass test
            with torch.no_grad():
                _ = model(inputs)
            
            # Check memory usage
            memory_usage = get_gpu_memory_usage()
            
            if memory_usage < 80.0:  # If memory usage < 80%, can try larger batch_size
                optimal_batch_size = batch_size
                print(f"Test batch_size={batch_size}: GPU memory usage {memory_usage:.1f}%, OK")
            else:
                print(f"Test batch_size={batch_size}: GPU memory usage {memory_usage:.1f}%, exceeds limit")
                break
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"batch_size={batch_size} caused OOM, using batch_size={optimal_batch_size}")
                torch.cuda.empty_cache()
                break
            else:
                raise e
    
    torch.cuda.empty_cache()
    model.train()
    return optimal_batch_size


# ==================== Training Function ====================
def network_train(mdl, train_data, val_data, num_epochs=50, loss_weights=[1, 1], lr=0.001, 
                         batch_size=None, use_amp=True, num_workers=None, pin_memory=True,
                         use_modification=False):
    """
    Train model in sector mode
    
    Args:
        mdl: Model
        train_data: Training dataset
        val_data: Validation dataset
        num_epochs: Number of training epochs
        loss_weights: [character loss weight, position loss weight]
        lr: Learning rate
        batch_size: Batch size, if None then automatically find optimal value
        use_amp: Whether to use mixed precision training (FP16)
        num_workers: DataLoader num_workers, if None then automatically set
        pin_memory: Whether to use pin_memory to accelerate data transfer
        use_modification: Whether to use modification settings (gradient clipping and learning rate decay)
                         - True: Enable gradient clipping (max_norm=2.0) and periodic LR decay (decay 0.5x at 25%, 50%, 75% epochs)
                         - False: No modifications, train in original way
    """
    # Place parameters according to model's internal device (can be 'cuda' or 'cpu')
    device = mdl.device
    mdl.to(device)
    
    # Automatically set num_workers (based on CPU cores)
    # In WSL, set num_workers=0 to avoid multiprocessing resource leaks
    if num_workers is None:
        # Check if running in WSL
        is_wsl = os.name == 'posix' and os.path.exists('/mnt/c')
        if is_wsl:
            num_workers = 0  # Disable multiprocessing in WSL to avoid semaphore leaks
            print("Running in WSL: setting num_workers=0 to avoid multiprocessing resource leaks")
        elif psutil is not None:
            num_workers = min(4, psutil.cpu_count(logical=False))  # Use physical cores, max 4
        else:
            num_workers = min(4, os.cpu_count() or 1)  # fallback to os.cpu_count()
    
    # Automatically find optimal batch_size
    if batch_size is None and not isinstance(mdl, GaWFRNNConv):
        print("Automatically finding optimal batch_size...")
        batch_size = find_optimal_batch_size(mdl, train_data, device=device, start_batch_size=8)
        print(f"Using batch_size = {batch_size}")
    else:
        batch_size = 256
        print(f"Detected GaWFRNNConv model, skipping batch_size search, using default batch_size = {batch_size}")
            
    optim = torch.optim.Adam(mdl.parameters(), lr=lr)
    criterion_char = nn.CrossEntropyLoss()
    criterion_pos = nn.CrossEntropyLoss()  # sector classification
    
    # Modification settings: gradient clipping and learning rate decay
    if use_modification:
        # Learning rate decay settings: decay LR at 25%, 50%, 75% epochs (decay factor 0.5)
        lr_decay_epochs = [int(num_epochs * 0.25), int(num_epochs * 0.5), int(num_epochs * 0.75)]
        lr_decay_factor = 0.5  # Decay to 0.5x each time
        initial_lr = lr  # Save initial learning rate
        print(f"Modification settings enabled:")
        print(f"  - Gradient clipping: max_norm=2.0")
        print(f"  - Learning rate decay: decay to {lr_decay_factor}x at epochs {lr_decay_epochs}")
    else:
        lr_decay_epochs = []
        lr_decay_factor = 1.0
        initial_lr = lr
        print("Modification settings disabled: training in original way (no gradient clipping, no LR decay)")
    
    # Mixed precision training
    if use_amp and device == 'cuda':
        try:
            scaler = GradScaler('cuda')
        except TypeError:
            # Compatible with older versions
            scaler = GradScaler()
    else:
        scaler = None

    def loss_fn(out_char, out_pos, labels):
        # labels: (B, T, 2) -> [char_id, sector_id]
        labels_char = labels[:, :, 0].long().view(-1)
        labels_pos = labels[:, :, 1].long().view(-1)
        outputs_char = out_char.view(-1, out_char.shape[-1])      # (B*T, num_classes)
        outputs_pos = out_pos.view(-1, out_pos.shape[-1])         # (B*T, num_sectors)
        loss_char = criterion_char(outputs_char, labels_char)
        loss_pos = criterion_pos(outputs_pos, labels_pos)

        # Keep consistent with original regularization (if model has mdl.rnn, add regularization)
        if hasattr(mdl, 'rnn'):
            rnn_hh = mdl.rnn.weight_hh_l0
            rnn_hh_diag = torch.diagonal(rnn_hh).abs().sum()
            loss = (loss_weights[0] * loss_char) + (loss_weights[1] * loss_pos) + rnn_hh_diag
        else:
            # FeedForward model doesn't have rnn, no regularization
            loss = (loss_weights[0] * loss_char) + (loss_weights[1] * loss_pos)
        return loss

    def evaluate(mdl, data_loader):
        mdl.eval()
        total_acc_char = 0
        total_acc_pos = 0
        with torch.no_grad():
            for batch in data_loader:
                inputs, labels = batch
                
                # Reset feedback at start of each batch (if GaWFRNNConv)
                if hasattr(mdl, 'prev_feedback'):
                    mdl.prev_feedback = None
                
                inputs = inputs.to(device, non_blocking=pin_memory)
                labels = labels.to(device, non_blocking=pin_memory)
                
                # Use mixed precision during validation to speed up
                if use_amp and scaler is not None:
                    try:
                        # New API: torch.amp.autocast(device_type='cuda')
                        with autocast(device_type='cuda' if device == 'cuda' else 'cpu'):
                            out_char, out_pos = mdl(inputs)
                    except TypeError:
                        # Compatible with older versions
                        with autocast():
                            out_char, out_pos = mdl(inputs)
                else:
                    out_char, out_pos = mdl(inputs)
                
                # Character accuracy
                total_acc_char += (torch.argmax(out_char, dim=2) == labels[:, :, 0].long()).float().mean().item()
                # Sector accuracy
                total_acc_pos += (torch.argmax(out_pos, dim=2) == labels[:, :, 1].long()).float().mean().item()
        return total_acc_char * 100 / len(data_loader), total_acc_pos * 100 / len(data_loader)

    # Data loader (optimized settings)
    train_dl = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and device == 'cuda',
        persistent_workers=num_workers > 0  # Keep worker processes to avoid repeated creation
    )
    val_dl = DataLoader(
        val_data, 
        batch_size=batch_size, 
        shuffle=False,  # Validation set doesn't need shuffle
        num_workers=num_workers,
        pin_memory=pin_memory and device == 'cuda',
        persistent_workers=num_workers > 0
    )
    train_acc_char = np.zeros(num_epochs)
    val_acc_char = np.zeros(num_epochs)
    train_acc_pos = np.zeros(num_epochs)
    val_acc_pos = np.zeros(num_epochs)

    for epoch in range(num_epochs):
        # Learning rate decay: decay LR at specified epochs
        if use_modification and epoch in lr_decay_epochs:
            current_lr = optim.param_groups[0]['lr']
            new_lr = current_lr * lr_decay_factor
            for param_group in optim.param_groups:
                param_group['lr'] = new_lr
            print(f"Epoch {epoch + 1}: Learning rate decayed from {current_lr:.6f} to {new_lr:.6f}")
        
        mdl.train()
        for batch_idx, batch in enumerate(train_dl):
            inputs, labels = batch
            
            # Reset feedback at start of each batch (if GaWFRNNConv)
            # This ensures feedback's batch_size and seq_len match current batch
            if hasattr(mdl, 'prev_feedback'):
                mdl.prev_feedback = None
            
            inputs = inputs.to(device, non_blocking=pin_memory)
            labels = labels.to(device, non_blocking=pin_memory)
            
            optim.zero_grad()
            
            # Mixed precision training
            if use_amp and scaler is not None:
                try:
                    # New API: torch.amp.autocast(device_type='cuda')
                    with autocast(device_type='cuda' if device == 'cuda' else 'cpu'):
                        out_char, out_pos = mdl(inputs)
                        loss = loss_fn(out_char, out_pos, labels)
                except TypeError:
                    # Compatible with older versions
                    with autocast():
                        out_char, out_pos = mdl(inputs)
                        loss = loss_fn(out_char, out_pos, labels)
                
                scaler.scale(loss).backward()
                # Gradient clipping (if modification enabled)
                if use_modification:
                    scaler.unscale_(optim)  # Need to unscale first in mixed precision training
                    torch.nn.utils.clip_grad_norm_(mdl.parameters(), max_norm=2.0)
                scaler.step(optim)
                scaler.update()
            else:
                out_char, out_pos = mdl(inputs)
                loss = loss_fn(out_char, out_pos, labels)
                loss.backward()
                # Gradient clipping (if modification enabled)
                if use_modification:
                    torch.nn.utils.clip_grad_norm_(mdl.parameters(), max_norm=2.0)
                optim.step()

            train_acc_char[epoch] += (torch.argmax(out_char, dim=2) == labels[:, :, 0].long()).float().mean().item()
            train_acc_pos[epoch] += (torch.argmax(out_pos, dim=2) == labels[:, :, 1].long()).float().mean().item()
            
            # Periodically clear cache (every 100 batches)
            if batch_idx % 100 == 0 and device == 'cuda':
                torch.cuda.empty_cache()

        train_acc_char[epoch] /= len(train_dl)
        train_acc_char[epoch] *= 100
        train_acc_pos[epoch] /= len(train_dl)
        train_acc_pos[epoch] *= 100
        
        # Display GPU usage
        gpu_info = ""
        if device == 'cuda' and torch.cuda.is_available():
            gpu_mem = get_gpu_memory_usage()
            gpu_info = f" | GPU memory: {gpu_mem:.1f}%"
        
        train_str = f"Epoch {epoch + 1}/{num_epochs} - Train (char, sector): ({train_acc_char[epoch]:.2f}%, {train_acc_pos[epoch]:.2f}%){gpu_info}"

        with torch.no_grad():
            val_acc_char[epoch], val_acc_pos[epoch] = evaluate(mdl, val_dl)
            val_str = f" Validation (char, sector): ({val_acc_char[epoch]:.2f}%, {val_acc_pos[epoch]:.2f}%)"
            print(train_str, val_str)

    torch.cuda.empty_cache()
    
    # Clean up DataLoader resources to prevent multiprocessing leaks
    # This is especially important in WSL environments
    try:
        if hasattr(train_dl, '_iterator'):
            train_dl._iterator = None
        if hasattr(val_dl, '_iterator'):
            val_dl._iterator = None
        # Explicitly close DataLoader if it has workers
        if num_workers > 0:
            train_dl._shutdown_workers()
            val_dl._shutdown_workers()
    except Exception:
        pass  # Ignore errors during cleanup
    
    # Force garbage collection
    import gc
    gc.collect()

    return {
        "train_acc_char": train_acc_char,
        "val_acc_char": val_acc_char,
        "train_acc_pos": train_acc_pos,
        "val_acc_pos": val_acc_pos,
        "model": mdl.to("cpu")
    }


# ==================== Utility Functions ====================
def save_results(results, filepath):
    """
    Save training results to local file
    
    Args:
        results: Training results dictionary
        filepath: Save path (e.g., 'results_rnn' or 'results/rnn_sector')
                  If directory doesn't exist, it will be created automatically
    """
    # Extract directory and filename
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Create save dictionary (excluding model, as model is too large)
    results_path = filepath + '.pkl'
    save_dict = {}
    for key, value in results.items():
        if key != "model":
            save_dict[key] = value
    
    # Save model state dict (can be saved separately if needed)
    model_path = results_path.replace('.pkl', '_model.pth')
    if "model" in results:
        # Verify model's num_pos before saving
        saved_num_pos = results["model"].fcpos.out_features
        torch.save(results["model"].state_dict(), model_path)
        print(f"Model state dict saved to: {model_path}")
    
    # Save other results
    with open(results_path, 'wb') as f:
        pickle.dump(save_dict, f)
    print(f"Results saved to: {results_path}")


# ==================== Main Training Code ====================
if __name__ == "__main__":
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Train RNN models for sector classification')
        parser.add_argument('--model_types', type=str, nargs='+', default=["rnn"],
                            choices=["rnn", "lstm", "gru", "gawf", "ffn"],
                            help='Model types to train (default: ["rnn"])')
        parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[256],
                            help='Hidden sizes to test (default: [256])')
        parser.add_argument('--num_epochs', type=int, default=200,
                            help='Number of training epochs (default: 200)')
        args = parser.parse_args()
        
        # Helper function to convert Windows path to WSL path if needed
        def convert_to_wsl_path(windows_path):
            """Convert Windows path to WSL path if running in WSL"""
            try:
                # Check if running in WSL by checking for /mnt/c directory
                if os.name == 'posix' and os.path.exists('/mnt/c'):
                    # Running in WSL, convert Windows path to WSL path
                    # C:\Users\... -> /mnt/c/Users/...
                    path = windows_path.replace('\\', '/')
                    if path.startswith('C:/') or path.startswith('c:/'):
                        path = '/mnt/c' + path[2:]
                    return path
                else:
                    # Running on Windows, use as is
                    return windows_path
            except Exception as e:
                # Fallback: assume Windows path format
                print(f"Warning: Could not determine environment, using Windows path format. Error: {e}")
                return windows_path
        
        # Data path configuration
        # Base path: C:\Users\12265\Desktop\SJC\archive\Aim3\stimuli
        base_path_windows = r"C:\Users\12265\Desktop\SJC\archive\Aim3\stimuli"
        base_path = convert_to_wsl_path(base_path_windows)
        stim_train_path = os.path.join(base_path, "stimulus_reg-train.npy")
        label_train_path = os.path.join(base_path, "stimulus_reg-train.tsv")
        stim_val_path = os.path.join(base_path, "stimulus_reg-validation.npy")
        label_val_path = os.path.join(base_path, "stimulus_reg-validation.tsv")
        
        # Verify paths exist before loading
        print(f"Checking data paths...")
        print(f"Base path: {base_path}")
        for path_name, path in [("train stim", stim_train_path), ("train label", label_train_path),
                                 ("val stim", stim_val_path), ("val label", label_val_path)]:
            if not os.path.exists(path):
                print(f"ERROR: {path_name} path does not exist: {path}")
                raise FileNotFoundError(f"Data file not found: {path}")
            else:
                print(f"  ✓ {path_name}: {path}")
        
        # Load data with error handling
        print("\nLoading data...")
        try:
            stims_train = np.load(stim_train_path, allow_pickle=True)
            print(f"  ✓ Loaded training stimuli: {stims_train.shape}")
            lbls_train = pd.read_csv(label_train_path, sep="\t", index_col=0)
            print(f"  ✓ Loaded training labels: {lbls_train.shape}")
            
            stims_val = np.load(stim_val_path, allow_pickle=True)
            print(f"  ✓ Loaded validation stimuli: {stims_val.shape}")
            lbls_val = pd.read_csv(label_val_path, sep="\t", index_col=0)
            print(f"  ✓ Loaded validation labels: {lbls_val.shape}")
        except Exception as e:
            print(f"ERROR: Failed to load data files: {e}")
            raise
        
        # Create datasets (sector mode, 3x3 grid -> 9 sectors)
        print("Creating datasets...")
        train_ds_sector = MC_RNN_Dataset(stims_train, lbls_train, use_sector=True, num_sectors=9)
        val_ds_sector = MC_RNN_Dataset(stims_val, lbls_val, use_sector=True, num_sectors=9)
        
        # Model class mapping
        MODEL_CLASSES = {
            "rnn": RNNConv,
            "lstm": LSTMConv,
            "gru": GRUConv,
            "gawf": GaWFRNNConv,
            "ffn": FeedForwardConv,
        }
        
        # Training configuration (from command line arguments)
        model_types = args.model_types
        hidden_sizes = args.hidden_sizes
        
        # Modification settings: control gradient clipping and learning rate decay
        use_modification = True  # Set to True to enable gradient clipping and LR decay, False to train in original way
        
        # Create results directory
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
            print(f"Created results directory: {results_dir}")
        
        # Training loop: 4 models × 3 hidden_sizes = 12 experiments
        total_experiments = len(model_types) * len(hidden_sizes)
        experiment_num = 0
        
        print(f"\n{'='*60}")
        print(f"Starting training loop: {total_experiments} experiments")
        print(f"Models: {model_types}")
        print(f"Hidden sizes: {hidden_sizes}")
        print(f"{'='*60}\n")
        
        for model_type in model_types:
            for hidden_size in hidden_sizes:
                experiment_num += 1
            print(f"\n{'='*60}")
            print(f"Experiment {experiment_num}/{total_experiments}: {model_type.upper()} with hidden_size={hidden_size}")
            print(f"{'='*60}\n")
            
            # Create model
            if model_type not in MODEL_CLASSES:
                print(f"Warning: Unsupported model type: {model_type}, skipping...")
                continue
            
            ModelClass = MODEL_CLASSES[model_type]
            
            # FFN model needs different parameters (input_channels)
            if model_type == "ffn":
                # FFN uses frame_num * channels as input_channels
                frame_num = train_ds_sector.frame_num  # Default 32
                chan_num = train_ds_sector.chan_num    # Default 2
                input_channels = frame_num * chan_num   # 64
                mdl = ModelClass(input_channels=input_channels, num_classes=10, num_pos=9, 
                               kernel_size=5, hidden_size=hidden_size)
                print(f"Created {model_type.upper()} model (num_pos=9, hidden_size={hidden_size}, input_channels={input_channels})")
            elif model_type == "gawf":
                # GaWFRNNConv needs dropout_rate parameter
                mdl = ModelClass(num_classes=10, num_pos=9, kernel_size=5, 
                               dropout_rate=0.3, hidden_size=hidden_size)
                print(f"Created {model_type.upper()} model (num_pos=9, hidden_size={hidden_size}, dropout_rate=0.3)")
            else:
                mdl = ModelClass(num_classes=10, num_pos=9, kernel_size=5, hidden_size=hidden_size)
                print(f"Created {model_type.upper()} model (num_pos=9, hidden_size={hidden_size})")
            
            # Train model
            print("Starting training...")
            print("Optimization settings:")
            print("  - Automatically find optimal batch_size")
            print("  - Use mixed precision training (FP16)")
            print("  - Optimize DataLoader (num_workers, pin_memory)")
            
            results = network_train(
                mdl, 
                train_ds_sector, 
                val_ds_sector, 
                num_epochs=args.num_epochs, 
                loss_weights=[1.0, 1.0],
                batch_size=None,  # Automatically find optimal value
                use_amp=True,     # Use mixed precision training
                num_workers=None, # Automatically set
                pin_memory=True,  # Accelerate data transfer
                use_modification=use_modification  # Control whether to enable modification settings
            )
            
            # Save training results
            print(f"\nSaving {model_type.upper()} (hidden_size={hidden_size}) results...")
            results_path = os.path.join(results_dir, f"{model_type}_sector_h{hidden_size}")
            
            save_results(results, results_path)
            print(f"Experiment {experiment_num}/{total_experiments} completed!\n")
        
        print(f"\n{'='*60}")
        print(f"All {total_experiments} experiments completed!")
        print(f"Results saved in: {results_dir}/")
        print(f"{'='*60}\n")
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user (Ctrl+C)")
        print("Exiting gracefully...")
        exit(0)
    except FileNotFoundError as e:
        print(f"\n\nERROR: File not found: {e}")
        print("Please check that the data paths are correct.")
        exit(1)
    except MemoryError as e:
        print(f"\n\nERROR: Out of memory: {e}")
        print("Try reducing batch_size or using a smaller model.")
        exit(1)
    except Exception as e:
        print(f"\n\nERROR: Unexpected error occurred: {e}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        exit(1)

