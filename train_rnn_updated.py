"""
Standalone RNN Sector training script
Used to train RNN models and save results
"""
import os
import gc
import sys
import signal
import argparse
import pickle
from collections import Counter
from itertools import product
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

torch.set_num_threads(4)


# ==================== Acceleration Configuration ====================
class AccelerationConfig:
    """
    Encapsulated acceleration module configuration for optimized training.
    Supports AMP, gradient accumulation, memory management, and DataLoader optimization.
    Can be enabled/disabled to isolate performance impact.
    """
    def __init__(self, use_acceleration=False, enable_amp=True, enable_dataloader_opt=True, 
                 enable_batch_auto=True, enable_gradient_scale=True, enable_grad_accum=False,
                 grad_accum_steps=4, enable_memory_opt=True, dataloader_prefetch_factor=2):
        """
        Args:
            use_acceleration: Master switch for all acceleration features
            enable_amp: Enable automatic mixed precision (AMP) for float16 computation
            enable_dataloader_opt: Enable multi-worker DataLoader optimization
            enable_batch_auto: Enable automatic batch size optimization
            enable_gradient_scale: Enable gradient scaling (used with AMP)
            enable_grad_accum: Enable gradient accumulation for effective larger batch size
            grad_accum_steps: Number of accumulation steps (effective_batch = batch_size * grad_accum_steps)
            enable_memory_opt: Enable memory optimization (cache clearing, gradient checkpointing prep)
            dataloader_prefetch_factor: Prefetch factor for DataLoader (reduces 2 if memory tight)
        """
        self.use_acceleration = use_acceleration
        self.enable_amp = use_acceleration and enable_amp
        self.enable_dataloader_opt = use_acceleration and enable_dataloader_opt
        self.enable_batch_auto = use_acceleration and enable_batch_auto
        self.enable_gradient_scale = use_acceleration and enable_gradient_scale
        self.enable_grad_accum = use_acceleration and enable_grad_accum
        self.grad_accum_steps = grad_accum_steps if self.enable_grad_accum else 1
        self.enable_memory_opt = use_acceleration and enable_memory_opt
        self.dataloader_prefetch_factor = dataloader_prefetch_factor if use_acceleration else 2
        
        # If use_acceleration=False, all sub-features are disabled
        if not use_acceleration:
            self.enable_amp = False
            self.enable_dataloader_opt = False
            self.enable_batch_auto = False
            self.enable_gradient_scale = False
            self.enable_grad_accum = False
            self.grad_accum_steps = 1
            self.enable_memory_opt = False
            self.dataloader_prefetch_factor = 2
    
    def summary(self):
        """Print acceleration configuration summary."""
        print("\n" + "="*60)
        print("Acceleration Configuration:")
        print(f"  Master switch (use_acceleration): {self.use_acceleration}")
        if self.use_acceleration:
            print(f"  - AMP (Automatic Mixed Precision): {self.enable_amp}")
            print(f"  - DataLoader Optimization: {self.enable_dataloader_opt}")
            print(f"  - Batch Size Auto Optimization: {self.enable_batch_auto}")
            print(f"  - Gradient Scaling: {self.enable_gradient_scale}")
            print(f"  - Gradient Accumulation: {self.enable_grad_accum} (steps={self.grad_accum_steps})")
            print(f"  - Memory Optimization: {self.enable_memory_opt}")
            print(f"  - DataLoader Prefetch Factor: {self.dataloader_prefetch_factor}")
        else:
            print("  All acceleration features disabled")
        print("="*60 + "\n")


# ==================== Acceleration Training Modules (Optional) ====================
# These modules are only used when acceleration is enabled
# By default (use_acceleration=False), they are not imported to keep the code clean

def _init_acceleration_modules():
    """Initialize acceleration training related modules (imported only when needed)"""
    try:
        from torch.amp import autocast, GradScaler
        try:
            import psutil
        except ImportError:
            psutil = None
        return autocast, GradScaler, psutil
    except ImportError:
        return None, None, None

def _get_gpu_memory_usage():
    """Get current GPU memory usage (only used in acceleration mode)"""
    if not torch.cuda.is_available():
        return 0.0
    dev = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(dev) / 1024**3
    reserved = torch.cuda.memory_reserved(dev) / 1024**3
    total = torch.cuda.get_device_properties(dev).total_memory / 1024**3
    return (reserved / total) * 100.0 if total > 0 else 0.0

def _find_optimal_batch_size(model, train_data, device='cuda', start_batch_size=32, max_batch_size=256, 
                              enable_grad_accum=False, grad_accum_steps=4):
    """
    Automatically find optimal batch_size (only used in acceleration mode)
    Uses smaller batch_size with gradient accumulation to save memory while maintaining effective batch size.
    
    Args:
        model: Model
        train_data: Training dataset
        device: Device
        start_batch_size: Starting batch_size
        max_batch_size: Maximum batch_size
        enable_grad_accum: Whether to enable gradient accumulation
        grad_accum_steps: Gradient accumulation steps
    
    Returns:
        Optimal batch_size, and adjusted num_workers if memory is tight
    """
    if device == 'cpu':
        return start_batch_size, 0
    
    model.eval()
    optimal_batch_size = start_batch_size
    num_workers_adjusted = 0  # Conservative: start with no workers
    
    # Test different batch sizes
    test_sizes = [start_batch_size, 64, 128, 256] if not enable_grad_accum else [start_batch_size, 32, 16, 8]
    
    for batch_size in test_sizes:
        if batch_size > max_batch_size:
            break
        
        try:
            torch.cuda.empty_cache()
            test_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=0)
            test_batch = next(iter(test_loader))
            inputs, labels = test_batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                _ = model(inputs)
            
            memory_usage = _get_gpu_memory_usage()
            
            if memory_usage < 70.0:  # Conservative threshold to avoid OOM
                optimal_batch_size = batch_size
                num_workers_adjusted = 2 if enable_grad_accum else 4  # Allow more workers with grad accum
                print(f"Testing batch_size={batch_size}: GPU memory usage {memory_usage:.1f}%, usable (num_workers will be {num_workers_adjusted})")
            else:
                print(f"Testing batch_size={batch_size}: GPU memory usage {memory_usage:.1f}%, exceeds limit")
                break
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"batch_size={batch_size} caused OOM, using batch_size={optimal_batch_size}")
                torch.cuda.empty_cache()
                break
            else:
                raise e
        finally:
            # Explicit cleanup of test_loader to prevent resource leaks
            if 'test_loader' in locals():
                del test_loader
                import gc
                gc.collect()
    
    torch.cuda.empty_cache()
    model.train()
    return optimal_batch_size, num_workers_adjusted


# ==================== Dataset Class ====================
class MC_RNN_Dataset(Dataset):
    def __init__(self, data, labels, frame_num=32, chan_num=2, use_sector=False, num_sectors=9, 
                 max_chars=10, predict_all_chars=False):
        """
        Args:
            data (np.ndarray): Array of shape (num_samples, num_frames, height, width)
            labels (np.ndarray): DataFrame with columns ['fg_char_id', 'fg_char_x', 'fg_char_y', 'bg_char_ids']
            frame_num (int): Number of frames to stack for input as multichannel image
            chan_num (int): Number of channels in the input images. Each channel is a previous frame.
            use_sector (bool): If True, map (x, y) position to sector id 0-(num_sectors-1)
            num_sectors (int): Number of sectors, e.g., 9 means 0-8 sectors (3x3 grid)
            max_chars (int): Maximum number of characters per frame (for padding)
            predict_all_chars (bool): If True, predict all characters (fg+bg), else only fg
        """
        self.data = data
        self.frame_num = frame_num
        self.chan_num = chan_num
        self.use_sector = use_sector
        self.num_sectors = num_sectors
        self.max_chars = max_chars
        self.predict_all_chars = predict_all_chars
        
        if predict_all_chars:
            # Store full DataFrame to access bg_char_ids
            self.labels_df = labels
            # Extract columns we need
            self.fg_char_ids = labels['fg_char_id'].values
            self.bg_char_ids_str = labels['bg_char_ids'].values
        else:
            # Original behavior: only fg char
            self.labels = labels[['fg_char_id', 'fg_char_x', 'fg_char_y']].values

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

        if self.predict_all_chars:
            # New mode: predict all characters (fg + bg)
            # Process each frame to extract all characters
            all_chars_per_frame = []
            for frame_idx in range(start_idx, end_idx):
                # Get fg char
                fg_char_id = int(self.fg_char_ids[frame_idx])
                
                # Get bg chars from comma-separated string
                bg_chars_str = str(self.bg_char_ids_str[frame_idx])
                if bg_chars_str and bg_chars_str != 'nan':
                    bg_char_ids = [int(x) for x in bg_chars_str.split(',') if x.strip()]
                else:
                    bg_char_ids = []
                
                # Combine fg and bg chars
                all_chars = [fg_char_id] + bg_char_ids
                
                # Pad to max_chars with -1 (no character)
                padded_chars = all_chars[:self.max_chars] + [-1] * max(0, self.max_chars - len(all_chars))
                all_chars_per_frame.append(padded_chars)
            
            # Convert to numpy array: (frame_num, max_chars)
            labels = np.array(all_chars_per_frame, dtype=np.int64)
        else:
            # Original mode: only predict fg char
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

        return stacked_frames, labels


# ==================== Model Classes ====================
class BaseRNNConv(nn.Module):
    """Base class for CNN-RNN models supporting different RNN types"""
    
    def __init__(self, num_classes, num_pos, rnn_class=nn.RNN, kernel_size=3, device='cuda', 
                 dropout_rate=0.3, hidden_size=256, max_chars=10, predict_all_chars=False):
        super(BaseRNNConv, self).__init__()
        self.device = device
        self.dropout_rate = dropout_rate
        self.max_chars = max_chars
        self.predict_all_chars = predict_all_chars
        self.conv1 = nn.Conv2d(2, 32, kernel_size=kernel_size, padding='same')
        self.MP1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.LNorm1 = nn.LayerNorm([32, 48, 48])
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.MP2 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.LNorm2 = nn.LayerNorm([64, 12, 12])
        self.rnn = rnn_class(input_size=64 * 12 * 12, hidden_size=hidden_size,
                             num_layers=1, batch_first=True)
        self.LNormRNN = nn.LayerNorm(hidden_size)
        
        if predict_all_chars:
            self.fcchars = nn.Linear(hidden_size, max_chars * num_classes)
            self.fcpos = None
        else:
            self.fcchar = nn.Linear(hidden_size, num_classes)
            self.fcpos = nn.Linear(hidden_size, num_pos)
        self.to(self.device)

    def encoder(self, x):
        x = self.conv1(x)
        x = self.MP1(x)
        x = self.LNorm1(x)
        x = F.relu(x)
        x = F.dropout2d(x, p=self.dropout_rate, training=self.training)
        
        x = self.conv2(x)
        x = self.MP2(x)
        x = self.LNorm2(x)
        x = F.relu(x)
        x = F.dropout2d(x, p=self.dropout_rate, training=self.training)
        return x

    def middle(self, x):
        x = self.rnn(x)[0]
        x = self.LNormRNN(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        return x

    def classifier(self, x):
        if self.predict_all_chars:
            chars_out = self.fcchars(x)
            batch_size, frame_num = chars_out.shape[:2]
            num_classes = chars_out.shape[-1] // self.max_chars
            chars_out = chars_out.view(batch_size, frame_num, self.max_chars, num_classes)
            return chars_out, None
        else:
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


class RNNConv(BaseRNNConv):
    def __init__(self, num_classes, num_pos, kernel_size=3, device='cuda', dropout_rate=0.3, hidden_size=256, 
                 max_chars=10, predict_all_chars=False):
        super(RNNConv, self).__init__(num_classes, num_pos, rnn_class=nn.RNN, kernel_size=kernel_size,
                                      device=device, dropout_rate=dropout_rate, hidden_size=hidden_size,
                                      max_chars=max_chars, predict_all_chars=predict_all_chars)


class GRUConv(BaseRNNConv):
    def __init__(self, num_classes, num_pos, kernel_size=3, device='cuda', dropout_rate=0.3, hidden_size=256,
                 max_chars=10, predict_all_chars=False):
        super(GRUConv, self).__init__(num_classes, num_pos, rnn_class=nn.GRU, kernel_size=kernel_size,
                                      device=device, dropout_rate=dropout_rate, hidden_size=hidden_size,
                                      max_chars=max_chars, predict_all_chars=predict_all_chars)


class LSTMConv(BaseRNNConv):
    def __init__(self, num_classes, num_pos, kernel_size=3, device='cuda', dropout_rate=0.3, hidden_size=256,
                 max_chars=10, predict_all_chars=False):
        super(LSTMConv, self).__init__(num_classes, num_pos, rnn_class=nn.LSTM, kernel_size=kernel_size,
                                       device=device, dropout_rate=dropout_rate, hidden_size=hidden_size,
                                       max_chars=max_chars, predict_all_chars=predict_all_chars)


class GaWFRNNConv(nn.Module):
    """
    GaWF (Gated with Feedback) RNN Model
    
    Main improvements:
    1. Use classifier output as feedback to RNN input
    2. Feedback is transformed by U @ diag(concat) @ V, then Hadamard product with RNN weights
    """
    def __init__(self, num_classes, num_pos, kernel_size=3, device='cuda', dropout_rate=0.3):
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
        hidden_size = 256
        
        # Create RNN (but not using built-in, manually implement to support feedback)
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                          num_layers=1, batch_first=True)
        
        # Feedback transformation matrices
        # Dimension after concatenating classifier outputs
        feedback_dim = num_classes + num_pos  # e.g., 10 + 9 = 19
        
        # RNN weight matrix shapes
        # weight_ih: (hidden_size, input_size) = (256, 9216)
        # weight_hh: (hidden_size, hidden_size) = (256, 256)
        # Concatenated shape: (256, 9216 + 256) = (256, 9472)
        combined_weight_size = input_size + hidden_size  # 9472
        
        # U: (hidden_size, feedback_dim) = (256, 19)
        # V: (feedback_dim, combined_weight_size) = (19, 9472)
        # diag(concat): (feedback_dim, feedback_dim) = (19, 19)
        # U @ diag @ V: (256, 19) @ (19, 19) @ (19, 9472) = (256, 9472)
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
        x = F.dropout2d(x, p=self.dropout_rate, training=self.training)
        
        x = self.conv2(x)
        x = self.MP2(x)
        x = self.LNorm2(x)
        x = F.relu(x)
        x = F.dropout2d(x, p=self.dropout_rate, training=self.training)
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
                # fb_t: (B, feedback_dim)
                # For each sample b, compute U @ diag(fb_t[b]) @ V
                # Equivalent to: U @ (fb_t[b] * I) @ V, where I is identity matrix
                # Can be written as: U @ (fb_t[b].unsqueeze(1) * V), but this is incorrect
                # Correct way: For diag(fb_t[b]), fb_t[b] serves as diagonal elements
                # U @ diag(fb_t[b]) @ V = U @ (torch.diag(fb_t[b])) @ V
                # Can be vectorized: use batch matrix multiplication
                
                # Method: Build diagonal matrix for each sample separately and compute (current implementation)
                # But can use more efficient vectorization
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
                # Compute for each sample separately
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
            x, _ = self.rnn(x)
            x = self.LNormRNN(x)
            x = F.relu(x)
        
        # Dropout
        x = F.dropout(x, p=0.5, training=self.training)
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
        
        # appl RNN
        x = self.middle(x, feedback=feedback)

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


# ==================== Training Function ====================
def network_train(mdl, train_data, val_data, num_epochs=50, loss_weights=None, lr=0.001, 
                  use_acceleration=False, use_modification=False, 
                  weight_decay=None, dropout_rate=None, rnn_diag_lambda=1e-4, use_tqdm=True):
    """
    Train model, supports sector mode and coordinate mode
    
    Args:
        mdl: Model
        train_data: Training dataset (MC_RNN_Dataset)
        val_data: Validation dataset (MC_RNN_Dataset)
        num_epochs: Number of training epochs
        loss_weights: [character loss weight, position loss weight], if None, automatically set based on use_sector
                     - sector mode default: [1, 1]
                     - coordinate mode default: [1, 0.001]
        lr: Learning rate
        use_acceleration: Whether to use acceleration training (default False)
                         - True: Enable mixed precision training (AMP), automatic batch_size optimization, 
                                 gradient accumulation for larger effective batch size, DataLoader optimization, etc.
                         - False: Use original training method, does not affect existing logic
                         - Features enabled when True: AMP (float16), gradient accumulation, memory optimization,
                           multi-worker DataLoader with prefetch, CPU-GPU pinned memory
        use_modification: Whether to use modification settings (weight_decay, dropout_rate)
                         - True: Use modification settings (default: weight_decay=1e-4, dropout_rate=0.3)
                         - False: Disable all modifications (weight_decay=0.0, dropout_rate=0.0)
                         - If True, you can still override with custom weight_decay, dropout_rate values
        weight_decay: L2 regularization coefficient (weight decay)
                     - If use_modification=False: ignored, set to 0.0
                     - If use_modification=True and None: default 1e-4
                     - If use_modification=True and specified: use custom value
        dropout_rate: Dropout rate (note: this is only for reference, actual dropout is set when creating the model)
                     - If use_modification=False: ignored, set to 0.0
                     - If use_modification=True and None: default 0.3
                     - If use_modification=True and specified: use custom value
        rnn_diag_lambda: Regularization coefficient for RNN hidden-to-hidden diagonal weight regularization, default 1e-4
                        This adds L1 penalty on the diagonal elements of RNN hidden weight matrix to improve stability
    """
    # Get use_sector and predict_all_chars information from dataset
    use_sector = train_data.use_sector
    predict_all_chars = train_data.predict_all_chars
    max_chars = train_data.max_chars if predict_all_chars else None
    
    # Set default loss_weights based on mode
    if loss_weights is None:
        if predict_all_chars:
            # All-chars mode: only predict character identity, no position prediction
            loss_weights = [1, 0]  # Only character loss, no position loss
        elif use_sector:
            loss_weights = [1, 1]  # sector mode: character and sector loss weights equal
        else:
            loss_weights = [1, 0.001]  # coordinate mode: position loss weight smaller (MSE usually has larger values)
    
    # Handle modification settings: weight_decay, dropout_rate
    if use_modification:
        # If use_modification=True, use provided values or defaults
        if weight_decay is None:
            weight_decay = 1e-4  # Default weight decay
        if dropout_rate is None:
            dropout_rate = 0.3  # Default dropout rate (for reference, actual dropout is set in model)
        print(f"Modification settings enabled: weight_decay={weight_decay}, dropout_rate={dropout_rate}")
    else:
        # If use_modification=False, force disable all modifications
        weight_decay = 0.0
        dropout_rate = 0.0
        print("Modification settings disabled: weight_decay=0.0, dropout_rate=0.0")
    
    # Place parameters according to model's internal device (can be 'cuda' or 'cpu')
    device = mdl.device
    mdl.to(device)
    
    # ========== Acceleration Configuration ==========
    accel_config = AccelerationConfig(use_acceleration=use_acceleration)
    accel_config.summary()
    
    # ========== Acceleration Training Module Initialization (only used when acceleration is enabled) ==========
    autocast_fn = None
    GradScaler_cls = None
    scaler = None
    psutil_module = None
    batch_size = 32  # Default batch_size
    num_workers = 0   # Default single process
    pin_memory = False  # Default not using pin_memory
    show_gpu_usage = False  # Default not showing GPU usage
    
    if use_acceleration:
        print("Enabling acceleration training...")
        autocast_fn, GradScaler_cls, psutil_module = _init_acceleration_modules()
        
        if autocast_fn is None or GradScaler_cls is None:
            print("Warning: Unable to import acceleration training modules, will use standard training")
            use_acceleration = False
        else:
            # Automatically find optimal batch_size (Ubuntu only)
            # Note: GaWFRNNConv model skips batch_size search, uses default value 32
            # Because GaWFRNNConv uses feedback mechanism, batch_size changes cause prev_feedback dimension mismatch
            if device == 'cuda' and not isinstance(mdl, GaWFRNNConv):
                print("Automatically finding optimal batch_size (with gradient accumulation support)...")
                batch_size, num_workers = _find_optimal_batch_size(
                    mdl, train_data, device=device, start_batch_size=32,
                    enable_grad_accum=accel_config.enable_grad_accum, 
                    grad_accum_steps=accel_config.grad_accum_steps
                )
                print(f"Using batch_size = {batch_size}, suggested num_workers = {num_workers}")
            elif isinstance(mdl, GaWFRNNConv):
                print(f"Detected GaWFRNNConv model, skipping batch_size search, using default batch_size = {batch_size}")
            
            # Set num_workers for Ubuntu environment (may be overridden by batch_size search)
            if num_workers == 0:  # Only set if not already set
                if psutil_module is not None:
                    num_workers = min(2, psutil_module.cpu_count(logical=False))  # Conservative default
                else:
                    num_workers = min(2, os.cpu_count() or 1)
            print(f"Ubuntu environment: num_workers={num_workers}, batch_size={batch_size}")
            
            # Enable pin_memory (GPU only)
            pin_memory = False
            show_gpu_usage = True
            
            # Initialize mixed precision training scaler
            if device == 'cuda':
                scaler = GradScaler_cls('cuda')
                print(f"Acceleration settings: batch_size={batch_size}, num_workers={num_workers}, "
                      f"pin_memory={pin_memory}, mixed precision training=enabled")
    # ========== Acceleration Module Initialization End ==========
    
    # Add weight decay (L2 regularization) to prevent overfitting
    optim = torch.optim.Adam(mdl.parameters(), lr=lr, weight_decay=weight_decay)
    criterion_char = nn.CrossEntropyLoss()
    
    # Select position loss function based on use_sector
    if use_sector:
        criterion_pos = nn.CrossEntropyLoss()  # sector classification
    else:
        criterion_pos = nn.MSELoss()  # coordinate regression
    
    def loss_fn(out_char, out_pos, labels):
        if predict_all_chars:
            # New mode: predict all characters (fg + bg)
            # out_char: (B, T, max_chars, num_classes)
            # labels: (B, T, max_chars) - each position is char_id or -1 (no char)
            batch_size, frame_num, max_chars_pred, num_classes = out_char.shape
            
            # Vectorized: compute softmax for all frames at once
            pred_probs = F.softmax(out_char, dim=-1)  # (B, T, max_chars, num_classes)
            
            total_loss = 0.0
            total_valid_chars = 0
            
            # Process each frame independently (still need loop for greedy matching logic)
            # But use vectorized operations within each frame
            for b in range(batch_size):
                for t in range(frame_num):
                    # Get true characters for this frame (remove padding -1)
                    true_chars = labels[b, t]  # (max_chars,)
                    valid_mask = true_chars >= 0  # (max_chars,)
                    valid_true_chars = true_chars[valid_mask]  # (num_valid_chars,)
                    
                    if len(valid_true_chars) == 0:
                        continue  # Skip frames with no characters
                    
                    # Get predictions for this frame (already computed softmax)
                    frame_probs = pred_probs[b, t]  # (max_chars, num_classes)
                    frame_logits = out_char[b, t]  # (max_chars, num_classes)
                    
                    # Vectorized greedy matching: use torch operations instead of Python loops
                    num_valid = len(valid_true_chars)
                    
                    # Create mask for used slots (vectorized)
                    used_mask = torch.zeros(max_chars, dtype=torch.bool, device=out_char.device)
                    
                    # Pre-allocate arrays for matched pairs
                    matched_pred_indices = []
                    matched_true_chars = []
                    
                    # Greedy matching: process each true char
                    for i, true_char_id in enumerate(valid_true_chars):
                        # Vectorized: get probabilities for this char across all slots
                        char_probs = frame_probs[:, true_char_id]  # (max_chars,)
                        
                        # Vectorized: mask out already used slots
                        char_probs = char_probs.masked_fill(used_mask, -1.0)
                        
                        # Find best matching slot (vectorized)
                        best_pred_idx = torch.argmax(char_probs).item()
                        
                        # Store matched pair
                        matched_pred_indices.append(best_pred_idx)
                        matched_true_chars.append(true_char_id)
                        
                        # Update used mask (vectorized)
                        used_mask[best_pred_idx] = True
                    
                    # Vectorized loss computation: batch all matched pairs
                    if len(matched_pred_indices) > 0:
                        # Convert to tensors for vectorized operations
                        matched_pred_indices_tensor = torch.tensor(matched_pred_indices, device=out_char.device)
                        matched_true_chars_tensor = torch.tensor(matched_true_chars, dtype=torch.long, device=out_char.device)
                        
                        # Gather matched logits: (num_matched, num_classes)
                        matched_logits = frame_logits[matched_pred_indices_tensor]  # (num_matched, num_classes)
                        
                        # Compute loss for all matched pairs at once (vectorized)
                        batch_loss = criterion_char(matched_logits, matched_true_chars_tensor)
                        total_loss += batch_loss
                        total_valid_chars += len(matched_pred_indices)
            
            if total_valid_chars == 0:
                loss_char = torch.tensor(0.0, device=out_char.device)
            else:
                loss_char = total_loss / total_valid_chars
            
            # No position loss in all-chars mode
            loss_pos = torch.tensor(0.0, device=out_char.device)
        else:
            # Original mode: single char + position
            # Character loss (same for both modes)
            labels_char = labels[:, :, 0].long().view(-1)
            outputs_char = out_char.view(-1, out_char.shape[-1])  # (B*T, num_classes)
            loss_char = criterion_char(outputs_char, labels_char)
            
            # Position loss (different methods based on use_sector)
            if use_sector:
                # sector mode: classification loss
                labels_pos = labels[:, :, 1].long().view(-1)
                outputs_pos = out_pos.view(-1, out_pos.shape[-1])  # (B*T, num_sectors)
                loss_pos = criterion_pos(outputs_pos, labels_pos)
            else:
                # coordinate mode: regression loss (MSE)
                labels_pos = labels[:, :, 1:].float()  # (B, T, 2) -> [x, y]
                outputs_pos = out_pos  # (B, T, 2) -> [x, y]
                loss_pos = criterion_pos(outputs_pos, labels_pos)

        # Keep regularization consistent with original (if model doesn't have mdl.rnn, need corresponding modification)
        if hasattr(mdl, 'rnn') and mdl.rnn is not None:
            rnn_hh = mdl.rnn.weight_hh_l0
            # rnn_hh_diag = torch.diagonal(rnn_hh).abs().sum()
            rnn_hh_diag = torch.diagonal(rnn_hh).abs().mean()   # ✅ mean 更稳定
        else:
            rnn_hh_diag = torch.tensor(0.0, device=out_char.device)
        
        # loss = (loss_weights[0] * loss_char) + (loss_weights[1] * loss_pos) + rnn_hh_diag
        loss = (loss_weights[0] * loss_char) + (loss_weights[1] * loss_pos) + rnn_diag_lambda * rnn_hh_diag
        return loss

    def evaluate(mdl, data_loader, use_tqdm):
        mdl.eval()
        total_acc_char = 0
        total_metric_pos = 0  # sector mode: accuracy; coordinate mode: MSE
        total_frames = 0
        # all-chars exact-match stats (scheme B): Counter(pred)==Counter(gt) per frame
        total_frames_eval = 0
        total_frames_exact = 0
        
        # Use tqdm for validation progress bar
        val_pbar = tqdm(data_loader, desc="[Val]", ncols=100, leave=False, unit="batch", disable=not use_tqdm)
        
        with torch.no_grad():
            for batch in val_pbar:
                inputs, labels = batch
                
                # Reset feedback at start of each batch (if GaWFRNNConv)
                # This ensures feedback's batch_size and seq_len match current batch
                if hasattr(mdl, 'prev_feedback'):
                    mdl.prev_feedback = None
                
                # Select data transfer method based on acceleration mode
                if use_acceleration and pin_memory:
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                else:
                    labels = labels.to(device)
                
                # Select whether to use mixed precision based on acceleration mode
                if use_acceleration and scaler is not None and autocast_fn is not None:
                    with autocast_fn('cuda'):
                        out_char, out_pos = mdl(inputs)
                else:
                    out_char, out_pos = mdl(inputs)
                
                if predict_all_chars:
                    # New mode: evaluate all characters prediction
                    # out_char: (B, T, max_chars, num_classes)
                    # labels: (B, T, max_chars)
                    batch_size, frame_num = labels.shape[:2]
                    total_frames += batch_size * frame_num
                    
                    # Vectorized: compute softmax for all frames at once
                    pred_probs = F.softmax(out_char, dim=-1)  # (B, T, max_chars, num_classes)
                    
                    # For each frame, match predictions to true chars and compute accuracy
                    for b in range(batch_size):
                        for t in range(frame_num):
                            true_chars = labels[b, t]  # (max_chars,)
                            valid_mask = true_chars >= 0
                            valid_true_chars = true_chars[valid_mask]
                            
                            if len(valid_true_chars) == 0:
                                continue
                            total_frames_eval += 1
                            
                            frame_probs = pred_probs[b, t]  # (max_chars, num_classes)
                            frame_logits = out_char[b, t]  # (max_chars, num_classes)
                            
                            # Greedy matching to build predicted chars list (same strategy as notebook)
                            used_mask = torch.zeros(max_chars, dtype=torch.bool, device=out_char.device)
                            matched_pred_chars = []

                            for true_char_id in valid_true_chars:
                                # Get probabilities for this GT char over all slots, mask used slots
                                char_probs = frame_probs[:, int(true_char_id)]
                                char_probs = char_probs.masked_fill(used_mask, -1.0)

                                # Find best matching slot
                                best_pred_idx = torch.argmax(char_probs).item()

                                # Predicted char at this slot
                                pred_char = torch.argmax(frame_logits[best_pred_idx]).item()
                                matched_pred_chars.append(pred_char)

                                # Mark slot as used
                                used_mask[best_pred_idx] = True

                            # Exact multiset/frame match using Counter, same as notebook logic
                            gt_chars = [int(c.item()) if isinstance(c, torch.Tensor) else int(c) for c in valid_true_chars]
                            pred_counter = Counter(matched_pred_chars)
                            gt_counter = Counter(gt_chars)
                            if pred_counter == gt_counter:
                                total_frames_exact += 1
                else:
                    # Original mode: single char + position
                    # char accuracy (same for both modes)
                    total_acc_char += (torch.argmax(out_char, dim=2) == labels[:, :, 0].long()).float().mean().item()
                    
                    # Position-related metrics (different methods based on use_sector)
                    if use_sector:
                        # sector mode: calculate accuracy
                        total_metric_pos += (torch.argmax(out_pos, dim=2) == labels[:, :, 1].long()).float().mean().item()
                    else:
                        # coordinate mode: calculate MSE
                        labels_pos = labels[:, :, 1:].float()  # (B, T, 2)
                        total_metric_pos += F.mse_loss(out_pos, labels_pos, reduction='mean').item()
        
        # Return results
        if predict_all_chars:
            # Scheme B: exact frame match rate (always in [0, 100])
            acc_char = (total_frames_exact / total_frames_eval) * 100 if total_frames_eval > 0 else 0.0
            metric_pos = 0.0  # No position metric in all-chars mode
        else:
            acc_char = total_acc_char * 100 / len(data_loader)
            if use_sector:
                metric_pos = total_metric_pos * 100 / len(data_loader)  # accuracy (percentage)
            else:
                metric_pos = total_metric_pos / len(data_loader)  # MSE (pixel squared)
        return acc_char, metric_pos

    # data loader (select different configurations based on acceleration mode)
    if use_acceleration:
        # Memory-optimized DataLoader configuration
        # - num_workers: CPU parallelism (careful not to overallocate)
        # - pin_memory: Faster CPU-GPU transfer (only if num_workers > 0)
        # - persistent_workers: Reuse worker processes to avoid overhead
        # - prefetch_factor: How many batches to prefetch (reduce if memory tight)
        # Note: Do NOT use num_workers with mmap_mode='r' (memory mapped data) 
        # as it causes memory duplication across processes
        
        train_dl = DataLoader(
            train_data, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory and num_workers > 0,  # Only pin if using workers
            persistent_workers=num_workers > 0,
            prefetch_factor=accel_config.dataloader_prefetch_factor if num_workers > 0 else 2,
            drop_last=True  # Drop last incomplete batch to avoid small batch issues
        )
        val_dl = DataLoader(
            val_data, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory and num_workers > 0,
            persistent_workers=num_workers > 0,
            prefetch_factor=accel_config.dataloader_prefetch_factor if num_workers > 0 else 2,
            drop_last=False  # Keep last incomplete batch for validation
        )
    else:
        # Original method: simple configuration (no workers to avoid memory overhead)
        train_dl = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
        val_dl = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=0)
    
    # Print dataset and batch information
    print(f"\nDataset Information:")
    print(f"  Training dataset size: {len(train_data)} samples")
    print(f"  Validation dataset size: {len(val_data)} samples")
    print(f"  Batch size per step: {batch_size}")
    effective_batch_size = batch_size * accel_config.grad_accum_steps
    print(f"  Effective batch size (with grad accum): {effective_batch_size}")
    print(f"  Number of batches per epoch: {len(train_dl)}")
    print(f"  use_sector mode: {use_sector}")
    print(f"  predict_all_chars mode: {predict_all_chars}")
    print(f"  use acceleration mode: {use_acceleration}")
    if use_acceleration:
        print(f"  Workers: {num_workers}, Pin Memory: {pin_memory and num_workers > 0}")
        print(f"  DataLoader Prefetch: {accel_config.dataloader_prefetch_factor}")
    print()
    train_acc_char = np.zeros(num_epochs)
    val_acc_char = np.zeros(num_epochs)
    train_metric_pos = np.zeros(num_epochs)  # sector mode: accuracy; coordinate mode: MSE
    val_metric_pos = np.zeros(num_epochs)

    # # Signal handler for graceful shutdown to prevent semaphore leaks
    # def cleanup_dataloaders(signum, frame):
    #     """Clean up DataLoader workers when receiving termination signal"""
    #     if use_acceleration and num_workers > 0:
    #         print("\nReceived termination signal, cleaning up DataLoader workers...")
    #         try:
    #             if 'train_dl' in locals():
    #                 train_dl._iterator = None
    #             if 'val_dl' in locals():
    #                 val_dl._iterator = None
    #             import gc
    #             gc.collect()
    #         except:
    #             pass
    #     sys.exit(0)
    
    # # Register signal handlers for graceful cleanup (only when using multiprocessing)
    # if use_acceleration and num_workers > 0:
    #     signal.signal(signal.SIGTERM, cleanup_dataloaders)
    #     signal.signal(signal.SIGINT, cleanup_dataloaders)

    _STOP_REQUESTED = False

    def _request_stop(signum, frame):
        global _STOP_REQUESTED
        _STOP_REQUESTED = True
        # 不做任何 DataLoader 操作！不 sys.exit！
        print(f"\n[signal] got {signum}, will stop after current step...", flush=True)

    if use_acceleration and num_workers > 0:
        signal.signal(signal.SIGTERM, _request_stop)
        signal.signal(signal.SIGINT, _request_stop)

    try:
        for epoch in range(num_epochs):
            mdl.train()
            
            # Training loop
            epoch_train_acc_char = 0.0
            epoch_train_metric_pos = 0.0
            # all-chars exact-match denominator (scheme B): number of evaluated frames (GT has >=1 char)
            epoch_train_frames_eval = 0
            num_batches = 0
            
            if use_tqdm:
                print(f"Epoch {epoch + 1}/{num_epochs}: Starting training...", flush=True)
            
            # Use tqdm for progress bar
            train_pbar = tqdm(enumerate(train_dl), total=len(train_dl), 
                             desc=f"Epoch {epoch + 1}/{num_epochs} [Train]",
                             ncols=100, leave=False, disable=not use_tqdm)
            
            for batch_idx, batch in train_pbar:

                if _STOP_REQUESTED:
                    raise KeyboardInterrupt

                inputs, labels = batch
                
                # Reset feedback at start of each batch (if GaWFRNNConv)
                # This ensures feedback's batch_size and seq_len match current batch
                if hasattr(mdl, 'prev_feedback'):
                    mdl.prev_feedback = None
                
                # Select data transfer method based on acceleration mode
                if use_acceleration and pin_memory:
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                else:
                    labels = labels.to(device)
                
                # Gradient accumulation: zero_grad only at the first step of accumulation
                if batch_idx % accel_config.grad_accum_steps == 0:
                    optim.zero_grad()
                
                # Normalize loss by accumulation steps for proper gradient scaling
                loss_scale = 1.0 / accel_config.grad_accum_steps
                
                # Select whether to use mixed precision training based on acceleration mode
                if use_acceleration and scaler is not None and autocast_fn is not None:
                    with autocast_fn('cuda'):
                        out_char, out_pos = mdl(inputs)
                        loss = loss_fn(out_char, out_pos, labels)
                        loss = loss * loss_scale  # Scale loss for accumulation
                    
                    scaler.scale(loss).backward()
                    # Optimizer step and scaling update only at accumulation boundary
                    if (batch_idx + 1) % accel_config.grad_accum_steps == 0:
                        scaler.unscale_(optim)
                        torch.nn.utils.clip_grad_norm_(mdl.parameters(), max_norm=2.0)
                        scaler.step(optim)
                        scaler.update()
                    current_loss = loss.item() / loss_scale  # Unscale for logging
                else:
                    # Original method: standard training
                    out_char, out_pos = mdl(inputs)
                    loss = loss_fn(out_char, out_pos, labels)
                    loss = loss * loss_scale  # Scale loss for accumulation
                    loss.backward()
                    
                    # Optimizer step only at accumulation boundary
                    if (batch_idx + 1) % accel_config.grad_accum_steps == 0:
                        torch.nn.utils.clip_grad_norm_(mdl.parameters(), max_norm=2.0)
                        optim.step()
                    current_loss = loss.item() / loss_scale  # Unscale for logging
                
                # Calculate training metrics
                if predict_all_chars:
                    # All-chars mode: use greedy matching to compute accuracy
                    # Optimize: only compute accuracy every N batches to speed up training
                    # Changed from every 10 batches to every 50 batches for better performance
                    if batch_idx % 50 == 0 or batch_idx == len(train_dl) - 1:
                        batch_size, frame_num = labels.shape[:2]
                        # Scheme B (exact multiset / frame match) stats for this eval pass
                        batch_exact = 0
                        batch_frames_eval = 0
                        
                        # Vectorized: compute softmax for all frames at once
                        pred_probs = F.softmax(out_char, dim=-1)  # (B, T, max_chars, num_classes)
                        
                        for b in range(batch_size):
                            for t in range(frame_num):
                                true_chars = labels[b, t]  # (max_chars,)
                                valid_mask = true_chars >= 0
                                valid_true_chars = true_chars[valid_mask]
                                
                                if len(valid_true_chars) == 0:
                                    continue
                                batch_frames_eval += 1
                                
                                frame_probs = pred_probs[b, t]  # (max_chars, num_classes)
                                frame_logits = out_char[b, t]  # (max_chars, num_classes)
                                
                                # Greedy matching to build predicted chars list (same strategy as notebook)
                                used_mask = torch.zeros(max_chars, dtype=torch.bool, device=out_char.device)
                                matched_pred_chars = []

                                for true_char_id in valid_true_chars:
                                    # Get probabilities for this GT char over all slots, mask used slots
                                    char_probs = frame_probs[:, int(true_char_id)]
                                    char_probs = char_probs.masked_fill(used_mask, -1.0)

                                    # Find best matching slot
                                    best_pred_idx = torch.argmax(char_probs).item()

                                    # Predicted char at this slot
                                    pred_char = torch.argmax(frame_logits[best_pred_idx]).item()
                                    matched_pred_chars.append(pred_char)

                                    # Mark slot as used
                                    used_mask[best_pred_idx] = True

                                # Exact multiset/frame match using Counter, same as notebook logic
                                gt_chars = [int(c.item()) if isinstance(c, torch.Tensor) else int(c) for c in valid_true_chars]
                                pred_counter = Counter(matched_pred_chars)
                                gt_counter = Counter(gt_chars)
                                if pred_counter == gt_counter:
                                    batch_exact += 1

                        # Accumulate epoch-level exact-match counts (denominator is real evaluated frames)
                        if batch_frames_eval > 0:
                            epoch_train_acc_char += batch_exact
                            epoch_train_frames_eval += batch_frames_eval
                    # For batches we skip, don't add anything (will be averaged correctly later)
                    # No position metric in all-chars mode
                else:
                    # Original mode: single char + position
                    # Character accuracy (same for both modes)
                    epoch_train_acc_char += (torch.argmax(out_char, dim=2) == labels[:, :, 0].long()).float().mean().item()
                    
                    # Position-related metrics (different methods based on use_sector)
                    if use_sector:
                        # sector mode: calculate accuracy
                        epoch_train_metric_pos += (torch.argmax(out_pos, dim=2) == labels[:, :, 1].long()).float().mean().item()
                    else:
                        # coordinate mode: calculate MSE
                        labels_pos = labels[:, :, 1:].float()  # (B, T, 2)
                        epoch_train_metric_pos += F.mse_loss(out_pos, labels_pos, reduction='mean').item()
                
                num_batches += 1
                
                # Memory optimization: periodically clear GPU cache and unused memory
                if use_acceleration and accel_config.enable_memory_opt:
                    if batch_idx % 50 == 0 and device == 'cuda':
                        torch.cuda.empty_cache()  # Clear GPU cache
                        torch.cuda.synchronize()  # Ensure all GPU operations complete
                
                # Update progress bar (every 10 batches to avoid too frequent updates)
                if batch_idx % 10 == 0:
                    if predict_all_chars:
                        # All-chars mode: only show loss
                        train_pbar.set_postfix({'loss': f'{current_loss:.4f}'})
                    else:
                        # Original mode: show loss and pos info
                        # Calculate current batch pos metric for display
                        if use_sector:
                            # sector mode: calculate accuracy for current batch
                            batch_pos_acc = (torch.argmax(out_pos, dim=2) == labels[:, :, 1].long()).float().mean().item() * 100
                            train_pbar.set_postfix({
                                'loss': f'{current_loss:.4f}',
                                'pos_acc': f'{batch_pos_acc:.2f}%'
                            })
                        else:
                            # coordinate mode: calculate MSE for current batch
                            labels_pos = labels[:, :, 1:].float()  # (B, T, 2)
                            batch_pos_mse = F.mse_loss(out_pos, labels_pos, reduction='mean').item()
                            train_pbar.set_postfix({
                                'loss': f'{current_loss:.4f}',
                                'pos_mse': f'{batch_pos_mse:.2f}'
                            })
                
                # Acceleration mode: periodically clear GPU cache
                if use_acceleration and batch_idx % 100 == 0 and device == 'cuda':
                    torch.cuda.empty_cache()
            
            # Calculate epoch average metrics
            if predict_all_chars:
                # Scheme B: epoch_train_acc_char = total exact-matched frames
                # epoch_train_frames_eval = total evaluated frames (denominator)
                train_acc_char[epoch] = (epoch_train_acc_char / epoch_train_frames_eval) * 100 if epoch_train_frames_eval > 0 else 0.0
                train_metric_pos[epoch] = 0.0  # No position metric in all-chars mode
            elif use_sector:
                train_acc_char[epoch] = (epoch_train_acc_char / num_batches) * 100
                train_metric_pos[epoch] = (epoch_train_metric_pos / num_batches) * 100  # accuracy (percentage)
            else:
                train_acc_char[epoch] = (epoch_train_acc_char / num_batches) * 100
                train_metric_pos[epoch] = epoch_train_metric_pos / num_batches  # MSE (pixel squared)
            
            # Format output string
            gpu_info = ""
            if use_acceleration and show_gpu_usage and device == 'cuda' and torch.cuda.is_available():
                gpu_mem = _get_gpu_memory_usage()
                gpu_info = f" | GPU memory: {gpu_mem:.1f}%"
            
            # Format output based on mode (ensure predict_all_chars is checked first)
            if predict_all_chars:
                # All-chars mode: only show character accuracy, no position
                train_str = f"Epoch {epoch + 1}/{num_epochs} - Train (all chars acc): {train_acc_char[epoch]:.2f}%{gpu_info}"
            elif use_sector:
                train_str = f"Epoch {epoch + 1}/{num_epochs} - Train (char, sector): ({train_acc_char[epoch]:.2f}%, {train_metric_pos[epoch]:.2f}%){gpu_info}"
            else:
                train_str = f"Epoch {epoch + 1}/{num_epochs} - Train (char, pos): ({train_acc_char[epoch]:.2f}%, {train_metric_pos[epoch]:.2f} pix^2){gpu_info}"

            # Training completed, now validate
            with torch.no_grad():
                val_acc_char[epoch], val_metric_pos[epoch] = evaluate(mdl, val_dl, use_tqdm)
                # Format validation output based on mode (ensure predict_all_chars is checked first)
                if predict_all_chars:
                    # All-chars mode: only show character accuracy, no position
                    val_str = f" Validation (all chars acc): {val_acc_char[epoch]:.2f}%"
                elif use_sector:
                    val_str = f" Validation (char, sector): ({val_acc_char[epoch]:.2f}%, {val_metric_pos[epoch]:.2f}%)"
                else:
                    val_str = f" Validation (char, pos): ({val_acc_char[epoch]:.2f}%, {val_metric_pos[epoch]:.2f} pix^2)"
                print(train_str + val_str, flush=True)

    
    except (KeyboardInterrupt, SystemExit):
        # Handle interruption gracefully
        print("\nTraining interrupted, cleaning up resources...")
    finally:
        # Explicit resource cleanup for DataLoader to prevent semaphore leaks
        # This is critical when using persistent_workers=True with multiprocessing
        if use_acceleration and num_workers > 0:
            try:
                # Shutdown DataLoader workers properly to prevent semaphore leaks
                if 'train_dl' in locals():
                    train_dl._iterator = None
                    del train_dl
                if 'val_dl' in locals():
                    val_dl._iterator = None
                    del val_dl
                gc.collect()
                print("DataLoader workers cleaned up successfully")
            except Exception as e:
                print(f"Warning: Error during DataLoader cleanup: {e}")

    torch.cuda.empty_cache()

    # If early stopping triggered, only return actual trained epochs (epoch starts from 0, so actually trained epoch+1 epochs)
    actual_epochs = epoch + 1
    
    # Return different key names based on mode, only return actual trained epochs
    if predict_all_chars:
        return {
            "train_acc_char": train_acc_char[:actual_epochs],
            "val_acc_char": val_acc_char[:actual_epochs],
            "model": mdl.to("cpu"),
            "actual_epochs": actual_epochs  # save actual trained epochs
        }
    elif use_sector:
        return {
            "train_acc_char": train_acc_char[:actual_epochs],
            "val_acc_char": val_acc_char[:actual_epochs],
            "train_acc_pos": train_metric_pos[:actual_epochs],  # sector accuracy
            "val_acc_pos": val_metric_pos[:actual_epochs],      # sector accuracy
            "model": mdl.to("cpu"),
            "actual_epochs": actual_epochs  # save actual trained epochs
        }
    else:
        return {
            "train_acc_char": train_acc_char[:actual_epochs],
            "val_acc_char": val_acc_char[:actual_epochs],
            "train_err_pos": train_metric_pos[:actual_epochs],  # coordinate MSE
            "val_err_pos": val_metric_pos[:actual_epochs],      # coordinate MSE
            "model": mdl.to("cpu"),
            "actual_epochs": actual_epochs  # save actual trained epochs
        }


# ==================== Utility Functions ====================
def save_results(results, filepath):
    """
    Save training results to local file
    
    Args:
        results: Training results dictionary
        filepath: Save path (e.g., 'results_rnn' or 'results/rnn_sector')
                  If directory doesn't exist, it will be created
    """
    import os
    # Extract directory and filename
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Create save dictionary (does not include model, because model is too large)
    results_path = filepath + '.pkl'
    save_dict = {}
    for key, value in results.items():
        if key != "model":
            save_dict[key] = value
    
    # Save model state dict (can be saved separately if needed)
    model_path = results_path.replace('.pkl', '_model.pth')
    if "model" in results:
        # Check if model has fcpos (position prediction layer)
        # In predict_all_chars mode, fcpos is None
        if hasattr(results["model"], 'fcpos') and results["model"].fcpos is not None:
            saved_num_pos = results["model"].fcpos.out_features
            print(f"Model has position prediction layer (num_pos={saved_num_pos})")
        elif hasattr(results["model"], 'predict_all_chars') and results["model"].predict_all_chars:
            print("Model is in predict_all_chars mode (no position prediction)")
        else:
            print("Model does not have position prediction layer")
        torch.save(results["model"].state_dict(), model_path)
        print(f"Model state dict saved to: {model_path}")
    
    # Save other results
    with open(results_path, 'wb') as f:
        pickle.dump(save_dict, f)
    print(f"Results saved to: {results_path}")


# ==================== Main Training Code ====================

def get_base_path() -> str:
    """Get the base stimulus path (Ubuntu/Linux only)."""
    base_path = "/G/MIMOlab/Codes/aim3_RNN/stimuli"
    print(f"Using base path: {base_path}")
    return base_path


def prepare_data_paths(base_path: str):
    """Construct and validate stimulus / label file paths."""
    stim_train_path = os.path.join(base_path, "stimulus_reg-train.npy")
    label_train_path = os.path.join(base_path, "stimulus_reg-train.tsv")
    stim_val_path = os.path.join(base_path, "stimulus_reg-validation.npy")
    label_val_path = os.path.join(base_path, "stimulus_reg-validation.tsv")

    print("Checking data paths...")
    print(f"Base path: {base_path}")
    for path_name, path in [
        ("train stim", stim_train_path),
        ("train label", label_train_path),
        ("val stim", stim_val_path),
        ("val label", label_val_path),
    ]:
        if not os.path.exists(path):
            print(f"ERROR: {path_name} path does not exist: {path}")
            raise FileNotFoundError(f"Data file not found: {path}")
        else:
            print(f"  ✓ {path_name}: {path}")

    return stim_train_path, label_train_path, stim_val_path, label_val_path


def load_raw_data(stim_train_path: str, label_train_path: str,
                  stim_val_path: str, label_val_path: str):
    """Load raw numpy and label data from disk."""
    print("\nLoading data...")
    stims_train = np.load(stim_train_path, allow_pickle=True, mmap_mode="r") # "r" for read-only memory mapping
    print(f"  ✓ Loaded training stimuli: {stims_train.shape}")
    lbls_train = pd.read_csv(label_train_path, sep="\t", index_col=0)
    print(f"  ✓ Loaded training labels: {lbls_train.shape}")

    stims_val = np.load(stim_val_path, allow_pickle=True, mmap_mode="r") # "r" for read-only memory mapping
    print(f"  ✓ Loaded validation stimuli: {stims_val.shape}")
    lbls_val = pd.read_csv(label_val_path, sep="\t", index_col=0)
    print(f"  ✓ Loaded validation labels: {lbls_val.shape}")

    return stims_train, lbls_train, stims_val, lbls_val


def create_datasets(stims_train, lbls_train, stims_val, lbls_val,
                    use_sector_mode: bool, predict_all_chars: bool,
                    max_chars: int = 10):
    """Create training / validation datasets and return dataset objects and num_pos."""
    print("Creating datasets...")

    if predict_all_chars:
        train_ds = MC_RNN_Dataset(
            stims_train, lbls_train, use_sector=False,
            predict_all_chars=True, max_chars=max_chars,
        )
        val_ds = MC_RNN_Dataset(
            stims_val, lbls_val, use_sector=False,
            predict_all_chars=True, max_chars=max_chars,
        )
        num_pos = 0
        print(f"Using all-chars mode: predict all characters (fg+bg) per frame, max_chars={max_chars}")
    elif use_sector_mode:
        num_pos = 9
        train_ds = MC_RNN_Dataset(
            stims_train, lbls_train, use_sector=True, num_sectors=num_pos,
            predict_all_chars=False,
        )
        val_ds = MC_RNN_Dataset(
            stims_val, lbls_val, use_sector=True, num_sectors=num_pos,
            predict_all_chars=False,
        )
        print("Using sector mode (3x3 grid, 9 sectors)")
    else:
        num_pos = 2
        train_ds = MC_RNN_Dataset(
            stims_train, lbls_train, use_sector=False, predict_all_chars=False,
        )
        val_ds = MC_RNN_Dataset(
            stims_val, lbls_val, use_sector=False, predict_all_chars=False,
        )
        print("Using coordinate mode (directly predict x, y coordinates)")

    return train_ds, val_ds, num_pos


def get_model_classes():
    """Return mapping from model type name to model class."""
    return {
        "rnn": RNNConv,
        "lstm": LSTMConv,
        "gru": GRUConv,
        "gawf": GaWFRNNConv,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser for command line options."""
    parser = argparse.ArgumentParser(description="Train RNN models for sector classification")
    parser.add_argument(
        "--model_types",
        type=str,
        nargs="+",
        default=["rnn"],
        choices=["rnn", "lstm", "gru", "gawf"],
        help='Model types to train (default: ["lstm"])',
    )
    parser.add_argument(
        "--hidden_sizes",
        type=int,
        nargs="+",
        default=[256],
        help="Hidden sizes to test (default: [256])",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=200,
        help="Number of training epochs (default: 200)",
    )
    parser.add_argument(
        "--lrs",
        type=float,
        nargs="+",
        default=[0.001],
        help="Learning rates to search over (default: [0.001])",
    )
    parser.add_argument(
        "--weight_decays",
        type=float,
        nargs="+",
        default=[0], #[1e-4],
        help="Weight decay values to search over (default: [1e-4])",
    )
    parser.add_argument(
        "--dropout_rates",
        type=float,
        nargs="+",
        default=[0], #[0.3],
        help="Dropout rates to search over for model creation (default: [0.3])",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--use_acceleration",
        action="store_true",
        default=False,
        help="Enable acceleration features for training (default: False)",
    )
    parser.add_argument(
        "--use_sector_mode",
        action="store_true",
        default=False,
        help="Use sector mode (3x3 grid, 9 sectors) instead of coordinate mode (default: False)",
    )
    parser.add_argument(
        "--predict_all_chars",
        action="store_true",
        default=False,
        help="Predict all characters (fg+bg) per frame instead of only foreground character (default: False)",
    )
    parser.add_argument(
        "--result_suffix",
        type=str,
        default="default_sector", # ""
        help="Suffix to append to result file names for distinguishing different training runs (default: empty string)",
    )
    return parser


if __name__ == "__main__":
    # Parse command line arguments
    parser = build_arg_parser()
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda:0"   # 可见设备中的 0（由 CUDA_VISIBLE_DEVICES 决定映射到物理哪张卡）
    else:
        device = "cpu"

    disable_tqdm_env = os.environ.get('DISABLE_TQDM', '').lower() in ['1', 'true', 'yes']
    enable_tqdm_env  = os.environ.get('ENABLE_TQDM', '').lower() in ['1', 'true', 'yes']
    term_ok = os.environ.get('TERM', '').lower() not in ['', 'dumb']
    use_tqdm = enable_tqdm_env or (
        not disable_tqdm_env and sys.stdout.isatty() and term_ok
    )
    
    # Data path configuration
    base_path = get_base_path()
    stim_train_path, label_train_path, stim_val_path, label_val_path = prepare_data_paths(base_path)
    stims_train, lbls_train, stims_val, lbls_val = load_raw_data(
        stim_train_path, label_train_path, stim_val_path, label_val_path,
    )

    # Dataset configuration
    use_sector_mode = args.use_sector_mode
    predict_all_chars = args.predict_all_chars
    use_acceleration = args.use_acceleration
    max_chars = 10

    train_ds, val_ds, num_pos = create_datasets(
        stims_train,
        lbls_train,
        stims_val,
        lbls_val,
        use_sector_mode=use_sector_mode,
        predict_all_chars=predict_all_chars,
        max_chars=max_chars,
    )

    # Model class mapping table
    model_classes = get_model_classes()

    # Training configuration (from command line arguments)
    model_types = args.model_types
    hidden_sizes = args.hidden_sizes
    lrs = args.lrs
    weight_decays = args.weight_decays
    dropout_rates = args.dropout_rates

    # Create results directory
    results_dir = f"results/models/{args.result_suffix}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
        print(f"Created results directory: {results_dir}")

    # Build hyperparameter combinations: (model_type, hidden_size, lr, weight_decay, dropout_rate)
    experiment_configs = list(
        product(model_types, hidden_sizes, lrs, weight_decays, dropout_rates)
    )

    # Modification settings: always enabled for this script
    use_modification = True

    # Training loop over all hyperparameter combinations
    total_experiments = len(experiment_configs)
    experiment_num = 0

    print(f"\n{'=' * 60}")
    print(f"Starting training loop: {total_experiments} experiments")
    print(f"Models: {model_types}")
    print(f"Hidden sizes: {hidden_sizes}")
    print(f"Learning rates: {lrs}")
    print(f"Weight decays: {weight_decays}")
    print(f"Dropout rates: {dropout_rates}")
    print(f"{'=' * 60}\n")

    for model_type, hidden_size, lr, weight_decay, dropout_rate in experiment_configs:
        experiment_num += 1
        print(f"\n{'=' * 60}")
        print(
            f"Experiment {experiment_num}/{total_experiments}: "
            f"{model_type.upper()} | hidden_size={hidden_size} | "
            f"lr={lr} | weight_decay={weight_decay} | dropout={dropout_rate}"
        )
        print(f"{'=' * 60}\n")

        # Create model
        if model_type not in model_classes:
            print(f"Warning: Unsupported model_type: {model_type}, skipping...")
            continue

        ModelClass = model_classes[model_type]

        if predict_all_chars:
            if model_type == "gawf":
                print("Warning: GaWFRNNConv does not support predict_all_chars mode, skipping...")
                continue
            mdl = ModelClass(
                num_classes=10,
                num_pos=0,
                kernel_size=5,
                dropout_rate=dropout_rate,
                hidden_size=hidden_size,
                max_chars=max_chars,
                predict_all_chars=True,
            )
            print(
                f"Created {model_type.upper()} model "
                f"(predict_all_chars=True, max_chars={max_chars}, "
                f"dropout_rate={dropout_rate}, hidden_size={hidden_size})"
            )
        else:
            mdl = ModelClass(
                num_classes=10,
                num_pos=num_pos,
                kernel_size=5,
                dropout_rate=dropout_rate,
                hidden_size=hidden_size,
            )
            print(
                f"Created {model_type.upper()} model "
                f"(num_pos={num_pos}, dropout_rate={dropout_rate}, "
                f"hidden_size={hidden_size})"
            )

        # Train model
        print("Starting training...")
        if use_acceleration:
            print("Acceleration training enabled")
        else:
            print("Using standard training method")

        results = network_train(
            mdl,
            train_ds,
            val_ds,
            num_epochs=args.num_epochs,
            lr=lr,
            use_acceleration=use_acceleration,
            use_modification=use_modification,
            weight_decay=weight_decay,
            dropout_rate=dropout_rate,
            rnn_diag_lambda=1e-4,
            use_tqdm=use_tqdm,
        )

        # Save training results
        print(f"\nSaving results for {model_type.upper()} (hidden_size={hidden_size})...")
        mode_suffix = "allchars" if predict_all_chars else ("sector" if use_sector_mode else "coord")
        acc_suffix = "_acc" if use_acceleration else ""
        hp_suffix = f"_lr{lr}_wd{weight_decay}_do{dropout_rate}"
        results_path = os.path.join(
            results_dir,
            f"{model_type}_{mode_suffix}{acc_suffix}_h{hidden_size}{hp_suffix}",
        )

        save_results(results, results_path)
        print(f"Experiment {experiment_num}/{total_experiments} completed!\n")

    print(f"\n{'=' * 60}")
    print(f"All {total_experiments} experiments completed!")
    print(f"Results saved to: {results_dir}/")
    print(f"{'=' * 60}\n")

