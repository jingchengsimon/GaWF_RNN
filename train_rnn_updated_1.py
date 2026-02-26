"""
Standalone RNN Sector training script
Used to train RNN models and save results
"""
import os
import gc
import sys
import signal
import argparse
from itertools import product
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

# Import helper and acceleration utilities
from utils.train_helpers import (
    save_results,
    get_base_path,
    prepare_data_paths,
    load_raw_data,
    create_datasets,
    set_seed,
    get_gpu_memory_usage,
    get_model_classes,
    setup_logger,
    log_experiment_config,
    log_experiment_start,
    log_dataset_and_batch_info,
)
from utils.train_acceleration_1 import (
    AccelerationConfig,
    setup_acceleration,
    build_loaders,
    run_forward_with_feedback,
    TrainStepper,
)
from utils.train_predict_all_chars import (
    build_loss_fn_all_chars,
    AllCharsMetricsMode,
)
from utils.train_sector import (
    get_loss_weights,
    get_criterion_pos,
    build_loss_fn_single,
    SingleCharMetricsMode,
)


def _create_metrics_mode(predict_all_chars, use_sector, max_chars=None, device=None):
    """Single factory: returns the metrics mode for train/eval so no if-else on predict_all_chars in loops."""
    if predict_all_chars:
        return AllCharsMetricsMode(max_chars, device)
    return SingleCharMetricsMode(use_sector)

torch.set_num_threads(4)

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def _cnn_feature_map_config(feature_size, mp1_out_hw=48):
    """
    Return (out_channels, out_h, out_w, mp2_kernel, mp2_stride) for the CNN encoder output.
    - large: (64, 12, 12), MP2(4, 4) so 48/4=12
    - small: (32, 6, 6), MP2(8, 8) so 48/8=6
    """
    if feature_size == "large":
        out_ch, out_h, out_w = 64, 12, 12
    elif feature_size == "small":
        out_ch, out_h, out_w = 16, 3, 3 #32, 6, 6
    else:
        raise ValueError(f"cnn_feature_size must be 'large' or 'small', got {feature_size!r}")

    # derive MP2 stride from target spatial size
    if mp1_out_hw % out_h != 0 or mp1_out_hw % out_w != 0:
        raise ValueError(
            f"MP1 output {mp1_out_hw}x{mp1_out_hw} cannot be pooled to "
            f"{out_h}x{out_w} with integer stride."
        )

    mp2_k = mp1_out_hw // out_h
    mp2_s = mp1_out_hw // out_w

    return out_ch, out_h, out_w, mp2_k, mp2_s


# ==================== Dataset Class ====================
class MC_RNN_Dataset(Dataset):
    def __init__(self, data, labels, frame_num=32, chan_num=2, use_sector=False, num_sectors=9, 
                 max_chars=15, predict_all_chars=False):
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
        for i in range(-(self.chan_num - 1), 1):
            if i == -(self.chan_num - 1):
                stacked_frames = np.expand_dims(self.data[(start_idx + i):(end_idx + i)], axis=1)
            else:
                stacked_frames = np.concatenate(
                    (stacked_frames, np.expand_dims(self.data[(start_idx + i):(end_idx + i)], axis=1)),
                    axis=1
                )
        stacked_frames = stacked_frames.astype(np.float32)

        if self.predict_all_chars:
            all_chars_per_frame = []
            for frame_idx in range(start_idx, end_idx):
                fg_char_id = int(self.fg_char_ids[frame_idx])

                bg_chars_str = str(self.bg_char_ids_str[frame_idx])
                if bg_chars_str and bg_chars_str != 'nan':
                    bg_char_ids = [int(x) for x in bg_chars_str.split(',') if x.strip()]
                else:
                    bg_char_ids = []

                all_chars = [fg_char_id] + bg_char_ids
                padded_chars = all_chars[:self.max_chars] + [-1] * max(0, self.max_chars - len(all_chars))
                all_chars_per_frame.append(padded_chars)

            labels = np.array(all_chars_per_frame, dtype=np.int64)
        else:
            labels = self.labels[start_idx:end_idx].copy()

            if self.use_sector:
                height = self.data.shape[-2]
                width = self.data.shape[-1]

                grid_size = int(np.sqrt(self.num_sectors))
                if grid_size * grid_size != self.num_sectors:
                    raise ValueError(
                        f"num_sectors={self.num_sectors} is not a perfect square, cannot form grid_size x grid_size grid"
                    )

                x = labels[:, 1].astype(np.float32)
                y = labels[:, 2].astype(np.float32)

                col = (x / max(width - 1, 1) * grid_size).astype(np.int64)
                row = (y / max(height - 1, 1) * grid_size).astype(np.int64)

                col = np.clip(col, 0, grid_size - 1)
                row = np.clip(row, 0, grid_size - 1)

                sector = row * grid_size + col
                labels = np.stack([labels[:, 0].astype(np.int64), sector], axis=1)

        # 关键：返回 idx，供 feedback table 使用
        return stacked_frames, labels, idx

# ==================== Model Classes ====================
# Note: self.training is provided by nn.Module. It is True when model.train() is called
# and False when model.eval(). Used by F.dropout* so dropout is applied only during training.
class BaseConvSequenceModel(nn.Module):
    """
    Shared encoder (CNN), classifier (fcchar/fcpos), and forward for Conv sequence models
    that use the same pipeline: (B,T,C,H,W) -> encoder -> (B,T,hidden) -> middle -> classifier.
    Subclasses must implement middle(x) and may add extra layers in __init__.
    cnn_feature_size: 'large' -> encoder output (64, 12, 12); 'small' -> (32, 6, 6).
    """
    def __init__(self, num_classes, num_pos, kernel_size=3, device='cuda', dropout_rate=0.3,
                 hidden_size=256, max_chars=15, predict_all_chars=False, cnn_feature_size='large'):
        super(BaseConvSequenceModel, self).__init__()
        self.device = device
        self.dropout_rate = dropout_rate
        self.max_chars = max_chars
        self.predict_all_chars = predict_all_chars
        out_ch, out_h, out_w, mp2_k, mp2_s = _cnn_feature_map_config(cnn_feature_size)
        self.encoder_flatten_size = out_ch * out_h * out_w
        # Shared encoder
        self.conv1 = nn.Conv2d(2, 32, kernel_size=kernel_size, padding='same')
        self.MP1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.LNorm1 = nn.LayerNorm([32, 48, 48])
        self.conv2 = nn.Conv2d(32, out_ch, kernel_size=3, padding=1)
        self.MP2 = nn.MaxPool2d(kernel_size=mp2_k, stride=mp2_s)
        self.LNorm2 = nn.LayerNorm([out_ch, out_h, out_w])
        # Shared classifier
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

    def classifier(self, x):
        if self.predict_all_chars:
            chars_out = self.fcchars(x)
            batch_size, frame_num = chars_out.shape[:2]
            num_classes = chars_out.shape[-1] // self.max_chars
            chars_out = chars_out.view(batch_size, frame_num, self.max_chars, num_classes)
            return chars_out, None
        else:
            return self.fcchar(x), self.fcpos(x)

    def middle(self, x):
        raise NotImplementedError("Subclass must implement middle(x)")

    def forward(self, x):
        x = x.to(self.device)
        batch_size, frame_num, channels, height, width = x.size()
        x = x.view(batch_size * frame_num, channels, height, width)
        x = self.encoder(x)
        x = x.view(batch_size, frame_num, -1)
        x = self.middle(x)
        char_out, pos_out = self.classifier(x)
        return char_out, pos_out


class BaseRNNConv(BaseConvSequenceModel):
    """Base class for CNN-RNN models supporting different RNN types. Middle = RNN + LayerNorm + ReLU + Dropout."""
    def __init__(self, num_classes, num_pos, rnn_class=nn.RNN, kernel_size=3, device='cuda',
                 dropout_rate=0.3, hidden_size=256, max_chars=15, predict_all_chars=False, cnn_feature_size='large'):
        super(BaseRNNConv, self).__init__(
            num_classes, num_pos, kernel_size=kernel_size, device=device,
            dropout_rate=dropout_rate, hidden_size=hidden_size,
            max_chars=max_chars, predict_all_chars=predict_all_chars,
            cnn_feature_size=cnn_feature_size,
        )
        self.rnn = rnn_class(input_size=self.encoder_flatten_size, hidden_size=hidden_size,
                             num_layers=1, batch_first=True)
        self.LNormRNN = nn.LayerNorm(hidden_size)

    def middle(self, x):
        x = self.rnn(x)[0]
        x = self.LNormRNN(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        return x


class RNNConv(BaseRNNConv):
    def __init__(self, num_classes, num_pos, kernel_size=3, device='cuda', dropout_rate=0.3, hidden_size=256,
                 max_chars=15, predict_all_chars=False, cnn_feature_size='large'):
        super(RNNConv, self).__init__(num_classes, num_pos, rnn_class=nn.RNN, kernel_size=kernel_size,
                                      device=device, dropout_rate=dropout_rate, hidden_size=hidden_size,
                                      max_chars=max_chars, predict_all_chars=predict_all_chars,
                                      cnn_feature_size=cnn_feature_size)


class GRUConv(BaseRNNConv):
    def __init__(self, num_classes, num_pos, kernel_size=3, device='cuda', dropout_rate=0.3, hidden_size=256,
                 max_chars=15, predict_all_chars=False, cnn_feature_size='large'):
        super(GRUConv, self).__init__(num_classes, num_pos, rnn_class=nn.GRU, kernel_size=kernel_size,
                                      device=device, dropout_rate=dropout_rate, hidden_size=hidden_size,
                                      max_chars=max_chars, predict_all_chars=predict_all_chars,
                                      cnn_feature_size=cnn_feature_size)


class LSTMConv(BaseRNNConv):
    def __init__(self, num_classes, num_pos, kernel_size=3, device='cuda', dropout_rate=0.3, hidden_size=256,
                 max_chars=15, predict_all_chars=False, cnn_feature_size='large'):
        super(LSTMConv, self).__init__(num_classes, num_pos, rnn_class=nn.LSTM, kernel_size=kernel_size,
                                       device=device, dropout_rate=dropout_rate, hidden_size=hidden_size,
                                       max_chars=max_chars, predict_all_chars=predict_all_chars,
                                       cnn_feature_size=cnn_feature_size)


class GaWFRNNConv(BaseConvSequenceModel):
    """
    GaWF (Gated with Feedback) RNN Model.
    Encoder and classifier from BaseConvSequenceModel. Forward overridden for feedback.
    Main improvements:
    1. Use classifier output as feedback to RNN input
    2. Feedback is transformed by U @ diag(concat) @ V, then Hadamard product with RNN weights
    """
    def __init__(self, num_classes, num_pos, kernel_size=3, device='cuda', dropout_rate=0.3, hidden_size=256,
                 max_chars=15, predict_all_chars=False, cnn_feature_size='large'):
        super(GaWFRNNConv, self).__init__(
            num_classes, num_pos, kernel_size=kernel_size, device=device,
            dropout_rate=dropout_rate, hidden_size=hidden_size,
            max_chars=15, predict_all_chars=False,
            cnn_feature_size=cnn_feature_size,
        )
        self.num_classes = num_classes
        self.num_pos = num_pos
        self.hidden_size = hidden_size
        input_size = self.encoder_flatten_size
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                          num_layers=1, batch_first=True)
        # Feedback transformation matrices
        # Dimension after concatenating classifier outputs
        feedback_dim = num_classes + num_pos
        # RNN weight matrix shapes
        # weight_ih: (hidden_size, input_size) = (256, 1152)
        # weight_hh: (hidden_size, hidden_size) = (256, 256)
        # Concatenated shape: (256, 1152 + 256) = (256, 9472)
        combined_weight_size = input_size + hidden_size
        # U: (hidden_size, feedback_dim) = (256, 19)
        # V: (feedback_dim, combined_weight_size) = (19, 9472)
        # diag(concat): (feedback_dim, feedback_dim) = (19, 19)
        # U @ diag @ V: (256, 19) @ (19, 19) @ (19, 9472) = (256, 9472)
        self.U = nn.Parameter(torch.randn(hidden_size, feedback_dim) * 0.01)
        self.V = nn.Parameter(torch.randn(feedback_dim, combined_weight_size) * 0.01)
        self.LNormRNN = nn.LayerNorm(hidden_size)
        self.register_buffer('prev_feedback', None)

    def set_feedback_frozen(self, freeze: bool):
        """Freeze or unfreeze feedback-related parameters (U, V). When frozen, feedback path does not receive gradients."""
        for p in (self.U, self.V):
            p.requires_grad = not freeze

    # encoder and classifier inherited from BaseConvSequenceModel (same implementation)

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

            if feedback.dim() == 2:
                # feedback from table is (B, fb_dim); broadcast to all timesteps
                feedback = feedback.unsqueeze(1).expand(-1, seq_len, -1)

            # Get RNN weights
            weight_ih = self.rnn.weight_ih_l0  # (hidden_size, input_size)
            weight_hh = self.rnn.weight_hh_l0  # (hidden_size, hidden_size)
            bias_ih = self.rnn.bias_ih_l0 if self.rnn.bias_ih_l0 is not None else None
            bias_hh = self.rnn.bias_hh_l0 if self.rnn.bias_hh_l0 is not None else None
            
            # Process each time step
            # outputs = []
            outputs = torch.empty(batch_size, seq_len, hidden_size, device=x.device, dtype=x.dtype)
            h = torch.zeros(batch_size, hidden_size, device=x.device)  # (B, hidden_size)
            
            # # Concatenate RNN input weight matrix and recurrent weight matrix (compute once)
            # combined_weight = torch.cat([weight_ih, weight_hh], dim=1)  # (hidden_size, input_size + hidden_size)
            
            for t in range(seq_len):
                # Current time step input and feedback
                x_t = x[:, t, :]  # (B, input_size)
                fb_t_clamped = feedback[:, t, :].clamp(-10, 10)  # (B, feedback_dim)
                
                # Feedback transformation: U @ diag(fb_t_clamped) @ V, then sigmoid
                # fb_t_clamped: (B, feedback_dim)
                # For each sample b, compute U @ diag(fb_t_clamped[b]) @ V
                # Equivalent to: U @ (fb_t_clamped[b] * I) @ V, where I is identity matrix
                # Can be written as: U @ (fb_t_clamped[b].unsqueeze(1) * V)
                
                # gated_weights, h_t_list = [], []
                # for b in range(batch_size):
                #     transformed = self.U @ (fb_t_clamped[b].unsqueeze(1) * self.V)

                #     tau = 2.0
                #     transformed = torch.sigmoid(transformed / tau)  # (hidden_size, combined_weight_size)
                    
                #     # Hadamard product: transformed * combined_weight
                #     gated_weight = transformed * combined_weight  # (hidden_size, combined_weight_size)
                #     gated_weights.append(gated_weight)
                
                # # Stack gated weights: (B, hidden_size, combined_weight_size)
                # gated_weights = torch.stack(gated_weights, dim=0)  # (B, hidden_size, combined_weight_size)
                
                # # Separate back into weight_ih and weight_hh parts
                # gated_weight_ih = gated_weights[:, :, :input_size]  # (B, hidden_size, input_size)
                # gated_weight_hh = gated_weights[:, :, input_size:]  # (B, hidden_size, hidden_size)
                
                # # Compute RNN output using gated weights
                # # Compute for each sample separately  
                # for b in range(batch_size):
                #     # Input to hidden: (1, hidden_size)
                #     ih = F.linear(x_t[b:b+1], gated_weight_ih[b], bias_ih)  # (1, hidden_size)
                #     # Hidden to hidden: (1, hidden_size)
                #     hh = F.linear(h[b:b+1], gated_weight_hh[b], bias_hh)  # (1, hidden_size)
                #     # Combine and apply activation
                #     h_t = torch.tanh(ih + hh)  # (1, hidden_size)
                #     h_t_list.append(h_t)
                
                # # Stack: (B, hidden_size)
                # h_t = torch.cat(h_t_list, dim=0)  # (B, hidden_size)

                # ===== [GAWF_BMM] batched gate + batched RNN update (no b-loop) =====

                # Ensure float32 for stability if you want (optional but recommended with AMP):
                # fb_t_clamped = fb_t_clamped.to(torch.float32)

                # Split V into input-part and hidden-part to avoid building full (B, hidden, combined) explicitly
                V_ih = self.V[:, :input_size]        # (feedback_dim, input_size)
                V_hh = self.V[:, input_size:]        # (feedback_dim, hidden_size)

                # Prepare scaled V per sample: (B, feedback_dim, K) where K is input_size or hidden_size
                # diag(fb) @ V == fb[:,None] * V  (row-wise scale of V)
                tmp_ih = fb_t_clamped.unsqueeze(2) * V_ih.unsqueeze(0)   # (B, feedback_dim, input_size)
                tmp_hh = fb_t_clamped.unsqueeze(2) * V_hh.unsqueeze(0)   # (B, feedback_dim, hidden_size)

                # U @ tmp => (B, hidden_size, K)
                # U: (hidden_size, feedback_dim)
                trans_ih = torch.matmul(self.U, tmp_ih)  # (B, hidden_size, input_size)
                trans_hh = torch.matmul(self.U, tmp_hh)  # (B, hidden_size, hidden_size)

                tau = 2.0
                gate_ih = torch.sigmoid(trans_ih / tau)  # (B, hidden_size, input_size)
                gate_hh = torch.sigmoid(trans_hh / tau)  # (B, hidden_size, hidden_size)

                # Apply gate to base weights (broadcast base weights to batch)
                # weight_ih: (hidden_size, input_size) -> (B, hidden_size, input_size)
                # weight_hh: (hidden_size, hidden_size) -> (B, hidden_size, hidden_size)
                gated_weight_ih = gate_ih * weight_ih.unsqueeze(0)  # (B, hidden_size, input_size)
                gated_weight_hh = gate_hh * weight_hh.unsqueeze(0)  # (B, hidden_size, hidden_size)

                # Batched linear:
                # ih[b] = x_t[b] @ gated_weight_ih[b]^T + bias_ih
                # Use bmm: (B, 1, input) x (B, input, hidden) -> (B, 1, hidden)
                ih = torch.bmm(x_t.unsqueeze(1), gated_weight_ih.transpose(1, 2)).squeeze(1)  # (B, hidden_size)
                hh = torch.bmm(h.unsqueeze(1), gated_weight_hh.transpose(1, 2)).squeeze(1)    # (B, hidden_size)

                if bias_ih is not None:
                    ih = ih + bias_ih.unsqueeze(0)
                if bias_hh is not None:
                    hh = hh + bias_hh.unsqueeze(0)

                h_t = torch.tanh(ih + hh)  # (B, hidden_size)
                # ===== [GAWF_BMM] end =====
                
                # LayerNorm and ReLU
                gated_output = self.LNormRNN(h_t)  # (B, hidden_size)
                gated_output = F.relu(gated_output)  # (B, hidden_size)
                
                # outputs.append(gated_output)
                outputs[:, t, :] = gated_output
                h = gated_output  # Update hidden state
            
            # Stack outputs: (B, T, hidden_size)
            # x = torch.stack(outputs, dim=1)  # (B, T, hidden_size)
            x = outputs

        else:
            # When no feedback, use standard RNN
            x, _ = self.rnn(x)
            x = self.LNormRNN(x)
            x = F.relu(x)
        
        # Dropout
        x = F.dropout(x, p=0.5, training=self.training)
        return x

    # classifier inherited from BaseConvSequenceModel (same implementation)

    def forward(self, x, use_feedback=True, reset_feedback=False):
        """
        Args:
            x: Input sequence (B, T, C, H, W)
            use_feedback: Single switch for the feedback path. When False, RNN runs as standard RNN (no feedback).
                        In training this is set from the same nofb/fb_start_epoch control (see network_train).
            reset_feedback: When True, do not use prev_feedback for this forward even if use_feedback=True.
                            (Useful for size-change / batch-size probing / eval isolation)
        """
        x = x.to(self.device)

        batch_size, frame_num, channels, height, width = x.size()

        # resize to process each frame individually
        x = x.view(batch_size * frame_num, channels, height, width)

        # apply CNN encoder
        x = self.encoder(x)

        # reshape back to batches of stacks of frames and flatten each image
        x = x.view(batch_size, frame_num, -1)

        # -------- feedback selection --------
        if use_feedback:
            if reset_feedback or self.prev_feedback is None:
                feedback = None
            else:
                # After FB_LAST_ONLY change, prev_feedback is (B, fb_dim) (NOT (B,T,fb_dim))
                feedback = self.prev_feedback.to(dtype=torch.float32)
        else:
            feedback = None
            self.prev_feedback = None  # keep state consistent when feedback is off

        # apply RNN / gated update
        x = self.middle(x, feedback=feedback)

        # apply classification heads
        char_out, pos_out = self.classifier(x)

        # -------- store feedback: ONLY last timestep to avoid (B,T,fb_dim) blow-up --------
        if use_feedback:
            with torch.no_grad():
                fb_full = torch.cat([char_out, pos_out], dim=-1)  # (B, T, fb_dim)
                fb_last = fb_full[:, -1, :]  # (B, fb_dim)
                self.prev_feedback = fb_last.detach()

        return char_out, pos_out


# ==================== Base for merge-frame ANN models (FFN / dANN) ====================
# Shared: (B, T, C, H, W) -> merge (B, T*C, H, W) -> encoder -> flatten -> middle -> classifier -> expand to (B, T, ...).
# Subclasses implement middle(x) only.
class BaseMergeConvModel(nn.Module):
    """
    Base for models that adapt ANN to RNN data format by merging frame_num and channels.
    Merges (B, T, C, H, W) -> (B, T*C, H, W); output expanded to (B, T, num_classes) etc.
    Subclasses must implement middle(x) where x is (B, encoder_flatten_size).
    cnn_feature_size: 'large' -> (64, 12, 12); 'small' -> (32, 6, 6).
    """
    def __init__(self, num_classes, num_pos, kernel_size=3, device='cuda', dropout_rate=0.3,
                 hidden_size=256, max_chars=15, predict_all_chars=False, cnn_feature_size='large'):
        super(BaseMergeConvModel, self).__init__()
        self.device = device
        self.dropout_rate = dropout_rate
        self.max_chars = max_chars
        self.predict_all_chars = predict_all_chars
        self._input_channels = None
        self.conv1 = None
        self.MP1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.LNorm1 = None
        out_ch, out_h, out_w, mp2_k, mp2_s = _cnn_feature_map_config(cnn_feature_size)
        self.encoder_flatten_size = out_ch * out_h * out_w
        self.conv2 = nn.Conv2d(32, out_ch, kernel_size=3, padding=1)
        self.MP2 = nn.MaxPool2d(kernel_size=mp2_k, stride=mp2_s)
        self.LNorm2 = nn.LayerNorm([out_ch, out_h, out_w])
        if predict_all_chars:
            self.fcchars = nn.Linear(hidden_size, max_chars * num_classes)
            self.fcpos = None
        else:
            self.fcchar = nn.Linear(hidden_size, num_classes)
            self.fcpos = nn.Linear(hidden_size, num_pos)
        self.to(self.device)

    def _ensure_conv1(self, input_channels):
        if self.conv1 is None or self._input_channels != input_channels:
            self._input_channels = input_channels
            self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding='same').to(self.device)
            self.LNorm1 = nn.LayerNorm([32, 48, 48]).to(self.device)

    def encoder(self, x):
        input_channels = x.size(1)
        self._ensure_conv1(input_channels)
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

    def classifier(self, x):
        if self.predict_all_chars:
            chars_out = self.fcchars(x)
            batch_size = chars_out.shape[0]
            num_classes = chars_out.shape[-1] // self.max_chars
            chars_out = chars_out.view(batch_size, self.max_chars, num_classes)
            return chars_out, None
        else:
            return self.fcchar(x), self.fcpos(x)

    def middle(self, x):
        raise NotImplementedError("Subclass must implement middle(x)")

    def forward(self, x):
        x = x.to(self.device)
        batch_size, frame_num, channels, height, width = x.size()
        x = x.view(batch_size, frame_num * channels, height, width)
        x = self.encoder(x)
        x = x.view(batch_size, -1)
        x = self.middle(x)
        char_out, pos_out = self.classifier(x)
        if self.predict_all_chars:
            char_out = char_out.unsqueeze(1).expand(-1, frame_num, -1, -1).contiguous()
        else:
            char_out = char_out.unsqueeze(1).expand(-1, frame_num, -1).contiguous()
            pos_out = pos_out.unsqueeze(1).expand(-1, frame_num, -1).contiguous()
        return char_out, pos_out


# ==================== Dendritic ANN (dANN) with Global RFs ====================
# Aligned with opt.py get_model(): model type 1 (dend_ann_global_rfs).
# - Dend: linear only (no nonlinearity on dendrite outputs).
# - Soma: fixed aggregation (non-learnable), i.e. sum over dendrites per soma; then LeakyReLU on soma (match opt).
class DendriticLayer(nn.Module):
    """
    Single dendritic layer aligned with opt.py.
    - Dendrite segment: one linear (input_dim -> num_soma * num_dends), no activation on dend outputs.
    - Soma segment: fixed aggregation (non-learnable) via registered buffer of ones; then LeakyReLU(0.1) on soma output.
    """
    def __init__(self, input_dim, num_dends, num_soma, dropout=0.0):
        super(DendriticLayer, self).__init__()
        self.num_dends = num_dends
        self.num_soma = num_soma
        # Dendrite: single linear, no nonlinearity on dend outputs
        self.fc = nn.Linear(input_dim, num_soma * num_dends)
        self.dropout = nn.Dropout(dropout)
        # Soma: non-learnable aggregation (fixed weights = 1). Buffer is not in parameters().
        self.register_buffer("soma_agg", torch.ones(num_soma, num_dends))

    def forward(self, x):
        # x: (B, input_dim)
        x = self.dropout(x)
        # Dend: linear only (no ReLU on dendrite outputs)
        x = self.fc(x)  # (B, num_soma * num_dends)
        x = x.view(x.size(0), self.num_soma, self.num_dends)
        # Soma: fixed weighted sum (soma_agg is non-learnable buffer), then LeakyReLU (opt relu_slope=0.1)
        x = F.leaky_relu((x * self.soma_agg).sum(dim=2), negative_slope=0.1)  # (B, num_soma)
        return x


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    More stable than LayerNorm, especially for large hidden sizes.
    Formula: x_norm = x / sqrt(mean(x^2) + eps) * scale
    """
    def __init__(self, dim, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (B, ..., dim)
        norm = x.norm(dim=-1, keepdim=True) / (x.shape[-1] ** 0.5)
        return self.scale * x / (norm + self.eps)


class DendriticANN(nn.Module):
    """
    dANN (model type 1, opt-aligned) for flat input: (B, input_dim) -> (B, hidden_size).
    Stack of DendriticLayers (dend linear, soma fixed sum + LeakyReLU) plus out_proj / Norm / Dropout.
    No ReLU after normalization to avoid representation collapse (negative pre-ReLU drift -> zero output).
    For h=512: uses RMSNorm (more stable than LayerNorm) and conservative out_proj initialization.
    """
    def __init__(self, input_dim, hidden_size, num_layers=2, num_dends=32, num_soma=256, dropout=0.5):
        super(DendriticANN, self).__init__()
        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            layers.append(DendriticLayer(in_dim, num_dends, num_soma, dropout=dropout))
            in_dim = num_soma
        self.layers = nn.ModuleList(layers)
        self.out_proj = nn.Linear(num_soma, hidden_size)
        # Use RMSNorm for h>=512 (more numerically stable), LayerNorm otherwise
        if hidden_size >= 512:
            self.norm = RMSNorm(hidden_size, eps=1e-5)
        else:
            self.norm = nn.LayerNorm(hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        
        # Initialize out_proj with smaller scale for stability (especially h=512)
        # Use Xavier uniform with smaller gain to prevent large activations
        init_gain = 0.5 if hidden_size >= 512 else 1.0
        nn.init.xavier_uniform_(self.out_proj.weight, gain=init_gain)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x):
        # x: (B, input_dim)
        for layer in self.layers:
            x = layer(x)
            # Check for NaN/Inf after each layer (especially important for h=512)
            if self.hidden_size >= 512:
                if torch.isnan(x).any() or torch.isinf(x).any():
                    print(f"Warning: NaN/Inf detected in DendriticLayer output, replacing with zeros")
                    x = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)
        x = self.out_proj(x)
        x = self.norm(x)
        # No ReLU here: avoids collapse when pre-ReLU activations drift negative (wd=0, etc.)
        # For h=512: clip extreme values before dropout to prevent numerical instability
        if self.hidden_size >= 512:
            x = torch.clamp(x, min=-10.0, max=10.0)
            # Final NaN/Inf check
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"Warning: NaN/Inf detected after normalization, replacing with zeros")
                x = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)
        x = self.dropout(x)
        return x  # (B, hidden_size)


class DendriticANNConv(BaseMergeConvModel):
    """
    Dendritic ANN (dANN) with global RFs, aligned with opt.py get_model() (model type 1).
    Encoder and classifier from BaseMergeConvModel. Middle: dANN (dend linear, soma non-learnable).
    """
    def __init__(self, num_classes, num_pos, kernel_size=3, device='cuda', dropout_rate=0.3, hidden_size=256,
                 max_chars=15, predict_all_chars=False,
                 num_layers=2, num_dends=32, num_soma=256, cnn_feature_size='large'):
        super(DendriticANNConv, self).__init__(
            num_classes, num_pos, kernel_size=kernel_size, device=device,
            dropout_rate=dropout_rate, hidden_size=hidden_size,
            max_chars=max_chars, predict_all_chars=predict_all_chars,
            cnn_feature_size=cnn_feature_size,
        )
        self.dann = DendriticANN(
            input_dim=self.encoder_flatten_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_dends=num_dends,
            num_soma=num_soma,
            dropout=0, # turn off dropout 
        )

    def middle(self, x):
        return self.dann(x)


class FeedForwardConv(BaseMergeConvModel):
    """
    FeedForward model, adapted to RNN data format (using frame_num as input_channels).
    Encoder and classifier from BaseMergeConvModel. Middle: single FC layer.
    FFN-exclusive: hidden_size for FC layer (default 512 when RNN default 256 is passed).
    """
    def __init__(self, num_classes, num_pos, kernel_size=3, device='cuda', dropout_rate=0.3, hidden_size=256,
                 max_chars=15, predict_all_chars=False, cnn_feature_size='large'):
        ffn_hidden_size = 512 if hidden_size == 256 else hidden_size
        super(FeedForwardConv, self).__init__(
            num_classes, num_pos, kernel_size=kernel_size, device=device,
            dropout_rate=dropout_rate, hidden_size=ffn_hidden_size,
            max_chars=max_chars, predict_all_chars=predict_all_chars,
            cnn_feature_size=cnn_feature_size,
        )
        self.fc1 = nn.Linear(self.encoder_flatten_size, ffn_hidden_size)
        self.dropout = nn.Dropout(0.5)

    def middle(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


# ==================== Training Function ====================
def network_train(mdl, train_data, val_data, num_epochs=50, loss_weights=None, lr=0.001,
                  use_acceleration=False, weight_decay=None, dropout_rate=None, rnn_diag_lambda=1e-4,
                  use_mmap=False, use_tqdm=True, nofb=False, fb_start_epoch=999999, seed=42, logger=None,
                  cnn_feature_size="large"):
    """
    Train model, supports sector mode and coordinate mode.
    Acceleration is config-driven; training loop is single-path (no AMP branches).
    """
    use_sector = train_data.use_sector
    predict_all_chars = train_data.predict_all_chars
    max_chars = train_data.max_chars if predict_all_chars else None

    if loss_weights is None:
        loss_weights = get_loss_weights(predict_all_chars, use_sector)

    device = mdl.device
    mdl.to(device)

    accel_config = AccelerationConfig(use_acceleration=use_acceleration)
    accel_config.summary()

    metrics_mode = _create_metrics_mode(predict_all_chars, use_sector, max_chars, device)

    (autocast_fn, scaler, batch_size, num_workers, pin_memory) = setup_acceleration(
        accel_config, mdl, train_data, device,
        is_gawf=isinstance(mdl, GaWFRNNConv),
        use_mmap=use_mmap,
        cnn_feature_size=cnn_feature_size,
    )

    train_dl, val_dl = build_loaders(
        train_data, val_data, batch_size, num_workers, pin_memory, accel_config, seed
    )

    # Add weight decay (L2 regularization) to prevent overfitting
    # For h>=512 models (especially dANN), use larger eps for Adam to prevent numerical instability
    # Default eps=1e-8 can be too small when variance estimates become very small
    
    if hasattr(mdl, 'hidden_size') and mdl.hidden_size >= 512:
        optim = torch.optim.Adam(mdl.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-6)
    else:
        optim = torch.optim.Adam(mdl.parameters(), lr=lr, weight_decay=weight_decay)
    criterion_char = nn.CrossEntropyLoss()
    criterion_pos = get_criterion_pos(use_sector) if not predict_all_chars else None

    if predict_all_chars:
        loss_fn = build_loss_fn_all_chars(mdl, criterion_char, max_chars, device, loss_weights, rnn_diag_lambda)
    else:
        loss_fn = build_loss_fn_single(mdl, criterion_char, criterion_pos, use_sector, loss_weights, rnn_diag_lambda, device)

    def evaluate(mdl, data_loader, use_tqdm=True, use_feedback=None, feedback_table=None, update_feedback_table=False):
        """
        Args:
            mdl: model
            data_loader: DataLoader
            use_feedback: None or bool (GaWFRNNConv)
            feedback_table: torch.Tensor on CPU, shape (len(dataset), fb_dim) or None
            update_feedback_table: whether to write new feedback_table from this pass
        Returns:
            (acc_char, metric_pos); if update_feedback_table=True, (acc_char, metric_pos, new_table).
        """
        device = mdl.device
        ds = data_loader.dataset
        eval_mode = _create_metrics_mode(
            getattr(ds, "predict_all_chars", False),
            getattr(ds, "use_sector", False),
            getattr(ds, "max_chars", 10),
            device,
        )
        acc = eval_mode.init_eval()
        new_table = None

        pbar = tqdm(enumerate(data_loader), total=len(data_loader),
                    desc="Validation", ncols=100, leave=False, disable=not use_tqdm)

        mdl.eval()
        with torch.no_grad():
            for batch_idx, batch in pbar:
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    inputs, labels, sample_idx = batch
                else:
                    inputs, labels = batch
                    sample_idx = None

                inputs = inputs.to(device)
                labels = labels.to(device)

                out_char, out_pos = run_forward_with_feedback(
                    mdl, inputs, device,
                    use_feedback=use_feedback,
                    feedback_table=feedback_table,
                    sample_idx=sample_idx,
                )

                if update_feedback_table and (use_feedback is True) and (sample_idx is not None):
                    fb_full = out_char.detach() if out_pos is None else torch.cat([out_char, out_pos], dim=-1).detach()
                    fb_last = fb_full[:, -1, :] if fb_full.dim() == 3 else fb_full
                    if new_table is None:
                        new_table = torch.zeros(len(ds), fb_last.shape[-1], dtype=torch.float32)
                    new_table[sample_idx] = fb_last.to("cpu", dtype=torch.float32)

                acc = eval_mode.update_eval_batch(acc, out_char, labels, out_pos)

        result = eval_mode.finalize_eval(acc, len(data_loader))

        if update_feedback_table:
            return result[0], result[1], new_table
        return result

    if logger is not None:
        log_dataset_and_batch_info(
            logger, train_data, val_data, batch_size, accel_config,
            train_dl, num_workers, pin_memory, use_sector, predict_all_chars,
        )

    stepper = TrainStepper(mdl, optim, loss_fn, accel_config, device, scaler, autocast_fn, pin_memory)
    train_acc_char, val_acc_char = np.zeros(num_epochs), np.zeros(num_epochs)
    train_metric_pos, val_metric_pos = np.zeros(num_epochs), np.zeros(num_epochs)  # sector: accuracy; coord: MSE
    # character CE loss (for both sector/coord single-char mode; all-chars keeps default zeros)
    train_loss_char, val_loss_char = np.zeros(num_epochs), np.zeros(num_epochs)
    # sector only: position CE loss (like coord saves MSE)
    train_loss_pos = np.zeros(num_epochs) if use_sector else None
    val_loss_pos = np.zeros(num_epochs) if use_sector else None

    _STOP_REQUESTED = False

    def _request_stop(signum, frame):
        global _STOP_REQUESTED
        _STOP_REQUESTED = True
        # 不做任何 DataLoader 操作！不 sys.exit！
        print(f"\n[signal] got {signum}, will stop after current step...", flush=True)

    if num_workers > 0:
        signal.signal(signal.SIGTERM, _request_stop)
        signal.signal(signal.SIGINT, _request_stop)

    # GaWFRNN: single nofb-based control for feedback. This value is passed to forward(use_feedback=...) and
    # governs both whether the feedback path is used and (when nofb) whether U,V are frozen.
    def _use_feedback_this_epoch(epoch):
        if not isinstance(mdl, GaWFRNNConv):
            return None  # not used
        if not nofb:
            return True  # default: always feedback
        return epoch >= fb_start_epoch

    try:
        prev_epoch_feedback_table = None  # CPU tensor: (len(train_data), fb_dim)
        for epoch in range(num_epochs):
            mdl.train()
            use_feedback_this_epoch = _use_feedback_this_epoch(epoch)
            new_epoch_feedback_table = None  # will be lazily initialized (CPU) when first fb is produced
            if isinstance(mdl, GaWFRNNConv):
                if nofb:
                    mdl.set_feedback_frozen(use_feedback_this_epoch is False)  # freeze when epoch < fb_start_epoch
                if use_tqdm and nofb and (epoch == 0 or epoch == fb_start_epoch):
                    print(f"GaWFRNN (nofb): epoch {epoch} use_feedback={use_feedback_this_epoch}", flush=True)
            
            epoch_acc = metrics_mode.init_epoch_train()
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

                current_loss, out_char, out_pos, feedback_update = stepper.step(
                    batch, batch_idx, use_feedback_this_epoch, prev_epoch_feedback_table
                )

                if feedback_update is not None:
                    sample_idx, fb_last = feedback_update
                    if new_epoch_feedback_table is None:
                        new_epoch_feedback_table = torch.zeros(len(train_data), fb_last.shape[-1], dtype=torch.float32)
                    new_epoch_feedback_table[sample_idx] = fb_last

                # Get labels for metrics (same batch format as in stepper)
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    _, labels, _ = batch
                else:
                    _, labels = batch

                labels = labels.to(device)

                epoch_acc = metrics_mode.update_train_batch(
                    epoch_acc, out_char, labels, batch_idx, len(train_dl), out_pos
                )

                num_batches += 1
                
                # Memory optimization: periodically clear GPU cache and unused memory
                if accel_config.enable_memory_opt and batch_idx % 50 == 0 and device == 'cuda':
                    torch.cuda.empty_cache()  # Clear GPU cache
                    torch.cuda.synchronize()  # Ensure all GPU operations complete
            
                if batch_idx % 10 == 0:
                    train_pbar.set_postfix(metrics_mode.postfix_for_pbar(
                        current_loss, out_char, out_pos, labels
                    ))
            
            train_result = metrics_mode.finalize_train_epoch(epoch_acc, num_batches)
            train_acc_char[epoch], train_metric_pos[epoch] = train_result[0], train_result[1]
            # For SingleCharMetricsMode: (acc_char, metric_pos, loss_pos, loss_char)
            # For AllCharsMetricsMode: (acc_char, metric_pos)
            if len(train_result) >= 3 and train_result[2] is not None and train_loss_pos is not None:
                train_loss_pos[epoch] = train_result[2]
            if len(train_result) >= 4 and train_result[3] is not None:
                train_loss_char[epoch] = train_result[3]

            gpu_info = ""
            if accel_config.use_acceleration and device == 'cuda' and torch.cuda.is_available():
                gpu_mem = get_gpu_memory_usage()
                gpu_info = f" | GPU memory: {gpu_mem:.1f}%"

            train_str = metrics_mode.format_train_str(
                epoch, num_epochs, train_acc_char[epoch], train_metric_pos[epoch], gpu_info
            )

            if use_feedback_this_epoch is True and new_epoch_feedback_table is not None:
                prev_epoch_feedback_table = new_epoch_feedback_table

            with torch.no_grad():
                val_res = evaluate(
                    mdl, val_dl, use_tqdm, use_feedback=use_feedback_this_epoch
                )
                val_acc_char[epoch], val_metric_pos[epoch] = val_res[0], val_res[1]
                # SingleCharMetricsMode: (acc_char, metric_pos, loss_pos, loss_char)
                # AllCharsMetricsMode: (acc_char, metric_pos)
                if len(val_res) >= 3 and val_res[2] is not None and val_loss_pos is not None:
                    val_loss_pos[epoch] = val_res[2]
                if len(val_res) >= 4 and val_res[3] is not None:
                    val_loss_char[epoch] = val_res[3]
            val_str = metrics_mode.format_val_str(val_acc_char[epoch], val_metric_pos[epoch])
            print(train_str + val_str, flush=True)
  
    except (KeyboardInterrupt, SystemExit):
        # Handle interruption gracefully
        print("\nTraining interrupted, cleaning up resources...")
    finally:
        # Explicit resource cleanup for DataLoader to prevent semaphore leaks
        # This is critical when using persistent_workers=True with multiprocessing
        if num_workers > 0:
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
    
    base = {
        "train_acc_char": train_acc_char[:actual_epochs],
        "val_acc_char": val_acc_char[:actual_epochs],
        "model": mdl.to("cpu"),
        "actual_epochs": actual_epochs,
    }
    return metrics_mode.add_pos_to_result_dict(
        base, train_metric_pos, val_metric_pos, actual_epochs,
        train_loss_pos=train_loss_pos, val_loss_pos=val_loss_pos,
        train_loss_char=train_loss_char, val_loss_char=val_loss_char,
    )


# ==================== Parse Arguments ====================
def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser for command line options."""
    parser = argparse.ArgumentParser(description="Train RNN models for sector classification")
    parser.add_argument(
        "--model_types",
        type=str,
        nargs="+",
        default=["rnn"],
        choices=["rnn", "lstm", "gru", "gawf", "ffn", "dann"],
        help='Model types to train (default: ["rnn"])',
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
        default=50,
        help="Number of training epochs (default: 100)",
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
        "--use_mmap",
        action="store_true",
        default=False,
        help="Load stimuli with memory mapping (mmap_mode='r'). If not set, load as ndarray in memory so num_workers can be used (default: False)",
    )
    parser.add_argument(
        "--nofb",
        action="store_true",
        default=False,
        help="GaWFRNN only: disable feedback. Behavior: (1) Omit --nofb -> full feedback throughout. "
             "(2) Use --nofb only -> no feedback throughout. (3) Use --nofb and --fb_start_epoch N -> no feedback until epoch N, then feedback on (default: False)",
    )
    parser.add_argument(
        "--fb_start_epoch",
        type=int,
        default=999999,
        help="GaWFRNN with --nofb: 0-based epoch at which to turn on feedback and unfreeze U,V. Only meaningful with --nofb; default 999999 means never turn on (default: 999999)",
    )
    parser.add_argument(
        "--data_suffix",
        type=str,
        default="",
        help="Suffix appended to stimulus_reg-* file names. "
             "Example: 'cplx' -> 'stimulus_reg-train-cplx.npy'. "
             "Default: empty string (no suffix).",
    )
    parser.add_argument(
        "--result_suffix",
        type=str,
        default="sector", # ""
        help="Suffix to append to result file names for distinguishing different training runs (default: empty string)",
    )
    parser.add_argument(
        "--cnn_feature_size",
        type=str,
        default="large",
        choices=["large", "small"],
        help="CNN encoder output feature map size: large (64, 12, 12), small (32, 6, 6). Default: large",
    )

    return parser


if __name__ == "__main__":
    # Parse command line arguments
    parser = build_arg_parser()
    args = parser.parse_args()

    # Set global random seed for reproducibility (torch, numpy, random)
    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"   # 可见设备中的 0（由 CUDA_VISIBLE_DEVICES 决定映射到物理哪张卡）

    disable_tqdm_env = os.environ.get('DISABLE_TQDM', '').lower() in ['1', 'true', 'yes']
    enable_tqdm_env  = os.environ.get('ENABLE_TQDM', '').lower() in ['1', 'true', 'yes']
    term_ok = os.environ.get('TERM', '').lower() not in ['', 'dumb']
    use_tqdm = enable_tqdm_env or (
        not disable_tqdm_env and sys.stdout.isatty() and term_ok
    )
    
    # Data path configuration
    base_path = get_base_path()
    stim_train_path, label_train_path, stim_val_path, label_val_path = prepare_data_paths(
        base_path, data_suffix=args.data_suffix
    )
    stims_train, lbls_train, stims_val, lbls_val = load_raw_data(
        stim_train_path, label_train_path, stim_val_path, label_val_path,
        use_mmap=args.use_mmap,
    )

    # Dataset configuration
    use_sector_mode = args.use_sector_mode
    predict_all_chars = args.predict_all_chars
    use_acceleration = args.use_acceleration
    max_chars = 15 # Num of bg digit in 40h is 12

    train_ds, val_ds, num_pos = create_datasets(
        stims_train, lbls_train, stims_val, lbls_val,
        use_sector_mode=args.use_sector_mode,
        predict_all_chars=args.predict_all_chars,
        max_chars=max_chars,
        dataset_class=MC_RNN_Dataset,
    )

    # Model class mapping table
    model_classes = get_model_classes(
        RNNConv,
        LSTMConv,
        GRUConv,
        GaWFRNNConv,
        FeedForwardConv,
        DendriticANNConv,
    )

    # Training configuration (from command line arguments)
    model_types = args.model_types
    hidden_sizes = args.hidden_sizes
    lrs = args.lrs
    weight_decays = args.weight_decays
    dropout_rates = args.dropout_rates
    cnn_feature_sizes = args.cnn_feature_size

    # Create results directory
    results_dir = f"results/models/{args.result_suffix}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
        print(f"Created results directory: {results_dir}")

    # Logger: console + optional file under results_dir
    log_file = os.path.join(results_dir, "train.log")
    logger = setup_logger("train", log_file=log_file)

    # Build hyperparameter combinations: (model_type, hidden_size, lr, weight_decay, dropout_rate)
    experiment_configs = list(
        product(model_types, hidden_sizes, lrs, weight_decays, dropout_rates)
    )

    # Training loop over all hyperparameter combinations
    total_experiments = len(experiment_configs)
    experiment_num = 0

    log_experiment_config(
        logger,
        total_experiments,
        model_types,
        hidden_sizes,
        lrs,
        weight_decays,
        dropout_rates,
        cnn_feature_sizes,
    )

    for model_type, hidden_size, lr, weight_decay, dropout_rate in experiment_configs:
        experiment_num += 1
        log_experiment_start(
            logger,
            experiment_num,
            total_experiments,
            model_type,
            hidden_size,
            lr,
            weight_decay,
            dropout_rate,
        )

        if predict_all_chars:
            num_pos = 0

        # Create model
        if model_type not in model_classes:
            print(f"Warning: Unsupported model_type: {model_type}, skipping...")
            continue

        ModelClass = model_classes[model_type]
        mdl = ModelClass(
                num_classes=10,
                num_pos=num_pos,
                kernel_size=5,
                dropout_rate=dropout_rate,
                hidden_size=hidden_size,
                max_chars=max_chars,
                predict_all_chars=predict_all_chars,
                cnn_feature_size=args.cnn_feature_size,
            )

        logger.info(
            "Created %s model (predict_all_chars=True, max_chars=%s, dropout_rate=%s, hidden_size=%s, cnn_feature_size=%s)",
            model_type.upper(), max_chars, dropout_rate, hidden_size, args.cnn_feature_size,
        )
       
        # [COMPILE] compile model for speed (PyTorch 2.x)
        try:
            mdl = torch.compile(mdl)  # 可选：torch.compile(mdl, mode="max-autotune")
        except Exception as e:
            logger.warning("[COMPILE] torch.compile failed, fallback to eager: %s", e)

        # Train model
        logger.info("Starting training...")
        logger.info("Acceleration training enabled" if use_acceleration else "Using standard training method")
        

        results = network_train(
            mdl,
            train_ds,
            val_ds,
            num_epochs=args.num_epochs,
            lr=lr,
            use_acceleration=use_acceleration,
            weight_decay=weight_decay,
            dropout_rate=dropout_rate,
            rnn_diag_lambda=1e-4,
            use_mmap=args.use_mmap,
            use_tqdm=use_tqdm,
            nofb=args.nofb,
            fb_start_epoch=args.fb_start_epoch,
            seed=args.seed,
            logger=logger,
            cnn_feature_size=args.cnn_feature_size,
        )

        # Save training results
        print(f"\nSaving results for {model_type.upper()} (hidden_size={hidden_size})...")
        mode_suffix = "allchars" if predict_all_chars else ("sector" if use_sector_mode else "coord")
        acc_suffix = "_acc" if use_acceleration else ""
        hp_suffix = f"_lr{lr}_wd{weight_decay}_do{dropout_rate}"
        cnn_feature_size_suffix = "_Lcnn" if args.cnn_feature_size == "large" else "_Scnn"
        # nofb/fb_start_epoch in result path: nofb only -> _nofb; nofb + fb_start_epoch -> _fb{N} only
        if args.nofb:
            if args.fb_start_epoch >= 999999:
                fb_path_suffix = "_nofb"
            else:
                fb_path_suffix = f"_fb{args.fb_start_epoch}"
        else:
            fb_path_suffix = ""
        results_path = os.path.join(
            results_dir,
            f"{model_type}_{mode_suffix}{acc_suffix}_h{hidden_size}{hp_suffix}{fb_path_suffix}{cnn_feature_size_suffix}",
        )

        save_results(results, results_path)
        logger.info("Experiment %s/%s completed!", experiment_num, total_experiments)

    logger.info("=" * 60)
    logger.info("All %s experiments completed! Results saved to: %s/", total_experiments, results_dir)
    logger.info("=" * 60)

