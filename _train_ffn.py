"""
Standalone FeedForward Network training script
Used to train FeedForward models and save results
"""
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==================== Acceleration Training Modules (Optional) ====================
# These modules are only used when use_acceleration=True
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
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
    return 0.0

def _find_optimal_batch_size(model, train_data, device='cuda', start_batch_size=32, max_batch_size=256):
    """
    Automatically find optimal batch_size (only used in acceleration mode)
    """
    model.eval()
    batch_size = start_batch_size
    
    while batch_size <= max_batch_size:
        try:
            # Create a small batch for testing
            test_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
            sample_batch = next(iter(test_loader))
            inputs, _ = sample_batch
            
            # Try forward pass
            with torch.no_grad():
                inputs = inputs.to(device)
                _ = model(inputs)
            
            # If successful, try next larger batch size
            batch_size *= 2
        except RuntimeError as e:
            if "out of memory" in str(e):
                # If OOM, return previous batch size
                torch.cuda.empty_cache()
                return batch_size // 2
            else:
                raise e
    
    torch.cuda.empty_cache()
    return min(batch_size // 2, max_batch_size)


# ==================== Dataset Class ====================
class MC_FF_Dataset(Dataset):
    """
    Dataset class for loading stacks of frames as multichannel images
    for use in testing the performance of purely feedforward models
    """
    def __init__(self, data, labels, stack_size=3, use_sector=False, num_sectors=9):
        """
        Args:
            data (np.ndarray): Array of shape (num_samples, num_frames, height, width)
            labels (np.ndarray): DataFrame with columns ['fg_char_id', 'fg_char_x', 'fg_char_y']
            stack_size (int): Number of frames to stack for input as multichannel image
            use_sector (bool): If True, map (x, y) position to sector id 0-(num_sectors-1)
            num_sectors (int): Number of sectors, e.g., 9 means 0-8 sectors (3x3 grid)
        """
        self.data = data
        self.labels = labels[['fg_char_id', 'fg_char_x', 'fg_char_y']].values
        self.stack_size = stack_size
        self.use_sector = use_sector
        self.num_sectors = num_sectors

    def __len__(self):
        return self.data.shape[0] - self.stack_size + 1

    def __getitem__(self, idx):
        # Stack frames to create a multichannel image
        stacked_frames = self.data[idx:(idx + self.stack_size)].astype(np.float32)
        label = self.labels[idx + self.stack_size - 1].copy()
        
        if self.use_sector:
            # Use image width and height to map (x, y) to a grid_size x grid_size grid,
            # obtaining sector id 0-(num_sectors-1) (e.g., num_sectors=9 -> 3x3 grid)
            height = self.data.shape[-2]
            width = self.data.shape[-1]

            # Derive grid_size for each dimension from num_sectors (assuming num_sectors is a perfect square, e.g., 9, 16)
            grid_size = int(np.sqrt(self.num_sectors))
            if grid_size * grid_size != self.num_sectors:
                raise ValueError(f"num_sectors={self.num_sectors} is not a perfect square, cannot form grid_size x grid_size grid")

            x = label[1].astype(np.float32)
            y = label[2].astype(np.float32)

            # Normalize coordinates to [0, grid_size) then round, using (width-1)/(height-1) to avoid out-of-bounds
            col = int((x / max(width - 1, 1)) * grid_size)
            row = int((y / max(height - 1, 1)) * grid_size)

            # Prevent out-of-bounds due to numerical or boundary issues
            col = np.clip(col, 0, grid_size - 1)
            row = np.clip(row, 0, grid_size - 1)

            # Encode sector id in row-major order: row * grid_size + col, range 0-(num_sectors-1)
            sector = row * grid_size + col

            # New label: [char_id, sector_id]
            label = np.array([label[0].astype(np.int64), sector], dtype=np.int64)
        else:
            # Coordinate mode: [char_id, x, y]
            label = np.array([label[0].astype(np.int64), label[1], label[2]], dtype=np.float32)
        
        return stacked_frames, label


# ==================== Model Class ====================
class FeedForwardConv(nn.Module):
    def __init__(self, input_channels, num_classes, num_pos, mnist_pre=None, kernel_size=3, device='cuda'):
        super(FeedForwardConv, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=kernel_size, padding='same')
        if mnist_pre is not None:
            # set weights of self.conv1 to mnist_pre.conv1
            self.conv1.weight.data = torch.cat([torch.zeros(mnist_pre.conv1.weight.shape).to(self.device), 
                                                  torch.zeros(mnist_pre.conv1.weight.shape).to(self.device),
                                                  mnist_pre.conv1.weight], dim=1)
            self.conv1.bias.data = mnist_pre.conv1.bias.data
        self.MP1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.LNorm1 = nn.LayerNorm([32, 48, 48])
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        if mnist_pre is not None:
            # set weights of self.conv2 to mnist_pre.conv2
            self.conv2.weight.data = mnist_pre.conv2.weight
            self.conv2.bias.data = mnist_pre.conv2.bias.data
        self.MP2 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.LNorm2 = nn.LayerNorm([64, 12, 12])
        self.fc1 = nn.Linear(64 * 12 * 12, 512)
        self.dropout = nn.Dropout(0.5)
        self.fcchar = nn.Linear(512, num_classes)
        self.fcpos = nn.Linear(512, num_pos)
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
        x = self.encoder(x)

        x = x.view(x.size(0), -1)
        x = self.middle(x)

        char_out, pos_out = self.classifier(x)
        return char_out, pos_out


# ==================== Training Function ====================
def network_train_ffn(mdl, train_data, val_data, num_epochs=50, loss_weights=None, lr=0.001, 
                      use_acceleration=False, weight_decay=1e-4, 
                      use_early_stopping=True, early_stopping_patience=15, min_delta=0.001):
    """
    Train FeedForward model, supports sector mode and coordinate mode
    
    Args:
        mdl: Model (FeedForwardConv)
        train_data: Training dataset (MC_FF_Dataset)
        val_data: Validation dataset (MC_FF_Dataset)
        num_epochs: Number of training epochs
        loss_weights: [character loss weight, position loss weight], if None, automatically set based on use_sector
                     - sector mode default: [1, 1]
                     - coordinate mode default: [1, 0.001]
        lr: Learning rate
        use_acceleration: Whether to use acceleration training (default False)
        weight_decay: L2 regularization coefficient (weight decay), default 1e-4
        use_early_stopping: Whether to use early stopping mechanism, default True
        early_stopping_patience: Early stopping patience, number of epochs without validation improvement, default 15
        min_delta: Minimum improvement threshold for early stopping, default 0.001
    """
    # Get use_sector information from dataset
    use_sector = train_data.use_sector
    
    # Set default loss_weights based on use_sector
    if loss_weights is None:
        if use_sector:
            loss_weights = [1, 1]  # sector mode: character and sector loss weights equal
        else:
            loss_weights = [1, 0.001]  # coordinate mode: position loss weight smaller (MSE usually has larger values)
    
    # Place parameters according to model's internal device (can be 'cuda' or 'cpu')
    device = mdl.device
    mdl.to(device)
    
    # ========== Acceleration Training Module Initialization (only used when use_acceleration=True) ==========
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
            # Automatically find optimal batch_size
            if device == 'cuda':
                print("Automatically finding optimal batch_size...")
                batch_size = _find_optimal_batch_size(mdl, train_data, device=device)
                print(f"Using batch_size = {batch_size}")
            
            # Automatically set num_workers
            if psutil_module is not None:
                num_workers = min(4, psutil_module.cpu_count(logical=False))
            else:
                import os
                num_workers = min(4, os.cpu_count() or 1)
            
            # Enable pin_memory (GPU only)
            pin_memory = (device == 'cuda')
            show_gpu_usage = True
            
            # Initialize mixed precision training scaler
            if device == 'cuda':
                scaler = GradScaler_cls('cuda')
                print(f"Acceleration settings: batch_size={batch_size}, num_workers={num_workers}, "
                      f"pin_memory={pin_memory}, mixed precision training=enabled")
            
            # In acceleration mode, limit batch_size to no more than 32
            if batch_size > 32:
                original_batch_size = batch_size
                batch_size = 32
                print(f"batch_size limited from {original_batch_size} to {batch_size} to maintain same convergence speed as original mode")
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
        # Character loss (same for both modes)
        labels_char = labels[:, 0].long()
        loss_char = criterion_char(out_char, labels_char)
        
        # Position loss (different methods based on use_sector)
        if use_sector:
            # sector mode: classification loss
            labels_pos = labels[:, 1].long()  # (B,) -> sector id
            loss_pos = criterion_pos(out_pos, labels_pos)
        else:
            # coordinate mode: regression loss (MSE)
            labels_pos = labels[:, 1:].float()  # (B, 2) -> [x, y]
            loss_pos = criterion_pos(out_pos, labels_pos)
        
        loss = loss_weights[0] * loss_char + loss_weights[1] * loss_pos
        return loss

    def evaluate(mdl, data_loader):
        mdl.eval()
        total_acc_char = 0
        total_metric_pos = 0  # sector mode: accuracy; coordinate mode: MSE
        num_batches = 0
        with torch.no_grad():
            for batch in data_loader:
                inputs, labels = batch
                
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
                
                # Character accuracy
                total_acc_char += (torch.argmax(out_char, dim=1) == labels[:, 0].long()).float().mean().item()
                
                # Position-related metrics (different methods based on use_sector)
                if use_sector:
                    # sector mode: calculate accuracy
                    total_metric_pos += (torch.argmax(out_pos, dim=1) == labels[:, 1].long()).float().mean().item()
                else:
                    # coordinate mode: calculate MSE
                    labels_pos = labels[:, 1:].float()  # (B, 2)
                    total_metric_pos += F.mse_loss(out_pos, labels_pos, reduction='mean').item()
                
                num_batches += 1
        
        # Return results
        acc_char = (total_acc_char / num_batches) * 100
        if use_sector:
            metric_pos = (total_metric_pos / num_batches) * 100  # accuracy (percentage)
        else:
            metric_pos = total_metric_pos / num_batches  # MSE (pixel squared)
        return acc_char, metric_pos

    # data loader (select different configurations based on acceleration mode)
    if use_acceleration:
        train_dl = DataLoader(
            train_data, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0
        )
        val_dl = DataLoader(
            val_data, 
            batch_size=batch_size, 
            shuffle=False,  # validation set doesn't need shuffle
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0
        )
    else:
        # Original method: simple configuration
        train_dl = DataLoader(train_data, batch_size=32, shuffle=True)
        val_dl = DataLoader(val_data, batch_size=512, shuffle=False)
    
    train_acc_char = np.zeros(num_epochs)
    val_acc_char = np.zeros(num_epochs)
    train_metric_pos = np.zeros(num_epochs)  # sector mode: accuracy; coordinate mode: MSE
    val_metric_pos = np.zeros(num_epochs)
    
    # Early stopping mechanism
    best_val_metric = -np.inf if use_sector else np.inf  # sector mode: larger is better; coordinate mode: smaller is better
    best_val_epoch = 0
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        mdl.train()
        
        # Training loop
        epoch_train_acc_char = 0.0
        epoch_train_metric_pos = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_dl):
            inputs, labels = batch
            
            # Select data transfer method based on acceleration mode
            if use_acceleration and pin_memory:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
            else:
                labels = labels.to(device)
            
            optim.zero_grad()
            
            # Select whether to use mixed precision training based on acceleration mode
            if use_acceleration and scaler is not None and autocast_fn is not None:
                with autocast_fn('cuda'):
                    out_char, out_pos = mdl(inputs)
                    loss = loss_fn(out_char, out_pos, labels)
                
                scaler.scale(loss).backward()
                # Gradient clipping (mixed precision training)
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(mdl.parameters(), max_norm=2.0)
                scaler.step(optim)
                scaler.update()
            else:
                # Original method: standard training
                out_char, out_pos = mdl(inputs)
                loss = loss_fn(out_char, out_pos, labels)
                loss.backward()
                # Gradient clipping (standard training)
                torch.nn.utils.clip_grad_norm_(mdl.parameters(), max_norm=2.0)
                optim.step()
            
            # Calculate training metrics
            # Character accuracy
            epoch_train_acc_char += (torch.argmax(out_char, dim=1) == labels[:, 0].long()).float().mean().item()
            
            # Position-related metrics (different methods based on use_sector)
            if use_sector:
                # sector mode: calculate accuracy
                epoch_train_metric_pos += (torch.argmax(out_pos, dim=1) == labels[:, 1].long()).float().mean().item()
            else:
                # coordinate mode: calculate MSE
                labels_pos = labels[:, 1:].float()  # (B, 2)
                epoch_train_metric_pos += F.mse_loss(out_pos, labels_pos, reduction='mean').item()
            
            num_batches += 1
            
            # Acceleration mode: periodically clear GPU cache
            if use_acceleration and batch_idx % 100 == 0 and device == 'cuda':
                torch.cuda.empty_cache()
        
        # Calculate epoch average metrics
        train_acc_char[epoch] = (epoch_train_acc_char / num_batches) * 100
        if use_sector:
            train_metric_pos[epoch] = (epoch_train_metric_pos / num_batches) * 100  # accuracy (percentage)
        else:
            train_metric_pos[epoch] = epoch_train_metric_pos / num_batches  # MSE (pixel squared)
        
        # Format output string
        gpu_info = ""
        if use_acceleration and show_gpu_usage and device == 'cuda' and torch.cuda.is_available():
            gpu_mem = _get_gpu_memory_usage()
            gpu_info = f" | GPU memory: {gpu_mem:.1f}%"
        
        if use_sector:
            train_str = f"Epoch {epoch + 1}/{num_epochs} - Train (char, pos): ({train_acc_char[epoch]:.2f}%, {train_metric_pos[epoch]:.2f}%){gpu_info}"
        else:
            train_str = f"Epoch {epoch + 1}/{num_epochs} - Train (char, pos): ({train_acc_char[epoch]:.2f}%, {train_metric_pos[epoch]:.2f} pix^2){gpu_info}"

        with torch.no_grad():
            val_acc_char[epoch], val_metric_pos[epoch] = evaluate(mdl, val_dl)
            if use_sector:
                val_str = f" Validation (char, pos): ({val_acc_char[epoch]:.2f}%, {val_metric_pos[epoch]:.2f}%)"
            else:
                val_str = f" Validation (char, pos): ({val_acc_char[epoch]:.2f}%, {val_metric_pos[epoch]:.2f} pix^2)"
            print(train_str, val_str)
            
            # Early stopping check: judge based on validation metric
            if use_early_stopping:
                if use_sector:
                    # sector mode: use position accuracy as main metric (higher is better)
                    current_val_metric = val_metric_pos[epoch]
                    improved = current_val_metric > best_val_metric + min_delta
                else:
                    # coordinate mode: use character accuracy as main metric (higher is better)
                    current_val_metric = val_acc_char[epoch]
                    improved = current_val_metric > best_val_metric + min_delta
                
                if improved:
                    best_val_metric = current_val_metric
                    best_val_epoch = epoch
                    patience_counter = 0
                    # Save best model state
                    best_model_state = mdl.state_dict().copy()
                    if use_sector:
                        print(f"  ✓ Validation performance improved! Current best: {best_val_metric:.2f}% (epoch {epoch + 1})")
                    else:
                        print(f"  ✓ Validation performance improved! Current best: {best_val_metric:.2f}% (epoch {epoch + 1})")
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"\nEarly stopping triggered: validation performance did not improve for {early_stopping_patience} epochs")
                        if use_sector:
                            print(f"Best validation performance: {best_val_metric:.2f}% (epoch {best_val_epoch + 1})")
                        else:
                            print(f"Best validation performance: {best_val_metric:.2f}% (epoch {best_val_epoch + 1})")
                        # Restore best model
                        if best_model_state is not None:
                            mdl.load_state_dict(best_model_state)
                            print("Best model state restored")
                        break

    torch.cuda.empty_cache()

    # If early stopping triggered, only return actual trained epochs (epoch starts from 0, so actually trained epoch+1 epochs)
    actual_epochs = epoch + 1
    
    # Return different key names based on use_sector, only return actual trained epochs
    if use_sector:
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
        filepath: Save path (e.g., 'results_ffn' or 'results_ffn_reg')
    """
    # Create save dictionary (does not include model, because model is too large)
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
    # Data path configuration
    stim_train_path = "/G/MIMOlab/Codes/aim3_RNN/stimuli/stimulus_reg-train.npy"
    label_train_path = "/G/MIMOlab/Codes/aim3_RNN/stimuli/stimulus_reg-train.tsv"
    stim_val_path = "/G/MIMOlab/Codes/aim3_RNN/stimuli/stimulus_reg-validation.npy"
    label_val_path = "/G/MIMOlab/Codes/aim3_RNN/stimuli/stimulus_reg-validation.tsv"
    
    # Load data
    print("Loading data...")
    stims_train = np.load(stim_train_path, allow_pickle=True)
    lbls_train = pd.read_csv(label_train_path, sep="\t", index_col=0)
    
    stims_val = np.load(stim_val_path, allow_pickle=True)
    lbls_val = pd.read_csv(label_val_path, sep="\t", index_col=0)
    
    # Configuration
    stack_size = 3  # Number of frames to stack
    num_classes = 10  # Character classes
    use_sector_mode = True  # Set to True to use sector mode, False to use coordinate mode
    use_acceleration = True  # Set to True to enable acceleration training, False to use original method
    
    if use_sector_mode:
        # sector mode: 3x3 grid -> 9 sectors
        num_pos = 9  # number of sectors
        train_ds = MC_FF_Dataset(stims_train, lbls_train, stack_size=stack_size, use_sector=True, num_sectors=num_pos)
        val_ds = MC_FF_Dataset(stims_val, lbls_val, stack_size=stack_size, use_sector=True, num_sectors=num_pos)
        print("Using sector mode (3x3 grid, 9 sectors)")
    else:
        # coordinate mode: directly predict (x, y) coordinates
        num_pos = 2  # x, y coordinates
        train_ds = MC_FF_Dataset(stims_train, lbls_train, stack_size=stack_size, use_sector=False)
        val_ds = MC_FF_Dataset(stims_val, lbls_val, stack_size=stack_size, use_sector=False)
        print("Using coordinate mode (directly predict x, y coordinates)")
    
    print(f"Using stack_size={stack_size}")
    
    # Create model
    print("Creating model...")
    mdl_ff = FeedForwardConv(input_channels=stack_size, num_classes=num_classes, num_pos=num_pos, kernel_size=5)
    print(f"Model created (num_classes={num_classes}, num_pos={num_pos}, kernel_size=5)")
    
    # Train model (loss_weights will be automatically set based on use_sector, can also be manually specified)
    print("Starting training...")
    
    if use_acceleration:
        print("Acceleration training enabled")
    else:
        print("Using standard training method")
    
    results_ff = network_train_ffn(
        mdl_ff, 
        train_ds, 
        val_ds, 
        num_epochs=200, 
        use_acceleration=use_acceleration,  # Control whether to use acceleration training
        weight_decay=1e-4,  # L2 regularization, prevent overfitting
        use_early_stopping=False,
        early_stopping_patience=15,  # Early stopping patience value
        min_delta=0.001  # Early stopping minimum improvement threshold
    )
    
    # Save training results
    print("\nSaving results...")
    mode_suffix = "sector" if use_sector_mode else "coord"
    acc_suffix = "_acc" if use_acceleration else ""
    # File naming includes both mode for easy distinction of different experiments
    # e.g., results_ffn_sector.pkl, results_ffn_coord_acc.pkl
    results_ff_path = f"results_ffn_{mode_suffix}{acc_suffix}"
    
    save_results(results_ff, results_ff_path)
    
    print("\nTraining completed!")

