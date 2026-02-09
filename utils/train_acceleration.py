"""
Acceleration-related utilities for training.
Contains AccelerationConfig and init_acceleration_modules for AMP, batch size, DataLoader.
"""
# AccelerationConfig and init_acceleration_modules are only used when acceleration is enabled.
# By default (use_acceleration=False), init_acceleration_modules is not called to keep imports lazy.


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


def init_acceleration_modules():
    """Initialize acceleration training related modules (imported only when needed)."""
    try:
        from torch.amp import autocast, GradScaler
        try:
            import psutil
        except ImportError:
            psutil = None
        return autocast, GradScaler, psutil
    except ImportError:
        return None, None, None
