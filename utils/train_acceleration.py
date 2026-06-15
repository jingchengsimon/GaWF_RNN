"""
Acceleration-related utilities for training.
Contains AccelerationConfig, setup_acceleration, build_loaders, run_forward_with_feedback, TrainStepper.
Device/dtype helpers for CUDA/MPS/CPU compatibility (MPS does not support float64).
"""
import json
import os
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from .train_helpers import worker_init_fn


# -----------------------------------------------------------------------------
# Device/dtype helpers (no extra deps). Use at data load / step to avoid float64 on MPS.
# -----------------------------------------------------------------------------

def _is_cuda(device) -> bool:
    """True if device is CUDA (string 'cuda' or 'cuda:N')."""
    if isinstance(device, torch.device):
        return device.type == "cuda"
    return device == "cuda" or (isinstance(device, str) and device.startswith("cuda:"))


def _device_type(device) -> str:
    """Return device_type for autocast: 'cuda' or 'cpu'."""
    if _is_cuda(device):
        return "cuda"
    if isinstance(device, torch.device):
        return device.type
    return device if isinstance(device, str) else "cpu"
 
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
    
    def summary(self, logger=None):
        """Log acceleration configuration summary."""
        import logging

        log = logger or logging.getLogger(__name__)
        sep = "=" * 60
        log.info(sep)
        log.info("Acceleration Configuration:")
        log.info("  Master switch (use_acceleration): %s", self.use_acceleration)
        if self.use_acceleration:
            log.info("  - AMP (Automatic Mixed Precision): %s", self.enable_amp)
            log.info("  - DataLoader Optimization: %s", self.enable_dataloader_opt)
            log.info("  - Batch Size Auto Optimization: %s", self.enable_batch_auto)
            log.info("  - Gradient Scaling: %s", self.enable_gradient_scale)
            log.info(
                "  - Gradient Accumulation: %s (steps=%s)",
                self.enable_grad_accum,
                self.grad_accum_steps,
            )
            log.info("  - Memory Optimization: %s", self.enable_memory_opt)
            log.info(
                "  - DataLoader Prefetch Factor: %s",
                self.dataloader_prefetch_factor,
            )
        else:
            log.info("  All acceleration features disabled")
        log.info(sep)


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


def setup_acceleration(accel_config, device, logger=None):
    """
    Setup acceleration artifacts from config. No if-else on use_acceleration inside training loop.
    Returns: (autocast_fn, scaler, batch_size, num_workers, pin_memory).
    """
    if not accel_config.use_acceleration:
        return (lambda _: nullcontext()), None, 32, 0, False

    autocast_cls, GradScaler_cls, _ = init_acceleration_modules()
    if autocast_cls is None or GradScaler_cls is None:
        if logger is not None:
            logger.warning(
                "Unable to import acceleration modules, using standard training"
            )
        return (lambda _: nullcontext()), None, 32, 0, False

    if logger is not None:
        logger.info("Enabling acceleration training...")

    batch_size = int(os.environ.get("AIM3_BATCH_SIZE", "256"))
    num_workers = int(os.environ.get("AIM3_NUM_WORKERS", "0"))
    pin_memory = os.environ.get("AIM3_PIN_MEMORY", "0").lower() in ("1", "true", "yes")
    use_amp = _is_cuda(device) and accel_config.enable_amp
    autocast_fn = (
        (lambda d: autocast_cls(device_type=_device_type(d))) if use_amp else (lambda _: nullcontext())
    )

    scaler = GradScaler_cls("cuda") if _is_cuda(device) and accel_config.enable_gradient_scale else None

    if logger is not None and _is_cuda(device) and scaler is not None:
        logger.info(
            "Acceleration: batch_size=%s, num_workers=%s, pin_memory=%s, AMP=enabled",
            batch_size,
            num_workers,
            pin_memory,
        )

    return autocast_fn, scaler, batch_size, num_workers, pin_memory


def build_loaders(train_data, val_data, batch_size, num_workers, pin_memory, accel_config, seed):
    """Build train, train-eval, and val DataLoaders from config. Single code path."""
    worker_init_fn_param = partial(worker_init_fn, seed=seed) if num_workers > 0 else None
    train_generator = torch.Generator().manual_seed(seed)

    train_kw = dict(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and num_workers > 0,
        persistent_workers=(num_workers > 0),
        drop_last=True,
        generator=train_generator,
        worker_init_fn=worker_init_fn_param,
    )
    # Validation loader: full-pass evaluation mode
    val_kw = dict(
        dataset=val_data,
        batch_size=batch_size,
        shuffle=False, #True,
        num_workers=num_workers,
        pin_memory=pin_memory and num_workers > 0,
        persistent_workers=(num_workers > 0),
        drop_last=False,
        worker_init_fn=worker_init_fn_param,
    )
    # Train-eval loader: subset of train_data with the same length as val_data (or smaller),
    # used only for eval-mode metrics to keep runtime reasonable.
    if val_data is not None:
        eval_len = min(len(train_data), len(val_data))
        eval_indices = list(range(eval_len))
        train_eval_dataset = Subset(train_data, eval_indices)
    else:
        train_eval_dataset = train_data

    train_eval_kw = dict(
        dataset=train_eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and num_workers > 0,
        persistent_workers=(num_workers > 0),
        drop_last=False,
        worker_init_fn=worker_init_fn_param,
    )
    if num_workers > 0:
        train_kw["prefetch_factor"] = accel_config.dataloader_prefetch_factor
        val_kw["prefetch_factor"] = accel_config.dataloader_prefetch_factor
        train_eval_kw["prefetch_factor"] = accel_config.dataloader_prefetch_factor

    train_dl = DataLoader(**train_kw)
    train_eval_dl = DataLoader(**train_eval_kw)
    val_dl = DataLoader(**val_kw)

    return train_dl, train_eval_dl, val_dl


def run_forward_with_feedback(mdl, inputs, use_feedback=None):
    """
    Single forward path for both training and evaluation.
    Stateless: always start with prev_feedback=None and reset_feedback=True so feedback
    is updated only within the same batch's timestep loop. No cross-batch/epoch table.
    Returns: (out_char, out_pos).
    """
    if hasattr(mdl, "prev_feedback"):
        mdl.prev_feedback = None
    if use_feedback is not None:
        out_char, out_pos = mdl(inputs, use_feedback=use_feedback, reset_feedback=True)
    else:
        out_char, out_pos = mdl(inputs)
    return out_char, out_pos


def _total_norm_from_tensors(tensors):
    total_sq = 0.0
    count = 0
    for tensor in tensors:
        if tensor is None:
            continue
        detached = tensor.detach().float()
        total_sq += float(torch.sum(detached * detached).item())
        count += 1
    return (total_sq ** 0.5) if count else None


def _named_parameter_groups(mdl):
    groups = {
        "all": [],
        "gawf_U": [],
        "gawf_V": [],
        "gawf_projector": [],
    }
    for name, param in mdl.named_parameters():
        groups["all"].append(param)
        if "U" in name:
            groups["gawf_U"].append(param)
        elif "V" in name:
            groups["gawf_V"].append(param)
        elif "proj_out" in name or "hidden_projectors" in name:
            groups["gawf_projector"].append(param)
    return groups


def _parameter_norms(mdl):
    out = {}
    for group_name, params in _named_parameter_groups(mdl).items():
        norm = _total_norm_from_tensors(params)
        if norm is not None:
            out[f"param_norm_{group_name}"] = norm
    return out


def _gradient_norms(mdl):
    out = {}
    for group_name, params in _named_parameter_groups(mdl).items():
        norm = _total_norm_from_tensors([p.grad for p in params if p.grad is not None])
        if norm is not None:
            out[f"grad_norm_preclip_{group_name}"] = norm
    return out


class GawfDiagnosticsRecorder:
    """Opt-in JSONL recorder for GaWF gate, feedback, gradient, and parameter stats."""

    def __init__(
        self,
        enabled=False,
        output_path=None,
        every_n_steps=1,
        gate_saturation_eps=0.01,
        logger=None,
    ):
        self.enabled = bool(enabled)
        self.output_path = output_path
        self.every_n_steps = max(1, int(every_n_steps))
        self.gate_saturation_eps = float(gate_saturation_eps)
        self.logger = logger
        self.global_batch = 0
        self.current_epoch = 0
        self.current_rows = []
        self.epoch_summaries = []
        if self.enabled and self.output_path:
            path = Path(self.output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
            if self.logger is not None:
                self.logger.info("GaWF diagnostics enabled: %s", path)

    def begin_epoch(self, epoch):
        if not self.enabled:
            return
        self.current_epoch = int(epoch) + 1
        self.current_rows = []

    def should_record_batch(self):
        if not self.enabled:
            return False
        return self.global_batch % self.every_n_steps == 0

    def begin_forward(self, mdl, record_batch):
        if record_batch and hasattr(mdl, "begin_gawf_diagnostics"):
            mdl.begin_gawf_diagnostics(self.gate_saturation_eps)

    def pop_forward(self, mdl, record_batch):
        if record_batch and hasattr(mdl, "pop_gawf_diagnostics"):
            return mdl.pop_gawf_diagnostics()
        return {}

    def write_step(self, row):
        if not self.enabled:
            return
        clean_row = {
            key: value
            for key, value in row.items()
            if value is not None
        }
        self.current_rows.append(clean_row)
        self._append_jsonl({"record_type": "step", **clean_row})

    def finish_epoch(self, epoch):
        if not self.enabled or not self.current_rows:
            return None
        summary = {
            "record_type": "epoch_summary",
            "epoch": int(epoch) + 1,
            "num_recorded_steps": len(self.current_rows),
        }
        numeric_keys = sorted(
            {
                key
                for row in self.current_rows
                for key, value in row.items()
                if isinstance(value, (int, float))
                and not isinstance(value, bool)
                and key not in ("epoch", "batch_idx", "global_batch")
            }
        )
        for key in numeric_keys:
            values = [
                float(row[key])
                for row in self.current_rows
                if key in row and isinstance(row[key], (int, float))
                and not isinstance(row[key], bool)
            ]
            if not values:
                continue
            summary[f"{key}_mean"] = sum(values) / len(values)
            summary[f"{key}_min"] = min(values)
            summary[f"{key}_max"] = max(values)
        self.epoch_summaries.append(summary)
        self._append_jsonl(summary)
        return summary

    def to_result_dict(self):
        if not self.enabled:
            return None
        return {
            "enabled": True,
            "path": self.output_path,
            "every_n_steps": self.every_n_steps,
            "gate_saturation_eps": self.gate_saturation_eps,
            "epoch_summaries": self.epoch_summaries,
        }

    def _append_jsonl(self, row):
        if not self.output_path:
            return
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")


class TrainStepper:
    """
    One-step training: forward, loss, backward, optional optimizer step.
    Encapsulates AMP/grad-accum so the training loop has no acceleration branches.
    """

    def __init__(
        self,
        mdl,
        optim,
        loss_fn,
        accel_config,
        device,
        scaler,
        autocast_fn,
        pin_memory,
        gawf_diagnostics=None,
    ):
        self.mdl = mdl
        self.optim = optim
        self.loss_fn = loss_fn
        self.accel_config = accel_config
        self.device = device
        self.scaler = scaler
        self.autocast_fn = autocast_fn
        self.pin_memory = pin_memory
        self.gawf_diagnostics = gawf_diagnostics

    def step(self, batch, batch_idx, use_feedback_this_epoch):
        """
        Run one batch: move to device (with MPS-safe dtypes), forward, loss, backward, step at accum boundary.
        Returns: (current_loss, out_char, out_pos, labels_device). labels_device is labels
        already on device to avoid duplicate .to(device) in the training loop.
        MPS does not support float64: inputs and float labels are cast to float32 when needed.
        """
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            inputs, labels = batch[0], batch[1]
        else:
            inputs, labels = batch, None

        # MPS/CUDA: avoid float64 on device (cast only when necessary to avoid CUDA overhead)
        if inputs.dtype == torch.float64:
            inputs = inputs.float()
        non_blocking = self.pin_memory
        inputs = inputs.to(self.device, non_blocking=non_blocking)
        if labels is not None:
            if labels.dtype == torch.float64:
                labels = labels.float()
            labels = labels.to(self.device, non_blocking=non_blocking)
            
        labels_device = labels


        accum_steps = self.accel_config.grad_accum_steps
        if batch_idx % accum_steps == 0:
            self.optim.zero_grad()
        loss_scale = 1.0 / accum_steps

        diag = self.gawf_diagnostics
        record_diag = diag.should_record_batch() if diag is not None else False
        if diag is not None:
            diag.begin_forward(self.mdl, record_diag)

        with self.autocast_fn(self.device):
            out_char, out_pos = run_forward_with_feedback(
                self.mdl, inputs,
                use_feedback=use_feedback_this_epoch,
            )
            loss = self.loss_fn(out_char, out_pos, labels_device) * loss_scale
        diag_row = diag.pop_forward(self.mdl, record_diag) if diag is not None else {}

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            if (batch_idx + 1) % accum_steps == 0:
                self.scaler.unscale_(self.optim)
                if record_diag:
                    diag_row.update(_gradient_norms(self.mdl))
                found_nonfinite = any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in self.mdl.parameters()
                )
                if found_nonfinite:
                    if record_diag:
                        diag_row.update(
                            {
                                "epoch": diag.current_epoch,
                                "batch_idx": int(batch_idx),
                                "global_batch": int(diag.global_batch),
                                "loss": float(loss.detach().item() / loss_scale),
                                "found_nonfinite_grad": True,
                            }
                        )
                        diag_row.update(_parameter_norms(self.mdl))
                        diag.write_step(diag_row)
                    if diag is not None:
                        diag.global_batch += 1
                    self.optim.zero_grad(set_to_none=True)
                    self.scaler.update()
                    return loss.detach().item(), out_char.detach(), out_pos.detach() if out_pos is not None else None, labels_device
                torch.nn.utils.clip_grad_value_(self.mdl.parameters(), clip_value=1.0)
                torch.nn.utils.clip_grad_norm_(self.mdl.parameters(), max_norm=1.0)
                self.scaler.step(self.optim)
                self.scaler.update()
                self.optim.zero_grad(set_to_none=True)
        else:
            loss.backward()
            if (batch_idx + 1) % accum_steps == 0:
                if record_diag:
                    diag_row.update(_gradient_norms(self.mdl))
                torch.nn.utils.clip_grad_value_(self.mdl.parameters(), clip_value=1.0)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.mdl.parameters(), max_norm=1.0)
                if not torch.isfinite(grad_norm):
                    if record_diag:
                        diag_row.update(
                            {
                                "epoch": diag.current_epoch,
                                "batch_idx": int(batch_idx),
                                "global_batch": int(diag.global_batch),
                                "loss": float(loss.detach().item() / loss_scale),
                                "found_nonfinite_grad": True,
                            }
                        )
                        diag_row.update(_parameter_norms(self.mdl))
                        diag.write_step(diag_row)
                    if diag is not None:
                        diag.global_batch += 1
                    self.optim.zero_grad(set_to_none=True)
                    return loss.detach().item(), out_char.detach(), out_pos.detach() if out_pos is not None else None, labels_device
                self.optim.step()
                self.optim.zero_grad(set_to_none=True)

        if record_diag:
            diag_row.update(
                {
                    "epoch": diag.current_epoch,
                    "batch_idx": int(batch_idx),
                    "global_batch": int(diag.global_batch),
                    "loss": float(loss.detach().item() / loss_scale),
                    "found_nonfinite_grad": False,
                }
            )
            diag_row.update(_parameter_norms(self.mdl))
            diag.write_step(diag_row)
        if diag is not None:
            diag.global_batch += 1

        return loss.detach().item(), out_char.detach(), out_pos.detach() if out_pos is not None else None, labels_device
