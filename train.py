from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import time
import os
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from pinn import TrainingConfig
from pinn.physics import residual_statistics
from pinn.losses import (
    PINNLossComponents,
    compute_pinn_loss_components,
    compute_pinn_loss_components_ssfm,
)


def _resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def _configure_scheduler(optimizer: torch.optim.Optimizer, config: TrainingConfig):
    if config.scheduler_patience <= 0:
        return None
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        verbose=True,
    )


def _estimate_teacher_residual_mse(dataset, config: TrainingConfig):
    """Estimate teacher residual MSE using finite differences on the ground-truth grid."""

    if dataset is None:
        return None

    stats = getattr(dataset, "stats", None)
    targets = getattr(dataset, "targets", None)
    z = getattr(dataset, "z", None)
    t = getattr(dataset, "t", None)

    if stats is None or targets is None or z is None or t is None:
        return None
    if not hasattr(stats, "denormalise_targets"):
        return None
    if not isinstance(targets, torch.Tensor):
        return None

    z_array = np.asarray(z, dtype=np.float64)
    t_array = np.asarray(t, dtype=np.float64)
    if z_array.ndim != 1 or t_array.ndim != 1:
        return None

    try:
        grid = targets.view(z_array.size, t_array.size, -1).cpu().numpy()
    except Exception:
        return None

    amplitudes = stats.denormalise_targets(grid)
    if amplitudes.shape[-1] < 2:
        return None

    amplitudes_complex = amplitudes[..., 0] + 1j * amplitudes[..., 1]
    _, _, mse = residual_statistics(z_array, t_array, amplitudes_complex, config)
    return mse


def train(
    model: nn.Module,
    dataloader: DataLoader,
    config: TrainingConfig,
    dataset=None,
) -> Dict[str, List[float]]:
    """Run a training loop supporting both supervised and PINN modes."""

    device = _resolve_device(config.device)
    print(f"Using device: {device}")
    model = model.to(device)

    start_time = time.time()

    # Propagate MATLAB call controls (if any) via environment variables.
    if getattr(config, "matlab_layer", None) is not None:
        os.environ["MATLAB_LAYER"] = str(config.matlab_layer)
    else:
        os.environ.pop("MATLAB_LAYER", None)
    if getattr(config, "matlab_call_idx", None) is not None:
        os.environ["MATLAB_CALL_IDX"] = str(config.matlab_call_idx)
    else:
        os.environ.pop("MATLAB_CALL_IDX", None)
    if getattr(config, "matlab_log_epoch_tag", None):
        os.environ["MATLAB_LOG_EPOCH"] = str(config.matlab_log_epoch_tag)

    # Prepare per-epoch checkpoint directory with descriptive naming.
    hw_flag = "hardware" if any(config.external_layers) else "soft"
    mat_flag = "exp" if config.matlab_layer is not None else "sim"
    folder_name = f"mode-{config.mode}_ff-{config.fourier_features}_{hw_flag}_{mat_flag}"
    run_dir = Path("checkpoints") / folder_name
    run_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = _configure_scheduler(optimizer, config)

    history: Dict[str, List[float]] = {"loss": []}
    teacher_residual_rms: Optional[float] = None
    if config.mode == "pinn":
        if dataset is None:
            raise ValueError("PINN mode requires access to the full dataset for condition sampling.")

        teacher_residual_mse = _estimate_teacher_residual_mse(dataset, config)
        if teacher_residual_mse is not None and teacher_residual_mse > 0.0:
            teacher_residual_rms = float(teacher_residual_mse ** 0.5)
            if config.residual_scale <= 0.0:
                config.residual_scale = float(1.0 / teacher_residual_mse)
                print(
                    f"Auto-tuned residual_scale to {config.residual_scale:.3e} "
                    f"(teacher residual RMS ~ {teacher_residual_rms:.3e})"
                )
            else:
                baseline = config.residual_weight * config.residual_scale * teacher_residual_mse
                print(
                    f"Teacher residual RMS ~ {teacher_residual_rms:.3e}; "
                    f"baseline weighted physics term ~ {baseline:.3e}"
                )
        elif config.residual_scale <= 0.0:
            config.residual_scale = 1.0
            print("Falling back to residual_scale=1.0 (auto calibration unavailable).")

        loss_fn = (
            compute_pinn_loss_components_ssfm
            if config.pde_variant == "ssfm"
            else compute_pinn_loss_components
        )

        history.update(
            {
                "data_loss": [],
                "initial_loss": [],
                "boundary_loss": [],
                "residual_loss": [],
                "physics_scale": [],
                "adaptive_balance": [],
            }
        )
        if teacher_residual_rms is not None:
            history["teacher_residual_rms"] = teacher_residual_rms

    balance_state: Optional[float] = None

    for epoch in range(1, config.num_epochs + 1):
        # Refresh MATLAB controls per epoch; if matlab_layer is None, skip entirely.
        if config.matlab_layer is not None:
            current_layer = config.matlab_layer
            current_call_idx = config.matlab_call_idx
            if config.matlab_layer_candidates:
                current_layer = random.choice(config.matlab_layer_candidates)
            if config.matlab_call_idx_candidates:
                current_call_idx = random.choice(config.matlab_call_idx_candidates)

            os.environ["MATLAB_LAYER"] = str(current_layer)
            if current_call_idx is not None:
                os.environ["MATLAB_CALL_IDX"] = str(current_call_idx)
            else:
                os.environ.pop("MATLAB_CALL_IDX", None)
            if getattr(config, "matlab_log_epoch_tag", None):
                os.environ["MATLAB_LOG_EPOCH"] = f"{config.matlab_log_epoch_tag}_{epoch}"
        else:
            os.environ.pop("MATLAB_LAYER", None)
            os.environ.pop("MATLAB_CALL_IDX", None)
            os.environ.pop("MATLAB_LOG_EPOCH", None)

        physics_scale = 1.0
        if config.mode == "pinn" and config.residual_warmup_epochs > 0:
            warmup = max(config.residual_warmup_epochs, 1)
            physics_scale = min(1.0, epoch / warmup)

        cumulative_loss = 0.0
        cumulative_data = 0.0
        cumulative_initial = 0.0
        cumulative_boundary = 0.0
        cumulative_residual = 0.0
        step_count = 0
        epoch_balance = 1.0

        for features, targets in dataloader:
            features = features.to(device)
            targets = targets.to(device)

            predictions = model(features)
            data_loss = criterion(predictions, targets)

            total_loss = data_loss
            if config.mode == "pinn":
                components: PINNLossComponents = loss_fn(
                    model,
                    dataset,
                    device=device,
                    data_batch=(features, targets),
                    predictions=predictions,
                    config=config,
                )

                adaptive_scale = 1.0
                if getattr(config, "adaptive_residual_balance", False):
                    eps = 1e-12
                    ratio = (components.data.item() + eps) / (components.residual.item() + eps)
                    smoothing = getattr(config, "adaptive_balance_smoothing", 0.1)
                    if balance_state is None:
                        balance_state = ratio
                    else:
                        balance_state = (1.0 - smoothing) * balance_state + smoothing * ratio
                    adaptive_scale = balance_state ** 0.5
                    min_scale = getattr(config, "adaptive_balance_min", 0.1)
                    max_scale = getattr(config, "adaptive_balance_max", 10.0)
                    adaptive_scale = float(max(min(adaptive_scale, max_scale), min_scale))

                physics_effective = physics_scale * adaptive_scale
                epoch_balance = adaptive_scale

                total_loss = (
                    config.data_weight * components.data
                    + physics_effective
                    * (
                        config.initial_weight * components.initial
                        + config.boundary_weight * components.boundary
                        + config.residual_weight * components.residual
                    )
                )

                cumulative_data += components.data.item()
                cumulative_initial += physics_effective * components.initial.item()
                cumulative_boundary += physics_effective * components.boundary.item()
                cumulative_residual += physics_effective * components.residual.item()
            else:
                cumulative_data += data_loss.item()

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()

            if config.max_grad_norm and config.max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            optimizer.step()

            cumulative_loss += total_loss.item()
            step_count += 1

        average_loss = cumulative_loss / max(step_count, 1)
        history["loss"].append(average_loss)

        if scheduler is not None:
            scheduler.step(average_loss)

        message = f"Epoch {epoch:04d}/{config.num_epochs} | Loss: {average_loss:.6e}"
        elapsed = time.time() - start_time
        avg_epoch_time = elapsed / max(epoch, 1)
        remaining = max(config.num_epochs - epoch, 0)
        eta_seconds = avg_epoch_time * remaining
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

        if config.mode == "pinn":
            avg_data = cumulative_data / max(step_count, 1)
            avg_initial = cumulative_initial / max(step_count, 1)
            avg_boundary = cumulative_boundary / max(step_count, 1)
            avg_residual = cumulative_residual / max(step_count, 1)

            history["data_loss"].append(avg_data)
            history["initial_loss"].append(avg_initial)
            history["boundary_loss"].append(avg_boundary)
            history["residual_loss"].append(avg_residual)
            history["physics_scale"].append(physics_scale)
            history["adaptive_balance"].append(epoch_balance)

            message += (
                f" | data: {avg_data:.3e} ic: {avg_initial:.3e} "
                f"bc: {avg_boundary:.3e} res: {avg_residual:.3e} "
                f"w: {physics_scale:.2f} bal: {epoch_balance:.2f}"
            )
        else:
            history.setdefault("data_loss", []).append(cumulative_data / max(step_count, 1))

        message += f" | ETA: {eta_str}"

        if epoch % config.print_every == 0 or epoch == config.num_epochs:
            print(message, flush=True)
        else:
            print(message, end="\r", flush=True)

        # Save checkpoint for this epoch
        ckpt_path = run_dir / f"epoch_{epoch:04d}.pt"
        torch.save({"epoch": epoch, "model_state": model.state_dict(), "config": config}, ckpt_path)

        # Persist history each epoch (JSON overwrite) and append CSV row.
        try:
            import json

            hist_path = run_dir / "history.json"
            with hist_path.open("w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)

            csv_path = run_dir / "history.csv"
            write_header = not csv_path.exists() or csv_path.stat().st_size == 0
            with csv_path.open("a", encoding="utf-8") as f:
                if write_header:
                    f.write("epoch,loss,data,initial,boundary,residual,physics_scale,adaptive_balance\n")
                data_val = history.get("data_loss", [None])[-1]
                init_val = history.get("initial_loss", [None])[-1] if "initial_loss" in history else None
                bc_val = history.get("boundary_loss", [None])[-1] if "boundary_loss" in history else None
                res_val = history.get("residual_loss", [None])[-1] if "residual_loss" in history else None
                phys = history.get("physics_scale", [None])[-1] if "physics_scale" in history else None
                bal = history.get("adaptive_balance", [None])[-1] if "adaptive_balance" in history else None
                f.write(f"{epoch},{average_loss},{data_val},{init_val},{bc_val},{res_val},{phys},{bal}\n")
        except Exception:
            pass

    if config.save_path:
        checkpoint_path = Path(config.save_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": model.state_dict(), "config": config}, checkpoint_path)
        print(f"Model checkpoint saved to {checkpoint_path}")

    return history
