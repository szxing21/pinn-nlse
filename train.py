from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from pinn import TrainingConfig
from pinn.losses import PINNLossComponents, compute_pinn_loss_components, compute_pinn_loss_components_ssfm


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

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = _configure_scheduler(optimizer, config)

    history: Dict[str, List[float]] = {"loss": []}
    if config.mode == "pinn":
        if dataset is None:
            raise ValueError("PINN mode requires access to the full dataset for condition sampling.")

        if config.pde_variant == "ssfm":
            loss_fn = compute_pinn_loss_components_ssfm
        else:
            loss_fn = compute_pinn_loss_components

        history.update(
            {
                "data_loss": [],
                "initial_loss": [],
                "boundary_loss": [],
                "residual_loss": [],
            }
        )

    for epoch in range(1, config.num_epochs + 1):
        cumulative_loss = 0.0
        cumulative_data = 0.0
        cumulative_initial = 0.0
        cumulative_boundary = 0.0
        cumulative_residual = 0.0
        step_count = 0

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
                total_loss = (
                    config.data_weight * components.data
                    + config.initial_weight * components.initial
                    + config.boundary_weight * components.boundary
                    + config.residual_weight * components.residual
                )

                cumulative_data += components.data.item()
                cumulative_initial += components.initial.item()
                cumulative_boundary += components.boundary.item()
                cumulative_residual += components.residual.item()
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

        if config.mode == "pinn":
            avg_data = cumulative_data / max(step_count, 1)
            avg_initial = cumulative_initial / max(step_count, 1)
            avg_boundary = cumulative_boundary / max(step_count, 1)
            avg_residual = cumulative_residual / max(step_count, 1)

            history["data_loss"].append(avg_data)
            history["initial_loss"].append(avg_initial)
            history["boundary_loss"].append(avg_boundary)
            history["residual_loss"].append(avg_residual)

            message += (
                f" | data: {avg_data:.3e} ic: {avg_initial:.3e} "
                f"bc: {avg_boundary:.3e} res: {avg_residual:.3e}"
            )
        else:
            history.setdefault("data_loss", []).append(cumulative_data / max(step_count, 1))

        if epoch % config.print_every == 0 or epoch == config.num_epochs:
            print(message, flush=True)
        else:
            print(message, end="\r", flush=True)

    if config.save_path:
        checkpoint_path = Path(config.save_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": model.state_dict(), "config": config}, checkpoint_path)
        print(f"Model checkpoint saved to {checkpoint_path}")

    return history
