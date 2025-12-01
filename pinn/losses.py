from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from .config import TrainingConfig
from .dataset import PulseEvolutionDataset, NormalisationStats
from .gradients import compute_output_gradients, GradientResult


@dataclass
class PINNLossComponents:
    data: torch.Tensor
    initial: torch.Tensor
    boundary: torch.Tensor
    residual: torch.Tensor
    gradients: Optional[GradientResult]


def _compute_pinn_components(
    model: torch.nn.Module,
    dataset: PulseEvolutionDataset,
    *,
    device: torch.device,
    config: TrainingConfig,
    data_batch: tuple[torch.Tensor, torch.Tensor],
    predictions: torch.Tensor,
    residual_fn,
) -> PINNLossComponents:
    features, targets = data_batch
    data_loss = F.mse_loss(predictions, targets)

    initial_inputs, initial_targets = dataset.sample_initial(config.ic_samples)
    boundary_inputs, boundary_targets = dataset.sample_boundary(config.bc_samples)
    interior_inputs, _ = dataset.sample_interior(config.residual_samples)

    initial_loss = _condition_loss(model, initial_inputs, initial_targets, device)
    boundary_loss = _condition_loss(model, boundary_inputs, boundary_targets, device)

    if interior_inputs.numel() == 0:
        residual_loss = torch.zeros((), device=device, dtype=predictions.dtype)
        gradient_result = None
    else:
        residual_loss, gradient_result = residual_fn(
            model,
            dataset.stats,
            interior_inputs.to(device),
            device=device,
            config=config,
        )

    return PINNLossComponents(
        data=data_loss,
        initial=initial_loss,
        boundary=boundary_loss,
        residual=residual_loss,
        gradients=gradient_result,
    )


def compute_pinn_loss_components(
    model: torch.nn.Module,
    dataset: PulseEvolutionDataset,
    *,
    device: torch.device,
    config: TrainingConfig,
    data_batch: tuple[torch.Tensor, torch.Tensor],
    predictions: torch.Tensor,
) -> PINNLossComponents:
    return _compute_pinn_components(
        model,
        dataset,
        device=device,
        config=config,
        data_batch=data_batch,
        predictions=predictions,
        residual_fn=_residual_loss_standard,
    )


def _condition_loss(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if inputs.numel() == 0:
        dtype = targets.dtype if targets.numel() > 0 else torch.float32
        return torch.zeros((), device=device, dtype=dtype)
    inputs = inputs.to(device)
    targets = targets.to(device)
    predictions = model(inputs)
    return F.mse_loss(predictions, targets)




def compute_pinn_loss_components_ssfm(
    model: torch.nn.Module,
    dataset: PulseEvolutionDataset,
    *,
    device: torch.device,
    config: TrainingConfig,
    data_batch: tuple[torch.Tensor, torch.Tensor],
    predictions: torch.Tensor,
) -> PINNLossComponents:
    return _compute_pinn_components(
        model,
        dataset,
        device=device,
        config=config,
        data_batch=data_batch,
        predictions=predictions,
        residual_fn=_residual_loss_ssfm,
    )


def _residual_loss_standard(
    model: torch.nn.Module,
    stats: NormalisationStats,
    inputs: torch.Tensor,
    *,
    device: torch.device,
    config: TrainingConfig,
) -> tuple[torch.Tensor, GradientResult]:
    """Compute PDE residual loss based on the generalized NLSE."""

    # Reparameterise inputs so autograd tracks physical coordinates
    input_min = torch.as_tensor(stats.input_min, device=device, dtype=inputs.dtype)
    input_scale = torch.as_tensor(stats.input_scale, device=device, dtype=inputs.dtype)

    inputs_norm = inputs.clone().detach()
    inputs_phys = ((inputs_norm + 1.0) * 0.5) * input_scale + input_min
    inputs_phys.requires_grad_(True)

    inputs_norm = 2.0 * (inputs_phys - input_min) / input_scale - 1.0
    predictions_norm = model(inputs_norm)

    target_mean = torch.as_tensor(stats.target_mean, device=device, dtype=predictions_norm.dtype)
    target_std = torch.as_tensor(stats.target_std, device=device, dtype=predictions_norm.dtype)
    amplitudes = predictions_norm * target_std + target_mean

    a_real = amplitudes[:, 0:1]
    a_imag = amplitudes[:, 1:2]

    # Gradients with respect to physical coordinates (z, t)
    grad_real = torch.autograd.grad(
        a_real,
        inputs_phys,
        grad_outputs=torch.ones_like(a_real),
        retain_graph=True,
        create_graph=True,
    )[0]
    grad_imag = torch.autograd.grad(
        a_imag,
        inputs_phys,
        grad_outputs=torch.ones_like(a_imag),
        retain_graph=True,
        create_graph=True,
    )[0]

    per_km = 1.0e3
    dA_r_dz = grad_real[:, 0:1] * per_km
    dA_r_dt = grad_real[:, 1:2]
    dA_i_dz = grad_imag[:, 0:1] * per_km
    dA_i_dt = grad_imag[:, 1:2]

    # Second derivatives with respect to physical time
    d2A_r_dt2 = torch.autograd.grad(
        dA_r_dt,
        inputs_phys,
        grad_outputs=torch.ones_like(dA_r_dt),
        retain_graph=True,
        create_graph=True,
    )[0][:, 1:2]
    d2A_i_dt2 = torch.autograd.grad(
        dA_i_dt,
        inputs_phys,
        grad_outputs=torch.ones_like(dA_i_dt),
        retain_graph=True,
        create_graph=True,
    )[0][:, 1:2]

    if config.beta3 != 0.0:
        d3A_r_dt3 = torch.autograd.grad(
            d2A_r_dt2,
            inputs_phys,
            grad_outputs=torch.ones_like(d2A_r_dt2),
            retain_graph=True,
            create_graph=False,
        )[0][:, 1:2]
        d3A_i_dt3 = torch.autograd.grad(
            d2A_i_dt2,
            inputs_phys,
            grad_outputs=torch.ones_like(d2A_i_dt2),
            retain_graph=True,
            create_graph=False,
        )[0][:, 1:2]
    else:
        d3A_r_dt3 = torch.zeros_like(d2A_r_dt2)
        d3A_i_dt3 = torch.zeros_like(d2A_i_dt2)

    # Residual assembly for the generalized NLSE
    power = a_real.pow(2) + a_imag.pow(2)

    residual_real = (
        dA_r_dz
        + 0.5 * config.alpha * a_real
        + 0.5 * config.beta2 * d2A_i_dt2
        - (config.beta3 / 6.0) * d3A_r_dt3
        + config.gamma * power * a_imag
    )

    residual_imag = (
        dA_i_dz
        + 0.5 * config.alpha * a_imag
        - 0.5 * config.beta2 * d2A_r_dt2
        - (config.beta3 / 6.0) * d3A_i_dt3
        - config.gamma * power * a_real
    )

    residual_loss = torch.mean(residual_real.pow(2) + residual_imag.pow(2)) * config.residual_scale

    gradient_result = compute_output_gradients(
        model,
        inputs,
        create_graph=False,
        retain_graph=False,
        noise_snr_db=config.gradient_noise_snr_db,
    )

    return residual_loss, gradient_result




def _residual_loss_ssfm(
    model: torch.nn.Module,
    stats: NormalisationStats,
    inputs: torch.Tensor,
    *,
    device: torch.device,
    config: TrainingConfig,
) -> tuple[torch.Tensor, GradientResult]:
    input_min = torch.as_tensor(stats.input_min, device=device, dtype=inputs.dtype)
    input_scale = torch.as_tensor(stats.input_scale, device=device, dtype=inputs.dtype)

    inputs_norm = inputs.clone().detach()
    inputs_phys = ((inputs_norm + 1.0) * 0.5) * input_scale + input_min
    inputs_phys.requires_grad_(True)

    inputs_norm = 2.0 * (inputs_phys - input_min) / input_scale - 1.0
    predictions_norm = model(inputs_norm)

    target_mean = torch.as_tensor(stats.target_mean, device=device, dtype=predictions_norm.dtype)
    target_std = torch.as_tensor(stats.target_std, device=device, dtype=predictions_norm.dtype)
    amplitudes = predictions_norm * target_std + target_mean

    a_real = amplitudes[:, 0:1]
    a_imag = amplitudes[:, 1:2]

    grad_real = torch.autograd.grad(
        a_real,
        inputs_phys,
        grad_outputs=torch.ones_like(a_real),
        retain_graph=True,
        create_graph=True,
    )[0]
    grad_imag = torch.autograd.grad(
        a_imag,
        inputs_phys,
        grad_outputs=torch.ones_like(a_imag),
        retain_graph=True,
        create_graph=True,
    )[0]

    per_km = 1.0e3
    dA_r_dz = grad_real[:, 0:1] * per_km
    dA_r_dt = grad_real[:, 1:2]
    dA_i_dz = grad_imag[:, 0:1] * per_km
    dA_i_dt = grad_imag[:, 1:2]

    d2A_r_dt2 = torch.autograd.grad(
        dA_r_dt,
        inputs_phys,
        grad_outputs=torch.ones_like(dA_r_dt),
        retain_graph=True,
        create_graph=True,
    )[0][:, 1:2]
    d2A_i_dt2 = torch.autograd.grad(
        dA_i_dt,
        inputs_phys,
        grad_outputs=torch.ones_like(dA_i_dt),
        retain_graph=True,
        create_graph=True,
    )[0][:, 1:2]

    if config.beta3 != 0.0:
        d3A_r_dt3 = torch.autograd.grad(
            d2A_r_dt2,
            inputs_phys,
            grad_outputs=torch.ones_like(d2A_r_dt2),
            retain_graph=True,
            create_graph=False,
        )[0][:, 1:2]
        d3A_i_dt3 = torch.autograd.grad(
            d2A_i_dt2,
            inputs_phys,
            grad_outputs=torch.ones_like(d2A_i_dt2),
            retain_graph=True,
            create_graph=False,
        )[0][:, 1:2]
    else:
        d3A_r_dt3 = torch.zeros_like(d2A_r_dt2)
        d3A_i_dt3 = torch.zeros_like(d2A_i_dt2)

    power = a_real.pow(2) + a_imag.pow(2)

    residual_real = (
        dA_r_dz
        + 0.5 * config.beta2 * d2A_i_dt2
        - (config.beta3 / 6.0) * d3A_r_dt3
        + config.gamma * power * a_imag
    )
    residual_imag = (
        dA_i_dz
        - 0.5 * config.beta2 * d2A_r_dt2
        - (config.beta3 / 6.0) * d3A_i_dt3
        - config.gamma * power * a_real
    )

    residual_loss = torch.mean(residual_real.pow(2) + residual_imag.pow(2)) * config.residual_scale

    gradient_result = compute_output_gradients(
        model,
        inputs,
        create_graph=False,
        retain_graph=False,
        noise_snr_db=config.gradient_noise_snr_db,
    )

    return residual_loss, gradient_result
