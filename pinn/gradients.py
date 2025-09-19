from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class GradientResult:
    gradients: torch.Tensor
    noisy: Optional[torch.Tensor] = None


def compute_output_gradients(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    *,
    create_graph: bool = False,
    retain_graph: bool = True,
    noise_snr_db: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
) -> GradientResult:
    """Compute \nabla_{inputs} model(inputs) with optional SNR-controlled noise."""

    inputs = inputs.clone().detach().requires_grad_(True)
    outputs = model(inputs)

    if outputs.ndim != 2:
        raise ValueError("Model outputs must be of shape [batch, features].")

    grads = []
    batch = outputs.size(0)
    feature_dim = outputs.size(1)

    for feature_idx in range(feature_dim):
        grad_outputs = torch.zeros_like(outputs)
        grad_outputs[:, feature_idx] = 1.0
        grad = torch.autograd.grad(
            outputs,
            inputs,
            grad_outputs=grad_outputs,
            retain_graph=retain_graph or feature_idx < feature_dim - 1,
            create_graph=create_graph,
        )[0]
        grads.append(grad)

    gradients = torch.stack(grads, dim=1)  # [batch, output_dim, input_dim]

    noisy = None
    if noise_snr_db is not None:
        signal_power = gradients.pow(2).mean()
        eps = torch.finfo(gradients.dtype).eps
        if torch.isnan(signal_power) or torch.isinf(signal_power) or signal_power <= eps:
            noise_std = torch.tensor(0.0, device=gradients.device, dtype=gradients.dtype)
        else:
            noise_power = signal_power / (10.0 ** (noise_snr_db / 10.0))
            noise_std = torch.sqrt(noise_power)
        noise = torch.randn(
            gradients.shape,
            device=gradients.device,
            dtype=gradients.dtype,
            generator=generator,
        ) * noise_std
        noisy = gradients + noise

    return GradientResult(gradients=gradients, noisy=noisy)
