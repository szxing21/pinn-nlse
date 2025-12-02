from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F


def _hardware_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    snr_db: float = 30.0,
    block: int = 4,
) -> torch.Tensor:
    """
    Placeholder for a hardware-backed 4x4 matmul pipeline.

    Performs software tiling into 4x4 blocks, simulating the hardware MVM with
    a standard matmul + optional noise. Replace the marked section with your
    actual hardware call (e.g., run_hw_mvm) to integrate the device.
    """

    B, in_dim = x.shape
    out_dim, _ = weight.shape

    pad_in = (block - in_dim % block) % block
    pad_out = (block - out_dim % block) % block

    x_pad = F.pad(x, (0, pad_in))
    W_pad = F.pad(weight, (0, pad_in, 0, pad_out))
    out_pad = out_dim + pad_out

    y_hw = x_pad.new_zeros(B, out_pad)

    for o in range(0, out_pad, block):
        acc = x_pad.new_zeros(B, block)
        for i in range(0, in_dim + pad_in, block):
            W_tile = W_pad[o:o + block, i:i + block]          # [4,4]
            X_tile = x_pad[:, i:i + block].T                  # [4,B]
            # TODO: replace this matmul with your hardware call:
            # Y_tile = run_hw_mvm(X_tile, W_tile, zeros(4,1))  # [4,B]
            Y_tile = W_tile @ X_tile                          # software simulation
            acc = acc + Y_tile.T                              # [B,4]
        y_hw[:, o:o + block] = acc

    y = y_hw[:, :out_dim]
    if bias is not None:
        y = y + bias

    # Additive noise calibrated by the measured output power.
    power = y.pow(2).mean()
    if torch.isfinite(power) and power > torch.finfo(y.dtype).eps:
        noise_power = power / (10.0 ** (snr_db / 10.0))
        noise_std = noise_power.sqrt()
        y = y + torch.randn_like(y) * noise_std
    return y


class ExternalLinear(nn.Module):
    """
    Linear layer whose forward values come from (simulated) hardware, while
    gradients are preserved via the software path.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, *, snr_db: float = 30.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.snr_db = snr_db
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -0.05, 0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Software path builds the autograd graph.
        y_model = F.linear(x, self.weight, self.bias)
        # Hardware (simulated) path provides the numerical output with noise.
        y_hw = _hardware_linear(x, self.weight, self.bias, snr_db=self.snr_db)
        # Preserve gradients from y_model, but use y_hw values.
        return y_hw.detach() + (y_model - y_model.detach())


class SimplePINN(nn.Module):
    """Baseline fully-connected network with optional Fourier features."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_layers: Sequence[int] = (128, 128, 128, 128),
        output_dim: int = 2,
        *,
        activation: nn.Module | None = None,
        fourier_features: int = 0,
        fourier_sigma: float = 10.0,
        generator: torch.Generator | None = None,
        external_layers: Sequence[bool] | None = None,
        external_snr_db: float = 30.0,
    ) -> None:
        super().__init__()
        activation = activation or nn.Tanh()

        if fourier_features < 0:
            raise ValueError("fourier_features must be non-negative.")
        if fourier_features > 0 and fourier_sigma <= 0:
            raise ValueError("fourier_sigma must be positive when using Fourier features.")

        self.input_dim = input_dim
        self.hidden_layers = list(hidden_layers)
        self.output_dim = output_dim
        self.activation_template = activation
        self.fourier_features = int(fourier_features)
        self.fourier_sigma = float(fourier_sigma)
        self.external_snr_db = float(external_snr_db)

        if self.fourier_features > 0:
            B = torch.randn(input_dim, self.fourier_features, generator=generator) * self.fourier_sigma
            self.register_buffer("fourier_B", B)
        else:
            self.register_buffer("fourier_B", None)

        effective_input_dim = input_dim + 2 * self.fourier_features
        num_linear_layers = len(hidden_layers) + 1  # hidden + output
        if external_layers is None:
            ext_flags = [True] + [False] * (num_linear_layers - 1)
        else:
            ext_flags = list(external_layers)
            if len(ext_flags) == 1:
                ext_flags = ext_flags * num_linear_layers
            elif len(ext_flags) != num_linear_layers:
                raise ValueError(
                    f"external_layers length must be 1 or {num_linear_layers} (hidden layers + output); got {len(ext_flags)}"
                )

        layers: list[nn.Module] = []
        prev_dim = effective_input_dim
        for idx, width in enumerate(hidden_layers):
            if ext_flags[idx]:
                layers.append(ExternalLinear(prev_dim, width, bias=True, snr_db=self.external_snr_db))
            else:
                layers.append(nn.Linear(prev_dim, width))
            layers.append(_clone_activation(activation))
            prev_dim = width
        # Output layer
        if ext_flags[-1]:
            layers.append(ExternalLinear(prev_dim, output_dim, bias=True, snr_db=self.external_snr_db))
        else:
            layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)
        self._initialise()

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, torch.Tensor],
        *,
        activation: nn.Module | None = None,
        external_layers: Sequence[bool] | None = None,
        external_snr_db: float = 30.0,
    ) -> "SimplePINN":
        activation = activation or nn.Tanh()

        fourier_B = state_dict.get("fourier_B")
        fourier_features = 0
        if isinstance(fourier_B, torch.Tensor) and fourier_B.ndim == 2 and fourier_B.numel() > 0:
            fourier_features = int(fourier_B.shape[1])
            input_dim = int(fourier_B.shape[0])
        else:
            input_dim = None

        weight_items = [
            (key, tensor)
            for key, tensor in state_dict.items()
            if key.endswith(".weight")
        ]
        if not weight_items:
            raise ValueError("State dict does not contain any linear weights.")
        weight_items.sort(key=lambda item: item[0])
        weights = [tensor for _, tensor in weight_items]

        first_weight = weights[0]
        if input_dim is None:
            input_dim = int(first_weight.shape[1] - 2 * fourier_features)
        if input_dim <= 0:
            raise ValueError("Inferred input dimension must be positive.")

        hidden_layers = [int(weight.shape[0]) for weight in weights[:-1]]
        output_dim = int(weights[-1].shape[0])

        model = cls(
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            output_dim=output_dim,
            activation=activation,
            fourier_features=fourier_features,
            fourier_sigma=1.0,
            external_layers=external_layers,
            external_snr_db=external_snr_db,
        )
        model.load_state_dict(state_dict)
        return model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self._apply_fourier_features(inputs)
        return self.network(features)

    def _apply_fourier_features(self, inputs: torch.Tensor) -> torch.Tensor:
        B = self.fourier_B
        if B is None or self.fourier_features == 0:
            return inputs
        projection = 2.0 * torch.pi * inputs @ B
        sin_feats = torch.sin(projection)
        cos_feats = torch.cos(projection)
        return torch.cat([inputs, sin_feats, cos_feats], dim=-1)

    def _initialise(self) -> None:
        for module in self.network:
            if isinstance(module, (nn.Linear, ExternalLinear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.uniform_(module.bias, -0.05, 0.05)


def _clone_activation(template: nn.Module) -> nn.Module:
    """Return a fresh activation instance matching the template type."""

    if isinstance(template, nn.Tanh):
        return nn.Tanh()
    if isinstance(template, nn.ReLU):
        return nn.ReLU()
    if isinstance(template, nn.Sigmoid):
        return nn.Sigmoid()
    try:
        return type(template)()  # type: ignore[call-arg]
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(
            "Activation template must be instantiable without arguments."  # noqa: TRY003
        ) from exc
