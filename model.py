from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import nn


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

        if self.fourier_features > 0:
            B = torch.randn(input_dim, self.fourier_features, generator=generator) * self.fourier_sigma
            self.register_buffer("fourier_B", B)
        else:
            self.register_buffer("fourier_B", None)

        effective_input_dim = input_dim + 2 * self.fourier_features

        layers: list[nn.Module] = []
        prev_dim = effective_input_dim
        for width in hidden_layers:
            layers.append(nn.Linear(prev_dim, width))
            layers.append(_clone_activation(activation))
            prev_dim = width
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)
        self._initialise()

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, torch.Tensor],
        *,
        activation: nn.Module | None = None,
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
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
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
