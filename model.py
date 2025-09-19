from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import nn


class SimplePINN(nn.Module):
    """Baseline fully-connected network with tanh activations."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_layers: Sequence[int] = (128, 128, 128, 128),
        output_dim: int = 2,
        *,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        activation = activation or nn.Tanh()

        layers: list[nn.Module] = []
        prev_dim = input_dim
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
        weight_items = [
            (key, tensor)
            for key, tensor in state_dict.items()
            if key.endswith(".weight")
        ]
        if not weight_items:
            raise ValueError("State dict does not contain any linear weights.")
        weight_items.sort(key=lambda item: item[0])
        weights = [tensor for _, tensor in weight_items]
        input_dim = int(weights[0].shape[1])
        hidden_layers = [int(weight.shape[0]) for weight in weights[:-1]]
        output_dim = int(weights[-1].shape[0])
        model = cls(
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            output_dim=output_dim,
            activation=activation,
        )
        model.load_state_dict(state_dict)
        return model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.network(inputs)

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

