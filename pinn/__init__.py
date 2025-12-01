"""Utilities for working with the PINN baseline project."""

from .config import TrainingConfig
from .dataset import (
    NormalisationStats,
    PulseEvolutionDataset,
    create_dataloader,
    load_pulse_evolution,
)
from .gradients import GradientResult, compute_output_gradients
from .losses import PINNLossComponents, compute_pinn_loss_components, compute_pinn_loss_components_ssfm
from .physics import residual_statistics
from .plotting import plot_loss_curve

__all__ = [
    "TrainingConfig",
    "PulseEvolutionDataset",
    "create_dataloader",
    "load_pulse_evolution",
    "NormalisationStats",
    "plot_loss_curve",
    "compute_output_gradients",
    "GradientResult",
    "PINNLossComponents",
    "compute_pinn_loss_components",
    "compute_pinn_loss_components_ssfm",
    "residual_statistics",
]
