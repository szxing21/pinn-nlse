from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DeviceChoice = Literal["cpu", "cuda", "auto"]
ModeChoice = Literal["mlp", "pinn"]
PDEVariant = Literal["standard", "ssfm"]


@dataclass
class TrainingConfig:
    """Hyperparameters controlling optimisation."""

    # Network architecture
    hidden_layers: tuple[int, ...] = (128, 128, 128, 128)
    external_layers: tuple[bool, ...] = (1,1,0,0,0)  # broadcast or per-linear-layer flags (hidden + output)
    external_snr_db: float = 30.0

    batch_size: int = 256
    learning_rate: float = 3e-3
    num_epochs: int = 200
    device: DeviceChoice = "auto"
    print_every: int = 5
    save_path: str = "checkpoints/pinn_mlp.pt"
    num_workers: int = 0
    pin_memory: bool = False
    max_grad_norm: float = 1.0
    scheduler_patience: int = 15
    scheduler_factor: float = 0.5
    mode: ModeChoice = "pinn"

    # Loss weights and sampling for PINN mode
    data_weight: float = 1.0
    initial_weight: float = 1.0
    boundary_weight: float = 0.0
    residual_weight: float = 0.5
    residual_warmup_epochs: int = 50
    residual_scale: float = 0.0  # <=0 enables automatic scaling from teacher residual
    ic_samples: int = 256
    bc_samples: int = 256
    residual_samples: int = 2048
    gradient_noise_snr_db: float = 80.0

    # Fourier feature encoding
    fourier_features: int = 32
    fourier_scale: float = 8.0

    # Optional supervised pre-training before PINN fine-tuning
    pretrain_epochs: int = 0
    pretrain_learning_rate: float = 3e-3

    # Adaptive balancing of physics/data losses
    adaptive_residual_balance: bool = False
    adaptive_balance_smoothing: float = 0.1
    adaptive_balance_min: float = 0.1
    adaptive_balance_max: float = 10.0

    pde_variant: PDEVariant = "ssfm"

    # PDE parameters expressed in ps/km units
    alpha: float = 0.0
    beta2: float = -21.242   # ps^2/km
    beta3: float = 0.166     # ps^3/km
    gamma: float = 1.3       # (W km)^-1
