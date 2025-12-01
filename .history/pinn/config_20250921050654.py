from dataclasses import dataclass
from typing import Literal

DeviceChoice = Literal["cpu", "cuda", "auto"]
ModeChoice = Literal["mlp", "pinn"]
PDEVariant = Literal["standard", "ssfm"]


@dataclass
class TrainingConfig:
    """Hyperparameters controlling optimisation."""

    batch_size: int = 128
    learning_rate: float = 5e-3
    num_epochs: int = 200
    device: DeviceChoice = "auto"
    print_every: int = 20
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
    boundary_weight: float = 0
    residual_weight: float = 1.0
    residual_scale: float = 1.0e4
    ic_samples: int = 256
    bc_samples: int = 256
    residual_samples: int = 2048
    gradient_noise_snr_db: float = 80.0
    fourier_features: int = 1
    fourier_scale: float = 10.0
    pde_variant: PDEVariant = "ssfm"

    # PDE parameters expressed in ps/km units
    alpha: float = 0.0
    beta2: float = -21.242   # ps^2/km
    beta3: float = 0.166     # ps^3/km
    gamma: float = 1.3       # (W km)^-1
