from __future__ import annotations

import os
from pathlib import Path

import argparse
from typing import Sequence

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")

import torch

from model import SimplePINN
from pinn import (
    TrainingConfig,
    PulseEvolutionDataset,
    create_dataloader,
    plot_loss_curve,
)
from train import train
from visualize import generate_visualizations

DEFAULT_FIG_DIR = "figures"


def parse_hidden_layers(spec: str) -> Sequence[int]:
    layers = [int(width.strip()) for width in spec.split(",") if width.strip()]
    if not layers:
        raise argparse.ArgumentTypeError("Hidden layer specification must contain at least one width.")
    return layers


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a baseline MLP PINN on the pulse evolution dataset.")
    default_config = TrainingConfig()
    parser.add_argument("--data-path", default="data/pulse_evolution.mat", help="Path to the MATLAB tensor.")
    parser.add_argument("--epochs", type=int, default=default_config.num_epochs, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=default_config.batch_size, help="Mini-batch size for the optimiser.")
    parser.add_argument("--learning-rate", type=float, default=default_config.learning_rate, help="Adam learning rate.")
    parser.add_argument(
        "--hidden",
        type=parse_hidden_layers,
        default=parse_hidden_layers("32,32,32,32"),
        help="Comma-separated hidden layer widths.",
    )
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default=default_config.device, help="Computation device preference.")
    parser.add_argument("--print-every", type=int, default=default_config.print_every, help="Epoch interval for logging losses.")
    parser.add_argument("--save-path", default=default_config.save_path, help="File to store the trained model state.")
    parser.add_argument("--num-workers", type=int, default=default_config.num_workers, help="Number of DataLoader workers.")
    parser.add_argument("--pin-memory", action="store_true", default=default_config.pin_memory, help="Enable pinned memory in the DataLoader.")
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=default_config.max_grad_norm,
        help="Gradient clipping threshold; disable with non-positive values.",
    )
    parser.add_argument(
        "--scheduler-patience",
        type=int,
        default=default_config.scheduler_patience,
        help="Epoch patience before reducing LR on plateau (non-positive disables).",
    )
    parser.add_argument(
        "--scheduler-factor",
        type=float,
        default=default_config.scheduler_factor,
        help="Multiplicative factor when reducing LR on plateau.",
    )
    parser.add_argument(
        "--mode",
        choices=["mlp", "pinn"],
        default=default_config.mode,
        help="Training mode: plain supervised MLP or physics-informed (PINN).",
    )
    parser.add_argument(
        "--pde-variant",
        choices=["standard", "ssfm"],
        default=default_config.pde_variant,
        help="PDE residual variant to use (standard NLSE or SSFM-based).",
    )
    parser.add_argument(
        "--data-weight",
        type=float,
        default=default_config.data_weight,
        help="Weight applied to supervised data loss when in PINN mode.",
    )
    parser.add_argument(
        "--initial-weight",
        type=float,
        default=default_config.initial_weight,
        help="Weight applied to initial-condition loss in PINN mode.",
    )
    parser.add_argument(
        "--boundary-weight",
        type=float,
        default=default_config.boundary_weight,
        help="Weight applied to boundary-condition loss in PINN mode.",
    )
    parser.add_argument(
        "--residual-weight",
        type=float,
        default=default_config.residual_weight,
        help="Weight applied to PDE residual loss in PINN mode.",
    )
    parser.add_argument(
        "--residual-scale",
        type=float,
        default=default_config.residual_scale,
        help="Optional multiplier applied to residual loss before weighting.",
    )
    parser.add_argument(
        "--ic-samples",
        type=int,
        default=default_config.ic_samples,
        help="Number of samples drawn for the initial-condition term each step.",
    )
    parser.add_argument(
        "--bc-samples",
        type=int,
        default=default_config.bc_samples,
        help="Number of samples drawn for the boundary-condition term each step.",
    )
    parser.add_argument(
        "--residual-samples",
        type=int,
        default=default_config.residual_samples,
        help="Number of samples drawn for the residual-term placeholder each step.",
    )
    parser.add_argument(
        "--gradient-noise-snr-db",
        type=float,
        default=default_config.gradient_noise_snr_db,
        help="Optional SNR (dB) for gradient noise injection when evaluating residuals.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=default_config.alpha,
        help="Linear loss coefficient (alpha/2 term) in the fiber NLSE.",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=default_config.beta2,
        help="Second-order dispersion coefficient beta2 (ps^2/mm).",
    )
    parser.add_argument(
        "--beta3",
        type=float,
        default=default_config.beta3,
        help="Third-order dispersion coefficient beta3 (ps^3/mm).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=default_config.gamma,
        help="Nonlinear Kerr coefficient gamma ((W mm)^-1).",
    )
    parser.add_argument(
        "--fig-output-dir",
        default=DEFAULT_FIG_DIR,
        help=(
            "Directory where diagnostic plots will be written. "
            "When left as the default, results go to figures/<mode>."
        ),
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()

    dataset = PulseEvolutionDataset(args.data_path)
    target_dim = dataset[0][1].numel()

    config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        device=args.device,
        print_every=args.print_every,
        save_path=args.save_path,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        max_grad_norm=args.max_grad_norm,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        mode=args.mode,
        data_weight=args.data_weight,
        initial_weight=args.initial_weight,
        boundary_weight=args.boundary_weight,
        residual_weight=args.residual_weight,
        residual_scale=args.residual_scale,
        ic_samples=args.ic_samples,
        bc_samples=args.bc_samples,
        residual_samples=args.residual_samples,
        gradient_noise_snr_db=args.gradient_noise_snr_db,
        pde_variant=args.pde_variant,
        alpha=args.alpha,
        beta2=args.beta2,
        beta3=args.beta3,
        gamma=args.gamma,
    )

    dataloader = create_dataloader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    model = SimplePINN(input_dim=2, hidden_layers=args.hidden, output_dim=target_dim, activation=torch.nn.Tanh())

    history = train(model, dataloader, config, dataset=dataset)

    fig_output_dir = Path(args.fig_output_dir)
    if args.fig_output_dir == DEFAULT_FIG_DIR:
        fig_output_dir = fig_output_dir / args.mode
    fig_output_dir.mkdir(parents=True, exist_ok=True)

    loss_path = plot_loss_curve(history.get("loss", []), output_dir=fig_output_dir)
    if loss_path is not None:
        print(f"Saved loss curve to {loss_path}")

    generate_visualizations(
        data_path=args.data_path,
        checkpoint_path=args.save_path,
        output_dir=fig_output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
