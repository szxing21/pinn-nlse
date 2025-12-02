from __future__ import annotations

import argparse
from dataclasses import replace
from typing import List, Sequence, Tuple

import optuna

import torch
from torch.utils.data import DataLoader, Subset

from model import SimplePINN
from pinn import PulseEvolutionDataset, TrainingConfig, create_dataloader
from train import train


def split_dataset(
    dataset: PulseEvolutionDataset,
    val_fraction: float,
    seed: int,
) -> tuple[Subset, Subset]:
    """Randomly split dataset into train/val subsets."""

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)
    val_size = max(1, int(len(indices) * val_fraction))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def evaluate_mse(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    """Compute mean MSE over a dataloader."""

    model.eval()
    criterion = torch.nn.MSELoss(reduction="mean")
    total = 0.0
    count = 0
    with torch.no_grad():
        for features, targets in dataloader:
            features = features.to(device)
            targets = targets.to(device)
            preds = model(features)
            total += criterion(preds, targets).item()
            count += 1
    return total / max(count, 1)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lightweight hyperparameter search over network depth/width.")
    parser.add_argument("--data-path", default="data/pulse_evolution.mat", help="Path to the MATLAB tensor.")
    parser.add_argument("--trials", type=int, default=10, help="Number of Optuna trials.")
    parser.add_argument("--epochs", type=int, default=30, help="Epochs per trial (keep small for quick sweeps / pruning).")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Validation split fraction.")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto", help="Computation device.")
    parser.add_argument("--mode", choices=["mlp", "pinn"], default="pinn", help="Training mode for trials.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--min-layers", type=int, default=2, help="Minimum hidden layers.")
    parser.add_argument("--max-layers", type=int, default=4, help="Maximum hidden layers.")
    parser.add_argument("--widths", default="64,128,256", help="Comma-separated choices for hidden widths.")
    parser.add_argument("--fourier", default="0,16,32", help="Comma-separated choices for Fourier features.")
    parser.add_argument("--external", default="0", help="Comma-separated choices for using external layers (broadcast). Default 0 disables external layers during tuning.")
    return parser


def parse_int_list(text: str) -> List[int]:
    return [int(tok) for tok in text.split(",") if tok.strip()]


def parse_bool_list(text: str) -> List[bool]:
    out: List[bool] = []
    for tok in text.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        if tok in ("1", "true", "t", "yes", "y"):
            out.append(True)
        elif tok in ("0", "false", "f", "no", "n"):
            out.append(False)
        else:
            raise argparse.ArgumentTypeError(f"Invalid boolean flag: {tok}")
    return out


def main() -> None:
    args = build_argparser().parse_args()

    dataset = PulseEvolutionDataset(args.data_path)
    train_ds, val_ds = split_dataset(dataset, val_fraction=args.val_fraction, seed=args.seed)

    width_choices = parse_int_list(args.widths)
    fourier_choices = parse_int_list(args.fourier)
    external_choices = parse_bool_list(args.external)

    default_cfg = TrainingConfig()
    device_choice = args.device

    def objective(trial: optuna.Trial) -> float:
        num_layers = trial.suggest_int("layers", args.min_layers, args.max_layers)
        hidden = tuple(trial.suggest_categorical(f"w{i}", width_choices) for i in range(num_layers))
        ff = trial.suggest_categorical("fourier_features", fourier_choices)
        ext_flag = trial.suggest_categorical("external", external_choices)
        ext = (bool(ext_flag),)  # broadcast

        cfg = replace(
            default_cfg,
            num_epochs=args.epochs,
            print_every=max(1, args.epochs // 5),
            device=device_choice,
            mode=args.mode,
            hidden_layers=hidden,
            fourier_features=ff,
            external_layers=ext,
        )

        model = SimplePINN(
            input_dim=2,
            hidden_layers=hidden,
            output_dim=dataset[0][1].numel(),
            activation=torch.nn.Tanh(),
            fourier_features=ff,
            fourier_sigma=cfg.fourier_scale,
            external_layers=ext,
            external_snr_db=cfg.external_snr_db,
        )

        train_loader = create_dataloader(
            train_ds,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )
        val_loader = create_dataloader(
            val_ds,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            shuffle=False,
        )

        print(f"[Trial {trial.number+1}/{args.trials}] hidden={hidden}, fourier={ff}, external={ext}")
        train(model, train_loader, cfg, dataset=dataset)

        device = torch.device("cuda" if device_choice == "cuda" and torch.cuda.is_available() else "cpu") if device_choice != "cpu" else torch.device("cpu")
        val_mse = evaluate_mse(model.to(device), val_loader, device)
        trial.report(val_mse, step=cfg.num_epochs)
        return val_mse

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=5, reduction_factor=2, min_early_stopping_rate=0)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=args.trials)

    best = study.best_trial
    params = best.params
    layers = params.get("layers", 0)
    hidden_best = [params[f"w{i}"] for i in range(layers)] if layers else []
    fourier_best = params.get("fourier_features", None)
    external_best = params.get("external", None)
    print("\nBest configuration:")
    print(f"  trial      : {best.number}")
    print(f"  hidden     : {hidden_best}")
    print(f"  fourier    : {fourier_best}")
    print(f"  external   : {external_best}")
    print(f"  val MSE    : {best.value:.4e}")


if __name__ == "__main__":
    main()
