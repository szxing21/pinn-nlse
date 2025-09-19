from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from model import SimplePINN
from pinn import PulseEvolutionDataset, TrainingConfig, create_dataloader
from train import train


def main() -> None:
    data_path = Path("data/pulse_evolution.mat")
    dataset = PulseEvolutionDataset(data_path)
    target_dim = dataset[0][1].numel()

    config = TrainingConfig(num_epochs=default_epochs(), print_every=10)

    dataloader = create_dataloader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        shuffle=True,
    )

    model = SimplePINN(input_dim=2, hidden_layers=(128, 128, 128, 128), output_dim=target_dim)

    history = train(model, dataloader, config)

    losses = history["loss"]
    save_curve(losses)


def default_epochs() -> int:
    return TrainingConfig().num_epochs


def save_curve(losses: list[float]) -> None:
    output_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(losses) + 1)

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, losses, label="Training loss")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("PINN training loss")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    path = output_dir / "loss_curve.png"
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    print(f"Saved loss curve to {path}")


if __name__ == "__main__":
    main()
