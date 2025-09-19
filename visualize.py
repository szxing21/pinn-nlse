from __future__ import annotations

import os

import argparse
from pathlib import Path
from typing import Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from model import SimplePINN
from pinn import NormalisationStats, load_pulse_evolution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualise teacher vs PINN predictions.")
    parser.add_argument("--data-path", default="data/pulse_evolution.mat", help="Path to the MATLAB dataset.")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/pinn_mlp.pt",
        help="Model checkpoint produced by train.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="figures",
        help="Directory for saving generated plots.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to run inference on.",
    )
    return parser.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def build_model(checkpoint_path: Path, device: torch.device) -> SimplePINN:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state"]
    model = SimplePINN.from_state_dict(state_dict, activation=nn.Tanh())
    return model.to(device).eval()


def reshape_to_grid(values: np.ndarray, grid_shape: Tuple[int, int]) -> np.ndarray:
    return values.reshape(*grid_shape, -1)


def to_magnitude(values: np.ndarray) -> np.ndarray:
    if values.shape[-1] == 1:
        return values[..., 0]
    if values.shape[-1] == 2:
        real = values[..., 0]
        imag = values[..., 1]
        return np.sqrt(real**2 + imag**2)
    return np.linalg.norm(values, axis=-1)


def denormalise_targets(stats: NormalisationStats, values: np.ndarray) -> np.ndarray:
    return stats.denormalise_targets(values)


def generate_visualizations(
    *,
    data_path: str | Path = "data/pulse_evolution.mat",
    checkpoint_path: str | Path = "checkpoints/pinn_mlp.pt",
    output_dir: str | Path = "figures",
    device: str = "auto",
) -> tuple[Path, Path]:
    """Create 3D surface and heatmap comparisons from the latest checkpoint."""

    device_obj = resolve_device(device)
    data_path = Path(data_path)
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = load_pulse_evolution(data_path, normalise=True)
    stats = payload["stats"]
    assert isinstance(stats, NormalisationStats)

    inputs = torch.from_numpy(payload["inputs"]).to(device_obj)
    targets_norm = payload["targets"]
    z = payload["z"]
    t = payload["t"]
    grid_shape = (z.size, t.size)

    model = build_model(checkpoint_path, device_obj)

    with torch.no_grad():
        predictions_norm = model(inputs).cpu().numpy()

    targets_grid_norm = reshape_to_grid(targets_norm, grid_shape)
    predictions_grid_norm = reshape_to_grid(predictions_norm, grid_shape)

    teacher = denormalise_targets(stats, targets_grid_norm)
    predicted = denormalise_targets(stats, predictions_grid_norm)
    error = predicted - teacher

    teacher_mag = to_magnitude(teacher)
    pred_mag = to_magnitude(predicted)
    error_mag = to_magnitude(error)

    T, Z = np.meshgrid(t, z, indexing="ij")

    fig3d = plt.figure(figsize=(14, 6))
    ax1 = fig3d.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(T, Z, teacher_mag.T, cmap="viridis")
    ax1.set_title("Teacher Magnitude")
    ax1.set_xlabel("Time (ps)")
    ax1.set_ylabel("Propagation (z)")
    ax1.set_zlabel("|A|")

    ax2 = fig3d.add_subplot(1, 2, 2, projection="3d")
    ax2.plot_surface(T, Z, pred_mag.T, cmap="plasma")
    ax2.set_title("Model Prediction Magnitude")
    ax2.set_xlabel("Time (ps)")
    ax2.set_ylabel("Propagation (z)")
    ax2.set_zlabel("|A|")

    fig3d.tight_layout()
    surface_path = output_dir / "surface_comparison.png"
    fig3d.savefig(surface_path, dpi=300, bbox_inches="tight")
    plt.close(fig3d)

    fig2d, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)

    im0 = axes[0].imshow(
        teacher_mag,
        aspect="auto",
        origin="lower",
        extent=(t.min(), t.max(), z.min(), z.max()),
        cmap="viridis",
    )
    axes[0].set_title("Teacher |A|")

    im1 = axes[1].imshow(
        pred_mag,
        aspect="auto",
        origin="lower",
        extent=(t.min(), t.max(), z.min(), z.max()),
        cmap="plasma",
    )
    axes[1].set_title("Prediction |A|")

    im2 = axes[2].imshow(
        error_mag,
        aspect="auto",
        origin="lower",
        extent=(t.min(), t.max(), z.min(), z.max()),
        cmap="coolwarm",
    )
    axes[2].set_title("|Prediction - Teacher|")

    for ax in axes:
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel("Propagation (z)")

    fig2d.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig2d.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig2d.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    fig2d.tight_layout()

    heatmap_path = output_dir / "heatmap_comparison.png"
    fig2d.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close(fig2d)

    print(f"Saved 3D surface comparison to {surface_path}")
    print(f"Saved heatmap comparison to {heatmap_path}")
    return surface_path, heatmap_path


def main() -> None:
    args = parse_args()
    generate_visualizations(
        data_path=args.data_path,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()