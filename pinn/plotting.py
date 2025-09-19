from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt


def plot_loss_curve(
    losses: Sequence[float],
    *,
    output_dir: str | Path = "figures",
    filename: str = "loss_curve.png",
    yscale: str = "log",
) -> Path | None:
    """Render the training loss curve and return the output path."""

    if not losses:
        return None

    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(losses) + 1), losses, label="Training loss")
    if yscale:
        ax.set_yscale(yscale)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("PINN training loss")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path
