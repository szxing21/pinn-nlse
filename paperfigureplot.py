from __future__ import annotations

import os
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from model import SimplePINN
from pinn import PulseEvolutionDataset
from visualize import generate_visualizations


def load_history(log_dir: Path, fallback_dir: Path | None = None) -> Dict[str, List[float]]:
    """Load history; prefer JSON (has data_loss), else CSV. Optionally fallback."""
    hist: Dict[str, List[float]] = {}

    json_path = log_dir / "history.json"
    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        for k, v in loaded.items():
            if isinstance(v, list):
                hist[k] = v
        if hist:
            return hist

    csv_path = log_dir / "history.csv"
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for k, v in row.items():
                    if k not in hist:
                        hist[k] = []
                    try:
                        hist[k].append(float(v))
                    except (ValueError, TypeError):
                        hist[k].append(np.nan)
        # If CSV lacked data_loss but JSON exists, try merging
        if "data_loss" not in hist and json_path.exists():
            with json_path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if "data_loss" in loaded and isinstance(loaded["data_loss"], list):
                hist["data_loss"] = loaded["data_loss"]
        return hist

    # Fallback
    if fallback_dir is not None and fallback_dir != log_dir:
        return load_history(fallback_dir, fallback_dir=None)

    raise FileNotFoundError(f"No history.csv or history.json found in {log_dir}")


def plot_data_loss(hist: Dict[str, List[float]], out_dir: Path):
    epochs = range(1, len(hist.get("loss", [])) + 1)
    data_loss = hist.get("data_loss")
    if not data_loss:
        raise ValueError("No data_loss found in history.")

    plt.figure()
    plt.plot(list(epochs), data_loss, marker="o")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Data loss" if "data_loss" in hist else "Loss")
    plt.title("Data loss vs Epoch")
    plt.grid(True, alpha=0.3)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_dir / "data_loss.png", dpi=200)
    plt.close()


def plot_heatmaps(ckpt_dir: Path, data_path: Path, paper_dir: Path, device: torch.device):
    ckpts = sorted(ckpt_dir.glob("epoch_*.pt"))
    if not ckpts:
        print(f"No checkpoints found in {ckpt_dir}")
        return

    paper_dir.mkdir(parents=True, exist_ok=True)

    for ckpt_path in ckpts:
        # Only plot every 10 epochs
        try:
            epoch_num = int(ckpt_path.stem.split("_")[1])
        except Exception:
            continue
        if epoch_num % 10 != 0:
            continue

        out_subdir = paper_dir / f"epoch_{epoch_num:04d}"
        generate_visualizations(
            data_path=data_path,
            checkpoint_path=ckpt_path,
            output_dir=out_subdir,
            device=str(device),
        )


def main():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    parser = argparse.ArgumentParser(description="Plot data loss and heatmaps from logs/checkpoints.")
    parser.add_argument("--log-dir", default="checkpoints/mode-pinn_ff-32_hardware_exp_1219", help="Directory containing history.csv/json")
    parser.add_argument("--ckpt-dir", default="checkpoints/mode-pinn_ff-32_hardware_exp_1219", help="Checkpoint directory")
    parser.add_argument("--data-path", default="data/pulse_evolution.mat", help="Path to dataset")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Device for inference")
    parser.add_argument("--paper-dir", default="figures/paperfigure", help="Output directory for figures")
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    log_dir = Path(args.log_dir)
    ckpt_dir = Path(args.ckpt_dir)
    paper_dir = Path(args.paper_dir)
    paper_dir.mkdir(parents=True, exist_ok=True)

    # Plot data loss
    hist = load_history(log_dir, fallback_dir=ckpt_dir)
    plot_data_loss(hist, paper_dir)

    # Heatmaps per 10 epochs
    # Use config from latest checkpoint to construct dataset
    latest_ckpt = sorted(ckpt_dir.glob("epoch_*.pt"))
    if not latest_ckpt:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    latest = latest_ckpt[-1]
    cfg = torch.load(latest, map_location="cpu")["config"]
    z_stride = getattr(cfg, "z_stride", 1)
    # Build dataset to respect z_stride if needed (not directly used in visualization helper).
    _ = PulseEvolutionDataset(args.data_path, z_stride=z_stride)

    plot_heatmaps(ckpt_dir, Path(args.data_path), paper_dir, device)

    print(f"Figures saved to {paper_dir}")


if __name__ == "__main__":
    main()
