from __future__ import annotations

import argparse
from pathlib import Path

import torch

from main import build_argument_parser  # reuse dataset/model construction args
from model import SimplePINN
from pinn import TrainingConfig, PulseEvolutionDataset, create_dataloader
from pinn.losses import (
    compute_pinn_loss_components,
    compute_pinn_loss_components_ssfm,
    PINNLossComponents,
)


def load_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> int:
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return int(ckpt.get("epoch", 0))


def main():
    parser = argparse.ArgumentParser(description="Approximate loss from a checkpoint (no training).")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint epoch_xxxx.pt")
    parser.add_argument("--data-path", default="data/pulse_evolution.mat", help="Path to dataset mat file")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--mode", choices=["pinn", "mlp"], default="pinn", help="Model mode")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Device for eval")
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    # Build config (minimal fields needed)
    base_cfg = TrainingConfig(mode=args.mode, batch_size=args.batch_size)

    # Load dataset/dataloader
    dataset = PulseEvolutionDataset(args.data_path, base_cfg)
    dataloader = create_dataloader(dataset, base_cfg, shuffle=False)

    # Build model
    sample_input, sample_target = dataset[0]
    input_dim = sample_input.numel()
    target_dim = sample_target.numel()
    model = SimplePINN(input_dim=input_dim, output_dim=target_dim, config=base_cfg).to(device)

    # Load checkpoint
    start_epoch = load_checkpoint(model, Path(args.ckpt), device)
    model.eval()

    # Choose loss fn
    if args.mode == "pinn":
        loss_fn = compute_pinn_loss_components_ssfm
    else:
        loss_fn = compute_pinn_loss_components

    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        for features, targets in dataloader:
            features = features.to(device)
            targets = targets.to(device)
            preds = model(features)
            data_loss = torch.nn.functional.mse_loss(preds, targets)
            if args.mode == "pinn":
                components: PINNLossComponents = loss_fn(
                    model, dataset, device=device, data_batch=(features, targets)
                )
                total = (
                    base_cfg.data_weight * components.data
                    + base_cfg.initial_weight * components.initial
                    + base_cfg.boundary_weight * components.boundary
                    + base_cfg.residual_weight * components.residual
                )
                total_loss += float(total.item())
            else:
                total_loss += float(data_loss.item())
            steps += 1

    approx = total_loss / max(steps, 1)
    print(f"Checkpoint epoch={start_epoch}, approx loss={approx:.6e}, steps={steps}, device={device}")


if __name__ == "__main__":
    main()
