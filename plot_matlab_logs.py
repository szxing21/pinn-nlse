from __future__ import annotations

"""
Helper to read MATLAB log files saved by `_call_matlab_real_mul_shift` and plot
theoretical (software) outputs against the logged hardware/MATLAB outputs.

Each log consists of:
  - `{epoch_tag}_call_xxxxxx.npz` with arrays `W` (4x4), `X` (4,B), `Y` (4,B)
  - optional JSON sidecar with metadata such as epoch/layer/tile indices.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _normalize_epoch_tag(epoch: str | int) -> str:
    """Return the epoch tag used in filenames (e.g., 1 -> 'epoch_1')."""
    if isinstance(epoch, int):
        return f"epoch_{epoch}"
    epoch_str = str(epoch)
    return epoch_str if epoch_str.startswith("epoch") else f"epoch_{epoch_str}"


def load_matlab_call(
    epoch: str | int,
    *,
    call_idx: int = 0,
    log_dir: str | Path = "matlab_logs",
) -> Dict[str, Any]:
    """
    Load one MATLAB log entry for the given epoch.

    Returns a dictionary containing:
      - W, X: inputs used for the hardware tile (numpy arrays)
      - y_actual: logged MATLAB output (Y)
      - y_theory: software matmul result W @ X
      - meta: metadata from the sidecar JSON (if present)
      - npz_path: path to the loaded .npz file
    """
    tag = _normalize_epoch_tag(epoch)
    log_dir = Path(log_dir)
    npz_files: List[Path] = sorted(log_dir.glob(f"{tag}_call_*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No logs found for {tag} in {log_dir}")
    if call_idx < 0 or call_idx >= len(npz_files):
        raise IndexError(f"call_idx {call_idx} out of range (0..{len(npz_files) - 1})")

    npz_path = npz_files[call_idx]
    data = np.load(npz_path)
    W = np.array(data["W"])
    X = np.array(data["X"])
    y_actual = np.array(data["Y"])
    y_theory = W @ X

    meta = {}
    meta_path = npz_path.with_suffix(".json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    return {
        "epoch": tag,
        "npz_path": npz_path,
        "meta": meta,
        "W": W,
        "X": X,
        "y_actual": y_actual,
        "y_theory": y_theory,
    }


def plot_actual_vs_theory(
    epoch: str | int,
    *,
    call_idx: int = 0,
    log_dir: str | Path = "matlab_logs",
    save_path: str | Path | None = None,
    show: bool = True,
):
    """
    Plot logged MATLAB outputs vs. theoretical W@X for a single call.

    NaN values in the logged output are left as-is (Matplotlib will show gaps).
    """
    record = load_matlab_call(epoch, call_idx=call_idx, log_dir=log_dir)
    y_actual = record["y_actual"]
    y_theory = record["y_theory"]

    n_rows, _ = y_theory.shape
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 2.5 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    for row, ax in enumerate(axes):
        ax.plot(y_theory[row], label="theory: W @ X", linewidth=1.6)
        ax.plot(y_actual[row], label="actual: MATLAB Y", linewidth=1.2, linestyle="--")
        ax.set_ylabel(f"out {row}")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("batch index")
    axes[0].legend(loc="upper right")

    title_parts = [record["epoch"], f"call {call_idx}"]
    if record["meta"]:
        meta_bits = ", ".join(f"{k}={v}" for k, v in record["meta"].items())
        title_parts.append(f"meta: {meta_bits}")
    fig.suptitle(" | ".join(title_parts))

    fig.tight_layout()
    if save_path:
        save_path = Path(save_path)
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load MATLAB logs for an epoch and plot theory vs. actual outputs."
    )
    parser.add_argument(
        "--epoch",
        required=True,
        help="Epoch tag or number (e.g., 1 or epoch_1).",
    )
    parser.add_argument(
        "--call-idx",
        type=int,
        default=0,
        help="Which call within the epoch to plot (0-based).",
    )
    parser.add_argument(
        "--log-dir",
        default="matlab_logs",
        help="Directory that stores the MATLAB log .npz/.json files.",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Optional path to save the plot instead of just showing it.",
    )
    args = parser.parse_args()

    # Show the requested call. Use --save to export to file.
    plot_actual_vs_theory(
        args.epoch,
        call_idx=args.call_idx,
        log_dir=args.log_dir,
        save_path=args.save,
        show=args.save is None,
    )


if __name__ == "__main__":
    main()
