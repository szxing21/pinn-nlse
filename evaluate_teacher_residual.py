import argparse
from pathlib import Path
import numpy as np
import scipy.io

from pinn.config import TrainingConfig
from pinn.physics import residual_statistics


def load_teacher(path: Path):
    data = scipy.io.loadmat(path)
    tensor = np.asarray(data['Tensor'], dtype=np.float64)  # [Z, T, 2]
    z = np.asarray(data['Z'], dtype=np.float64).reshape(-1)
    t = np.asarray(data['T_ps'], dtype=np.float64).reshape(-1)
    if np.max(np.abs(t)) < 1e-6:
        t = t * 1e12  # seconds -> picoseconds
    a_complex = tensor[..., 0] + 1j * tensor[..., 1]
    return z, t, a_complex


def compute_residual(z_m, t_ps, amplitudes, config: TrainingConfig):
    rms, max_val, _ = residual_statistics(z_m, t_ps, amplitudes, config)
    return rms, max_val


def main():
    parser = argparse.ArgumentParser(description="Evaluate PDE residuals for teacher data.")
    parser.add_argument('--data', default='data/pulse_evolution.mat')
    args = parser.parse_args()

    config = TrainingConfig()
    z, t, a = load_teacher(Path(args.data))
    rms, max_val = compute_residual(z, t, a, config)

    print(f"Teacher residual RMS: {rms:.6e}")
    print(f"Teacher residual max: {max_val:.6e}")


if __name__ == '__main__':
    main()
