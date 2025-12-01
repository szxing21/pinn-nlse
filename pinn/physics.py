from __future__ import annotations

from typing import Tuple

import numpy as np

from .config import TrainingConfig


def _finite_difference_derivatives(
    z: np.ndarray,
    t: np.ndarray,
    amplitudes: np.ndarray,
    config: TrainingConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return real and imaginary PDE residual components via finite differences."""

    if amplitudes.ndim != 2:
        raise ValueError("Amplitudes must be a 2-D complex grid with shape [z, t].")

    z = np.asarray(z, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)

    if z.ndim != 1 or t.ndim != 1:
        raise ValueError("Spatial axes z and t must be 1-D arrays.")

    if amplitudes.shape != (z.size, t.size):
        raise ValueError(
            f"Amplitude grid shape {amplitudes.shape} does not match z ({z.size}) and t ({t.size})."
        )

    a_real = amplitudes.real
    a_imag = amplitudes.imag

    per_km = 1.0e3
    dz = float(np.abs(z[1] - z[0])) if z.size > 1 else 1.0
    dt = float(np.abs(t[1] - t[0])) if t.size > 1 else 1.0

    dA_r_dz = np.gradient(a_real, dz, axis=0, edge_order=2) * per_km
    dA_i_dz = np.gradient(a_imag, dz, axis=0, edge_order=2) * per_km

    dA_r_dt = np.gradient(a_real, dt, axis=1, edge_order=2)
    dA_i_dt = np.gradient(a_imag, dt, axis=1, edge_order=2)

    d2A_r_dt2 = np.gradient(dA_r_dt, dt, axis=1, edge_order=2)
    d2A_i_dt2 = np.gradient(dA_i_dt, dt, axis=1, edge_order=2)

    if config.beta3 != 0.0:
        d3A_r_dt3 = np.gradient(d2A_r_dt2, dt, axis=1, edge_order=2)
        d3A_i_dt3 = np.gradient(d2A_i_dt2, dt, axis=1, edge_order=2)
    else:
        d3A_r_dt3 = np.zeros_like(d2A_r_dt2)
        d3A_i_dt3 = np.zeros_like(d2A_i_dt2)

    power = a_real ** 2 + a_imag ** 2

    alpha_term = 0.5 * config.alpha if config.pde_variant == "standard" else 0.0

    residual_real = (
        dA_r_dz
        + alpha_term * a_real
        + 0.5 * config.beta2 * d2A_i_dt2
        - (config.beta3 / 6.0) * d3A_r_dt3
        + config.gamma * power * a_imag
    )

    residual_imag = (
        dA_i_dz
        + alpha_term * a_imag
        - 0.5 * config.beta2 * d2A_r_dt2
        - (config.beta3 / 6.0) * d3A_i_dt3
        - config.gamma * power * a_real
    )

    return residual_real, residual_imag


def residual_statistics(
    z: np.ndarray,
    t: np.ndarray,
    amplitudes: np.ndarray,
    config: TrainingConfig,
) -> Tuple[float, float, float]:
    """Compute RMS, max, and MSE of the PDE residual for a complex amplitude grid."""

    residual_real, residual_imag = _finite_difference_derivatives(z, t, amplitudes, config)
    residual_sq = residual_real ** 2 + residual_imag ** 2
    mse = float(np.mean(residual_sq))
    rms = float(np.sqrt(mse))
    max_val = float(np.sqrt(np.max(residual_sq)))
    return rms, max_val, mse
