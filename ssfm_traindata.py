"""
Python port of SSFM_traindata.m for generating pulse evolution data.

The numerical scheme matches the original MATLAB script, including sampling
and save cadence. The only intentional change is that the output .mat file is
written to ``data/pulse_evolution.mat``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend; we save figures instead of showing
import matplotlib.pyplot as plt
import numpy as np
import scipy.io


def run_ssfm() -> None:
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Parameter setup (units mirror the MATLAB script).
    L = 100 * 1e3  # total length (m)
    dz = 100.0  # SSFM step (m)
    Nz = int(round(L / dz))

    beta2 = -21.242e-27
    beta3 = 16.6e-41
    gamma = 1.3
    gamma_SI = gamma / 1e3  # (1/W/m)

    T = 3000e-12
    N = 2**10
    dt = T / N
    t = np.arange(-N // 2, N // 2, dtype=np.float64) * dt

    P0 = 0.5
    FWHM = 100e-12
    sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
    delays = np.array([-400, 0, 400], dtype=np.float64) * 1e-12

    A0 = np.zeros_like(t, dtype=np.complex128)
    for delay in delays:
        A0 += np.sqrt(P0) * np.exp(-((t - delay) ** 2) / (2 * sigma**2))

    df = 1.0 / (N * dt)
    f = np.arange(-N // 2, N // 2, dtype=np.float64) * df
    omega = 2 * np.pi * f

    # Save tensor initialisation.
    save_interval = 10e3  # save every 10 km (value kept identical to MATLAB)
    num_save = int(round(L / save_interval))
    Tensor = np.zeros((num_save + 1, N, 2), dtype=np.float64)
    Tensor[0, :, 0] = A0.real
    Tensor[0, :, 1] = A0.imag

    A = A0.copy()
    counter = 0
    save_every_steps = int(round(save_interval / dz))

    dispersion_phase = np.exp(-1j * (beta2 / 2 * omega**2 + beta3 / 6 * omega**3) * dz)

    for step in range(1, Nz + 1):
        # Dispersion in frequency domain.
        A_freq = np.fft.fftshift(np.fft.fft(A))
        A_freq *= dispersion_phase
        A = np.fft.ifft(np.fft.ifftshift(A_freq))

        # Nonlinearity in time domain.
        A *= np.exp(1j * gamma_SI * np.abs(A) ** 2 * dz)

        if step % save_every_steps == 0:
            counter += 1
            Tensor[counter, :, 0] = A.real
            Tensor[counter, :, 1] = A.imag

    # 3D waterfall plot (z-t evolution).
    Z = np.arange(0, L + save_interval, save_interval, dtype=np.float64)
    T_ps_plot = t * 1e12
    Power = Tensor[:, :, 0] ** 2 + Tensor[:, :, 1] ** 2

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for k, z_val in enumerate(Z):
        ax.plot(T_ps_plot, np.full_like(T_ps_plot, z_val), Power[k, :], color="b", linewidth=1.2)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Distance (km)")
    ax.set_zlabel("Power (W)")
    ax.set_title("Pulse evolution along fiber (blue case)")
    ax.set_xlim([t.min() * 1e12, t.max() * 1e12])
    ax.set_box_aspect((1, 3, 1))
    ax.view_init(elev=13.5784, azim=65.25)
    ax.set_proj_type("ortho")
    fig.savefig(data_dir / "pulse_waterfall.png", dpi=300, bbox_inches="tight")

    # Initial vs final pulse comparison.
    A_initial = Tensor[0, :, 0] + 1j * Tensor[0, :, 1]
    A_final = Tensor[-1, :, 0] + 1j * Tensor[-1, :, 1]

    fig2 = plt.figure()
    plt.plot(T_ps_plot, np.abs(A_initial) ** 2, "b-", linewidth=2, label="Initial")
    plt.plot(T_ps_plot, np.abs(A_final) ** 2, "r--", linewidth=2, label="Final")
    plt.xlabel("Time (ps)")
    plt.ylabel("Power (W)")
    plt.title("Initial vs Final Pulse")
    plt.legend()
    plt.grid(True)
    plt.xlim([t.min() * 1e12, t.max() * 1e12])
    fig2.savefig(data_dir / "pulse_initial_final.png", dpi=300, bbox_inches="tight")

    # Save outputs to .mat (path intentionally changed to data/pulse_evolution.mat).
    T_ps_save = t  # seconds, consistent with the MATLAB export
    output_path = data_dir / "pulse_evolution.mat"
    scipy.io.savemat(output_path, {"Tensor": Tensor, "Z": Z, "T_ps": T_ps_save})
    print(f"Saved tensor data to {output_path.resolve()}")
    plt.close("all")


if __name__ == "__main__":
    run_ssfm()
