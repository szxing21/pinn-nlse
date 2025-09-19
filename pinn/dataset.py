from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import scipy.io
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class NormalisationStats:
    input_min: np.ndarray
    input_scale: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray

    def denormalise_inputs(self, values: np.ndarray) -> np.ndarray:
        return ((values + 1.0) * 0.5) * self.input_scale + self.input_min

    def denormalise_targets(self, values: np.ndarray) -> np.ndarray:
        return values * self.target_std + self.target_mean


class PulseEvolutionDataset(Dataset):
    """Dataset exposing (z, t) -> complex field samples with normalisation."""

    def __init__(
        self,
        mat_path: str | Path,
        *,
        dtype: torch.dtype = torch.float32,
        normalise: bool = True,
    ) -> None:
        payload = load_pulse_evolution(mat_path, normalise=normalise)
        self._inputs = torch.from_numpy(payload["inputs"]).to(dtype)
        self._targets = torch.from_numpy(payload["targets"]).to(dtype)

        self.z = payload["z"]
        self.t = payload["t"]
        self.stats = payload["stats"]

        self._initial_indices, self._boundary_indices, self._interior_indices = self._build_region_indices()

    def __len__(self) -> int:
        return self._inputs.shape[0]

    def __getitem__(self, idx: int):
        return self._inputs[idx], self._targets[idx]

    @property
    def inputs(self) -> torch.Tensor:
        return self._inputs

    @property
    def targets(self) -> torch.Tensor:
        return self._targets

    def denormalise_targets(self, values: torch.Tensor) -> torch.Tensor:
        """Map normalised network outputs back to the original scale."""

        if not isinstance(self.stats, NormalisationStats):
            raise ValueError("Normalisation statistics are unavailable for denormalisation.")

        mean = torch.as_tensor(
            self.stats.target_mean,
            dtype=values.dtype,
            device=values.device,
        )
        std = torch.as_tensor(
            self.stats.target_std,
            dtype=values.dtype,
            device=values.device,
        )
        return values * std + mean

    def sample_initial(self, num_samples: int | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._sample_from_indices(self._initial_indices, num_samples)

    def sample_boundary(self, num_samples: int | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._sample_from_indices(self._boundary_indices, num_samples)

    def sample_interior(self, num_samples: int | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._sample_from_indices(self._interior_indices, num_samples)

    def _sample_from_indices(
        self,
        indices: torch.Tensor,
        num_samples: int | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if indices.numel() == 0:
            empty = torch.empty((0, self._inputs.size(1)), dtype=self._inputs.dtype)
            return empty, torch.empty((0, self._targets.size(1)), dtype=self._targets.dtype)

        if num_samples is not None and 0 < num_samples < indices.numel():
            choice = torch.randperm(indices.numel())[:num_samples]
            indices = indices[choice]

        inputs = self._inputs.index_select(0, indices)
        targets = self._targets.index_select(0, indices)
        return inputs, targets

    def _build_region_indices(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tol = 1e-5
        z_channel = self._inputs[:, 0]
        t_channel = self._inputs[:, 1]

        z_min_val = torch.full_like(z_channel, -1.0)
        z_max_val = torch.full_like(z_channel, 1.0)
        t_min_val = torch.full_like(t_channel, -1.0)
        t_max_val = torch.full_like(t_channel, 1.0)

        initial_mask = torch.isclose(z_channel, z_min_val, atol=tol)
        boundary_mask = torch.isclose(t_channel, t_min_val, atol=tol) | torch.isclose(t_channel, t_max_val, atol=tol)

        initial_indices = torch.nonzero(initial_mask, as_tuple=True)[0]
        boundary_indices = torch.nonzero(boundary_mask, as_tuple=True)[0]
        interior_mask = ~(initial_mask | boundary_mask)
        interior_indices = torch.nonzero(interior_mask, as_tuple=True)[0]

        return initial_indices, boundary_indices, interior_indices


def load_pulse_evolution(
    mat_path: str | Path,
    *,
    normalise: bool = True,
) -> Dict[str, np.ndarray | NormalisationStats]:
    """Load and reshape the raw MATLAB tensor into learning-ready arrays."""

    mat_path = Path(mat_path)
    mat = scipy.io.loadmat(mat_path)

    try:
        t = np.asarray(mat["T_ps"], dtype=np.float32).reshape(-1)
        z = np.asarray(mat["Z"], dtype=np.float32).reshape(-1)
        tensor = np.asarray(mat["Tensor"], dtype=np.float32)
    except KeyError as exc:  # pragma: no cover - defensive
        missing = ", ".join(sorted(set(name for name in ("T_ps", "Z", "Tensor") if name not in mat)))
        raise KeyError(f"Required key(s) missing from MATLAB file: {missing}") from exc

    if tensor.shape[0] != z.size or tensor.shape[1] != t.size:
        raise ValueError(
            "Tensor dimensions must align with Z and T_ps axes; "
            f"got tensor={tensor.shape}, Z={z.shape}, T_ps={t.shape}."
        )

    z_grid, t_grid = np.meshgrid(z, t, indexing="ij")
    inputs = np.stack([z_grid, t_grid], axis=-1).reshape(-1, 2)
    targets = tensor.reshape(-1, tensor.shape[-1])

    # Compute scaling statistics once.
    input_min = inputs.min(axis=0)
    input_max = inputs.max(axis=0)
    input_scale = np.where(input_max - input_min == 0.0, 1.0, input_max - input_min)
    inputs_norm = 2.0 * (inputs - input_min) / input_scale - 1.0

    target_mean = targets.mean(axis=0)
    target_std = targets.std(axis=0)
    target_std = np.where(target_std == 0.0, 1.0, target_std)
    targets_norm = (targets - target_mean) / target_std

    if normalise:
        inputs_out = inputs_norm
        targets_out = targets_norm
    else:
        inputs_out = inputs
        targets_out = targets

    stats = NormalisationStats(
        input_min=input_min.astype(np.float32),
        input_scale=input_scale.astype(np.float32),
        target_mean=target_mean.astype(np.float32),
        target_std=target_std.astype(np.float32),
    )

    return {
        "inputs": inputs_out.astype(np.float32),
        "targets": targets_out.astype(np.float32),
        "z": z.astype(np.float32),
        "t": t.astype(np.float32),
        "stats": stats,
    }


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    *,
    num_workers: int = 0,
    pin_memory: bool = False,
    shuffle: bool = True,
) -> DataLoader:
    """Helper for wrapping datasets with a standard DataLoader."""

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
