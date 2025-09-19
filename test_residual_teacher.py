import torch

from pinn.dataset import PulseEvolutionDataset
from pinn.losses import _residual_loss
from pinn import TrainingConfig


class TeacherModel(torch.nn.Module):
    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor):
        super().__init__()
        self.register_buffer("targets", targets)
        # Map normalised coordinate tuples to row indices for exact lookup.
        mapping = {}
        for idx, row in enumerate(inputs.cpu()):
            mapping[tuple(row.tolist())] = idx
        self._mapping = mapping

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rows = []
        for row in x.detach().cpu():
            key = tuple(row.tolist())
            rows.append(self.targets[self._mapping[key]])
        return torch.stack(rows, dim=0).to(x.device)


def main() -> None:
    dataset = PulseEvolutionDataset("data/pulse_evolution.mat")
    model = TeacherModel(dataset.inputs, dataset.targets)
    config = TrainingConfig()

    inputs_norm, _ = dataset.sample_interior(min(len(dataset), 4096))
    residual_loss, _ = _residual_loss(
        model,
        dataset.stats,
        inputs_norm,
        device=torch.device("cpu"),
        config=config,
    )
    print("Teacher residual loss:", residual_loss.item())


if __name__ == "__main__":
    main()
