"""Simple PyTorch MLP model for classification tasks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import Tensor, nn

from fedbrew.models.config_keys import reject_unknown_model_keys


class TorchMLP(nn.Module):  # type: ignore[misc]
    """Small feed-forward network for synthetic classification."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.0,
    ) -> None:
        """Build a Linear -> ReLU -> (Dropout) -> Linear stack.

        Args:
            input_dim: Features per example. Inputs arrive already flattened;
                this module does no reshaping, so a (N, C, H, W) image tensor
                must be flattened by the caller first.
            hidden_dim: Width of the single hidden layer.
            num_classes: Number of output logits.
            dropout: Drop probability in [0, 1). Applied between the hidden
                ReLU and the output layer, and only in training mode.

        Raises:
            ValueError: If ``dropout`` is outside [0, 1).
        """

        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        ]
        # Off by default, so no existing config changes. It exists because the
        # dev fixtures had no stochastic layer at all: nn.Dropout is what draws
        # from the process-wide RNG during training, and without it a run can
        # be reproducible for the trivial reason that nothing is random after
        # initialisation. See tests/test_reproducibility.py.
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        """Compute class logits for a batch of inputs.

        Args:
            inputs: Float tensor of shape (N, input_dim). Already flattened
                and already on the module's device.

        Returns:
            Float tensor of shape (N, num_classes): raw logits, not
            probabilities. The loss is applied by the task adapter, which
            expects logits.
        """

        return self.net(inputs)


#: Every key build_torch_mlp below reads. The first three are named
#: ModelConfig fields; dropout comes through model.extra.
_KNOWN_KEYS = frozenset({"input_dim", "hidden_dim", "num_classes", "dropout"})


def build_torch_mlp(config: Mapping[str, Any] | None = None) -> nn.Module:
    """Build a TorchMLP from a ``model`` config mapping.

    Args:
        config: Keys ``input_dim`` (default 5), ``hidden_dim`` (16),
            ``num_classes`` (2) and ``dropout`` (0.0). An unrecognised key
            raises rather than being ignored, so a misspelling cannot leave the
            model silently at a default.

    Returns:
        A TorchMLP on the CPU, in whatever mode nn.Module defaults to. The
        caller moves it to the run device.
    """

    config = config or {}
    reject_unknown_model_keys(config, _KNOWN_KEYS, "mlp")
    input_dim = int(config.get("input_dim", 5))
    hidden_dim = int(config.get("hidden_dim", 16))
    num_classes = int(config.get("num_classes", 2))
    dropout = float(config.get("dropout", 0.0) or 0.0)
    return TorchMLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout,
    )
