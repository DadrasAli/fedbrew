"""Small CNN models for torch image classification tasks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn

from fedbrew.models.config_keys import reject_unknown_model_keys


class TorchSmallCNN(nn.Module):
    """Compact CNN suitable for small CIFAR-10 FL smoke tests.

    Two conv/pool stages then a two-layer head. **Input must be 32x32.** The
    classifier's first Linear is built with a hardcoded 64 * 8 * 8 input, which
    is what two stride-2 max-pools leave of a 32x32 image; any other spatial
    size reaches it with a different flattened width and raises a shape
    mismatch at the first forward pass, not at construction.
    """

    def __init__(
        self,
        input_channels: int = 3,
        hidden_dim: int = 128,
        num_classes: int = 10,
    ) -> None:
        """Build the feature stack and classifier head.

        Args:
            input_channels: Channels per input image; 3 for RGB, 1 for
                grayscale. Only the first convolution depends on it.
            hidden_dim: Width of the hidden Linear in the classifier head.
            num_classes: Number of output logits.
        """

        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):  # type: ignore[no-untyped-def]
        """Compute class logits for a batch of images.

        Args:
            x: Float tensor of shape (N, input_channels, 32, 32), already
                normalised and on the module's device. The 32x32 spatial size
                is required -- see the class docstring.

        Returns:
            Float tensor of shape (N, num_classes): raw logits, not
            probabilities.
        """

        return self.classifier(self.features(x))


#: Every key build_torch_cnn below reads; hidden_dim and num_classes are also
#: named ModelConfig fields, which _model_config injects under the same names.
_KNOWN_KEYS = frozenset({"input_channels", "hidden_dim", "num_classes"})


def build_torch_cnn(config: Mapping[str, Any] | None = None) -> TorchSmallCNN:
    """Build a TorchSmallCNN from a ``model`` config mapping.

    Args:
        config: Keys ``input_channels`` (default 3), ``hidden_dim`` (128) and
            ``num_classes`` (10). An unrecognised key raises rather than being
            ignored. There is no spatial-size key: the model is fixed at 32x32.

    Returns:
        A TorchSmallCNN on the CPU. The caller moves it to the run device.
    """

    values = dict(config or {})
    reject_unknown_model_keys(values, _KNOWN_KEYS, "cnn")
    return TorchSmallCNN(
        input_channels=int(values.get("input_channels", 3)),
        hidden_dim=int(values.get("hidden_dim", 128)),
        num_classes=int(values.get("num_classes", 10)),
    )


def build_torch_small_cnn(config: Mapping[str, Any] | None = None) -> TorchSmallCNN:
    """Build the model registered as ``small_cnn``.

    An alias for :func:`build_torch_cnn` with identical behaviour and config
    keys; the two registry names exist so a config can say which it means.
    """

    return build_torch_cnn(config)
