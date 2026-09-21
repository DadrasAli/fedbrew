"""Residual image classifier tailored to naturally partitioned FEMNIST."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from fedbrew.models.config_keys import reject_unknown_model_keys
from fedbrew.models.group_norm import group_norm

#: Input normalization, fixed rather than configurable. These describe the
#: [0, 255] uint8 contract the FEMNIST shards are written under, not a tunable:
#: PIXEL_SCALE is what uint8 means, and the 0.5/0.5 shift is the standard
#: [0, 1] -> [-1, 1] map. Changing either without regenerating the shards
#: trains on the wrong units, which is the failure the class docstring warns
#: about, so there is nothing a config could usefully say here.
PIXEL_SCALE = 255.0
PIXEL_MEAN = 0.5
PIXEL_STD = 0.5


class FEMNISTResidualBlock(nn.Module):
    """Basic residual block using batch-size-independent GroupNorm.

    GroupNorm rather than BatchNorm throughout: BatchNorm's running mean and
    variance are client-specific statistics, and averaging them across clients
    mixes quantities that were never estimated on the same distribution.
    GroupNorm has no running state, so a client's normalisation depends only on
    the batch in front of it and the aggregated model carries no leaked
    statistics.
    """

    expansion = 1

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        stride: int,
        group_norm_groups: int,
    ) -> None:
        """Build the two-convolution residual path and its shortcut.

        Args:
            input_channels: Channels of the incoming feature map.
            output_channels: Channels of the outgoing feature map.
            stride: Spatial stride of the first convolution. 1 preserves H and
                W; 2 halves both.
            group_norm_groups: Requested GroupNorm groups, capped and reduced
                to a divisor of ``output_channels`` by
                :func:`~fedbrew.models.group_norm.group_norm`.

        The shortcut is a 1x1 projection whenever stride != 1 or the channel
        count changes, and nn.Identity otherwise -- so the residual sum always
        has matching shapes.
        """

        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm1 = group_norm(output_channels, group_norm_groups)
        self.activation = nn.GELU()
        self.conv2 = nn.Conv2d(
            output_channels,
            output_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm2 = group_norm(output_channels, group_norm_groups)

        self.shortcut: nn.Module
        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                group_norm(output_channels, group_norm_groups),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply the residual block.

        Args:
            inputs: Float tensor of shape (N, input_channels, H, W).

        Returns:
            Float tensor of shape (N, output_channels, H // stride,
            W // stride). The activation is applied *after* the residual sum,
            so the output is non-negative wherever GELU is.
        """

        residual = self.shortcut(inputs)
        outputs = self.activation(self.norm1(self.conv1(inputs)))
        outputs = self.norm2(self.conv2(outputs))
        return self.activation(outputs + residual)


class FEMNISTResNet(nn.Module):
    """Compact ResNet-18 variant for 28x28 grayscale FEMNIST images.

    GroupNorm avoids aggregating client-specific BatchNorm running statistics.
    Input normalization lives in the model so generated shards can retain compact
    uint8 pixels on disk.

    **Input units.** ``forward`` expects raw pixel values in [0, 255] as
    floats, not pre-normalised inputs: it divides by :data:`PIXEL_SCALE` and
    then standardises by :data:`PIXEL_MEAN` / :data:`PIXEL_STD` itself. Handing
    it data already scaled to [0, 1] trains on inputs 255x too small.

    **Spatial size is not fixed.** A stride-1 stem and three stride-2 stages
    downsample by 8x, and an AdaptiveAvgPool2d((1, 1)) collapses whatever is
    left, so any H and W work -- unlike
    :class:`~fedbrew.models.torch_cnn.TorchSmallCNN`. 28x28 is what FEMNIST
    ships and what the defaults are tuned for.

    At the shipped defaults the model holds ~2.81M parameters, all of which are
    communicated every round.
    """

    def __init__(
        self,
        input_channels: int = 1,
        base_channels: int = 32,
        num_classes: int = 62,
        blocks_per_stage: Sequence[int] = (2, 2, 2, 2),
        group_norm_groups: int = 8,
        dropout: float = 0.1,
    ) -> None:
        """Build the stem, four residual stages and the classifier head.

        Args:
            input_channels: Channels per image; 1 for FEMNIST grayscale.
            base_channels: Width of stage 0. Stages widen 1x, 2x, 4x, 8x from
                it, so the classifier sees ``base_channels * 8`` features.
            num_classes: Output logits. 62 for FEMNIST (10 digits + 52 letters).
            blocks_per_stage: Exactly four positive integers, one residual
                block count per stage. (2, 2, 2, 2) is ResNet-18's shape.
            group_norm_groups: Requested GroupNorm groups per block; reduced
                per stage to a divisor of that stage's width.
            dropout: Drop probability in [0, 1), applied to the pooled features
                just before the classifier, in training mode only.

        Raises:
            ValueError: If any dimension is non-positive, ``blocks_per_stage``
                is not four positive integers, or ``dropout`` is outside
                [0, 1).

        The final GroupNorm of every residual block is zero-initialised, so each
        block starts as an identity map and the network begins training as a
        much shallower one.
        """

        super().__init__()
        if input_channels <= 0:
            raise ValueError("input_channels must be positive")
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if len(blocks_per_stage) != 4 or any(
            not isinstance(count, int) or isinstance(count, bool) or count <= 0
            for count in blocks_per_stage
        ):
            raise ValueError("blocks_per_stage must contain four positive integers")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self._current_channels = base_channels

        self.stem = nn.Sequential(
            nn.Conv2d(
                input_channels,
                base_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            group_norm(base_channels, group_norm_groups),
            nn.GELU(),
        )
        widths = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        )
        self.stages = nn.Sequential(
            *[
                self._make_stage(
                    output_channels=width,
                    num_blocks=int(num_blocks),
                    first_stride=1 if stage_index == 0 else 2,
                    group_norm_groups=group_norm_groups,
                )
                for stage_index, (width, num_blocks) in enumerate(
                    zip(widths, blocks_per_stage, strict=True)
                )
            ]
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=float(dropout))
        self.classifier = nn.Linear(widths[-1], num_classes)
        self._initialize_parameters()

    def _make_stage(
        self,
        output_channels: int,
        num_blocks: int,
        first_stride: int,
        group_norm_groups: int,
    ) -> nn.Sequential:
        blocks: list[nn.Module] = [
            FEMNISTResidualBlock(
                input_channels=self._current_channels,
                output_channels=output_channels,
                stride=first_stride,
                group_norm_groups=group_norm_groups,
            )
        ]
        self._current_channels = output_channels
        blocks.extend(
            FEMNISTResidualBlock(
                input_channels=output_channels,
                output_channels=output_channels,
                stride=1,
                group_norm_groups=group_norm_groups,
            )
            for _ in range(num_blocks - 1)
        )
        return nn.Sequential(*blocks)

    def _initialize_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

        for module in self.modules():
            if isinstance(module, FEMNISTResidualBlock):
                nn.init.zeros_(module.norm2.weight)

    def forward(self, inputs: Tensor) -> Tensor:
        """Compute class logits for raw FEMNIST pixels.

        Args:
            inputs: Float tensor of shape (N, input_channels, H, W) holding
                **raw pixel values in [0, 255]**. Normalisation happens here,
                not in the dataset -- see the class docstring.

        Returns:
            Float tensor of shape (N, num_classes): raw logits, not
            probabilities.
        """

        normalized = inputs / PIXEL_SCALE
        normalized = (normalized - PIXEL_MEAN) / PIXEL_STD
        features = self.stages(self.stem(normalized))
        pooled = torch.flatten(self.pool(features), 1)
        return self.classifier(self.dropout(pooled))


#: Every key build_femnist_resnet18 below reads.
_KNOWN_KEYS = frozenset(
    {
        "input_channels",
        "base_channels",
        "blocks_per_stage",
        "group_norm_groups",
        "dropout",
    }
)


def build_femnist_resnet18(
    config: Mapping[str, Any] | None = None,
) -> FEMNISTResNet:
    """Build the FL-friendly FEMNIST ResNet-18 variant.

    Args:
        config: ``model`` block keys, all optional: ``input_channels`` (1),
            ``base_channels`` (32), ``num_classes`` (62), ``blocks_per_stage``
            ([2, 2, 2, 2]), ``group_norm_groups`` (8), ``dropout`` (0.1),
            An unrecognised key raises rather than being ignored. Input
            normalization is not a key: see :data:`PIXEL_SCALE`.

    Returns:
        A FEMNISTResNet on the CPU. The caller moves it to the run device.

    Raises:
        ValueError: If ``blocks_per_stage`` is not a non-string sequence, or if
            any value fails FEMNISTResNet's own validation.
    """

    values = dict(config or {})
    reject_unknown_model_keys(values, _KNOWN_KEYS, "femnist_resnet18")
    raw_blocks = values.get("blocks_per_stage", (2, 2, 2, 2))
    if not isinstance(raw_blocks, Sequence) or isinstance(raw_blocks, str | bytes):
        raise ValueError("blocks_per_stage must be a sequence")
    blocks = tuple(int(count) for count in raw_blocks)
    return FEMNISTResNet(
        input_channels=int(values.get("input_channels", 1)),
        base_channels=int(values.get("base_channels", 32)),
        num_classes=int(values.get("num_classes", 62)),
        blocks_per_stage=blocks,
        group_norm_groups=int(values.get("group_norm_groups", 8)),
        dropout=float(values.get("dropout", 0.1)),
    )
