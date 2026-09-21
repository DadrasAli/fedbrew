"""ShuffleNet-V2 variant for federated OpenImage classification.

Two deliberate departures from the reference architecture, both forced by the
federated setting rather than by taste:

**GroupNorm instead of BatchNorm.** BatchNorm keeps running mean/variance
buffers that are averaged across clients like weights, but under a non-IID
partition each client's statistics describe a different distribution, so the
average describes none of them. This is the same reason the FEMNIST model here
uses GroupNorm. It costs a little accuracy centrally and avoids an artifact
that would otherwise be confounded with the optimizer differences under study.

``group_norm_groups`` is one number for widths that are not all multiples of
it, so most of this network does not get the count the config asks for. At the
shipped settings, ``8`` builds:

    channels   24    58    116   232   1024
    groups      8     2      4     8      8
    layers      2    13     26    14      1

-- 39 of the 56 normalisation layers at 2 or 4 groups.
:func:`~fedbrew.models.group_norm.group_norm` keeps the request on each module
it builds so ``run.json`` records this under
``federated_model_state.group_norm_reductions`` rather than only echoing the
8 that was asked for. The alternative, refusing a request that cannot be
honoured everywhere, would make 8 illegal here: the only counts dividing every
width above are 1 and 2. The FEMNIST model's widths are all multiples of 8, so
nothing is reduced there and the key is silent.

**A stem that does not throw away a low-resolution input.** The reference stem
is a stride-2 conv followed by a stride-2 max pool, which is right for 224x224
but reduces a 64x64 input to 16x16 before the first stage and 2x2 by the last.
Dropping the pool keeps the final feature map at 4x4.

ShuffleNet-V2 rather than another ResNet because cross-device FL is premised on
clients being mobile-class devices, and this is the architecture family the
FedScale OpenImage benchmark uses.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from fedbrew.models.config_keys import reject_unknown_model_keys
from fedbrew.models.group_norm import group_norm

#: Input normalization, fixed rather than configurable -- the same [0, 255]
#: uint8 contract fedbrew.models.femnist_resnet states, for the same reason:
#: these describe how the shards were written, not a tunable. Restated here
#: rather than imported so the two models can diverge if their datasets ever do.
PIXEL_SCALE = 255.0
PIXEL_MEAN = 0.5
PIXEL_STD = 0.5


def _channel_shuffle(inputs: Tensor, groups: int) -> Tensor:
    """Interleave channels across groups so the two branches can mix.

    Args:
        inputs: Float tensor of shape (N, C, H, W).
        groups: Number of groups to interleave across. ``C`` must be divisible
            by it.

    Returns:
        Float tensor of the same shape (N, C, H, W), channels permuted so that
        channel ``g * (C // groups) + i`` moves to ``i * groups + g``. Without
        this the concatenated halves of a ShuffleUnit would never exchange
        information, since each unit only ever transforms one of them.

    Raises:
        ValueError: If ``C`` is not divisible by ``groups``.
    """

    batch, channels, height, width = inputs.shape
    if channels % groups != 0:
        raise ValueError(f"channels {channels} not divisible by groups {groups}")
    inputs = inputs.view(batch, groups, channels // groups, height, width)
    inputs = torch.transpose(inputs, 1, 2).contiguous()
    return inputs.view(batch, channels, height, width)


class ShuffleUnit(nn.Module):
    """One ShuffleNet-V2 unit, stride 1 (split) or stride 2 (downsample).

    **``output_channels`` must be even**, and is checked at construction. Both
    branches are built at ``output_channels // 2`` and concatenated, so an odd
    request would yield ``output_channels - 1`` channels -- a mismatch that
    surfaces as a shape error in whatever consumes the output, arbitrarily far
    downstream. It is refused here instead, naming ``model.stage_channels``.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        stride: int,
        group_norm_groups: int,
    ) -> None:
        """Build the unit's one or two branches.

        Args:
            input_channels: Channels of the incoming feature map.
            output_channels: Channels of the outgoing feature map. Must be
                even -- see the class docstring.
            stride: 1 for the split/identity form, which requires
                ``input_channels == output_channels`` and halves the input to
                feed one branch; 2 for the downsampling form, which feeds the
                whole input to both branches and halves H and W.
            group_norm_groups: Requested GroupNorm groups, reduced per tensor
                to a divisor of its channel count.

        Raises:
            ValueError: If ``stride`` is not 1 or 2, or if a stride-1 unit is
                asked to change the channel count.
        """

        super().__init__()
        if stride not in (1, 2):
            raise ValueError("ShuffleUnit stride must be 1 or 2")
        # Both branches are built at output_channels // 2 and concatenated, so
        # an odd width silently yields output_channels - 1. Nothing downstream
        # notices until a later convolution is handed a feature map one channel
        # narrower than the one it was sized for -- and for the last stage, not
        # until the head. Refusing here names the config key and the value
        # instead.
        if output_channels % 2 != 0:
            raise ValueError(
                f"ShuffleUnit output_channels must be even, got {output_channels}. "
                "Set an even value in model.stage_channels: the unit splits into "
                "two branches of output_channels // 2 and concatenates them, so "
                f"an odd width would silently build {output_channels - 1} channels."
            )
        self.stride = stride
        branch_channels = output_channels // 2

        if stride == 1:
            if input_channels != output_channels:
                raise ValueError("stride-1 ShuffleUnit requires matching input/output channels")
            # The unit splits its input in half and only transforms one half,
            # so the right branch starts from half the channels.
            right_in = input_channels // 2
            self.left: nn.Module = nn.Identity()
        else:
            right_in = input_channels
            self.left = nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    input_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=input_channels,
                    bias=False,
                ),
                group_norm(input_channels, group_norm_groups),
                nn.Conv2d(input_channels, branch_channels, kernel_size=1, bias=False),
                group_norm(branch_channels, group_norm_groups),
                nn.GELU(),
            )

        self.right = nn.Sequential(
            nn.Conv2d(right_in, branch_channels, kernel_size=1, bias=False),
            group_norm(branch_channels, group_norm_groups),
            nn.GELU(),
            nn.Conv2d(
                branch_channels,
                branch_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=branch_channels,
                bias=False,
            ),
            group_norm(branch_channels, group_norm_groups),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=1, bias=False),
            group_norm(branch_channels, group_norm_groups),
            nn.GELU(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply the unit and shuffle its two concatenated halves.

        Args:
            inputs: Float tensor of shape (N, input_channels, H, W).

        Returns:
            Float tensor of shape (N, output_channels, H, W) at stride 1, or
            (N, output_channels, ceil(H / 2), ceil(W / 2)) at stride 2.
        """

        if self.stride == 1:
            left, right = inputs.chunk(2, dim=1)
            out = torch.cat((left, self.right(right)), dim=1)
        else:
            out = torch.cat((self.left(inputs), self.right(inputs)), dim=1)
        return _channel_shuffle(out, 2)


class OpenImageShuffleNet(nn.Module):
    """ShuffleNet-V2 with GroupNorm, sized for low-resolution FL inputs.

    **Input units.** ``forward`` expects raw pixel values in [0, 255] as
    floats. It divides by :data:`PIXEL_SCALE` and standardises by
    :data:`PIXEL_MEAN` / :data:`PIXEL_STD` itself, so pre-normalised input
    trains on values 255x too small. Same contract as
    :class:`~fedbrew.models.femnist_resnet.FEMNISTResNet`.

    **Downsampling.** The stride-2 stem plus one stride-2 unit per stage
    reduces H and W by ``2 ** (1 + len(stage_channels))`` -- 16x at the shipped
    three stages, so a 64x64 input reaches the head as 4x4. Setting
    ``stem_pool: true`` restores the reference stem's extra max pool and halves
    that again, which is right for 224x224 and throws away most of a 64x64
    input. An AdaptiveAvgPool2d(1) collapses whatever remains, so no spatial
    size is structurally required -- but a size that reaches the head below 1x1
    will have been reduced to nothing well before then.

    At the shipped defaults the model holds ~1.86M parameters.
    """

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 596,
        stage_channels: Sequence[int] = (116, 232, 464),
        blocks_per_stage: Sequence[int] = (4, 8, 4),
        stem_channels: int = 24,
        final_channels: int = 1024,
        group_norm_groups: int = 8,
        stem_pool: bool = False,
        dropout: float = 0.1,
    ) -> None:
        """Build the stem, the shuffle stages, the 1x1 head and the classifier.

        Args:
            input_channels: Channels per image; 3 for RGB.
            num_classes: Output logits. 596 for the FedScale OpenImage label
                set.
            stage_channels: Output width of each stage. Every entry must be
                even -- an odd one is refused by :class:`ShuffleUnit`. Must be
                the same length as ``blocks_per_stage``.
            blocks_per_stage: Units per stage. The first unit of each stage is
                stride 2; the rest are stride 1.
            stem_channels: Width after the stride-2 stem convolution.
            final_channels: Width of the 1x1 head convolution, and the input
                width of the classifier.
            group_norm_groups: Requested GroupNorm groups, reduced per tensor
                to a divisor of its channel count.
            stem_pool: Add the reference stem's stride-2 max pool. Off by
                default -- see the class and module docstrings.
            dropout: Drop probability applied to the pooled features before the
                classifier, in training mode only.

        Raises:
            ValueError: If ``stage_channels`` and ``blocks_per_stage`` differ in
                length, or if a ShuffleUnit rejects its arguments.
        """

        super().__init__()
        if len(stage_channels) != len(blocks_per_stage):
            raise ValueError("stage_channels and blocks_per_stage must align")

        stem: list[nn.Module] = [
            nn.Conv2d(
                input_channels,
                stem_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            group_norm(stem_channels, group_norm_groups),
            nn.GELU(),
        ]
        if stem_pool:
            stem.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.stem = nn.Sequential(*stem)

        stages: list[nn.Module] = []
        channels = stem_channels
        for stage_out, num_blocks in zip(stage_channels, blocks_per_stage, strict=True):
            blocks: list[nn.Module] = [ShuffleUnit(channels, stage_out, 2, group_norm_groups)]
            blocks += [
                ShuffleUnit(stage_out, stage_out, 1, group_norm_groups)
                for _ in range(num_blocks - 1)
            ]
            stages.append(nn.Sequential(*blocks))
            channels = stage_out
        self.stages = nn.Sequential(*stages)

        self.head = nn.Sequential(
            nn.Conv2d(channels, final_channels, kernel_size=1, bias=False),
            group_norm(final_channels, group_norm_groups),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(final_channels, num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        """Compute class logits for raw uint8-scaled pixels.

        Args:
            inputs: Float tensor of shape (N, input_channels, H, W) holding
                **raw pixel values in [0, 255]**. Normalisation happens here,
                not in the dataset.

        Returns:
            Float tensor of shape (N, num_classes): raw logits, not
            probabilities.
        """

        normalized = inputs / PIXEL_SCALE
        normalized = (normalized - PIXEL_MEAN) / PIXEL_STD
        features = self.head(self.stages(self.stem(normalized)))
        pooled = torch.flatten(self.pool(features), 1)
        return self.classifier(self.dropout(pooled))


#: Every key build_openimage_shufflenet below reads.
_KNOWN_KEYS = frozenset(
    {
        "input_channels",
        "stage_channels",
        "blocks_per_stage",
        "stem_channels",
        "final_channels",
        "group_norm_groups",
        "stem_pool",
        "dropout",
    }
)


def build_openimage_shufflenet(
    config: Mapping[str, Any] | None = None,
) -> OpenImageShuffleNet:
    """Build the FL-friendly ShuffleNet-V2 variant for OpenImage.

    Args:
        config: ``model`` block keys, all optional: ``input_channels`` (3),
            ``num_classes`` (596), ``stage_channels`` ([116, 232, 464]),
            ``blocks_per_stage`` ([4, 8, 4]), ``stem_channels`` (24),
            ``final_channels`` (1024), ``group_norm_groups`` (8), ``stem_pool``
            (False), ``dropout`` (0.1). An unrecognised key raises rather than
            being ignored. Input normalization is not a key: see
            :data:`PIXEL_SCALE`.

    Returns:
        An OpenImageShuffleNet on the CPU. The caller moves it to the run
        device.

    Raises:
        ValueError: If ``stage_channels`` or ``blocks_per_stage`` is not a
            non-string sequence, or if the model rejects the values.
    """

    values = dict(config or {})
    reject_unknown_model_keys(values, _KNOWN_KEYS, "openimage_shufflenet")

    def _sequence(key: str, default: tuple[int, ...]) -> tuple[int, ...]:
        raw = values.get(key, default)
        if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
            raise ValueError(f"{key} must be a sequence")
        return tuple(int(item) for item in raw)

    return OpenImageShuffleNet(
        input_channels=int(values.get("input_channels", 3)),
        num_classes=int(values.get("num_classes", 596)),
        stage_channels=_sequence("stage_channels", (116, 232, 464)),
        blocks_per_stage=_sequence("blocks_per_stage", (4, 8, 4)),
        stem_channels=int(values.get("stem_channels", 24)),
        final_channels=int(values.get("final_channels", 1024)),
        group_norm_groups=int(values.get("group_norm_groups", 8)),
        stem_pool=bool(values.get("stem_pool", False)),
        dropout=float(values.get("dropout", 0.1)),
    )
