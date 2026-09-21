"""One definition of the GroupNorm both vision models build.

``nn.GroupNorm`` requires ``num_channels % num_groups == 0``, but a single
``model.group_norm_groups`` has to serve every width in a network whose widths
are not all multiples of it. Both models carried a private copy of the same
four lines walking the request down to a divisor, and neither said it had:
``group_norm_groups: 8`` on the shipped OpenImage ShuffleNet builds **2** groups
over its 58-channel branches and **4** over its 116-channel ones, while the
config and ``run.json`` both said 8. P10-F32.

The backoff stays. Refusing a request it cannot honour everywhere would make 8
illegal for that model, and the only counts dividing every one of its widths
are 1 and 2 -- a different, worse model than the one that has been run. What
changes is that the request is kept on the module it built, so
:func:`group_norm_reductions` can report every width where the answer is not
the question, read off the model that was actually built rather than off a
second copy of the architecture.
"""

from __future__ import annotations

from torch import nn

#: Attribute :func:`group_norm` leaves on each module it builds. Plain int, so
#: it stays out of ``state_dict`` and out of everything that aggregates one.
REQUESTED_ATTRIBUTE = "requested_num_groups"


def honoured_groups(channels: int, requested_groups: int) -> int:
    """Return the largest divisor of ``channels`` not above ``requested_groups``.

    Args:
        channels: Width of the tensor to normalise.
        requested_groups: What ``model.group_norm_groups`` asked for.

    Returns:
        The group count GroupNorm can actually be built with: the request
        capped at ``channels`` and then walked down to a divisor. Always
        terminates, since 1 divides everything.

    Raises:
        ValueError: If either argument is not positive.
    """

    if requested_groups <= 0:
        raise ValueError("group_norm_groups must be positive")
    if channels <= 0:
        raise ValueError("GroupNorm needs a positive channel count")
    groups = min(channels, requested_groups)
    while channels % groups != 0:
        groups -= 1
    return groups


def group_norm(channels: int, requested_groups: int) -> nn.GroupNorm:
    """Build a GroupNorm over ``channels``, remembering what was requested."""

    norm = nn.GroupNorm(
        num_groups=honoured_groups(channels, requested_groups),
        num_channels=channels,
    )
    setattr(norm, REQUESTED_ATTRIBUTE, int(requested_groups))
    return norm


def group_norm_reductions(model: nn.Module) -> list[dict[str, int]]:
    """Report every width in ``model`` whose group request was not honoured.

    Args:
        model: Any built module. Layers not built by :func:`group_norm` carry
            no request and are skipped -- a GroupNorm constructed directly with
            the count it wanted has nothing to report.

    Returns:
        One record per distinct ``(channels, requested, groups)``, ordered by
        width, each naming how many layers it covers. Empty when every request
        was honoured, which is what the FEMNIST model's widths all give.
    """

    counted: dict[tuple[int, int, int], int] = {}
    for module in model.modules():
        if not isinstance(module, nn.GroupNorm):
            continue
        requested = getattr(module, REQUESTED_ATTRIBUTE, None)
        if not isinstance(requested, int) or requested == module.num_groups:
            continue
        key = (int(module.num_channels), requested, int(module.num_groups))
        counted[key] = counted.get(key, 0) + 1
    return [
        {"channels": channels, "requested": requested, "groups": groups, "layers": layers}
        for (channels, requested, groups), layers in sorted(counted.items())
    ]
