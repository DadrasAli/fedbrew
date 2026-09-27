"""Every client's batch order for a round, planned together, exactly as each loader draws it.

The batched executor needs each sampled client's batches, as row indices,
before any client runs. Replaying each client's loader for them -- building
it, iterating it on row numbers, cutting and collecting its batches in Python
-- was most of the planning a round did at 1000 clients. This module plans
them for every client together from what each task declares about its loader
(:class:`fedbrew.tasks.base.LoaderOrder`):

- the loaders' seeds, derived as ``dataloader_seed`` derives each one, with
  the hash of the fields every client shares taken once
  (:func:`fedbrew.core.seeding.dataloader_seeds`);
- each shuffled loader's permutations, drawn by the calls the loader itself
  makes on a ``torch.Generator`` seeded as the loader seeds it: a torch
  ``DataLoader`` (and ``_DeviceTensorBatches``, which reproduces it) draws,
  per epoch, one int64 base seed, then the sampler's ``randperm`` and, once
  the epoch's last batch has been taken, one more ``randperm`` it discards;
  the linear examples' loaders draw one ``randperm`` when they are built and
  yield that order every time they are iterated. The draws are torch's own,
  so the orders are the loaders' by construction;
- each epoch's batches, cut and dropped as the loader cuts and drops them,
  the batches each applied update consumes, and every client's batch at every
  step as row indices, for all clients at once (:func:`plan_orders`).

``tests/test_batch_orders.py`` checks every order against the loader's own,
iterated as each update rule's loop iterates it, for every update mode,
shuffle setting, ``drop_last`` and ``max_local_steps``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from fedbrew.tasks.base import LoaderOrder

_LONG = torch.int64


# ---------------------------------------------------------------------------
# A round's orders
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LocalLoop:
    """Which batches a client's local update consumes, of its loader's epochs.

    ``epochs`` iterations of the loader, each one full epoch, or, with
    ``single_batch``, ``epochs`` batches taken one at a time from the epochs
    in turn. ``per_update``: ``"batch"`` makes each batch one applied update,
    ``"epoch"`` each epoch; stopped after ``max_updates`` updates when set.
    """

    epochs: int
    per_update: str = "batch"
    single_batch: bool = False
    max_updates: int | None = None


@dataclass(slots=True)
class RoundOrders:
    """Every client's batches, stacked: what :func:`plan_orders` returns.

    ``indices[c, t, :lengths[c, t]]`` are client ``c``'s ``t``-th batch, as row
    indices, and past its length the batch's own rows; past ``steps[c]`` there
    are none. ``structure[c]`` is how many batches each of its applied updates
    consumes. ``contiguous[c]`` says its order is unshuffled, so its batch
    ``t`` is its rows from ``starts[c, t]`` on. All CPU int64 tensors.
    """

    indices: Tensor
    lengths: Tensor
    starts: Tensor
    steps: Tensor
    structure: list[tuple[int, ...]]
    contiguous: Tensor

    @classmethod
    def from_updates(cls, clients: Sequence[Sequence[Sequence[Tensor]]]) -> RoundOrders:
        """Orders given as each client's updates, each a list of index batches (a replay)."""

        flat = [[batch for update in updates for batch in update] for updates in clients]
        steps = torch.tensor([len(batches) for batches in flat], dtype=_LONG)
        longest = max((len(batches) for batches in flat), default=0)
        widest = max((len(batch) for batches in flat for batch in batches), default=1)
        indices = torch.zeros((len(flat), longest, max(widest, 1)), dtype=_LONG)
        lengths = torch.zeros((len(flat), longest), dtype=_LONG)
        for client, batches in enumerate(flat):
            for step, batch in enumerate(batches):
                values = batch.to(dtype=_LONG)
                lengths[client, step] = len(values)
                if len(values):
                    indices[client, step, : len(values)] = values
                    indices[client, step, len(values) :] = values[0]
        return cls(
            indices=indices,
            lengths=lengths,
            starts=torch.zeros_like(lengths),
            steps=steps,
            structure=[tuple(len(update) for update in updates) for updates in clients],
            contiguous=torch.zeros(len(flat), dtype=torch.bool),
        )

    @classmethod
    def merge(
        cls, parts: Sequence[RoundOrders], groups: Sequence[Sequence[int]], total: int
    ) -> RoundOrders:
        """``parts[k]``'s clients placed at ``groups[k]``'s positions of ``total`` clients."""

        if len(parts) == 1 and list(groups[0]) == list(range(total)):
            return parts[0]
        longest = max((part.lengths.shape[1] for part in parts), default=0)
        widest = max((part.indices.shape[2] for part in parts), default=1)
        indices = torch.zeros((total, longest, widest), dtype=_LONG)
        lengths = torch.zeros((total, longest), dtype=_LONG)
        starts = torch.zeros((total, longest), dtype=_LONG)
        steps = torch.zeros(total, dtype=_LONG)
        contiguous = torch.zeros(total, dtype=torch.bool)
        structure: list[tuple[int, ...]] = [()] * total
        for part, group in zip(parts, groups, strict=True):
            where = torch.tensor(list(group), dtype=_LONG)
            count, width = part.lengths.shape[1], part.indices.shape[2]
            indices[where, :count, :width] = part.indices
            if width < widest:
                indices[where, :count, width:] = part.indices[:, :, :1]
            lengths[where, :count] = part.lengths
            starts[where, :count] = part.starts
            steps[where] = part.steps
            contiguous[where] = part.contiguous
            for position, client in enumerate(group):
                structure[client] = part.structure[position]
        return cls(indices, lengths, starts, steps, structure, contiguous)


def batches_per_epoch(order: LoaderOrder) -> int:
    """How many batches one epoch of ``order``'s loader yields."""

    whole, rest = divmod(order.rows, order.batch_size)
    if not order.drop_last or (order.keep_single_batch and whole + (rest > 0) <= 1):
        return whole + (rest > 0)
    return whole


def plan_orders(
    orders: Sequence[LoaderOrder],
    loops: Sequence[LocalLoop],
    seeds: Sequence[int | None] | None = None,
) -> RoundOrders:
    """Each client's batches for its local loop, from its loader's declared order.

    ``seeds[c]``, or ``orders[c].seed`` without them, is the seed client
    ``c``'s loader is seeded with; ``loops[c]`` which of its batches the
    update consumes. A client whose loader yields no batch gets no steps, and
    the caller refuses it in its rule's words.
    """

    clients = len(orders)
    rows = torch.tensor([order.rows for order in orders], dtype=_LONG)
    sizes = torch.tensor([order.batch_size for order in orders], dtype=_LONG)
    drop = torch.tensor([order.drop_last for order in orders], dtype=torch.bool)
    keep = torch.tensor([order.keep_single_batch for order in orders], dtype=torch.bool)
    whole = torch.div(rows, sizes, rounding_mode="floor")
    every = whole + (rows % sizes > 0).to(_LONG)
    per_epoch = torch.where(~drop | (keep & (every <= 1)), every, whole)
    counts = per_epoch.tolist()
    step_counts: list[int] = []
    structure: list[tuple[int, ...]] = []
    # Clients share a loop and an epoch's batch count far more often than
    # not, so each distinct pair is worked out once.
    known: dict[tuple[LocalLoop, int], tuple[tuple[int, ...], int]] = {}
    for loop, count in zip(loops, counts, strict=True):
        held = known.get((loop, count))
        if held is None:
            held = known[(loop, count)] = _loop_structure(loop, count)
        structure.append(held[0])
        step_counts.append(held[1])
    steps = torch.tensor(step_counts, dtype=_LONG)
    epochs_used = torch.div(-steps, torch.clamp(per_epoch, min=1), rounding_mode="floor").neg()

    longest_steps = max(step_counts, default=0)
    widest = int(sizes.max()) if clients else 1
    # Batch t of client c is batch t % per_epoch of epoch t // per_epoch.
    step = torch.arange(longest_steps, dtype=_LONG).unsqueeze(0)
    safe = torch.clamp(per_epoch, min=1).unsqueeze(1)
    epoch = torch.div(step, safe, rounding_mode="floor")
    starts = (step % safe) * sizes.unsqueeze(1)
    lengths = torch.minimum(torch.clamp(rows.unsqueeze(1) - starts, min=0), sizes.unsqueeze(1))
    lengths = torch.where(step < steps.unsqueeze(1), lengths, 0)

    shuffled = torch.tensor([order.shuffle for order in orders], dtype=torch.bool)
    positions = starts.unsqueeze(2) + torch.arange(widest, dtype=_LONG).view(1, 1, -1)
    positions = torch.minimum(positions, torch.clamp(rows - 1, min=0).view(-1, 1, 1))
    indices = positions.clone()
    if bool(shuffled.any()):
        if seeds is None:
            seeds = [order.seed for order in orders]
        _shuffle_positions(orders, seeds, shuffled, rows, epoch, epochs_used, positions, indices)
    return RoundOrders(
        indices=indices,
        lengths=lengths,
        starts=starts,
        steps=steps,
        structure=structure,
        contiguous=~shuffled,
    )


def _loop_structure(loop: LocalLoop, count: int) -> tuple[tuple[int, ...], int]:
    """The batches each update takes, and their total, for epochs of ``count`` batches."""

    if count == 0:
        return (), 0
    if loop.per_update == "epoch":
        updates = loop.epochs
        if loop.max_updates is not None:
            updates = min(updates, loop.max_updates)
        return (count,) * updates, updates * count
    total = loop.epochs if loop.single_batch else loop.epochs * count
    if loop.max_updates is not None:
        total = min(total, loop.max_updates)
    return (1,) * total, total


def _shuffle_positions(
    orders: Sequence[LoaderOrder],
    all_seeds: Sequence[int | None],
    shuffled: Tensor,
    rows: Tensor,
    epoch: Tensor,
    epochs_used: Tensor,
    positions: Tensor,
    indices: Tensor,
) -> None:
    """Map each shuffled client's positions through its epochs' permutations, in place."""

    chosen = torch.nonzero(shuffled).view(-1)
    picked = chosen.tolist()
    counts = torch.clamp(epochs_used[chosen], min=1).tolist()
    drawn: list[Tensor] = []
    first: list[int] = []
    # One generator, re-seeded for each loader: manual_seed sets the whole
    # state, so it is each loader's fresh generator in turn.
    generator = torch.Generator()
    scratch = torch.empty((), dtype=_LONG)
    for client, epochs in zip(picked, counts, strict=True):
        seed = all_seeds[client]
        if seed is None:
            raise ValueError("a shuffled loader's order is drawn from its own seed")
        first.append(len(drawn))
        generator.manual_seed(int(seed))
        drawn.extend(_permutations(orders[client], generator, scratch, epochs))
    # Every permutation end to end; client c's epoch e is permutation
    # first[c] + e, or its only one for a loader that permutes once, and its
    # position p is element starts[that] + p.
    joined = torch.cat(drawn)
    starts = torch.zeros(len(drawn), dtype=_LONG)
    if len(drawn) > 1:
        starts[1:] = torch.tensor([len(permutation) for permutation in drawn[:-1]]).cumsum(0)
    per_loader = torch.tensor([not orders[client].per_epoch for client in picked])
    last = torch.tensor(counts, dtype=_LONG) - 1
    which = torch.minimum(epoch[chosen], last.unsqueeze(1))
    which = torch.where(per_loader.unsqueeze(1), 0, which)
    which = which + torch.tensor(first, dtype=_LONG).unsqueeze(1)
    indices[chosen] = joined[starts[which].unsqueeze(2) + positions[chosen]]


def _permutations(
    order: LoaderOrder, generator: torch.Generator, scratch: Tensor, epochs: int
) -> list[Tensor]:
    """The permutations ``order``'s loader draws over ``epochs`` iterations from ``generator``."""

    if not order.per_epoch:
        return [torch.randperm(order.rows, generator=generator)]
    drawn = []
    for number in range(epochs):
        # The DataLoader iterator's base seed, an int64 draw, then the
        # sampler's order.
        scratch.random_(generator=generator)
        drawn.append(torch.randperm(order.rows, generator=generator))
        if number + 1 < epochs:
            # RandomSampler draws one more permutation after the epoch's
            # last batch, and discards it.
            torch.randperm(order.rows, generator=generator)
    return drawn
