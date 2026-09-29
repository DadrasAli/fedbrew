"""``client.sampling``: how a client's training loader draws its rows.

``without_replacement``, the default, is the task's own loader: each pass an
epoch over the train split, in order or permuted (``client.train_shuffle``).

``with_replacement`` makes every training pass **one** batch of
``client.batch_size`` rows drawn independently and uniformly with replacement
from the train split -- ``torch.randint(rows, (batch_size,))`` on a
``torch.Generator`` seeded once per loader, with the seed the task's loader
would have been handed (``dataloader_seed``) -- so successive passes, clients,
rounds and phases draw independently, and the batch may hold more rows than
the split. It is the iid stochastic oracle of the optimisation literature:
``update_mode: single_batch`` takes K iid batches in K local iterations,
``sequential_epoch`` (one pass is one batch) does too, and a pass of
``frozen_batch_gradients`` is one batch.

It is general, not a task's: a batch is the task's rows at the drawn indices,
``split_rows(data)`` gathered with ``index_select`` -- which a
``BatchableTask`` declares a batch of its loader to be -- so any task that
gives its rows can be sampled so, and one that cannot is refused when its
clients are built. The same draws are declared to the batched executor as an
iid ``LoaderOrder`` (``replacement=True``), which plans every client's batches
in one ``randint`` call per client from the same seed, so the sequential,
batched and resident paths train on the same rows.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
from torch import Tensor

from fedbrew.tasks.base import LoaderOrder

WITHOUT_REPLACEMENT = "without_replacement"
WITH_REPLACEMENT = "with_replacement"

#: The values ``client.sampling`` takes; the first is the default.
SAMPLINGS = (WITHOUT_REPLACEMENT, WITH_REPLACEMENT)


class ReplacementBatches:
    """The row indices of a with-replacement loader over ``rows`` rows; re-iterable.

    Each iteration yields one batch of ``batch_size`` indices from one
    generator, seeded once here, so iterations differ and a loader rebuilt from
    the same seed draws the same batches. Unseeded, its seed is drawn from
    torch's global generator, as a task's unseeded loader draws its order.
    """

    def __init__(self, rows: int, batch_size: int, seed: int | None) -> None:
        if rows <= 0:
            raise ValueError("a with-replacement loader needs at least one row to draw")
        self.rows, self.batch_size = int(rows), max(1, int(batch_size))
        if seed is None:
            seed = int(torch.randint(0, 2**62, (1,)).item())
        self.generator = torch.Generator()
        self.generator.manual_seed(int(seed))

    def __iter__(self) -> Iterator[Tensor]:
        yield torch.randint(0, self.rows, (self.batch_size,), generator=self.generator)

    def __len__(self) -> int:
        return 1


class ReplacementLoader:
    """A training loader over a split's rows, one with-replacement batch per pass.

    ``rows`` is the task's ``split_rows(data)``; a batch is each of its tensors
    at the drawn indices, which is what the task's loader yields for those rows.
    """

    def __init__(self, rows: tuple[Tensor, ...], batch_size: int, seed: int | None) -> None:
        self.rows = tuple(rows)
        self.batches = ReplacementBatches(len(self.rows[0]), batch_size, seed)

    def __iter__(self) -> Iterator[tuple[Tensor, ...]]:
        for index in self.batches:
            yield tuple(tensor.index_select(0, index.to(tensor.device)) for tensor in self.rows)

    def __len__(self) -> int:
        return 1


def replacement_order(rows: int, batch_size: int, seed: int | None) -> LoaderOrder:
    """What a with-replacement loader over ``rows`` rows yields, declared (``LoaderOrder``)."""

    return LoaderOrder(
        rows=int(rows),
        batch_size=max(1, int(batch_size)),
        shuffle=True,
        drop_last=False,
        seed=None if seed is None else int(seed),
        per_epoch=False,
        replacement=True,
    )


def split_row_count(task: Any, data: Any) -> int:
    """How many rows a split holds, as the task gives them (``split_rows``)."""

    return len(task.split_rows(data)[0])
