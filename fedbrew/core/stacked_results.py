"""Several clients' fit results handed over as one: the seam's stacked path.

A ``ClientExecutor`` yields one ``FitResult`` per client, and an
``Aggregator`` folds them one at a time (``fedbrew/core/execution.py``). An
executor that trained clients together can hand them over together instead:
one :class:`StackedFitResults` per chunk, whose states are the rows of the
chunk's stacked tensors and whose example counts and metrics are one tensor
each over the clients. It stands for exactly the ``FitResult`` s it replaces:
:meth:`StackedFitResults.result` builds each one, the same client id, count,
payload and metrics, so a server that folds results one at a time is handed
those (:class:`StackedResults`), and one that folds a stack whole
(``FedAvgServer``) does per chunk what it did per result.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from fedbrew.core.protocol import FitResult
from fedbrew.core.torch_utils import StackedRow, StateStack


@dataclass(slots=True)
class StackedFitResults:
    """Several clients' fit results of one round, as tensors over the clients.

    Clients are in request order, and position ``p`` is client
    ``client_ids[p]``. Its model state is a row of one of ``states``' stacks:
    each entry is a stack and, per row of it, the position of that row's
    client. ``metrics`` holds one float64 tensor per name, in the order the
    names are first reported; where some client does not report a name,
    ``reported`` holds which do, and that client's value is never read.
    ``payload`` is what every client's result payload holds beside its model
    state -- the scope and metadata, each client's its own copy.
    """

    round_id: int
    client_ids: list[str]
    num_examples: Tensor
    states: list[tuple[StateStack, list[int]]]
    metrics: dict[str, Tensor]
    payload: dict[str, Any]
    reported: dict[str, Tensor] = field(default_factory=dict)
    _rows: list[dict[str, float]] | None = None

    def __len__(self) -> int:
        return len(self.client_ids)

    def counts(self) -> list[int]:
        """Each client's example count, as its FitResult's ``num_examples``."""

        return [int(count) for count in self.num_examples.tolist()]

    def metric_columns(self) -> tuple[dict[str, list[float]], dict[str, list[bool]]]:
        """Each name's values over the clients, and, for a name not all report, which do."""

        return (
            {name: values.tolist() for name, values in self.metrics.items()},
            {name: mask.tolist() for name, mask in self.reported.items()},
        )

    def metric_rows(self) -> list[dict[str, float]]:
        """Each client's metrics, as its FitResult holds them; built once."""

        if self._rows is None:
            columns, reported = self.metric_columns()
            if not columns:
                self._rows = [{} for _ in self.client_ids]
            elif not reported:
                names = list(columns)
                self._rows = [
                    dict(zip(names, row, strict=True))
                    for row in zip(*columns.values(), strict=True)
                ]
            else:
                self._rows = [
                    {
                        name: values[position]
                        for name, values in columns.items()
                        if name not in reported or reported[name][position]
                    }
                    for position in range(len(self.client_ids))
                ]
        return self._rows

    def rows(self) -> list[StackedRow]:
        """Each client's model state, a row of its stack, in request order."""

        placed: list[StackedRow | None] = [None] * len(self.client_ids)
        for stack, positions in self.states:
            for row, position in enumerate(positions):
                placed[position] = stack.row(row)
        return placed  # type: ignore[return-value]

    def results(self) -> Iterator[FitResult]:
        """The FitResult of every client, in request order."""

        rows = self.metric_rows()
        counts = self.counts()
        for position, state in enumerate(self.rows()):
            yield self.result(position, state, rows[position], counts[position])

    def result(
        self,
        position: int,
        state: Mapping[str, Tensor],
        metrics: Mapping[str, float],
        num_examples: int,
    ) -> FitResult:
        """One client's FitResult: what the executor's per-client path returns for it."""

        payload: dict[str, Any] = {"model_state": state}
        for key, value in self.payload.items():
            payload[key] = dict(value) if isinstance(value, dict) else value
        return FitResult(
            round_id=self.round_id,
            client_id=self.client_ids[position],
            num_examples=num_examples,
            payload=payload,
            metrics=dict(metrics),
        )


class MetricColumns:
    """Per-client metrics gathered name by name, in the order each is first reported."""

    __slots__ = ("size", "values", "reported")

    def __init__(self, size: int) -> None:
        self.size = size
        self.values: dict[str, list[float]] = {}
        self.reported: dict[str, list[bool]] = {}

    def put(self, name: str, positions: Sequence[int], values: Sequence[float]) -> None:
        """Clients ``positions`` report ``values`` under ``name``."""

        column = self.values.get(name)
        if column is None:
            column = self.values[name] = [0.0] * self.size
            self.reported[name] = [False] * self.size
        mask = self.reported[name]
        for position, value in zip(positions, values, strict=True):
            column[position] = value
            mask[position] = True

    def tensors(self) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """The columns as float64 tensors, and the masks of the names not every client reports."""

        metrics = {
            name: torch.tensor(values, dtype=torch.float64) for name, values in self.values.items()
        }
        reported = {
            name: torch.tensor(mask, dtype=torch.bool)
            for name, mask in self.reported.items()
            if not all(mask)
        }
        return metrics, reported


class StackedResults:
    """A round's stacked results, as a server folds them.

    Iterated, it is every client's FitResult in request order -- what a server
    that folds one result at a time reads, the same results the per-client
    path hands it. :meth:`stacks` gives the stacks themselves to a server that
    folds a stack whole. Either way the round is consumed once, a stack at a
    time, so no more than a chunk is held.
    """

    __slots__ = ("_stacks", "_taken")

    def __init__(self, stacks: Iterable[StackedFitResults]) -> None:
        self._stacks = stacks
        self._taken = False

    def stacks(self) -> Iterator[StackedFitResults]:
        """The stacks, each once."""

        if self._taken:
            raise RuntimeError("a round's stacked results are consumed once")
        self._taken = True
        return iter(self._stacks)

    def __iter__(self) -> Iterator[FitResult]:
        for stacked in self.stacks():
            yield from stacked.results()
