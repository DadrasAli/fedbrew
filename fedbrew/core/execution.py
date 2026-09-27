"""The seam between what a round computes and how its clients are run.

An algorithm -- a ServerStrategy and a ClientUpdate rule -- decides what the
server and the clients compute. Two parts decide how a round gets it done,
and each can be replaced on its own:

- a ``ClientExecutor`` runs the round's sampled clients: it yields their
  ``FitResult``s, one per request, in request order, and reports each to a
  ``FitObserver``;
- an ``Aggregator`` folds those results into the server's new state.

The references are ``SequentialExecutor`` (``fedbrew/core/loop.py``) and
``StreamingAggregator`` below, and the loop runs and folds through nothing
else. They are what every run used before the seam existed, call for call. A
batched executor is held to them by tolerance and never replaces them as the
reference.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any, Protocol

from fedbrew.clients.base import ClientUpdate
from fedbrew.core.protocol import FitRequest, FitResult, RoundInfo
from fedbrew.servers.base import ServerStrategy

#: One client update for every client, or a mapping from client id to each
#: client's own -- LazyClientPool is one.
ClientPool = ClientUpdate | Mapping[str, ClientUpdate]

#: (round_id, done, total, phase), done 1-indexed, phase "fit" or "client_eval".
ProgressCallback = Callable[[int, int, int, str], None]


class FitObserver(Protocol):
    """What an executor reports for each client it has run, before yielding its result.

    One call per result, in the order the results are yielded, so the loop's
    records -- the per-client update row, the example count, the fit time and
    the progress footer -- are written exactly as they were when the loop ran
    the clients itself. An executor that ran clients together reports each
    client's share of the time.
    """

    def fitted(self, result: FitResult, seconds: float, done: int, total: int) -> None:
        """Record one client's result; ``done`` counts from 1 up to ``total``."""


class ClientExecutor(Protocol):
    """How a round's sampled clients are run."""

    def fit(
        self,
        clients: ClientPool,
        requests: Sequence[FitRequest],
        observer: FitObserver,
    ) -> Iterator[FitResult]:
        """Yield one result per request, in request order, reporting each to ``observer``.

        A generator, pulled by the aggregator: a result that has been folded
        in is no longer referenced, so peak memory does not grow with the
        number of sampled clients.
        """


class Aggregator(Protocol):
    """How a round's results become the server's new state."""

    def aggregate(
        self,
        server: ServerStrategy,
        round_info: RoundInfo,
        results: Iterable[FitResult],
    ) -> dict[str, Any]:
        """Fold ``results`` into ``server`` and return the next broadcast payload.

        Raises:
            NonFiniteStateError: If the round's aggregate is not finite; the
                server's state is left as it was.
        """


class StreamingAggregator:
    """The reference Aggregator: the server folds each result as it is pulled.

    Every strategy's ``aggregate_stream`` consumes the results one at a time,
    so a result is folded, and dropped, before the next client runs.
    """

    def aggregate(
        self,
        server: ServerStrategy,
        round_info: RoundInfo,
        results: Iterable[FitResult],
    ) -> dict[str, Any]:
        return server.aggregate_stream(round_info, results)
