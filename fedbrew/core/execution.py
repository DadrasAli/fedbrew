"""The seam between what a round computes and how its clients are run.

An algorithm -- a ServerStrategy and a ClientUpdate rule -- decides what the
server and the clients compute. A ``ClientExecutor`` decides how a round's
sampled clients are run: it yields their ``FitResult``s, one per request, in
request order, and reports each to a ``FitObserver``.

The loop (``fedbrew/core/loop.py``) holds the reference executor,
``SequentialExecutor``, and runs clients through nothing else. It is what
every run used before the seam existed, call for call. A batched executor
is held to it by tolerance and never replaces it as the reference.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Protocol

from fedbrew.clients.base import ClientUpdate
from fedbrew.core.protocol import FitRequest, FitResult

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
