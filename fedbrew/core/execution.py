"""The seam between what a round computes and how its clients are run.

An algorithm -- a ServerStrategy and a ClientUpdate rule -- decides what the
server and the clients compute. Three parts decide how a round gets it done,
and each can be replaced on its own:

- a ``ClientExecutor`` runs the round's sampled clients: it yields their
  ``FitResult``s, one per request, in request order, and reports each to a
  ``FitObserver``;
- an ``Aggregator`` folds those results into the server's new state;
- an ``Evaluator`` measures the new model on clients and on the central test
  set.

The references are ``SequentialExecutor`` and ``SequentialEvaluator``
(``fedbrew/core/loop.py``) and ``StreamingAggregator`` below, and the loop
runs, folds and measures through nothing else. They are what every run used
before the seam existed, call for call. A batched executor is held to them by
tolerance and never replaces them as the reference.

The stacked path. An executor that trains clients together may also offer
``fit_stacked``: the same results handed over a chunk at a time, each chunk
one ``StackedFitResults`` (``fedbrew/core/stacked_results.py``) whose states,
example counts and metrics are tensors over its clients. The loop takes it
when the executor offers it for the round and the aggregator takes it
(``aggregate_stacked``); otherwise the round goes through ``fit`` and
``aggregate``. A stacked result stands for exactly the ``FitResult`` s it
replaces: a server that folds one result at a time is handed them, built from
the stacks, and the observer writes each client's records from them as it
writes them from a result.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any, Protocol

from fedbrew.clients.base import ClientUpdate
from fedbrew.core.protocol import ClientInfo, EvalResult, FitRequest, FitResult, RoundInfo
from fedbrew.core.stacked_results import StackedFitResults, StackedResults
from fedbrew.data.dataset import FederatedDataset
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

    # Optional, for the stacked path:
    #
    #   def fitted_stack(self, stacked, seconds, done, total) -> None
    #
    # records every client of one StackedFitResults, in order, as ``fitted``
    # records each, ``done`` counting up to its last client and ``seconds``
    # the stack's. An observer without it is handed each client's result.


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

    # Optional, for the stacked path:
    #
    #   def fit_stacked(self, clients, requests, observer)
    #       -> Iterator[StackedFitResults] | None
    #
    # the same results, one StackedFitResults per chunk in request order,
    # each reported to the observer (``fitted_stack``) before it is yielded;
    # None, before anything runs, when the round's rules cannot stack them.


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

    # Optional, for the stacked path:
    #
    #   def aggregate_stacked(self, server, round_info, stacks) -> dict[str, Any]
    #
    # ``aggregate`` over an executor's ``fit_stacked``: the same fold of the
    # same results, handed over a chunk at a time.


class Evaluator(Protocol):
    """How the aggregated model is measured."""

    def evaluate_clients(
        self,
        clients: ClientPool,
        round_id: int,
        work: list[tuple[ClientInfo, list[str]]],
        server_payload: Mapping[str, Any],
        model_scope: str,
        on_progress: ProgressCallback | None,
    ) -> list[tuple[EvalResult, list[str]]]:
        """Evaluate each client on its splits, in ``work``'s order."""

    def evaluate_central(
        self,
        server: ServerStrategy,
        dataset: FederatedDataset,
    ) -> dict[str, float]:
        """The ``central_test_*`` metrics of the server's model."""

    def evaluate_grad_norm(
        self,
        server: ServerStrategy,
        dataset: FederatedDataset,
    ) -> dict[str, float]:
        """``grad_norm_sq`` of the server's model (``fedbrew/core/grad_norm.py``).

        Asked only on the rounds ``evaluation.grad_norm.every`` schedules.
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

    def aggregate_stacked(
        self,
        server: ServerStrategy,
        round_info: RoundInfo,
        stacks: Iterable[StackedFitResults],
    ) -> dict[str, Any]:
        """``aggregate`` of stacked results: the server's own stream over them.

        A server that folds a stack whole reads the stacks
        (``StackedResults.stacks``, ``FedAvgServer``); any other iterates the
        same results one by one, as ``aggregate`` hands them.
        """

        return server.aggregate_stream(round_info, StackedResults(stacks))
