"""A round runs, folds and measures through the executor seam, and nothing else.

``run_fl_loop`` takes a ClientExecutor, an Aggregator and an Evaluator
(``fedbrew/core/execution.py``); None is the reference for each, which is
what every run used before the seam. What is pinned here:

- the defaults are the references;
- a replacement is what the loop calls, every round, for every client: an
  executor sees each round's requests in order and reports each result to the
  observer before yielding it; the aggregator is handed exactly what the
  executor yields; the evaluator is asked for the plan's clients and for the
  central pass on its schedule;
- a replacement that delegates to the references changes nothing: the round
  history, the per-client records and the progress calls are the default
  run's.
"""

from __future__ import annotations

import inspect
import unittest
from collections.abc import Iterator, Sequence
from dataclasses import asdict, replace
from typing import Any
from unittest import mock

from fedbrew.core import loop
from fedbrew.core.config import CentralTestConfig
from fedbrew.core.execution import FitObserver, StreamingAggregator
from fedbrew.core.protocol import FitRequest, FitResult
from fedbrew.core.state import ExperimentState
from tests.test_resume_metrics_continuity import _EVALUATION, _Client, _Dataset, _Server

ROUNDS = 3
_FIT = loop.SequentialExecutor.fit
_AGGREGATE = StreamingAggregator.aggregate
_EVALUATE = loop.SequentialEvaluator.evaluate_clients


class _MeasuredServer(_Server):
    """The fixture's server, with a central pass that reports a loss."""

    def evaluate_global(self, global_data: Any) -> dict[str, float]:
        return {"loss": 0.5}


class _RecordingExecutor:
    def __init__(self) -> None:
        self.rounds: list[list[str]] = []
        self.observed: list[tuple[str, int, int]] = []
        self.yielded: list[FitResult] = []

    def fit(
        self, clients: Any, requests: Sequence[FitRequest], observer: FitObserver
    ) -> Iterator[FitResult]:
        self.rounds.append([request.client_id for request in requests])

        class Watching:
            def fitted(inner, result: FitResult, seconds: float, done: int, total: int) -> None:
                self.observed.append((result.client_id, done, total))
                observer.fitted(result, seconds, done, total)

        for result in loop.SequentialExecutor().fit(clients, requests, Watching()):
            self.yielded.append(result)
            yield result


class _RecordingAggregator:
    def __init__(self) -> None:
        self.folded: list[FitResult] = []

    def aggregate(self, server: Any, round_info: Any, results: Any) -> dict[str, Any]:
        def watched() -> Iterator[FitResult]:
            for result in results:
                self.folded.append(result)
                yield result

        return StreamingAggregator().aggregate(server, round_info, watched())


class _RecordingEvaluator:
    def __init__(self) -> None:
        self.clients: list[list[str]] = []
        self.central = 0

    def evaluate_clients(self, clients: Any, round_id: int, work: Any, *args: Any) -> Any:
        self.clients.append([info.client_id for info, _ in work])
        return loop.SequentialEvaluator().evaluate_clients(clients, round_id, work, *args)

    def evaluate_central(self, server: Any, dataset: Any) -> dict[str, float]:
        self.central += 1
        return loop.SequentialEvaluator().evaluate_central(server, dataset)


def _run(**seam: Any) -> tuple[ExperimentState, list[tuple[int, int, int, str]]]:
    progress: list[tuple[int, int, int, str]] = []
    state = loop.run_fl_loop(
        server=_MeasuredServer(),
        client={"client_0": _Client(), "client_1": _Client()},
        dataset=_Dataset(),
        global_rounds=ROUNDS,
        evaluation=replace(_EVALUATION, central_test=CentralTestConfig(every=1)),
        on_client_progress=lambda *call: progress.append(call),
        **seam,
    )
    return state, progress


def _numbers(state: ExperimentState) -> list[Any]:
    """The round history without its timings, and the per-client records."""

    return [
        [(r.round_id, r.metrics, r.num_clients, r.num_examples) for r in state.metrics_history],
        [asdict(record) for record in state.client_update_metrics_history],
        [asdict(record) for record in state.client_metrics_history],
    ]


class TheDefaultsAreTheReferencesTest(unittest.TestCase):
    def test_none_is_each_reference(self) -> None:
        for name in ("executor", "aggregator", "evaluator"):
            self.assertIsNone(inspect.signature(loop.run_fl_loop).parameters[name].default)
        with (
            mock.patch.object(
                loop.SequentialExecutor, "fit", autospec=True, side_effect=_FIT
            ) as fit,
            mock.patch.object(
                StreamingAggregator, "aggregate", autospec=True, side_effect=_AGGREGATE
            ) as aggregate,
            mock.patch.object(
                loop.SequentialEvaluator,
                "evaluate_clients",
                autospec=True,
                side_effect=_EVALUATE,
            ) as evaluate,
        ):
            state, _ = _run()
        self.assertEqual(len(state.metrics_history), ROUNDS)
        self.assertEqual((fit.call_count, aggregate.call_count), (ROUNDS, ROUNDS))
        self.assertEqual(evaluate.call_count, ROUNDS)


class AReplacementIsWhatTheLoopCallsTest(unittest.TestCase):
    def test_every_round_every_client(self) -> None:
        executor, aggregator, evaluator = (
            _RecordingExecutor(),
            _RecordingAggregator(),
            _RecordingEvaluator(),
        )
        replaced, replaced_progress = _run(
            executor=executor, aggregator=aggregator, evaluator=evaluator
        )
        default, default_progress = _run()

        self.assertEqual(executor.rounds, [["client_0", "client_1"]] * ROUNDS)
        self.assertEqual(executor.observed, [("client_0", 1, 2), ("client_1", 2, 2)] * ROUNDS)
        self.assertEqual(
            [id(result) for result in aggregator.folded],
            [id(result) for result in executor.yielded],
        )
        self.assertEqual(len(evaluator.clients), ROUNDS)
        self.assertEqual(evaluator.central, ROUNDS)

        self.assertEqual(_numbers(replaced), _numbers(default))
        self.assertEqual(replaced_progress, default_progress)


if __name__ == "__main__":
    unittest.main()
