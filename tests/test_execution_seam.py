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
  run's;
- the stacked path: when the executor offers ``fit_stacked`` for the round
  and the aggregator takes ``aggregate_stacked``, the loop takes it every
  round, the observer is told of each stack before it is yielded, the
  aggregator is handed exactly the stacks yielded, and the server that folds
  one result at a time is handed each client's result, in order; the records
  and progress calls are the default run's. An executor that declines the
  round (None) or an aggregator without ``aggregate_stacked`` sends the round
  through ``fit`` and ``aggregate``.
"""

from __future__ import annotations

import inspect
import unittest
from collections.abc import Iterator, Sequence
from dataclasses import asdict, replace
from typing import Any
from unittest import mock

import torch

from fedbrew.core import loop
from fedbrew.core.config import CentralTestConfig
from fedbrew.core.execution import FitObserver, StreamingAggregator
from fedbrew.core.protocol import FitRequest, FitResult
from fedbrew.core.stacked_results import MetricColumns, StackedFitResults
from fedbrew.core.state import ExperimentState
from fedbrew.core.torch_utils import StateStack
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


def _stack(results: list[FitResult]) -> StackedFitResults:
    """A chunk's results as the stacked path hands them: counts and metrics as tensors."""

    columns = MetricColumns(len(results))
    for position, result in enumerate(results):
        for name, value in result.metrics.items():
            columns.put(name, [position], [value])
    metrics, reported = columns.tensors()
    return StackedFitResults(
        round_id=results[0].round_id,
        client_ids=[result.client_id for result in results],
        num_examples=torch.tensor([result.num_examples for result in results]),
        states=[(StateStack({"w": torch.zeros(len(results), 1)}), list(range(len(results))))],
        metrics=metrics,
        reported=reported,
        payload={},
    )


class _StackingExecutor(_RecordingExecutor):
    """Runs the round as the reference does, and hands it over as one stack."""

    def __init__(self, decline: bool = False) -> None:
        super().__init__()
        self.decline = decline
        self.stacked_rounds: list[list[str]] = []
        self.stacks: list[StackedFitResults] = []
        self.told: list[tuple[int, int, int]] = []

    def fit_stacked(
        self, clients: Any, requests: Sequence[FitRequest], observer: Any
    ) -> Iterator[StackedFitResults] | None:
        if self.decline:
            return None
        self.stacked_rounds.append([request.client_id for request in requests])

        class Silent:
            def fitted(inner, *args: Any) -> None:
                pass

        def run() -> Iterator[StackedFitResults]:
            stacked = _stack(list(loop.SequentialExecutor().fit(clients, requests, Silent())))
            self.told.append((id(stacked), len(stacked), len(requests)))
            observer.fitted_stack(stacked, 0.0, len(stacked), len(requests))
            self.stacks.append(stacked)
            yield stacked

        return run()


class _StackedAggregator(_RecordingAggregator):
    def __init__(self) -> None:
        super().__init__()
        self.handed: list[StackedFitResults] = []

    def aggregate_stacked(self, server: Any, round_info: Any, stacks: Any) -> dict[str, Any]:
        def watched() -> Iterator[StackedFitResults]:
            for stacked in stacks:
                self.handed.append(stacked)
                yield stacked

        return StreamingAggregator().aggregate_stacked(server, round_info, watched())


class _WatchedServer(_MeasuredServer):
    """Records what its one-at-a-time fold is handed."""

    folded: list[list[tuple[str, int, dict[str, float]]]] = []

    def aggregate(self, round_info: Any, results: Any) -> dict[str, Any]:
        _WatchedServer.folded.append(
            [(r.client_id, r.num_examples, dict(r.metrics)) for r in results]
        )
        return super().aggregate(round_info, results)


class TheStackedPathTest(unittest.TestCase):
    def _run(self, **seam: Any) -> tuple[ExperimentState, list[Any], list[Any]]:
        _WatchedServer.folded = []
        progress: list[tuple[int, int, int, str]] = []
        state = loop.run_fl_loop(
            server=_WatchedServer(),
            client={"client_0": _Client(), "client_1": _Client()},
            dataset=_Dataset(),
            global_rounds=ROUNDS,
            evaluation=replace(_EVALUATION, central_test=CentralTestConfig(every=1)),
            on_client_progress=lambda *call: progress.append(call),
            **seam,
        )
        return state, progress, list(_WatchedServer.folded)

    def test_taken_every_round_and_the_same_as_the_default(self) -> None:
        executor, aggregator = _StackingExecutor(), _StackedAggregator()
        stacked, stacked_progress, stacked_folds = self._run(
            executor=executor, aggregator=aggregator
        )
        default, default_progress, default_folds = self._run()

        self.assertEqual(executor.stacked_rounds, [["client_0", "client_1"]] * ROUNDS)
        self.assertEqual(executor.rounds, [])
        self.assertEqual(aggregator.folded, [])
        self.assertEqual(executor.told, [(id(s), 2, 2) for s in executor.stacks])
        self.assertEqual([id(s) for s in aggregator.handed], [id(s) for s in executor.stacks])
        self.assertEqual(stacked_folds, default_folds)
        self.assertEqual(_numbers(stacked), _numbers(default))
        self.assertEqual(stacked_progress, default_progress)

    def test_declined_or_not_taken_goes_one_result_at_a_time(self) -> None:
        default, default_progress, _ = self._run()
        for label, executor, aggregator in (
            ("the executor declines", _StackingExecutor(decline=True), _StackedAggregator()),
            ("the aggregator does not take it", _StackingExecutor(), _RecordingAggregator()),
        ):
            with self.subTest(label):
                state, progress, _ = self._run(executor=executor, aggregator=aggregator)
                self.assertEqual(executor.stacked_rounds, [])
                self.assertEqual(executor.rounds, [["client_0", "client_1"]] * ROUNDS)
                self.assertEqual(len(aggregator.folded), 2 * ROUNDS)
                self.assertEqual(_numbers(state), _numbers(default))
                self.assertEqual(progress, default_progress)


if __name__ == "__main__":
    unittest.main()
