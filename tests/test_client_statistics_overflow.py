"""A finite loss can still be too large to take a standard deviation of.

`_client_distribution_statistics` short-circuits to NaN when any per-client
value is non-finite, and the check is right about what it checks. It is the
wrong check for the failure here: every value is a finite float, the guard
passes them through, and `statistics.pstdev` raises anyway. It sums the squared
deviations as exact rationals and converts once at the end, so a value near
1e155 -- finite, printable, already in the CSV -- has a variance around 1e310
that no float can hold. `statistics.fmean` has the same shape of failure from
fsum's accumulator, about 150 orders of magnitude further along.

The cost is POST-F01's cost, in the same place in the run: the OverflowError
escapes `run_fl_loop` mid-round, so `round_metrics.csv` stops short with no row
saying why, `run.json` still reads `status: running` with `num_rounds` equal to
the rounds that did finish, and a collector reads a finished run of the wrong
length. Reproduced with `examples/nonconvex-simplex` at
`client.learning_rate: 0.1`: 148 of 150 rounds on disk, `status: running`,
`error: null`.

It was also version-dependent, which is how it survived a 112-finding census.
CPython 3.11 rewrote `pstdev` to take the square root before converting, so
that same run **completes on 3.12 and dies on 3.10** -- both of which the test
matrix builds. `pvariance` still raises on every supported version.

So the guard goes on the statistic rather than on the data, and answers the NaN
the caller already reserves for a distribution nothing can summarise. It is
deliberately not a switch to float arithmetic: that would move the `_std`
column of every run in the repository by an ulp or two to fix a case that only
a diverged run reaches.

FINDINGS.csv POST-F02.
"""

from __future__ import annotations

import math
import statistics
import unittest
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from fedbrew.clients.base import ClientUpdate
from fedbrew.core.config import (
    CentralTestConfig,
    ClientStatisticsConfig,
    EvaluationConfig,
    SplitEvaluationConfig,
)
from fedbrew.core.loop import (
    _aggregate_client_split_metrics,
    _overflow_safe,
    run_fl_loop,
)
from fedbrew.core.protocol import (
    ClientInfo,
    EvalRequest,
    EvalResult,
    FitRequest,
    FitResult,
    RoundInfo,
)
from fedbrew.data.dataset import FederatedDataset
from fedbrew.servers.base import ServerStrategy

pytestmark = pytest.mark.fast

# The per-client losses `examples/nonconvex-simplex` reported in the round that
# used to kill the process, rounded to the digits the CSV carries. Every one is
# a finite float; their squared deviations are not.
DIVERGED_LOSSES = [
    -1.8319862076344268e156,
    -1.8750770667562233e156,
    -1.8750770667562233e156,
    -1.8319862076344268e156,
    -1.8750770667562233e156,
    -1.8319862076344268e156,
    -1.8750770667562233e156,
    -1.8319862076344268e156,
]

# Far enough along that fsum's accumulator overflows too, which is the other
# statistic on this path and 150 orders of magnitude later.
FSUM_OVERFLOW_LOSSES = [1e308, 1e308, -1e308, 5e307]

HEALTHY_LOSSES = [1.0, 3.0, 5.0, 7.0]

ALL_STATISTICS = ClientStatisticsConfig(
    std=True,
    variance=True,
    min=True,
    max=True,
    worst_percent=25.0,
)


def _results(losses: Sequence[float]) -> list[EvalResult]:
    return [
        EvalResult(
            round_id=1,
            client_id=f"client_{index}",
            num_examples=4,
            metrics={"test_loss": float(loss)},
            payload={"model_scope": "global", "num_examples_by_split": {"test": 4}},
        )
        for index, loss in enumerate(losses)
    ]


class OverflowSafeTests(unittest.TestCase):
    def test_the_values_are_finite_and_their_variance_is_not(self) -> None:
        """The premise, stated in a way no interpreter version changes."""

        for loss in DIVERGED_LOSSES:
            with self.subTest(loss=loss):
                self.assertTrue(math.isfinite(loss))
        largest = max(abs(loss) for loss in DIVERGED_LOSSES)
        self.assertTrue(math.isinf(largest * largest))

    def test_a_statistic_that_cannot_fit_answers_nan(self) -> None:
        # pvariance converts the exact rational at the end on every supported
        # version, so this is the one that raises everywhere.
        with self.assertRaises(OverflowError):
            statistics.pvariance(DIVERGED_LOSSES)
        self.assertTrue(math.isnan(_overflow_safe(statistics.pvariance, DIVERGED_LOSSES)))

    def test_fmean_is_guarded_on_the_same_path(self) -> None:
        with self.assertRaises(OverflowError):
            statistics.fmean(FSUM_OVERFLOW_LOSSES)
        self.assertTrue(math.isnan(_overflow_safe(statistics.fmean, FSUM_OVERFLOW_LOSSES)))

    def test_a_statistic_that_fits_passes_through_bit_for_bit(self) -> None:
        """Not `math.isclose`: the guard must not move an existing column.

        Every `_std` and `_avg` a run writes comes from these two functions. A
        fix that computed them in floats instead would agree to a few ulps and
        still change every one of them.
        """

        for statistic in (statistics.pstdev, statistics.pvariance, statistics.fmean):
            with self.subTest(statistic=statistic.__name__):
                self.assertEqual(
                    _overflow_safe(statistic, HEALTHY_LOSSES),
                    statistic(HEALTHY_LOSSES),
                )

    def test_nothing_but_overflow_is_swallowed(self) -> None:
        """A blanket `except Exception` here would hide an empty round."""

        with self.assertRaises(statistics.StatisticsError):
            _overflow_safe(statistics.pstdev, [])


class AggregateTests(unittest.TestCase):
    def test_the_round_that_used_to_raise_now_aggregates(self) -> None:
        aggregated = _aggregate_client_split_metrics(
            _results(DIVERGED_LOSSES), "test", ALL_STATISTICS
        )
        self.assertTrue(math.isnan(aggregated["test_loss_variance"]))
        # The mean is well inside float range and must survive intact: only the
        # statistic that cannot fit is allowed to go missing.
        self.assertEqual(aggregated["test_loss_avg"], statistics.fmean(DIVERGED_LOSSES))
        self.assertEqual(aggregated["test_loss_min"], min(DIVERGED_LOSSES))
        self.assertEqual(aggregated["test_loss_max"], max(DIVERGED_LOSSES))

    def test_std_and_variance_are_guarded_separately(self) -> None:
        """A finite `_std` beside a NaN `_variance` is allowed, on purpose.

        From CPython 3.11 `pstdev` takes the square root before converting, so
        it can return a float on the round where `pvariance` overflows. Both
        answers are true -- the variance exceeds float range, its root does
        not -- and the guard must not force the pair to NaN together for the
        row's sake. On 3.10 both overflow; the assertion below holds either
        way, which is the point: the column follows its own statistic.
        """

        aggregated = _aggregate_client_split_metrics(
            _results(DIVERGED_LOSSES), "test", ALL_STATISTICS
        )
        self.assertTrue(math.isnan(aggregated["test_loss_variance"]))
        expected = _overflow_safe(statistics.pstdev, DIVERGED_LOSSES)
        if math.isnan(expected):
            self.assertTrue(math.isnan(aggregated["test_loss_std"]))
        else:
            self.assertTrue(math.isfinite(expected))
            self.assertEqual(aggregated["test_loss_std"], expected)

    def test_the_column_set_does_not_change_partway_through_a_run(self) -> None:
        healthy = _aggregate_client_split_metrics(_results(HEALTHY_LOSSES), "test", ALL_STATISTICS)
        diverged = _aggregate_client_split_metrics(
            _results(DIVERGED_LOSSES), "test", ALL_STATISTICS
        )
        overflowing = _aggregate_client_split_metrics(
            _results(FSUM_OVERFLOW_LOSSES), "test", ALL_STATISTICS
        )
        self.assertEqual(set(healthy), set(diverged))
        self.assertEqual(set(healthy), set(overflowing))

    def test_a_non_finite_value_still_short_circuits_every_statistic(self) -> None:
        """The older guard is the one that owns NaN input; it must still fire.

        `min` and `max` are the reason it exists: NaN comparisons are all
        False, so they return whichever element sat next to the NaN and say
        nothing about it. `_overflow_safe` never sees these.
        """

        aggregated = _aggregate_client_split_metrics(
            _results([1.0, math.nan, 3.0]), "test", ALL_STATISTICS
        )
        for column in ("std", "variance", "min", "max", "worst25"):
            with self.subTest(column=column):
                self.assertTrue(math.isnan(aggregated[f"test_loss_{column}"]))


class _Dataset(FederatedDataset):
    def list_clients(self) -> list[str]:
        return [f"client_{index}" for index in range(len(DIVERGED_LOSSES))]

    def get_client_data(self, client_id: str) -> dict[str, Any]:
        raise AssertionError("the fake clients already own their data")

    def get_client_metadata(self, client_id: str) -> dict[str, Any]:
        return {
            "client_id": client_id,
            "num_examples": 8,
            "num_train_examples": 4,
            "num_eval_examples": 4,
        }

    def get_global_data(self) -> None:
        raise AssertionError("central global data must not be used")

    def get_metadata(self) -> dict[str, Any]:
        return {"name": "overflow-test"}


class _Client(ClientUpdate):
    """Reports one of the diverged losses, unchanged, every round."""

    def __init__(self, client_id: str, loss: float) -> None:
        self.client_id = client_id
        self.loss = loss

    def setup(self, client_info: ClientInfo) -> None:
        self.client_id = client_info.client_id

    def fit(self, request: FitRequest) -> FitResult:
        return FitResult(
            round_id=request.round_id,
            client_id=self.client_id,
            num_examples=4,
            payload={"model_state": {"version": request.round_id}},
            metrics={"fit_loss": self.loss},
        )

    def evaluate(self, request: EvalRequest) -> EvalResult:
        return EvalResult(
            round_id=request.round_id,
            client_id=self.client_id,
            num_examples=4,
            metrics={"test_loss": self.loss},
            payload={"model_scope": "global", "num_examples_by_split": {"test": 4}},
        )

    def get_state(self) -> dict[str, Any]:
        return {"client_id": self.client_id}

    def load_state(self, state: Mapping[str, Any]) -> None:
        self.client_id = str(state.get("client_id", self.client_id))


class _Server(ServerStrategy):
    def initialize(self) -> dict[str, Any]:
        return {"model_state": {"version": 0}}

    def configure_round(
        self,
        round_info: RoundInfo,
        clients: Sequence[ClientInfo],
    ) -> Sequence[FitRequest]:
        return [
            FitRequest(
                round_id=round_info.round_id,
                client_id=client.client_id,
                payload={"model_state": {"version": round_info.round_id - 1}},
            )
            for client in clients
        ]

    def aggregate(
        self,
        round_info: RoundInfo,
        results: Sequence[FitResult],
    ) -> dict[str, Any]:
        return {"model_state": {"version": round_info.round_id}, "metrics": {}}

    def evaluate(
        self,
        round_info: RoundInfo,
        results: Sequence[EvalResult],
    ) -> dict[str, float]:
        return {}

    def save_state(self) -> dict[str, Any]:
        return {}

    def load_state(self, state: Mapping[str, Any]) -> None:
        return None


class LoopTests(unittest.TestCase):
    def test_the_run_finishes_every_round_it_was_asked_for(self) -> None:
        """The defect's real cost: the rounds that never got written."""

        rounds = 3
        clients = {
            f"client_{index}": _Client(f"client_{index}", loss)
            for index, loss in enumerate(DIVERGED_LOSSES)
        }

        state = run_fl_loop(
            _Server(),
            clients,
            _Dataset(),
            global_rounds=rounds,
            checkpointing={"enabled": False},
            evaluation=EvaluationConfig(
                train=SplitEvaluationConfig(every="never", clients="all"),
                val=SplitEvaluationConfig(every="never", clients="all"),
                test=SplitEvaluationConfig(every=1, clients="all"),
                central_test=CentralTestConfig(every="never"),
            ),
            client_statistics=ALL_STATISTICS,
        )

        self.assertEqual(len(state.metrics_history), rounds)
        self.assertEqual(state.metrics_history[-1].round_id, rounds)
        last = state.metrics_history[-1].metrics
        self.assertTrue(math.isnan(last["test_loss_variance"]))
        self.assertEqual(last["test_loss_avg"], statistics.fmean(DIVERGED_LOSSES))


if __name__ == "__main__":
    unittest.main()
