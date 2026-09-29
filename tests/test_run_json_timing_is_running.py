"""run.json's timing block is kept as running aggregates, not re-scanned.

run.json is rewritten every round, and its timing block summed, averaged,
sorted for the median and took the extremes of every round's timings since
round 1, each time -- O(rounds so far) per round. The loop's round history now
keeps those aggregates as records arrive (RoundTimingSummary). What is pinned
here:

- every aggregate is the value the full computation gives, bit for bit, before
  the rounding run.json applies: the total and the per-phase sums added left
  to right in round order, statistics.fmean, statistics.median, min and max,
  over durations with duplicates, zeros and magnitudes far apart, at every
  length from 1. The sums are not sum()'s: since Python 3.12 sum() adds
  floats with a running compensation, so it no longer adds in order;
- the block written from the running history is the block written from the
  same records in a plain list, which is what every earlier writer did;
- the per-round path reads the aggregates and never iterates the history.
"""

from __future__ import annotations

import random
import statistics
import unittest
from dataclasses import fields
from typing import Any

import pytest

from fedbrew.core.artifacts import _timing_summary
from fedbrew.core.state import ExperimentState, MetricRecord, RoundHistory, RoundTimings

_PHASES = [timing.name for timing in fields(RoundTimings) if timing.name != "total"]


def _duration(rng: random.Random) -> float:
    kind = rng.randrange(5)
    if kind == 0:
        return 0.0
    if kind == 1:
        return rng.choice([0.1, 0.25, 1.0 / 3.0])
    if kind == 2:
        return rng.uniform(0.0, 1e-6)
    if kind == 3:
        return rng.uniform(0.0, 1e6)
    return rng.expovariate(10.0)


def _timings(rng: random.Random) -> RoundTimings:
    return RoundTimings(**{timing.name: _duration(rng) for timing in fields(RoundTimings)})


def _in_round_order(values: Any) -> float:
    """The values added left to right, one rounding per addition: sum() before Python 3.12."""

    total = 0.0
    for value in values:
        total += value
    return total


def _record(round_id: int, timings: RoundTimings | None) -> MetricRecord:
    return MetricRecord(
        round_id=round_id, metrics={}, num_clients=1, num_examples=1, timings=timings
    )


@pytest.mark.fast
class TheRunningAggregatesAreTheFullOnesTest(unittest.TestCase):
    def test_bit_for_bit_at_every_length(self) -> None:
        rng = random.Random(20260926)
        for trial in range(40):
            history = RoundHistory()
            timed: list[RoundTimings] = []
            for round_id in range(1, 200):
                timings = None if rng.random() < 0.05 else _timings(rng)
                history.append(_record(round_id, timings))
                if timings is None:
                    continue
                timed.append(timings)
                durations = [t.total for t in timed]
                running = history.summary
                with self.subTest(trial=trial, rounds=round_id):
                    self.assertEqual(running.timed_rounds, len(timed))
                    self.assertEqual(running.total_sec.hex(), _in_round_order(durations).hex())
                    self.assertEqual(
                        running.mean_sec().hex(), float(statistics.fmean(durations)).hex()
                    )
                    self.assertEqual(
                        running.median_sec().hex(), float(statistics.median(durations)).hex()
                    )
                    self.assertEqual(running.minimum_sec, min(durations))
                    self.assertEqual(running.maximum_sec, max(durations))
                    for phase in _PHASES:
                        self.assertEqual(
                            running.phase_sec[phase].hex(),
                            _in_round_order(getattr(t, phase) for t in timed).hex(),
                        )

    def test_the_written_block_is_the_plain_lists(self) -> None:
        rng = random.Random(7)
        history = RoundHistory()
        history.extend(_record(round_id, _timings(rng)) for round_id in range(1, 60))
        metadata = {"duration_sec": 12.5}
        self.assertEqual(
            _timing_summary(history, metadata), _timing_summary(list(history), metadata)
        )

    def test_the_loops_history_keeps_them(self) -> None:
        self.assertIsInstance(ExperimentState().metrics_history, RoundHistory)


class _Unreadable(RoundHistory):
    __slots__ = ()

    def __iter__(self) -> Any:
        raise AssertionError("the per-round timing block iterated the whole history")


@pytest.mark.fast
class ThePerRoundPathDoesNotScanTest(unittest.TestCase):
    def test_the_history_is_not_iterated(self) -> None:
        rng = random.Random(3)
        history = _Unreadable()
        for round_id in range(1, 20):
            history.append(_record(round_id, _timings(rng)))
        self.assertEqual(_timing_summary(history, {})["timed_rounds"], 19)


if __name__ == "__main__":
    unittest.main()
