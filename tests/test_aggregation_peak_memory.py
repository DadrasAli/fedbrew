"""Peak aggregation memory does not scale with participation. Measured.

Three chapters state it -- docs/01 section 2, docs/07 section 3.1, docs/11
section 3 -- and so does WeightedStateAccumulator's own docstring, which is the
justification for the class existing. Nothing observed it. No test in the suite
touched an allocation count, a storage size or a liveness check, so a change
that buffered every client state before averaging would have cost one model
copy per participant at cross-device scale and passed every existing guard: the
averaged result is identical either way.

What is measured here is bytes retained, not bytes allocated. tracemalloc is
useless for this -- torch allocates tensor storage outside Python's allocator,
so a 64-client round with 800 KB states reports a 10 KiB peak -- and RSS is a
process-wide high-water mark that cannot go back down between two measurements
in one process. Walking the accumulator's own storages, deduplicated by data
pointer, is exact and deterministic instead.

The claim is asserted as a shape and as a constant, because the two fail
differently. The shape -- retained bytes identical at 4, 16 and 64 clients --
is what "regardless of participation" means, and it is the one that matters.
The constant is pinned as well so that an extra full copy, which does not scale
with participation and would pass the shape check, still fails.

The constant is **two** model states, not the "one model" three chapters used
to claim: the running sum, and one reference copy the accumulator keeps to fix
the key set, dtypes and shapes and to compare non-floating buffers against. The
chapters now say two. Buffering would be N.
"""

from __future__ import annotations

import gc
import inspect
import unittest
from collections.abc import Iterator
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch

from fedbrew.core.loop import _build_client_metric_record, _stream_fit_results
from fedbrew.core.protocol import FitRequest, FitResult, RoundInfo
from fedbrew.core.state import ExperimentState
from fedbrew.core.torch_utils import WeightedStateAccumulator
from fedbrew.servers.fedavg import FedAvgServer

REPO_ROOT = Path(__file__).resolve().parent.parent

#: 100k float32 = 400 KB per model state. Large enough that one copy per client
#: at 64 clients is 25 MB rather than a rounding error, small enough to stay in
#: the default suite.
SIZE = 100_000
MODEL_BYTES = SIZE * 4

#: Participation counts to compare. The claim is that the measurement is the
#: same for all of them; three points make "flat" distinguishable from "grows
#: slowly", which two would not.
CLIENT_COUNTS = (4, 16, 64)

#: What the accumulator is expected to hold at any moment: the running weighted
#: sum, plus the first client's state kept as the reference for the key set,
#: dtypes, shapes and the non-floating comparison.
EXPECTED_MODEL_STATES_RETAINED = 2


def _fit_result(client_id: str, value: float) -> FitResult:
    """One client's result, its state filled with a value identifying it."""

    return FitResult(
        round_id=1,
        client_id=client_id,
        num_examples=1,
        payload={
            "model_state": {"w": torch.full((SIZE,), value)},
            "model_state_scope": "full",
            "model_state_metadata": {"model_state_scope": "full"},
        },
        metrics={"fit_loss": 1.0},
    )


def _server() -> FedAvgServer:
    server = FedAvgServer(participation_rate=1.0, seed=0, aggregation_weighting="uniform")
    server._model_state = {"w": torch.zeros(SIZE)}
    server._model_state_scope = "full"
    server._model_state_metadata = {"model_state_scope": "full"}
    return server


def _retained_bytes(accumulator: WeightedStateAccumulator) -> int:
    """Storage bytes the accumulator holds, deduplicated by data pointer.

    Deduplicated because on CPU `.detach().cpu()` is a view: the reference copy
    shares storage with the first client's tensor, and counting both would
    report memory that was never allocated twice.
    """

    seen: set[int] = set()
    total = 0
    for store in (accumulator._totals, accumulator._reference):
        for tensor in store.values():
            storage = tensor.untyped_storage()
            if storage.data_ptr() in seen:
                continue
            seen.add(storage.data_ptr())
            total += storage.nbytes()
    return total


class _CapturingAccumulator(WeightedStateAccumulator):
    """Records itself so the test can measure it while a round is in flight."""

    instances: list[WeightedStateAccumulator] = []

    def __init__(self) -> None:
        super().__init__()
        type(self).instances.append(self)


def _measure(client_count: int) -> tuple[int, list[float]]:
    """Aggregate `client_count` clients, sampling what is retained per fold.

    Returns the peak retained bytes and, per fold, which client's values the
    reference copy is holding.
    """

    import fedbrew.servers.fedavg as fedavg_module

    _CapturingAccumulator.instances = []
    original = fedavg_module.WeightedStateAccumulator
    fedavg_module.WeightedStateAccumulator = _CapturingAccumulator
    peak = 0
    referenced: list[float] = []

    def stream() -> Iterator[FitResult]:
        nonlocal peak
        for index in range(client_count):
            yield _fit_result(f"c{index}", float(index))
            gc.collect()
            accumulator = _CapturingAccumulator.instances[0]
            peak = max(peak, _retained_bytes(accumulator))
            referenced.append(float(accumulator._reference["w"][0]))

    try:
        _server().aggregate_stream(RoundInfo(round_id=1), stream())
    finally:
        fedavg_module.WeightedStateAccumulator = original
    return peak, referenced


class PeakDoesNotScaleWithParticipationTest(unittest.TestCase):
    """The shape claim, which is the one three chapters make."""

    def setUp(self) -> None:
        self.measurements = {count: _measure(count)[0] for count in CLIENT_COUNTS}

    def test_the_same_bytes_are_retained_at_every_participation(self) -> None:
        distinct = set(self.measurements.values())
        self.assertEqual(
            len(distinct),
            1,
            f"retained bytes vary with participation: {self.measurements}",
        )

    def test_it_is_the_expected_constant_and_not_merely_flat(self) -> None:
        """A flat measurement one copy too high would pass the check above."""

        for count, peak in sorted(self.measurements.items()):
            with self.subTest(clients=count):
                self.assertEqual(peak, EXPECTED_MODEL_STATES_RETAINED * MODEL_BYTES)

    def test_buffering_would_have_cost_one_copy_per_client(self) -> None:
        """The saving the class exists for, stated rather than implied."""

        buffered = max(CLIENT_COUNTS) * MODEL_BYTES
        streamed = self.measurements[max(CLIENT_COUNTS)]
        self.assertGreater(buffered, 8 * streamed)


class FoldedStatesAreReleasedTest(unittest.TestCase):
    """ "A client's model state becomes unreachable as soon as it is folded."

    True of every client but the first, which the reference copy holds for the
    round. The chapters say a constant, not zero, and this is why.
    """

    def test_the_reference_is_the_first_client_and_stays_the_first_client(self) -> None:
        _, referenced = _measure(max(CLIENT_COUNTS))
        self.assertEqual(set(referenced), {0.0}, "the reference copy changed client mid-round")

    def test_nothing_is_retained_once_the_round_is_over(self) -> None:
        _CapturingAccumulator.instances = []
        import fedbrew.servers.fedavg as fedavg_module

        original = fedavg_module.WeightedStateAccumulator
        fedavg_module.WeightedStateAccumulator = _CapturingAccumulator
        try:
            _server().aggregate_stream(
                RoundInfo(round_id=1),
                (_fit_result(f"c{index}", float(index)) for index in range(8)),
            )
        finally:
            fedavg_module.WeightedStateAccumulator = original
        gc.collect()
        self.assertEqual(_retained_bytes(_CapturingAccumulator.instances[0]), 0)


class TheLoopStreamsRatherThanBuffersTest(unittest.TestCase):
    """The other half: the loop must hand the server a generator, not a list."""

    def test_the_fit_phase_is_a_generator_function(self) -> None:
        self.assertTrue(
            inspect.isgeneratorfunction(_stream_fit_results),
            "_stream_fit_results returning a list would buffer every client "
            "state for the whole round, which no other guard would notice",
        )

    def test_nothing_is_produced_before_the_first_result_is_asked_for(self) -> None:
        """A list would have run every client before the server saw one."""

        from fedbrew.core.loop import _FitPhaseTotals

        fitted: list[str] = []

        class _Pool:
            """Not a Mapping: _fit_client dispatches on that."""

            def fit(self, request: FitRequest) -> FitResult:
                fitted.append(request.client_id)
                return _fit_result(request.client_id, 1.0)

        requests = [FitRequest(round_id=1, client_id=f"c{index}", payload={}) for index in range(4)]
        stream = _stream_fit_results(
            _Pool(), requests, ExperimentState(), _FitPhaseTotals(), round_id=1
        )
        self.assertTrue(inspect.isgenerator(stream))
        self.assertEqual(fitted, [], "clients ran before the consumer asked for one")

        next(stream)
        self.assertEqual(fitted, ["c0"], "more than one client ran per result")
        self.assertEqual(sum(1 for _ in stream) + 1, len(requests))

    def test_the_per_client_record_carries_no_model_state(self) -> None:
        """It is appended once per client per round and kept for the whole run."""

        record = _build_client_metric_record(_fit_result("c0", 1.0))
        for item in fields(record):
            with self.subTest(field=item.name):
                value: Any = getattr(record, item.name)
                self.assertNotIsInstance(value, torch.Tensor)
        self.assertNotIn("model_state", {item.name for item in fields(record)})


def _flattened(path: str) -> str:
    """A chapter with its hard wrapping removed.

    Every phrase checked below spans a line break in at least one of the three
    chapters, and a raw `assertIn` would fail on the wrapping rather than on
    the claim -- while dumping the whole chapter into the failure message.
    """

    return " ".join((REPO_ROOT / path).read_text(encoding="utf-8").split())


class TheChaptersStateTheMeasuredConstantTest(unittest.TestCase):
    """Three chapters and a docstring said "one model". It is two."""

    CLAIMS = {
        "docs/01-architecture.md": "peak memory is a constant number of model states",
        "docs/07-algorithms.md": "peak memory is a constant number of model states",
        "docs/11-performance-and-cost.md": "does not scale with participation",
    }

    def test_each_chapter_states_the_shape(self) -> None:
        for path, phrase in sorted(self.CLAIMS.items()):
            with self.subTest(chapter=path):
                self.assertTrue(
                    phrase in _flattened(path),
                    f"{path} must state: {phrase!r}",
                )

    def test_no_chapter_still_claims_one_model(self) -> None:
        for path in sorted(self.CLAIMS):
            with self.subTest(chapter=path):
                flattened = _flattened(path)
                for stale in ("peak memory is one model state", "Peak model memory is one model,"):
                    self.assertNotIn(stale, flattened, f"{path} still says {stale!r}")

    def test_the_accumulator_docstring_matches_what_it_holds(self) -> None:
        docstring = " ".join((WeightedStateAccumulator.__doc__ or "").split())
        self.assertIn("two model states", docstring)
        self.assertNotIn("one model regardless", docstring)

    def test_the_number_the_chapters_give_is_the_measured_one(self) -> None:
        words = {2: "two", 3: "three", 4: "four"}
        stated = words[EXPECTED_MODEL_STATES_RETAINED]
        for path in sorted(self.CLAIMS):
            with self.subTest(chapter=path):
                self.assertIn(
                    f"{stated} model states",
                    _flattened(path),
                    f"{path} must give the measured constant, {stated}",
                )


if __name__ == "__main__":
    unittest.main()
