"""One real run, frozen: the concrete numbers correctness lands on.

tests/test_reproducibility.py already asserts that two runs at one seed --
or a run and its resume -- agree with each other. That is an internal
consistency check: a bug that changes the answer but changes it the same way
every time still passes it. What it cannot catch is a change to the *correct*
answer -- a wrong denominator, a control variate scaled by the wrong N, a
buffer averaged that should have been kept bit-identical -- landing on a new
number that is still self-consistent from run to run.

This file freezes what those numbers actually are, recorded from one real
execution of the pipeline the aggregation, participation and split fixes
touched:
FedAvgServer.sample_clients under partial participation (seeded, not "all"),
WeightedStateAccumulator's per-round weighted mean, a checkpoint write, and a
--resume-latest that must continue the same run. A future change that moves
any of these away from their fixed, tested behaviour -- even one that is
internally consistent -- makes this fail.

Model and data: the in-repo SyntheticClassificationDataset (fedbrew.data.
synthetic_classification), no network access, and a dropout layer. The
dropout is required, not decorative: this file's resume test is the one that
has to notice if the RNG stream not surviving a checkpoint ever regresses,
and a model with no stochastic layer cannot notice that at all. Verified by
injecting the regression directly: with
_restore_rng_state (loop.py) turned into a no-op, a dropout-free version of
this exact config still produced a bit-identical resume, because nothing else
in the run reads the global RNG stream -- sample_clients and the DataLoader
seed are both pure functions of (seed, round_id, ...), never of shared mutable
state. With dropout: 0.3 the same mutation immediately desyncs the resumed
run's fit_loss from the uninterrupted one, which is what the checkpoint file's
own docstring warns: "Any model with dropout or another global-RNG consumer
will diverge" without it.

client eval/test are set equal on this fixture by construction
(synthetic_classification.py's client_test_source: partitioned_global_test),
and central_test is exactly the pooled concatenation of every client's test
slice at equal per-client size -- so val_accuracy_sample_weighted_avg and
central_test_accuracy are mathematically the same number here, not a
resurfacing of the FEMNIST split leak (which was specific to FEMNIST reusing the
eval slice as global_test.pt while claiming client_test_source:
within_client_holdout).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from fedbrew.core.runner import run

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The torch release FROZEN_METRICS below was recorded under.
#:
#: These numbers are not portable across torch releases and are not meant to
#: be. The fixture carries dropout: 0.3 on purpose -- see the module docstring
#: -- which makes the run a consumer of torch's global RNG stream, so a release
#: that changes how dropout draws sends the whole trajectory somewhere else
#: from round 1 onward. Re-freezing 2.5.1 -> 2.13.0 moved fit_loss at round 1
#: by 0.118, and fit_accuracy, val_accuracy_sample_weighted_avg and
#: central_test_accuracy at later rounds by up to 0.083. That is drift in
#: torch, not a regression here.
#:
#: Two fixes were rejected. Widening TOLERANCE to absorb it would put the
#: tolerance above the defects this file exists to catch -- the accumulator
#: denominator mutation it was sized against moves a comparable mean by 17.6%,
#: and a 0.118 tolerance would swallow that whole. Pinning torch in
#: pyproject.toml would make a benchmark framework dictate its users' torch
#: build, which chapter 02 declines to do on purpose. So the frozen floats are
#: gated on the release instead, and CI pins torch so CI stays reproducible.
FROZEN_TORCH_RELEASE = "2.13.0"


def _torch_release() -> str:
    """``2.13.0`` from ``2.13.0+cu130``; ``2.5.1`` from ``2.5.1.post303``.

    The local and build suffixes are dropped deliberately. This fixture pins
    runtime.device to cpu, and a CPU wheel and a CUDA wheel of one release run
    the same CPU kernels: 2.13.0+cpu and 2.13.0+cu130 were measured on
    2026-09-02 to produce the numbers below bit for bit. Gating on the full
    string would skip whichever build CI happens to resolve, for no numerical
    reason.
    """

    match = re.match(r"\d+\.\d+\.\d+", torch.__version__)
    return match.group(0) if match else torch.__version__


#: Named once so the skip says what to do, not just that it skipped.
_REFREEZE = (
    "Re-freeze with `python tests/test_regression_baseline.py --freeze`, which "
    "prints the replacement literal; update FROZEN_METRICS and "
    "FROZEN_TORCH_RELEASE together, in one commit saying which torch produced "
    "them and why a new baseline was needed."
)

#: Kept small on both axes: enough rounds for a checkpoint/resume split to be
#: meaningful (round 3 of 6), enough clients for participation_rate < 1 to
#: draw a genuinely different subset each round (ceil(8 * 0.5) = 4 of 8).
ROUNDS = 6
CHECKPOINT_ROUND = 3
NUM_CLIENTS = 8
PARTICIPATION_RATE = 0.5
#: Non-zero so the model is a global-RNG consumer -- see the module
#: docstring: a dropout-free model cannot fail the resume test even when the
#: checkpoint silently drops the RNG stream.
DROPOUT = 0.3


def _config(
    output_dir: Path,
    *,
    rounds: int = ROUNDS,
    resume_latest: bool = False,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "experiment": {"seed": 1337, "output_dir": str(output_dir)},
        "server": {
            "strategy": "fedavg",
            "participation_rate": PARTICIPATION_RATE,
            "metrics": ["fit_loss", "fit_accuracy"],
        },
        "client": {
            "update_rule": "local_sgd",
            "batch_size": 4,
            "learning_rate": 0.05,
            "momentum": 0.0,
            "weight_decay": 0.0,
            "nesterov": False,
            "learning_rate_schedule": "constant",
            "min_learning_rate": 0.0,
            "metrics": ["fit_loss", "fit_accuracy"],
        },
        "data": {
            "num_clients": NUM_CLIENTS,
            "samples_per_client": 12,
            "input_dim": 4,
            "num_classes": 2,
        },
        "model": {
            "name": "mlp",
            "input_dim": 4,
            "hidden_dim": 8,
            "num_classes": 2,
            "dropout": DROPOUT,
        },
        "runtime": {
            "deterministic": True,
            "deterministic_warn_only": False,
            "device": "cpu",
            "use_amp": False,
            "checkpointing": {
                "enabled": True,
                "save_last": True,
                "save_every_round": True,
                "keep_last": 0,
            },
        },
        "evaluation": {
            "train": {"every": 1, "clients": "all"},
            "val": {"every": 1, "clients": "all"},
            "test": {"every": 1, "clients": "all"},
            "central_test": {"every": 1},
        },
        "defaults": {"global_rounds": rounds, "local_iterations": 1},
    }
    if resume_latest:
        config["runtime"]["extra"] = {"resume_latest": True}
    return config


def _write(root: Path, name: str, **kwargs: Any) -> Path:
    path = root / f"{name}.yaml"
    path.write_text(yaml.safe_dump(_config(root / name, **kwargs)), encoding="utf-8")
    return path


def _run(path: Path, *, resume_latest: bool = False) -> Path:
    args = argparse.Namespace(resume_latest=resume_latest) if resume_latest else None
    state = run(path, args=args)
    assert state.metrics_history, "the run produced no rounds"
    return Path(yaml.safe_load(path.read_text())["experiment"]["output_dir"])


def _rows(output_dir: Path) -> list[dict[str, str]]:
    with (output_dir / "round_metrics.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


#: Columns that legitimately differ between two runs of the same config --
#: wall-clock timing depends on the machine's load, not on the computation.
TIMING_COLUMNS = {
    "duration_sec",
    "fit_sec",
    "aggregate_sec",
    "client_eval_sec",
    "global_eval_sec",
    "checkpoint_sec",
}


def _stripped(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [{key: value for key, value in row.items() if key not in TIMING_COLUMNS} for row in rows]


#: One real execution of this exact config, seed 1337, recorded 2026-09-02
#: under torch 2.13.0 on CPU. Regenerate with the command in the class
#: docstring below if a deliberate change to aggregation, participation or the
#: client update rule requires a new baseline -- and say in the commit message
#: why the numbers moved.
#:
#: num_clients is 4 every round, not 8: proof this run exercises partial,
#: seeded participation (FedAvgServer.sample_clients under
#: participation_rate: 0.5), not the participation_rate: 1 path every other
#: fixture in this repo's test suite uses.
FROZEN_METRICS: list[dict[str, Any]] = [
    {
        "round_id": 1,
        "num_clients": 4,
        "num_examples": 48,
        "fit_loss": 0.7828545421361923,
        "fit_accuracy": 0.6041666666666666,
        "val_accuracy_sample_weighted_avg": 0.4270833333333333,
        "central_test_accuracy": 0.4270833333333333,
        "central_test_loss": 0.7408025860786438,
    },
    {
        "round_id": 2,
        "num_clients": 4,
        "num_examples": 48,
        "fit_loss": 0.6933911889791489,
        "fit_accuracy": 0.4791666666666667,
        "val_accuracy_sample_weighted_avg": 0.40625,
        "central_test_accuracy": 0.40625,
        "central_test_loss": 0.7277345657348633,
    },
    {
        "round_id": 3,
        "num_clients": 4,
        "num_examples": 48,
        "fit_loss": 0.666963741183281,
        "fit_accuracy": 0.625,
        "val_accuracy_sample_weighted_avg": 0.40625,
        "central_test_accuracy": 0.40625,
        "central_test_loss": 0.7217652797698975,
    },
    {
        "round_id": 4,
        "num_clients": 4,
        "num_examples": 48,
        "fit_loss": 0.6647893786430359,
        "fit_accuracy": 0.5416666666666666,
        "val_accuracy_sample_weighted_avg": 0.4166666666666667,
        "central_test_accuracy": 0.4166666666666667,
        "central_test_loss": 0.7148580551147461,
    },
    {
        "round_id": 5,
        "num_clients": 4,
        "num_examples": 48,
        "fit_loss": 0.6371231824159622,
        "fit_accuracy": 0.6458333333333334,
        "val_accuracy_sample_weighted_avg": 0.40625,
        "central_test_accuracy": 0.40625,
        "central_test_loss": 0.711749792098999,
    },
    {
        "round_id": 6,
        "num_clients": 4,
        "num_examples": 48,
        "fit_loss": 0.637186273932457,
        "fit_accuracy": 0.625,
        "val_accuracy_sample_weighted_avg": 0.4270833333333333,
        "central_test_accuracy": 0.4270833333333333,
        "central_test_loss": 0.7046942114830017,
    },
]


#: Compared bit-for-bit: whatever changed the algorithm did not change these,
#: so any difference is meaningful, not rounding.
EXACT_COLUMNS = {"round_id", "num_clients", "num_examples"}

#: Numeric tolerance for the columns above, derived rather than eyeballed.
#:
#: This exact config, run on this machine with torch.set_num_threads forced
#: to 1 and to 4 (a real, measurable proxy for a different BLAS/scheduling
#: environment reducing the same sums in a different order), produced
#: bit-identical fit_loss and fit_accuracy at every round -- the tensors here
#: (batch 4, hidden_dim 8) are far too small to invoke a multi-threaded
#: reduction path, so the measured order-sensitivity on this build is exactly
#: zero. That leaves cross-machine drift (a different PyTorch version, a
#: different CPU's float32 unit) as the only real source of disagreement, and
#: nothing here can measure that directly, so the bound below is analytic:
#:
#:   float32 machine epsilon        eps = 1.1920929e-07
#:   sequential steps in this run   6 rounds * 4 clients * 3 batches = 72
#:   generous per-step FLOP count   ~2,000 (two small matmuls, their
#:                                  backward pass, and the loss/softmax)
#:   total FLOPs, training alone    ~1.4e5; rounded up 7x for the four
#:                                  evaluation passes every round          -> n = 1e6
#:   probabilistic error bound      sqrt(n) * eps  =  1000 * 1.1920929e-07
#:                                                  =  1.1920929e-04
#:   margin for iterative compounding across 6 rounds (each round's output
#:   is the next round's input, so this is not a single flat sum) and for
#:   the BLAS drift the thread-count probe above could not exercise: 10x
#:
#: giving ~1.2e-3, rounded to a clean 1e-3. For scale: the mutation that
#: swapped WeightedStateAccumulator's denominator for the client *count*
#: instead of the summed weight (tests/test_aggregation_correctness.py) moved
#: a comparable small-scale mean by 17.6% relative -- five orders of
#: magnitude above this tolerance, so nothing here can mask a real defect.
TOLERANCE = 1e-3


class _TempRoot(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)


class FrozenBaselineTests(_TempRoot):
    """The uninterrupted run must still land on the numbers above.

    Split in two on purpose. The integer columns -- how many rounds ran, how
    many clients each drew, how many examples they held -- are what partial
    seeded participation produces, and no torch release changes them, so they
    are checked on every torch. The floats are a recording of one release's
    arithmetic and are checked only on that release; see FROZEN_TORCH_RELEASE.
    """

    def test_the_run_has_the_frozen_shape(self) -> None:
        """Runs on every torch: these columns are counts, not arithmetic."""

        rows = _rows(_run(_write(self.root, "shape")))
        self.assertEqual(len(rows), len(FROZEN_METRICS))
        for row, expected in zip(rows, FROZEN_METRICS, strict=True):
            with self.subTest(round_id=expected["round_id"]):
                for key in EXACT_COLUMNS:
                    self.assertEqual(
                        int(row[key]),
                        expected[key],
                        f"{key} at round {expected['round_id']} -- this is a "
                        "count, so no torch release explains a change here",
                    )

    @unittest.skipUnless(
        _torch_release() == FROZEN_TORCH_RELEASE,
        f"FROZEN_METRICS was recorded under torch {FROZEN_TORCH_RELEASE}; this "
        f"environment runs torch {torch.__version__}. The frozen floats are not "
        "portable across releases -- a release that changes torch's dropout RNG "
        "stream moves this fixture's whole trajectory -- so they are checked "
        f"only on {FROZEN_TORCH_RELEASE}. The shape check and both resume "
        f"equivalences ran. {_REFREEZE}",
    )
    def test_the_run_matches_the_frozen_numbers(self) -> None:
        rows = _rows(_run(_write(self.root, "baseline")))
        self.assertEqual(len(rows), len(FROZEN_METRICS))

        for row, expected in zip(rows, FROZEN_METRICS, strict=True):
            with self.subTest(round_id=expected["round_id"]):
                mismatched = {
                    key: (float(row[key]), value)
                    for key, value in expected.items()
                    if key not in EXACT_COLUMNS and abs(float(row[key]) - value) > TOLERANCE
                }
                # Reported together rather than one assertAlmostEqual per key:
                # raising on the first key hides how far the divergence spread,
                # which is what made a torch upgrade look like a fit_loss-only
                # problem when it had moved four columns.
                self.assertEqual(
                    mismatched,
                    {},
                    f"at round {expected['round_id']}, (actual, frozen) beyond "
                    f"{TOLERANCE}: {mismatched}. If torch was upgraded, "
                    f"FROZEN_TORCH_RELEASE ({FROZEN_TORCH_RELEASE}) is stale "
                    f"and this ran anyway. {_REFREEZE}",
                )

    @pytest.mark.fast
    def test_participation_is_seeded_and_partial(self) -> None:
        # Guards the guard: if participation_rate stopped being applied and
        # every client fit every round, num_clients would read 8, not 4, and
        # the frozen baseline above would not be exercising
        # sample_clients(..., round_id) at all.
        for expected in FROZEN_METRICS:
            self.assertEqual(expected["num_clients"], 4)
            self.assertLess(expected["num_clients"], NUM_CLIENTS)


@pytest.mark.fast
class CiPinMatchesTheFrozenReleaseTest(unittest.TestCase):
    """CI's torch pin and FROZEN_TORCH_RELEASE must name one release.

    Guards the guard, and the failure it prevents is the quiet one. If they
    drift apart, nothing breaks: CI installs some other torch, the skipUnless
    above fires, the job stays green, and the frozen numbers -- the only check
    in this repository that can catch a change to the *correct* answer -- are
    silently no longer being checked anywhere. A green run that verifies less
    than it did yesterday is worse than a red one.
    """

    def test_the_workflow_pins_the_release_the_numbers_were_frozen_under(self) -> None:
        workflow = REPO_ROOT / ".github" / "workflows" / "tests.yml"
        self.assertTrue(workflow.is_file(), "the CI workflow moved; update this guard")
        pinned = set(re.findall(r"pip install torch==(\S+)", workflow.read_text(encoding="utf-8")))
        self.assertTrue(
            pinned,
            "CI no longer pins torch. It must, or the frozen assertions skip "
            "in CI and stop protecting anything.",
        )
        self.assertEqual(
            pinned,
            {FROZEN_TORCH_RELEASE},
            f"CI pins torch {sorted(pinned)} but FROZEN_METRICS was recorded "
            f"under {FROZEN_TORCH_RELEASE}, so the frozen assertions would "
            f"skip in CI and the job would pass without them. {_REFREEZE}",
        )


class ResumeMatchesTheUninterruptedRunTests(_TempRoot):
    """A checkpoint, restored, must continue the run it was written for.

    Compared bit-for-bit rather than against TOLERANCE: both runs execute in
    this same test, in this same process, on this same build, so nothing
    legitimate can make them differ -- any gap is the resume path itself
    losing state: a stream the checkpoint does not carry, silently restarting
    from a different point.
    """

    def test_a_resumed_run_reproduces_the_uninterrupted_run_exactly(self) -> None:
        full = _run(_write(self.root, "full", rounds=ROUNDS))
        full_rows = _rows(full)

        partial_path = _write(self.root, "part", rounds=CHECKPOINT_ROUND)
        _run(partial_path)
        checkpoint_dir = (
            Path(yaml.safe_load(partial_path.read_text())["experiment"]["output_dir"])
            / "checkpoints"
        )
        self.assertTrue(
            (checkpoint_dir / "latest.pt").exists(),
            "checkpointing.enabled: true must have written a checkpoint",
        )

        resumed_path = _write(self.root, "part", rounds=ROUNDS)
        resumed = _run(resumed_path, resume_latest=True)
        resumed_rows = _rows(resumed)

        self.assertEqual(len(resumed_rows), ROUNDS)
        self.assertEqual(_stripped(full_rows), _stripped(resumed_rows))

    def test_the_run_metadata_records_that_it_was_resumed(self) -> None:
        partial_path = _write(self.root, "solo", rounds=CHECKPOINT_ROUND)
        output_dir = _run(partial_path)
        resumed_path = _write(self.root, "solo", rounds=ROUNDS)
        _run(resumed_path, resume_latest=True)

        import json

        run_metadata = json.loads((output_dir / "run.json").read_text())
        self.assertTrue(run_metadata["resumed"])
        self.assertIsNotNone(run_metadata["resume_from"])


def _print_frozen_literal() -> None:
    """Print a FROZEN_METRICS body recorded from one run on this torch.

    The re-freeze path is code rather than prose because the alternative --
    "run it and copy the numbers out of round_metrics.csv" -- is where a
    hand-transcribed baseline goes wrong, and because the columns that belong
    in the literal are a subset: the timing columns are machine-dependent and
    must never be frozen.
    """

    with tempfile.TemporaryDirectory() as scratch:
        rows = _rows(_run(_write(Path(scratch), "baseline")))

    print(f"# recorded under torch {torch.__version__}")
    print(f'FROZEN_TORCH_RELEASE = "{_torch_release()}"')
    print("FROZEN_METRICS: list[dict[str, Any]] = [")
    for row in rows:
        print("    {")
        for key in FROZEN_METRICS[0]:
            value = int(row[key]) if key in EXACT_COLUMNS else float(row[key])
            print(f"        {key!r}: {value!r},")
        print("    },")
    print("]")


if __name__ == "__main__":
    if "--freeze" in sys.argv:
        _print_frozen_literal()
    else:
        unittest.main()
