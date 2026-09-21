"""A classification client's aggregation weight is its train split's size.

Not the number of examples its local update touched. The weight a client
reports is `TaskAdapter.federated_aggregation_weight`, which for
classification returns the example count of the post-fit evaluation pass over
the client's whole train split (`TorchSGDClient._evaluate_model`, whose loader
never drops a batch). `single_batch`, `max_local_steps` and `drop_last` all
make the update touch fewer rows than that, and `sequential_epoch` with more
than one iteration touches each row more than once; none of them moves the
weight. That is a dataset-size convention, the one FedAvg's `n_k` is, and
chapter 07 §3.1 defines it.

The audit's probe (03-final-release-gate, B03) trained four rows in one step
and returned eight. The assertion is on what the server receives: every run
here records `FitResult.num_examples` where `FedAvgServer` checks each result
before folding it in, so a client that reported a different quantity, or a
loop that rewrote it on the way, would fail here and not only in a unit of the
task hook.
"""

from __future__ import annotations

import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any
from unittest import mock

import yaml

from fedbrew.core import runner
from fedbrew.servers.fedavg import FedAvgServer

#: Each client's train split; the in-process synthetic dataset gives every
#: client exactly `samples_per_client` training rows.
TRAIN_ROWS = 12
BATCH_SIZE = 5


def _config(directory: Path, local_iterations: int, client: dict[str, Any]) -> Path:
    raw = yaml.safe_load(Path("configs/dev/smoke.yaml").read_text(encoding="utf-8"))
    raw["experiment"]["output_dir"] = str(directory / "run")
    raw["data"]["samples_per_client"] = TRAIN_ROWS
    raw["defaults"]["local_iterations"] = local_iterations
    raw["client"]["batch_size"] = BATCH_SIZE
    raw["client"]["metrics"] = ["fit_loss", "optimizer_steps"]
    raw["server"]["metrics"] = ["fit_loss", "optimizer_steps"]
    raw["client"].update(client)
    # `local_adamw` refuses the SGD keys the smoke config carries.
    raw["client"] = {key: value for key, value in raw["client"].items() if value is not None}
    raw["runtime"]["checkpointing"] = {"enabled": False}
    path = directory / "run.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def _received(config: Path) -> tuple[dict[str, int], float]:
    """What the server was handed per client, and the round's mean step count."""

    received: dict[str, int] = {}
    original = FedAvgServer._compatible_model_state

    def record(self: FedAvgServer, result: Any) -> Any:
        received[result.client_id] = result.num_examples
        return original(self, result)

    with (
        mock.patch.object(FedAvgServer, "_compatible_model_state", record),
        redirect_stdout(StringIO()),
    ):
        runner.run(config, runner.parse_args(["--quiet"]))
    rounds = (config.parent / "run" / "round_metrics.csv").read_text(encoding="utf-8")
    header, row = (line.split(",") for line in rounds.splitlines()[:2])
    return received, float(row[header.index("optimizer_steps")])


FEDAVG = {"update_rule": "fedavg", "frozen_gradient_weighting": "examples"}


class TheWeightIsTheTrainSplitNotTheRowsTouchedTest(unittest.TestCase):
    #: (case, local_iterations, client keys, optimizer steps, rows the update touched)
    CASES = (
        ("single_batch", 1, {**FEDAVG, "update_mode": "single_batch"}, 1, BATCH_SIZE),
        (
            "sequential_epoch, drop_last, two passes",
            2,
            {**FEDAVG, "update_mode": "sequential_epoch", "drop_last": True},
            4,
            2 * 2 * BATCH_SIZE,
        ),
        (
            "local_adamw, unset update_mode, max_local_steps 1",
            1,
            {
                "update_rule": "local_adamw",
                "max_local_steps": 1,
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1.0e-8,
                "momentum": None,
                "nesterov": None,
            },
            1,
            BATCH_SIZE,
        ),
    )

    def test_each_mode_reports_the_train_split_size(self) -> None:
        for case, local_iterations, client, steps, touched in self.CASES:
            with self.subTest(case), tempfile.TemporaryDirectory() as directory:
                self.assertNotEqual(touched, TRAIN_ROWS)
                received, optimizer_steps = _received(
                    _config(Path(directory), local_iterations, client)
                )
                self.assertEqual(optimizer_steps, float(steps))
                self.assertEqual(received, {"client_0": TRAIN_ROWS, "client_1": TRAIN_ROWS})


if __name__ == "__main__":
    unittest.main()
