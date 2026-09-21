"""A decaying schedule may reach zero; the run must still finish its last round.

`client.min_learning_rate: 0.0` is inside the range `config.py` documents, and a
cosine schedule that anneals to zero at the horizon is the standard form of the
thing. `_round_learning_rate` returns exactly `min_learning_rate` when
`round_id == total_rounds`, and `run_sgd_update_mode` refuses a non-positive
rate -- so that pair used to raise `ValueError: learning_rate must be positive`
on the **final** round, after every round before it had been spent.

The cost is not the exception. It is what the exception leaves behind: the
artifacts on disk are a complete-looking shorter run. `round_metrics.csv` holds
`global_rounds - 1` rows, `run.json` still says `status: running` with
`num_rounds` equal to the rounds that did finish, `checkpoints/latest.pt` is the
second-to-last round's, and nothing on disk says the run failed. A sweep
collector reads a finished run of the wrong length.

So the fix is a floor rather than a refusal at config load -- a config that
would have worked for 499 of 500 rounds should not fail to load -- and it is
`local_update_modes.MIN_POSITIVE_LEARNING_RATE`, the smallest positive float,
placed beside the guard that requires positivity. It satisfies the guard and
steps by nothing, which is what a schedule that reached zero asked for.

FINDINGS.csv POST-F01.
"""

from __future__ import annotations

import math
import unittest

import pytest
import torch

from fedbrew.clients.fedavg_client import FedAvgClient
from fedbrew.clients.local_update_modes import MIN_POSITIVE_LEARNING_RATE
from fedbrew.clients.torch_sgd_client import TorchSGDClient
from fedbrew.core.protocol import ClientInfo, FitRequest
from fedbrew.core.torch_utils import get_model_state
from tests.test_empty_training_batches import _data, _Task

TOTAL_ROUNDS = 5
PEAK = 0.1

SGD_SETTINGS = {
    "momentum": 0.0,
    "weight_decay": 0.0,
    "nesterov": False,
    "total_rounds": TOTAL_ROUNDS,
}


def _sgd_client(**overrides: object) -> TorchSGDClient:
    settings: dict[str, object] = {
        "client_id": "client_0",
        "task": _Task(),
        "model_config": {},
        "client_data": _data(4),
        "local_iterations": 1,
        "batch_size": 2,
        "learning_rate": PEAK,
        "learning_rate_schedule": "cosine",
        "min_learning_rate": 0.0,
        **SGD_SETTINGS,
    }
    settings.update(overrides)
    return TorchSGDClient(**settings)  # type: ignore[arg-type]


class ScheduleFloorTest(unittest.TestCase):
    @pytest.mark.fast
    def test_the_final_round_rate_is_positive_when_the_schedule_reaches_zero(self) -> None:
        client = _sgd_client()

        self.assertEqual(client._round_learning_rate(TOTAL_ROUNDS), MIN_POSITIVE_LEARNING_RATE)
        self.assertGreater(client._round_learning_rate(TOTAL_ROUNDS), 0.0)

    @pytest.mark.fast
    def test_the_floor_is_the_smallest_positive_float_and_so_steps_by_nothing(self) -> None:
        """A larger floor would be a step size nobody configured.

        The value is not a chosen magnitude, it is the smallest one that
        satisfies the guard -- so the clamped round is the no-op the schedule
        asked for. `p - lr*g` is `p` in float64 for any `|g|` below about
        5e291, which is every gradient a run produces and then some;
        `test_the_floored_round_leaves_the_model_where_it_found_it` checks the
        same thing through a real step.
        """

        self.assertGreater(MIN_POSITIVE_LEARNING_RATE, 0.0)
        for magnitude in (1.0, 1e6, 1e200):
            with self.subTest(gradient=magnitude):
                self.assertEqual(1.0 - MIN_POSITIVE_LEARNING_RATE * magnitude, 1.0)

    @pytest.mark.fast
    def test_no_other_round_moves(self) -> None:
        """The clamp may only bite where the schedule is already below it.

        Checked against the closed form rather than against a recorded list, so
        a change to the schedule itself fails here instead of being absorbed.
        """

        client = _sgd_client()

        for round_id in range(1, TOTAL_ROUNDS):
            progress = (round_id - 1) / (TOTAL_ROUNDS - 1)
            expected = PEAK * 0.5 * (1.0 + math.cos(math.pi * progress))
            with self.subTest(round_id=round_id):
                self.assertAlmostEqual(client._round_learning_rate(round_id), expected)

    @pytest.mark.fast
    def test_a_positive_minimum_reaches_that_minimum_untouched(self) -> None:
        """The floor is far below any rate anyone configures, so it never rounds one up."""

        client = _sgd_client(min_learning_rate=PEAK / 100.0)

        self.assertAlmostEqual(client._round_learning_rate(TOTAL_ROUNDS), PEAK / 100.0)

    @pytest.mark.fast
    def test_a_configured_zero_is_still_refused(self) -> None:
        """The floor is for a schedule that arrived at zero, not for a config that says so."""

        with self.assertRaises(ValueError):
            _sgd_client(learning_rate=0.0, learning_rate_schedule="constant")

    def test_fedavg_completes_the_final_round(self) -> None:
        """The regression itself: this used to raise on round TOTAL_ROUNDS."""

        client = FedAvgClient(
            client_id="client_0",
            task=_Task(),
            model_config={},
            client_data=_data(4),
            local_iterations=1,
            batch_size=2,
            learning_rate=PEAK,
            learning_rate_schedule="cosine",
            min_learning_rate=0.0,
            update_mode="sequential_epoch",
            frozen_gradient_weighting="examples",
            **SGD_SETTINGS,
        )
        client.setup(ClientInfo(client_id="client_0", num_examples=4))
        payload = {"model_state": get_model_state(_Task().build_model())}

        request = FitRequest(round_id=TOTAL_ROUNDS, client_id="client_0", payload=payload)
        result = client.fit(request)

        self.assertEqual(result.round_id, TOTAL_ROUNDS)
        self.assertEqual(
            result.metrics["client_learning_rate"],
            MIN_POSITIVE_LEARNING_RATE,
        )

    def test_the_floored_round_leaves_the_model_where_it_found_it(self) -> None:
        """A no-op step, not a small one: the last round must change nothing."""

        client = FedAvgClient(
            client_id="client_0",
            task=_Task(),
            model_config={},
            client_data=_data(4),
            local_iterations=1,
            batch_size=2,
            learning_rate=PEAK,
            learning_rate_schedule="cosine",
            min_learning_rate=0.0,
            update_mode="sequential_epoch",
            frozen_gradient_weighting="examples",
            **SGD_SETTINGS,
        )
        client.setup(ClientInfo(client_id="client_0", num_examples=4))
        before = get_model_state(_Task().build_model())

        result = client.fit(
            FitRequest(round_id=TOTAL_ROUNDS, client_id="client_0", payload={"model_state": before})
        )

        after = result.payload["model_state"]
        for name, value in before.items():
            with self.subTest(parameter=name):
                self.assertTrue(torch.equal(value, after[name]))


if __name__ == "__main__":
    unittest.main()
