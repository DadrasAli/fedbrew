"""One client, one local iteration, one round == plain SGD on the same data.

The identity FedAvg is built on: with a single client holding the whole
dataset, K = 1 local iteration (one pass, for this rule) and full
participation, a federated round is centralized minibatch SGD over one pass.
If that does not hold, nothing built on top of it means what its name says --
every "FedAvg vs centralized" gap in a results table would be measuring the
harness rather than federation.

The comparison is bit-exact on purpose. Both sides run the same ops on the same
batches in the same order, so anything that moves the result -- an extra step,
a dropped batch, a rescaled learning rate, a stray zero_grad, the round's LR
schedule firing when it was configured constant -- shows up as an inequality
rather than as a tolerance judgement call.

train_shuffle is False throughout so the batch order is reproducible by hand;
the shuffled path is what test_reproducibility.py covers.
"""

from __future__ import annotations

import unittest
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from fedbrew.clients.torch_sgd_client import TorchSGDClient
from fedbrew.core.protocol import FitRequest, RoundInfo
from fedbrew.servers.fedavg import FedAvgServer
from fedbrew.tasks.base import TaskAdapter

INPUT_DIM = 3
NUM_CLASSES = 2
NUM_EXAMPLES = 12
LEARNING_RATE = 0.1


class _Task(TaskAdapter):
    """A linear model and a cross-entropy step, with no hidden state.

    Deliberately minimal: the point of the test is the loop around train_step,
    so train_step itself has to be something the centralized reference can
    reproduce exactly, line for line.
    """

    device = torch.device("cpu")

    def build_model(self, config: Mapping[str, Any] | None = None) -> nn.Module:
        torch.manual_seed(11)
        return nn.Linear(INPUT_DIM, NUM_CLASSES)

    def build_dataloader(self, data: Any, config: Mapping[str, Any]) -> DataLoader[Any]:
        return DataLoader(
            TensorDataset(data["x"], data["y"]),
            batch_size=int(config.get("batch_size", NUM_EXAMPLES)),
            shuffle=bool(config.get("shuffle", False)),
            drop_last=bool(config.get("drop_last", False)),
        )

    def train_step(self, model, batch, optimizer=None) -> dict[str, float]:
        if optimizer is None:
            raise ValueError("optimizer is required")
        x, y = batch
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        return {"loss": float(loss.item()), "total": float(len(y))}

    def eval_step(self, model, batch) -> dict[str, float]:
        x, y = batch
        with torch.no_grad():
            logits = model(x)
        return {
            "loss": float(nn.functional.cross_entropy(logits, y).item()),
            "correct": float((logits.argmax(1) == y).sum().item()),
            "total": float(len(y)),
        }

    def compute_metrics(self, outputs: Any, targets: Tensor | None = None) -> dict[str, float]:
        total = sum(float(o["total"]) for o in outputs) or 1.0
        return {
            "loss": sum(float(o["loss"]) * float(o["total"]) for o in outputs) / total,
            "accuracy": sum(float(o["correct"]) for o in outputs) / total,
        }


def _dataset() -> dict[str, Tensor]:
    generator = torch.Generator().manual_seed(5)
    return {
        "x": torch.randn(NUM_EXAMPLES, INPUT_DIM, generator=generator),
        "y": torch.randint(0, NUM_CLASSES, (NUM_EXAMPLES,), generator=generator),
    }


def _initial_state() -> dict[str, Tensor]:
    return {
        key: value.detach().clone() for key, value in _Task().build_model().state_dict().items()
    }


def _client(batch_size: int, local_iterations: int = 1) -> TorchSGDClient:
    split = _dataset()
    return TorchSGDClient(
        client_id="only",
        task=_Task(),
        model_config={},
        # The one client holds the whole dataset, and the same rows are its
        # eval split, so num_examples is the full corpus.
        client_data={"train": split, "eval": split},
        local_iterations=local_iterations,
        batch_size=batch_size,
        learning_rate=LEARNING_RATE,
        momentum=0.0,
        weight_decay=0.0,
        nesterov=False,
        train_shuffle=False,
        drop_last=False,
        # Constant, so _round_learning_rate returns learning_rate for every
        # round and the centralized reference below can use a bare lr.
        learning_rate_schedule="constant",
        min_learning_rate=0.0,
        total_rounds=1,
    )


def _centralized(batch_size: int, epochs: int = 1) -> dict[str, Tensor]:
    """Plain SGD over the same data, written out longhand."""

    task = _Task()
    model = task.build_model()
    model.load_state_dict(_initial_state())
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    loader = DataLoader(TensorDataset(*_dataset().values()), batch_size=batch_size, shuffle=False)
    for _ in range(epochs):
        for x, y in loader:
            optimizer.zero_grad()
            nn.functional.cross_entropy(model(x), y).backward()
            optimizer.step()
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _assert_identical(left: Mapping[str, Tensor], right: Mapping[str, Tensor]) -> None:
    assert set(left) == set(right), (set(left), set(right))
    for key in left:
        torch.testing.assert_close(left[key], right[key], rtol=0.0, atol=0.0)


class SingleClientIsCentralizedTests(unittest.TestCase):
    def test_one_full_batch_epoch_is_one_centralized_sgd_step(self) -> None:
        client = _client(batch_size=NUM_EXAMPLES)
        result = client.fit(
            FitRequest(round_id=1, client_id="only", payload={"model_state": _initial_state()})
        )
        self.assertEqual(result.metrics["optimizer_steps"], 1.0)
        _assert_identical(result.payload["model_state"], _centralized(NUM_EXAMPLES))

    def test_one_minibatch_epoch_is_one_centralized_pass(self) -> None:
        for batch_size in (1, 4, 5):
            with self.subTest(batch_size=batch_size):
                client = _client(batch_size=batch_size)
                result = client.fit(
                    FitRequest(
                        round_id=1,
                        client_id="only",
                        payload={"model_state": _initial_state()},
                    )
                )
                # 5 does not divide 12: the ragged last batch must be taken,
                # not dropped, or this is not one pass over the data.
                expected_steps = -(-NUM_EXAMPLES // batch_size)
                self.assertEqual(result.metrics["optimizer_steps"], float(expected_steps))
                _assert_identical(result.payload["model_state"], _centralized(batch_size))

    def test_e_local_iterations_are_e_centralized_passes(self) -> None:
        client = _client(batch_size=4, local_iterations=3)
        result = client.fit(
            FitRequest(round_id=1, client_id="only", payload={"model_state": _initial_state()})
        )
        _assert_identical(result.payload["model_state"], _centralized(4, epochs=3))

    def test_a_federated_round_over_the_one_client_is_that_client(self) -> None:
        """The client identity is only half of it: the server must not touch it.

        Closes the loop with tests/test_aggregation_correctness.py -- a single
        client's weighted mean is the client -- through the real server path,
        so a round is centralized SGD end to end and not just inside fit().
        """

        client = _client(batch_size=NUM_EXAMPLES)
        result = client.fit(
            FitRequest(round_id=1, client_id="only", payload={"model_state": _initial_state()})
        )
        server = FedAvgServer(participation_rate=1.0, seed=0)
        server._model_state = _initial_state()
        server._model_state_scope = "full"
        server._model_state_metadata = {"model_state_scope": "full"}
        server.aggregate(RoundInfo(round_id=1), [result])

        centralized = _centralized(NUM_EXAMPLES)
        for key, expected in centralized.items():
            # One float32 multiply-then-divide by num_examples, as in
            # test_aggregation_properties.test_one_client_is_returned_unchanged.
            torch.testing.assert_close(server._model_state[key], expected, rtol=1e-6, atol=1e-6)

    def test_the_client_reports_the_whole_dataset_as_its_weight(self) -> None:
        # If this drifts, the round is still centralized SGD but the identity
        # stops being observable from the server's side.
        client = _client(batch_size=NUM_EXAMPLES)
        result = client.fit(
            FitRequest(round_id=1, client_id="only", payload={"model_state": _initial_state()})
        )
        self.assertEqual(result.num_examples, NUM_EXAMPLES)

    def test_the_reference_actually_moves_the_model(self) -> None:
        """Guards every assertion above against comparing two no-ops."""

        initial = _initial_state()
        trained = _centralized(NUM_EXAMPLES)
        self.assertFalse(all(torch.equal(initial[key], trained[key]) for key in initial))


if __name__ == "__main__":
    unittest.main()
