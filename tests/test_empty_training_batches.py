"""A client that trains on nothing must say so, not report success.

With drop_last: true a client holding fewer than batch_size training examples
produces an empty loader. Five of the seven update rules raise on that; local_sgd
and fedprox hand-roll their epoch loop and ran the body zero times instead --
returning the global state they were sent, at the client's full aggregation
weight, with the run exiting "completed". Measured on the 3-client synthetic
dataset at batch_size 32 with 20 train examples each, fit_loss was bit-identical
across two rounds (0.74479769 both times) and both runs reported
"Finished successfully".
"""

from __future__ import annotations

import unittest
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from fedbrew.clients.torch_fedprox_client import TorchFedProxClient
from fedbrew.clients.torch_sgd_client import TorchSGDClient
from fedbrew.core.protocol import FitRequest
from fedbrew.tasks.base import TaskAdapter


class _Task(TaskAdapter):
    device = torch.device("cpu")

    def build_model(self, config: Mapping[str, Any] | None = None) -> nn.Module:
        torch.manual_seed(0)
        return nn.Linear(1, 2)

    def build_dataloader(self, data: Any, config: Mapping[str, Any]) -> DataLoader[Any]:
        dataset = TensorDataset(data["x"], data["y"])
        return DataLoader(
            dataset,
            batch_size=int(config.get("batch_size", 4)),
            shuffle=False,
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
            loss = nn.functional.cross_entropy(logits, y)
        return {
            "loss": float(loss.item()),
            "correct": float((logits.argmax(1) == y).sum().item()),
            "total": float(len(y)),
        }

    def compute_metrics(self, outputs: Any, targets: Tensor | None = None) -> dict[str, float]:
        total = sum(float(o["total"]) for o in outputs) or 1.0
        return {
            "loss": sum(float(o["loss"]) * float(o["total"]) for o in outputs) / total,
            "accuracy": sum(float(o["correct"]) for o in outputs) / total,
        }


def _data(size: int) -> dict[str, dict[str, Tensor]]:
    split = {"x": torch.ones(size, 1), "y": torch.zeros(size, dtype=torch.long)}
    return {"train": split, "eval": split}


#: The keys TorchFedProxClient accepts; the base client takes more.
_FEDPROX_KEYS = {
    "client_id",
    "task",
    "model_config",
    "client_data",
    "local_iterations",
    "batch_size",
    "learning_rate",
    "drop_last",
    "train_shuffle",
}


def _fedprox_kwargs(**kwargs: Any) -> dict[str, Any]:
    return {k: v for k, v in kwargs.items() if k in _FEDPROX_KEYS}


def _kwargs(train_size: int, batch_size: int, drop_last: bool) -> dict[str, Any]:
    return {
        "client_id": "c0",
        "task": _Task(),
        "model_config": {},
        "client_data": _data(train_size),
        "local_iterations": 2,
        "batch_size": batch_size,
        "learning_rate": 0.1,
        "momentum": 0.0,
        "weight_decay": 0.0,
        "nesterov": False,
        "drop_last": drop_last,
        "train_shuffle": False,
        "learning_rate_schedule": "constant",
        "min_learning_rate": 0.0,
        "total_rounds": 1,
    }


def _request() -> FitRequest:
    torch.manual_seed(0)
    state = {k: v.detach().clone() for k, v in nn.Linear(1, 2).state_dict().items()}
    return FitRequest(round_id=1, client_id="c0", payload={"model_state": state})


class EmptyLoaderIsRefusedTests(unittest.TestCase):
    """A client with zero training batches must be refused.

    It used to return the unchanged global model and report success, so the
    round folded a no-op in at full weight and nothing said so.
    """

    def test_local_sgd_refuses_a_client_with_no_batches(self) -> None:
        client = TorchSGDClient(**_kwargs(train_size=20, batch_size=32, drop_last=True))

        with self.assertRaises(ValueError) as caught:
            client.fit(_request())

        message = str(caught.exception)
        self.assertIn("has no training batches", message)
        # The three numbers that explain why, so the message is actionable.
        self.assertIn("batch_size 32", message)
        self.assertIn("drop_last True", message)

    def test_fedprox_refuses_a_client_with_no_batches(self) -> None:
        client = TorchFedProxClient(
            **_fedprox_kwargs(**_kwargs(train_size=20, batch_size=32, drop_last=True)),
            proximal_mu=0.1,
        )

        with self.assertRaises(ValueError) as caught:
            client.fit(_request())

        self.assertIn("has no training batches", str(caught.exception))

    def test_an_empty_train_split_is_refused_without_drop_last(self) -> None:
        """The same path is reached by any client whose train split is empty."""

        client = TorchSGDClient(**_kwargs(train_size=0, batch_size=4, drop_last=False))

        with self.assertRaises(ValueError):
            client.fit(_request())

    def test_a_client_that_does_train_is_unaffected(self) -> None:
        base = _kwargs(train_size=64, batch_size=32, drop_last=True)
        for rule in ("local_sgd", "fedprox"):
            with self.subTest(rule=rule):
                client = (
                    TorchSGDClient(**base)
                    if rule == "local_sgd"
                    else TorchFedProxClient(**_fedprox_kwargs(**base), proximal_mu=0.1)
                )
                self.assertEqual(client.fit(_request()).num_examples, 64)

    def test_a_sub_batch_client_is_fine_when_drop_last_is_off(self) -> None:
        client = TorchSGDClient(**_kwargs(train_size=20, batch_size=32, drop_last=False))

        self.assertEqual(client.fit(_request()).num_examples, 20)


if __name__ == "__main__":
    unittest.main()
