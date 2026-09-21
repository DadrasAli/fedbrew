"""SCAFFOLD and FedProx now work with any TaskAdapter, not just classification.

Before the correcting-optimizer refactor, both hand-rolled forward/loss/
backward against ``task._move_batch`` and ``task._criterion`` -- two private
attributes named after, and defined only on, TorchClassificationTask. Both
classes were generic-parameterized to it specifically
(``TorchSGDClient[TorchClassificationTask]``), not to ``TaskAdapter``, and
TorchCausalLMTask has no ``_criterion`` at all (it uses the differently
shaped ``_loss_and_counts``, returning ``(loss, correct, total)``) and no
``_scaler``. ``client.update_rule: scaffold`` with ``task.name: causal_lm``
was refused nowhere and would have raised ``AttributeError`` on the first
round.

The refactor calls ``task.train_step(model, batch, optimizer)`` -- the one
method every ``TaskAdapter`` already implements -- through an optimizer that
corrects ``.grad`` in its own ``.step()``. Nothing about that reach depends on
which concrete task is underneath, so the finding is resolved as a
consequence of the refactor, not by a task-specific pairing check.

Two kinds of evidence:

``AnyConformingTaskTest`` uses a hand-rolled ``TaskAdapter`` that implements
only the five abstract methods -- no ``_move_batch``, ``_criterion`` or
``_scaler`` anywhere on it -- and was already relied on by
test_scaffold_fedprox_communication_cost.py before this file existed.

``CausalLMTaskTest`` goes further: the real ``TorchCausalLMTask``, not a
stand-in, with a tiny transformers-free model swapped in for build_model so
the test does not need the optional ``transformers`` dependency to exercise
the task's real train_step/_move_batch/_model_inputs/_loss_and_counts.
"""

from __future__ import annotations

import unittest
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from fedbrew.clients.torch_fedprox_client import TorchFedProxClient
from fedbrew.clients.torch_scaffold_client import TorchScaffoldClient
from fedbrew.core.protocol import FitRequest
from fedbrew.core.torch_utils import get_model_state
from fedbrew.tasks.base import TaskAdapter
from fedbrew.tasks.causal_lm.torch_causal_lm import TorchCausalLMTask


class _MinimalTask(TaskAdapter):
    """Only the five abstract methods. No _move_batch, _criterion or
    _scaler anywhere -- the shape a third-party task actually has."""

    device = torch.device("cpu")

    def build_model(self, config: Mapping[str, Any] | None = None) -> nn.Module:
        torch.manual_seed(0)
        return nn.Linear(2, 2)

    def build_dataloader(self, data: Any, config: Mapping[str, Any]) -> DataLoader[Any]:
        dataset = TensorDataset(data["X"], data["y"])
        return DataLoader(dataset, batch_size=int(config.get("batch_size", 4)), shuffle=False)

    def train_step(self, model: nn.Module, batch: Any, optimizer: Any = None) -> dict[str, float]:
        x, y = batch
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        return {"loss": float(loss.item()), "total": float(len(y))}

    def eval_step(self, model: nn.Module, batch: Any) -> dict[str, float]:
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
        return {"loss": sum(float(o["loss"]) * float(o["total"]) for o in outputs) / total}


def _minimal_task_data() -> dict[str, dict[str, torch.Tensor]]:
    split = {"X": torch.randn(4, 2), "y": torch.zeros(4, dtype=torch.long)}
    return {"train": split, "eval": split}


class AnyConformingTaskTest(unittest.TestCase):
    def test_scaffold_runs_with_a_task_that_defines_only_the_abstract_methods(self) -> None:
        client = TorchScaffoldClient(
            client_id="c0",
            task=_MinimalTask(),
            model_config={},
            client_data=_minimal_task_data(),
            local_iterations=1,
            batch_size=4,
            learning_rate=0.1,
            train_shuffle=False,
        )
        model = client.task.build_model()
        state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        result = client.fit(
            FitRequest(
                round_id=1,
                client_id="c0",
                payload={
                    "model_state": state,
                    "server_control": {n: torch.zeros_like(v) for n, v in state.items()},
                },
            )
        )
        self.assertIn("fit_loss", result.metrics)

    def test_fedprox_runs_with_a_task_that_defines_only_the_abstract_methods(self) -> None:
        client = TorchFedProxClient(
            client_id="c0",
            task=_MinimalTask(),
            model_config={},
            client_data=_minimal_task_data(),
            local_iterations=1,
            batch_size=4,
            learning_rate=0.1,
            train_shuffle=False,
            proximal_mu=0.1,
        )
        model = client.task.build_model()
        state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        result = client.fit(FitRequest(round_id=1, client_id="c0", payload={"model_state": state}))
        self.assertIn("fit_proximal_loss", result.metrics)


class _TinyLMModel(nn.Module):
    """A stand-in for an HF causal LM: takes input_ids (+ optional
    attention_mask), returns a plain (N, seq_len, vocab) logits tensor --
    exactly what _extract_logits accepts without an HF ModelOutput wrapper."""

    def __init__(self, vocab_size: int, hidden: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        del attention_mask
        return self.head(self.embed(input_ids))


class _TinyCausalLMTask(TorchCausalLMTask):
    """The real TorchCausalLMTask -- real _move_batch, _model_inputs,
    _loss_and_counts, train_step -- over a transformers-free model, so this
    does not need the optional transformers dependency installed."""

    def build_model(self, config: Mapping[str, Any] | None = None) -> nn.Module:
        torch.manual_seed(0)
        return _TinyLMModel(vocab_size=16, hidden=8)


def _causal_lm_data() -> dict[str, dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(1)
    inputs = torch.randint(1, 16, (4, 5), generator=generator)
    targets = torch.randint(1, 16, (4, 5), generator=generator)
    split = {"X": inputs, "y": targets}
    return {"train": split, "eval": split}


class CausalLMTaskTest(unittest.TestCase):
    """The task the finding named specifically, not a stand-in for it."""

    def test_scaffold_runs_with_the_real_causal_lm_task(self) -> None:
        client = TorchScaffoldClient(
            client_id="c0",
            task=_TinyCausalLMTask(model_config={}, batch_size=4, device="cpu"),
            model_config={},
            client_data=_causal_lm_data(),
            local_iterations=1,
            batch_size=4,
            learning_rate=0.1,
            train_shuffle=False,
        )
        model = client.task.build_model()
        state = get_model_state(model)
        result = client.fit(
            FitRequest(
                round_id=1,
                client_id="c0",
                payload={
                    "model_state": state,
                    "server_control": {n: torch.zeros_like(v) for n, v in state.items()},
                },
            )
        )
        self.assertIn("fit_loss", result.metrics)

    def test_fedprox_runs_with_the_real_causal_lm_task(self) -> None:
        client = TorchFedProxClient(
            client_id="c0",
            task=_TinyCausalLMTask(model_config={}, batch_size=4, device="cpu"),
            model_config={},
            client_data=_causal_lm_data(),
            local_iterations=1,
            batch_size=4,
            learning_rate=0.1,
            train_shuffle=False,
            proximal_mu=0.1,
        )
        model = client.task.build_model()
        state = get_model_state(model)
        result = client.fit(FitRequest(round_id=1, client_id="c0", payload={"model_state": state}))
        self.assertIn("fit_proximal_loss", result.metrics)


if __name__ == "__main__":
    unittest.main()
