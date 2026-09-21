"""FedAvg + local fine-tuning at evaluation time: a personalization baseline.

Training is FedAvg unchanged. At evaluation the client fine-tunes the global
model on its OWN train split, measures the result, and throws it away -- so
the arm keeps no per-client state between rounds.

Two properties carry the weight here:

* **No leakage.** Fine-tuning reads the train split and nothing else. A client
  that personalized on val or test and then reported a score there would not
  be a weak baseline, it would be a wrong one.
* **No contamination.** runtime.performance.reuse_model makes build_model
  return one cached module per architecture, so the personalized pass mutates
  the very object the global pass reads. The global numbers under
  model_scope: both must be identical to those under model_scope: global.
"""

from __future__ import annotations

import copy
import unittest
from collections.abc import Mapping
from typing import Any

import pytest
import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader, TensorDataset

from fedbrew.clients.fedavg_ft_client import FedAvgFTClient
from fedbrew.core.config import (
    PERSONAL_SPLIT_PREFIX,
    load_config,
    validate_config,
)
from fedbrew.core.factory import build_components
from fedbrew.core.protocol import EvalRequest
from fedbrew.core.validation import validate_full_config
from fedbrew.tasks.base import TaskAdapter

#: Each split's features carry its own marker value, so anything the trainer
#: touches can be traced back to the split it came from.
_TRAIN_MARKER = 1.0
_VAL_MARKER = 2.0
_TEST_MARKER = 3.0
_MARKER_SPLITS = {_TRAIN_MARKER: "train", _VAL_MARKER: "val", _TEST_MARKER: "test"}

LEARNING_RATE = 0.5


class _MarkedTask(TaskAdapter):
    """Two-class task that records which split every training batch came from."""

    device = torch.device("cpu")

    def __init__(self) -> None:
        self.trained_on_splits: list[str] = []

    def build_model(self, config: Mapping[str, Any] | None = None) -> nn.Module:
        torch.manual_seed(0)
        model = nn.Linear(1, 2, bias=True)
        with torch.no_grad():
            # Confidently predicts class 0, which is wrong for every client
            # below. Fine-tuning has somewhere to go.
            model.weight.zero_()
            model.bias.copy_(torch.tensor([4.0, -4.0]))
        return model

    def build_dataloader(self, data: Any, config: Mapping[str, Any]) -> DataLoader[Any]:
        dataset = TensorDataset(data["x"], data["y"])
        return DataLoader(dataset, batch_size=int(config.get("batch_size", 4)), shuffle=False)

    def train_step(
        self,
        model: nn.Module,
        batch: Any,
        optimizer: optim.Optimizer | None = None,
    ) -> dict[str, float]:
        if optimizer is None:
            raise ValueError("optimizer is required")
        features, targets = batch
        for marker in features.flatten().unique().tolist():
            self.trained_on_splits.append(_MARKER_SPLITS[round(marker, 3)])
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(features), targets)
        loss.backward()
        optimizer.step()
        return {"loss": float(loss.detach()), "correct": 0.0, "total": float(targets.numel())}

    def eval_step(self, model: nn.Module, batch: Any) -> dict[str, float]:
        features, targets = batch
        with torch.no_grad():
            logits = model(features)
            loss = nn.functional.cross_entropy(logits, targets)
            correct = float((logits.argmax(dim=-1) == targets).sum())
        return {"loss": float(loss), "correct": correct, "total": float(targets.numel())}

    def compute_metrics(self, outputs: Any, targets: Tensor | None = None) -> dict[str, float]:
        total = sum(float(o["total"]) for o in outputs) or 1.0
        return {
            "loss": sum(float(o["loss"]) * float(o["total"]) for o in outputs) / total,
            "accuracy": sum(float(o["correct"]) for o in outputs) / total,
        }


def _split(marker: float, label: int, size: int = 8) -> dict[str, Tensor]:
    """One split in the shape the client-data helpers expect: {"x": ..., "y": ...}."""

    return {
        "x": torch.full((size, 1), marker),
        "y": torch.full((size,), label, dtype=torch.long),
    }


class _CachingMarkedTask(_MarkedTask):
    """One module for every caller, as runtime.performance.reuse_model gives.

    The plain _MarkedTask builds a fresh nn.Linear per call, which hides every
    defect that depends on two clients sharing a module.
    """

    def __init__(self) -> None:
        super().__init__()
        self._cached: nn.Module | None = None

    def build_model(self, config: Mapping[str, Any] | None = None) -> nn.Module:
        if self._cached is None:
            self._cached = super().build_model(config)
        return self._cached


def _client_data() -> dict[str, dict[str, Tensor]]:
    """Every split is class 1, which the initial model never predicts.

    train/val/test carry different marker features so training on the wrong one
    is detectable, but the same label, so fine-tuning on train is genuinely
    expected to help on test.
    """

    return {
        "train": _split(_TRAIN_MARKER, label=1),
        "eval": _split(_VAL_MARKER, label=1),
        "test": _split(_TEST_MARKER, label=1),
    }


def _client(task: _MarkedTask | None = None, **overrides: Any) -> FedAvgFTClient:
    kwargs: dict[str, Any] = {
        "client_id": "c0",
        "task": task or _MarkedTask(),
        "model_config": {},
        "client_data": _client_data(),
        "local_iterations": 1,
        "batch_size": 4,
        "learning_rate": LEARNING_RATE,
        "momentum": 0.0,
        "weight_decay": 0.0,
        "nesterov": False,
        "update_mode": "sequential_epoch",
        "frozen_gradient_weighting": "examples",
        "finetune_epochs": 5,
        "train_shuffle": False,
    }
    kwargs.update(overrides)
    return FedAvgFTClient(**kwargs)


def _global_model_state() -> dict[str, Tensor]:
    """The broadcast state: the confidently-wrong initial model."""

    return {
        name: tensor.detach().clone()
        for name, tensor in _MarkedTask().build_model().state_dict().items()
    }


def _request(model_scope: str, splits: list[str] | None = None) -> EvalRequest:
    payload: dict[str, Any] = {
        "model_state": _global_model_state(),
        "model_scope": model_scope,
        "metrics": ["loss", "accuracy"],
    }
    if splits is not None:
        payload["splits"] = splits
    return EvalRequest(round_id=1, client_id="c0", payload=payload)


class LeakageTest(unittest.TestCase):
    """The property that must never regress."""

    def test_finetuning_reads_the_train_split_and_nothing_else(self) -> None:
        task = _MarkedTask()
        _client(task).evaluate(_request("personal", ["train", "val", "test"]))

        self.assertTrue(task.trained_on_splits, "fine-tuning never ran")
        self.assertEqual(set(task.trained_on_splits), {"train"})

    @pytest.mark.fast
    def test_the_split_is_not_configurable(self) -> None:
        """_finetune takes no split argument, so no config can redirect it."""

        import inspect

        parameters = set(inspect.signature(FedAvgFTClient._finetune).parameters)
        self.assertEqual(parameters, {"self", "model", "request"})


class PersonalizationTest(unittest.TestCase):
    def test_finetuning_improves_the_client_test_score(self) -> None:
        """The initial model always predicts class 0; every client is class 1.
        A personalized pass that did nothing would score 0 like the global one."""

        result = _client().evaluate(_request("both", ["test"]))
        self.assertEqual(result.metrics["test_accuracy"], 0.0)
        self.assertEqual(result.metrics["personal_test_accuracy"], 1.0)
        self.assertLess(result.metrics["personal_test_loss"], result.metrics["test_loss"])

    def test_the_global_pass_is_not_contaminated_by_the_personal_one(self) -> None:
        """reuse_model hands both passes the same module, so the order in
        evaluate() is load-bearing: global first, then fine-tune in place."""

        global_only = _client().evaluate(_request("global", ["test"]))
        both = _client().evaluate(_request("both", ["test"]))
        for name in ("test_loss", "test_accuracy"):
            with self.subTest(metric=name):
                self.assertAlmostEqual(both.metrics[name], global_only.metrics[name])

    def test_each_scope_emits_only_its_own_columns(self) -> None:
        personal = _client().evaluate(_request("personal", ["test"]))
        self.assertIn("personal_test_accuracy", personal.metrics)
        self.assertNotIn("test_accuracy", personal.metrics)

        global_only = _client().evaluate(_request("global", ["test"]))
        self.assertIn("test_accuracy", global_only.metrics)
        self.assertFalse([n for n in global_only.metrics if n.startswith(PERSONAL_SPLIT_PREFIX)])

    def test_counts_are_reported_under_the_prefixed_split(self) -> None:
        result = _client().evaluate(_request("both", ["train", "test"]))
        counts = result.payload["num_examples_by_split"]
        for split in ("train", "test"):
            self.assertEqual(counts[split], counts[f"{PERSONAL_SPLIT_PREFIX}{split}"])

    @pytest.mark.fast
    def test_a_non_global_scope_needs_an_explicit_split_list(self) -> None:
        """The single-split path reports one unprefixed metric set, which has
        nowhere to put a second scope's numbers."""

        with self.assertRaises(ValueError):
            _client().evaluate(_request("personal"))

    def test_more_finetune_epochs_do_more_work(self) -> None:
        one = _MarkedTask()
        five = _MarkedTask()
        _client(one, finetune_epochs=1).evaluate(_request("personal", ["test"]))
        _client(five, finetune_epochs=5).evaluate(_request("personal", ["test"]))
        self.assertEqual(len(five.trained_on_splits), 5 * len(one.trained_on_splits))


class FinetuningDoesNotReachTrainingTests(unittest.TestCase):
    """Regression tests for evaluation moving the training stream.

    _finetune runs real SGD on the evaluation path, in train() mode, so every
    fine-tuned client used to draw from the process-wide generator the NEXT
    round's training draws from. That made evaluation.val.clients a
    hyperparameter of the training curve rather than the pure cost knob its
    config comment describes: on FEMNIST, sample:4 and sample:8 gave the global
    model different fit_accuracy, and the arm stopped matching the plain-fedavg
    baseline it is paired with at the seed they share.
    """

    def _stream_after_evaluating(self, clients: int, scope: str = "both") -> float:
        """The next draw from the global stream after evaluating N clients."""

        torch.manual_seed(99)
        task = _CachingMarkedTask()
        state = _global_model_state()
        for index in range(clients):
            request = EvalRequest(
                round_id=1,
                client_id=f"c{index}",
                payload={
                    "model_state": state,
                    "model_scope": scope,
                    "metrics": ["loss", "accuracy"],
                    "splits": ["test"],
                },
            )
            _client(task=task, client_id=f"c{index}", base_seed=7).evaluate(request)
        return float(torch.rand(1).item())

    def test_the_number_of_evaluated_clients_does_not_move_the_stream(self) -> None:
        self.assertEqual(self._stream_after_evaluating(2), self._stream_after_evaluating(8))

    def test_a_personal_pass_leaves_the_stream_where_a_global_pass_does(self) -> None:
        """model_scope must be a measurement choice, not a training input."""

        self.assertEqual(
            self._stream_after_evaluating(3, scope="both"),
            self._stream_after_evaluating(3, scope="global"),
        )

    def test_fine_tuning_is_still_reproducible_within_the_fork(self) -> None:
        """Forking must not make the personalized number random run to run."""

        def personal_losses() -> list[float]:
            torch.manual_seed(99)
            task = _CachingMarkedTask()
            state = _global_model_state()
            out = []
            for index in range(3):
                request = EvalRequest(
                    round_id=1,
                    client_id=f"c{index}",
                    payload={
                        "model_state": state,
                        "model_scope": "both",
                        "metrics": ["loss", "accuracy"],
                        "splits": ["test"],
                    },
                )
                result = _client(task=task, client_id=f"c{index}", base_seed=7).evaluate(request)
                out.append(result.metrics["personal_test_loss"])
            return out

        self.assertEqual(personal_losses(), personal_losses())

    def test_a_client_without_a_base_seed_still_fine_tunes(self) -> None:
        result = _client(base_seed=None).evaluate(_request("both", ["test"]))

        self.assertLess(result.metrics["personal_test_loss"], result.metrics["test_loss"])


class SharedModuleContaminationTest(unittest.TestCase):
    """The two conditions a real FEMNIST round has, and the unit tests did not.

    reuse_model=True means one cached module for every client in the round, and
    the loop shallow-copies one server payload per round, so every request
    carries the *same* model_state object. Together those are what let an
    identity-keyed "already resident" check skip a load.
    """

    def _evaluate_round(self, scope: str) -> list[dict[str, float]]:
        """Three clients through one module and one payload object, as the loop does."""

        task = _CachingMarkedTask()
        model_state = _global_model_state()
        metrics = []
        for client_id in ("c0", "c1", "c2"):
            request = EvalRequest(
                round_id=1,
                client_id=client_id,
                # Shallow, exactly like loop.py's per-request payload: a fresh
                # dict whose "model_state" is the one shared object.
                payload={
                    "model_state": model_state,
                    "model_scope": scope,
                    "metrics": ["loss", "accuracy"],
                    "splits": ["test"],
                },
            )
            metrics.append(_client(task=task, client_id=client_id).evaluate(request).metrics)
        return metrics

    def test_every_client_global_pass_sees_the_unmodified_global_model(self) -> None:
        """The global model does not change within a round, so neither may its score.

        Regression test: _finetune mutated a module that
        had been loaded with mutates=False without retracting the residency
        claim, so clients 2..N skipped their load and were measured on client
        1's fine-tuned weights -- monotonically improving, and wrong.
        """

        baseline = _client().evaluate(_request("global", ["test"])).metrics
        for index, metrics in enumerate(self._evaluate_round("both")):
            with self.subTest(client=index):
                self.assertAlmostEqual(metrics["test_loss"], baseline["test_loss"])
                self.assertAlmostEqual(metrics["test_accuracy"], baseline["test_accuracy"])

    def test_the_personal_pass_still_personalizes(self) -> None:
        """Guards the test above from passing because nothing fine-tunes at all."""

        for index, metrics in enumerate(self._evaluate_round("both")):
            with self.subTest(client=index):
                self.assertLess(metrics["personal_test_loss"], metrics["test_loss"])

    def test_a_personal_only_round_does_not_carry_across_clients(self) -> None:
        """model_scope: personal takes the same path with no global pass to anchor it."""

        losses = [m["personal_test_loss"] for m in self._evaluate_round("personal")]
        for index, loss in enumerate(losses[1:], start=1):
            with self.subTest(client=index):
                self.assertAlmostEqual(loss, losses[0])


class ConfigurationTest(unittest.TestCase):
    @pytest.mark.fast
    def test_the_finetune_rate_defaults_to_the_training_rate(self) -> None:
        self.assertEqual(_client().finetune_learning_rate, LEARNING_RATE)
        self.assertEqual(_client(finetune_learning_rate=0.01).finetune_learning_rate, 0.01)

    @pytest.mark.fast
    def test_invalid_finetune_settings_are_rejected(self) -> None:
        for override in (
            {"finetune_epochs": 0},
            {"finetune_epochs": 1.5},
            {"finetune_epochs": True},
            {"finetune_learning_rate": 0.0},
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                _client(**override)

    def test_the_client_is_stateless(self) -> None:
        """The fine-tuned model is rebuilt at every evaluation and discarded,
        so nothing personalized is checkpointed per client."""

        client = _client()
        client.evaluate(_request("personal", ["test"]))
        for value in client.get_state().values():
            self.assertNotIsInstance(value, torch.Tensor)

    @pytest.mark.fast
    def test_settings_survive_a_state_round_trip(self) -> None:
        """When the config still says the same thing, which a resume does.

        This used to load a checkpoint written at `finetune_epochs: 3` into a
        client configured at the default and assert that 3 won -- P10-F14
        written as a guarantee. A resume across a changed config is the case
        below.
        """

        settings = {"finetune_epochs": 3, "finetune_learning_rate": 0.02}
        source = _client(**settings)
        restored = _client(**settings)
        restored.load_state(source.get_state())
        self.assertEqual(restored.finetune_epochs, 3)
        self.assertEqual(restored.finetune_learning_rate, 0.02)

    @pytest.mark.fast
    def test_a_checkpoint_that_disagrees_with_the_config_is_refused(self) -> None:
        """P10-F14, on the two settings this client adds."""

        source = _client(finetune_epochs=3, finetune_learning_rate=0.02)
        restored = _client()
        with self.assertRaises(ValueError) as caught:
            restored.load_state(source.get_state())
        self.assertIn("finetune_epochs", str(caught.exception))


def _config(model_scope: str = "both", **client_extra: Any) -> Any:
    config = copy.deepcopy(_BASE_CONFIG)
    config.client.update_rule = "fedavg_ft"
    # fedavg_ft runs the shared local-update modes, like fedavg itself.
    config.client.extra.update(
        {
            "update_mode": "sequential_epoch",
            "frozen_gradient_weighting": "examples",
            "finetune_epochs": 2,
            **client_extra,
        }
    )
    config.evaluation.model_scope = model_scope
    return config


_BASE_CONFIG = load_config("configs/dev/synthetic.yaml")


@pytest.mark.fast
class ValidationTest(unittest.TestCase):
    def _errors(self, config: Any) -> list[str]:
        return [
            issue.code for issue in validate_full_config(config).issues if issue.severity == "error"
        ]

    def test_personal_and_both_are_accepted(self) -> None:
        for scope in ("personal", "both"):
            with self.subTest(scope=scope):
                config = _config(scope)
                validate_config(config)
                self.assertEqual(self._errors(config), [])

    def test_global_scope_is_rejected_as_a_wasted_run(self) -> None:
        """Under model_scope: global this arm trains and evaluates exactly like
        plain fedavg, so the name would claim something the run did not do."""

        config = _config("global")
        with self.assertRaises(ValueError):
            validate_config(config)
        self.assertIn("algorithm.fedavg_ft_scope_missing", self._errors(config))

    def test_finetune_epochs_is_required(self) -> None:
        config = _config()
        del config.client.extra["finetune_epochs"]
        with self.assertRaises(ValueError):
            validate_config(config)
        self.assertIn("algorithm.fedavg_ft_epochs_missing", self._errors(config))

    def test_invalid_finetune_settings_are_rejected(self) -> None:
        for extra, code in (
            ({"finetune_epochs": 0}, "algorithm.fedavg_ft_epochs_invalid"),
            (
                {"finetune_learning_rate": -1.0},
                "algorithm.fedavg_ft_learning_rate_invalid",
            ),
        ):
            with self.subTest(extra=extra):
                config = _config(**extra)
                with self.assertRaises(ValueError):
                    validate_config(config)
                self.assertIn(code, self._errors(config))

    def test_evaluating_every_client_is_flagged_as_a_cost(self) -> None:
        """clients: all fine-tunes every client on every evaluated round, not
        just the round's participants."""

        config = _config()
        config.evaluation.test.clients = "all"
        codes = {issue.code: issue.severity for issue in validate_full_config(config).issues}
        self.assertEqual(codes.get("algorithm.fedavg_ft_test_scope_cost"), "info")

    def test_the_factory_builds_it_against_a_plain_fedavg_server(self) -> None:
        components = build_components(_config())
        self.assertEqual(components.config.server.strategy, "fedavg")
        client = components.clients[next(iter(components.clients))]
        self.assertIsInstance(client, FedAvgFTClient)
        self.assertEqual(client.finetune_epochs, 2)
        # Unset in the config, so it inherits the training rate.
        self.assertEqual(client.finetune_learning_rate, client.learning_rate)


if __name__ == "__main__":
    unittest.main()
