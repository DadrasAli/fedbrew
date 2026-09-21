"""Evaluation must not reload a model state the model already holds.

A cross-device round evaluates the same global model on every client through
one shared model instance (``reuse_model=True``), so the state load after the
first client was copying a state dict onto itself -- about half of client_eval
on FEMNIST. These tests pin both halves of the optimization: that the redundant
load is actually skipped, and that it is never skipped when the model may have
been changed underneath it.
"""

from __future__ import annotations

import unittest
from typing import Any

import pytest
import torch

from fedbrew.clients.torch_sgd_client import TorchSGDClient
from fedbrew.core.protocol import EvalRequest, FitRequest
from fedbrew.core.torch_utils import RESIDENT_STATE_ATTR
from fedbrew.tasks.classification.torch_classification import (
    TorchClassificationTask,
)

MODEL_CONFIG = {
    "name": "mlp",
    "input_dim": 2,
    "hidden_dim": 4,
    "num_classes": 2,
}


class _CountingTask(TorchClassificationTask):
    """Classification task that counts real state loads."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.load_count = 0

    def load_federated_model_state(self, model, state) -> None:  # type: ignore[no-untyped-def]
        self.load_count += 1
        super().load_federated_model_state(model, state)


def _client_data() -> dict[str, Any]:
    generator = torch.Generator().manual_seed(0)
    return {
        "train": {
            "x": torch.randn(8, 2, generator=generator),
            "y": torch.randint(0, 2, (8,), generator=generator),
        },
        "eval": {
            "x": torch.randn(4, 2, generator=generator),
            "y": torch.randint(0, 2, (4,), generator=generator),
        },
    }


class ClientEvalStateReuseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.task = _CountingTask(
            model_config=dict(MODEL_CONFIG),
            batch_size=4,
            device="cpu",
            reuse_model=True,
        )
        # One shared model instance, exactly as reuse_model gives the loop.
        self.model = self.task.build_model(MODEL_CONFIG)

    def _client(self, client_id: str) -> TorchSGDClient:
        return TorchSGDClient(
            client_id=client_id,
            task=self.task,
            model_config=dict(MODEL_CONFIG),
            client_data=_client_data(),
            local_iterations=1,
            batch_size=4,
            learning_rate=0.1,
            momentum=0.0,
            weight_decay=0.0,
            nesterov=False,
            learning_rate_schedule="constant",
            min_learning_rate=0.0,
            total_rounds=1,
            device="cpu",
            metrics=["loss", "accuracy"],
            # Pinned so a fit is a pure function of the state it loads: the
            # comparison below runs one fit after an evaluation and one without,
            # and shuffling seeded from global RNG would diverge on that alone.
            base_seed=0,
            train_shuffle=False,
        )

    def _server_payload(self, seed: int) -> dict[str, Any]:
        """Build a fresh global state object, as the server does each round."""
        torch.manual_seed(seed)
        source = self.task._construct_model(MODEL_CONFIG)
        state = self.task.get_federated_model_state(source)
        metadata = self.task.federated_model_state_metadata(source)
        return {
            "model_state": state,
            "model_state_scope": str(metadata["model_state_scope"]),
            "model_state_metadata": metadata,
        }

    def _eval_payload(self, server_payload: dict[str, Any]) -> dict[str, Any]:
        payload = dict(server_payload)
        payload.update(
            {
                "metrics": ["loss", "accuracy"],
                "splits": ["train"],
                "model_scope": "global",
            }
        )
        return payload

    def _evaluate(self, client: TorchSGDClient, payload: dict[str, Any]):
        return client.evaluate(EvalRequest(round_id=1, client_id=client.client_id, payload=payload))

    @pytest.mark.fast
    def test_shared_state_object_is_loaded_once_across_clients(self) -> None:
        """The whole point: N clients, one shared state object, one load."""

        payload = self._eval_payload(self._server_payload(seed=0))
        clients = [self._client(f"client_{index}") for index in range(5)]

        results = [self._evaluate(client, payload) for client in clients]

        self.assertEqual(self.task.load_count, 1, "state reloaded per client")
        self.assertEqual(len(results), 5)
        self.assertIs(getattr(self.model, RESIDENT_STATE_ATTR, None), payload["model_state"])

    @pytest.mark.fast
    def test_skipping_does_not_change_metrics(self) -> None:
        """Skipped loads must be unobservable in the reported metrics."""

        payload = self._eval_payload(self._server_payload(seed=0))
        client = self._client("client_0")

        first = self._evaluate(client, payload)
        # Force the next evaluation to take the slow path, then compare.
        setattr(self.model, RESIDENT_STATE_ATTR, None)
        second = self._evaluate(client, payload)

        self.assertEqual(self.task.load_count, 2, "the forced reload did not happen")
        self.assertEqual(first.metrics.keys(), second.metrics.keys())
        for name, value in first.metrics.items():
            self.assertAlmostEqual(value, second.metrics[name], places=12, msg=name)

    @pytest.mark.fast
    def test_new_state_object_is_always_loaded(self) -> None:
        """A different round means a different state object, so reload."""

        client = self._client("client_0")
        self._evaluate(client, self._eval_payload(self._server_payload(seed=0)))
        self._evaluate(client, self._eval_payload(self._server_payload(seed=1)))

        self.assertEqual(self.task.load_count, 2)

    def test_fit_invalidates_the_resident_state(self) -> None:
        """Local training rewrites the shared model, so the claim must drop."""

        server_payload = self._server_payload(seed=0)
        eval_payload = self._eval_payload(server_payload)
        client = self._client("client_0")

        self._evaluate(client, eval_payload)
        self.assertEqual(self.task.load_count, 1)

        client.fit(FitRequest(round_id=1, client_id=client.client_id, payload=server_payload))
        self.assertIsNone(getattr(self.model, RESIDENT_STATE_ATTR, None))

        # Same state object as before the fit: must NOT be skipped, because the
        # weights were trained away from it in between.
        self._evaluate(client, eval_payload)
        self.assertGreaterEqual(self.task.load_count, 3)

    def test_fit_result_matches_an_untouched_model(self) -> None:
        """Fit must be unaffected by whatever eval left stamped on the model."""

        server_payload = self._server_payload(seed=0)
        client = self._client("client_0")

        self._evaluate(client, self._eval_payload(server_payload))
        after_eval = client.fit(
            FitRequest(round_id=1, client_id=client.client_id, payload=server_payload)
        )

        setattr(self.model, RESIDENT_STATE_ATTR, None)
        clean = client.fit(
            FitRequest(round_id=1, client_id=client.client_id, payload=server_payload)
        )

        for key, tensor in after_eval.payload["model_state"].items():
            self.assertTrue(
                torch.equal(tensor, clean.payload["model_state"][key]),
                f"fit diverged for {key}",
            )


class EvaluationDoesNotTouchTheTrainingStreamTests(ClientEvalStateReuseTests):
    """Regression tests for evaluation moving the training stream, at the base client.

    Evaluation is a measurement, so it must not move the generator the next
    round trains from -- otherwise evaluation.*.clients stops being the pure
    cost knob its config comment describes and becomes a hyperparameter of the
    training curve. Two things draw here: every DataLoader iterator takes a base
    seed from the global generator, and any rule that trains on the eval path
    draws dropout masks on top of that. The first applies to every algorithm,
    not just fedavg_ft.
    """

    def _stream_after_evaluating(self, clients: int) -> bytes:
        torch.manual_seed(4321)
        payload = self._server_payload(seed=1)
        for index in range(clients):
            self._client(f"c{index}").evaluate(
                EvalRequest(round_id=1, client_id=f"c{index}", payload=payload)
            )
        # bytes(tolist()) rather than .numpy().tobytes(): the state is a uint8
        # tensor, so the two are identical, and this one does not make the
        # suite depend on numpy. Nothing in fedbrew/ requires numpy -- every
        # use there is a guarded lazy import -- and a test that quietly did
        # would be the undeclared dependency the core-only CI job exists to
        # catch.
        return bytes(torch.get_rng_state().clone().tolist())

    @pytest.mark.fast
    def test_evaluating_more_clients_does_not_move_the_stream(self) -> None:
        self.assertEqual(self._stream_after_evaluating(1), self._stream_after_evaluating(6))

    @pytest.mark.fast
    def test_evaluating_at_all_does_not_move_the_stream(self) -> None:
        torch.manual_seed(4321)
        payload = self._server_payload(seed=1)
        before = torch.get_rng_state().clone()
        self._client("c0").evaluate(EvalRequest(round_id=1, client_id="c0", payload=payload))

        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    @pytest.mark.fast
    def test_the_evaluation_result_is_still_reproducible(self) -> None:
        """Isolating the draws must not make the measurement itself vary."""

        def measure() -> dict[str, float]:
            torch.manual_seed(4321)
            payload = self._server_payload(seed=1)
            return dict(
                self._client("c0")
                .evaluate(EvalRequest(round_id=1, client_id="c0", payload=payload))
                .metrics
            )

        self.assertEqual(measure(), measure())


if __name__ == "__main__":
    unittest.main()
