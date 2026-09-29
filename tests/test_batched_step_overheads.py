"""What the batched executor stopped doing a step changes no bit of what it computes.

A per-step cost is gone from a stack's steps, and each is held here against
the executor that still pays it -- the same run with the change turned off --
bit for bit, in every CSV and every round's checkpoint:

- the combination of a pass's gradients and the update run on the stacked
  tensors where they are elementwise (``_Bucket._stepwise``: unclipped, not
  compiled), not under ``torch.func.vmap``: over FedAvg's four modes and each
  frozen weighting, local SGD with momentum, Nesterov and weight decay under a
  cosine schedule, AdamW, FedProx and SCAFFOLD, on the float64 linear example
  and on the float32 classification task, under both gradient forms;
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, nullcontext
from typing import Any
from unittest import mock

from fedbrew.core import batched_executor
from tests.test_batched_executor_tolerance import (
    ExecutorRuns,
    classification_rule_config,
    example_config,
    fedavg_modes,
    ragged_clients,
    rule_arms,
    rule_config,
    with_client,
)


def _unclipped(client: dict[str, Any]) -> bool:
    return client.get("max_grad_norm") is None


def cases() -> Iterator[tuple[str, dict[str, Any], Any]]:
    for mode, client in fedavg_modes():
        if _unclipped(client):
            yield f"fedavg/{mode}", with_client(example_config("fed-lasso-l2"), **client), None
    for label, client in rule_arms():
        yield label, rule_config(client), ragged_clients
    for mode in ("single_batch", "sequential_epoch", "frozen_batch_gradients", "full_gradient"):
        client = {
            "update_rule": "fedavg",
            "update_mode": mode,
            "frozen_gradient_weighting": "examples",
        }
        yield f"mlp/fedavg/{mode}", classification_rule_config(client), None
    yield (
        "mlp/local_sgd",
        classification_rule_config({"momentum": 0.9, "nesterov": True, "weight_decay": 0.01}),
        None,
    )
    yield "mlp/scaffold", classification_rule_config({"update_rule": "scaffold"}), None


@contextmanager
def paying(stepwise: bool, repeats: bool) -> Iterator[None]:
    """The executor with the removed costs put back, as asked."""

    with ExitStack() as stack:
        if not stepwise:
            stack.enter_context(
                mock.patch.object(batched_executor._Bucket, "_stepwise", lambda self: False)
            )
        yield


class NothingMovesTest(ExecutorRuns):
    def _pair(self, config: dict[str, Any], data: Any, **off: bool) -> tuple[Any, Any]:
        with data() if data is not None else nullcontext():
            now = self.run_config(config, "batched")
            with paying(**off):
                before = self.run_config(config, "batched")
        return now, before

    def test_the_stacked_arithmetic_under_either_gradient_form(self) -> None:
        for form in ("vmap_grad", "summed"):
            for label, config, data in cases():
                with (
                    self.subTest(form=form, case=label),
                    mock.patch.object(
                        batched_executor, "gradient_form", lambda task, form=form: form
                    ),
                ):
                    now, before = self._pair(config, data, stepwise=False, repeats=True)
                    self.assertAgree(now, before, exact=True)
