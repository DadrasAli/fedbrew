"""The FedAvg+FT fine-tuning pass does not replay the fit pass's batch order.

Both passes read the same client's train split in the same round, and both
start from the same weights -- the round's global model. The fit pass loads it
and trains; the fine-tuning pass, on the evaluation path, loads the same
payload and trains again. What is supposed to make them different passes is
that one runs `local_iterations` at `learning_rate` and the other `finetune_epochs`
at `finetune_learning_rate`.

The batch order was not among the differences. `_train_loader_config` derived
its seed under the phase name `"fit"`, and `_finetune` called that same method,
so `dataloader_seed(base, round, client, "fit")` served both. Measured before
the fix, 24 examples in 6 batches:

    fit      [15, 2, 3, 12, 18, 6, 14, 10, 21, 17, 16, 8, ...]
    finetune [15, 2, 3, 12, 18, 6, 14, 10, 21, 17, 16, 8, ...]

Same seed, same order, element for element. With the shipped
`configs/femnist/fedavg_ft.yaml` -- `local_iterations: 20`, `finetune_epochs: 1`,
`finetune_learning_rate: null` inheriting `learning_rate` -- the personal model
was the fit pass's first epoch recomputed, not an adaptation drawn
independently of it. Only the dropout masks differed, because `evaluate()` runs
inside `isolated_evaluation_rng` and the fit pass does not.

`dataloader_seed` has taken a phase since it was written, and the eval loader
already passes `"eval"`. This pass simply never named its own. It passes
`"finetune"` now.

This module pins three things: that the two phases derive different seeds, that
the seeds produce different orders on real data, and that neither has stopped
being reproducible -- an independent order is worth nothing if it is also an
unseeded one.

See FINDINGS.csv P09-F09.
"""

from __future__ import annotations

import unittest
from typing import Any

import pytest
import torch

from fedbrew.clients.fedavg_ft_client import FedAvgFTClient
from fedbrew.core.protocol import EvalRequest, FitRequest
from fedbrew.core.seeding import dataloader_seed
from fedbrew.tasks.classification.torch_classification import TorchClassificationTask

BASE_SEED = 42
NUM_EXAMPLES = 24
BATCH_SIZE = 4

#: Every phase name that reaches `dataloader_seed`. Written here rather than
#: derived: a test that reads its expectation off the thing under test cannot
#: notice the thing changing, and the whole finding was two callers sharing one
#: name.
PHASES = ("fit", "eval", "finetune")


def _data() -> dict[str, dict[str, torch.Tensor]]:
    """One split whose x values are their own indices, so an order is readable."""

    rows = {
        "x": torch.arange(NUM_EXAMPLES, dtype=torch.float32).reshape(NUM_EXAMPLES, 1),
        "y": torch.zeros(NUM_EXAMPLES, dtype=torch.long),
    }
    return {"train": dict(rows), "eval": dict(rows)}


def _client() -> FedAvgFTClient:
    task = TorchClassificationTask(
        model_config={"name": "mlp", "input_dim": 1, "num_classes": 2},
        batch_size=BATCH_SIZE,
    )
    return FedAvgFTClient(
        client_id="c0",
        task=task,
        model_config={},
        client_data=_data(),
        local_iterations=1,
        batch_size=BATCH_SIZE,
        learning_rate=0.1,
        base_seed=BASE_SEED,
        update_mode="sequential_epoch",
        frozen_gradient_weighting="examples",
        finetune_epochs=1,
        finetune_learning_rate=None,
        train_shuffle=True,
        momentum=0.0,
        weight_decay=0.0,
        nesterov=False,
        learning_rate_schedule="constant",
        min_learning_rate=0.0,
    )


def _order(client: FedAvgFTClient, config: dict[str, Any]) -> list[int]:
    """The example indices the loader yields, in the order it yields them."""

    loader = client.task.build_dataloader(_data()["train"], config)
    return [int(value) for batch in loader for value in batch[0].flatten().tolist()]


@pytest.mark.fast
class TheTwoPhasesDeriveDifferentSeedsTest(unittest.TestCase):
    def test_every_phase_name_gives_its_own_seed(self) -> None:
        seeds = {phase: dataloader_seed(BASE_SEED, 7, "c0", phase) for phase in PHASES}
        self.assertEqual(len(set(seeds.values())), len(PHASES), seeds)

    def test_finetune_is_not_fit(self) -> None:
        """The collision, named on its own so a failure says which pair."""

        self.assertNotEqual(
            dataloader_seed(BASE_SEED, 7, "c0", "fit"),
            dataloader_seed(BASE_SEED, 7, "c0", "finetune"),
        )

    def test_the_loader_config_carries_the_phase_through(self) -> None:
        """A distinct seed is worth nothing if the caller cannot select it."""

        client = _client()
        self.assertNotEqual(
            client._train_loader_config(1)["seed"],
            client._train_loader_config(1, phase="finetune")["seed"],
        )

    def test_the_default_phase_is_still_fit(self) -> None:
        """Every other caller of this method trains the fit pass."""

        client = _client()
        self.assertEqual(
            client._train_loader_config(3)["seed"],
            dataloader_seed(BASE_SEED, 3, "c0", "fit"),
        )


@pytest.mark.fast
class TheOrdersDifferOnRealDataTest(unittest.TestCase):
    """Different seeds are the mechanism; different orders are the claim."""

    def test_the_finetune_pass_does_not_replay_the_fit_pass(self) -> None:
        client = _client()
        fit = _order(client, client._train_loader_config(1))
        finetune = _order(client, client._train_loader_config(1, phase="finetune"))
        self.assertNotEqual(fit, finetune)

    def test_both_orders_are_still_permutations_of_the_split(self) -> None:
        """Anti-vacuity: differing because one drops rows would also pass above."""

        client = _client()
        for phase in ("fit", "finetune"):
            with self.subTest(phase=phase):
                order = _order(client, client._train_loader_config(1, phase=phase))
                self.assertEqual(sorted(order), list(range(NUM_EXAMPLES)))

    def test_they_differ_in_every_round_not_just_the_first(self) -> None:
        """One shared seed would have collided in all of them equally."""

        client = _client()
        for round_id in (1, 2, 5, 17):
            with self.subTest(round_id=round_id):
                self.assertNotEqual(
                    _order(client, client._train_loader_config(round_id)),
                    _order(client, client._train_loader_config(round_id, phase="finetune")),
                )


@pytest.mark.fast
class BothPassesStayReproducibleTest(unittest.TestCase):
    """An independent order is worth nothing if it is an unseeded one."""

    def test_each_phase_repeats_itself_exactly(self) -> None:
        for phase in ("fit", "finetune"):
            with self.subTest(phase=phase):
                first = _order(_client(), _client()._train_loader_config(4, phase=phase))
                second = _order(_client(), _client()._train_loader_config(4, phase=phase))
                self.assertEqual(first, second)

    def test_the_order_still_depends_on_the_round(self) -> None:
        """A per-phase seed that ignored the round would freeze the order."""

        client = _client()
        self.assertNotEqual(
            _order(client, client._train_loader_config(1, phase="finetune")),
            _order(client, client._train_loader_config(2, phase="finetune")),
        )

    def test_the_order_still_depends_on_the_client(self) -> None:
        client = _client()
        other = _client()
        other.client_id = "c1"
        self.assertNotEqual(
            _order(client, client._train_loader_config(1, phase="finetune")),
            _order(other, other._train_loader_config(1, phase="finetune")),
        )

    def test_an_unseeded_client_is_left_unseeded(self) -> None:
        """`base_seed: None` means the loader draws from the global stream."""

        client = _client()
        client.base_seed = None
        self.assertNotIn("seed", client._train_loader_config(1, phase="finetune"))


class TheFinetunePassActuallyUsesItTest(unittest.TestCase):
    """The seed can be right and the call site still pass the wrong phase.

    Checked by recording what `_finetune` asks for, not by grepping its source.
    The first version of this test did grep, and a deliberate revert of the
    call site passed it -- because the comment above that line quotes
    `phase="finetune"` to explain why it is there. A guard that a nearby
    comment can satisfy is not guarding the code.
    """

    def _recorded_phases(self) -> list[str]:
        """Every phase `_finetune` asks `_train_loader_config` for."""

        client = _client()
        seen: list[str] = []
        original = client._train_loader_config

        def recording(round_id: int, phase: str = "fit") -> dict[str, Any]:
            seen.append(phase)
            return original(round_id, phase)

        client._train_loader_config = recording  # type: ignore[method-assign]
        model = client.task.build_model(client.model_config)
        client._finetune(model, EvalRequest(round_id=1, client_id="c0", payload={}))
        return seen

    def test_finetune_asks_for_its_own_phase(self) -> None:
        self.assertEqual(self._recorded_phases(), ["finetune"])

    @pytest.mark.fast
    def test_it_is_the_only_caller_that_does(self) -> None:
        """Everything else trains the fit pass, and must keep the default."""

        from pathlib import Path

        clients = Path(__file__).resolve().parent.parent / "fedbrew" / "clients"
        callers = {
            path.name
            for path in clients.glob("*.py")
            if 'phase="finetune"' in path.read_text(encoding="utf-8")
        }
        self.assertEqual(callers, {"fedavg_ft_client.py"})

    def test_the_fit_pass_still_asks_for_nothing(self) -> None:
        """`fit()` takes the default, so the two cannot drift back together."""

        client = _client()
        seen: list[str] = []
        original = client._train_loader_config

        def recording(round_id: int, phase: str = "fit") -> dict[str, Any]:
            seen.append(phase)
            return original(round_id, phase)

        client._train_loader_config = recording  # type: ignore[method-assign]
        model = client.task.build_model(client.model_config)
        payload = {"model_state": client.task.get_federated_model_state(model)}
        client.fit(FitRequest(round_id=1, client_id="c0", payload=payload))
        self.assertEqual(seen, ["fit"])


if __name__ == "__main__":
    unittest.main()
