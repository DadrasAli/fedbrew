"""`max_grad_norm` bounds a different quantity in each `update_mode`.

The engine clips once per applied update, and the modes apply different numbers
of updates per epoch:

- `single_batch` and `sequential_epoch` clip **each batch gradient**, once per
  optimizer step, through `_ClippingOptimizer`. An epoch over N batches moves
  at most `N * lr * max_grad_norm`.
- `frozen_batch_gradients` clips the **combined epoch gradient** once, after
  the batch gradients have been weighted and summed. The same epoch moves at
  most `lr * max_grad_norm`.

Per mode the behaviour is coherent -- one clip per applied update either way --
and nothing here changes it. What does not hold is that two arms sharing a
threshold across modes share a bound, and a sentence saying so is not a thing a
later reader can check. So this measures the factor of N instead, at a
threshold low enough that both modes clip on every step, which turns "different
quantity" into a number.

The second half is the AMP refusal. Both wrappers the engine hands
`task.train_step` -- `_ClippingOptimizer` when clipping and
`_GradientOnlyOptimizer` in the frozen mode -- are optimizer-shaped without
being `torch.optim.Optimizer`, and `train_step` used to test exactly that.
Measured on a CUDA device under `use_amp: true`, four of the six
(mode x clipping) combinations raised `TypeError` at the first training step
of round 1:

    single_batch           max_grad_norm=1.0   TypeError  _ClippingOptimizer
    sequential_epoch       max_grad_norm=1.0   TypeError  _ClippingOptimizer
    frozen_batch_gradients max_grad_norm=None  TypeError  _GradientOnlyOptimizer
    frozen_batch_gradients max_grad_norm=1.0   TypeError  _GradientOnlyOptimizer

The census filed the frozen mode alone. Clipping was the same crash through
the other wrapper, on all three modes, reaching `fedavg`, `centralized` and
`fedavg_ft` -- three rules that do not refuse `use_amp`. Both
were refused at config load, so neither reached round 1.

**The two were never the same defect**, which the table above hides and a
later measurement exposed. `GradScaler` needs `param_groups`, not the torch
type, and with that guard corrected `_ClippingOptimizer` takes an AMP step
while `_GradientOnlyOptimizer` still cannot -- it wraps no optimizer, so it
has no groups. Only the frozen mode is refused now; clipping's refusal was
lifted on its own five-round trajectory, alongside scaffold's and fedprox's.
`tests/test_amp_composes_with_wrapped_optimizers.py` carries that measurement;
this module pins the predicate underneath it, since `use_amp` is forced false
off CUDA and the crash cannot be reproduced here.

See FINDINGS.csv P03-F05.
"""

from __future__ import annotations

import copy
import math
import unittest
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn, optim

from fedbrew.clients.local_update_modes import (
    _ClippingOptimizer,
    _GradientOnlyOptimizer,
    run_sgd_update_mode,
)
from fedbrew.core.config import (
    SGD_ENGINE_CLIENT_RULES,
    amp_unsupported_sgd_engine_setting,
    load_config,
    validate_config,
)
from fedbrew.core.validation import validate_full_config
from fedbrew.tasks.classification.torch_classification import TorchClassificationTask

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Batches per epoch. The factor the two clipping scopes differ by.
NUM_BATCHES = 4

#: Low enough that every batch gradient exceeds it, so both modes clip on every
#: step and the comparison is between the scopes rather than between two
#: gradients that happened not to reach the threshold.
MAX_GRAD_NORM = 1e-3

LEARNING_RATE = 0.5

#: Relative tolerance on a clipped step's length. The clip itself is tight --
#: `_clip_accumulated_update` lands ~1e-5 *under* the threshold, from its 1e-6
#: denominator epsilon -- but applying a ~1e-4 update to O(1) float32
#: parameters rounds at ~1e-7 absolute, which is ~1e-4 of the step. Comfortably
#: below the factor-of-N these tests are about, and comfortably above float32.
STEP_TOLERANCE = 1e-3


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(4, 3)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def _batches() -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(0)
    return [
        (
            torch.randn(8, 4, generator=generator),
            torch.randint(0, 3, (8,), generator=generator),
        )
        for _ in range(NUM_BATCHES)
    ]


def _run(update_mode: str, max_grad_norm: float | None) -> tuple[float, int]:
    """One epoch of a mode: how far the model moved, and in how many updates."""

    torch.manual_seed(0)
    model = _Tiny()
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}

    result = run_sgd_update_mode(
        task=TorchClassificationTask(device="cpu"),
        model=model,
        train_loader=_batches(),
        local_iterations=1,
        learning_rate=LEARNING_RATE,
        update_mode=update_mode,
        frozen_gradient_weighting="examples",
        client_id="c0",
        max_grad_norm=max_grad_norm,
    )

    after = model.state_dict()
    moved = math.sqrt(sum(float(((after[name] - before[name]) ** 2).sum()) for name in before))
    return moved, result.optimizer_steps


def _displacement(update_mode: str, max_grad_norm: float | None) -> float:
    """L2 distance the model moved over one epoch of the given mode."""

    return _run(update_mode, max_grad_norm)[0]


class TheBoundIsNotTheSameQuantityTest(unittest.TestCase):
    """The measured gap, not a sentence saying there is one.

    At `MAX_GRAD_NORM` the clip binds in every mode -- each clipped run below
    moves three orders of magnitude less than the same run unclipped -- so
    what these compare is the clipping scope and not the modes' own
    differences.
    """

    def test_the_frozen_mode_moves_exactly_one_clipped_step(self) -> None:
        """One applied update per epoch, and the clip sets its length."""

        moved = _displacement("frozen_batch_gradients", MAX_GRAD_NORM)
        one_step = LEARNING_RATE * MAX_GRAD_NORM
        self.assertAlmostEqual(moved / one_step, 1.0, delta=STEP_TOLERANCE)

    def test_the_update_counts_are_why(self) -> None:
        """The mechanism under the numbers, stated without them.

        One clip per applied update in both modes; the modes disagree on how
        many updates an epoch is. That is the whole of the difference, and it
        is what a reader would check first.
        """

        self.assertEqual(_run("frozen_batch_gradients", MAX_GRAD_NORM)[1], 1)
        self.assertEqual(_run("sequential_epoch", MAX_GRAD_NORM)[1], NUM_BATCHES)

    def test_the_sequential_mode_passes_that_ceiling_at_the_same_threshold(self) -> None:
        """The finding in one number: same `max_grad_norm`, same batches.

        The frozen mode cannot move further than `lr * max_grad_norm` in an
        epoch. The sequential mode does, on identical data, because the
        threshold bounds each of its N steps rather than their combination.
        """

        one_step = LEARNING_RATE * MAX_GRAD_NORM
        moved = _displacement("sequential_epoch", MAX_GRAD_NORM)
        self.assertGreater(moved, one_step * (1.0 + STEP_TOLERANCE))
        self.assertLessEqual(moved, one_step * NUM_BATCHES * (1.0 + STEP_TOLERANCE))

    def test_the_two_ceilings_differ_by_the_batch_count(self) -> None:
        """What the chapter's table says, against what the modes can reach.

        The realised displacement is below the sequential ceiling because
        successive clipped steps partly cancel; the ceiling is still N times
        the frozen mode's, and it is the ceiling that a shared threshold is
        supposed to make comparable.
        """

        frozen_ceiling = LEARNING_RATE * MAX_GRAD_NORM
        sequential_ceiling = frozen_ceiling * NUM_BATCHES
        self.assertAlmostEqual(sequential_ceiling / frozen_ceiling, float(NUM_BATCHES))
        self.assertLessEqual(
            _displacement("frozen_batch_gradients", MAX_GRAD_NORM),
            frozen_ceiling * (1.0 + STEP_TOLERANCE),
        )
        self.assertLessEqual(
            _displacement("sequential_epoch", MAX_GRAD_NORM),
            sequential_ceiling * (1.0 + STEP_TOLERANCE),
        )

    def test_the_clip_binds_in_both_modes(self) -> None:
        """Otherwise the comparison above would be between two free runs."""

        for update_mode in ("sequential_epoch", "frozen_batch_gradients"):
            with self.subTest(update_mode=update_mode):
                clipped = _displacement(update_mode, MAX_GRAD_NORM)
                free = _displacement(update_mode, None)
                self.assertGreater(free / clipped, 100.0)


@pytest.mark.fast
class OnlyOneWrapperCannotBeScaledTest(unittest.TestCase):
    """The predicate `train_step` tests, pinned off CUDA.

    `use_amp` is forced false off CUDA, so the crash cannot be reproduced
    here. What can be is what causes it: `GradScaler.unscale_` reads
    `param_groups`, and one of these two wrappers has none. Measured on an
    A100 on 2026-09-07 with the guard lifted -- `_ClippingOptimizer` took an AMP step,
    `_GradientOnlyOptimizer` raised `AttributeError` on exactly this
    attribute.
    """

    def test_the_clipping_wrapper_exposes_param_groups(self) -> None:
        """So it can be scaled. Its refusal is a decision, not a mechanism."""

        model = _Tiny()
        wrapper = _ClippingOptimizer(optim.SGD(model.parameters(), lr=0.1), model, 1.0)
        self.assertTrue(hasattr(wrapper, "param_groups"))
        self.assertEqual(len(wrapper.param_groups), 1)

    def test_the_gradient_only_wrapper_does_not(self) -> None:
        """It wraps no optimizer, so it has no groups to expose."""

        self.assertFalse(hasattr(_GradientOnlyOptimizer(_Tiny().parameters()), "param_groups"))

    def test_neither_is_a_torch_optimizer_and_that_is_not_the_test(self) -> None:
        """Both fail `isinstance`, and only one fails under AMP.

        `train_step` used to test `isinstance(optimizer, optim.Optimizer)`,
        which is stricter than what GradScaler needs and refused three
        wrappers that work. This pins that the two facts are different, so a
        future editor cannot restore the stricter guard on the grounds that
        it says the same thing.
        """

        model = _Tiny()
        for wrapper in (
            _ClippingOptimizer(optim.SGD(model.parameters(), lr=0.1), model, 1.0),
            _GradientOnlyOptimizer(model.parameters()),
        ):
            with self.subTest(wrapper=type(wrapper).__name__):
                self.assertNotIsInstance(wrapper, optim.Optimizer)

    def test_train_step_tests_param_groups_and_says_so(self) -> None:
        """The backstop names the requirement, so a crash is legible."""

        source = (
            REPO_ROOT / "fedbrew" / "tasks" / "classification" / "torch_classification.py"
        ).read_text(encoding="utf-8")
        for phrase in (
            'hasattr(optimizer, "param_groups")',
            "runtime.use_amp needs an optimizer exposing param_groups",
        ):
            if phrase not in source:
                self.fail(f"torch_classification.py no longer contains {phrase!r}")
        if "isinstance(optimizer, optim.Optimizer)" in source:
            self.fail("train_step is back to a guard stricter than GradScaler's requirement")


class RefusedAtConfigLoadTest(unittest.TestCase):
    """What the crash is replaced by."""

    def _config(self) -> Any:
        return copy.deepcopy(load_config("configs/femnist/fedavg.yaml"))

    def _preflight_errors(self, config: Any) -> list[str]:
        return [
            issue.code for issue in validate_full_config(config).issues if issue.severity == "error"
        ]

    @pytest.mark.fast
    def test_the_shipped_configs_are_accepted_as_they_ship(self) -> None:
        for path in (
            "configs/femnist/fedavg.yaml",
            "configs/femnist/centralized.yaml",
            "configs/mnist/fedavg.yaml",
            "configs/medmcqa/fedavg_lora.yaml",
        ):
            with self.subTest(path=path):
                config = load_config(path)
                validate_config(config)
                self.assertIsNone(amp_unsupported_sgd_engine_setting(config))

    def test_the_frozen_mode_is_refused(self) -> None:
        config = self._config()
        config.runtime.use_amp = True
        config.client.extra["update_mode"] = "frozen_batch_gradients"
        with self.assertRaisesRegex(ValueError, "update_mode: frozen_batch_gradients"):
            validate_config(config)
        self.assertIn("algorithm.sgd_engine_amp_unsupported", self._preflight_errors(config))

    @pytest.mark.fast
    def test_clipping_is_accepted_in_the_modes_that_can_run(self) -> None:
        """It was refused in all three, on a guard stricter than GradScaler's.

        The refusal named `max_grad_norm`; the mechanism was
        `train_step` testing the nominal optimizer type. With that corrected,
        `_ClippingOptimizer` scales fine and only the frozen mode is left --
        refused for its own wrapper, not for the clip.
        """

        for update_mode in ("single_batch", "sequential_epoch"):
            with self.subTest(update_mode=update_mode):
                config = self._config()
                config.runtime.use_amp = True
                config.client.extra["update_mode"] = update_mode
                config.client.extra["max_grad_norm"] = 1.0
                validate_config(config)
                self.assertIsNone(amp_unsupported_sgd_engine_setting(config))

    @pytest.mark.fast
    def test_the_refusal_names_the_mode_not_the_clip(self) -> None:
        """With both set, the message must send the reader to the right key."""

        config = self._config()
        config.runtime.use_amp = True
        config.client.extra["update_mode"] = "frozen_batch_gradients"
        config.client.extra["max_grad_norm"] = 1.0
        with self.assertRaises(ValueError) as caught:
            validate_config(config)
        message = str(caught.exception)
        self.assertIn("update_mode: frozen_batch_gradients", message)
        self.assertNotIn("max_grad_norm", message)

    def test_amp_alone_is_still_allowed(self) -> None:
        """The refusal is per setting, not per rule: plain AMP still runs."""

        config = self._config()
        config.runtime.use_amp = True
        validate_config(config)
        self.assertNotIn("algorithm.sgd_engine_amp_unsupported", self._preflight_errors(config))

    @pytest.mark.fast
    def test_every_rule_that_receives_the_settings_is_covered(self) -> None:
        """Derived from the factory, so a new rule cannot slip the refusal.

        `delta_sgd` also runs through the engine and takes both settings; it
        refuses `use_amp` outright, which is why it is excluded rather than
        listed.
        """

        from fedbrew.core.factory import DELTA_SGD_CLIENT_RULES

        self.assertEqual(
            SGD_ENGINE_CLIENT_RULES,
            {
                "fedavg",
                "centralized",
                "fedavg_ft",
                "fedprox",
                "scaffold",
                "fedlalr",
                "local_sgd",
                "local_adamw",
            },
        )
        self.assertEqual(SGD_ENGINE_CLIENT_RULES & DELTA_SGD_CLIENT_RULES, set())


@pytest.mark.fast
class TheChapterSaysSoTest(unittest.TestCase):
    """Chapter 07 §4.1 now carries both halves.

    `assertIn` is avoided: the chapter runs to forty thousand characters and a
    failing `assertIn` prints the haystack.
    """

    def setUp(self) -> None:
        self.text = (REPO_ROOT / "docs" / "07-algorithms.md").read_text(encoding="utf-8")

    def _require(self, phrase: str) -> None:
        if phrase not in self.text:
            self.fail(f"docs/07-algorithms.md no longer says {phrase!r}")

    def test_it_says_the_threshold_is_per_mode(self) -> None:
        self._require("`max_grad_norm` does not mean the same thing in all four")

    def test_it_says_the_refusal_is_per_setting_not_per_rule(self) -> None:
        self._require("This refusal is per setting, not per")

    def test_it_says_the_surviving_refusal_is_mechanical(self) -> None:
        """The chapter has hedged twice here and must not hedge again.

        First it called this refusal "measured rather than predicted", which
        was true. Then it said one of the two settings was refused by decision
        with its trajectory quoted beside it -- a wall whose stated reason is
        "we chose to". Both settings have now been measured and only the one
        that cannot work is refused.
        """

        self._require("the refusal covers exactly the wrapper that cannot")


if __name__ == "__main__":
    unittest.main()
