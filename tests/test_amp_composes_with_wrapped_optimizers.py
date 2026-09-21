"""SCAFFOLD and FedProx run under `runtime.use_amp: true`. Measured, not argued.

Both refused it until now, at config load, at preflight, and inside `fit()`.
The reason recorded in all three places was the same, and it was a prediction:
composing a correcting wrapper with the task's `GradScaler` path "should work,
but nothing in this repository has measured it -- AMP requires CUDA".

It has now been measured, on an A100, on 2026-09-07.

**The guard was stricter than the requirement.** `GradScaler.unscale_` reads
`param_groups`; `train_step` tested `isinstance(optimizer, optim.Optimizer)`.
One AMP training step per wrapper, with that guard lifted:

    torch.optim.SGD (control)      ok
    _ClippingOptimizer             ok
    _ScaffoldCorrectingOptimizer   ok
    _FedProxCorrectingOptimizer    ok
    _GradientOnlyOptimizer         AttributeError: no attribute 'param_groups'

**And the numbers agree.** Five rounds on the synthetic label-skew manifest,
same seed and data, `central_test_loss` per round. The fp32 arm was run twice
to establish what run-to-run noise looks like before comparing anything to it:

    arm                       fp32 repeat    fp32 vs AMP     relative
    scaffold                  0.0            6.13e-05        5.14e-05
    fedprox                   0.0            6.13e-05        5.14e-05
    fedavg + max_grad_norm    0.0            3.97e-05        3.33e-05

A zero noise floor: the same config run twice reproduces bit-identically, so
the AMP differences are the AMP differences. 5e-05 relative is two orders
inside fp16's own precision. `scaffold` and `fedprox` land on the same delta to
three significant figures; that coincidence was not investigated. The
acceptance criterion was a trajectory comparison rather than a completed run,
because a run that completes is not the claim.

So the refusal is lifted for `scaffold`, `fedprox` and `max_grad_norm`, in
every layer each was stated in. `fedlalr`, `delta_sgd` and `update_mode:
frozen_batch_gradients` keep theirs: all three step through
`_GradientOnlyOptimizer`, which wraps no optimizer and so exposes no
`param_groups`. Their refusal is now a measurement too.

`max_grad_norm` was held back one commit longer than the other two, refused by
decision with its own trajectory sitting in the table above. That is a worse
place to stand than either a mechanism or nothing: a reader hits a wall whose
stated reason is "we chose to". Every AMP refusal left in the tree names the
same mechanism, and this module asserts there are no others.

See FINDINGS.csv P03-F05.
"""

from __future__ import annotations

import copy
import unittest
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from fedbrew.clients.local_update_modes import _ClippingOptimizer, _GradientOnlyOptimizer
from fedbrew.clients.torch_fedprox_client import _FedProxCorrectingOptimizer
from fedbrew.clients.torch_scaffold_client import _ScaffoldCorrectingOptimizer
from fedbrew.core.config import (
    amp_unsupported_sgd_engine_setting,
    load_config,
    validate_config,
)
from fedbrew.core.validation import validate_full_config

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Rules that now accept `use_amp`, with the config that ships them.
COMPOSES: dict[str, str] = {
    "scaffold": "configs/femnist/scaffold.yaml",
    "fedprox": "configs/femnist/fedprox.yaml",
}

#: Rules that still refuse it, and the wrapper that is why.
STILL_REFUSED: dict[str, str] = {
    "fedlalr": "configs/femnist/fedlalr.yaml",
    "delta_sgd": "configs/femnist/delta_sgd.yaml",
}

#: The codes the two lifted refusals used to raise. Neither may come back
#: under its old name, which is the cheapest way to catch a revert.
RETIRED_CODES = (
    "algorithm.scaffold_amp_unsupported",
    "algorithm.fedprox_amp_unsupported",
)


def _errors(config: Any) -> list[str]:
    return [
        issue.code for issue in validate_full_config(config).issues if issue.severity == "error"
    ]


def _with_amp(path: str) -> Any:
    config = copy.deepcopy(load_config(path))
    config.runtime.use_amp = True
    return config


class TheRefusalIsLiftedTest(unittest.TestCase):
    @pytest.mark.fast
    def test_both_rules_load_under_amp(self) -> None:
        for rule, path in sorted(COMPOSES.items()):
            with self.subTest(rule=rule):
                validate_config(_with_amp(path))

    def test_preflight_raises_no_amp_error_for_either(self) -> None:
        for rule, path in sorted(COMPOSES.items()):
            with self.subTest(rule=rule):
                codes = _errors(_with_amp(path))
                for code in RETIRED_CODES:
                    self.assertNotIn(code, codes)

    @pytest.mark.fast
    def test_the_shipped_configs_still_load_as_they_ship(self) -> None:
        for path in (*COMPOSES.values(), *STILL_REFUSED.values()):
            with self.subTest(path=path):
                validate_config(load_config(path))

    @pytest.mark.fast
    def test_the_retired_codes_are_gone_from_the_source(self) -> None:
        """Not merely unreachable: removed, so a grep for them finds nothing."""

        source = (REPO_ROOT / "fedbrew" / "core" / "validation.py").read_text(encoding="utf-8")
        for code in RETIRED_CODES:
            with self.subTest(code=code):
                if code in source:
                    self.fail(f"validation.py still emits {code}")

    @pytest.mark.fast
    def test_neither_client_keeps_its_runtime_backstop(self) -> None:
        for module in ("torch_scaffold_client", "torch_fedprox_client"):
            with self.subTest(module=module):
                source = (REPO_ROOT / "fedbrew" / "clients" / f"{module}.py").read_text(
                    encoding="utf-8"
                )
                if 'getattr(self.task, "_scaler"' in source:
                    self.fail(f"{module}.py still refuses use_amp inside fit()")


@pytest.mark.fast
class TheOtherRefusalsSurviveTest(unittest.TestCase):
    """Lifting two is not lifting all four. The wrapper is the difference."""

    def test_fedlalr_and_delta_sgd_still_refuse_at_load(self) -> None:
        for rule, path in sorted(STILL_REFUSED.items()):
            with self.subTest(rule=rule):
                with self.assertRaisesRegex(ValueError, f"{rule} is incompatible"):
                    validate_config(_with_amp(path))

    def test_the_frozen_mode_still_refuses(self) -> None:
        config = _with_amp("configs/femnist/fedavg.yaml")
        config.client.extra["update_mode"] = "frozen_batch_gradients"
        self.assertEqual(
            amp_unsupported_sgd_engine_setting(config),
            "update_mode: frozen_batch_gradients",
        )

    def test_all_three_step_through_the_one_wrapper_that_cannot_be_scaled(self) -> None:
        """Why those three and not the other two, checked rather than asserted."""

        model = nn.Linear(4, 2)
        inner = torch.optim.SGD(model.parameters(), lr=0.1)
        zeros = {name: torch.zeros_like(t) for name, t in model.state_dict().items()}

        self.assertFalse(hasattr(_GradientOnlyOptimizer(model.parameters()), "param_groups"))
        for wrapper in (
            _ScaffoldCorrectingOptimizer(inner, model, zeros, zeros),
            _FedProxCorrectingOptimizer(inner, model, list(zeros.values()), 0.1),
        ):
            with self.subTest(wrapper=type(wrapper).__name__):
                self.assertTrue(hasattr(wrapper, "param_groups"))

    def test_clipping_is_no_longer_among_them(self) -> None:
        """It composes, so it is accepted -- in every mode, and beside the one
        setting that is still refused.

        This was the last AMP refusal standing on a decision rather than a
        mechanism. Held one commit longer than scaffold and fedprox, with its
        own passing trajectory in the module docstring's table.
        """

        model = nn.Linear(4, 2)
        wrapper = _ClippingOptimizer(torch.optim.SGD(model.parameters(), lr=0.1), model, 1.0)
        self.assertTrue(hasattr(wrapper, "param_groups"))

        for update_mode in ("single_batch", "sequential_epoch"):
            with self.subTest(update_mode=update_mode):
                config = _with_amp("configs/femnist/fedavg.yaml")
                config.client.extra["update_mode"] = update_mode
                config.client.extra["max_grad_norm"] = 1.0
                self.assertIsNone(amp_unsupported_sgd_engine_setting(config))
                validate_config(config)

    def test_clipping_does_not_rescue_the_frozen_mode(self) -> None:
        """The refusal follows the wrapper, not the pair of settings.

        `frozen_batch_gradients` still passes `_GradientOnlyOptimizer` for its
        per-batch gradients whether or not clipping is on -- the clip happens
        afterwards, by hand, on the combined update.
        """

        config = _with_amp("configs/femnist/fedavg.yaml")
        config.client.extra["update_mode"] = "frozen_batch_gradients"
        config.client.extra["max_grad_norm"] = 1.0
        self.assertEqual(
            amp_unsupported_sgd_engine_setting(config),
            "update_mode: frozen_batch_gradients",
        )

    def test_no_refusal_left_in_the_tree_stands_on_a_decision(self) -> None:
        """The point of the two lifts, stated as one check.

        Every AMP refusal names `_GradientOnlyOptimizer`'s missing
        `param_groups`. If a future refusal is added on other grounds, it
        belongs in this list with its reason, not silently beside these.
        """

        for path in STILL_REFUSED.values():
            with self.subTest(path=path):
                with self.assertRaises(ValueError) as caught:
                    validate_config(_with_amp(path))
                self.assertIn("use_amp", str(caught.exception))

        config = _with_amp("configs/femnist/fedavg.yaml")
        config.client.extra["update_mode"] = "frozen_batch_gradients"
        with self.assertRaises(ValueError) as caught:
            validate_config(config)
        self.assertIn("param_groups", str(caught.exception))


@pytest.mark.fast
class TheChapterSaysSoTest(unittest.TestCase):
    """Chapter 07 hedged in three places. It must not hedge in any.

    `assertIn` is avoided: the chapter runs to forty thousand characters and a
    failing `assertIn` prints the haystack.
    """

    def setUp(self) -> None:
        self.text = (REPO_ROOT / "docs" / "07-algorithms.md").read_text(encoding="utf-8")

    def test_the_hedge_is_gone(self) -> None:
        for phrase in (
            "prediction, not a fact",
            "prediction-not-fact",
        ):
            with self.subTest(phrase=phrase):
                if phrase in self.text:
                    self.fail(f"docs/07-algorithms.md still hedges: {phrase!r}")

    def test_it_says_the_two_now_compose(self) -> None:
        if "**Composes with `runtime.use_amp: true`**" not in self.text:
            self.fail("docs/07-algorithms.md no longer says scaffold and fedprox compose")

    def test_it_gives_the_numbers_behind_that(self) -> None:
        """A lifted refusal with no number is the hedge facing the other way.

        The rows, not just the figures. A first attempt at breaking this
        deleted one table row and one sentence and the guard still passed,
        because both figures survived elsewhere in the chapter -- so it now
        asks for the artefact a reader would actually check.
        """

        for phrase in (
            "| `scaffold` | 0.0 | 6.13e-05 | 5.1e-05 |",
            "| `fedprox` | 0.0 | 6.13e-05 | 5.1e-05 |",
            "| `fedavg` + `max_grad_norm` | 0.0 | 3.97e-05 | 3.3e-05 |",
            "noise floor",
        ):
            with self.subTest(phrase=phrase):
                if phrase not in self.text:
                    self.fail(f"docs/07-algorithms.md no longer quotes {phrase!r}")

    def test_it_gives_the_wrapper_table_too(self) -> None:
        """Which wrapper works is the other half, and the reusable half."""

        for phrase in (
            "| `_ScaffoldCorrectingOptimizer` | ok |",
            "| `_GradientOnlyOptimizer` | `AttributeError`: no `param_groups` |",
        ):
            with self.subTest(phrase=phrase):
                if phrase not in self.text:
                    self.fail(f"docs/07-algorithms.md no longer quotes {phrase!r}")


if __name__ == "__main__":
    unittest.main()
