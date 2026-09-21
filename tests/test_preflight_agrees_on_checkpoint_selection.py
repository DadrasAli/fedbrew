"""`--validate-only` reports what the run path decides about `best.pt`.

`validation.py` used to hold its own opinion of two checkpointing keys, and it
was wrong in both directions at once: it accepted `best_mode` and checked it
was `max` or `min`, a key `_validate_checkpoint_selection` has removed
outright, and it checked `best_metric` for non-emptiness alone while the run
path checks the `val_` prefix, the direction word, the `model_scope` pairing
and whether the column is emitted at all. So `best_mode: min` passed preflight
and died at load, and `best_metric: central_test_accuracy` -- selecting `best.pt`
on the test set, which is the leak chapter 08 §11 exists to prevent -- passed
preflight with no complaint. P07-F09.

Preflight now reports `checkpoint_selection_problem`, which is what
`validate_config` raises. The guard is the equivalence, checked case by case:
preflight raises an issue exactly when the run path refuses, with the same
words.
"""

from __future__ import annotations

import copy
import unittest
from typing import Any

import pytest

from fedbrew.core.config import checkpoint_selection_problem, load_config, validate_config
from fedbrew.core.validation import validate_full_config

CODE = "runtime.checkpointing_best_metric_invalid"

#: Checkpointing edits, each a reason the run path does or does not refuse.
#: The point is the pairing, so both outcomes are represented.
CASES: dict[str, dict[str, Any]] = {
    "best_mode is a removed key": {"best_mode": "min"},
    "best_mode with a value preflight used to call valid": {"best_mode": "max"},
    "best_mode with a value preflight used to call invalid": {"best_mode": "sideways"},
    "selection on the test set": {"best_metric": "central_test_accuracy"},
    "selection on a training metric": {"best_metric": "fit_loss"},
    "a val_ name with no direction word": {"best_metric": "val_thing"},
    "a val_ column nothing emits": {"best_metric": "val_accuracy_bottom10"},
    "a worst_percent that is not this run's": {"best_metric": "val_accuracy_worst5"},
    "a statistic whose toggle is off": {"best_metric": "val_accuracy_variance"},
    "an empty metric": {"best_metric": ""},
    "a personal metric without the personalized pass": {"best_metric": "personal_val_accuracy_avg"},
    "the shipped selection": {},
    "no selection at all": {"save_best": False, "best_metric": "central_test_accuracy"},
}


def _config(edit: dict[str, Any]) -> Any:
    config = load_config("configs/femnist/fedavg.yaml")
    checkpointing = dict(config.runtime.extra["checkpointing"])
    checkpointing.update(edit)
    config.runtime.extra = dict(config.runtime.extra)
    config.runtime.extra["checkpointing"] = checkpointing
    return config


def _run_path_refusal(config: Any) -> str | None:
    try:
        validate_config(copy.deepcopy(config))
    except ValueError as exc:
        return str(exc)
    return None


def _preflight_refusals(config: Any) -> list[str]:
    return [issue.message for issue in validate_full_config(config).issues if issue.code == CODE]


class PreflightAgreesWithTheRunPathTest(unittest.TestCase):
    def test_preflight_refuses_exactly_what_the_run_path_refuses(self) -> None:
        for label, edit in CASES.items():
            with self.subTest(case=label):
                config = _config(edit)
                refusal = _run_path_refusal(config)
                self.assertEqual(
                    _preflight_refusals(config),
                    [] if refusal is None else [refusal],
                    "preflight and validate_config disagree about best.pt selection",
                )

    @pytest.mark.fast
    def test_both_outcomes_are_represented(self) -> None:
        """A table that only ever refuses would pass a preflight that refuses everything."""

        outcomes = {_run_path_refusal(_config(edit)) is None for edit in CASES.values()}
        self.assertEqual(outcomes, {True, False})

    @pytest.mark.fast
    def test_the_composer_returns_what_the_run_path_raises(self) -> None:
        for label, edit in CASES.items():
            with self.subTest(case=label):
                config = _config(edit)
                self.assertEqual(checkpoint_selection_problem(config), _run_path_refusal(config))

    def test_a_test_metric_is_refused_by_both(self) -> None:
        """The case the preflight silently passed: selection on the reported split."""

        config = _config({"best_metric": "central_test_accuracy"})
        self.assertIsNotNone(_run_path_refusal(config))
        self.assertEqual(len(_preflight_refusals(config)), 1)
        self.assertIn("validation metric", _preflight_refusals(config)[0])

    def test_a_removed_key_is_named_as_removed_by_both(self) -> None:
        """The case the preflight refused for the wrong reason: `best_mode`."""

        config = _config({"best_mode": "min"})
        message = _preflight_refusals(config)[0]
        self.assertIn("has been removed", message)
        self.assertNotIn("must be max or min", message)


if __name__ == "__main__":
    unittest.main()
