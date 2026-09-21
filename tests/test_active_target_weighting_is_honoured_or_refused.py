"""`model.active_target_weighting` is honoured, or the run is refused.

The causal-LM task decides a client's aggregation weight through
`TaskAdapter.federated_aggregation_weight`, and `active_target_weighting` is
what it reads: on, the weight is the active target tokens of the batches the
round trained on; off, those of the whole train split (chapter 07 §3.1). It is
on by default for a `causal_lm_sft` dataset. `fedprox` and `scaffold` never ask
the task: they report their post-fit evaluation count, the train split's
tokens, whatever the key says. So a run with the weighting on under either rule
recorded one weighting in `run.json` and aggregated with the other.

It is refused now (FINDINGS.csv POST-F30). The key itself at config load,
through `fedbrew run` and `--validate-only`; the SFT default where each path
first reads the manifest — `factory._model_config` on the run path, preflight's
`data` check under `--validate-only`. The message names the rules that honour
it, and the first test here derives that list from what each client's `fit`
actually does, not from the list.
"""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import yaml

from fedbrew.cli import dispatch
from fedbrew.core import registry, validation
from fedbrew.core.config import load_config
from fedbrew.core.factory import _model_config
from fedbrew.core.federated_state import (
    AGGREGATION_WEIGHT_HOOK_BYPASS_RULES,
    AGGREGATION_WEIGHT_HOOK_CLIENT_RULES,
)
from fedbrew.core.refusal import RunRefused
from tests.test_client_communication_cost import BUILDERS, _request
from tests.test_empty_training_batches import _Task

HONOURING = ", ".join(sorted(AGGREGATION_WEIGHT_HOOK_CLIENT_RULES))
SENTINEL = 7777


def _raw(rule: str, directory: Path, **model: Any) -> dict[str, Any]:
    raw = yaml.safe_load(Path("configs/dev/tiny_causal_lm.yaml").read_text(encoding="utf-8"))
    raw["experiment"]["output_dir"] = str(directory / f"out_{rule}")
    metrics = raw["client"]["metrics"]
    # local_adamw keeps the shipped client block.
    if rule == "fedprox":
        raw["client"] = {
            "update_rule": "fedprox",
            "batch_size": 4,
            "learning_rate": 0.0005,
            "proximal_mu": 0.01,
            "metrics": metrics,
        }
    elif rule == "scaffold":
        raw["server"]["strategy"] = "scaffold"
        raw["client"] = {
            "update_rule": "scaffold",
            "batch_size": 4,
            "learning_rate": 0.0005,
            "metrics": metrics,
        }
    raw["model"].update(model)
    return raw


def _write(raw: dict[str, Any], directory: Path, name: str) -> Path:
    path = directory / f"{name}.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def _fedbrew(*argv: str) -> tuple[int | str | None, str]:
    output = StringIO()
    code: int | str | None = None
    with redirect_stderr(output), redirect_stdout(output):
        try:
            dispatch.main(list(argv))
        except SystemExit as exit_:
            code = exit_.code
    return code, " ".join(output.getvalue().split())


class TheTwoSetsAreWhatEachClientDoesTest(unittest.TestCase):
    def test_every_rule_either_asks_the_task_or_is_on_the_bypass_list(self) -> None:
        registry.register_builtin_components()
        builtin = set(registry.client_updates.builtin())
        self.assertEqual(
            AGGREGATION_WEIGHT_HOOK_CLIENT_RULES | AGGREGATION_WEIGHT_HOOK_BYPASS_RULES, builtin
        )
        self.assertFalse(
            AGGREGATION_WEIGHT_HOOK_CLIENT_RULES & AGGREGATION_WEIGHT_HOOK_BYPASS_RULES
        )
        self.assertEqual(set(BUILDERS), builtin)
        with mock.patch.object(
            _Task, "federated_aggregation_weight", lambda self, outputs, evaluated: SENTINEL
        ):
            for rule, build in BUILDERS.items():
                with self.subTest(rule=rule):
                    weight = build().fit(copy.deepcopy(_request())).num_examples
                    asks = weight == SENTINEL
                    self.assertEqual(asks, rule in AGGREGATION_WEIGHT_HOOK_CLIENT_RULES)


@pytest.mark.fast
class TheKeyIsRefusedAtLoadTest(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)

    def _assert_message(self, rule: str, text: str) -> None:
        self.assertIn(
            f"client.update_rule={rule} does not honour model.active_target_weighting: true",
            text,
        )
        self.assertIn(f"use a rule that honours it: {HONOURING}.", text)

    def test_fedbrew_run_refuses_it_and_writes_nothing(self) -> None:
        for rule in sorted(AGGREGATION_WEIGHT_HOOK_BYPASS_RULES):
            with self.subTest(rule=rule):
                raw = _raw(rule, self.root, active_target_weighting=True)
                code, text = _fedbrew("run", "--config", str(_write(raw, self.root, rule)))
                self.assertEqual(code, 2)
                self.assertIn("RUN REFUSED", text)
                self._assert_message(rule, text)
                self.assertFalse((self.root / f"out_{rule}").exists())

    def test_validate_only_refuses_it(self) -> None:
        for rule in sorted(AGGREGATION_WEIGHT_HOOK_BYPASS_RULES):
            with self.subTest(rule=rule):
                raw = _raw(rule, self.root, active_target_weighting=True)
                path = _write(raw, self.root, rule)
                code, text = _fedbrew("run", "--config", str(path), "--validate-only")
                self.assertEqual(code, 1)
                self._assert_message(rule, text)
                self.assertFalse((self.root / f"out_{rule}").exists())

    def test_the_combinations_that_mean_what_they_say_still_load(self) -> None:
        cases = [(rule, False) for rule in sorted(AGGREGATION_WEIGHT_HOOK_BYPASS_RULES)]
        cases += [("local_adamw", True), ("local_adamw", False)]
        for rule, weighting in cases:
            with self.subTest(rule=rule, active_target_weighting=weighting):
                raw = _raw(rule, self.root, active_target_weighting=weighting)
                load_config(_write(raw, self.root, f"{rule}_{weighting}"))
        for rule in sorted(AGGREGATION_WEIGHT_HOOK_BYPASS_RULES):
            with self.subTest(rule=rule, active_target_weighting="unset"):
                load_config(_write(_raw(rule, self.root), self.root, f"{rule}_unset"))


class _Dataset:
    def __init__(self, task: str) -> None:
        self.task = task

    def get_metadata(self) -> dict[str, Any]:
        return {"task": self.task}


@pytest.mark.fast
class TheSftDefaultIsRefusedWhereTheManifestIsReadTest(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)

    def _config(self, rule: str, **model: Any) -> Any:
        return load_config(_write(_raw(rule, self.root, **model), self.root, rule))

    def test_the_run_path_refuses_it_when_it_builds_the_model_config(self) -> None:
        for rule in sorted(AGGREGATION_WEIGHT_HOOK_BYPASS_RULES):
            with self.subTest(rule=rule):
                config = self._config(rule)
                with self.assertRaises(RunRefused) as caught:
                    _model_config(config, _Dataset("causal_lm_sft"))
                self.assertIn(
                    "which a causal_lm_sft dataset turns on by default", str(caught.exception)
                )
                self.assertIn(HONOURING, str(caught.exception))
                _model_config(config, _Dataset("causal_lm"))
                _model_config(
                    self._config(rule, active_target_weighting=False), _Dataset("causal_lm_sft")
                )
        _model_config(self._config("local_adamw"), _Dataset("causal_lm_sft"))

    def test_preflight_reports_it_from_the_manifest(self) -> None:
        manifest = self.root / "manifest.json"
        codes = {}
        for task in ("causal_lm_sft", "causal_lm"):
            manifest.write_text(json.dumps({"task": task}), encoding="utf-8")
            raw = _raw("fedprox", self.root)
            raw["data"]["path"] = str(manifest)
            config = load_config(_write(raw, self.root, f"fedprox_{task}"))
            codes[task] = {issue.code for issue in validation.run_checks(config)}
        self.assertIn("client.active_target_weighting_ignored", codes["causal_lm_sft"])
        self.assertNotIn("client.active_target_weighting_ignored", codes["causal_lm"])


if __name__ == "__main__":
    unittest.main()
