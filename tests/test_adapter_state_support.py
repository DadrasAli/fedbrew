"""Which client rules train adapter-only (LoRA) state, and the ones refused.

Chapter 07 listed LoRA under "any rule". Three rules could not: `fedprox` and
`scaffold` loaded the broadcast into the whole model's state_dict and failed
in round 1 on an adapter-only one, and `fedlalr` looked its AMSGrad moments up
by parameter names that under a PEFT adapter carry the adapter name. Each got
past config load. They are now refused at load for an adapter-scoped model the
package builds, and by the client before any update for a task that reports
adapter scope. FINDINGS.csv POST-F29.

Every combination still advertised trains one real adapter-only round here, on
the tiny GPT-2 fixture `tests/test_lora_runner_resume.py` builds, with the
input embeddings given a seeded random fill: the fixture's constant embeddings
tie to a constant LM head, whose gradient is exactly zero, so nothing trains
on it and "the round ran" would say nothing. A combination counts as supported
only if its adapter moved.
"""

from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import torch
import yaml

from fedbrew.cli import dispatch
from fedbrew.core import registry, runner
from fedbrew.core.config import ADAPTER_SCOPED_MODELS, load_config
from fedbrew.core.federated_state import (
    ADAPTER_STATE_CLIENT_RULES,
    FULL_STATE_ONLY_CLIENT_RULES,
)
from fedbrew.tasks.classification.torch_classification import TorchClassificationTask

_HAS_LLM_DEPS = (
    importlib.util.find_spec("transformers") is not None
    and importlib.util.find_spec("peft") is not None
)

FEDAVG_SERVERS = {
    "fedavg": {},
    "fedopt": {
        "server_optimizer": "fedadam",
        "server_learning_rate": 0.01,
        "beta1": 0.9,
        "beta2": 0.99,
        "tau": 0.001,
    },
    "fedadam": {"server_learning_rate": 0.01, "beta1": 0.9, "beta2": 0.99, "tau": 0.001},
    "fedyogi": {"server_learning_rate": 0.01, "beta1": 0.9, "beta2": 0.99, "tau": 0.001},
    "fedadagrad": {"server_learning_rate": 0.01, "beta1": 0.0, "tau": 0.001},
}
_SGD = {
    "learning_rate": 0.05,
    "learning_rate_schedule": "constant",
    "min_learning_rate": 0.0,
    "momentum": 0.0,
    "weight_decay": 0.0,
    "nesterov": False,
}
_ENGINE = {**_SGD, "update_mode": "sequential_epoch", "frozen_gradient_weighting": "examples"}
RULES: dict[str, dict[str, Any]] = {
    "fedavg": _ENGINE,
    "centralized": _ENGINE,
    "fedavg_ft": {**_ENGINE, "finetune_epochs": 1, "finetune_learning_rate": None},
    "local_sgd": _SGD,
    "local_adamw": {
        "learning_rate": 0.01,
        "learning_rate_schedule": "constant",
        "min_learning_rate": 0.0,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8,
        "weight_decay": 0.0,
    },
    "delta_sgd": {"eta_0": 0.05},
    "fedprox": {"learning_rate": 0.05, "proximal_mu": 0.01},
    "scaffold": {"learning_rate": 0.05},
    "fedlalr": {"learning_rate": 0.01},
}

#: (server.strategy, client.update_rule) for every combination that trains
#: adapter state: each rule that can, with the fedavg server and every FedOpt
#: server, and centralized with its own strategy.
SUPPORTED = [
    (server, rule)
    for server in FEDAVG_SERVERS
    for rule in sorted(ADAPTER_STATE_CLIENT_RULES - {"centralized"})
] + [("centralized", "centralized")]
#: Every combination config load allows that pairs an adapter model with a rule
#: that cannot train it: fedprox with each server it pairs with, and the two
#: rules that bring their own.
REFUSED = [(server, "fedprox") for server in FEDAVG_SERVERS] + [
    ("scaffold", "scaffold"),
    ("fedlalr", "fedlalr"),
]


def _config(
    server: str,
    rule: str,
    output_dir: Path,
    *,
    asset_manifest: Path | str = "assets/asset_manifest.json",
    data_manifest: Path | str = "generated/manifest.json",
) -> dict[str, Any]:
    evaluation: dict[str, Any] = {
        "train": {"every": 1, "clients": "all"},
        "val": {"every": "never"},
        "test": {"every": "never"},
        "central_test": {"every": "never"},
    }
    if rule == "fedavg_ft":
        evaluation["model_scope"] = "both"
    return {
        "experiment": {"seed": 0, "output_dir": str(output_dir), "use_run_subdir": False},
        "server": {
            "strategy": server,
            "participation_rate": 1,
            "metrics": ["fit_loss"],
            **FEDAVG_SERVERS.get(server, {}),
        },
        "client": {"update_rule": rule, "batch_size": 1, "metrics": ["fit_loss"], **RULES[rule]},
        "data": {"name": "manifest_dataset", "path": str(data_manifest)},
        "model": {
            "name": "hf_causal_lm_lora",
            "asset_manifest": str(asset_manifest),
            "preparation_config": str(Path("configs/llm_assets/tiny_gpt2.yaml").resolve()),
            "sequence_length": 4,
            "local_files_only": True,
            "trust_remote_code": False,
            "adapter_name": "support_probe",
            "r": 2,
            "lora_alpha": 4,
            "lora_dropout": 0.0,
            "target_modules": ["c_attn"],
            "bias": "none",
        },
        "runtime": {
            "deterministic": True,
            "deterministic_warn_only": True,
            "device": "cpu",
            "use_amp": False,
            "checkpointing": {
                "enabled": True,
                "save_last": True,
                "save_best": False,
                "save_every_round": False,
                "keep_last": 0,
            },
            "performance": {"matmul_precision": "highest"},
        },
        "evaluation": evaluation,
        "defaults": {"global_rounds": 1, "local_iterations": 1},
    }


def _fedbrew(*argv: str) -> tuple[int | str | None, str]:
    output = StringIO()
    code: int | str | None = None
    with redirect_stderr(output), redirect_stdout(output):
        try:
            dispatch.main(list(argv))
        except SystemExit as exit_:
            code = exit_.code
    return code, " ".join(output.getvalue().split())


@pytest.mark.fast
class TheTwoListsCoverEveryRuleTest(unittest.TestCase):
    def test_every_built_in_rule_is_on_exactly_one_list(self) -> None:
        registry.register_builtin_components()
        builtin = set(registry.client_updates.builtin())
        self.assertEqual(ADAPTER_STATE_CLIENT_RULES | set(FULL_STATE_ONLY_CLIENT_RULES), builtin)
        self.assertFalse(ADAPTER_STATE_CLIENT_RULES & set(FULL_STATE_ONLY_CLIENT_RULES))

    def test_the_adapter_model_is_the_lora_builder(self) -> None:
        self.assertEqual(ADAPTER_SCOPED_MODELS, {"hf_causal_lm_lora"})


class TheConfigRefusesThemBeforeAnythingRunsTest(unittest.TestCase):
    """Through `fedbrew run` and `--validate-only`; needs no `llm` extra."""

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)

    def _path(self, server: str, rule: str) -> tuple[Path, Path]:
        output = self.root / f"out_{server}_{rule}"
        path = self.root / f"{server}_{rule}.yaml"
        path.write_text(yaml.safe_dump(_config(server, rule, output)), encoding="utf-8")
        return path, output

    def _assert_names_rule_and_alternatives(self, rule: str, text: str) -> None:
        self.assertIn(f"client.update_rule={rule} cannot train adapter-only (LoRA) state", text)
        self.assertIn(", ".join(sorted(ADAPTER_STATE_CLIENT_RULES)), text)

    def test_fedbrew_run_refuses_each_and_writes_nothing(self) -> None:
        for server, rule in REFUSED:
            with self.subTest(server=server, rule=rule):
                path, output = self._path(server, rule)
                code, text = _fedbrew("run", "--config", str(path), "--quiet")
                self.assertEqual(code, 2)
                self.assertIn("RUN REFUSED", text)
                self._assert_names_rule_and_alternatives(rule, text)
                self.assertFalse(output.exists(), "a refused run wrote its output dir")

    def test_validate_only_refuses_each_and_writes_nothing(self) -> None:
        for server, rule in REFUSED:
            with self.subTest(server=server, rule=rule):
                path, output = self._path(server, rule)
                code, text = _fedbrew("run", "--config", str(path), "--validate-only")
                self.assertEqual(code, 1)
                self._assert_names_rule_and_alternatives(rule, text)
                self.assertFalse(output.exists())

    def test_every_supported_combination_loads(self) -> None:
        for server, rule in SUPPORTED:
            with self.subTest(server=server, rule=rule):
                load_config(self._path(server, rule)[0])


def _adapter_metadata(task: Any, model: Any) -> dict[str, Any]:
    """What a task reporting adapter scope says, e.g. an extension's."""

    return {
        "model_state_scope": "adapter",
        "base_model_identifier": "probe",
        "base_model_resolved_revision": "probe",
        "adapter_name": "probe",
        "lora_config": {},
    }


class TheClientRefusesATaskThatReportsAdapterScopeTest(unittest.TestCase):
    """A model config load cannot judge: the task says adapter at run time."""

    RULES = {
        "fedprox": {"learning_rate": 0.01, "proximal_mu": 0.01},
        "scaffold": {"learning_rate": 0.01},
        "fedlalr": {"learning_rate": 0.01},
    }

    def test_each_rule_refuses_before_its_first_update(self) -> None:
        for rule, options in self.RULES.items():
            with self.subTest(rule=rule), tempfile.TemporaryDirectory() as directory:
                raw = yaml.safe_load(Path("configs/dev/smoke.yaml").read_text(encoding="utf-8"))
                output = Path(directory) / "out"
                raw["experiment"]["output_dir"] = str(output)
                raw["server"]["strategy"] = rule if rule != "fedprox" else "fedavg"
                raw["client"] = {
                    "update_rule": rule,
                    "batch_size": 4,
                    "metrics": ["fit_loss"],
                    **options,
                }
                path = Path(directory) / "run.yaml"
                path.write_text(yaml.safe_dump(raw), encoding="utf-8")
                steps = mock.Mock(wraps=TorchClassificationTask.train_step)
                with (
                    mock.patch.object(
                        TorchClassificationTask,
                        "federated_model_state_metadata",
                        _adapter_metadata,
                    ),
                    mock.patch.object(TorchClassificationTask, "train_step", steps),
                    redirect_stdout(StringIO()),
                    self.assertRaises(ValueError) as caught,
                ):
                    runner.run(path, runner.parse_args(["--quiet"]))
                message = str(caught.exception)
                self.assertIn(f"client.update_rule={rule} cannot train adapter-only", message)
                self.assertIn(", ".join(sorted(ADAPTER_STATE_CLIENT_RULES)), message)
                self.assertEqual(steps.call_count, 0, "a training step ran before the refusal")
                written = sorted(p.name for p in output.rglob("*") if p.is_file())
                self.assertEqual(written, [], "the refused run left artifacts")


@unittest.skipUnless(_HAS_LLM_DEPS, "PEFT and Transformers are optional dependencies")
class EverySupportedCombinationTrainsAnAdapterRoundTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from test_hf_causal_lm_runtime import (
            _write_federated_fixture,
            _write_local_model_fixture,
        )
        from transformers import GPT2LMHeadModel

        cls._directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls._directory.name)
        cls.assets = _write_local_model_fixture(cls.root / "assets")
        model = GPT2LMHeadModel.from_pretrained(cls.root / "assets" / "model")
        generator = torch.Generator().manual_seed(0)
        with torch.no_grad():
            weight = model.get_input_embeddings().weight
            weight.copy_(torch.randn(weight.shape, generator=generator) * 0.5)
        model.save_pretrained(cls.root / "assets" / "model")
        cls.data = _write_federated_fixture(cls.root / "generated", cls.assets)
        cls._environment = mock.patch.dict(
            os.environ, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        )
        cls._environment.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._environment.stop()
        cls._directory.cleanup()

    def test_each_trains_one_adapter_only_round_that_moves_the_adapter(self) -> None:
        from fedbrew.core.checkpointing import load_checkpoint

        for server, rule in SUPPORTED:
            with self.subTest(server=server, rule=rule):
                output = self.root / f"out_{server}_{rule}"
                path = self.root / f"{server}_{rule}.yaml"
                config = _config(
                    server, rule, output, asset_manifest=self.assets, data_manifest=self.data
                )
                path.write_text(yaml.safe_dump(config), encoding="utf-8")
                with redirect_stdout(StringIO()):
                    state = runner.run(path, runner.parse_args(["--quiet"]))
                self.assertEqual(state.status, "completed")
                checkpoint = load_checkpoint(output / "checkpoints" / "latest.pt")
                self.assertEqual(checkpoint["model_state_scope"], "adapter")
                adapter = checkpoint["model_state"]
                self.assertTrue(adapter)
                self.assertTrue(all("lora_" in name for name in adapter))
                # PEFT starts lora_B at zero; a trained round leaves it nonzero.
                moved = [
                    name
                    for name, tensor in adapter.items()
                    if "lora_B" in name and bool(tensor.abs().max() > 0)
                ]
                self.assertTrue(moved, "the round ran but the adapter did not move")


if __name__ == "__main__":
    unittest.main()
