"""The causal-LM task must honour reuse_model, like the classification task.

TorchCausalLMTask had no model cache and factory.py passed reuse_model only for
`classification`, so the five LLM configs that set performance.reuse_model:
true got a fresh model on every fit and every evaluate. Beyond the throughput
cost -- a from_pretrained of a 0.5B-parameter model per client per phase --
PEFT initialises lora_A with kaiming_uniform_ from the process-wide RNG, so
each rebuild also advanced the global stream. The number of draws then depends
on how many clients are *evaluated* -- evaluation contaminating the training
stream, arriving here without a fine-tuning step.
"""

from __future__ import annotations

import unittest
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from fedbrew.core.registry import MODEL_TASKS, models, register_builtin_components
from fedbrew.tasks.base import model_config_key
from fedbrew.tasks.causal_lm.torch_causal_lm import TorchCausalLMTask

pytestmark = pytest.mark.fast

#: A stand-in for the LoRA adapter: it draws from the process-wide RNG at
#: construction, exactly as kaiming_uniform_ does, with none of the LLM
#: dependencies.
BUILDS = "rng_probe"


class _RngProbeModel(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(width, width))


def _build_rng_probe(values: Mapping[str, Any]) -> nn.Module:
    return _RngProbeModel(int(values.get("width", 4)))


class _RegistryFixture(unittest.TestCase):
    """Register the probe model once, and put the registry back afterwards."""

    def setUp(self) -> None:
        register_builtin_components()
        if not models.exists(BUILDS):
            models.register(BUILDS, _build_rng_probe, task="causal_lm")
            self.addCleanup(models._items.pop, BUILDS, None)
            self.addCleanup(models._origins.pop, BUILDS, None)
            self.addCleanup(MODEL_TASKS.pop, BUILDS, None)


def _rng_moves(build) -> bool:
    before = torch.get_rng_state().clone()
    build()
    return not torch.equal(before, torch.get_rng_state())


class CausalLmModelReuseTests(_RegistryFixture):
    CONFIG = {"name": BUILDS, "width": 4}

    def _task(self, reuse_model: bool) -> TorchCausalLMTask:
        return TorchCausalLMTask(
            model_config=dict(self.CONFIG),
            batch_size=2,
            device="cpu",
            reuse_model=reuse_model,
        )

    def test_reuse_returns_one_instance_and_stops_drawing(self) -> None:
        task = self._task(True)
        torch.manual_seed(0)
        first = task.build_model()
        self.assertIs(task.build_model(), first)
        # The first build constructs and draws; later ones must not.
        self.assertFalse(_rng_moves(task.build_model))

    def test_reuse_false_gives_independent_instances(self) -> None:
        task = self._task(False)
        torch.manual_seed(0)
        first = task.build_model()
        second = task.build_model()
        self.assertIsNot(first, second)
        self.assertFalse(torch.equal(first.weight, second.weight))
        self.assertTrue(_rng_moves(task.build_model))

    def test_reuse_is_on_by_default(self) -> None:
        task = TorchCausalLMTask(model_config=dict(self.CONFIG), batch_size=2, device="cpu")
        self.assertTrue(task.reuse_model)
        torch.manual_seed(0)
        self.assertIs(task.build_model(), task.build_model())

    def test_a_different_architecture_gets_its_own_cache_entry(self) -> None:
        task = self._task(True)
        torch.manual_seed(0)
        narrow = task.build_model({"width": 4})
        wide = task.build_model({"width": 8})
        self.assertIsNot(narrow, wide)
        self.assertIs(task.build_model({"width": 8}), wide)


class FactoryPassesReuseModelTests(unittest.TestCase):
    def test_the_causal_lm_task_is_built_with_reuse_model(self) -> None:
        from fedbrew.core.factory import _build_task

        seen: dict[str, Any] = {}

        def _factory(**kwargs: Any) -> Any:
            seen.update(kwargs)
            return object()

        for task_name in ("classification", "causal_lm"):
            with self.subTest(task=task_name):
                seen.clear()
                _build_task(_config(task_name, reuse_model=False), _factory, {})
                self.assertIn("reuse_model", seen)
                self.assertFalse(seen["reuse_model"])
                seen.clear()
                _build_task(_config(task_name, reuse_model=True), _factory, {})
                self.assertTrue(seen["reuse_model"])

    def test_reuse_model_defaults_to_true_when_unset(self) -> None:
        from fedbrew.core.factory import _build_task

        seen: dict[str, Any] = {}
        _build_task(_config("causal_lm", reuse_model=None), lambda **kw: seen.update(kw), {})
        self.assertTrue(seen["reuse_model"])


class SharedCacheKeyTests(unittest.TestCase):
    def test_both_tasks_use_the_same_key_function(self) -> None:
        # Two copies of this logic could disagree about when two configs
        # describe the same architecture; there is one.
        import fedbrew.tasks.causal_lm.torch_causal_lm as causal
        import fedbrew.tasks.classification.torch_classification as classification

        self.assertIs(causal.model_config_key, model_config_key)
        self.assertIs(classification.model_config_key, model_config_key)

    def test_the_key_is_order_independent(self) -> None:
        self.assertEqual(
            model_config_key({"name": "mlp", "hidden_dim": 16}),
            model_config_key({"hidden_dim": 16, "name": "mlp"}),
        )
        self.assertNotEqual(
            model_config_key({"name": "mlp", "hidden_dim": 16}),
            model_config_key({"name": "mlp", "hidden_dim": 32}),
        )


#: A real config, so _build_task reads the fields it actually reads. smoke.yaml
#: is the classification arm; tiny_causal_lm.yaml is the causal_lm one.
TASK_CONFIGS = {
    "classification": "configs/dev/smoke.yaml",
    "causal_lm": "configs/dev/tiny_causal_lm.yaml",
}


def _config(task_name: str, reuse_model: bool | None) -> Any:
    from dataclasses import replace

    from fedbrew.core.config import load_config

    root = Path(__file__).resolve().parent.parent
    config = load_config(root / TASK_CONFIGS[task_name])
    performance = dict(config.runtime.extra.get("performance") or {})
    performance.pop("reuse_model", None)
    if reuse_model is not None:
        performance["reuse_model"] = reuse_model
    extra = {**config.runtime.extra, "performance": performance}
    return replace(config, runtime=replace(config.runtime, extra=extra))


if __name__ == "__main__":
    unittest.main()
