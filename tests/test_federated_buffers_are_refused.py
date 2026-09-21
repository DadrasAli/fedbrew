"""A model that would federate a registered buffer is refused before round 1.

`WeightedStateAccumulator` is handed tensors, not a model, so it cannot tell a
buffer from a parameter. Its handling of the two is one-sided, and by accident
rather than by design:

    if value_cpu.is_floating_point():        -> add into the running sum
    elif ... not torch.equal(...):           -> raise

The `elif` is BatchNorm's `num_batches_tracked`, and it is the loud case
`tests/test_batchnorm_is_unsupported.py` and docs/06 §4 already cover: clients
that took different numbers of steps fail aggregation outright. The `if` is
`running_mean` and `running_var`, and it is silent. So a BatchNorm model whose
clients happen to take the *same* number of steps -- equal shard sizes with
`drop_last: true`, or `max_local_steps` -- averages running statistics across
non-IID clients as though they were weights, and FedAdam or FedYogi then
applies an adaptive update to them. Nothing at any layer said so.

The refusal has to be somewhere that has the model, since the accumulator does
not, and `factory.build_components` is the one place with both the model and
the federated state, once per run rather than once per fit.

`persistent_buffer_keys` is the predicate, and both of its exclusions were
measured rather than assumed:

    tiny GPT-2   named_buffers()  attn.bias, attn.masked_bias  -- non-persistent
                 state_dict - named_parameters()  lm_head.weight -- a tied alias
                 persistent_buffer_keys()  []
    Qwen2        named_buffers()  model.rotary_emb.inv_freq    -- non-persistent
                 persistent_buffer_keys()  []

A check written on `named_buffers()` would have refused both LLM models; one
that skipped the tied-alias exclusion would have refused GPT-2. Every model
this repository ships returns an empty list, so this refuses only a model that
does not exist yet.
"""

from __future__ import annotations

import unittest
from importlib.util import find_spec

import pytest
import torch
from torch import nn

from fedbrew.core.registry import MODEL_TASKS, models, register_builtin_components
from fedbrew.core.torch_utils import WeightedStateAccumulator, persistent_buffer_keys

pytestmark = pytest.mark.fast

HAS_LLM_DEPS = find_spec("transformers") is not None

#: The five classification builders, with the config each needs. Built rather
#: than read, because the claim is about what the model registers at runtime.
CLASSIFICATION_MODELS = ("mlp", "cnn", "small_cnn", "femnist_resnet18", "openimage_shufflenet")


class _BatchNormNet(nn.Module):
    """The model the refusal exists for: three buffers, two of them floating."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.norm = nn.BatchNorm1d(4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.norm(self.linear(inputs))


class _NonPersistentBufferNet(nn.Module):
    """A buffer that never reaches the state dict, so never reaches the wire."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.register_buffer("scratch", torch.ones(4), persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs) * self.scratch


class _TiedNet(nn.Module):
    """Two state-dict keys, one parameter -- GPT-2's shape, minimally."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Linear(4, 4, bias=False)
        self.head = nn.Linear(4, 4, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(inputs))


class ThePredicateTest(unittest.TestCase):
    def test_a_batch_norm_model_reports_its_three_buffers(self) -> None:
        self.assertEqual(
            persistent_buffer_keys(_BatchNormNet()),
            ["norm.num_batches_tracked", "norm.running_mean", "norm.running_var"],
        )

    def test_a_non_persistent_buffer_is_not_reported(self) -> None:
        model = _NonPersistentBufferNet()
        self.assertIn("scratch", dict(model.named_buffers()))
        self.assertNotIn("scratch", model.state_dict())
        self.assertEqual(persistent_buffer_keys(model), [])

    def test_a_tied_parameter_is_not_reported(self) -> None:
        """It is in the state dict and not in named_parameters, and is not a buffer."""

        model = _TiedNet()
        parameters = dict(model.named_parameters())
        untracked = [name for name in model.state_dict() if name not in parameters]
        self.assertEqual(untracked, ["head.weight"])
        self.assertEqual(persistent_buffer_keys(model), [])

    def test_a_plain_model_reports_nothing(self) -> None:
        self.assertEqual(persistent_buffer_keys(nn.Linear(4, 4)), [])


class NoShippedModelHasOneTest(unittest.TestCase):
    def setUp(self) -> None:
        register_builtin_components()

    def test_the_classification_builders(self) -> None:
        for name in CLASSIFICATION_MODELS:
            with self.subTest(model=name):
                self.assertEqual(persistent_buffer_keys(models.get(name)(None)), [])

    @unittest.skipUnless(HAS_LLM_DEPS, "requires the llm optional dependencies")
    def test_tiny_gpt2_despite_two_named_buffers_and_a_tied_head(self) -> None:
        model = models.get("tiny_gpt2")(
            {"vocab_size": 64, "sequence_length": 8, "n_layer": 1, "n_head": 1, "n_embd": 8}
        )
        self.assertTrue([name for name, _ in model.named_buffers()], "the mask buffers moved")
        self.assertEqual(persistent_buffer_keys(model), [])


class TheFactoryRefusesItTest(unittest.TestCase):
    """End to end, through the path a run takes."""

    NAME = "buffer_probe_model"

    def setUp(self) -> None:
        register_builtin_components()
        models.register(self.NAME, lambda config=None: _BatchNormNet(), task="classification")
        self.addCleanup(models._items.pop, self.NAME, None)
        self.addCleanup(models._origins.pop, self.NAME, None)
        self.addCleanup(MODEL_TASKS.pop, self.NAME, None)

    def _config(self, model_name: str) -> object:
        import copy
        import tempfile
        from pathlib import Path

        import yaml

        from fedbrew.core.config import load_config

        base = Path("configs/dev/synthetic.yaml")
        raw = yaml.safe_load(base.read_text(encoding="utf-8"))
        raw["model"] = {"name": model_name}
        if model_name == "mlp":
            raw["model"] = {"name": "mlp", "input_dim": 5, "hidden_dim": 16, "num_classes": 2}
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            yaml.safe_dump(raw, handle)
            path = handle.name
        return copy.deepcopy(load_config(path))

    def test_building_a_batch_norm_arm_raises_and_names_every_buffer(self) -> None:
        from fedbrew.core.factory import build_components

        with self.assertRaises(ValueError) as caught:
            build_components(self._config(self.NAME))
        message = str(caught.exception)
        for name in ("norm.running_mean", "norm.running_var", "norm.num_batches_tracked"):
            self.assertIn(name, message)
        self.assertIn("GroupNorm", message)
        self.assertIn("get_federated_model_state", message)

    def test_the_shipped_arm_beside_it_still_builds(self) -> None:
        """A refusal that also refused the working case would be worse."""

        from fedbrew.core.factory import build_components

        self.assertIsNotNone(build_components(self._config("mlp")))


class WhatTheRefusalPreventsTest(unittest.TestCase):
    """The silent path, kept runnable so the reason is a number.

    Unchanged behaviour -- the accumulator still does this, and
    tests/test_batchnorm_is_unsupported.py still pins it. What changed is that
    nothing can reach it through the factory any more.
    """

    def test_equal_step_counts_average_running_statistics_without_complaint(self) -> None:
        first = {
            "norm.running_mean": torch.tensor([0.0, 1.0]),
            "norm.num_batches_tracked": torch.tensor(5),
        }
        second = {
            "norm.running_mean": torch.tensor([2.0, 3.0]),
            "norm.num_batches_tracked": torch.tensor(5),
        }
        accumulator = WeightedStateAccumulator()
        accumulator.add(first, 30.0)
        accumulator.add(second, 10.0)
        result = accumulator.result()

        # 30/40 of one client's statistics plus 10/40 of the other's -- an
        # example-weighted mean of two running means, which is a quantity no
        # client ever computed and no batch ever produced.
        self.assertTrue(
            torch.allclose(result["norm.running_mean"], torch.tensor([0.5, 1.5])),
            result["norm.running_mean"],
        )
        # And the counter came through untouched, because the two agreed.
        self.assertEqual(int(result["norm.num_batches_tracked"]), 5)

    def test_unequal_step_counts_are_the_loud_half_of_the_same_asymmetry(self) -> None:
        accumulator = WeightedStateAccumulator()
        accumulator.add({"norm.num_batches_tracked": torch.tensor(5)}, 1.0)
        with self.assertRaisesRegex(ValueError, "non-floating state tensor"):
            accumulator.add({"norm.num_batches_tracked": torch.tensor(7)}, 1.0)


if __name__ == "__main__":
    unittest.main()
