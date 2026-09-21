"""`deterministic: true` must mean deterministic, not "warns when it is not".

Every run config used to set deterministic_warn_only: true, and both the library
and the runner defaulted to it. warn_only=True hands torch a licence to pick a
nondeterministic kernel and print one warning -- once per process, to
logs_and_errs/*.err, recorded nowhere. Measured on an A100-40GB (torch 2.5.1,
CUDA 11.8, August 2026), Qwen2.5-0.5B at the shipped shape gave 2 distinct
gradient digests over 6 identical passes with warn_only=True and 1 with
warn_only=False.
"""

from __future__ import annotations

import unittest
import warnings
from pathlib import Path
from typing import Any

import pytest
import yaml

from fedbrew.core.runtime_setup import _enable_torch_determinism, seed_everything

CONFIG_ROOT = Path(__file__).resolve().parent.parent / "configs"

#: Models whose attention has no deterministic kernel. Only these may opt out.
#: scaled_dot_product_attention's flash and memory-efficient backends have no
#: deterministic implementation, so warn_only: false raises for them rather
#: than falling back.
ATTENTION_MODELS = frozenset({"hf_causal_lm", "hf_causal_lm_lora", "tiny_gpt2"})


class _FakeBackendCudnn:
    def __init__(self) -> None:
        self.benchmark = True
        self.deterministic = False


class _FakeTorch:
    """Records what use_deterministic_algorithms was actually asked for."""

    def __init__(self) -> None:
        self.calls: list[tuple[bool, Any]] = []
        self.backends = type("_B", (), {"cudnn": _FakeBackendCudnn()})()

    def use_deterministic_algorithms(self, mode: bool, warn_only: bool = False) -> None:
        self.calls.append((mode, warn_only))


def _run_configs() -> list[tuple[Path, dict]]:
    configs = []
    for path in sorted(CONFIG_ROOT.rglob("*.yaml")):
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and "runtime" in loaded:
            configs.append((path, loaded))
    return configs


@pytest.mark.fast
class LibraryDefaultTests(unittest.TestCase):
    def test_enabling_determinism_asks_torch_to_raise(self) -> None:
        fake = _FakeTorch()
        _enable_torch_determinism(fake, warn_only=False)
        self.assertEqual(fake.calls, [(True, False)])
        # And the cudnn flags it also owns.
        self.assertFalse(fake.backends.cudnn.benchmark)
        self.assertTrue(fake.backends.cudnn.deterministic)

    def test_warn_only_is_passed_through_when_asked_for(self) -> None:
        fake = _FakeTorch()
        _enable_torch_determinism(fake, warn_only=True)
        self.assertEqual(fake.calls, [(True, True)])

    def test_seed_everything_defaults_to_strict(self) -> None:
        metadata = seed_everything(7, deterministic=True)
        self.assertFalse(metadata["deterministic_warn_only"])
        self.assertTrue(metadata["deterministic"])


@pytest.mark.fast
class RunnerDefaultTests(unittest.TestCase):
    def test_a_config_that_is_silent_gets_strict_determinism(self) -> None:
        from fedbrew.core.runner import _runtime_extra_bool

        class _Runtime:
            extra: dict[str, Any] = {"deterministic": True}

        class _Config:
            runtime = _Runtime()

        self.assertIs(_runtime_extra_bool(_Config(), "deterministic_warn_only", False), False)


@pytest.mark.fast
class ShippedConfigTests(unittest.TestCase):
    def test_only_attention_models_may_downgrade_to_a_warning(self) -> None:
        seen = 0
        for path, config in _run_configs():
            runtime = config.get("runtime") or {}
            if "deterministic_warn_only" not in runtime:
                continue
            seen += 1
            model = (config.get("model") or {}).get("name")
            with self.subTest(config=str(path.relative_to(CONFIG_ROOT))):
                if runtime["deterministic_warn_only"]:
                    # An opt-out is a throughput decision. It has to be one the
                    # model actually forces, not one inherited by copy-paste.
                    self.assertIn(model, ATTENTION_MODELS)
                else:
                    self.assertNotIn(model, ATTENTION_MODELS)
        # 26 dataset arms, plus the 69 examples/ arms: drift-quad's three
        # dial settings of eight, pl-1d's seven, fed-lasso's nine in each of
        # its two full settings plus the one-arm null control, simplex-lsq's
        # eight plus three, and nonconvex-simplex's eight. The count is
        # pinned so a config cannot quietly stop stating its strictness.
        self.assertEqual(seen, 95)

    def test_every_config_that_sets_determinism_states_its_strictness(self) -> None:
        for path, config in _run_configs():
            runtime = config.get("runtime") or {}
            if not runtime.get("deterministic"):
                continue
            with self.subTest(config=str(path.relative_to(CONFIG_ROOT))):
                self.assertIn("deterministic_warn_only", runtime)


class VisionModelsSurviveStrictDeterminismTests(unittest.TestCase):
    """The flip must cost the vision tracks nothing.

    Probed on CPU here; the A100-40GB measurement behind the config change
    (torch 2.5.1, CUDA 11.8, August 2026) covered
    AdaptiveAvgPool2d, Conv2d, GroupNorm, CrossEntropyLoss, Dropout, Linear,
    index_select and Embedding backward under warn_only=False, none of which
    warns or raises.
    """

    def test_a_forward_and_backward_neither_warns_nor_raises(self) -> None:
        import torch

        from fedbrew.models.femnist_resnet import build_femnist_resnet18
        from fedbrew.models.openimage_shufflenet import build_openimage_shufflenet
        from fedbrew.models.torch_mlp import build_torch_mlp

        cases = {
            "femnist_resnet18": (
                lambda: build_femnist_resnet18({"num_classes": 62}),
                (4, 1, 28, 28),
                62,
            ),
            "openimage_shufflenet": (
                lambda: build_openimage_shufflenet({"num_classes": 596}),
                (2, 3, 96, 96),
                596,
            ),
            "mlp": (
                lambda: build_torch_mlp({"input_dim": 784, "hidden_dim": 64, "num_classes": 10}),
                (4, 784),
                10,
            ),
        }
        previous = torch.are_deterministic_algorithms_enabled()
        previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(True, warn_only=False)
        try:
            for name, (build, shape, num_classes) in cases.items():
                with self.subTest(model=name):
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        torch.manual_seed(0)
                        model = build()
                        features = torch.randn(*shape)
                        targets = torch.randint(0, num_classes, (shape[0],))
                        loss = torch.nn.functional.cross_entropy(model(features), targets)
                        loss.backward()
                    determinism_warnings = [
                        str(item.message)
                        for item in caught
                        if "determinis" in str(item.message).lower()
                    ]
                    self.assertEqual(determinism_warnings, [])
        finally:
            torch.use_deterministic_algorithms(previous, warn_only=previous_warn)


if __name__ == "__main__":
    unittest.main()
