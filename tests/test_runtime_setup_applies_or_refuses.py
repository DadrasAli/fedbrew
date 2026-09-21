"""A performance setting is applied, or the run stops. It is never dropped.

`configure_runtime` wrapped the torch import, the CUDA probe and all three
performance settings in one `try` under a bare `except Exception` whose handler
returned early. So a `torch_num_threads` that `int()` could not read left
`cudnn_benchmark` and `matmul_precision` unapplied *and absent from the record*,
and the run trained at torch's default precision while the config asked for
another. The only trace was `runtime_setup_error`, a `run.json` key nothing
reads. Separately, `device: auto` on a node whose GPU is busy fell back to the
CPU without a word. P04-F07.

Three claims are guarded: the three settings a config can hold are refused by
`validate_config` before `configure_runtime` sees them; a setting that is
accepted is actually applied, `matmul_precision` included, whatever else is in
the block; and the `auto` fallback says why when there is a GPU it could not
use.
"""

from __future__ import annotations

import builtins
import unittest
import warnings
from typing import Any
from unittest import mock

import pytest
import torch

from fedbrew.core.config import load_config, validate_config
from fedbrew.core.runtime_setup import configure_runtime
from fedbrew.core.torch_utils import resolve_torch_device

pytestmark = pytest.mark.fast

BASE = "configs/dev/smoke.yaml"

#: Values a config can hold that torch rejects or silently misreads.
REFUSED: dict[str, tuple[Any, ...]] = {
    # int() raises on the first; set_num_threads raises on the other three.
    "torch_num_threads": ("not-an-int", 0, -4, 1.5, True),
    # bool("false") is True, so this one is not an error torch would raise.
    "cudnn_benchmark": ("false", "no", 1, None.__class__),
    "matmul_precision": ("highest ", "float32", 3),
}

ACCEPTED: dict[str, tuple[Any, ...]] = {
    "torch_num_threads": (1, 4),
    "cudnn_benchmark": (True, False),
    "matmul_precision": ("highest", "high", "medium"),
}


def _config(performance: dict[str, Any]) -> Any:
    config = load_config(BASE)
    config.runtime.extra = dict(config.runtime.extra)
    config.runtime.extra["performance"] = performance
    return config


class TheConfigIsRefusedBeforeTheRuntimeSeesItTest(unittest.TestCase):
    def test_a_value_the_runtime_cannot_apply_is_refused(self) -> None:
        for name, values in REFUSED.items():
            for value in values:
                with self.subTest(key=name, value=value):
                    with self.assertRaises(ValueError) as caught:
                        validate_config(_config({name: value}))
                    self.assertIn(name, str(caught.exception))

    def test_the_accepted_values_really_are_accepted(self) -> None:
        """A table that refused everything would pass a validator that does too."""

        for name, values in ACCEPTED.items():
            for value in values:
                with self.subTest(key=name, value=value):
                    validate_config(_config({name: value}))

    def test_cudnn_benchmark_false_as_a_string_is_the_inverted_one(self) -> None:
        """bool('false') is True: the config turned the autotuner on, not off."""

        self.assertTrue(bool("false"))
        with self.assertRaises(ValueError) as caught:
            validate_config(_config({"cudnn_benchmark": "false"}))
        self.assertIn("bool()", str(caught.exception))


class EverySettingIsAppliedTest(unittest.TestCase):
    def setUp(self) -> None:
        self.addCleanup(torch.set_float32_matmul_precision, torch.get_float32_matmul_precision())
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())

    def test_matmul_precision_is_applied_beside_every_other_key(self) -> None:
        """The one key that changes the numbers, and the one that was dropped."""

        for precision in ("high", "medium", "highest"):
            with self.subTest(matmul_precision=precision):
                torch.set_float32_matmul_precision("highest")
                config = _config(
                    {
                        "torch_num_threads": 2,
                        "cudnn_benchmark": False,
                        "matmul_precision": precision,
                    }
                )
                validate_config(config)
                record = configure_runtime(config, deterministic=False)
                self.assertEqual(torch.get_float32_matmul_precision(), precision)
                self.assertEqual(record["matmul_precision"], precision)
                self.assertEqual(torch.get_num_threads(), 2)
                self.assertNotIn("runtime_setup_error", record)

    def test_a_setting_that_cannot_be_applied_stops_the_run(self) -> None:
        """Past validate_config, a failure is a fault and must not be swallowed."""

        config = _config({"torch_num_threads": 2, "matmul_precision": "high"})
        with mock.patch.object(torch, "set_num_threads", side_effect=RuntimeError("no")):
            with self.assertRaises(RuntimeError):
                configure_runtime(config, deterministic=False)

    def test_a_missing_torch_is_still_recorded_rather_than_raised(self) -> None:
        """The one thing the except still covers: torch is an optional install."""

        real_import = builtins.__import__

        def refuse_torch(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "torch":
                raise ImportError("No module named 'torch'")
            return real_import(name, *args, **kwargs)

        config = _config({"matmul_precision": "high"})
        with mock.patch.object(builtins, "__import__", refuse_torch):
            record = configure_runtime(config, deterministic=False)
        self.assertIn("runtime_setup_error", record)
        self.assertIs(record["torch_available"], False)

    def test_a_cuda_probe_failure_is_recorded_and_does_not_stop_a_cpu_run(self) -> None:
        config = _config({"matmul_precision": "high"})
        with mock.patch.object(torch.cuda, "is_available", side_effect=RuntimeError("driver")):
            record = configure_runtime(config, deterministic=False)
        self.assertEqual(record["cuda_probe_error"], "driver")
        self.assertEqual(record["matmul_precision"], "high")


class TheAutoFallbackSaysWhyTest(unittest.TestCase):
    def _resolve(self, *, cuda: bool, allocates: bool) -> tuple[str, list[str]]:
        allocate = mock.DEFAULT if allocates else mock.Mock(side_effect=RuntimeError("all busy"))
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=cuda),
            mock.patch.object(torch, "empty", allocate),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            resolved = resolve_torch_device("auto")
        return resolved, [str(warning.message) for warning in caught]

    def test_a_gpu_that_cannot_be_allocated_on_is_named(self) -> None:
        resolved, messages = self._resolve(cuda=True, allocates=False)
        self.assertEqual(resolved, "cpu")
        self.assertEqual(len(messages), 1)
        self.assertIn("all busy", messages[0])

    def test_a_node_with_no_cuda_says_nothing(self) -> None:
        """Not surprising, so not a warning; only the unusable GPU is."""

        resolved, messages = self._resolve(cuda=False, allocates=False)
        self.assertEqual(resolved, "cpu")
        self.assertEqual(messages, [])

    def test_a_usable_gpu_says_nothing(self) -> None:
        resolved, messages = self._resolve(cuda=True, allocates=True)
        self.assertEqual(resolved, "cuda")
        self.assertEqual(messages, [])

    def test_an_explicit_device_is_never_probed(self) -> None:
        for device in ("cpu", "cuda"):
            with self.subTest(device=device):
                with mock.patch.object(torch.cuda, "is_available") as available:
                    self.assertEqual(resolve_torch_device(device), device)
                    available.assert_not_called()


if __name__ == "__main__":
    unittest.main()
