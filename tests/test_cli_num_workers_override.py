"""--num-workers has to reach the DataLoader, including from a bare config.

`runtime.num_workers` was a required field that nothing read. The loader takes
its worker count from `runtime.performance.dataloader.num_workers`, and
apply_cli_overrides wrote the flag into that nested dict only when both blocks
already existed -- so on the ten shipped configs with no `runtime.performance`
block, `--num-workers 8` set a field nobody consumed and left the loader at 0.
Two shipped MNIST configs stated `num_workers: 10` and ran at 0 for the same
reason.
"""

from __future__ import annotations

import argparse
import unittest

import pytest

from fedbrew.core.config import load_config, validate_config
from fedbrew.core.factory import _dataloader_config
from fedbrew.core.runner import apply_cli_overrides

#: The config the bare case is derived from. Every shipped config now carries
#: a `runtime.performance` block -- matmul_precision was written into all 30 --
#: so the block-less case has to be built rather than picked. It is still the
#: case that matters: a config written from scratch, or any of the ten this
#: repository shipped until that commit.
FULL_CONFIG = "configs/femnist/fedavg.yaml"


def _args(**overrides: object) -> argparse.Namespace:
    return argparse.Namespace(**overrides)


def _bare_config() -> object:
    """A loaded config with no `runtime.performance` block at all."""

    config = load_config(FULL_CONFIG)
    config.runtime.extra.pop("performance", None)
    return config


@pytest.mark.fast
class NumWorkersOverrideTest(unittest.TestCase):
    def test_a_config_with_no_performance_block_still_validates(self) -> None:
        """The block is optional, so the bare case is a config a user can
        write -- not merely one this repository no longer ships."""

        config = _bare_config()
        self.assertNotIn("performance", config.runtime.extra)
        validate_config(config)

    def test_flag_reaches_the_loader_from_a_config_with_no_performance_block(
        self,
    ) -> None:
        config = _bare_config()
        self.assertEqual(_dataloader_config(config).get("num_workers"), None)

        overridden = apply_cli_overrides(config, _args(num_workers=4))

        self.assertEqual(_dataloader_config(overridden).get("num_workers"), 4)
        validate_config(overridden)

    def test_flag_overrides_an_existing_dataloader_block(self) -> None:
        config = load_config(FULL_CONFIG)
        self.assertEqual(_dataloader_config(config).get("num_workers"), 0)

        overridden = apply_cli_overrides(config, _args(num_workers=6))

        self.assertEqual(_dataloader_config(overridden).get("num_workers"), 6)
        validate_config(overridden)

    def test_the_override_leaves_the_other_performance_keys_alone(self) -> None:
        """Creating the block must not drop what the config already set."""

        config = load_config(FULL_CONFIG)
        before = dict(config.runtime.extra["performance"])

        overridden = apply_cli_overrides(config, _args(num_workers=2))
        after = overridden.runtime.extra["performance"]

        for key, value in before.items():
            if key == "dataloader":
                continue
            with self.subTest(key=key):
                self.assertEqual(after[key], value)
        self.assertEqual(
            after["dataloader"]["pin_memory"],
            before["dataloader"]["pin_memory"],
        )

    def test_no_flag_leaves_the_config_untouched(self) -> None:
        overridden = apply_cli_overrides(_bare_config(), _args(num_workers=None))
        self.assertNotIn("performance", overridden.runtime.extra)

    def test_runtime_num_workers_is_refused_and_names_its_replacement(self) -> None:
        config = load_config(FULL_CONFIG)
        config.runtime.extra["num_workers"] = 10

        with self.assertRaises(ValueError) as caught:
            validate_config(config)

        message = str(caught.exception)
        self.assertIn("runtime.num_workers has been removed", message)
        self.assertIn("runtime.performance.dataloader.num_workers", message)


@pytest.mark.fast
class DataloaderBlockKeysTest(unittest.TestCase):
    """The block accepts only what the loader takes from a config.

    build_dataloader also reads batch_size, shuffle, drop_last and seed, but
    the client supplies all four per call and the per-call dict wins the merge
    -- except drop_last on the evaluation path, which the client does not set,
    so a config-level drop_last: true silently dropped the last partial batch
    of every evaluated split.
    """

    SHADOWED = ("batch_size", "shuffle", "drop_last", "seed")

    def test_each_shadowed_key_is_refused_with_its_reason(self) -> None:
        for key in self.SHADOWED:
            with self.subTest(key=key):
                config = load_config(FULL_CONFIG)
                config.runtime.extra["performance"]["dataloader"][key] = 1

                with self.assertRaises(ValueError) as caught:
                    validate_config(config)

                message = str(caught.exception)
                self.assertIn(
                    f"runtime.performance.dataloader.{key} has been removed",
                    message,
                )

    def test_a_misspelled_loader_key_is_refused(self) -> None:
        config = load_config(FULL_CONFIG)
        config.runtime.extra["performance"]["dataloader"]["pin_memoy"] = True

        with self.assertRaises(ValueError) as caught:
            validate_config(config)

        self.assertIn("pin_memoy", str(caught.exception))

    def test_the_four_live_keys_validate(self) -> None:
        config = load_config(FULL_CONFIG)
        config.runtime.extra["performance"]["dataloader"].update(
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )
        validate_config(config)


class ShippedConfigTest(unittest.TestCase):
    def test_no_shipped_config_still_sets_the_removed_field(self) -> None:
        import glob

        for path in sorted(glob.glob("configs/**/*.yaml", recursive=True)):
            with self.subTest(config=path):
                # load_config raises on the removed key, so loading every
                # shipped config is the assertion.
                if path.startswith("configs/llm_assets/"):
                    continue
                load_config(path)


if __name__ == "__main__":
    unittest.main()
