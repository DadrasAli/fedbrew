"""Whether clients are built on demand follows from the dataset, not a key.

client.lazy_clients could override it. No shipped config set it and no test
exercised it, and neither dataset has a reason to want the other's behaviour:
a manifest dataset reads thousands of client shards from disk, which is what
LazyClientPool exists to avoid materialising up front, and the in-process
synthetic dataset is already resident, where the pool would add indirection
and nothing else.
"""

from __future__ import annotations

import unittest
from typing import Any

import pytest

from fedbrew.core.config import (
    ClientConfig,
    DataConfig,
    ExperimentConfig,
    FullConfig,
    ModelConfig,
    RuntimeConfig,
    ServerConfig,
    TaskConfig,
    validate_config,
)
from fedbrew.core.factory import _use_lazy_clients

pytestmark = pytest.mark.fast

#: fedavg's required client extras, so validate_config reaches the key under
#: test instead of stopping at a missing hyperparameter.
_FEDAVG_EXTRA: dict[str, Any] = {
    "momentum": 0.0,
    "weight_decay": 0.0,
    "nesterov": False,
    "learning_rate_schedule": "constant",
    "min_learning_rate": 0.0,
    "update_mode": "sequential_epoch",
    "frozen_gradient_weighting": "examples",
}


def _config(data_name: str, **client_extra: Any) -> FullConfig:
    return FullConfig(
        experiment=ExperimentConfig(seed=0, output_dir="outputs/test"),
        server=ServerConfig(strategy="fedavg", global_rounds=1, participation_rate=1.0, metrics=[]),
        client=ClientConfig(
            update_rule="fedavg",
            local_iterations=1,
            batch_size=1,
            metrics=[],
            learning_rate=0.1,
            extra={**_FEDAVG_EXTRA, **client_extra},
        ),
        task=TaskConfig(name="classification"),
        data=DataConfig(name=data_name),
        model=ModelConfig(name="mlp"),
        runtime=RuntimeConfig(device="cpu", use_amp=False),
    )


class LazyClientSelectionTest(unittest.TestCase):
    def test_a_manifest_dataset_builds_clients_lazily(self) -> None:
        self.assertTrue(_use_lazy_clients(_config("manifest_dataset")))

    def test_the_synthetic_dataset_builds_clients_eagerly(self) -> None:
        self.assertFalse(_use_lazy_clients(_config("synthetic_classification")))

    def test_the_override_is_refused_and_says_what_decides(self) -> None:
        with self.assertRaises(ValueError) as caught:
            validate_config(_config("manifest_dataset", lazy_clients=False))

        message = str(caught.exception)
        self.assertIn("client.lazy_clients has been removed", message)
        self.assertIn("follows from the dataset", message)


if __name__ == "__main__":
    unittest.main()
