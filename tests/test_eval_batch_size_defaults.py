"""Evaluation batch-size defaults must respect the task's logit size."""

from __future__ import annotations

import unittest
from dataclasses import replace

import pytest

from fedbrew.core.config import load_config
from fedbrew.core.factory import DEFAULT_MIN_EVAL_BATCH_SIZE, _eval_batch_size

pytestmark = pytest.mark.fast

CLASSIFICATION_CONFIG = "configs/dev/smoke.yaml"
CAUSAL_LM_CONFIG = "configs/dev/tiny_causal_lm.yaml"


class EvalBatchSizeDefaultsTest(unittest.TestCase):
    def test_classification_raises_the_batch_size_to_the_floor(self) -> None:
        config = load_config(CLASSIFICATION_CONFIG)
        self.assertEqual(config.task.name, "classification")
        self.assertLess(config.client.batch_size, DEFAULT_MIN_EVAL_BATCH_SIZE)
        self.assertEqual(_eval_batch_size(config), DEFAULT_MIN_EVAL_BATCH_SIZE)

    def test_causal_lm_evaluates_at_the_training_batch_size(self) -> None:
        # A causal-LM forward allocates batch x sequence_length x vocabulary
        # logits, so the classification floor would need tens of gigabytes.
        config = load_config(CAUSAL_LM_CONFIG)
        self.assertEqual(config.task.name, "causal_lm")
        self.assertEqual(_eval_batch_size(config), config.client.batch_size)

    def test_an_explicit_setting_overrides_both_defaults(self) -> None:
        for path in (CLASSIFICATION_CONFIG, CAUSAL_LM_CONFIG):
            with self.subTest(config=path):
                config = load_config(path)
                extra = dict(config.client.extra)
                extra["eval_batch_size"] = 16
                config = replace(config, client=replace(config.client, extra=extra))
                self.assertEqual(_eval_batch_size(config), 16)


if __name__ == "__main__":
    unittest.main()
