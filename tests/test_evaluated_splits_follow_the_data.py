"""A run evaluates only the client splits its data carries, and says so before it trains.

``evaluation.val`` and ``evaluation.test`` default to every 5 and every 10
rounds whatever the data holds. A dataset without one of them -- the
``synthetic_classification`` generator writes no val split, no causal-LM
generator a test split -- used to train its first round and then stop on "no
client reported a non-empty val split", after preflight had said "READY TO
RUN". ``fedbrew/core/evaluated_splits.py`` resolves the schedules against the
manifest's client records when the config loads. Pinned here:

- a split the data does not carry and the config does not name is not
  evaluated, the resolved config records it, and the plan header says
  "not evaluated (no <split> data)";
- a config that asks for such a split is refused at load, which is also
  preflight's first check, naming the split and the data;
- ``every: never`` for it is accepted;
- a count a record does not state is taken from its total when the other two
  are stated (the causal-LM generators write no test count), and a split
  whose count is unknown for any client is left as configured;
- the shipped configs on such data (dev/synthetic_manifest, the six
  causal-LM configs) load as shipped.
"""

from __future__ import annotations

import json
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Any

import yaml

from fedbrew.core.config import load_config
from fedbrew.core.evaluated_splits import splits_without_data
from fedbrew.core.refusal import RunRefused
from fedbrew.data.generate import generate_from_config

REPO = Path(__file__).resolve().parent.parent
BASE = REPO / "configs" / "dev" / "synthetic_manifest.yaml"


def _generate_without_val(root: Path) -> Path:
    """The synthetic_classification generator's layout: train and test, no val."""

    config = root / "generator.yaml"
    config.write_text(
        textwrap.dedent(f"""
            dataset:
              name: synthetic_classification
              output_dir: {root / "generated"}
              seed: 42
            synthetic:
              num_samples: 90
              input_dim: 5
              num_classes: 2
            partition:
              strategy: iid
              num_clients: 3
            splits:
              train_ratio: 0.8
              test_ratio: 0.2
            """).lstrip(),
        encoding="utf-8",
    )
    return generate_from_config(config)


def _run_config(root: Path, manifest: Path, evaluation: dict[str, Any] | None) -> Path:
    config = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    config["data"]["path"] = str(manifest)
    config["experiment"]["output_dir"] = str(root / "run")
    if evaluation is None:
        config.pop("evaluation", None)
    else:
        config["evaluation"] = evaluation
    path = root / "run.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _rewrite_records(manifest: Path, edit: Any) -> None:
    clients = manifest.parent / json.loads(manifest.read_text())["clients_file"]
    records = [json.loads(line) for line in clients.read_text().splitlines() if line.strip()]
    clients.write_text("".join(json.dumps(edit(dict(record))) + "\n" for record in records))


class ASplitTheDataDoesNotCarryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.manifest = _generate_without_val(self.root)

    def test_is_not_evaluated_when_the_config_leaves_it_to_the_default(self) -> None:
        config = load_config(_run_config(self.root, self.manifest, None))
        self.assertEqual(config.evaluation.val.every, "never")
        self.assertEqual(config.evaluation.splits_without_data, ("val",))
        # The splits the data carries keep their defaults.
        self.assertEqual(config.evaluation.test.every, 10)
        self.assertEqual(config.evaluation.train.every, 10)

    def test_the_plan_header_says_why(self) -> None:
        from fedbrew.core.logging import _metrics_rows

        config = load_config(_run_config(self.root, self.manifest, None))
        rows = {row.label: row.value for row in _metrics_rows(config, verbose=True)}
        self.assertEqual(rows["evaluation.val"], "not evaluated (no val data)")
        self.assertIn("every 10 rounds", rows["evaluation.test"])

    def test_is_refused_at_load_when_the_config_asks_for_it(self) -> None:
        path = _run_config(self.root, self.manifest, {"val": {"every": 5, "clients": "all"}})
        with self.assertRaisesRegex(
            RunRefused, r"evaluation\.val\.every is 5.*no client val split"
        ):
            load_config(path)

    def test_never_is_accepted(self) -> None:
        config = load_config(
            _run_config(self.root, self.manifest, {"val": {"every": "never", "clients": "all"}})
        )
        self.assertEqual(config.evaluation.val.every, "never")
        self.assertEqual(config.evaluation.splits_without_data, ("val",))

    def test_preflight_refuses_it(self) -> None:
        import argparse

        from fedbrew.core.runner import _run_preflight

        path = _run_config(self.root, self.manifest, {"val": {"every": 5, "clients": "all"}})
        args = argparse.Namespace(config=str(path), quiet=True, plain=True, verbose=False)
        for name in (
            "rounds",
            "seed",
            "output_dir",
            "local_iterations",
            "learning_rate",
            "batch_size",
            "participation_rate",
            "device",
        ):
            setattr(args, name, None)
        self.assertTrue(_run_preflight(args), "preflight reported no error")


class WhatARecordSaysTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.manifest = _generate_without_val(self.root)

    def _absent(self) -> list[str]:
        return splits_without_data(
            load_config(
                _run_config(
                    self.root,
                    self.manifest,
                    {
                        "val": {"every": "never", "clients": "all"},
                        "test": {"every": "never", "clients": "all"},
                    },
                )
            )
        )

    def test_an_unstated_count_is_the_total_less_the_other_two(self) -> None:
        # As the causal-LM generators write their records: no test count, a
        # total of train plus eval.
        def as_lm(record: dict[str, Any]) -> dict[str, Any]:
            record["num_eval_examples"] = 2
            record["num_examples"] = record["num_train_examples"] + 2
            del record["num_test_examples"]
            return record

        _rewrite_records(self.manifest, as_lm)
        self.assertEqual(self._absent(), ["test"])

    def test_an_unknown_count_leaves_the_split_as_configured(self) -> None:
        def unknown(record: dict[str, Any]) -> dict[str, Any]:
            del record["num_eval_examples"]
            del record["num_examples"]
            return record

        _rewrite_records(self.manifest, unknown)
        self.assertEqual(self._absent(), [])

    def test_one_client_holding_the_split_keeps_it(self) -> None:
        first = []

        def one_with_val(record: dict[str, Any]) -> dict[str, Any]:
            if not first:
                first.append(record["client_id"])
                record["num_eval_examples"] = 1
            return record

        _rewrite_records(self.manifest, one_with_val)
        self.assertEqual(self._absent(), [])


class TheShippedConfigsOnSuchDataTest(unittest.TestCase):
    """The configs whose data lacks a split load as shipped when the data is there."""

    def test_the_causal_lm_configs_do_not_ask_for_a_client_test_split(self) -> None:
        configs = [
            REPO / "configs" / "dev" / "tiny_causal_lm.yaml",
            *sorted((REPO / "configs" / "oasst1").glob("*.yaml")),
            *sorted((REPO / "configs" / "medmcqa").glob("*.yaml")),
        ]
        self.assertEqual(len(configs), 6)
        for path in configs:
            with self.subTest(config=path.name):
                evaluation = yaml.safe_load(path.read_text(encoding="utf-8"))["evaluation"]
                self.assertEqual(evaluation["test"]["every"], "never")

    def test_synthetic_manifest_loads_with_val_off(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _generate_without_val(root)
            config = yaml.safe_load(BASE.read_text(encoding="utf-8"))
            self.assertNotIn("evaluation", config, "the shipped config leaves val to the default")
            loaded = load_config(_run_config(root, manifest, None))
            self.assertEqual(loaded.evaluation.splits_without_data, ("val",))


if __name__ == "__main__":
    unittest.main()
