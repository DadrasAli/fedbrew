"""A number has to be provable against the shard set that produced it.

run.json recorded the manifest *path*. The generated data is not in the
repository and the same path holds a different dataset before and after a
regeneration, so a FEMNIST accuracy in a table could not be shown to have come
from a three-way per-writer split rather than from shards whose global_test.pt
was a copy of the eval slice the checkpoint was selected on. Those two answers
are the difference between a test number and the validation number under
another name, and the run artifact could not tell them apart. The LLM runs have
had corpus_hash for exactly this reason; the classification runs had nothing.

Two halves, both here because they are one claim:

- `build_dataset_provenance` records the manifest's sha256 and the keys that
  decide what the number means, with a null for a key the manifest does not
  declare -- a manifest predating `client_shard_format` is a split_v1 shard
  set, and that must not read the same as nobody having looked.
- The ten FEMNIST configs describe those semantics in a comment beside
  `evaluation.test`. That comment described split_v1 for as long as the
  generator has written split_v2. It is now diffed against a manifest the real
  generator produces, so it cannot say the wrong thing again.
"""

from __future__ import annotations

import glob
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import pytest

from fedbrew.core.config import load_config
from fedbrew.core.run_metadata import _DATASET_PROVENANCE_KEYS, build_dataset_provenance
from fedbrew.data.femnist import generate_femnist_from_config
from tests.test_femnist_support import _SPLITS, _fake_femnist_source, _generator_config

FEMNIST_CONFIGS = sorted(glob.glob("configs/femnist/*.yaml"))


def _generated_femnist(directory: Path) -> Path:
    """A real FEMNIST manifest from the real generator, over a fake corpus."""

    summary = generate_femnist_from_config(
        config=_generator_config(),
        output_dir=directory,
        seed=17,
        client_splits=_SPLITS,
        source_dataset=_fake_femnist_source(),
    )
    return Path(summary.manifest_path)


def _config_for(manifest_path: Path) -> Any:
    config = load_config("configs/dev/synthetic_manifest.yaml")
    config.data.path = str(manifest_path)
    return config


class ItRecordsWhatDecidesTheMeaningTest(unittest.TestCase):
    def test_a_generated_femnist_manifest_is_recorded_key_for_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = _generated_femnist(Path(directory) / "femnist")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            provenance = build_dataset_provenance(_config_for(manifest_path))

            assert provenance is not None
            for key in _DATASET_PROVENANCE_KEYS:
                with self.subTest(key=key):
                    self.assertEqual(provenance[key], manifest.get(key))

    def test_the_two_keys_the_finding_turned_on_are_not_null(self) -> None:
        """Without these a number cannot be placed on either side of the split."""

        with tempfile.TemporaryDirectory() as directory:
            manifest_path = _generated_femnist(Path(directory) / "femnist")
            provenance = build_dataset_provenance(_config_for(manifest_path))

            assert provenance is not None
            self.assertEqual(provenance["client_shard_format"], "split_v2")
            self.assertEqual(
                provenance["client_test_source"],
                "within_client_holdout_disjoint_from_eval",
            )

    def test_the_hash_is_the_manifest_the_run_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = _generated_femnist(Path(directory) / "femnist")
            provenance = build_dataset_provenance(_config_for(manifest_path))

            assert provenance is not None
            self.assertEqual(
                provenance["manifest_sha256"],
                hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            )

    def test_the_hash_changes_when_the_shard_format_does(self) -> None:
        """The anchor for every question this key list does not anticipate."""

        with tempfile.TemporaryDirectory() as directory:
            manifest_path = _generated_femnist(Path(directory) / "femnist")
            before = build_dataset_provenance(_config_for(manifest_path))

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["client_shard_format"] = "split_v1"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            after = build_dataset_provenance(_config_for(manifest_path))

            assert before is not None and after is not None
            self.assertNotEqual(before["manifest_sha256"], after["manifest_sha256"])
            self.assertEqual(after["client_shard_format"], "split_v1")

    def test_an_undeclared_key_is_null_rather_than_absent(self) -> None:
        """ "The manifest did not say" must not read as "nobody looked"."""

        with tempfile.TemporaryDirectory() as directory:
            manifest_path = _generated_femnist(Path(directory) / "femnist")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            del manifest["client_shard_format"]
            del manifest["client_test_source"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            provenance = build_dataset_provenance(_config_for(manifest_path))
            assert provenance is not None
            self.assertIn("client_shard_format", provenance)
            self.assertIsNone(provenance["client_shard_format"])
            self.assertIsNone(provenance["client_test_source"])


@pytest.mark.fast
class ItNeverLosesARunTest(unittest.TestCase):
    """Same discipline as capture_code_state: report the failure, do not raise."""

    def test_a_missing_manifest_is_an_error_field_not_an_exception(self) -> None:
        provenance = build_dataset_provenance(_config_for(Path("/nonexistent/manifest.json")))
        assert provenance is not None
        self.assertIn("manifest_error", provenance)
        self.assertNotIn("client_shard_format", provenance)

    def test_unreadable_json_is_an_error_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text("{not json", encoding="utf-8")
            provenance = build_dataset_provenance(_config_for(path))
            assert provenance is not None
            self.assertIn("manifest_error", provenance)

    def test_an_error_is_distinguishable_from_a_dataset_with_nothing_to_say(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text("{}", encoding="utf-8")
            provenance = build_dataset_provenance(_config_for(path))
            assert provenance is not None
            self.assertNotIn("manifest_error", provenance)
            self.assertIsNone(provenance["client_shard_format"])


@pytest.mark.fast
class OneProvenanceBlockPerRunTest(unittest.TestCase):
    def test_a_synthetic_dataset_has_no_manifest_to_record(self) -> None:
        self.assertIsNone(build_dataset_provenance(load_config("configs/dev/synthetic.yaml")))

    def test_a_causal_lm_run_records_its_manifest_under_llm_instead(self) -> None:
        """corpus_hash and the tokenizer revisions are already there."""

        config = load_config("configs/dev/tiny_causal_lm.yaml")
        self.assertEqual(config.task.name, "causal_lm")
        self.assertIsNone(build_dataset_provenance(config))


class ItReachesRunJsonTest(unittest.TestCase):
    """Computed is not recorded. code_state was read and never written once."""

    def _run(self, root: Path) -> dict[str, Any]:
        import yaml

        from fedbrew.core import runner
        from fedbrew.data.generate import generate_from_config

        source = Path("data/configs/synthetic_label_skew.yaml")
        generator = yaml.safe_load(source.read_text(encoding="utf-8"))
        generator["dataset"]["output_dir"] = str(root / "data")
        generator_path = root / "generator.yaml"
        generator_path.write_text(yaml.safe_dump(generator), encoding="utf-8")
        manifest_path = Path(generate_from_config(generator_path))

        config_path = root / "run.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "experiment": {"seed": 42, "output_dir": str(root / "run")},
                    "server": {
                        "strategy": "fedavg",
                        "participation_rate": 1,
                        "metrics": ["fit_loss"],
                    },
                    "client": {
                        "update_rule": "local_sgd",
                        "batch_size": 4,
                        "learning_rate": 0.05,
                        "learning_rate_schedule": "constant",
                        "min_learning_rate": 0.0,
                        "momentum": 0.0,
                        "weight_decay": 0.0,
                        "nesterov": False,
                        "metrics": ["fit_loss"],
                    },
                    "data": {"path": str(manifest_path)},
                    "model": {
                        "name": "mlp",
                        "input_dim": 6,
                        "hidden_dim": 4,
                        "num_classes": 3,
                    },
                    "runtime": {"device": "cpu", "use_amp": False, "deterministic": False},
                    "defaults": {"global_rounds": 1, "local_iterations": 1},
                }
            ),
            encoding="utf-8",
        )
        runner.run(config_path, runner.parse_args(["--quiet"]))
        written = sorted((root / "run").rglob("run.json"))
        self.assertEqual(len(written), 1)
        return json.loads(written[0].read_text(encoding="utf-8")), manifest_path

    def test_a_finished_run_records_the_shard_set_it_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run, manifest_path = self._run(Path(directory))
            dataset = run["reproducibility"]["dataset"]
            self.assertEqual(
                dataset["manifest_sha256"],
                hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(dataset["client_shard_format"], "split_v2")
            self.assertEqual(dataset["partition_strategy"], "label_skew")
            self.assertEqual(dataset["manifest_path"], str(manifest_path))


class TheFemnistConfigsDescribeTheGeneratorTest(unittest.TestCase):
    """The comment that was wrong, diffed against the generator's own output.

    It said the shards were split_v1 with no per-client test split, and that
    global_test.pt was the concatenation of every client's *eval* slice -- the
    same examples best.pt is selected on. The generator has written a three-way
    split and pooled the *test* slices since before these configs shipped.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        manifest_path = _generated_femnist(Path(cls._directory.name) / "femnist")
        cls.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def test_there_are_ten_of_them(self) -> None:
        self.assertEqual(len(FEMNIST_CONFIGS), 10, "the comment lives in every FEMNIST config")

    def test_each_states_the_shard_format_the_generator_writes(self) -> None:
        shard_format = self.manifest["client_shard_format"]
        for path in FEMNIST_CONFIGS:
            with self.subTest(path=path):
                text = Path(path).read_text(encoding="utf-8")
                self.assertIn(
                    f"client_shard_format {shard_format}",
                    text,
                    f"{path} does not say the shards are {shard_format}",
                )

    def test_each_states_where_the_pooled_test_set_comes_from(self) -> None:
        source = self.manifest["client_test_source"]
        for path in FEMNIST_CONFIGS:
            with self.subTest(path=path):
                text = Path(path).read_text(encoding="utf-8")
                self.assertIn(f"client_test_source: {source}", text)

    def test_none_of_them_still_calls_the_pool_the_eval_slice(self) -> None:
        """The specific false sentence, so it cannot come back by copy-paste."""

        for path in FEMNIST_CONFIGS:
            with self.subTest(path=path):
                text = Path(path).read_text(encoding="utf-8")
                self.assertNotIn("concatenation of every\n  # client's eval slice", text)
                self.assertNotIn("client's eval slice", text)


if __name__ == "__main__":
    unittest.main()
