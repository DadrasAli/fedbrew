"""Regenerating a dataset must not be able to leave a half-old directory.

Generators wrote straight into output_dir with mkdir(exist_ok=True), replaced
the shards one at a time, and saved manifest.json and clients.jsonl last. A
regeneration with a different seed, num_clients or split ratios that died
part-way left the previous run's manifest and clients.jsonl beside the new
run's shards, and every existing check passed: the files are all there and the
shards all have x and y. FedAvg then weighted each client by a count from the
old file.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import pytest
import yaml

from fedbrew.data import generate
from fedbrew.data.manifest_validation import validate_manifest

#: The smallest shipped generator config: 150 rows, 5 clients, no downloads.
SOURCE_CONFIG = Path("data/configs/synthetic_label_skew.yaml")


def _config(root: Path, name: str, num_clients: int) -> Path:
    config: dict[str, Any] = yaml.safe_load(SOURCE_CONFIG.read_text(encoding="utf-8"))
    config["dataset"]["output_dir"] = str(root / "dataset")
    config["partition"]["num_clients"] = num_clients
    path = root / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _snapshot(directory: Path) -> dict[str, int]:
    return {
        str(path.relative_to(directory)): path.stat().st_size
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


class AtomicReplacementTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.destination = self.root / "dataset"
        generate.generate_from_config(_config(self.root, "five", 5))

    def _manifest(self) -> dict[str, Any]:
        return json.loads((self.destination / "manifest.json").read_text(encoding="utf-8"))

    def test_a_successful_regeneration_leaves_no_stale_shards(self) -> None:
        self.assertEqual(self._manifest()["num_clients"], 5)
        generate.generate_from_config(_config(self.root, "three", 3))
        self.assertEqual(self._manifest()["num_clients"], 3)
        shards = sorted(p.name for p in (self.destination / "shards").iterdir())
        # The two extra client shards from the 5-client run must be gone, not
        # sitting beside a manifest that no longer lists them.
        self.assertEqual(len(shards), 4)
        self.assertNotIn("client_4.pt", shards)

    def test_a_generation_that_dies_leaves_the_previous_dataset_untouched(self) -> None:
        before = _snapshot(self.destination)
        original = generate.save_manifest

        def _die(*args: Any, **kwargs: Any) -> Path:
            raise RuntimeError("time limit")

        generate.save_manifest = _die
        try:
            with self.assertRaisesRegex(RuntimeError, "time limit"):
                generate.generate_from_config(_config(self.root, "three", 3))
        finally:
            generate.save_manifest = original

        self.assertEqual(_snapshot(self.destination), before)
        self.assertEqual(self._manifest()["num_clients"], 5)

    def test_a_failed_generation_cleans_up_after_itself(self) -> None:
        original = generate.save_manifest
        generate.save_manifest = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("time limit"))
        try:
            with self.assertRaises(RuntimeError):
                generate.generate_from_config(_config(self.root, "three", 3))
        finally:
            generate.save_manifest = original
        leftovers = [p.name for p in self.root.iterdir() if "incomplete" in p.name]
        self.assertEqual(leftovers, [])

    def test_the_staging_directory_is_a_sibling_not_a_child(self) -> None:
        # A staging directory inside the destination would be swept into the
        # dataset, and rglob over it would find two of every shard.
        seen: list[Path] = []
        with generate._staged_output(self.destination) as staging:
            seen.append(staging)
            staging.mkdir(parents=True)
        self.assertEqual(seen[0].parent, self.destination.parent)
        self.assertNotEqual(seen[0], self.destination)

    def test_the_returned_manifest_path_is_in_the_destination(self) -> None:
        path = generate.generate_from_config(_config(self.root, "three", 3))
        self.assertEqual(Path(path), self.destination / "manifest.json")
        self.assertTrue(Path(path).is_file())


class DeclaredCountValidationTests(unittest.TestCase):
    """The check that sees a mixed directory that predates the fix."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.destination = self.root / "dataset"
        generate.generate_from_config(_config(self.root, "five", 5))

    def _errors(self) -> list[str]:
        return [
            issue.message
            for issue in validate_manifest(self.destination / "manifest.json")
            if issue.severity == "error"
        ]

    def test_a_consistent_dataset_reports_nothing(self) -> None:
        self.assertEqual(self._errors(), [])

    def test_a_stale_count_is_an_error(self) -> None:
        path = self.destination / "clients.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        record = json.loads(lines[0])
        record["num_train_examples"] += 7
        lines[0] = json.dumps(record)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        errors = self._errors()
        self.assertTrue(errors)
        self.assertTrue(any("num_train_examples" in message for message in errors), errors)
        # Name both numbers, so the reader does not have to load the shard.
        self.assertTrue(any("rows" in message for message in errors), errors)

    def test_every_split_count_is_checked(self) -> None:
        for key in ("num_train_examples", "num_eval_examples", "num_test_examples"):
            with self.subTest(key=key):
                path = self.destination / "clients.jsonl"
                lines = path.read_text(encoding="utf-8").splitlines()
                record = json.loads(lines[0])
                if key not in record:
                    self.skipTest(f"this generator does not declare {key}")
                original = record[key]
                record[key] = original + 5
                lines[0] = json.dumps(record)
                path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                try:
                    self.assertTrue(any(key in message for message in self._errors()))
                finally:
                    record[key] = original
                    lines[0] = json.dumps(record)
                    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class NumExamplesDefinitionTests(unittest.TestCase):
    """num_examples is every split, and one definition for every writer.

    generate.py wrote train + eval, leaving out the official test rows it had
    just partitioned to the client; femnist.py wrote all three. A reader could
    not tell which convention a record followed, and the one consumer -- the
    "no training batches" message in torch_sgd_client -- printed the total
    under the label "train split".
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.destination = self.root / "dataset"
        generate.generate_from_config(_config(self.root, "five", 5))
        self.records = [
            json.loads(line)
            for line in (self.destination / "clients.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]

    def test_the_total_is_the_sum_of_every_split(self) -> None:
        for record in self.records:
            with self.subTest(client=record["client_id"]):
                self.assertEqual(
                    record["num_examples"],
                    record["num_train_examples"]
                    + record["num_eval_examples"]
                    + record["num_test_examples"],
                )

    def test_the_total_counts_the_official_test_rows(self) -> None:
        # The defect: it used to be train + eval, so a client with any test
        # rows declared fewer examples than it holds.
        self.assertTrue(any(r["num_test_examples"] > 0 for r in self.records))
        for record in self.records:
            with self.subTest(client=record["client_id"]):
                self.assertGreater(
                    record["num_examples"],
                    record["num_train_examples"] + record["num_eval_examples"],
                )

    def test_a_total_that_disagrees_with_the_splits_is_an_error(self) -> None:
        path = self.destination / "clients.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        record = json.loads(lines[0])
        # Exactly the old convention: the total without the test rows.
        record["num_examples"] = record["num_train_examples"] + record["num_eval_examples"]
        lines[0] = json.dumps(record)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        errors = [
            issue
            for issue in validate_manifest(self.destination / "manifest.json")
            if issue.severity == "error"
        ]
        self.assertTrue(
            any(issue.code == "manifest.client_num_examples_inconsistent" for issue in errors),
            errors,
        )

    def test_the_manifest_dataset_reader_agrees(self) -> None:
        from fedbrew.data.manifest_dataset import ManifestFederatedDataset

        dataset = ManifestFederatedDataset(str(self.destination / "manifest.json"))
        for record in self.records:
            with self.subTest(client=record["client_id"]):
                metadata = dataset.get_client_metadata(record["client_id"])
                self.assertEqual(metadata["num_examples"], record["num_examples"])
                self.assertEqual(
                    metadata["num_examples"],
                    metadata["num_train_examples"]
                    + metadata["num_eval_examples"]
                    + metadata["num_test_examples"],
                )

    def test_the_in_memory_synthetic_dataset_agrees(self) -> None:
        from fedbrew.data.synthetic_classification import (
            SyntheticClassificationDataset,
        )

        dataset = SyntheticClassificationDataset(num_clients=2, samples_per_client=7)
        for source in (dataset.get_client_metadata, dataset.get_client_data):
            with self.subTest(source=source.__name__):
                record = source("client_0")
                self.assertEqual(
                    record["num_examples"],
                    record["num_train_examples"]
                    + record["num_eval_examples"]
                    + record["num_test_examples"],
                )


@pytest.mark.fast
class TrainingBatchMessageTests(unittest.TestCase):
    def test_the_message_names_the_train_split_not_the_total(self) -> None:
        from fedbrew.clients.torch_sgd_client import (
            _infer_num_examples,
            _infer_train_num_examples,
        )

        client_data = {
            "train": {"y": list(range(6))},
            "eval": {"y": list(range(2))},
            "test": {"y": list(range(4))},
            "num_examples": 12,
            "num_train_examples": 6,
        }
        self.assertEqual(_infer_train_num_examples(client_data), 6)
        self.assertEqual(_infer_num_examples(client_data), 12)

    def test_the_total_falls_back_to_every_split(self) -> None:
        from fedbrew.clients.torch_sgd_client import _infer_num_examples

        self.assertEqual(
            _infer_num_examples(
                {
                    "train": {"y": list(range(6))},
                    "eval": {"y": list(range(2))},
                    "test": {"y": list(range(4))},
                }
            ),
            12,
        )


if __name__ == "__main__":
    unittest.main()
