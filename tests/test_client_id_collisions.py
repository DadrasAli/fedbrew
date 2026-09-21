"""Two group names must never collapse into one client id.

_safe_client_id folds case and turns every run of non-alphanumerics into a
single underscore, so "Anatomy" and "anatomy", "Skin" and " skin ", and "A-B"
and "A B" all produce the same id. Nothing checked for that. The second shard
overwrote the first, clients.jsonl carried two rows with the same id, and
ManifestFederatedDataset -- which keys its lookup by client_id but lists every
row -- then handed the loop the surviving shard twice a round under two
separate FedAvg weights, while the client that lost its shard was never seen.

The writer, the checker and the reader each carried the defect, so each is
pinned here: generic_sft refuses to emit a colliding roster, validate_manifest
calls an existing duplicate an error rather than a warning, and the loader
refuses to serve one.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pytest
import torch

from fedbrew.data.generic_sft import FieldMapping, _client_ids, _safe_client_id
from fedbrew.data.manifest_dataset import ManifestFederatedDataset
from fedbrew.data.manifest_validation import validate_manifest


def _mapping(*, anonymize: bool = False) -> FieldMapping:
    return FieldMapping(
        prompt_template="{question}",
        response_template="{answer}",
        client_field="subject",
        group_field="subject",
        required_fields=("question", "answer", "subject"),
        choice=None,
        anonymize_client_ids=anonymize,
        system_prompt=None,
    )


def _write_dataset(root: Path, client_ids: list[str]) -> Path:
    """Write a manifest whose roster carries exactly the ids given."""

    shards = root / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    clients = []
    for index, client_id in enumerate(client_ids):
        payload = {
            "train": {
                "x": torch.full((4, 2), index, dtype=torch.long),
                "y": torch.full((4,), index, dtype=torch.long),
            },
            "test": {
                "x": torch.full((2, 2), index, dtype=torch.long),
                "y": torch.full((2,), index, dtype=torch.long),
            },
        }
        torch.save(payload, shards / f"{client_id}.pt")
        clients.append(
            {
                "client_id": client_id,
                "shard": f"shards/{client_id}.pt",
                "num_examples": 6,
                "num_train_examples": 4,
                "num_test_examples": 2,
            }
        )

    (root / "clients.jsonl").write_text(
        "\n".join(json.dumps(client) for client in clients), encoding="utf-8"
    )
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "clients_file": "clients.jsonl",
                "shards_dir": "shards",
                "num_clients": len(clients),
            }
        ),
        encoding="utf-8",
    )
    return manifest


@pytest.mark.fast
class SafeClientIdTest(unittest.TestCase):
    """The folding is genuinely lossy -- the collisions below are real."""

    def test_distinct_group_names_share_an_id(self) -> None:
        for first, second in (
            ("Anatomy", "anatomy"),
            ("Skin", " skin "),
            ("A-B", "A B"),
            ("Cell Biology", "cell__biology"),
        ):
            with self.subTest(first=first, second=second):
                self.assertNotEqual(first, second)
                self.assertEqual(_safe_client_id(first), _safe_client_id(second))


@pytest.mark.fast
class WriterTest(unittest.TestCase):
    """generic_sft must refuse the roster instead of overwriting a shard."""

    def test_a_collision_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            _client_ids(["Anatomy", "anatomy"], _mapping())
        message = str(caught.exception)
        # Both names and the id they share, so the config can be fixed.
        self.assertIn("Anatomy", message)
        self.assertIn("anatomy", message)
        self.assertIn("client_anatomy", message)

    def test_every_collision_is_reported_not_just_the_first(self) -> None:
        with self.assertRaises(ValueError) as caught:
            _client_ids(["Skin", "skin", "A-B", "A B"], _mapping())
        message = str(caught.exception)
        self.assertIn("2 client id collision", message)
        self.assertIn("client_skin", message)
        self.assertIn("client_a_b", message)

    def test_a_distinct_roster_maps_every_group_to_its_own_id(self) -> None:
        selected = ["anatomy", "skin", "cell biology"]
        ids = _client_ids(selected, _mapping())
        self.assertEqual(list(ids), selected)
        self.assertEqual(len(set(ids.values())), len(selected))
        self.assertEqual(ids["cell biology"], "client_cell_biology")

    def test_anonymized_ids_do_not_collide(self) -> None:
        """The hash is the escape hatch the error message points at."""

        ids = _client_ids(["Anatomy", "anatomy"], _mapping(anonymize=True))
        self.assertEqual(len(set(ids.values())), 2)


class ValidatorTest(unittest.TestCase):
    """A duplicate already on disk is an error, not a warning."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_a_duplicate_client_id_is_an_error(self) -> None:
        manifest = _write_dataset(self.root, ["client_a", "client_a", "client_b"])
        issues = validate_manifest(manifest)
        duplicates = [issue for issue in issues if issue.code == "manifest.client_id_duplicate"]
        self.assertEqual(len(duplicates), 1)
        self.assertEqual(duplicates[0].severity, "error")
        self.assertIn("client_a", duplicates[0].message)

    def test_a_distinct_roster_raises_no_duplicate_issue(self) -> None:
        manifest = _write_dataset(self.root, ["client_a", "client_b"])
        issues = validate_manifest(manifest)
        self.assertEqual(
            [issue for issue in issues if issue.code == "manifest.client_id_duplicate"],
            [],
        )


class LoaderTest(unittest.TestCase):
    """The loader must refuse a roster it cannot serve one-shard-per-client."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_a_duplicate_roster_is_refused_at_load(self) -> None:
        manifest = _write_dataset(self.root, ["client_a", "client_a", "client_b"])
        with self.assertRaises(ValueError) as caught:
            ManifestFederatedDataset(manifest)
        message = str(caught.exception)
        self.assertIn("client_a", message)
        self.assertNotIn("client_b", message)

    def test_a_distinct_roster_loads_and_lists_every_client(self) -> None:
        manifest = _write_dataset(self.root, ["client_a", "client_b", "client_c"])
        dataset = ManifestFederatedDataset(manifest)
        self.assertEqual(sorted(dataset.list_clients()), ["client_a", "client_b", "client_c"])


if __name__ == "__main__":
    unittest.main()
