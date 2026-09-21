"""The manifest records what produced the partition, not only its shape.

`manifest.json` carried `partition_strategy` and `num_clients` and nothing else
about the cut. Verified on the pre-fix tree with `data/configs/
synthetic_label_skew.yaml` at `labels_per_client` 1 and 2 -- two partitions with
no client in common, `client_0` holding 45 examples of one label against 16 and
7 of two others -- whose manifests were **byte-identical**. `dataset.seed`, the
seed every random choice in the partition comes from, appeared in neither
`manifest.json` nor `partition_stats.json`.

That is what makes a regeneration invisible. The generated data is not in the
repository and `generate` does not clear the directory it writes into (finding
7 of the same report), so a dataset rebuilt under the same path with a different
alpha is a different dataset that nothing downstream can distinguish -- and
`run.json`'s dataset provenance is copied from the manifest, so a run recorded
the same provenance either way.

`partition_parameters` is read from `_PARTITION_PARAMETERS`, the mapping the
rail line already uses, so it names the knobs the configured strategy actually
reads and no others: a config carrying `alpha` under `label_skew` records no
alpha, because none was applied.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pytest
import yaml

from fedbrew.core.run_metadata import _DATASET_PROVENANCE_KEYS
from fedbrew.data.generate import _PARTITION_PARAMETERS, generate_from_config

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG = REPO_ROOT / "data" / "configs" / "synthetic_label_skew.yaml"


class _GeneratesTestCase(unittest.TestCase):
    def setUp(self) -> None:
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.root = Path(holder.name)
        self.base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))

    def _generate(self, name: str, *, partition: dict, seed: int | None = None) -> Path:
        document = json.loads(json.dumps(self.base))
        document["partition"] = partition
        document["dataset"]["output_dir"] = str(self.root / name)
        if seed is not None:
            document["dataset"]["seed"] = seed
        config_path = self.root / f"{name}.yaml"
        config_path.write_text(yaml.safe_dump(document), encoding="utf-8")
        return Path(generate_from_config(config_path))

    def _manifest(self, manifest_path: Path) -> dict:
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def _stats(self, manifest_path: Path) -> dict:
        return json.loads(
            (manifest_path.parent / "partition_stats.json").read_text(encoding="utf-8")
        )


class TwoPartitionsAreDistinguishableTests(_GeneratesTestCase):
    def test_a_different_labels_per_client_gives_a_different_manifest(self) -> None:
        one = self._manifest(
            self._generate(
                "lpc1",
                partition={"strategy": "label_skew", "num_clients": 5, "labels_per_client": 1},
            )
        )
        two = self._manifest(
            self._generate(
                "lpc2",
                partition={"strategy": "label_skew", "num_clients": 5, "labels_per_client": 2},
            )
        )

        self.assertNotEqual(one, two)
        self.assertEqual(one["partition_parameters"], {"labels_per_client": 1})
        self.assertEqual(two["partition_parameters"], {"labels_per_client": 2})

    def test_a_different_seed_gives_a_different_manifest(self) -> None:
        partition = {"strategy": "label_skew", "num_clients": 5, "labels_per_client": 2}
        first = self._manifest(self._generate("seed7", partition=partition, seed=7))
        second = self._manifest(self._generate("seed8", partition=partition, seed=8))

        self.assertNotEqual(first, second)
        self.assertEqual(first["seed"], 7)
        self.assertEqual(second["seed"], 8)

    def test_partition_stats_records_the_same_two(self) -> None:
        """It is the file a reader opens to ask how the partition landed, and
        it could not say what was asked for."""

        stats = self._stats(
            self._generate(
                "stats",
                partition={"strategy": "dirichlet", "num_clients": 5, "alpha": 0.25},
                seed=11,
            )
        )
        self.assertEqual(stats["partition_parameters"], {"alpha": 0.25})
        self.assertEqual(stats["seed"], 11)


class OnlyTheKnobsTheStrategyReadsTests(_GeneratesTestCase):
    def test_a_knob_belonging_to_another_strategy_is_not_recorded(self) -> None:
        """`alpha` under `label_skew` did nothing; recording it would say it
        had."""

        manifest = self._manifest(
            self._generate(
                "mixed",
                partition={
                    "strategy": "label_skew",
                    "num_clients": 5,
                    "labels_per_client": 2,
                    "alpha": 0.5,
                },
            )
        )
        self.assertEqual(manifest["partition_parameters"], {"labels_per_client": 2})

    def test_quantity_skew_records_its_three(self) -> None:
        manifest = self._manifest(
            self._generate(
                "qskew",
                partition={
                    "strategy": "quantity_skew",
                    "num_clients": 5,
                    "min_size": 10,
                    "max_size": 60,
                    "sigma": 0.5,
                },
            )
        )
        self.assertEqual(
            manifest["partition_parameters"],
            {"sigma": 0.5, "min_size": 10, "max_size": 60},
        )

    def test_iid_records_an_empty_mapping_rather_than_omitting_the_key(self) -> None:
        """A strategy with no knobs and a record nobody wrote have to be
        distinguishable, which is why the key is always present."""

        manifest = self._manifest(
            self._generate("iid", partition={"strategy": "iid", "num_clients": 5})
        )
        self.assertIn("partition_parameters", manifest)
        self.assertEqual(manifest["partition_parameters"], {})

    def test_the_recorded_names_come_from_the_dispatch_table(self) -> None:
        """Not a second hand-kept list. Widening `_PARTITION_PARAMETERS` --
        the mapping the rail line already reads -- widens the record, so a
        strategy that gains a knob there records it without anyone
        remembering to."""

        partition = {
            "strategy": "label_skew",
            "num_clients": 5,
            "labels_per_client": 2,
            "alpha": 0.5,
        }
        with mock.patch.dict(
            _PARTITION_PARAMETERS,
            {"label_skew": ("labels_per_client", "alpha")},
        ):
            widened = self._manifest(self._generate("widened", partition=partition))
        self.assertEqual(widened["partition_parameters"], {"labels_per_client": 2, "alpha": 0.5})


@pytest.mark.fast
class RunProvenanceCarriesThemTests(unittest.TestCase):
    def test_both_keys_are_copied_into_run_json(self) -> None:
        """The manifest is not in the repository; run.json is the only record
        a result table can be traced through."""

        self.assertIn("partition_parameters", _DATASET_PROVENANCE_KEYS)
        self.assertIn("seed", _DATASET_PROVENANCE_KEYS)


if __name__ == "__main__":
    unittest.main()
