"""A generator config names its extensions, and the manifest records them.

``dataset.extensions`` is the generator-side twin of ``experiment.extensions``:
loaded before ``dataset.name`` is looked up, so an out-of-tree generator is
dispatched exactly as a built-in is, through the same ``GeneratorSpec``, with
the sections it declared refused-if-undeclared like everyone else's. What it
generated is then described by the manifest -- so the manifest records which
extension wrote it, with the file's SHA-256, and the run that reads the
manifest copies that into run.json beside ``reference``, the generator's
statement of what a run on this data is scored against.

The probe generator here is an analytic problem's shape: two clients, every
split the same rows, ``client_test_source: identical_to_train``. Preflight
must say that a test number on such data is a training number.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import textwrap
import unittest
from pathlib import Path

import pytest
import yaml

from fedbrew.core import extensions, registry
from fedbrew.core.config import load_config
from fedbrew.core.run_metadata import build_dataset_provenance
from fedbrew.core.validation import run_checks
from fedbrew.data.generate import generate_from_config
from fedbrew.data.manifest_validation import IDENTICAL_TO_TRAIN, validate_manifest

GENERATOR_EXTENSION = textwrap.dedent(
    '''
    """A shards generator for an analytic problem: the data is the spec."""

    import csv
    import json
    from dataclasses import dataclass
    from pathlib import Path

    import torch

    from fedbrew.core import registry
    from fedbrew.data.manifest_validation import IDENTICAL_TO_TRAIN
    from fedbrew.data.writers.manifest import save_clients_jsonl, save_manifest
    from fedbrew.data.writers.torch_shards import save_client_shard, save_split_client_shard


    @dataclass(frozen=True)
    class Summary:
        manifest_path: Path
        num_clients: int
        num_examples: int
        num_test_examples: int


    def generate_probe_from_config(config, output_dir, seed, client_splits):
        del seed, client_splits
        rows = int(config["probe"]["rows"])
        clients = int(config["partition"]["num_clients"])
        output_dir = Path(output_dir)
        (output_dir / "shards").mkdir(parents=True, exist_ok=True)
        records = []
        every_x = []
        for index in range(clients):
            x = torch.full((rows, 2), float(index), dtype=torch.float64)
            y = torch.zeros(rows, dtype=torch.float64)
            shard = output_dir / "shards" / f"client_{{index}}.pt"
            save_split_client_shard(shard, x, y, x, y, x, y)
            every_x.append(x)
            records.append(
                {{
                    "client_id": f"client_{{index}}",
                    "shard": f"shards/client_{{index}}.pt",
                    "num_examples": 3 * rows,
                    "num_train_examples": rows,
                    "num_eval_examples": rows,
                    "num_test_examples": rows,
                }}
            )
        global_x = torch.cat(every_x)
        global_y = torch.zeros(len(global_x))
        save_client_shard(output_dir / "shards" / "global_test.pt", global_x, global_y)
        (output_dir / "partition_stats.json").write_text(
            json.dumps(
                {{"num_clients": clients, "total_examples": clients * rows, "clients": records}}
            ),
            encoding="utf-8",
        )
        with (output_dir / "client_stats.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["client_id", "num_examples"])
            for record in records:
                writer.writerow([record["client_id"], record["num_examples"]])
        manifest = {{
            "dataset_name": "{name}",
            "format": "torch_shards",
            "client_shard_format": "split_v2",
            "client_test_source": IDENTICAL_TO_TRAIN,
            "num_clients": clients,
            "clients_file": "clients.jsonl",
            "shards_dir": "shards",
            "global_test": "shards/global_test.pt",
            "partition_stats_file": "partition_stats.json",
            "client_stats_file": "client_stats.csv",
            "partition_strategy": "analytic",
            "client_splits": {{"train_ratio": 1.0, "eval_ratio": 1.0}},
            "reference": {{"x_star": [0.0, 0.0], "f_star": 0.0, "rows": rows}},
        }}
        manifest_path = save_manifest(output_dir, manifest)
        save_clients_jsonl(output_dir, records)
        return Summary(manifest_path, clients, clients * rows, clients * rows)


    def register():
        registry.generators.register(
            "{name}", generate_probe_from_config, sections={{"probe": {{"rows"}}}}
        )
    '''
)

#: The same generator, declaring its section by name alone -- the form the
#: built-ins use because their key lists live in generate.py. Out of tree
#: there is no such list, so this is the registration that used to leave a
#: misspelled key unchecked, and now is refused on the first generate.
NAME_ONLY_SECTION = 'sections={{"probe": {{"rows"}}}}'
GENERATOR_EXTENSION_NAME_ONLY = GENERATOR_EXTENSION.replace(
    NAME_ONLY_SECTION, 'sections={{"probe"}}'
)
assert GENERATOR_EXTENSION_NAME_ONLY != GENERATOR_EXTENSION


class GeneratorExtensionFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp())
        self.name = f"ext_probe_generator_{self.id().rsplit('.', 1)[-1]}"
        self.entry = str(self.directory / "problem.py")
        Path(self.entry).write_text(GENERATOR_EXTENSION.format(name=self.name), encoding="utf-8")
        self.addCleanup(extensions._loaded.pop, str(Path(self.entry).resolve()), None)
        self.addCleanup(registry.generators._items.pop, self.name, None)
        self.addCleanup(registry.generators._origins.pop, self.name, None)
        self.addCleanup(registry.generators._config_keys.pop, self.name, None)
        self.output = self.directory / "generated"

    def _generator_config(self, **sections: object) -> Path:
        raw = {
            "dataset": {
                "name": self.name,
                "output_dir": str(self.output),
                "seed": 1,
                "extensions": [self.entry],
            },
            "partition": {"strategy": "analytic", "num_clients": 2},
            # No client_splits: the ratios describe a cut, and an analytic
            # problem has none -- every split is the same rows. The parser
            # defaults to train 1.0 and the generator ignores it.
            "probe": {"rows": 3},
        }
        raw.update(sections)
        path = self.directory / "generator.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        return path

    def _manifest(self, manifest_path: Path) -> dict:
        return json.loads(manifest_path.read_text(encoding="utf-8"))


class GenerateTest(GeneratorExtensionFixture):
    def test_the_extension_generator_is_dispatched_like_a_builtin(self) -> None:
        manifest_path = generate_from_config(self._generator_config())

        self.assertEqual(manifest_path, self.output / "manifest.json")
        manifest = self._manifest(manifest_path)
        self.assertEqual(manifest["dataset_name"], self.name)
        self.assertEqual(manifest["num_clients"], 2)
        self.assertEqual(registry.generators.origin(self.name), self.entry)
        errors = [issue for issue in validate_manifest(manifest_path) if issue.severity == "error"]
        self.assertEqual(errors, [])

    def test_the_manifest_records_the_extension_and_its_hash(self) -> None:
        manifest = self._manifest(generate_from_config(self._generator_config()))
        self.assertEqual(
            manifest["extensions"],
            [
                {
                    "entry": self.entry,
                    "resolved": str(Path(self.entry).resolve()),
                    "sha256": hashlib.sha256(Path(self.entry).read_bytes()).hexdigest(),
                }
            ],
        )

    @pytest.mark.fast
    def test_an_undeclared_section_is_refused_for_an_extension_too(self) -> None:
        with self.assertRaises(ValueError) as caught:
            generate_from_config(self._generator_config(prob={"rows": 3}))
        message = str(caught.exception)
        self.assertIn("prob", message)
        self.assertIn("probe", message)

    @pytest.mark.fast
    def test_without_the_extension_the_name_is_unknown(self) -> None:
        path = self._generator_config(
            dataset={"name": self.name, "output_dir": str(self.output), "seed": 1}
        )
        with self.assertRaises(ValueError) as caught:
            generate_from_config(path)
        message = str(caught.exception)
        self.assertIn(f"unknown dataset.name={self.name!r}", message)
        self.assertIn("femnist", message)

    @pytest.mark.fast
    def test_a_misspelled_key_inside_the_declared_section_is_refused(self) -> None:
        """The case that found the gap: `condition_numbr:` generated the default.

        `problem.condition_numbr` in examples/drift-quad's generator config
        was dropped and a kappa = 100 dataset written, because the section was
        declared and its keys were not. Declaring the keys with the section
        is what closes it, and this is the same misspelling one level down in
        the probe.
        """

        with self.assertRaises(ValueError) as caught:
            generate_from_config(self._generator_config(probe={"rowz": 3}))
        message = str(caught.exception)
        self.assertIn("probe.rowz", message)
        self.assertIn("It reads: rows", message)
        self.assertFalse(self.output.exists(), "a refused config must generate nothing")

    @pytest.mark.fast
    def test_a_section_declared_by_name_alone_is_refused_at_generate(self) -> None:
        """Out of tree, a name-only section has no key list anywhere. Say so."""

        entry = self.directory / "name_only.py"
        name = f"{self.name}_name_only"
        entry.write_text(GENERATOR_EXTENSION_NAME_ONLY.format(name=name), encoding="utf-8")
        self.addCleanup(extensions._loaded.pop, str(entry.resolve()), None)
        self.addCleanup(registry.generators._items.pop, name, None)
        self.addCleanup(registry.generators._origins.pop, name, None)
        self.addCleanup(registry.generators._config_keys.pop, name, None)
        path = self._generator_config(
            dataset={
                "name": name,
                "output_dir": str(self.output),
                "seed": 1,
                "extensions": [str(entry)],
            }
        )

        with self.assertRaises(ValueError) as caught:
            generate_from_config(path)
        message = str(caught.exception)
        self.assertIn(f"generator {name!r} declares the section 'probe' but not the keys", message)
        self.assertIn("sections={'probe': {...}}", message)
        self.assertFalse(self.output.exists())

    @pytest.mark.fast
    def test_a_malformed_extensions_list_is_refused(self) -> None:
        path = self._generator_config(
            dataset={
                "name": self.name,
                "output_dir": str(self.output),
                "seed": 1,
                "extensions": self.entry,
            }
        )
        with self.assertRaises(ValueError) as caught:
            generate_from_config(path)
        self.assertIn("dataset.extensions must be a list", str(caught.exception))


class RunAgainstItTest(GeneratorExtensionFixture):
    def _run_config(self, manifest_path: Path) -> Path:
        raw = yaml.safe_load(Path("configs/dev/smoke.yaml").read_text(encoding="utf-8"))
        raw["data"] = {"path": str(manifest_path)}
        raw["model"] = {"name": "mlp", "input_dim": 2, "hidden_dim": 4, "num_classes": 2}
        raw["experiment"]["output_dir"] = str(self.directory / "out")
        path = self.directory / "run.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        return path

    def test_reference_and_extensions_reach_the_dataset_provenance(self) -> None:
        manifest_path = generate_from_config(self._generator_config())
        config = load_config(self._run_config(manifest_path))

        provenance = build_dataset_provenance(config)

        assert provenance is not None
        self.assertEqual(provenance["reference"], {"x_star": [0.0, 0.0], "f_star": 0.0, "rows": 3})
        self.assertEqual(provenance["extensions"][0]["entry"], self.entry)
        self.assertEqual(provenance["client_test_source"], IDENTICAL_TO_TRAIN)

    def test_preflight_says_the_test_numbers_are_training_numbers(self) -> None:
        manifest_path = generate_from_config(self._generator_config())
        config = load_config(self._run_config(manifest_path))

        issues = run_checks(config)

        notes = [issue for issue in issues if issue.code == "data.test_is_training_data"]
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0].severity, "info")
        self.assertIn(IDENTICAL_TO_TRAIN, notes[0].message)

    @pytest.mark.fast
    def test_a_dataset_with_held_out_data_gets_no_such_note(self) -> None:
        config = load_config("configs/dev/synthetic_manifest.yaml")
        issues = run_checks(config)
        self.assertEqual(
            [issue for issue in issues if issue.code == "data.test_is_training_data"], []
        )


if __name__ == "__main__":
    unittest.main()
