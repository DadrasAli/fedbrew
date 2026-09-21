"""`input_dtype` and `input_range` describe the shards, and are checked against them.

Both keys were written and read by nobody. `femnist.py` declared
`input_dtype: uint8` and `input_range: [0, 255]`; the four LLM generators
declare `input_dtype: int64`; `grep -rn "input_range\\|input_dtype"` over
`fedbrew/` found no consumer, and the shared writer declared neither, so the
difference the pair exists to record -- FEMNIST shards are `uint8` over
`[0, 255]`, MNIST shards are `float32` over `[0, 1]` -- was stated for one
dataset and not the other.

The audit's own failure mode has since gone: `input_scale` stopped being a
config key, and `femnist_resnet18` normalises internally on the `[0, 255]`
contract instead. That makes the manifest's claim *more* load-bearing, not
less -- shards regenerated in `[0, 1]` under the same path would train at
1/255 of the intended scale, with the manifest still saying `[0, 255]` and
nothing comparing the two.

The two claims have deliberately different strengths:

- `input_dtype` is **exact**. It is the half that catches the case above: a
  `[0, 1]` float shard set under a `uint8` declaration fails here, where the
  range alone would not, since `[0, 1]` sits inside `[0, 255]`.
- `input_range` is a **bound**. A shard whose values fall outside it is an
  error; one whose values are narrower is not, because a declared domain is not
  a promise that some client reached both ends of it.

The check costs one pass over the values of every shard, which
`validate_manifest` already loads -- 188 MB for a 100-client MNIST set
(60,000 x 784 float32).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from fedbrew.data.generate import _write_torch_shard_dataset
from fedbrew.data.manifest_validation import validate_manifest


def _errors(manifest_path: Path) -> list[str]:
    return [issue.code for issue in validate_manifest(manifest_path) if issue.severity == "error"]


class _GeneratedShardsTestCase(unittest.TestCase):
    def setUp(self) -> None:
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.directory = Path(holder.name)
        self.manifest_path = _write_torch_shard_dataset(
            output_dir=self.directory,
            dataset_name="mnist",
            train_x=torch.arange(12, dtype=torch.float32).reshape(12, 1) / 11.0,
            train_y=torch.tensor([0] * 6 + [1] * 6),
            test_x=torch.arange(8, dtype=torch.float32).reshape(8, 1) / 7.0,
            test_y=torch.tensor([0] * 4 + [1] * 4),
            partitions={"client_0": list(range(6)), "client_1": list(range(6, 12))},
            num_clients=2,
            metadata={"input_dim": 1, "num_classes": 2, "source": "test fixture"},
            partition_strategy="iid",
            client_splits={"train_ratio": 0.5, "eval_ratio": 0.5},
            seed=17,
        )

    def _manifest(self) -> dict:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def _rewrite(self, **overrides: object) -> None:
        manifest = self._manifest()
        manifest.update(overrides)
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


class TheGeneratorStatesThemTests(_GeneratedShardsTestCase):
    def test_the_claims_are_read_off_the_tensors_written(self) -> None:
        manifest = self._manifest()
        self.assertEqual(manifest["input_dtype"], "float32")
        self.assertEqual(manifest["input_range"], [0.0, 1.0])

    def test_a_dataset_it_wrote_validates_against_its_own_claims(self) -> None:
        self.assertEqual(_errors(self.manifest_path), [])


class TheClaimsAreCheckedTests(_GeneratedShardsTestCase):
    def test_a_wrong_dtype_is_an_error(self) -> None:
        """Exact, so the FEMNIST contract over float shards fails here."""

        self._rewrite(input_dtype="uint8", input_range=[0, 255])
        codes = _errors(self.manifest_path)
        self.assertTrue(codes)
        self.assertEqual(set(codes), {"manifest.input_dtype_mismatch"})

    def test_values_outside_the_declared_range_are_an_error(self) -> None:
        self._rewrite(input_range=[0.0, 0.25])
        self.assertEqual(set(_errors(self.manifest_path)), {"manifest.input_range_violated"})

    def test_a_wider_range_is_not(self) -> None:
        """A declared domain is not a promise that a client reached both ends
        of it, and every shard here is a slice of the pooled tensors."""

        self._rewrite(input_range=[-1.0, 255.0])
        self.assertEqual(_errors(self.manifest_path), [])

    def test_a_malformed_declaration_is_reported_once_rather_than_ignored(self) -> None:
        for value, code in (
            ("0 to 255", "manifest.input_range_invalid"),
            ([0.0], "manifest.input_range_invalid"),
            ([1.0, 0.0], "manifest.input_range_invalid"),
            ([None, 1.0], "manifest.input_range_invalid"),
        ):
            with self.subTest(input_range=value):
                self._rewrite(input_range=value)
                self.assertEqual(_errors(self.manifest_path), [code])

    def test_an_empty_dtype_declaration_is_reported(self) -> None:
        self._rewrite(input_dtype="   ")
        self.assertEqual(_errors(self.manifest_path), ["manifest.input_dtype_invalid"])

    def test_a_manifest_claiming_neither_is_still_valid(self) -> None:
        """Older shard sets declared no input keys; unclaimed is not wrong."""

        manifest = self._manifest()
        del manifest["input_dtype"]
        del manifest["input_range"]
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self.assertEqual(_errors(self.manifest_path), [])


class EveryShardIsHeldToThemTests(_GeneratedShardsTestCase):
    def test_one_stale_shard_is_enough_to_fail(self) -> None:
        """The case this compounds with: `generate` does not clear the
        directory it writes into, so a shard left by an earlier generation
        keeps its old units under the new manifest."""

        stale = self.directory / "shards" / "client_1.pt"
        shard = torch.load(stale, map_location="cpu", weights_only=False)
        shard["train"]["x"] = shard["train"]["x"] * 255.0
        torch.save(shard, stale)

        codes = _errors(self.manifest_path)
        self.assertIn("manifest.input_range_violated", codes)


if __name__ == "__main__":
    unittest.main()
