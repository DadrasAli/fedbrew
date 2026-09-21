"""A staged dataset is the one that was asked for, and is whole when it is read.

`data_staging` keyed the scratch destination by the source directory's
basename, so two datasets whose directories share one -- say
`$FL_DATA_ROOT/oasst1_qwen05b_4clients` and
`data/generated/oasst1_qwen05b_4clients` -- staged into the same place. And
`copytree(dirs_exist_ok=True)` overwrites what it matches and removes nothing
it does not, so the shared directory ended up holding the second manifest
beside both sets of shards.

The copy was also not atomic: it wrote into the destination as it went, and
the only completeness check was that the manifest had appeared there
afterwards -- a file `copytree` may write long before the shards beside it. A
second run packed on the same node could read a tree still being written.

Not enabled by any shipped config, so nothing published was affected.
P10-F25.

What is *not* fixed here, because it is not a defect: nothing deletes the
staged copy. Chapter 11 §5 says so deliberately -- node-local cleanup is the
cluster's job -- and this guard pins that decision rather than changing it.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

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
)
from fedbrew.core.data_staging import (
    STAGING_MARKER,
    maybe_stage_manifest_dataset,
    staged_directory_name,
)

pytestmark = pytest.mark.fast


def _config(data_path: str, local_root: str) -> FullConfig:
    return FullConfig(
        experiment=ExperimentConfig(seed=0, output_dir="outputs/test"),
        server=ServerConfig(strategy="fedavg", global_rounds=1, participation_rate=1.0, metrics=[]),
        client=ClientConfig(update_rule="fedavg", local_iterations=1, batch_size=1, metrics=[]),
        task=TaskConfig(name="classification"),
        data=DataConfig(name="manifest_dataset", path=data_path),
        model=ModelConfig(name="mlp"),
        runtime=RuntimeConfig(
            device="cpu",
            use_amp=False,
            extra={"data_staging": {"enabled": True, "local_root": local_root}},
        ),
    )


def _dataset(root: Path, tag: str, *, name: str = "shared_basename") -> Path:
    directory = root / name
    (directory / "shards").mkdir(parents=True)
    (directory / "manifest.json").write_text(json.dumps({"dataset_name": tag}), encoding="utf-8")
    (directory / "shards" / f"{tag}.pt").write_text(tag, encoding="utf-8")
    return directory / "manifest.json"


def _stage(manifest: Path, scratch: Path) -> Path:
    staged = maybe_stage_manifest_dataset(_config(str(manifest), str(scratch)))
    return Path(staged.data.path)


class TwoSourcesOneBasenameTest(unittest.TestCase):
    def test_they_stage_to_different_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = _dataset(root / "siteA", "dataset_A")
            second = _dataset(root / "siteB", "dataset_B")
            scratch = root / "scratch"

            staged_first = _stage(first, scratch)
            staged_second = _stage(second, scratch)

            self.assertNotEqual(staged_first.parent, staged_second.parent)
            for staged, tag in ((staged_first, "dataset_A"), (staged_second, "dataset_B")):
                with self.subTest(dataset=tag):
                    self.assertEqual(json.loads(staged.read_text())["dataset_name"], tag)
                    self.assertEqual(
                        sorted(path.name for path in (staged.parent / "shards").iterdir()),
                        [f"{tag}.pt"],
                        "the staged tree holds another dataset's shards",
                    )

    def test_the_name_keeps_the_basename_and_adds_the_source(self) -> None:
        """Legible on the node, and unambiguous."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = (root / "siteA" / "shared_basename").resolve()
            second = (root / "siteB" / "shared_basename").resolve()
            for source in (first, second):
                self.assertTrue(staged_directory_name(source).startswith("shared_basename-"))
            self.assertNotEqual(staged_directory_name(first), staged_directory_name(second))

    def test_the_same_source_always_gets_the_same_name(self) -> None:
        """Two runs of one dataset share the copy; that is what staging is for."""

        source = Path("/data/generated/femnist_natural")
        self.assertEqual(staged_directory_name(source), staged_directory_name(source))


class TheCopyIsWholeOrAbsentTest(unittest.TestCase):
    def test_a_finished_tree_is_reused_rather_than_recopied(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _dataset(root / "src", "A", name="ds")
            scratch = root / "scratch"

            staged = _stage(manifest, scratch)
            marker = staged.parent / STAGING_MARKER
            before = marker.stat().st_mtime_ns

            self.assertEqual(_stage(manifest, scratch), staged)
            self.assertEqual(marker.stat().st_mtime_ns, before, "the warm copy was rewritten")

    def test_a_truncated_tree_is_not_mistaken_for_a_finished_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _dataset(root / "src", "A", name="ds")
            scratch = root / "scratch"

            staged = _stage(manifest, scratch)
            (staged.parent / "shards" / "A.pt").unlink()

            restaged = _stage(manifest, scratch)
            self.assertEqual(restaged, staged)
            self.assertTrue((staged.parent / "shards" / "A.pt").exists())

    def test_a_tree_with_no_marker_is_replaced(self) -> None:
        """Debris from a copy that died, or from before the digest existed."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _dataset(root / "src", "A", name="ds")
            scratch = root / "scratch"

            staged = _stage(manifest, scratch)
            (staged.parent / STAGING_MARKER).unlink()
            (staged.parent / "left_over.pt").write_text("stale", encoding="utf-8")

            restaged = _stage(manifest, scratch)
            self.assertEqual(restaged, staged)
            self.assertTrue((staged.parent / STAGING_MARKER).exists())
            self.assertFalse(
                (staged.parent / "left_over.pt").exists(),
                "the replacement merged into the debris instead of replacing it",
            )

    def test_a_marker_naming_another_source_is_not_trusted(self) -> None:
        """Belt as well as braces: the name carries the digest, and so does
        the marker. One scratch root reached under two mount paths, or a tree
        copied by hand, is a directory whose contents are not what its name
        says -- and re-copying is cheap against reading the wrong dataset."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _dataset(root / "src", "A", name="ds")
            scratch = root / "scratch"

            staged = _stage(manifest, scratch)
            marker_path = staged.parent / STAGING_MARKER
            marker = json.loads(marker_path.read_text())
            # Only the source, so the file count still checks out and this
            # isolates the source comparison rather than the count.
            marker["source"] = "/somewhere/else/ds"
            marker_path.write_text(json.dumps(marker), encoding="utf-8")

            restaged = _stage(manifest, scratch)
            self.assertEqual(restaged, staged)
            self.assertEqual(
                json.loads(marker_path.read_text())["source"],
                str(manifest.parent),
                "a marker naming another source was accepted as this one's copy",
            )

    def test_no_partial_directory_survives_a_successful_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _dataset(root / "src", "A", name="ds")
            scratch = root / "scratch"
            _stage(manifest, scratch)
            self.assertEqual(
                [path.name for path in scratch.iterdir() if path.name.endswith(".partial")],
                [],
            )

    def test_the_marker_names_the_source_it_copied(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _dataset(root / "src", "A", name="ds")
            scratch = root / "scratch"
            staged = _stage(manifest, scratch)
            marker = json.loads((staged.parent / STAGING_MARKER).read_text())
            self.assertEqual(marker["source"], str(manifest.parent))
            self.assertEqual(
                marker["files"], sum(1 for p in staged.parent.rglob("*") if p.is_file())
            )


class NothingDeletesTheStagedCopyTest(unittest.TestCase):
    """Pinned, not fixed: chapter 11 §5 states this as a deliberate decision."""

    def test_the_staged_tree_outlives_the_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _dataset(root / "src", "A", name="ds")
            scratch = root / "scratch"
            staged = _stage(manifest, scratch)
            self.assertTrue(staged.exists())

    def test_the_chapter_says_so(self) -> None:
        chapter = Path("docs/11-performance-and-cost.md").read_text(encoding="utf-8")
        self.assertIn("The runtime\ndoes not delete staged data", chapter)


if __name__ == "__main__":
    unittest.main()
