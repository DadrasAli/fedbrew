"""MNIST records which reader produced its tensors, and verifies what it read.

`generate_mnist_tensors` caught `Exception` around torchvision and switched to
downloading the four idx archives from a Google Cloud Storage mirror, with no
checksum, while `_prepare_mnist_tensors` declared `source:
torchvision.datasets.MNIST` either way. `source` is a `_DATASET_PROVENANCE_KEYS`
entry, so that label is copied into the `run.json` of every run trained on the
data -- the mislabelling outlived the generation it happened in.

Three separate things were wrong and each has its own test here: the catch was
wide enough to swallow a torchvision API change, the mirror was trusted
unverified where torchvision would have checked an MD5, and the recorded source
named the reader that was tried rather than the one that ran.

The digests are torchvision's own, so the last test pins the two tables equal.
They were also checked against real bytes when they were written: the four
archives torchvision had already fetched into `data/raw/datasets/mnist` matched,
and so did the mirror's `t10k-labels-idx1-ubyte.gz` -- a check the fix needed,
since a digest the mirror could not satisfy would have turned an unverified
fallback into a guaranteed-broken one.
"""

from __future__ import annotations

import gzip
import hashlib
import struct
import tempfile
import unittest
import warnings
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import torch

from fedbrew.data import generate

pytestmark = pytest.mark.fast

IMAGE_FILES = ("train-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte.gz")
LABEL_FILES = ("train-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte.gz")


def _write_images(path: Path, count: int) -> None:
    payload = struct.pack(">IIII", 2051, count, 28, 28) + bytes(count * 28 * 28)
    path.write_bytes(gzip.compress(payload))


def _write_labels(path: Path, count: int) -> None:
    payload = struct.pack(">II", 2049, count) + bytes(count)
    path.write_bytes(gzip.compress(payload))


class MnistSourceProvenanceTests(unittest.TestCase):
    """What the manifest says about MNIST has to be what happened."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.raw_dir = Path(self._tmp.name) / "mnist"
        self.raw_dir.mkdir(parents=True)
        for name in IMAGE_FILES:
            _write_images(self.raw_dir / name, 4)
        for name in LABEL_FILES:
            _write_labels(self.raw_dir / name, 4)
        # The archives are stand-ins, so the table they are checked against is
        # the fixture's own digests. Patching it here is also what proves the
        # check reads that table rather than trusting the file name.
        self._digests = mock.patch.dict(
            generate._MNIST_ARCHIVE_MD5,
            {
                name: hashlib.md5((self.raw_dir / name).read_bytes()).hexdigest()
                for name in IMAGE_FILES + LABEL_FILES
            },
        )
        self._digests.start()
        self.addCleanup(self._digests.stop)
        self.config = {"dataset": {"raw_dir": str(self.raw_dir)}, "mnist": {}}

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _torchvision_returns(self) -> Any:
        images = torch.zeros((4, 28, 28), dtype=torch.uint8)
        labels = torch.zeros((4,), dtype=torch.long)
        return mock.patch.object(
            generate,
            "_load_mnist_torchvision",
            return_value=(images, labels, images, labels),
        )

    def test_a_torchvision_read_is_recorded_as_torchvision(self) -> None:
        with self._torchvision_returns(), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            *_, metadata = generate.generate_mnist_tensors(self.config, seed=0)
        self.assertEqual(metadata["source"], "torchvision.datasets.MNIST")
        self.assertEqual([str(entry.message) for entry in caught], [])

    def test_a_mirror_read_is_recorded_as_the_mirror_and_said_out_loud(self) -> None:
        failed = mock.patch.object(
            generate,
            "_load_mnist_torchvision",
            side_effect=RuntimeError("Error downloading train-images-idx3-ubyte.gz"),
        )
        with failed, self.assertWarns(UserWarning) as warned:
            *_, metadata = generate.generate_mnist_tensors(self.config, seed=0)
        self.assertEqual(metadata["source"], "https://storage.googleapis.com/cvdf-datasets/mnist")
        self.assertIn("Error downloading", str(warned.warning))

    def test_a_failure_that_is_not_a_download_failure_is_not_rerouted(self) -> None:
        # .data or .targets moving is a bug the mirror cannot stand in for: it
        # used to be caught, and produced a dataset that looked fine.
        broken = mock.patch.object(
            generate,
            "_load_mnist_torchvision",
            side_effect=AttributeError("'MNIST' object has no attribute 'targets'"),
        )
        mirror = mock.patch.object(generate, "_load_mnist_idx_files")
        with broken, mirror as never_read, self.assertRaises(AttributeError):
            generate.generate_mnist_tensors(self.config, seed=0)
        never_read.assert_not_called()

    def test_an_archive_that_is_not_the_one_it_is_named_after_is_refused(self) -> None:
        corrupt = self.raw_dir / "t10k-labels-idx1-ubyte.gz"
        corrupt.write_bytes(gzip.compress(struct.pack(">II", 2049, 3) + bytes(3)))
        failed = mock.patch.object(
            generate,
            "_load_mnist_torchvision",
            side_effect=RuntimeError("Error downloading t10k-labels-idx1-ubyte.gz"),
        )
        # A file already on disk is never re-fetched, so a corrupt one must not
        # reach the network to be refused.
        offline = mock.patch.object(
            generate.urllib.request,
            "urlopen",
            side_effect=AssertionError("a file already on disk was re-downloaded"),
        )
        with failed, offline, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaises(ValueError) as raised:
                generate.generate_mnist_tensors(self.config, seed=0)
        self.assertIn("t10k-labels-idx1-ubyte.gz", str(raised.exception))

    def test_the_declared_digests_are_the_ones_torchvision_enforces(self) -> None:
        self._digests.stop()
        self.addCleanup(self._digests.start)
        try:
            from torchvision.datasets import MNIST
        except ImportError:  # pragma: no cover - torchvision is an extra
            self.skipTest("torchvision is not installed")
        self.assertEqual(generate._MNIST_ARCHIVE_MD5, dict(MNIST.resources))


if __name__ == "__main__":
    unittest.main()
