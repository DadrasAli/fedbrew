"""Tests for selective runtime-artifact cleanup."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pytest

from fedbrew.cli.cleanup import clean_runtime_artifacts

pytestmark = pytest.mark.fast


class CleanupRuntimeArtifactsTests(unittest.TestCase):
    def test_excluded_output_directory_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kept_file = root / "outputs" / "keep" / "summary.json"
            removed_file = root / "outputs" / "discard" / "summary.json"
            generated_file = root / "data" / "generated" / "manifest.json"
            for path in (kept_file, removed_file, generated_file):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("artifact", encoding="utf-8")

            clean_runtime_artifacts(str(root), excludes=["outputs/keep"])

            self.assertTrue(kept_file.exists())
            self.assertFalse(removed_file.exists())
            self.assertFalse(generated_file.exists())

    def test_entire_outputs_directory_can_be_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_file = root / "outputs" / "run" / "summary.json"
            generated_file = root / "data" / "generated" / "manifest.json"
            for path in (output_file, generated_file):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("artifact", encoding="utf-8")

            clean_runtime_artifacts(str(root), excludes=["outputs"])

            self.assertTrue(output_file.exists())
            self.assertFalse(generated_file.exists())

    def test_excluded_file_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kept_file = root / "outputs" / "keep.json"
            removed_file = root / "outputs" / "discard.json"
            for path in (kept_file, removed_file):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("artifact", encoding="utf-8")

            clean_runtime_artifacts(str(root), excludes=["outputs/keep.json"])

            self.assertTrue(kept_file.exists())
            self.assertFalse(removed_file.exists())

    def test_outside_project_exclusion_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root.parent / "outside"

            with self.assertRaises(ValueError):
                clean_runtime_artifacts(str(root), excludes=[str(outside)])


if __name__ == "__main__":
    unittest.main()
