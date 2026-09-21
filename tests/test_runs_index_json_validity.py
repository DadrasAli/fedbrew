"""runs_index.jsonl has to be valid JSON, including for a diverged run.

A diverged run's final_metrics and termination.value are inf or nan.
json.dumps defaults to allow_nan=True and writes the bare tokens NaN and
Infinity, which RFC 8259 does not define: Python's json and pandas.read_json
read them back and jq 1.6 rewrites them, but Go, serde_json and JSON.parse
refuse the line. run.json
was sanitised for exactly this reason; the index, written 116 lines earlier in
the same module, was not -- and the index is the sweep-level record of which
hyperparameters diverged, the one file meant to be read long after the run.
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any

import pytest

from fedbrew.core.artifacts import append_run_index

pytestmark = pytest.mark.fast


def _diverged_metadata() -> dict[str, Any]:
    return {
        "run_id": "r0",
        "experiment_name": "blowup",
        "seed": 42,
        "output_dir": "outputs/blowup",
        "created_at": "2026-01-01T00:00:00+00:00",
        "tags": ["sweep"],
        "code_state": {"git_commit": "abc123", "git_dirty": False},
        "status": "diverged",
        "stopped_round": 1,
        "termination": {"detector": "non_finite", "value": math.nan},
        "final_metrics": {"fit_accuracy": 0.5166666666666667, "fit_loss": math.nan},
    }


def _strict(line: str) -> Any:
    """Parse as a reader that does not extend JSON with NaN/Infinity would."""

    def refuse(token: str) -> float:
        raise ValueError(f"invalid JSON constant {token!r}")

    return json.loads(line, parse_constant=refuse)


class RunsIndexJsonValidityTest(unittest.TestCase):
    def _write(self, metadata: dict[str, Any]) -> str:
        with tempfile.TemporaryDirectory() as directory:
            path = append_run_index(metadata, root_output_dir=directory)
            return Path(path).read_text(encoding="utf-8").strip()

    def test_a_diverged_run_writes_a_strictly_valid_line(self) -> None:
        line = self._write(_diverged_metadata())
        self.assertNotIn("NaN", line)
        self.assertNotIn("Infinity", line)
        record = _strict(line)
        self.assertIsNone(record["final_metrics"]["fit_loss"])
        self.assertIsNone(record["termination"]["value"])

    def test_the_finite_values_beside_it_survive(self) -> None:
        """Sanitising must null the non-finite value, not the whole record."""

        record = _strict(self._write(_diverged_metadata()))
        self.assertAlmostEqual(record["final_metrics"]["fit_accuracy"], 0.5166666667)
        self.assertEqual(record["status"], "diverged")
        self.assertEqual(record["stopped_round"], 1)
        self.assertEqual(record["termination"]["detector"], "non_finite")

    def test_an_infinity_is_nulled_too(self) -> None:
        metadata = _diverged_metadata()
        metadata["final_metrics"] = {"fit_loss": math.inf, "fit_grad": -math.inf}
        record = _strict(self._write(metadata))
        self.assertEqual(record["final_metrics"], {"fit_loss": None, "fit_grad": None})

    def test_a_completed_run_is_unchanged(self) -> None:
        metadata = _diverged_metadata()
        metadata["status"] = "completed"
        metadata["stopped_round"] = None
        metadata["termination"] = None
        metadata["final_metrics"] = {"fit_loss": 0.25, "fit_accuracy": 0.9}
        record = _strict(self._write(metadata))
        self.assertEqual(record["final_metrics"], {"fit_loss": 0.25, "fit_accuracy": 0.9})
        self.assertIsNone(record["termination"])

    def test_appending_keeps_every_line_parseable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            append_run_index(_diverged_metadata(), root_output_dir=directory)
            second = _diverged_metadata() | {"run_id": "r1", "status": "completed"}
            path = append_run_index(second, root_output_dir=directory)
            lines = Path(path).read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual([_strict(line)["run_id"] for line in lines], ["r0", "r1"])


class MetricJsonWritersTest(unittest.TestCase):
    """Every writer that serialises a measured metric runs it through json_safe.

    The defect was not that one writer forgot: it was that forgetting is
    invisible. This enumerates the repo's json.dump/dumps call sites and holds
    the ones that can carry a measured float to the contract, so a fourth
    writer added later fails here instead of writing an invalid file.
    """

    #: Call sites whose record provably cannot hold a non-finite float, with
    #: the reason. Everything else must pass allow_nan=False.
    EXEMPT: dict[str, str] = {
        "fedbrew/data/femnist.py": "partition_stats: counts, and ratios "
        "_parse_client_splits already validated to be finite and sum to 1",
        "fedbrew/data/generate.py": "partition_stats: same",
        "fedbrew/data/generic_sft.py": "partition_stats and a token mapping",
        "fedbrew/data/hf_causal_lm_text.py": "partition_stats: same",
        "fedbrew/data/oasst1.py": "asset manifest: hashes, paths, revisions",
        "fedbrew/data/oasst1_sft.py": "partition_stats: same",
        "fedbrew/data/tiny_causal_lm.py": "partition_stats: same",
        "fedbrew/data/writers/manifest.py": "manifest and per-client counts",
        "fedbrew/data/llm_assets/prepare.py": "asset manifest: no floats",
        "fedbrew/tasks/base.py": "model_config_key builds a cache key "
        "string, not a file; shared by both tasks that cache models",
        "tools/generate_openimage_shaped_synthetic.py": "two integer totals",
    }

    def _call_sites(self) -> list[tuple[str, int, bool]]:
        import ast

        sites: list[tuple[str, int, bool]] = []
        root = Path(__file__).resolve().parent.parent
        for path in sorted(root.rglob("*.py")):
            relative = path.relative_to(root)
            if any(
                part in {"tests", ".git", "AUDIT", "__pycache__", ".venv"}
                for part in relative.parts
            ):
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:  # pragma: no cover - not expected in-tree
                continue
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in {"dump", "dumps"}
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "json"
                ):
                    strict = any(kw.arg == "allow_nan" for kw in node.keywords)
                    sites.append((str(relative), node.lineno, strict))
        return sites

    def test_every_metric_writer_refuses_non_finite_floats(self) -> None:
        sites = self._call_sites()
        self.assertGreater(len(sites), 10, "the scan found nothing to check")
        for path, line, strict in sites:
            if path in self.EXEMPT:
                continue
            with self.subTest(site=f"{path}:{line}"):
                self.assertTrue(
                    strict,
                    f"{path}:{line} writes JSON without allow_nan=False. If its "
                    "record can hold a measured float, run it through "
                    "fedbrew.core.metrics.json_safe and pass "
                    "allow_nan=False; if it provably cannot, add it to EXEMPT "
                    "with the reason.",
                )

    def test_the_exemptions_all_still_exist(self) -> None:
        """An exemption for a deleted file would silently stop protecting."""

        scanned = {path for path, _, _ in self._call_sites()}
        self.assertEqual(set(self.EXEMPT) - scanned, set())


if __name__ == "__main__":
    unittest.main()
