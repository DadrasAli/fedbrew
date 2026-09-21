"""Every dataset path a run config names must be one a generator config produces.

The sibling of tests/test_docs_references_resolve.py, for dataset manifests
instead of chapters, and it exists for the same reason. Three paths were dead:
``configs/mnist/fedavg.yaml`` offered ``mnist_label_skew`` and
``mnist_dirichlet_a01`` as commented alternatives and no generator config
produced either, while ``configs/openimage/fedavg.yaml`` named a manifest
nothing could produce as its **active** path.

Commented alternatives are a showcase and are meant to stay, which is exactly
why they are scanned: a commented path is copied into the active line by
someone who reads it as an offer. A dead one fails at load, minutes into
setting up a run, having looked authoritative in the file that suggested it.

Walks both trees rather than a kept list, so a new config cannot reintroduce
the defect.

**The exemption is a declaration, not a list.** An arm may legitimately need
data this repository cannot produce -- ``openimage/fedavg.yaml`` needs
FedScale's OpenImage partition -- and the honest form of that is the config
saying so where the path is. So a path that no generator produces passes only
if its config carries ``UNSHIPPED_DECLARATION`` verbatim. That keeps the
exemption at a location rather than on a name: it cannot be granted by adding
a word to an allow-list somewhere else, and it cannot be granted silently,
because the sentence has to be written next to the thing it excuses.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.fast

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_CONFIGS = REPO_ROOT / "configs"
GENERATOR_CONFIGS = REPO_ROOT / "data" / "configs"

#: The sentence a config must carry to name a path nothing produces. Written
#: in full rather than matched loosely, so it is a deliberate statement and not
#: something a passing mention of "does not ship" could satisfy.
UNSHIPPED_DECLARATION = "THIS ARM NEEDS DATA THIS REPOSITORY DOES NOT SHIP."

#: A dataset directory. Both roots that shipped configs use: the in-tree one
#: and the HPC variable the oasst1 4-client arm and its generator share.
_DATASET_PATH = re.compile(r"(?:data/generated|\$FL_DATA_ROOT)/[\w./-]+")


def _producible() -> set[str]:
    """Every dataset directory some generator config writes."""

    produced = set()
    for path in sorted(GENERATOR_CONFIGS.rglob("*.yaml")):
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            continue
        output_dir = (loaded.get("dataset") or {}).get("output_dir")
        if output_dir:
            produced.add(str(output_dir).rstrip("/"))
    return produced


def _named() -> list[tuple[Path, int, str, bool]]:
    """(config, line, dataset directory, is_active) for every path named."""

    found = []
    for path in sorted(RUN_CONFIGS.rglob("*.yaml")):
        if "llm_assets" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        loaded = yaml.safe_load(text)
        active = (loaded or {}).get("data", {}).get("path") if isinstance(loaded, dict) else None
        for number, line in enumerate(text.splitlines(), start=1):
            for match in _DATASET_PATH.findall(line):
                directory = match.rsplit("/manifest.json", 1)[0].rstrip("/")
                is_active = active is not None and match == str(active)
                found.append((path, number, directory, is_active))
    return found


def _declares_unshipped(path: Path) -> bool:
    return UNSHIPPED_DECLARATION in path.read_text(encoding="utf-8")


class DatasetPathsResolveTest(unittest.TestCase):
    def test_every_named_path_is_produced_or_declared_unshipped(self) -> None:
        producible = _producible()
        offenders = [
            f"{path.relative_to(REPO_ROOT)}:{number}  {directory}"
            f"{'  (active)' if is_active else '  (commented alternative)'}"
            for path, number, directory, is_active in _named()
            if directory not in producible and not _declares_unshipped(path)
        ]
        self.assertEqual(
            offenders,
            [],
            f"dataset paths no generator config produces: {offenders}. Either "
            "add a generator config that writes the directory, repoint the "
            f"path at one that exists, or write {UNSHIPPED_DECLARATION!r} into "
            "the config beside the path if the data genuinely comes from "
            "outside this repository.",
        )

    def test_a_declaration_is_only_honoured_where_it_is_written(self) -> None:
        """The exemption is per config, and no config has it by accident."""

        declaring = sorted(
            str(path.relative_to(REPO_ROOT))
            for path in RUN_CONFIGS.rglob("*.yaml")
            if _declares_unshipped(path)
        )
        self.assertEqual(declaring, ["configs/openimage/fedavg.yaml"])

    def test_the_declaring_config_would_otherwise_fail(self) -> None:
        """The exemption earns itself.

        A declaration on a config whose paths all resolve is a sentence saying
        something untrue, and it would sit there excusing a future breakage.
        """

        producible = _producible()
        unresolved = [
            directory
            for path, _, directory, _ in _named()
            if _declares_unshipped(path) and directory not in producible
        ]
        self.assertNotEqual(unresolved, [])

    def test_the_scan_reaches_both_kinds_of_path(self) -> None:
        """A walk matching nothing, or only active paths, would check little."""

        named = _named()
        self.assertGreater(len(_producible()), 5, "almost no generator outputs found")
        self.assertGreater(len([n for n in named if n[3]]), 20, "few active paths found")
        self.assertGreater(
            len([n for n in named if not n[3]]),
            3,
            "no commented alternatives found; the showcase is what this guards",
        )

    def test_every_generator_writes_somewhere_distinct(self) -> None:
        """Two configs writing one directory means one silently overwrites the
        other, and the filename-to-directory correspondence that
        data/configs/README.md calls the provenance record stops holding."""

        seen: dict[str, str] = {}
        clashes = []
        for path in sorted(GENERATOR_CONFIGS.rglob("*.yaml")):
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                continue
            output_dir = (loaded.get("dataset") or {}).get("output_dir")
            if not output_dir:
                continue
            if output_dir in seen:
                clashes.append(f"{seen[output_dir]} and {path.name} both write {output_dir}")
            seen[str(output_dir)] = path.name
        self.assertEqual(clashes, [])


if __name__ == "__main__":
    unittest.main()
