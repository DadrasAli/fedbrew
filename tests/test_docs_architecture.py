"""docs/01-architecture.md names real modules and the registered components.

An architecture chapter goes stale in a specific way: the prose stays roughly
true while the file paths under it rot. A reader following "this lives in
core/foo.py" to a file that moved two refactors ago loses more time than one
who was told nothing, because they assume they are looking in the wrong place.

So every path the chapter names is checked to exist, and the registry listings
are checked against the registries. The claims this cannot check -- that
aggregation is streaming, that the phase order is what it says -- are guarded
by the behavioural tests the chapter's For agents section names, which are
themselves checked to exist.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import pytest

from fedbrew.core import registry

pytestmark = pytest.mark.fast

REPO_ROOT = Path(__file__).resolve().parent.parent
CHAPTER = REPO_ROOT / "docs" / "01-architecture.md"

#: Registry attribute -> the heading its listing sits under in the chapter.
_REGISTRIES = (
    "server_strategies",
    "client_updates",
    "models",
    "datasets",
    "tasks",
    "generators",
)


def _chapter_text() -> str:
    return CHAPTER.read_text(encoding="utf-8")


class ChapterShapeTest(unittest.TestCase):
    def test_present_and_agent_facing(self) -> None:
        self.assertTrue(CHAPTER.is_file())
        self.assertIn("\n## For agents\n", _chapter_text())


class RegistryListingTest(unittest.TestCase):
    """The names block must be exactly what register_builtin_components makes."""

    def setUp(self) -> None:
        registry.register_builtin_components()
        self.text = _chapter_text()

    def _names_block(self) -> str:
        start = self.text.index("Registered names:")
        opening = self.text.index("```", start)
        closing = self.text.index("```", opening + 3)
        return self.text[opening + 3 : closing]

    def test_every_registered_name_is_listed(self) -> None:
        block = self._names_block()
        listed = set(block.split())
        for attribute in _REGISTRIES:
            registered = set(getattr(registry, attribute).builtin())
            with self.subTest(registry=attribute):
                self.assertTrue(
                    registered <= listed,
                    f"{attribute} names missing from the chapter: {sorted(registered - listed)}",
                )

    def test_the_block_invents_nothing(self) -> None:
        listed = set(self._names_block().split())
        known = set(_REGISTRIES)
        for attribute in _REGISTRIES:
            known |= set(getattr(registry, attribute).builtin())
        self.assertEqual(
            sorted(listed - known),
            [],
            "the chapter lists components that are not registered",
        )

    def test_the_registry_counts_are_right(self) -> None:
        """The chapter prints a count per registry in its table."""

        for attribute, config_key in (
            ("server_strategies", "server.strategy"),
            ("client_updates", "client.update_rule"),
            ("models", "model.name"),
            ("datasets", "data.name"),
            ("tasks", "task.name"),
            ("generators", "dataset.name"),
        ):
            count = len(getattr(registry, attribute).builtin())
            with self.subTest(registry=attribute):
                self.assertRegex(
                    self.text,
                    rf"`{re.escape(config_key)}` \| {count} \|",
                    f"{attribute} has {count} entries; the chapter's table says otherwise",
                )


class CitedPathsTest(unittest.TestCase):
    """Every module, test and config the chapter points at must exist."""

    def setUp(self) -> None:
        self.text = _chapter_text()

    def test_cited_modules_exist(self) -> None:
        cited = set(re.findall(r"`(fedbrew/[\w/]+\.py)", self.text))
        self.assertGreater(len(cited), 15, "an architecture chapter cites paths")
        missing = sorted(path for path in cited if not (REPO_ROOT / path).is_file())
        self.assertEqual(missing, [])

    def test_cited_directories_exist(self) -> None:
        cited = set(re.findall(r"`(fedbrew/\w+/)`", self.text))
        missing = sorted(path for path in cited if not (REPO_ROOT / path).is_dir())
        self.assertEqual(missing, [])

    def test_cited_tests_exist(self) -> None:
        cited = set(re.findall(r"`(tests/test_\w+\.py)`", self.text))
        self.assertTrue(cited)
        missing = sorted(path for path in cited if not (REPO_ROOT / path).is_file())
        self.assertEqual(missing, [])

    def test_cited_configs_exist(self) -> None:
        # Not backtick-delimited: configs are named inside fenced command
        # blocks as often as they are inline.
        cited = set(re.findall(r"(configs/[\w/]+\.yaml)", self.text))
        self.assertTrue(cited)
        missing = sorted(path for path in cited if not (REPO_ROOT / path).is_file())
        self.assertEqual(missing, [])


class AbstractBaseTest(unittest.TestCase):
    """The methods listed for each base must be the abstract ones."""

    def test_declared_methods_are_real(self) -> None:
        from fedbrew.clients.base import ClientUpdate
        from fedbrew.data.dataset import FederatedDataset
        from fedbrew.servers.base import ServerStrategy
        from fedbrew.tasks.base import TaskAdapter

        text = _chapter_text()
        for base in (ServerStrategy, ClientUpdate, TaskAdapter, FederatedDataset):
            for method in sorted(base.__abstractmethods__):
                with self.subTest(base=base.__name__, method=method):
                    self.assertIn(
                        f"`{method}`",
                        text,
                        f"{base.__name__}.{method} is abstract but the chapter does not name it",
                    )


if __name__ == "__main__":
    unittest.main()
