"""A generator may not read a config key its section's allow-list rejects.

``_GENERATOR_SECTION_KEYS`` is what ``_validate_generator_keys`` refuses
against, and it runs in ``generate_from_config`` before any generator is
dispatched. So a ``.get()`` for a key that is not on its section's list can
never fire: the config carrying it was rejected at load.

Code like that is worse than dead. It reads as an option -- four
``dataset.corpus_path`` / ``dataset.source_path`` / ``dataset.input_path``
fallbacks and two ``dataset.sequence_length`` defaults sat in the two text
generators, spelling out a precedence order between config locations, one of
which the loader refuses. A reader tracing "where can the corpus path come
from?" found six spellings and three that worked.

Derived both ways: the section for each local is read from the assignment that
binds it, and the accepted keys from the allow-list itself, so neither side is
a list kept here.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

import pytest

from fedbrew.data.generate import _GENERATOR_SECTION_KEYS

pytestmark = pytest.mark.fast

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "fedbrew" / "data"


def _section_of(node: ast.AST) -> str | None:
    """The section name in ``_mapping(config["x"])`` or ``config.get("x", ...)``."""

    for inner in ast.walk(node):
        if not isinstance(inner, ast.Subscript):
            continue
        if isinstance(inner.value, ast.Name) and inner.value.id == "config":
            if isinstance(inner.slice, ast.Constant) and isinstance(inner.slice.value, str):
                return inner.slice.value
    for inner in ast.walk(node):
        if (
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Attribute)
            and inner.func.attr == "get"
            and isinstance(inner.func.value, ast.Name)
            and inner.func.value.id == "config"
            and inner.args
            and isinstance(inner.args[0], ast.Constant)
            and isinstance(inner.args[0].value, str)
        ):
            return inner.args[0].value
    return None


def _offenders() -> list[str]:
    found: list[str] = []
    for path in sorted(DATA.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        # local name -> the config section it was bound from
        bound: dict[str, str] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            section = _section_of(node.value)
            if section in _GENERATOR_SECTION_KEYS:
                bound[target.id] = section

        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in bound
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                continue
            section = bound[node.func.value.id]
            key = node.args[0].value
            if key not in _GENERATOR_SECTION_KEYS[section]:
                found.append(
                    f"{path.relative_to(REPO_ROOT)}:{node.lineno}  "
                    f"{node.func.value.id}.get({key!r}) -- {section} accepts "
                    f"{sorted(_GENERATOR_SECTION_KEYS[section])}"
                )
    return sorted(found)


class GeneratorReadsAreReachableTest(unittest.TestCase):
    def test_no_generator_reads_a_key_its_section_rejects(self) -> None:
        offenders = _offenders()
        self.assertEqual(
            offenders,
            [],
            "these reads can never fire -- _validate_generator_keys refuses the "
            f"key before the generator runs: {offenders}",
        )

    def test_the_scan_binds_the_sections_it_claims_to_check(self) -> None:
        """A walk that binds nothing would pass while checking nothing."""

        bound_sections = set()
        for path in DATA.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and len(node.targets) == 1:
                    section = _section_of(node.value)
                    if section in _GENERATOR_SECTION_KEYS:
                        bound_sections.add(section)
        self.assertIn("causal_lm", bound_sections)
        self.assertIn("partition", bound_sections)
        self.assertGreaterEqual(len(bound_sections), 4)


if __name__ == "__main__":
    unittest.main()
