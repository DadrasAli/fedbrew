"""Nothing in fedbrew/ may import an optional dependency at module level.

This is the assumption the extras rest on, and it is invisible in review: a
module-level `import torchvision` looks like every other import, and on the
developer's machine -- where the extra is installed -- nothing goes wrong. It
goes wrong for a reader who ran `pip install -e .`, as an ImportError at
`import fedbrew` rather than the guarded message the loader raises, and it goes
wrong for every command, not just the one that needed the dependency.

torchvision is the reason this exists. It was a core dependency until the
release pin inside it -- every torchvision release requires one exact torch
version -- was found to be discarding pre-installed, driver-matched torch
builds on `pip install -e .`. Moving it into the `vision` extra is only safe
while every use stays inside the function that needs it, so that is what is
checked here, for every optional dependency rather than for torchvision alone.

The check is syntactic: `ast` walked over each module, flagging an Import or
ImportFrom whose parent chain reaches the module body. A lazy import lives
inside a function or method and is not flagged. `TYPE_CHECKING` blocks are
exempt -- they never execute.
"""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE = REPO_ROOT / "fedbrew"
PYPROJECT = REPO_ROOT / "pyproject.toml"

#: Distribution name -> the top-level module(s) importing it would bind.
#: Only names that differ, or that are not a simple lowercase of the
#: distribution, need an entry; the rest fall through to the name itself.
_IMPORT_NAMES = {
    "huggingface-hub": {"huggingface_hub"},
    "Pillow": {"PIL"},
    "pytest": {"pytest"},
    "hypothesis": {"hypothesis"},
    "ruff": set(),  # a binary, never imported
}


def _optional_distributions() -> set[str]:
    """Every distribution named by an extra in pyproject.

    Hand-parsed for the same reason tests/test_docs_installation.py is:
    tomllib is 3.11+ and pyproject declares a 3.10 floor.
    """

    text = PYPROJECT.read_text(encoding="utf-8")
    start = text.index("[project.optional-dependencies]")
    end = text.find("\n[", start + 1)
    block = text[start : end if end > 0 else len(text)]
    names = set()
    for raw in re.findall(r'"([^"]+)"', block):
        name = re.split(r"[<>=!\[;\s]", raw, maxsplit=1)[0]
        if name and name != "fedbrew":  # the femnist alias points at this package
            names.add(name)
    return names


def _optional_modules() -> set[str]:
    modules: set[str] = set()
    for distribution in _optional_distributions():
        modules |= _IMPORT_NAMES.get(distribution, {distribution.replace("-", "_")})
    return modules


#: Statements whose bodies run when the module is imported. A function body
#: does not, which is exactly what makes a lazy import lazy, so FunctionDef and
#: AsyncFunctionDef are absent here on purpose -- and ast.walk cannot be used,
#: because it descends into them and would flag every guarded import as a
#: module-level one.
_EXECUTES_ON_IMPORT = (ast.If, ast.Try, ast.With, ast.For, ast.While, ast.ClassDef)


def _is_type_checking_guard(node: ast.stmt) -> bool:
    """`if TYPE_CHECKING:` -- never executed, so its imports are free."""

    if not isinstance(node, ast.If):
        return False
    test = node.test
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def _module_level_imports(source: str) -> set[str]:
    """Top-level module names bound by an import that runs at import time."""

    found: set[str] = set()

    def visit(body: list[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, ast.Import):
                found.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
            elif _is_type_checking_guard(node):
                continue
            elif isinstance(node, _EXECUTES_ON_IMPORT):
                visit(node.body)
                visit(getattr(node, "orelse", []))
                visit(getattr(node, "finalbody", []))
                for handler in getattr(node, "handlers", []):
                    visit(handler.body)

    visit(ast.parse(source).body)
    return found


class OptionalDependenciesAreImportedLazilyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.optional = _optional_modules()

    def test_the_optional_module_set_was_actually_derived(self) -> None:
        """Guards the guard: an empty set would pass every check below."""

        self.assertIn("torchvision", self.optional)
        self.assertIn("transformers", self.optional)
        self.assertIn("PIL", self.optional)

    def test_no_module_in_the_package_imports_one_at_module_level(self) -> None:
        offenders: list[str] = []
        scanned = 0
        for path in sorted(PACKAGE.rglob("*.py")):
            scanned += 1
            imported = _module_level_imports(path.read_text(encoding="utf-8"))
            for name in sorted(imported & self.optional):
                offenders.append(f"{path.relative_to(REPO_ROOT).as_posix()}: {name}")
        self.assertGreater(scanned, 10, "the scan found almost no modules; the glob is wrong")
        self.assertEqual(
            offenders,
            [],
            "these import an optional dependency at module level, so "
            "`pip install -e .` without the extra would fail at `import "
            f"fedbrew` instead of at the loader that needs it: {offenders}",
        )

    def test_torch_is_not_treated_as_optional(self) -> None:
        """The counterpart: torch is core, so importing it at module level is fine."""

        self.assertNotIn("torch", self.optional)
        client = PACKAGE / "clients" / "torch_sgd_client.py"
        self.assertIn(
            "torch",
            _module_level_imports(client.read_text(encoding="utf-8")),
            "a core dependency imported at module level must not be flagged; "
            "if this fails the walk has stopped seeing module-level imports "
            "and the check above would pass vacuously",
        )


class TorchvisionIsNotACoreDependencyTest(unittest.TestCase):
    """The move itself, so it cannot be undone without a failing test."""

    def test_it_is_declared_only_by_an_extra(self) -> None:
        text = PYPROJECT.read_text(encoding="utf-8")
        core = text[text.index("dependencies = [") : text.index("[project.optional-dependencies]")]
        self.assertNotIn(
            "torchvision",
            core,
            "torchvision is back in [project.dependencies]. Every torchvision "
            "release pins torch to one exact version, so a core dependency on "
            "it replaces a pre-installed driver-matched torch on install.",
        )
        self.assertIn("torchvision", _optional_distributions())


if __name__ == "__main__":
    unittest.main()
