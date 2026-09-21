"""Only fedbrew/core/console.py may paint a terminal.

The palette, the marker set and the TTY gate are worth nothing if a second
module can quietly pick its own cyan. That is not a hypothetical: before this
guard, three modules rendered independently -- fedbrew/core/logging.py,
fedbrew/core/validation.py and fedbrew/cli/inspect_generated_data.py -- each
with its own Console, its own colours and its own idea of what a redirected
stream should get. Two of them coloured errors red and one coloured them
yellow-on-red-border, and none of them agreed on whether a non-terminal should
receive escape sequences at all.

It is also invisible in review. `console.print("[bold cyan]Round 3[/]")` looks
like every other print, and it works on the author's machine. It goes wrong in
a SLURM log, where the escape sequences land in the file, and it goes wrong
for the palette, which stops meaning anything once amber is used for whatever
the last author felt was noteworthy.

So the check is syntactic and it has three parts, all of them run over every
module under fedbrew/ except console.py itself:

1. **No rich import**, at module level or inside a function. Composing a
   `rich.table.Table` somewhere else is exactly the second console layer this
   forbids, and a lazy import hides it from a module-level scan.
2. **No ANSI escape literal** -- `\\x1b[`, `\\033[`, `\\u001b[`. The way round
   a rich ban is to write the escape by hand.
3. **No rich console markup** -- `[bold cyan]`, `[/]`, `[red]`. The tag
   vocabulary is enumerated rather than matched loosely, so `f"[{index}]"` and
   a list literal are not findings.

Each check is paired with a positive control asserting it still detects what
it is looking for, because a scan narrowed by a later edit passes silently and
a guard that cannot fail is worse than no guard.
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
CONSOLE = PACKAGE / "core" / "console.py"

#: The one module allowed to do all three.
RENDERER = CONSOLE.relative_to(REPO_ROOT).as_posix()

#: Style words rich's markup parser understands, plus the closing forms. Only
#: these open a tag, so an index expression or a list display is not a hit.
_STYLE_WORD = (
    r"bold|dim|italic|underline|blink|reverse|strike|conceal|"
    r"black|red|green|yellow|blue|magenta|cyan|white|"
    r"grey\d*|gray\d*|bright_\w+|orange\d*|gold\d*|cornsilk\d*|"
    r"default|none|on\s+\w+|#[0-9A-Fa-f]{6}"
)
_MARKUP = re.compile(rf"\[/?(?:{_STYLE_WORD})\b[^\]]*\]|\[/\]")

_ANSI = re.compile(r"\\x1b\[|\\033\[|\\u001[bB]\[|\x1b\[")


def _package_modules() -> list[Path]:
    return sorted(PACKAGE.rglob("*.py"))


def _imports_rich(source: str) -> bool:
    """Whether the module binds anything from rich, at any nesting depth.

    ast.walk rather than a top-level visit, the opposite of
    tests/test_optional_dependency_imports.py: there a lazy import is the
    point, here it is the loophole.
    """

    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] == "rich" for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] == "rich":
                return True
    return False


def _relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


class ConsoleIsTheOnlyRendererTest(unittest.TestCase):
    def setUp(self) -> None:
        self.modules = _package_modules()
        self.assertGreater(
            len(self.modules),
            10,
            "the scan found almost no modules; the glob is wrong and every "
            "check below would pass vacuously",
        )
        self.assertIn(
            CONSOLE,
            self.modules,
            "fedbrew/core/console.py is missing; it is the module every other "
            "one is required to defer to",
        )

    def test_no_other_module_imports_rich(self) -> None:
        offenders = [
            _relative(path)
            for path in self.modules
            if path != CONSOLE and _imports_rich(path.read_text(encoding="utf-8"))
        ]
        self.assertEqual(
            offenders,
            [],
            "these import rich, so they can render a terminal without going "
            f"through {RENDERER}'s palette, markers and TTY gate: {offenders}",
        )

    def test_no_other_module_writes_an_escape_sequence(self) -> None:
        offenders = [
            _relative(path)
            for path in self.modules
            if path != CONSOLE and _ANSI.search(path.read_text(encoding="utf-8"))
        ]
        self.assertEqual(
            offenders,
            [],
            "these contain a hand-written ANSI escape, which is the way round "
            f"a rich ban rather than an alternative to it: {offenders}",
        )

    def test_no_other_module_writes_rich_markup(self) -> None:
        offenders: list[str] = []
        for path in self.modules:
            if path == CONSOLE:
                continue
            found = _MARKUP.findall(path.read_text(encoding="utf-8"))
            if found:
                offenders.append(f"{_relative(path)}: {sorted(set(found))[:4]}")
        self.assertEqual(
            offenders,
            [],
            "these embed rich console markup. Compose the content and pass it "
            f"to {RENDERER}, which styles values as Text rather than as markup "
            f"-- a value can be a path, and rich reads outputs/run[1]/ as a "
            f"style tag: {offenders}",
        )


class TheChecksStillDetectWhatTheyLookForTest(unittest.TestCase):
    """Positive controls. Each check must fail on a module that does the thing."""

    def test_the_rich_import_check_sees_a_module_level_import(self) -> None:
        self.assertTrue(_imports_rich("from rich.console import Console\n"))
        self.assertTrue(_imports_rich("import rich.table\n"))

    def test_the_rich_import_check_sees_a_lazy_import(self) -> None:
        self.assertTrue(
            _imports_rich("def render():\n    from rich.table import Table\n    return Table\n"),
            "a function-level import is the loophole this check exists for",
        )

    def test_the_rich_import_check_does_not_fire_on_unrelated_names(self) -> None:
        self.assertFalse(_imports_rich("from fedbrew.core.console import build_surface\n"))
        self.assertFalse(_imports_rich("import richness\n"))

    def test_the_escape_check_sees_every_spelling(self) -> None:
        for source in (r'"\x1b[1m"', r'"\033[31m"', r'"\u001b[0m"', '"\x1b[0m"'):
            self.assertTrue(_ANSI.search(source), f"missed {source!r}")

    def test_the_markup_check_sees_the_forms_that_were_in_use(self) -> None:
        for source in (
            '"[bold cyan]EXPERIMENT VALIDATION REPORT[/]"',
            '"[bold]Manifest:[/] [cyan]path[/]"',
            '"[yellow]0 warning(s)[/]"',
            '"[bold red]INVALID EXPERIMENT[/]"',
        ):
            self.assertTrue(_MARKUP.search(source), f"missed {source!r}")

    def test_the_markup_check_does_not_fire_on_ordinary_brackets(self) -> None:
        for source in (
            'f"round [{index}]"',
            "values = [1, 2, 3]",
            'name = row["client_id"]',
            'f"{path}[{shard}]"',
            '"[INFO] starting"',
        ):
            self.assertFalse(_MARKUP.search(source), f"false positive on {source!r}")

    def test_the_renderer_itself_is_what_the_checks_would_flag(self) -> None:
        """Guards the guard: if console.py stopped importing rich, every check
        above would be trivially satisfiable by deleting all output."""

        self.assertTrue(
            _imports_rich(CONSOLE.read_text(encoding="utf-8")),
            "fedbrew/core/console.py no longer imports rich; either the "
            "renderer moved or the package stopped rendering at all",
        )


if __name__ == "__main__":
    unittest.main()
