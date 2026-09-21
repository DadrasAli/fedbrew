"""``runtime.deterministic`` is read once and passed down.

It used to be read twice, in ``runner.run`` and again in
``runtime_setup.configure_runtime``, each supplying its own ``False``.

The two readers applied different rules -- the runner's goes through a bool
check that raises on ``deterministic: "false"``, the other took
``.get(..., False)`` and would have treated that string as true -- but that
difference was not reachable, because ``validate_config`` runs
``_validate_extra_bools`` over both keys before either reader sees them. The
defect was the second default, not the second parser.

Both defaults agreeing is what made it invisible, and is exactly the state a
duplicated read is in right up until one of them changes.

``logging.py`` was the third reader, for the plan header. That one could not
change what the run did, only what it *said* about it -- but it carried its
own implicit default, so a change to the runner's default would have printed
"off" for a run that was deterministic. A report disagreeing with the run is
the defect this whole area exists to prevent, so the header is handed the
resolved pair and reads neither key.

The invariant is therefore one reader, and it is ``runner.py``: the module
that resolves them, applies them, and hands them to everything else.

The check is structural rather than behavioural because the behaviour is
currently identical -- that is the point. A test asserting the two agree would
pass today and pass again after a divergence in the half it did not exercise.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

REPO_ROOT = Path(__file__).resolve().parent.parent
CORE = REPO_ROOT / "fedbrew" / "core"

#: Flags the runner resolves once and hands to whoever needs them. Each is a
#: claim that exactly one module reads it from the config.
SINGLY_READ_FLAGS = ("deterministic", "deterministic_warn_only")

#: The one module allowed to read them. Everything else -- the module that
#: applies determinism, and the one that prints it -- receives the resolved
#: value. Named rather than derived: a second reader appearing anywhere in
#: fedbrew/core is the defect, so the check cannot be "wherever they happen
#: to be read".
OWNER = "runner.py"


def _reads_of(flag: str) -> list[str]:
    """Every ``...extra.get("<flag>", ...)`` or ``_runtime_extra_bool`` site."""

    found = []
    for path in sorted(CORE.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            first = node.args[0] if not isinstance(node.func, ast.Attribute) else None
            key = node.args[0] if isinstance(node.func, ast.Attribute) else None
            # runner's helper takes (config, name, default)
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "_runtime_extra_bool"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value == flag
            ):
                found.append(f"{path.name}:{node.lineno}")
                continue
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and isinstance(key, ast.Constant)
                and key.value == flag
                and "extra" in ast.dump(node.func.value)
            ):
                found.append(f"{path.name}:{node.lineno}")
            del first
    return found


class OneReadPerFlagTest(unittest.TestCase):
    def test_exactly_one_module_reads_each_flag(self) -> None:
        for flag in SINGLY_READ_FLAGS:
            with self.subTest(flag=flag):
                sites = [s for s in _reads_of(flag) if not s.startswith(OWNER)]
                self.assertEqual(
                    sites,
                    [],
                    f"runtime.{flag} is read outside {OWNER}: {sites}. It is "
                    f"resolved in {OWNER} and passed down; a second read "
                    "carries a second default and a second idea of what the "
                    "key accepts, and the two agree only until one changes.",
                )

    def test_the_scan_finds_the_owning_read(self) -> None:
        """A walk matching nothing would pass while checking nothing."""

        for flag in SINGLY_READ_FLAGS:
            with self.subTest(flag=flag):
                self.assertTrue(_reads_of(flag), f"no read of {flag} found at all")

    def test_configure_runtime_takes_it_as_an_argument(self) -> None:
        """The other half: it must still receive the value it stopped reading."""

        from fedbrew.core.runtime_setup import configure_runtime

        names = configure_runtime.__code__.co_varnames[: configure_runtime.__code__.co_argcount]
        self.assertIn("deterministic", names)

    def test_the_plan_header_takes_both_and_defaults_neither(self) -> None:
        """A default here would be the second answer all over again: the
        header would print it whenever a caller forgot, which is exactly the
        case where the printed line and the run disagree."""

        import inspect

        from fedbrew.core.logging import print_plan_header

        parameters = inspect.signature(print_plan_header).parameters
        for flag in SINGLY_READ_FLAGS:
            with self.subTest(flag=flag):
                self.assertIn(flag, parameters)
                self.assertIs(parameters[flag].default, inspect.Parameter.empty)


if __name__ == "__main__":
    unittest.main()
