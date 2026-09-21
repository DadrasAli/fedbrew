"""A rendering test must not be able to read the shell that started pytest.

`build_surface(force_rich=True)` says "render as if interactive" and cannot say
"and ignore the operator": rich's `Console` reads `NO_COLOR` itself and strips
every escape sequence, which `fedbrew/core/console.py` deliberately leaves to
it rather than keeping a second answer to one question. So eight tests here --
`test_console_surface`, `test_logging`, and six in `test_plan_header` -- failed
under `NO_COLOR=1`, on a machine where nothing they test had changed. A ninth,
`test_run_reporting.py::TheLiveClientLineTest`, fails the same way under
`COLUMNS=40`: it asserts the redrawn footer fills the terminal, which it does
at 80 columns and does not at 40.

`console_env` is the fix and this is its guard. Two directions, because either
alone passes for the wrong reason: that the mechanism is real (rich does strip
colour when the variable is set), and that the neutraliser removes it (the
rendered bytes are identical whatever the caller configured). Without the
first, a rich release that stopped reading `NO_COLOR` would leave the
neutraliser guarding nothing and this file green.
"""

from __future__ import annotations

import io
import os
import unittest
from unittest import mock

import console_env
import pytest

from fedbrew.core.console import Row, build_surface

pytestmark = pytest.mark.fast

ESCAPE = "\x1b["


def _rendered() -> str:
    buffer = io.StringIO()
    build_surface(file=buffer, force_rich=True).rows([Row("Device", "cuda")])
    return buffer.getvalue()


class TheMechanismIsRealTest(unittest.TestCase):
    """Without the neutraliser the caller wins, which is why it exists."""

    def test_no_color_strips_the_escapes_a_forced_surface_would_emit(self) -> None:
        with mock.patch.dict(os.environ, {"NO_COLOR": "1"}):
            stripped = _rendered()
        self.assertNotIn(ESCAPE, stripped)

    def test_the_same_call_is_coloured_with_the_variable_removed(self) -> None:
        with mock.patch.dict(os.environ):
            os.environ.pop("NO_COLOR", None)
            coloured = _rendered()
        self.assertIn(ESCAPE, coloured)


class TheNeutraliserRemovesEveryOpinionTest(unittest.TestCase):
    """One rendering per variable, all required to be byte-identical.

    Written over `CONSOLE_ENV` rather than over a list here, so adding a
    variable to the helper adds a case rather than a second list to keep.
    """

    def setUp(self) -> None:
        console_env.setUpModule()
        self.addCleanup(console_env.tearDownModule)
        self.neutral = _rendered()

    def test_the_neutral_rendering_is_coloured_at_all(self) -> None:
        """Guards the guard: identical-but-empty would satisfy every case."""

        self.assertIn(ESCAPE, self.neutral)

    def test_the_variable_that_actually_broke_the_suite_is_named(self) -> None:
        """The loop below is written over `CONSOLE_ENV`, so it cannot notice a
        variable leaving it -- removing `NO_COLOR` from the helper shrinks the
        loop rather than failing it. That one is measured, so it is pinned by
        name as well."""

        self.assertIn("NO_COLOR", console_env.CONSOLE_ENV)

    def test_no_variable_the_helper_clears_can_change_the_output(self) -> None:
        for name in console_env.CONSOLE_ENV:
            # "40" so COLUMNS and LINES get a value rich will actually use:
            # it ignores anything that is not all digits.
            for value in ("1", "0", "", "dumb", "40"):
                with self.subTest(variable=name, value=value):
                    with mock.patch.dict(os.environ, {name: value}):
                        # The helper is already active; a variable set now is
                        # exactly the case it cannot see, so re-enter it.
                        console_env.setUpModule()
                        try:
                            self.assertEqual(_rendered(), self.neutral)
                        finally:
                            console_env.tearDownModule()


class TheHelperRestoresWhatItFoundTest(unittest.TestCase):
    def test_a_variable_the_caller_set_is_put_back(self) -> None:
        with mock.patch.dict(os.environ, {"NO_COLOR": "1"}):
            console_env.setUpModule()
            self.assertNotIn("NO_COLOR", os.environ)
            console_env.tearDownModule()
            self.assertEqual(os.environ["NO_COLOR"], "1")

    def test_a_variable_that_was_absent_stays_absent(self) -> None:
        with mock.patch.dict(os.environ):
            os.environ.pop("NO_COLOR", None)
            console_env.setUpModule()
            console_env.tearDownModule()
            self.assertNotIn("NO_COLOR", os.environ)


if __name__ == "__main__":
    unittest.main()
