"""Neutralise the shell's opinion about colour for a whole test module.

Eight tests in this suite failed when the caller's environment set
`NO_COLOR=1`, and the fault was entirely on this side of the line.
`build_surface(force_rich=True)` exists so a test can exercise the interactive
path its own captured stdout would otherwise rule out, and it says both halves
of "render as if interactive": `force_terminal=True` and `color_system="256"`.
What it cannot say is "and ignore the operator" -- rich's `Console` reads
`NO_COLOR` itself and strips every escape, which is correct behaviour and is
why `fedbrew/core/console.py` deliberately does not reimplement it. A test that
then asserts an escape sequence is present is asserting something about the
shell that started pytest.

So the fix belongs here rather than in the package: a test that renders must do
it in an environment the caller cannot have configured. Import both names into
a module and every class in it renders into the same neutral environment:

    from console_env import setUpModule, tearDownModule  # noqa: F401

Module scope rather than a per-class mixin because the forced surfaces are
built in module-level helpers -- `_tone_prefix`, `_render_rich`, `_amber_lines`
-- that several classes share, so a mixin would have to be applied to all of
them and a new class could still forget.

Two variables were measured to change what this suite produces: `NO_COLOR`
breaks the eight tests above, and `COLUMNS` breaks
`test_run_reporting.py::TheLiveClientLineTest`, which asserts that the redrawn
footer is exactly one column short of the terminal -- true at 80 columns and
false at 40. The rest of `CONSOLE_ENV` is cleared because rich consults it for
the same three decisions -- colour, terminal-ness, size -- and a suite that is
immune to one variable by accident is not immune. That claim is guarded rather
than asserted: `tests/test_console_env.py` renders once per variable per value
and requires the output to be byte-identical to the neutral one.

The first version of this file left `COLUMNS` out, on the stated grounds that
no assertion depended on it. That was wrong, and it was wrong in the way this
whole finding is about: it had been reasoned about rather than run. The
measurement is now in the guard.
"""

from __future__ import annotations

import os
from unittest import mock

#: Every variable rich's `Console` consults, minus the two it reads only under
#: Jupyter. Three decisions: whether to emit colour (`NO_COLOR`, and
#: `TERM`/`COLORTERM` for the colour system), whether the destination counts as
#: a terminal (`FORCE_COLOR`, `TTY_COMPATIBLE`, `TTY_INTERACTIVE`), and how wide
#: it is (`COLUMNS`, `LINES`). Nothing in `fedbrew/` reads any of them -- the
#: package asks rich, which is the point: the answer is the operator's and a
#: test must not inherit it.
CONSOLE_ENV = (
    "NO_COLOR",
    "FORCE_COLOR",
    "TTY_COMPATIBLE",
    "TTY_INTERACTIVE",
    "TERM",
    "COLORTERM",
    "COLUMNS",
    "LINES",
)

#: A stack, so a module that neutralises while another already has stays
#: correct. Restoring is `mock.patch.dict`'s job: it snapshots the whole
#: environment on start and puts it back on stop, so a test that sets one of
#: these itself is also undone.
_ACTIVE: list[mock._patch_dict] = []


def setUpModule() -> None:
    """Remove the caller's colour configuration for the module's lifetime."""

    patcher = mock.patch.dict(os.environ)
    patcher.start()
    _ACTIVE.append(patcher)
    for name in CONSOLE_ENV:
        os.environ.pop(name, None)


def tearDownModule() -> None:
    """Put back exactly what was there, including variables a test set."""

    _ACTIVE.pop().stop()
