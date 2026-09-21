"""The console layer's own behaviour: the gate, the row builder, the rail.

tests/test_console_is_the_only_renderer.py says nothing else may render.
This says the one thing that may, renders correctly -- otherwise the guard
protects a monopoly on being wrong.

What is pinned here is what the rest of the package now depends on being true:
a redirected stream gets no escape sequences at all, a block's label column is
measured rather than guessed, and a value is never read as markup. That last
one is not hypothetical: every value this package prints is a path, an
identifier or a number, and `outputs/run[1]/` is valid rich markup that would
be silently swallowed.
"""

from __future__ import annotations

import io
import unicodedata
import unittest

import pytest
from console_env import setUpModule, tearDownModule  # noqa: F401

from fedbrew.core.console import (
    AMBER,
    DONE,
    FAIL,
    GOLD,
    PENDING,
    PULSE,
    RAIL,
    WARN,
    Row,
    Verbosity,
    build_surface,
    measure,
)

pytestmark = pytest.mark.fast

ESCAPE = "\x1b["


class TheTtyGateTest(unittest.TestCase):
    """One question, asked once: is the destination a terminal."""

    def test_a_redirected_stream_gets_no_escape_sequence(self) -> None:
        buffer = io.StringIO()
        surface = build_surface(file=buffer)
        surface.rule("SECTION")
        surface.rows([Row("Device", "cuda")])
        surface.line("done", tone=GOLD, marker=DONE)

        rendered = buffer.getvalue()
        self.assertNotIn(ESCAPE, rendered)
        self.assertIn("Device", rendered)
        self.assertIn("cuda", rendered)

    def test_no_rich_stays_plain_even_when_forced_interactive(self) -> None:
        """--no-rich has to win over the gate, not merely agree with it: it is
        the flag a reader reaches for when the auto-detection is wrong."""

        buffer = io.StringIO()
        surface = build_surface(file=buffer, no_rich=True, force_rich=False)
        surface.rows([Row("Device", "cuda")])
        self.assertNotIn(ESCAPE, buffer.getvalue())

    def test_an_interactive_destination_is_coloured(self) -> None:
        buffer = io.StringIO()
        surface = build_surface(file=buffer, force_rich=True)
        surface.rows([Row("Device", "cuda")])
        self.assertIn(ESCAPE, buffer.getvalue())

    def test_a_redirected_stream_never_redraws(self) -> None:
        """A carriage-return redraw in a log file is one line per update
        instead of one per result, which is how a SLURM log reaches a gigabyte
        of progress bar."""

        buffer = io.StringIO()
        build_surface(file=buffer).redraw("loading tokenizer")
        self.assertEqual(buffer.getvalue(), "")

    def test_quiet_prints_nothing_at_all(self) -> None:
        buffer = io.StringIO()
        surface = build_surface(file=buffer, quiet=True)
        surface.rule("SECTION")
        surface.rows([Row("Device", "cuda")])
        surface.line("done")
        surface.blank()
        self.assertEqual(buffer.getvalue(), "")
        self.assertIs(surface.verbosity, Verbosity.QUIET)

    def test_verbose_is_a_third_level_not_a_second_flag(self) -> None:
        self.assertIs(build_surface(verbose=True).verbosity, Verbosity.VERBOSE)
        self.assertIs(build_surface().verbosity, Verbosity.NORMAL)
        # Quiet wins: a caller that passed both asked for less, and printing
        # everything would be the surprising reading.
        self.assertIs(build_surface(quiet=True, verbose=True).verbosity, Verbosity.QUIET)


class TheRowBuilderTest(unittest.TestCase):
    def test_the_label_column_is_the_widest_label_in_the_block(self) -> None:
        rows = [Row("Device", "cuda"), Row("Experiment name", "smoke"), Row("Seed", "42")]
        self.assertEqual(measure(rows), len("Experiment name"))

        buffer = io.StringIO()
        build_surface(file=buffer).rows(rows)
        lines = buffer.getvalue().splitlines()
        starts = [
            line.index(value) for line, value in zip(lines, ("cuda", "smoke", "42"), strict=True)
        ]
        self.assertEqual(len(set(starts)), 1, f"values are not in one column: {lines}")

    def test_a_row_with_no_value_is_not_padded(self) -> None:
        """A bare label is a heading. Trailing spaces on it turn a diff of two
        captured logs into a difference that is not one."""

        buffer = io.StringIO()
        build_surface(file=buffer).rows([Row("Metrics"), Row("Device", "cpu")])
        self.assertEqual(buffer.getvalue().splitlines()[0], "Metrics")

    def test_the_plain_and_interactive_forms_carry_the_same_text(self) -> None:
        rows = [Row("Device", "cuda"), Row("Experiment name", "smoke")]

        plain, rich = io.StringIO(), io.StringIO()
        build_surface(file=plain).rows(rows)
        build_surface(file=rich, force_rich=True).rows(rows)

        stripped = _strip(rich.getvalue())
        self.assertEqual(
            [line.rstrip() for line in stripped.splitlines()],
            [line.rstrip() for line in plain.getvalue().splitlines()],
            "the two paths must differ in colour only",
        )

    def test_a_value_is_not_read_as_markup(self) -> None:
        for value in ("outputs/run[1]/best.pt", "[bold red]not a style[/]", "loss[0]"):
            for force_rich in (False, True):
                buffer = io.StringIO()
                build_surface(file=buffer, force_rich=force_rich).rows([Row("Path", value)])
                self.assertIn(
                    value,
                    _strip(buffer.getvalue()),
                    f"{value!r} was swallowed as markup (force_rich={force_rich})",
                )


class TheRailTest(unittest.TestCase):
    def test_a_stage_settles_as_done_and_carries_what_it_learned(self) -> None:
        buffer = io.StringIO()
        rail = build_surface(file=buffer).rail(["load config", "resolve revision"])
        with rail.stage("resolve revision") as stage:
            stage.done("abc123def456")

        rendered = buffer.getvalue()
        self.assertIn(f"{RAIL} {DONE} resolve revision", rendered)
        self.assertIn("abc123def456", rendered)
        self.assertFalse(rail.stopped)
        self.assertEqual((rail.warnings, rail.failures), (0, 0))

    def test_a_stage_that_says_nothing_still_settles(self) -> None:
        buffer = io.StringIO()
        rail = build_surface(file=buffer).rail()
        with rail.stage("mkdir cache"):
            pass
        self.assertIn(f"{RAIL} {DONE} mkdir cache", buffer.getvalue())

    def test_a_warning_is_amber_and_counted(self) -> None:
        buffer = io.StringIO()
        rail = build_surface(file=buffer, force_rich=True).rail()
        with rail.stage("stage dataset") as stage:
            stage.warn("copied to node-local scratch")

        self.assertIn(WARN, _strip(buffer.getvalue()))
        self.assertIn(_ansi(AMBER), buffer.getvalue())
        self.assertEqual(rail.warnings, 1)
        self.assertFalse(rail.stopped)

    def test_a_raising_stage_settles_as_failed_stops_the_rail_and_reraises(self) -> None:
        """The rail reports what it knows and then gets out of the way: the
        real exception is what tells the caller what went wrong, and swallowing
        it to keep the display tidy is how a 30-second silent failure becomes a
        30-second silent success."""

        buffer = io.StringIO()
        rail = build_surface(file=buffer).rail()
        with self.assertRaises(ValueError):
            with rail.stage("download model"):
                raise ValueError("no such revision")

        self.assertIn(f"{RAIL} {FAIL} download model", buffer.getvalue())
        self.assertTrue(rail.stopped)
        self.assertEqual(rail.failures, 1)

    def test_the_pending_line_is_a_terminal_only_redraw(self) -> None:
        """The settled line carries the same information, once. A log file
        gets that and not the pulse."""

        buffer = io.StringIO()
        rail = build_surface(file=buffer).rail()
        with rail.stage("tokenize") as stage:
            for _ in range(5):
                stage.tick()
            stage.done("24,239 messages")

        rendered = buffer.getvalue()
        self.assertEqual(rendered.count("\n"), 1, f"expected one settled line: {rendered!r}")
        self.assertNotIn("\r", rendered)

    def test_a_settled_result_needs_no_stage(self) -> None:
        """Every stage of path A of `generate` is short. It gets result lines;
        animating it would be a lie about where the time went."""

        buffer = io.StringIO()
        rail = build_surface(file=buffer).rail(["partition", "write shards"])
        rail.result("partition", "5 clients, smallest 18 examples")
        rail.result("write shards", "10 files")

        lines = buffer.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(all(line.startswith(f"{RAIL} {DONE} ") for line in lines))
        columns = {
            line.index("5 clients") if "5 clients" in line else line.index("10 files")
            for line in lines
        }
        self.assertEqual(
            len(columns),
            1,
            "a rail told its labels measures them, so its values line up",
        )


class TheMarkersShareOneFontTest(unittest.TestCase):
    """Every marker is a Dingbat. Mixing a box-drawing rail with an emoji tick
    is how a run's output ends up two weights and two baselines on the same
    line, differently on every terminal."""

    def test_every_marker_is_in_the_dingbats_block(self) -> None:
        for marker in (RAIL, DONE, PENDING, FAIL, WARN, *PULSE):
            self.assertEqual(len(marker), 1, f"{marker!r} is not one code point")
            self.assertTrue(
                0x2700 <= ord(marker) <= 0x27BF,
                f"{marker!r} (U+{ord(marker):04X}, {unicodedata.name(marker)}) is "
                "outside the Dingbats block",
            )

    def test_the_pulse_returns_to_where_it_started(self) -> None:
        """A cycle, not a ramp: ✧ ✦ ❖ ✦ reads as breathing rather than as
        progress towards something."""

        self.assertEqual(PULSE[1], PULSE[3])
        self.assertEqual(len(PULSE), 4)


def _strip(rendered: str) -> str:
    import re

    return re.sub(r"\x1b\[[0-9;]*m", "", rendered)


def _ansi(tone: str) -> str:
    buffer = io.StringIO()
    build_surface(file=buffer, force_rich=True).line("x", tone=tone)
    return buffer.getvalue().split("x")[0]


if __name__ == "__main__":
    unittest.main()
