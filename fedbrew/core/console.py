"""Everything visual: the palette, the marker set, the row builder, the
section rule, and the TTY gate.

This is the only module in the package permitted to import rich, name a
colour, or emit an escape sequence. Every other module composes *content* --
labels, values, tones picked from the constants below -- and hands it here.
tests/test_console_is_the_only_renderer.py enforces that for every module
under fedbrew/, in the same shape as tests/test_optional_dependency_imports.py:
a second place that knows how to paint a terminal is how a palette becomes
six palettes, and it is invisible in review because a markup string looks like
any other string.

Three properties this module exists to keep true:

- **One palette.** Gold for values, ivory for the active label, dim for
  completed ones, faint for structure, amber for warnings, red for errors.
  Amber is deliberately scarce -- it marks a value that changes numerics or
  was defaulted rather than chosen -- and it stays scarce only while one
  module owns it.
- **One font.** The markers and the rail glyph are all Dingbats (U+2717 to
  U+2762), so they render from a single font at a single weight rather than
  mixing a box-drawing rail with an emoji tick.
- **One gate.** Whether the destination is a terminal is asked once, here,
  through `sys.stdout.isatty()`. A redirected stream -- a SLURM log, a pipe
  into grep -- gets the same information as flat lines: no redraw, no colour,
  no cursor movement. Nothing else in the package decides this for itself.

Values are never passed to rich as markup. They are appended to a `Text` with
an explicit style, because a value can be a filesystem path and rich would
read `outputs/run[1]/` as a style tag and swallow it.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, TextIO

# --------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------
#: 256-colour names rather than the eight standard ones: gold and amber are
#: the distinction the palette rests on (a defaulted value must not read as an
#: ordinary one) and both collapse to "yellow" in an 8-colour system. rich
#: downgrades these itself on a terminal that cannot show them.
GOLD = "gold3"
"""Values. Every measured number, path and resolved identifier."""

IVORY = "cornsilk1"
"""The label of the row currently being worked on."""

DIM = "grey58"
"""The label of a row that has settled."""

FAINT = "grey35"
"""Structure: rules, the rail, separators, continuation text."""

AMBER = "orange3"
"""Warnings, and any value that changes numerics or was defaulted rather than
chosen. Scarce on purpose -- see the module docstring."""

RED = "red3"
"""Errors."""

# --------------------------------------------------------------------------
# The run surface's second palette
# --------------------------------------------------------------------------
#: Truecolor rather than 256-colour names, because these eight are read
#: against each other rather than against the terminal's default: a round
#: block puts four splits in one column and four group headings down the page,
#: and the distinctions are between neighbouring hues rather than between a
#: value and its label. rich downgrades them on a terminal that cannot show
#: them, exactly as it does the named five above.
#:
#: The train split is the palette's existing GOLD rather than a paler ninth
#: colour: a second gold beside the plan header's would read as a mistake.
SPLIT_VAL = "rgb(150,185,220)"
"""The validation split. Soft blue -- the split model selection may look at."""

SPLIT_TEST = "rgb(150,200,160)"
"""The test split, measured per client. Sage."""

SPLIT_CENTRAL = "rgb(196,172,220)"
"""The server's own held-out set. Lavender: not a client measurement at all."""

GROUP_LOSS = "rgb(232,168,124)"
"""The loss group heading. Warm amber -- distinct from AMBER, which means
"this value was defaulted"; a group heading is not a value."""

GROUP_ACCURACY = "rgb(150,200,160)"
"""The accuracy group heading. Sage."""

GROUP_SPREAD = "rgb(196,172,220)"
"""The client-spread group heading. Lavender."""

GROUP_ALGORITHM = "rgb(150,185,220)"
"""The algorithm-state group heading. Soft blue."""

UNCLASSIFIED = "rgb(220,120,120)"
"""A column the name-to-group mapping did not recognise.

Deliberately loud. The mapping replaced sixteen hand-written labels with a
derivation, and a derivation that quietly files an unknown name under a
plausible group is worse than the table it replaced: the column still prints,
so nothing looks wrong, and it prints under a heading that is not true of it.
tests/test_round_block.py fails on any column the real vocabulary can produce
that lands here."""

# --------------------------------------------------------------------------
# Markers
# --------------------------------------------------------------------------
RAIL = "❙"
"""U+2759 MEDIUM VERTICAL BAR -- the rail every staged surface hangs from."""

DONE = "❖"
"""U+2756 BLACK DIAMOND MINUS WHITE X -- a stage that finished."""

PENDING = "✧"
"""U+2727 WHITE FOUR POINTED STAR -- a stage that has not run yet."""

WARN = "❢"
"""U+2762 HEAVY EXCLAMATION MARK ORNAMENT -- a stage that warned."""

FAIL = "✗"
"""U+2717 BALLOT X -- a stage that failed."""

PULSE = ("✧", "✦", "❖", "✦")
"""The running stage's marker cycle: ✧ ✦ ❖ ✦. Advanced by Stage.tick(), so a
stage pulses only while it has something to report -- a bounded loop over a
known count -- and a blocking call with no internal signal shows a still
marker instead of a spinner that promises progress it cannot see."""

SELECTED = "✦"
"""U+2726 BLACK FOUR POINTED STAR -- marks the one column a run's
checkpointing selects its best model on. Also the bright phase of PULSE,
which is the point: the two mean the same thing in different tenses."""

#: The progress bar. Box-drawing rather than Dingbats, which the marker set
#: above is drawn from: the bar is structure, like the section rule, and it is
#: read as a continuous line rather than as a glyph. A heavy/light pair from
#: one block is what makes filled and unfilled the same line at two weights.
BAR_FILLED = "━"
"""U+2501 BOX DRAWINGS HEAVY HORIZONTAL -- rounds completed."""

BAR_EMPTY = "─"
"""U+2500 BOX DRAWINGS LIGHT HORIZONTAL -- rounds remaining."""

#: Fallback label column width for a rail whose stage labels are not known up
#: front. A rail told its labels measures them instead.
DEFAULT_LABEL_WIDTH = 28

#: The round block's three columns: split label, qualifier, value. Fixed
#: rather than measured, unlike `Row`: a block is reprinted every evaluation
#: round and a column that moved when --verbose added a wider name would make
#: two rounds of the same run impossible to compare by eye.
SPLIT_COLUMN_WIDTH = 11
QUALIFIER_COLUMN_WIDTH = 22
VALUE_COLUMN_WIDTH = 8

#: Terminal width assumed when the stream cannot report one.
_FALLBACK_WIDTH = 100


class Verbosity(IntEnum):
    """How much a surface says.

    QUIET is the final line only, VERBOSE is every key and every round, and
    NORMAL is the curated middle every command defaults to.
    """

    QUIET = 0
    NORMAL = 1
    VERBOSE = 2


@dataclass(frozen=True)
class Row:
    """One line of a measured block: an optional marker, a label, a value.

    The label column is measured across the whole block rather than fixed per
    row, and the rich and plain paths measure it identically, so a redirected
    run and an interactive one line up column-for-column and differ only in
    colour.
    """

    label: str
    value: str = ""
    #: None means the terminal's own foreground. Prose -- a metric's
    #: plain-language gloss -- is not a value and is not painted gold.
    tone: str | None = GOLD
    label_tone: str = DIM
    marker: str | None = None
    marker_tone: str = FAINT
    note: str | None = None
    note_tone: str = FAINT


@dataclass(frozen=True)
class MetricRow:
    """One line of a round block: which split, which statistic, what number.

    Three fixed columns rather than `Row`'s measured two. The split label is
    coloured and the qualifier is not, because the split is the thing a reader
    scans down the column for and the qualifier is the thing they read once
    they have found it.
    """

    split: str
    qualifier: str
    value: str
    #: The split's own hue. Unrecognised names arrive here as UNCLASSIFIED.
    tone: str = GOLD
    #: Whether checkpointing selects the run's best model on this column.
    selected: bool = False


@dataclass
class Stage:
    """One rail stage, handed to the caller for the duration of its body."""

    label: str
    _rail: Rail
    _pulse: int = 0
    _last_tick: float = field(default=0.0, repr=False)
    _settled: bool = False

    def tick(self, note: str | None = None) -> None:
        """Advance the pulse, at most ten times a second.

        Throttled here rather than at every call site: the natural caller is a
        loop over tens of thousands of items (the inventory's 24,239-message
        tokenise pass), and a redraw per item costs more than the work.
        """

        now = time.perf_counter()
        if now - self._last_tick < 0.1:
            return
        self._last_tick = now
        self._pulse += 1
        self._rail._redraw_running(self, note)

    def note(self, text: str) -> None:
        """Replace the running line's trailing note without advancing the pulse."""

        self._rail._redraw_running(self, text)

    def done(self, value: str = "", *, tone: str = GOLD, note: str | None = None) -> None:
        """Settle this stage as finished, carrying what it learned."""

        self._settle(DONE, value, tone=tone, note=note)

    def warn(self, value: str = "", *, note: str | None = None) -> None:
        """Settle this stage as finished-with-a-caveat. Amber, and it counts."""

        self._settle(WARN, value, tone=AMBER, note=note, warned=True)

    def fail(self, value: str = "", *, note: str | None = None) -> None:
        """Settle this stage as failed. The rail stops after this line."""

        self._settle(FAIL, value, tone=RED, note=note, failed=True)

    def _settle(
        self,
        marker: str,
        value: str,
        *,
        tone: str,
        note: str | None,
        warned: bool = False,
        failed: bool = False,
    ) -> None:
        if self._settled:
            return
        self._settled = True
        self._rail._settle(self, marker, value, tone=tone, note=note)
        if warned:
            self._rail.warnings += 1
        if failed:
            self._rail.failures += 1
            self._rail.stopped = True


class Surface:
    """A destination for output, and the decisions that go with it.

    Built once per command (or per call, for the config-driven functions in
    fedbrew.core.logging that predate this) and passed down. Holds the
    verbosity, the rich console when there is one, and nothing else: a surface
    knows how to draw a row, not what belongs in one.
    """

    def __init__(
        self,
        *,
        verbosity: Verbosity = Verbosity.NORMAL,
        file: TextIO | None = None,
        console: Any | None = None,
    ) -> None:
        self.verbosity = verbosity
        # Deliberately not resolved to sys.stdout here. contextlib's
        # redirect_stdout swaps the module attribute, and a surface built
        # before the redirect must still land inside it -- which is how every
        # test in this suite captures output. rich's Console does the same
        # with file=None, so the two agree.
        self._file = file
        self._console = console

    # -- destination ------------------------------------------------------

    @property
    def stream(self) -> TextIO:
        return self._file if self._file is not None else sys.stdout

    @property
    def is_rich(self) -> bool:
        return self._console is not None

    @property
    def is_tty(self) -> bool:
        """Whether the destination is an interactive terminal.

        The single gate. A false answer means flat lines: no redraw, no
        colour, no cursor movement -- what a log file should hold.

        rich's console is the authority when there is one: its own is_terminal
        already folds in isatty() plus the FORCE_COLOR/TERM overrides, and two
        answers to this question is how a redraw ends up in a log file.
        """

        if self._console is not None:
            return bool(self._console.is_terminal)
        try:
            return bool(self.stream.isatty())
        except (AttributeError, ValueError):
            return False

    @property
    def width(self) -> int:
        if self._console is not None:
            return int(self._console.width)
        return _FALLBACK_WIDTH

    def flush(self) -> None:
        flush = getattr(self.stream, "flush", None)
        if callable(flush):
            flush()

    # -- primitives -------------------------------------------------------

    def blank(self) -> None:
        if self.verbosity is Verbosity.QUIET:
            return
        self._write("")

    def rule(self, title: str, *, tone: str = IVORY) -> None:
        """A section rule: the title, and structure either side of it."""

        if self.verbosity is Verbosity.QUIET:
            return
        if self._console is not None:
            from rich.text import Text

            self._console.rule(Text(title, style=tone), style=FAINT)
            self.flush()
            return
        # ASCII in the plain path on purpose: the rule carries no information
        # beyond its title, so a log file gets the cheapest legible form.
        padding = max(0, 72 - len(title) - 4)
        self._write(f"-- {title} {'-' * padding}")

    def line(
        self,
        text: str,
        *,
        tone: str | None = None,
        marker: str | None = None,
        marker_tone: str = FAINT,
        prefix: str = "",
        wrap: bool = False,
    ) -> None:
        """One free-standing line, with no label column."""

        if self.verbosity is Verbosity.QUIET:
            return
        self._emit(
            [(prefix, FAINT)] if prefix else [],
            [(marker + " ", marker_tone)] if marker else [],
            [(text, tone)],
            wrap=wrap,
        )

    def spans(self, parts: Sequence[tuple[str, str | None]], *, wrap: bool = False) -> None:
        """One line assembled from differently toned pieces.

        For the summary lines that are a run of `label: value` pairs -- a
        round's clients/examples/time/ETA -- where four separate rows would
        cost four lines per round and say the same thing.
        """

        if self.verbosity is Verbosity.QUIET:
            return
        self._emit(parts, wrap=wrap)

    def row(
        self,
        row: Row,
        *,
        width: int | None = None,
        prefix: str = "",
        wrap: bool = False,
    ) -> None:
        """One measured row. `width` is the block's label column, when known."""

        if self.verbosity is Verbosity.QUIET:
            return
        self._write_row(row, width if width is not None else len(row.label), prefix, wrap)

    def rows(self, rows: Sequence[Row], *, prefix: str = "", wrap: bool = False) -> None:
        """A measured block: the label column is the widest label in it.

        `wrap` is for blocks whose value is a sentence rather than a number --
        a metric's gloss -- where cropping at the terminal edge would lose the
        half of the sentence that distinguishes it from its neighbour.
        """

        if self.verbosity is Verbosity.QUIET or not rows:
            return
        width = measure(rows)
        for row in rows:
            self._write_row(row, width, prefix, wrap)

    def final(self, text: str, *, tone: str | None = None) -> None:
        """The one line `--quiet` does not suppress.

        Quiet means "tell me how it went, not how it is going", not "tell me
        nothing": a sweep that prints nothing at all is indistinguishable from
        a sweep whose jobs never started. Every other method on this surface
        returns early under QUIET; this one does not, which is why it is a
        separate method rather than a flag on `line`.
        """

        self._emit([(text, tone)])

    def redraw(self, text: str, *, tone: str | None = None) -> None:
        """Redraw one line in place. A no-op's worth of care: on anything but
        a terminal this prints nothing at all, because a carriage-return
        redraw in a log file is one line per update rather than one per
        result."""

        if self.verbosity is Verbosity.QUIET or not self.is_tty:
            return
        if self._console is not None:
            from rich.text import Text

            self._console.print(Text(text, style=tone or ""), end="\r", highlight=False)
        else:
            print(text, end="\r", file=self.stream, flush=True)
        self.flush()

    def heading(self, text: str, *, tone: str, prefix: str = "  ") -> None:
        """A group heading inside a round block: the group's name, its hue."""

        if self.verbosity is Verbosity.QUIET:
            return
        self._emit([(prefix, None)] if prefix else [], [(text, tone)])

    def metric_rows(self, rows: Sequence[MetricRow], *, prefix: str = "    ") -> None:
        """A round block's rows: split, qualifier, value, selection mark.

        The value is right-aligned so that a column of losses and a column of
        percentages both end on the same edge -- the digits are what is being
        compared between rounds, and a left-aligned number moves its own
        decimal point as the integer part grows.
        """

        if self.verbosity is Verbosity.QUIET or not rows:
            return
        for row in rows:
            self._emit(
                [(prefix, None)] if prefix else [],
                [(row.split.ljust(SPLIT_COLUMN_WIDTH), row.tone)],
                [(row.qualifier.ljust(QUALIFIER_COLUMN_WIDTH), FAINT)],
                [(row.value.rjust(VALUE_COLUMN_WIDTH), IVORY)],
                [(f" {SELECTED}", GOLD)] if row.selected else [],
            )

    def bar(self, fraction: float, width: int) -> list[tuple[str, str]]:
        """The spans of a progress bar, for `spans` to draw beside a tail.

        Returned rather than printed because the bar never appears alone: it
        opens a header line or a live footer, and assembling those here would
        put the run surface's sentence structure in the renderer.
        """

        width = max(0, width)
        filled = min(width, max(0, int(round(width * _clamped(fraction)))))
        return [(BAR_FILLED * filled, GOLD), (BAR_EMPTY * (width - filled), FAINT)]

    def redraw_spans(self, parts: Sequence[tuple[str, str | None]]) -> None:
        """`redraw`, for a line made of differently toned pieces.

        The live footer is a gold bar, a faint remainder and a dim tail on one
        line; `redraw` takes a single tone, and painting the bar would mean
        assembling escape sequences outside this module.
        """

        if self.verbosity is Verbosity.QUIET or not self.is_tty:
            return
        if self._console is not None:
            from rich.text import Text

            text = Text(no_wrap=True, overflow="ignore")
            for content, tone in parts:
                text.append(content, style=tone or "")
            self._console.print(text, end="\r", highlight=False, soft_wrap=True)
        else:
            print(
                "".join(content for content, _ in parts),
                end="\r",
                file=self.stream,
                flush=True,
            )
        self.flush()

    def rail(self, labels: Sequence[str] = ()) -> Rail:
        """Open a rail. Pass the stage labels when they are known up front and
        the label column is measured from them."""

        return Rail(self, labels)

    # -- rendering --------------------------------------------------------

    def _write_row(self, row: Row, width: int, prefix: str, wrap: bool = False) -> None:
        marker = f"{row.marker} " if row.marker else ""
        # Padded only when something follows it. A label with no value is a
        # heading, and trailing spaces on it would show up in a diff of two
        # captured logs as a difference that is not one.
        label = row.label.ljust(width) if row.value or row.note else row.label
        indent = len(prefix) + len(marker) + width
        values = self._wrapped(row.value, indent) if wrap else [row.value]
        self._emit(
            [(prefix, FAINT)] if prefix else [],
            [(marker, row.marker_tone)] if marker else [],
            [(label, row.label_tone)],
            [(f"  {values[0]}", row.tone)] if row.value else [],
            [(f"  {row.note}", row.note_tone)] if row.note else [],
        )
        # Continuation lines hang under the value column. Left at column zero
        # they read as new rows, and the block stops being a table at exactly
        # the rows whose text was long enough to need reading carefully.
        for continued in values[1:]:
            self._emit(
                [(prefix, FAINT)] if prefix else [],
                [(" " * (len(marker) + width), None)],
                [(f"  {continued}", row.tone)],
            )

    def _wrapped(self, value: str, indent: int) -> list[str]:
        """`value` split to fit beside a label column `indent` wide.

        Only on a terminal. A redirected stream keeps every value on one line
        however long it is: the point of the plain path is that a line can be
        grepped, and a gloss broken across two lines matches neither half.
        """

        if not value or not self.is_tty:
            return [value]
        available = self.width - indent - 2
        if available < 24 or len(value) <= available:
            return [value]
        # Neither long words nor hyphens are break points: the long values in
        # this package are filesystem paths and metric names, and a path split
        # across "…-FL-\ncodebase/" cannot be copied out of the terminal or
        # matched by eye against the one in the config.
        return textwrap.wrap(
            value,
            width=available,
            break_long_words=False,
            break_on_hyphens=False,
        ) or [value]

    def _emit(self, *parts: Sequence[tuple[str, str | None]], wrap: bool = False) -> None:
        segments = [segment for part in parts for segment in part]
        if self._console is not None:
            from rich.text import Text

            text = Text(no_wrap=not wrap, overflow=None if wrap else "ignore")
            for content, tone in segments:
                text.append(content, style=tone or "")
            # soft_wrap=True is rich's "leave my line alone" -- no wrapping,
            # no cropping, no padding to the terminal width. Off only for the
            # blocks that asked to wrap.
            self._console.print(text, soft_wrap=not wrap)
            self.flush()
            return
        self._write("".join(content for content, _ in segments))

    def _write(self, text: str) -> None:
        print(text, file=self.stream, flush=True)


class Rail:
    """Checks or stages, streamed one per line, hanging from a single rail.

    Used where a command has a bounded list of things to do and the reader
    wants to watch them settle: `--validate-only`'s checks, and the staged
    data commands. Deliberately not used for the round loop -- 500 rounds is
    not a list.
    """

    def __init__(self, surface: Surface, labels: Sequence[str] = ()) -> None:
        self.surface = surface
        self.width = max((len(label) for label in labels), default=DEFAULT_LABEL_WIDTH)
        self.warnings = 0
        self.failures = 0
        #: Set once a stage fails. Callers check it rather than catching: a
        #: failing check stops the rail, and nothing after it is claimed.
        self.stopped = False
        self._running: Stage | None = None

    @contextmanager
    def stage(self, label: str) -> Iterator[Stage]:
        """Run one stage, settling it as done unless the body says otherwise.

        An exception settles the stage as failed and propagates: the rail
        reports what it knows, then gets out of the way of the real error.
        """

        stage = Stage(label=label, _rail=self)
        self._running = stage
        self._redraw_running(stage, None)
        try:
            yield stage
        except BaseException:
            stage.fail("failed")
            raise
        else:
            if not stage._settled:
                stage.done()
        finally:
            self._running = None

    def result(
        self,
        label: str,
        value: str = "",
        *,
        tone: str = GOLD,
        note: str | None = None,
    ) -> None:
        """A settled line for work already done. No pending state, no
        animation -- what a stage measured in single-digit milliseconds
        deserves."""

        self._line(DONE, label, value, tone=tone, note=note, label_tone=DIM)

    def detail(self, label: str, value: str = "", *, note: str | None = None) -> None:
        """A result line that only --verbose shows.

        The rule everywhere else is that a line appears only if it carries
        information the reader does not already have from typing the command.
        This is where the lines that fail that test go rather than being
        deleted: a resolved cache root or a seed is exactly what someone
        debugging a wrong asset wants, and exactly what everyone else does not.
        """

        if self.surface.verbosity is Verbosity.VERBOSE:
            self.result(label, value, note=note)

    def warn(self, label: str, value: str = "", *, note: str | None = None) -> None:
        self.warnings += 1
        self._line(WARN, label, value, tone=AMBER, note=note, label_tone=DIM)

    def fail(self, label: str, value: str = "", *, note: str | None = None) -> None:
        self.failures += 1
        self.stopped = True
        self._line(FAIL, label, value, tone=RED, note=note, label_tone=DIM)

    # -- internals used by Stage -----------------------------------------

    def _redraw_running(self, stage: Stage, note: str | None) -> None:
        """The running stage's line: ivory label, pulsing marker.

        On a terminal this is redrawn in place, so a stage occupies one line
        from start to finish. Anywhere else it prints nothing -- the settled
        line below carries the same information, once.
        """

        if not self.surface.is_tty:
            return
        marker = PULSE[stage._pulse % len(PULSE)]
        label = stage.label.ljust(self.width)
        text = f"{RAIL} {marker} {label}"
        if note:
            text = f"{text}  {note}"
        self.surface.redraw(text.ljust(self.surface.width - 1)[: self.surface.width - 1])

    def _settle(
        self,
        stage: Stage,
        marker: str,
        value: str,
        *,
        tone: str,
        note: str | None,
    ) -> None:
        self._line(marker, stage.label, value, tone=tone, note=note, label_tone=DIM)

    def _line(
        self,
        marker: str,
        label: str,
        value: str,
        *,
        tone: str,
        note: str | None,
        label_tone: str,
    ) -> None:
        marker_tone = {WARN: AMBER, FAIL: RED}.get(marker, FAINT)
        self.surface.row(
            Row(
                label=label,
                value=value,
                tone=tone,
                label_tone=label_tone,
                marker=marker,
                marker_tone=marker_tone,
                note=note,
            ),
            width=self.width,
            prefix=f"{RAIL} ",
        )


def _clamped(fraction: float) -> float:
    """`fraction` into [0, 1]. A resumed run can report a round past its total."""

    return min(1.0, max(0.0, float(fraction)))


def measure(rows: Sequence[Row]) -> int:
    """The label column width for a block: the widest label in it."""

    return max((len(row.label) for row in rows), default=0)


def silent_rail() -> Rail:
    """A rail that renders nothing.

    The null object for a library function that reports its stages: every
    method still works and still counts, and nothing reaches a terminal. It is
    what lets `prepare_oasst1_assets(path)` keep its old signature and its old
    silence for every caller that does not pass a real rail.
    """

    return build_surface(quiet=True).rail()


def add_output_arguments(parser: argparse.ArgumentParser) -> None:
    """Add --quiet/--verbose/--no-rich to a subcommand's parser.

    One definition for every command rather than one per command. A flag that
    works on `fedbrew run` and not on `fedbrew generate` is worse than no flag:
    the reader who learned it once now has to remember which half of the CLI
    it belongs to, and the failure is an argparse error at the moment they are
    already busy.
    """

    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "--quiet",
        action="store_true",
        help="Report only the final outcome line.",
    )
    verbosity.add_argument(
        "--verbose",
        action="store_true",
        help="Report every stage and every column, not the ones that earn a line.",
    )
    parser.add_argument(
        "--no-rich",
        action="store_true",
        help="Disable colour and redrawing; print flat lines.",
    )


def surface_from_args(args: argparse.Namespace, file: TextIO | None = None) -> Surface:
    """The surface the flags above ask for.

    getattr with defaults throughout: a command that has not adopted the flags
    still gets a working surface, which is what lets them be added one command
    at a time without a flag day.
    """

    return build_surface(
        quiet=bool(getattr(args, "quiet", False)),
        verbose=bool(getattr(args, "verbose", False)),
        no_rich=bool(getattr(args, "no_rich", False)),
        file=file,
    )


def build_surface(
    *,
    quiet: bool = False,
    verbose: bool = False,
    no_rich: bool = False,
    file: TextIO | None = None,
    force_rich: bool | None = None,
) -> Surface:
    """Build the surface a command should write to.

    The rich path is taken only for an interactive terminal that did not ask
    for plain output. A redirected stream takes the plain path -- same rows,
    same measured columns, no colour and no redraw -- which is both what a log
    file wants and what makes `grep` on a SLURM log work.

    `force_rich` overrides the gate in both directions, for tests that need to
    exercise a path their own captured stdout would otherwise decide for them.
    Forcing it on also forces terminal-ness: rich emits no colour into a
    non-terminal, so "render as if interactive" has to say both.
    """

    verbosity = Verbosity.QUIET if quiet else (Verbosity.VERBOSE if verbose else Verbosity.NORMAL)
    console = None
    if _wants_rich(no_rich, file, force_rich):
        console = _build_console(file, force_terminal=force_rich is True)
    return Surface(verbosity=verbosity, file=file, console=console)


def _wants_rich(no_rich: bool, file: TextIO | None, force_rich: bool | None) -> bool:
    if force_rich is not None:
        return force_rich
    if no_rich:
        return False
    stream = file if file is not None else sys.stdout
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):
        return False


def _build_console(file: TextIO | None, *, force_terminal: bool = False) -> Any | None:
    """A rich Console, or None when rich is not installed.

    rich is a core dependency, so None is not the expected path -- but the
    package degrades to plain output rather than failing to import, which is
    the same contract every optional dependency here has.

    NO_COLOR is not read here: rich's own Console honours it, along with
    TERM=dumb and FORCE_COLOR, and reimplementing that would mean two answers
    to one question.
    """

    try:
        from rich.console import Console
    except ModuleNotFoundError:  # pragma: no cover - plain fallback for minimal envs.
        return None
    if force_terminal:
        return Console(file=file, highlight=False, force_terminal=True, color_system="256")
    return Console(file=file, highlight=False)


__all__ = [
    "AMBER",
    "add_output_arguments",
    "DIM",
    "DONE",
    "FAIL",
    "FAINT",
    "GOLD",
    "IVORY",
    "PENDING",
    "PULSE",
    "RAIL",
    "RED",
    "Rail",
    "Row",
    "Stage",
    "Surface",
    "Verbosity",
    "WARN",
    "build_surface",
    "measure",
    "silent_rail",
    "surface_from_args",
]
