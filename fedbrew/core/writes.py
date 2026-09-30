"""The file writes a run's flush makes, done now or collected for the writer thread.

A flush formats its CSV rows and serialises its checkpoints -- Python work that
holds the GIL -- and writes, fsyncs and renames files, which release it. On a
thread of its own the Python half only competes with the round loop for the
GIL (a two-client round of the dev MLP took 5.8 ms with its flush on the
writer and 3.8 ms with it inline, measured 2026-09-30), while the I/O half is
what overlaps: an fsync on /proj's NFS is ~7 ms. So the loop does the Python
half and hands the writer the I/O, which these functions describe:

- outside :func:`collected`, each is done at once, as every write outside a
  flush is;
- inside it, each is appended, in order, to the list the block yields, for
  :func:`run_all` to do later -- on the writer thread, which runs them in the
  order they were made.

What a file ends up holding is the same either way: the text or bytes are
fixed when the write is made.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

#: The list the writes of the current thread's :func:`collected` block go to.
_collecting = threading.local()


@contextmanager
def collected() -> Iterator[list[Callable[[], None]]]:
    """Collect this thread's writes, in order, instead of making them."""

    held = getattr(_collecting, "writes", None)
    writes: list[Callable[[], None]] = []
    _collecting.writes = writes
    try:
        yield writes
    finally:
        _collecting.writes = held


def run_all(writes: list[Callable[[], None]]) -> None:
    """Make collected writes, in the order they were collected."""

    for write in writes:
        write()


def _do(write: Callable[[], None]) -> None:
    writes = getattr(_collecting, "writes", None)
    if writes is None:
        write()
    else:
        writes.append(write)


def write_text_atomically(path: Path, text: str) -> None:
    """``text`` to a sibling ``<name>.tmp``, fsynced, then renamed over ``path``.

    The visible file is the previous complete version or the new one, never
    part of either.
    """

    def write() -> None:
        temp_path = path.with_name(path.name + ".tmp")
        try:
            with temp_path.open("w", encoding="utf-8", newline="") as file:
                file.write(text)
                file.flush()
                os.fsync(file.fileno())
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
        os.replace(temp_path, path)

    _do(write)


def append_text(path: Path, text: str) -> None:
    """``text`` appended to ``path`` in one write, then fsynced."""

    def write() -> None:
        with path.open("a", encoding="utf-8", newline="") as file:
            file.write(text)
            file.flush()
            os.fsync(file.fileno())

    _do(write)


def replace(source: Path, target: Path) -> None:
    """``os.replace(source, target)``: a staged file made visible."""

    _do(lambda: os.replace(source, target))
