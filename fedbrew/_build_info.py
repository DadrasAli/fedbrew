"""The commit a release archive was exported from.

A source tarball carries no `.git`, so `capture_code_state` had nothing to ask
and every run made from one recorded a null commit -- the one field that ties a
number back to the code that produced it, missing exactly where it is hardest
to recover by hand. `git archive` rewrites the placeholder below for any file
marked `export-subst` in `.gitattributes`, so the archive carries the answer
the checkout would have given.

In a working tree the placeholder is never rewritten and stays the literal it
is written as, which is how `source_commit()` tells the two apart: a 40-character
hex string is an export, anything else is a checkout that should ask git instead.

Not covered, stated rather than implied: a `tar` of the working tree, and
`python -m build`, which copies files rather than exporting them. Both produce
an unstamped archive, and an unstamped archive reports `git_error` exactly as
before -- absent, not wrong.
"""

from __future__ import annotations

import re

#: Rewritten to the full commit hash by `git archive`. Nothing reads this name
#: except `source_commit()`; it exists to be substituted.
SOURCE_COMMIT = "$Format:%H$"

_FULL_COMMIT = re.compile(r"\A[0-9a-f]{40}\Z")


def source_commit() -> str | None:
    """The commit this archive was exported from, or None in a checkout."""

    return SOURCE_COMMIT if _FULL_COMMIT.match(SOURCE_COMMIT) else None
