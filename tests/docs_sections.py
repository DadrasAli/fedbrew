"""Scope a documentation check to the part of the chapter it is about.

Five guards in this suite have now been found unable to fail because they
checked a table by searching the whole chapter. "Every dataloader key is
documented with its gate" was `assertIn(f"`{key}`", chapter_text)`, and every
one of those keys also appears in the prose above the table and in the
`## For agents` tables below it -- so deleting the row a reader actually reads
left the guard green. The same held for chapter 2's core-dependency row and
chapter 9's artifact list.

tests/test_docs_artifacts.py had already found this once and fixed it locally,
with a `_key_table` helper and a comment explaining exactly the trap. It did
not stop the shape spreading to another check in the same file. A local fix
does not generalise, so the scoping lives here and every docs guard reaches
for it.

**Every function raises rather than returning empty.** That is the whole point.
A scoping helper that returns "" for a heading that has been renamed converts a
guard that could fail into one that cannot, silently, which is the defect this
module exists to remove rather than relocate. A missing anchor is a broken
guard and has to say so.
"""

from __future__ import annotations

import re


class SectionNotFound(AssertionError):
    """An anchor a guard scopes to is gone. The guard is broken, not the chapter."""


def _require(condition: object, message: str) -> None:
    if not condition:
        raise SectionNotFound(message)


def section_of(text: str, heading: str) -> str:
    """The body under `heading`, to the next heading at the same or higher level.

    `heading` is the heading line as written, with its hashes:
    `section_of(chapter, "### 4.1 `dataloader`")`. Matching is exact and
    anchored to the start of a line, so a renamed section raises instead of
    quietly scoping to nothing.
    """

    _require(heading.startswith("#"), f"pass the heading with its hashes, not {heading!r}")
    level = len(heading) - len(heading.lstrip("#"))
    lines = text.splitlines()
    starts = [index for index, line in enumerate(lines) if line.rstrip() == heading.rstrip()]
    _require(starts, f"no heading {heading!r} in this chapter")
    _require(
        len(starts) == 1, f"heading {heading!r} appears {len(starts)} times; scope is ambiguous"
    )

    start = starts[0]
    boundary = re.compile(rf"^#{{1,{level}}}\s")
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if boundary.match(lines[index]):
            end = index
            break
    body = "\n".join(lines[start + 1 : end]).strip()
    _require(body, f"section {heading!r} is empty")
    return body


def table_after(text: str, anchor: str) -> str:
    """The first Markdown table following `anchor`, header row included.

    `anchor` is any literal that appears once before the table -- a heading, or
    the sentence that introduces it. The table ends at the first line that is
    not a table row, so the prose after it is out of scope.
    """

    count = text.count(anchor)
    _require(count, f"no anchor {anchor!r} in this chapter")
    _require(count == 1, f"anchor {anchor!r} appears {count} times; scope is ambiguous")

    lines = text[text.index(anchor) :].splitlines()
    rows: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|"):
            rows.append(stripped)
        elif rows:
            break
    _require(rows, f"no table follows {anchor!r}")
    _require(len(rows) > 2, f"the table after {anchor!r} has no body rows")
    return "\n".join(rows)


def fenced_block_after(text: str, anchor: str) -> str:
    """The first ``` fenced block following `anchor`, fences excluded.

    Scoping by index rather than by a character count: chapter 6's injected-key
    check took 400 characters from its anchor, which reached past the block and
    into the sentence after it, so a key moved out of the block but mentioned
    nearby still passed.
    """

    count = text.count(anchor)
    _require(count, f"no anchor {anchor!r} in this chapter")
    _require(count == 1, f"anchor {anchor!r} appears {count} times; scope is ambiguous")

    rest = text[text.index(anchor) :]
    opening = rest.find("```")
    _require(opening != -1, f"no fenced block follows {anchor!r}")
    after_open = rest.index("\n", opening) + 1
    closing = rest.find("```", after_open)
    _require(closing != -1, f"the fenced block after {anchor!r} is never closed")
    body = rest[after_open:closing].strip()
    _require(body, f"the fenced block after {anchor!r} is empty")
    return body
