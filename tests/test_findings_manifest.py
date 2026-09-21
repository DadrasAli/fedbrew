"""FINDINGS.csv is the audit census as data; nothing else checks it.

The manifest exists because the findings were only ever prose, and prose cannot
be inspected. That makes the file the first thing a reader opens, and a row that
lies -- a severity outside the taxonomy, a hash that resolves to nothing, a path
that opens nothing -- is worse than the prose it replaced, because it looks
machine-checked.

So the checks here are the ones a reader would run by hand: the header is
exactly what FINDINGS.md documents, the census is the audit's 112 rows less the
three removed with a component, at the class split that leaves, `fix_commit` is
empty on every row, and every path named either opens or carries the marker that
says why it does not. The marker is the one escape hatch, so it is guarded too:
a pattern in FINDINGS.md's removals table has to cover each marked path, or the
marker becomes a way to write anything.

The file is allowed to outgrow the census. A finding raised after the audit
closed carries `pass: post` and is counted separately, so the census stays the
number the reports produced, less the documented removals, and does not quietly
drift upward as later work adds rows. That is a second escape hatch and is
guarded the same way: `pass` is two digits or exactly `post`, and nothing else.

One section of FINDINGS.md is checked against something other than the CSV. The
two corrections in *Post-census rows* are documentation errors rather than
defects in `fedbrew`, so they carry no row and no id, which puts them outside
every check above; what is left is one file quoting numbers out of another. So
the quoted values are read out of the note and looked for in the README it names.

`fix_commit` is checked for being empty, not for resolving. The commits it named
are in a development history that is not published, so a check that resolved
them could run only on a disk that still held that history and would skip
everywhere else -- a guard that cannot fail for anyone who clones the
repository. An empty column that FINDINGS.md explains is checkable everywhere.
"""

from __future__ import annotations

import csv
import fnmatch
import re
import unittest
from collections import Counter
from pathlib import Path

import pytest
from docs_sections import section_of, table_after

pytestmark = pytest.mark.fast

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "FINDINGS.csv"
DOCUMENTATION = REPO_ROOT / "FINDINGS.md"

#: The header FINDINGS.md documents, column for column and in order.
HEADER = [
    "id",
    "pass",
    "pass_title",
    "severity",
    "component",
    "location",
    "summary",
    "confidence",
    "fix_commit",
    "regression_test",
    "status",
]

#: The four severity classes the audit filed findings under. There is no fifth.
SEVERITIES = ("wrong-results", "silent-degradation", "fragile", "style")

#: What the audit reports filed, before any row was removed.
AUDITED_ROWS = 112

#: Rows removed from the census together with the component they were about.
#: FINDINGS.md names each under REMOVED_ROWS_HEADING, beside the audited count.
#: Retired: none may come back, for that finding or for a new one.
RETIRED_IDS = ("P01-F08", "P07-F05", "P08-F06")
REMOVED_ROWS_HEADING = "## Rows removed with a component"

#: The census the file carries, and its class split.
EXPECTED_ROWS = AUDITED_ROWS - len(RETIRED_IDS)
EXPECTED_SPLIT = {
    "wrong-results": 9,
    "silent-degradation": 34,
    "fragile": 47,
    "style": 19,
}

#: The `pass` value of a row raised after the audit closed. Such a row is in the
#: file and outside the census: it takes the same columns, the same taxonomy and
#: the same guards, and it is not counted into the census.
POST_CENSUS_PASS = "post"

#: The note whose values are quoted out of two READMEs rather than out of the
#: CSV. Scoping to the heading rather than to the whole file keeps the check off
#: the census's own tables, which quote nothing.
CORRECTIONS_HEADING = "### Two corrections that are not rows"

#: The severity-correction table beside it. Different subject -- a row that the
#: census filed under the wrong class -- and different shape: it quotes ids and
#: severities out of FINDINGS.csv rather than numbers out of a README.
SEVERITY_CORRECTIONS_HEADING = "### Two severities the census got wrong"

#: A location that names code deleted since the audit carries this prefix.
REMOVED = "removed:"

#: Every value `status` may take. `fixed` and `obsolete` close a finding, and
#: empty means there is no record either way. None says how a fix was attributed,
#: because no attribution survives; the section under FIX_COMMIT_HEADING says why.
STATUSES = ("", "fixed", "obsolete")

#: The FINDINGS.md section that says why `fix_commit` is empty on every row.
FIX_COMMIT_HEADING = "## Why `fix_commit` is empty"

#: The heading the status table lives under. Its counts are the file's own and
#: were kept by hand until this guard; see the class docstring below.
COVERAGE_STATUS_ANCHOR = "`status` takes one of three values over"

#: The section that names every open census row, one table row each.
OPEN_ROWS_HEADING = "## What is left, and why"

#: The sentences that state how many census rows are open.
OPEN_ROW_COUNT_STATEMENTS = (
    r"census's own (\d+) open rows",
    r"All \*\*(\d+)\*\* rows still open",
)

#: The sentence that introduces FINDINGS.md's removals table.
REMOVALS_ANCHOR = "The removals, as path patterns:"


def _rows() -> list[dict[str, str]]:
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _census_rows() -> list[dict[str, str]]:
    """The census. Every check on totals is on these, and never on the whole file."""

    return [row for row in _rows() if row["pass"] != POST_CENSUS_PASS]


def _documented_removals() -> list[str]:
    """The path patterns in FINDINGS.md's removals table.

    The table is the only place a `removed:` location is explained, so it is
    read rather than restated here: a marker whose path no pattern covers is an
    undocumented marker.
    """

    table = table_after(DOCUMENTATION.read_text(encoding="utf-8"), REMOVALS_ANCHOR)
    return [
        pattern
        for line in table.splitlines()[2:]
        for pattern in re.findall(r"`([^`]+)`", line.strip("|").split("|")[0])
    ]


class HeaderTest(unittest.TestCase):
    def test_the_file_parses_as_csv_with_exactly_the_declared_header(self) -> None:
        with MANIFEST.open(newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle))
        self.assertEqual(header, HEADER)

    def test_every_row_has_every_column(self) -> None:
        """csv.DictReader pads a short row with None and buries a long one in a key."""

        with MANIFEST.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            next(reader)
            widths = {index: len(row) for index, row in enumerate(reader, start=2)}
        wrong = {line: width for line, width in widths.items() if width != len(HEADER)}
        self.assertEqual(wrong, {}, f"rows whose width is not {len(HEADER)}: {wrong}")

    def test_the_documentation_declares_the_same_header(self) -> None:
        text = DOCUMENTATION.read_text(encoding="utf-8")
        for column in HEADER:
            with self.subTest(column=column):
                self.assertIn(f"| `{column}` |", text)


class CensusTest(unittest.TestCase):
    def test_every_severity_is_one_of_the_four_classes(self) -> None:
        stray = sorted({row["severity"] for row in _rows()} - set(SEVERITIES))
        self.assertEqual(stray, [], f"severities outside the taxonomy: {stray}")

    def test_the_manifest_carries_exactly_the_census(self) -> None:
        """Post-census rows are excluded, so later work cannot inflate the census."""

        self.assertEqual(len(_census_rows()), EXPECTED_ROWS)

    def test_the_class_split_is_the_one_the_reports_produce(self) -> None:
        self.assertEqual(
            Counter(row["severity"] for row in _census_rows()), Counter(EXPECTED_SPLIT)
        )

    def test_every_pass_is_two_digits_or_the_post_census_marker(self) -> None:
        """The marker is the only way out of the census, so it is the only spelling."""

        stray = sorted(
            {row["pass"] for row in _rows()}
            - {POST_CENSUS_PASS}
            - {row["pass"] for row in _rows() if re.fullmatch(r"\d{2}", row["pass"])}
        )
        self.assertEqual(stray, [], f"pass values that are neither NN nor {POST_CENSUS_PASS!r}")

    def test_a_post_census_row_declares_itself_in_its_id_and_its_title(self) -> None:
        """A row outside the census has to say so where a reader will see it."""

        for row in _rows():
            if row["pass"] != POST_CENSUS_PASS:
                continue
            with self.subTest(finding=row["id"]):
                self.assertTrue(row["id"].startswith("POST-F"), row["id"])
                self.assertIn("Post-census", row["pass_title"])

    def test_the_documentation_says_the_census_and_the_file_can_differ(self) -> None:
        """FINDINGS.md says the census and the file can differ; that has to stay true."""

        text = DOCUMENTATION.read_text(encoding="utf-8")
        self.assertIn(POST_CENSUS_PASS, text)
        self.assertIn("POST-F", text)

    def test_the_documented_row_counts_are_the_file_s_own(self) -> None:
        """The one table in FINDINGS.md that is a count rather than a claim.

        Every other number there describes the census and cannot move. These three
        move every time a post-census row lands, and they were kept by hand:
        POST-F02 was added to the CSV and the table said 113 until someone
        noticed. A manifest that exists so counts can be checked should not
        carry an unchecked one.
        """

        text = DOCUMENTATION.read_text(encoding="utf-8")
        rows = _rows()
        post = [row for row in rows if row["pass"] == POST_CENSUS_PASS]
        for label, expected in (
            ("Census (passes 01–13)", len(_census_rows())),
            (f"Post-census (`pass: {POST_CENSUS_PASS}`)", len(post)),
        ):
            with self.subTest(label=label):
                self.assertRegex(
                    text,
                    rf"\| {re.escape(label)} \| {expected} \|",
                    f"FINDINGS.md does not count {label} as {expected}",
                )
        self.assertRegex(
            text,
            rf"\| \*\*File\*\* \| \*\*{len(rows)}\*\* \|",
            f"FINDINGS.md does not give the file's total as {len(rows)}",
        )

    def test_every_post_census_row_is_written_out_in_full(self) -> None:
        """The prose table beside the counts, kept by hand for the same reason.

        The Fix cell is checked against the row rather than searched for in the
        file. An earlier form searched the whole file for the row's values, which
        degenerates to vacuous searches on the empty strings an open row carries,
        so an open row could have claimed any fix it liked in prose. Now a fixed
        row's cell has to say so and name its guard, and an open row's has to say
        `open` and name none.
        """

        text = DOCUMENTATION.read_text(encoding="utf-8")
        for row in _rows():
            if row["pass"] != POST_CENSUS_PASS:
                continue
            with self.subTest(finding=row["id"]):
                marker = f"| `{row['id']}` |"
                self.assertIn(marker, text)
                line = next(line for line in text.splitlines() if line.startswith(marker))
                fix_cell = line.rsplit("|", 2)[-2]
                if row["status"] == "fixed":
                    self.assertNotEqual(row["regression_test"], "")
                    self.assertIn("fixed", fix_cell)
                    self.assertIn(row["regression_test"], fix_cell)
                else:
                    self.assertEqual(row["status"], "")
                    self.assertEqual(row["regression_test"], "")
                    self.assertIn("open", fix_cell)
                    self.assertNotIn("tests/", fix_cell)

    def test_no_id_is_used_twice(self) -> None:
        """The pass number is half the key; dropping it silently merges two findings."""

        ids = [row["id"] for row in _rows()]
        duplicated = sorted({value for value in ids if ids.count(value) > 1})
        self.assertEqual(duplicated, [], f"repeated ids: {duplicated}")


class RetiredRowsTest(unittest.TestCase):
    """Three rows left the census with the component they were about.

    FINDINGS.md says which, and that the audit filed 112. What can drift is an
    id coming back -- re-added from its report, or reused for a new finding --
    or the note losing the count it departs from.
    """

    def test_no_retired_id_is_in_the_manifest(self) -> None:
        back = sorted({row["id"] for row in _rows()} & set(RETIRED_IDS))
        self.assertEqual(back, [], f"retired ids are back in FINDINGS.csv: {back}")

    def test_the_documentation_tables_exactly_the_retired_ids(self) -> None:
        """Against the table's id column, not a search of the section.

        The section's prose names a retired id as well, so a search passed with
        that id's table row deleted. Both directions: a row for an id that is
        not retired is as wrong as a missing one.
        """

        table = table_after(DOCUMENTATION.read_text(encoding="utf-8"), REMOVED_ROWS_HEADING)
        tabled = re.findall(r"^\| `([A-Z0-9]+-F\d{2})` \|", table, flags=re.MULTILINE)
        self.assertEqual(sorted(tabled), sorted(RETIRED_IDS))

    def test_the_documentation_states_both_counts(self) -> None:
        section = section_of(DOCUMENTATION.read_text(encoding="utf-8"), REMOVED_ROWS_HEADING)
        self.assertIn(f"**{AUDITED_ROWS}**", section)
        self.assertIn(f"**{EXPECTED_ROWS}**", section)


class StatusTest(unittest.TestCase):
    """`status` had a vocabulary in FINDINGS.md and no vocabulary in code.

    Nothing checked the column at all: it was documented as three values and
    the file already held a fourth combination the documentation's counts had
    drifted past. So the vocabulary is pinned here, and the counts are read out
    of FINDINGS.md's status table rather than restated, because the counts are
    what drifted.
    """

    def test_every_status_is_one_of_the_documented_values(self) -> None:
        unknown = sorted({row["status"] for row in _rows() if row["status"] not in STATUSES})
        self.assertEqual(unknown, [], f"status values outside the vocabulary: {unknown}")

    def test_the_documented_status_counts_are_the_file_s_own(self) -> None:
        """The same class of hand-kept count the census table already guards.

        Scoped to the census, because that is what every other table here
        counts and what the documentation says it counts. The table has one row
        per value and no more, so a value dropped from the vocabulary cannot
        linger in the documentation.
        """

        counts = Counter(row["status"] for row in _census_rows())
        table = table_after(DOCUMENTATION.read_text(encoding="utf-8"), COVERAGE_STATUS_ANCHOR)
        self.assertEqual(len(table.splitlines()) - 2, len(STATUSES), table)
        for status in STATUSES:
            label = "empty" if status == "" else f"`{status}`"
            with self.subTest(status=label):
                self.assertRegex(
                    table,
                    rf"\| {re.escape(label)} \| {counts[status]} \|",
                    f"FINDINGS.md does not count {label} as {counts[status]}",
                )
        self.assertEqual(sum(counts.values()), EXPECTED_ROWS)


class FixCommitColumnTest(unittest.TestCase):
    """`fix_commit` is empty on every row, and FINDINGS.md says why.

    The hashes it held named commits in a development history that is not
    published, so in any clone they resolve to nothing -- and a hash that
    resolves to nothing reads as checkable while checking nothing, which is
    worse than an empty cell the file explains. The check is that the column
    stays empty, which fails in every checkout the moment a hash comes back.
    """

    def test_the_column_is_empty_on_every_row(self) -> None:
        filled = [f"{row['id']}  {row['fix_commit']}" for row in _rows() if row["fix_commit"]]
        self.assertEqual(filled, [], f"fix_commit must stay empty: {filled}")

    def test_the_documentation_says_why(self) -> None:
        section = section_of(DOCUMENTATION.read_text(encoding="utf-8"), FIX_COMMIT_HEADING)
        self.assertIn("not published", " ".join(section.split()))


class OpenRowsTest(unittest.TestCase):
    """How many census rows are open is stated in prose, and was wrong in one place.

    The post-census section said an open row carries what "the census's own 21
    open rows" carry, when the census had 5. The status table beside it was
    guarded and right; the sentence restating its count was not, and nothing
    noticed. So every sentence that states the count is read here, and so is the
    table that names the open rows.
    """

    def _open_ids(self) -> list[str]:
        return sorted(row["id"] for row in _census_rows() if row["status"] == "")

    def test_every_stated_open_row_count_is_the_census_s_own(self) -> None:
        text = DOCUMENTATION.read_text(encoding="utf-8")
        for statement in OPEN_ROW_COUNT_STATEMENTS:
            with self.subTest(statement=statement):
                stated = re.findall(statement, text)
                self.assertNotEqual(stated, [], f"FINDINGS.md no longer says {statement!r}")
                self.assertEqual({int(value) for value in stated}, {len(self._open_ids())})

    def test_the_open_rows_table_names_exactly_the_open_rows(self) -> None:
        table = table_after(DOCUMENTATION.read_text(encoding="utf-8"), OPEN_ROWS_HEADING)
        tabled = re.findall(r"^\| `([A-Z0-9]+-F\d{2})` \|", table, flags=re.MULTILINE)
        self.assertEqual(sorted(tabled), self._open_ids())


class SeverityCorrectionsTest(unittest.TestCase):
    """The note says which rows were misfiled; the rows must still say so.

    Deliberately not a check that the severity was *changed*: the whole point
    of the note is that the CSV keeps what the audit filed. What can drift is
    the pairing -- a row renumbered, or its severity edited to agree with the
    note, either of which would leave the note describing nothing. Both fail
    here.
    """

    def _rows(self) -> list[tuple[str, str]]:
        """Each correction as (finding id, the severity it says was filed)."""

        table = table_after(DOCUMENTATION.read_text(encoding="utf-8"), SEVERITY_CORRECTIONS_HEADING)
        rows = []
        for line in table.splitlines()[2:]:
            quoted = re.findall(r"`([^`]+)`", line)
            self.assertGreaterEqual(len(quoted), 2, f"a correction row quotes too little: {line}")
            rows.append((quoted[0], quoted[1]))
        self.assertNotEqual(rows, [], "the severity-correction table has no rows")
        return rows

    def test_each_correction_names_a_finding_in_the_manifest(self) -> None:
        ids = {row["id"] for row in _rows()}
        for finding, _ in self._rows():
            with self.subTest(finding=finding):
                self.assertIn(finding, ids)

    def test_the_severity_it_says_was_filed_is_the_one_on_the_row(self) -> None:
        """If a later editor "corrects" the CSV, this is what catches it."""

        by_id = {row["id"]: row for row in _rows()}
        for finding, filed in self._rows():
            with self.subTest(finding=finding):
                self.assertEqual(
                    by_id[finding]["severity"],
                    filed,
                    f"{finding} is no longer {filed} in FINDINGS.csv; the note says it is, "
                    "and the note's whole claim is that the CSV keeps what the audit filed",
                )

    def test_the_note_says_the_csv_is_not_edited(self) -> None:
        text = DOCUMENTATION.read_text(encoding="utf-8")
        self.assertIn("neither `severity` is edited in `FINDINGS.csv`", text)


class CorrectionsNoteTest(unittest.TestCase):
    """The corrections note is prose quoting prose, and only this holds them together.

    Each row names a README and quotes what that README got wrong and what it
    says instead. Nothing else links the two files: re-measure the example,
    reword the correction, and the note is left asserting a number that appears
    nowhere -- the failure the whole manifest exists to make impossible.

    The values are read out of the note rather than restated here, so the note
    stays the record and this stays a check that two files agree.
    """

    def _corrections(self) -> list[tuple[str, list[str]]]:
        """Each row as (the README it names, everything else it quotes)."""

        table = table_after(DOCUMENTATION.read_text(encoding="utf-8"), CORRECTIONS_HEADING)
        rows = []
        for line in table.splitlines()[2:]:
            quoted = re.findall(r"`([^`]+)`", line)
            self.assertNotEqual(quoted, [], f"this corrections row quotes nothing: {line}")
            rows.append((quoted[0], quoted[1:]))
        self.assertNotEqual(rows, [], "the corrections table has no rows")
        return rows

    def test_each_correction_names_a_readme_that_is_in_the_tree(self) -> None:
        for path, _ in self._corrections():
            with self.subTest(path=path):
                self.assertTrue(path.endswith("README.md"), f"not a README: {path}")
                self.assertTrue((REPO_ROOT / path).is_file(), f"no such file: {path}")

    def test_every_value_the_note_quotes_is_still_in_the_readme_it_names(self) -> None:
        for path, values in self._corrections():
            readme = (REPO_ROOT / path).read_text(encoding="utf-8")
            self.assertNotEqual(values, [], f"{path}: the row quotes no value to check")
            for value in values:
                with self.subTest(path=path, value=value):
                    # self.fail rather than assertIn: these READMEs run to tens of
                    # thousands of characters and assertIn prints the haystack.
                    if value not in readme:
                        self.fail(
                            f"FINDINGS.md quotes {value!r} from {path}, which no "
                            "longer contains it. Either the README moved on and the "
                            "note has to follow it, or the note quotes something "
                            "that was never there."
                        )


class LocationTest(unittest.TestCase):
    def test_every_unmarked_location_names_a_path_that_exists(self) -> None:
        missing = [
            f"{row['id']}  {row['location']}"
            for row in _rows()
            if row["location"]
            and not row["location"].startswith(REMOVED)
            and not (REPO_ROOT / row["location"]).exists()
        ]
        self.assertEqual(
            missing,
            [],
            "these name a path that is not in the tree. Either the file moved and "
            "the row should follow it, or the file is gone and the row should say "
            f"so with the {REMOVED!r} marker, documented in FINDINGS.md: {missing}",
        )

    def test_a_marked_location_is_not_secretly_a_live_path(self) -> None:
        """The marker says 'deleted'. On a path that exists it says something false."""

        wrong = [
            f"{row['id']}  {row['location']}"
            for row in _rows()
            if row["location"].startswith(REMOVED)
            and (REPO_ROOT / row["location"][len(REMOVED) :]).exists()
        ]
        self.assertEqual(wrong, [], f"marked removed, but present in the tree: {wrong}")

    def test_every_marked_location_is_explained_in_the_documentation(self) -> None:
        removals = _documented_removals()
        self.assertNotEqual(removals, [], "FINDINGS.md's removals table did not parse")
        undocumented = []
        for row in _rows():
            if not row["location"].startswith(REMOVED):
                continue
            path = row["location"][len(REMOVED) :]
            if not any(
                fnmatch.fnmatch(path, pattern.replace("**", "*")) or path == pattern
                for pattern in removals
            ):
                undocumented.append(f"{row['id']}  {path}")
        self.assertEqual(
            undocumented,
            [],
            "these carry the removed: marker but no row of FINDINGS.md's removals "
            f"table covers them, so nothing explains them: {undocumented}",
        )


class RegressionTestColumnTest(unittest.TestCase):
    def test_every_named_regression_test_exists(self) -> None:
        missing = [
            f"{row['id']}  {path}"
            for row in _rows()
            for path in row["regression_test"].split()
            if not (REPO_ROOT / path).is_file()
        ]
        self.assertEqual(missing, [], f"named guards that are not in tests/: {missing}")

    def test_every_named_regression_test_is_a_test_module(self) -> None:
        stray = [
            f"{row['id']}  {path}"
            for row in _rows()
            for path in row["regression_test"].split()
            if not (path.startswith("tests/") and path.endswith(".py"))
        ]
        self.assertEqual(stray, [], f"not tests/ modules: {stray}")


if __name__ == "__main__":
    unittest.main()
