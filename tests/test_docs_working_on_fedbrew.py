"""docs/14 is method, so only its factual scaffolding can be guarded.

The chapter says that itself, in a section headed "What no test checks", and
this file is the other half of that honesty: it pins everything the chapter
asserts that a machine can verify, and nothing else.

Guardable here:

- every module path and function name section 6 cites still exists, at the
  path it names -- five of the traps were fixed under the tree's *old*
  layout (fl_framework/, clients/, servers/), so every one of those paths was
  rewritten by hand for this chapter and every one of them can rot;
- the fix each trap describes is still in the code, checked by a phrase from
  the code itself rather than by a commit hash: the repository was first
  published as a single commit, so those fixes have no hash to resolve, and
  section 5 says so;
- CONTRIBUTING.md points here and stays a pointer;
- the chapter keeps saying which of its claims are not guarded.

Not guardable, and deliberately not attempted: whether anyone followed any of
it.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

REPO_ROOT = Path(__file__).resolve().parent.parent
CHAPTER = REPO_ROOT / "docs" / "14-working-on-fedbrew.md"
CONTRIBUTING = REPO_ROOT / "CONTRIBUTING.md"

#: Inline `path/like/this.py` anywhere in the chapter.
_MODULE = re.compile(r"`((?:fedbrew|tests|tools|configs|data|examples)/[\w./-]+\.(?:py|yaml|md))`")

#: A `_leading_underscore` or plain function name in backticks. Restricted to
#: the ones section 6 uses as evidence; a general scan would match prose.
CITED_SYMBOLS = {
    "_assign_allowed_labels": "fedbrew/data/partitioners/label_skew.py",
    "_require_label_coverage": "fedbrew/data/partitioners/label_skew.py",
    "synthetic_teacher": "fedbrew/data/synthetic_classification.py",
    "synthetic_labels": "fedbrew/data/synthetic_classification.py",
    "flush_client_csvs": "fedbrew/core/artifacts.py",
    "_load_existing_metric_history": "fedbrew/core/loop.py",
    "_build_checkpoint_payload": "fedbrew/core/loop.py",
    "_restore_server_state": "fedbrew/core/loop.py",
    "client_metric_names": "fedbrew/core/config.py",
    "validate_config": "fedbrew/core/config.py",
    "validate_full_config": "fedbrew/core/validation.py",
    # Section 6.8's seven. The section's claim is that one insertion moved all
    # of them out from under their citations at once, so the names have to keep
    # resolving for the claim to stay checkable -- and a rename that does not
    # reach the chapter is the same defect the section is about.
    "_stream_fit_results": "fedbrew/core/loop.py",
    "_release_client": "fedbrew/core/loop.py",
    "_evaluate_central_test_set": "fedbrew/core/loop.py",
    "_aggregate_client_split_metrics": "fedbrew/core/loop.py",
    "_client_distribution_statistics": "fedbrew/core/loop.py",
    "_refuse_a_foreign_seed": "fedbrew/core/runner.py",
    "_artifact_file_names": "fedbrew/core/runner.py",
    # Section 6.9's: the refusal whose test fixture had gone stale, and the
    # fixture.
    "_require_disjoint_client_windows": "fedbrew/data/hf_causal_lm_text.py",
    "_generator_config": "tests/test_hf_causal_lm_text_generator_offline.py",
}


def _text() -> str:
    return CHAPTER.read_text(encoding="utf-8")


def _fenced_stripped(text: str) -> str:
    return re.sub(r"```.*?```", "", text, flags=re.DOTALL)


class ChapterShapeTest(unittest.TestCase):
    def test_present_and_agent_facing(self) -> None:
        self.assertTrue(CHAPTER.is_file())
        self.assertIn("\n## For agents\n", _text())

    def test_the_traps_section_is_the_longest(self) -> None:
        """The chapter claims section 6 is the point of it. Hold it to that."""

        text = _text()
        sections = re.split(r"\n## ", text)
        by_length = {part.split("\n", 1)[0].strip(): len(part) for part in sections[1:]}
        traps = next(name for name in by_length if name.startswith("6. Recurring traps"))
        longest = max(by_length, key=lambda name: by_length[name])
        self.assertEqual(
            traps,
            longest,
            "section 6 is meant to have the most space; it now has "
            f"{by_length[traps]} characters against {by_length[longest]} for "
            f"{longest!r}",
        )

    def test_the_traps_intro_counts_and_names_every_trap(self) -> None:
        """Section 6.10 arrived under an intro that still said nine entries."""

        from docs_sections import section_of

        text = _text()
        traps = [
            number
            for number, title in re.findall(r"^### 6\.(\d+) (.*)$", text, re.MULTILINE)
            if title != "The shape they share"
        ]
        intro = " ".join(section_of(text, "## 6. Recurring traps").split("\n### ", 1)[0].split())
        words = "zero one two three four five six seven eight nine ten eleven twelve".split()
        self.assertIn(f"{words[len(traps)].capitalize()} entries.", intro)
        # The intro groups the first five and names every trap after them.
        for number in traps[5:]:
            with self.subTest(section=f"6.{number}"):
                self.assertIn(f"§6.{number} ", intro)

    def test_it_says_which_claims_are_not_guarded(self) -> None:
        """A method chapter that implies it is guarded would be the worst case."""

        text = " ".join(_text().split())
        self.assertIn("What no test checks", text)
        self.assertIn("cannot be guarded", text)


class CitedCodeExistsTest(unittest.TestCase):
    """Five of the traps predate the current tree layout."""

    def test_every_module_path_resolves(self) -> None:
        paths = sorted(set(_MODULE.findall(_text())))
        self.assertGreater(len(paths), 8, "the path scan found almost nothing")
        missing = [path for path in paths if not (REPO_ROOT / path).exists()]
        self.assertEqual(missing, [], f"paths cited in docs/14 that do not exist: {missing}")

    def test_every_cited_symbol_is_defined_where_the_chapter_says(self) -> None:
        text = _text()
        for symbol, module in sorted(CITED_SYMBOLS.items()):
            with self.subTest(symbol=symbol):
                self.assertIn(f"`{symbol}`", text, f"the chapter no longer cites {symbol}")
                source = (REPO_ROOT / module).read_text(encoding="utf-8")
                # re.search with MULTILINE, not assertRegex: assertRegex takes
                # no flags, so ^ anchors to the whole file and the failure
                # message prints the entire module. Section 2 of the chapter
                # this guards says so.
                self.assertIsNotNone(
                    re.search(rf"^def {re.escape(symbol)}\(", source, re.MULTILINE),
                    f"{symbol} is not defined in {module}",
                )

    def test_the_shared_scoping_helper_exists_and_is_used(self) -> None:
        """Section 6.6's fix, which the section argues had to be shared.

        The trap is that a correct local fix with its reasoning written beside
        it did not stop the shape spreading. So the check is not that the note
        is still there -- it is that the helper exists and that the guards
        actually reach for it. One importer would satisfy the letter of 6.6
        and none of its point.
        """

        helper = REPO_ROOT / "tests" / "docs_sections.py"
        self.assertTrue(helper.is_file(), "docs/14 section 6.6 names a helper that is gone")
        source = helper.read_text(encoding="utf-8")
        for function in ("section_of", "table_after", "fenced_block_after"):
            with self.subTest(function=function):
                self.assertIn(f"def {function}(", source)

        importers = sorted(
            path.name
            for path in (REPO_ROOT / "tests").glob("test_docs_*.py")
            if "docs_sections import" in path.read_text(encoding="utf-8")
        )
        self.assertGreaterEqual(
            len(importers),
            5,
            "section 6.6 says every docs guard reaches for the shared scoping "
            f"helper; only {importers} do. A helper one guard uses is the local "
            "fix the section is about.",
        )

    def test_the_helper_refuses_a_missing_anchor(self) -> None:
        """The property that keeps a conversion from silently loosening a check."""

        from docs_sections import SectionNotFound, section_of, table_after

        with self.assertRaises(SectionNotFound):
            section_of("## Real\n\nbody\n", "## Absent")
        with self.assertRaises(SectionNotFound):
            table_after("## Real\n\nbody\n", "## Absent")

    def test_the_deliberate_strict_false_is_still_deliberate(self) -> None:
        """Section 6.1's whole point is a comment that has to stay attached."""

        source = (REPO_ROOT / "fedbrew/data/partitioners/label_skew.py").read_text(encoding="utf-8")
        self.assertIn("strict=False", source)
        self.assertIn("synthetic_label_skew", source)
        self.assertIn(
            "strict=False, deliberately",
            source,
            "the reason was removed from beside the call; docs/14 section 6.1 "
            "argues from the fact that it is written there",
        )

    def test_the_label_skew_fixture_is_still_the_slack_case(self) -> None:
        """3 labels against 10 slots is the number section 6.1 quotes."""

        config = (REPO_ROOT / "data/configs/synthetic_label_skew.yaml").read_text(encoding="utf-8")
        values = {
            key: int(match)
            for key in ("num_classes", "num_clients", "labels_per_client")
            for match in re.findall(rf"^\s*{key}:\s*(\d+)", config, re.MULTILINE)[:1]
        }
        self.assertEqual(values["num_classes"], 3)
        self.assertEqual(values["num_clients"] * values["labels_per_client"], 10)


class TheDescribedFixIsStillInPlaceTest(unittest.TestCase):
    """What replaced the commit hashes.

    Section 6 used to cite a seven-character hash per trap. The repository was
    first published as a single orphan commit, so those resolved to nothing,
    and the guard that checked them checked nothing.

    The replacement is better than what it replaces. A hash proved a commit
    once existed; this proves the fix the chapter describes is still in the
    code. Each entry is a phrase the fix's own source carries -- a comment, a
    docstring line, a call -- so reverting the fix fails here even though the
    chapter still reads correctly.
    """

    #: section -> (module, phrase from that module, what the phrase is)
    FIXES = {
        "6.1": (
            "fedbrew/data/partitioners/label_skew.py",
            "strict=False, deliberately",
            "the deliberate exception and its invariant",
        ),
        "6.2": (
            "fedbrew/data/synthetic_classification.py",
            "mean(dim=0",
            "the teacher's columns centred to sum to zero",
        ),
        "6.3": (
            "fedbrew/core/artifacts.py",
            "Throttling the rewrite instead is not an option",
            "the rejected alternative, recorded at the function",
        ),
        "6.4": (
            "fedbrew/core/logging.py",
            "client_metric_names(",
            "the legend asking the emitter instead of restating it",
        ),
        "6.5": (
            "fedbrew/core/loop.py",
            "resume from freshly initialised weights",
            "the refusal that the deduplication made necessary",
        ),
        "6.6": (
            "tests/docs_sections.py",
            "**Every function raises rather than returning empty.**",
            "the reason the scoping helper raises instead of returning empty",
        ),
        # Not a fix in fedbrew/: the trap is a claim, and what replaced it is a
        # retraction. The phrase is the README heading the chapter names, so
        # rewording the retraction away fails here.
        "6.7": (
            "examples/simplex-lsq/README.md",
            "### A withdrawn claim about thread count",
            "the retraction, under the heading the chapter names",
        ),
        # Also not a fix in fedbrew/: the trap was a form of citation, and what
        # replaced it is a guard. The phrase is the class that checks the
        # replacement form, so deleting the check fails here even though every
        # chapter still reads correctly -- which is the trap itself.
        "6.8": (
            "tests/test_docs_references_resolve.py",
            "class DocsSymbolCitationsResolveTest",
            "the guard that checks the symbol citations that replaced the ranges",
        ),
        # A test fixture, not a fix in fedbrew/: the code was right and the
        # tests were stale. The phrase is the reason the stride sits above
        # sequence_length rather than at it, so a later edit back to the
        # default spelling -- which passes every assertion -- fails here.
        "6.9": (
            "tests/test_hf_causal_lm_text_generator_offline.py",
            "not tell a configured stride from a defaulted one",
            "the fixture's stride above sequence_length, with the reason beside it",
        ),
        # Not a fix in fedbrew/: the trap is a job script's exit status. What the
        # tree carries of the lesson is the packed example's closing loop, whose
        # comment says why the job's status cannot be left to the last background
        # process.
        "6.10": (
            "SLURMs/example_sweep_packed.sh",
            "exit code would be the last background process's",
            "the wait loop that makes a pack's status every run's",
        ),
    }

    def test_each_trap_s_fix_is_still_there(self) -> None:
        for section, (module, phrase, what) in sorted(self.FIXES.items()):
            with self.subTest(section=section):
                source = (REPO_ROOT / module).read_text(encoding="utf-8")
                self.assertIn(
                    phrase,
                    source,
                    f"docs/14 section {section} describes {what} in {module}, "
                    "and it is no longer there",
                )

    def test_the_chapter_still_describes_each_one(self) -> None:
        """The other direction: a trap deleted from the chapter."""

        text = _text()
        for section in sorted(self.FIXES):
            with self.subTest(section=section):
                self.assertIn(f"### {section} ", text)

    def test_no_commit_hash_is_cited(self) -> None:
        """The fixes predate the first published commit; their hashes resolve to nothing."""

        bare = _fenced_stripped(_text())
        hashes = re.findall(r"`([0-9a-f]{7,40})`", bare)
        self.assertEqual(
            hashes,
            [],
            f"docs/14 cites commit hashes again: {hashes}. The repository was "
            "first published as a single commit; cite the module and the symbol.",
        )


class ContributingIsAPointerTest(unittest.TestCase):
    """Two locations for one rule is how the previous docs/ drifted."""

    def test_it_exists_and_points_at_the_chapter(self) -> None:
        self.assertTrue(CONTRIBUTING.is_file())
        text = CONTRIBUTING.read_text(encoding="utf-8")
        self.assertIn("docs/14-working-on-fedbrew.md", text)
        self.assertIn("docs/00-index.md", text)

    def test_it_stays_thin(self) -> None:
        lines = CONTRIBUTING.read_text(encoding="utf-8").splitlines()
        self.assertLess(
            len(lines),
            60,
            "CONTRIBUTING.md is growing into a second copy of chapter 14. Put "
            "the material in the chapter, where a guard checks it.",
        )

    def test_it_does_not_restate_the_traps(self) -> None:
        text = CONTRIBUTING.read_text(encoding="utf-8")
        for heading in ("6.1", "strict=True", "bottom10", "flush_client_csvs"):
            with self.subTest(term=heading):
                self.assertNotIn(heading, text)


if __name__ == "__main__":
    unittest.main()
