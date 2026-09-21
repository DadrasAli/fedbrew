"""Every docs/ path written anywhere in the tree must exist.

This is the sibling of tests/test_cli_commands_exist.py, for documentation
paths instead of commands, and it exists for the same reason: the tree carried
36 references to three chapters of the deleted documentation set -- its
performance, experiment-checklist and first-experiments pages -- and 34 of
them were in shipped configs, where a user reads them while deciding whether a
setting is safe to change.

Nothing checked, so nothing noticed. A stale documentation pointer is worse
than no pointer: it tells a reader there is an explanation and then does not
provide it, and in a config comment it appears exactly when someone is
deciding whether they may touch the value beside it.

Walks the tree rather than a kept list, so a new file cannot reintroduce the
defect.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

REPO_ROOT = Path(__file__).resolve().parent.parent

SCANNED_SUFFIXES = (".py", ".sh", ".md", ".yaml", ".yml")

#: Kept in step with tests/test_cli_commands_exist.py.
SKIPPED_DIRS = frozenset(
    {
        ".git",
        ".hypothesis",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "AUDIT",
        "__pycache__",
        "build",
        "dist",
        "generated",
        "logs_and_errs",
        "outputs",
        "raw",
        "venv",
    }
)

#: A docs/ path. The trailing bound stops a sentence's full stop being taken
#: as part of the filename.
_DOC_REFERENCE = re.compile(r"\b(docs/[\w./-]*\.md)\b")


def _shipped_files() -> list[Path]:
    found = []
    for path in REPO_ROOT.rglob("*"):
        if path.suffix not in SCANNED_SUFFIXES or not path.is_file():
            continue
        if SKIPPED_DIRS.intersection(path.relative_to(REPO_ROOT).parts):
            continue
        found.append(path)
    return sorted(found)


def _references() -> list[tuple[Path, int, str]]:
    found = []
    for path in _shipped_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), start=1):
            for reference in _DOC_REFERENCE.findall(line):
                found.append((path, number, reference))
    return found


class DocumentationReferenceTests(unittest.TestCase):
    def test_the_scan_reaches_the_tree_it_claims_to_check(self) -> None:
        """A scan-based guard that silently matches nothing always passes."""

        files = _shipped_files()
        self.assertGreater(len(files), 100, "file walk found almost nothing")

        references = _references()
        self.assertGreater(
            len(references),
            30,
            "no documentation references found; the scan is broken",
        )
        parents = {path.relative_to(REPO_ROOT).parts[0] for path, _, _ in references}
        # The configs are where the stale ones hid, so they must be in scope.
        # docs/ is deliberately absent: chapters link each other with relative
        # names (`08-metrics.md`), which is correct and which this regex does
        # not match.
        for expected in ("configs", "fedbrew"):
            with self.subTest(directory=expected):
                self.assertIn(expected, parents)

    def test_every_referenced_chapter_exists(self) -> None:
        offenders = [
            f"{path.relative_to(REPO_ROOT)}:{number}  {reference}"
            for path, number, reference in _references()
            if not (REPO_ROOT / reference).is_file()
        ]
        self.assertEqual(
            offenders,
            [],
            "these point at documentation that does not exist. A stale pointer "
            "is worse than none: it promises an explanation and does not "
            "deliver one, and in a config comment it is read exactly when "
            "someone is deciding whether they may change the value beside it.",
        )

    def test_config_comments_point_at_the_chapter_that_owns_the_claim(self) -> None:
        """The performance block's comments are reproducibility claims.

        They explain that matmul_precision changes results and that three
        neighbouring keys do not. That is chapter 10's subject, and
        test_matmul_precision.py guards the chapter's half of it.
        """

        # By directory, not by suffix: .github/workflows/tests.yml is also
        # YAML and legitimately names the quickstart chapter.
        referenced = {
            reference
            for path, _, reference in _references()
            if path.relative_to(REPO_ROOT).parts[0] == "configs"
            and path.suffix in (".yaml", ".yml")
        }
        self.assertEqual(
            referenced,
            {"docs/10-reproducibility.md"},
            "a shipped config points somewhere other than the reproducibility "
            "chapter; if a new claim needs a different chapter, add it here "
            "deliberately",
        )


#: ``module.symbol`` in backticks, the citation form examples/ uses. Narrowed
#: to symbols that are private, CamelCase or CONSTANT_CASE: those are code
#: names and nothing else, whereas a lowercase word after a dot is far more
#: often a config key -- `divergence.metric`, `model.tolerance` -- and the
#: sections of a config share names with modules (`data`, `client`, `server`).
_SYMBOL_CITATION = re.compile(r"`([a-z][a-z0-9_]*)\.(_\w+|[A-Z]\w*)`")

#: A file:line citation. What this guard exists to forbid in examples/ and
#: docs/ alike.
_LINE_CITATION = re.compile(r"\b[\w./-]+\.py:\d+")

#: A commit hash as prose writes one: seven to forty lowercase hex digits
#: standing alone, with at least one digit and at least one letter among them,
#: so that neither a number nor an English word is taken for one.
_COMMIT_HASH = re.compile(r"(?<![\w.-])(?=[0-9a-f]*[0-9])(?=[0-9a-f]*[a-f])[0-9a-f]{7,40}(?![\w-])")

#: The three shapes a docs/ citation may take, each pairing one symbol with the
#: file it is claimed to live in. Anything looser cannot be checked: a paths
#: table's second cell holds registry names and config keys as often as symbols
#: -- `| `fedbrew/models/torch_mlp.py` | builds `mlp` |` is a true row naming
#: something that file deliberately does not define -- so the table form is
#: matched only when the symbol leads the cell and the path is under fedbrew/.
_SYMBOL = r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
_PATH = r"[\w./-]+\.py"
_SYMBOL_BESIDE_FILE = (
    re.compile(rf"`({_SYMBOL})`\s*\(`({_PATH})`\)"),
    re.compile(rf"\(`({_SYMBOL})`,\s*`({_PATH})`\)"),
    re.compile(rf"^\|\s*`(fedbrew/{_PATH})`\s*\|\s*`({_SYMBOL})`", re.M),
)


def _doc_files() -> list[Path]:
    # rglob, not glob: a chapter filed under a future subdirectory is still a
    # chapter, and a guard that walks only the top level would stop covering it
    # without saying so.
    return sorted((REPO_ROOT / "docs").rglob("*.md"))


def _symbol_citations() -> list[tuple[Path, int, str, str]]:
    """Every (doc, line, symbol, path) pair written in one of the three forms."""

    found = []
    for path in _doc_files():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for index, form in enumerate(_SYMBOL_BESIDE_FILE):
                for match in form.finditer(line):
                    symbol, reference = match.group(1), match.group(2)
                    if index == 2:  # the table form reads path first.
                        symbol, reference = reference, symbol
                    found.append((path, number, symbol, reference))
    return found


def _example_files() -> list[Path]:
    return sorted(
        path
        for path in (REPO_ROOT / "examples").rglob("*")
        if path.suffix in (".md", ".py") and "__pycache__" not in path.parts
    )


def _module_index() -> dict[str, list[Path]]:
    """Module basename -> the files with that name, package and examples alike."""

    index: dict[str, list[Path]] = {}
    roots = (REPO_ROOT / "fedbrew", REPO_ROOT / "examples")
    for root in roots:
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts or path.stem == "__init__":
                continue
            index.setdefault(path.stem, []).append(path)
    return index


def _top_level_names(path: Path) -> set[str]:
    """Every name a module binds at its top level, plus its classes' attributes.

    Attributes matter: the citations here name things like ``_move_batch`` and
    ``_criterion``, which are methods and instance attributes rather than
    module-level definitions.
    """

    import ast

    names: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
            names.add(node.attr)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Attribute):
            names.add(node.target.attr)
    return names


class ExampleCitationsResolveTest(unittest.TestCase):
    """examples/ cites the package by symbol, never by line number.

    Both of the file:line citations this example once carried went stale
    within three commits of being written, and neither failed anything: they
    named real files, so they read as evidence while pointing at code that had
    moved. That is the whole defect -- a decorative citation is worse than
    none, for the same reason the stale chapter pointers above were.

    A line range cannot be checked in a way that would have caught it, either.
    Verifying only that the file is that long passes while the range points at
    something unrelated; verifying that the range *contains* the cited symbol
    means the symbol is doing the work and the range is pure maintenance debt.
    So examples/ names the symbol, and this checks the symbol resolves in the
    module the text says it is in.

    This was scoped to examples/, on the argument that a docs/ chapter carries
    its line citations under a different bargain: each chapter has a guard that
    fails when its subject moves. That bargain did not hold. Of the 177 line
    citations docs/ carried, **33 already pointed somewhere their own sentence
    contradicted** -- 31 naming a symbol their module defines and citing lines
    that do not contain it, and two citing a range past the end of the file.
    Six pointed at unrelated code rather than merely drifting: `factory.py`'s
    `shard_cache_bytes` citation landed in the buffer-averaging refusal, and its
    `client.epsilon` one in the AMP task allow-list. The chapter guards had held
    every claim those sentences made and none of the pointers beside them,
    because a line number is not a claim any guard was checking.

    The drift is not per-citation either. `fedbrew/core/loop.py` grew by about
    120 lines above the cited region, so a whole family moved together: seven
    symbols were off by 114 to 187 lines at once. Nothing about that is
    detectable by reading, which is why it survived.

    So docs/ is swept too, and the replacement form is checked rather than
    trusted -- see :class:`DocsSymbolCitationsResolveTest`.
    """

    def test_no_file_line_citation_appears_in_examples_or_docs(self) -> None:
        offenders = [
            f"{path.relative_to(REPO_ROOT)}:{number}  {match.group(0)}"
            for path in _example_files() + _doc_files()
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
            for match in [_LINE_CITATION.search(line)]
            if match
        ]
        self.assertEqual(
            offenders,
            [],
            f"line-number citations: {offenders}. Name the symbol instead -- "
            "`symbol` (`path/to/module.py`) in docs/, `module._symbol` in "
            "examples/ -- which this file checks and which does not go stale "
            "when the code above it moves.",
        )

    def test_every_cited_symbol_is_defined_in_the_module_named(self) -> None:
        index = _module_index()
        offenders = []
        for path in _example_files():
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                for module, symbol in _SYMBOL_CITATION.findall(line):
                    if module not in index:
                        continue
                    if any(symbol in _top_level_names(target) for target in index[module]):
                        continue
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{number}  {module}.{symbol}")
        self.assertEqual(
            offenders,
            [],
            f"cited symbols that their module does not define: {offenders}",
        )

    def test_the_scan_reaches_the_citations_it_claims_to_check(self) -> None:
        """A scan matching nothing would pass while checking nothing."""

        files = _example_files()
        self.assertGreater(len(files), 2, f"examples/ walk found almost nothing: {files}")

        index = _module_index()
        checked = [
            f"{module}.{symbol}"
            for path in _example_files()
            for module, symbol in _SYMBOL_CITATION.findall(path.read_text(encoding="utf-8"))
            if module in index
        ]
        self.assertGreater(len(checked), 3, f"almost no citation was checked: {checked}")

    def test_it_resolves_into_the_package_and_not_only_the_example(self) -> None:
        """The citations that rot are the ones pointing at code examples/ does
        not own, so at least one must resolve into fedbrew/ itself."""

        index = _module_index()
        into_package = {
            module
            for path in _example_files()
            for module, _ in _SYMBOL_CITATION.findall(path.read_text(encoding="utf-8"))
            if module in index
            and any((REPO_ROOT / "fedbrew") in target.parents for target in index[module])
        }
        self.assertNotEqual(into_package, set())


class DocsSymbolCitationsResolveTest(unittest.TestCase):
    """A symbol docs/ cites beside a file must be defined in that file.

    This is what replaced the line ranges, so it is what has to be checked.
    A line range could not be: verifying only that the file is that long passes
    while the range points at something unrelated, and verifying that the range
    contains the cited symbol makes the symbol do the work and the range pure
    maintenance debt. A symbol beside a path is one assertion, and this is it.

    It reaches further than the sweep did. The forms match every symbol-beside-
    file citation in docs/, including the ones that never carried a line number,
    so chapters are held to more claims than they used to make.

    What it deliberately does not check is a bare symbol with no file beside it.
    Naming the file is what makes the claim checkable, and a citation that
    names only a symbol is asking the reader to guess which of two `config.py`
    it meant -- which docs/04 was doing, in a form nothing could resolve, until
    this found it.
    """

    def test_every_cited_symbol_is_defined_in_the_file_beside_it(self) -> None:
        offenders = []
        for path, number, symbol, reference in _symbol_citations():
            target = REPO_ROOT / reference
            if not target.is_file():
                offenders.append(f"{path.name}:{number}  no such file: {reference}")
                continue
            if symbol.split(".")[-1] not in _top_level_names(target):
                offenders.append(f"{path.name}:{number}  {symbol} is not in {reference}")
        self.assertEqual(
            offenders,
            [],
            f"citations naming something their file does not define: {offenders}",
        )

    def test_the_scan_reaches_the_citations_it_claims_to_check(self) -> None:
        """A scan matching nothing would pass while checking nothing."""

        citations = _symbol_citations()
        self.assertGreater(len(_doc_files()), 10, "the docs/ walk found almost nothing")
        self.assertGreater(len(citations), 100, f"almost nothing was checked: {citations}")
        chapters = {path.name for path, _, _, _ in citations}
        self.assertGreater(
            len(chapters),
            8,
            f"only {sorted(chapters)} were reached; the forms match too narrowly",
        )

    def test_all_three_citation_forms_are_in_use(self) -> None:
        """A form nothing uses is a form nothing would notice breaking."""

        counts = []
        for form in _SYMBOL_BESIDE_FILE:
            matched = sum(
                len(form.findall(path.read_text(encoding="utf-8"))) for path in _doc_files()
            )
            counts.append(matched)
        self.assertTrue(all(counts), f"an unused citation form: {counts}")


def _documentation_files() -> list[Path]:
    """Every Markdown file the tree ships, and the findings manifest beside them."""

    markdown = [path for path in _shipped_files() if path.suffix == ".md"]
    return [*markdown, REPO_ROOT / "FINDINGS.csv"]


class NoCommitHashCitationTest(unittest.TestCase):
    """No document in the tree cites a commit by its hash.

    The line-number ban above, one level up. A hash reads as the most checkable
    citation there is -- paste it into `git show` -- and it is only ever as good
    as the history it was written against. When that history is rewritten or
    left unpublished, every hash in the tree resolves to nothing, and nothing in
    the sentence around it says so. FINDINGS.csv and FINDINGS.md carried more
    than a hundred of them, beside a guard that could resolve them only in the
    one checkout still holding that history, and the quickstart's sample
    `run.json` showed one as its `git_commit` -- a value a reader might well try.

    So a document names what a commit did -- the finding it fixed, the change it
    made -- and a sample value that stands for a hash says it is a placeholder.
    Markdown and the findings manifest only: code comments already cite findings
    by id, and a test fixture may need a hash-shaped string.
    """

    def test_no_document_cites_a_commit_hash(self) -> None:
        offenders = [
            f"{path.relative_to(REPO_ROOT)}:{number}  {match.group(0)}"
            for path in _documentation_files()
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
            for match in _COMMIT_HASH.finditer(line)
        ]
        self.assertEqual(
            offenders,
            [],
            f"commit hashes in documentation: {offenders}. Name the finding id or the "
            "change instead; a hash resolves only against the history it was written in.",
        )

    def test_the_scan_reaches_the_documents_it_claims_to_check(self) -> None:
        """A scan matching nothing would pass while checking nothing."""

        scanned = {path.relative_to(REPO_ROOT).as_posix() for path in _documentation_files()}
        self.assertGreater(len(scanned), 20, f"documentation walk found almost nothing: {scanned}")
        for expected in (
            "README.md",
            "FINDINGS.md",
            "FINDINGS.csv",
            "docs/03-quickstart.md",
            "examples/fed-lasso/README.md",
        ):
            with self.subTest(document=expected):
                self.assertIn(expected, scanned)

    def test_the_pattern_takes_hashes_and_nothing_else(self) -> None:
        """Both lengths git prints, bare or in a URL; not a number, a word or a literal."""

        for text, expected in (
            ("fixed in `1a2b3c4`", ["1a2b3c4"]),
            ("a1" * 20, ["a1" * 20]),
            ("github.com/owner/repo/commit/9f8e7d6", ["9f8e7d6"]),
            ("2026090512 rows", []),
            ("deadbeefcafe", []),
            ("0x7f3a9c1d", []),
            ("7.694928e-05", []),
        ):
            with self.subTest(text=text):
                self.assertEqual(_COMMIT_HASH.findall(text), expected)


class SubdirectoryReadmeTest(unittest.TestCase):
    """The four subdirectory READMEs must point at a chapter, not restate one.

    Each of these describes something no chapter covers -- the submit scripts,
    the two config inventories, the dividing line between tools/ and the
    installed package. That is why they still exist. What they must not do is
    carry a second copy of a chapter's subject, which is how README.md drifted
    and how the deleted docs/ drifted before it.

    The relative link is checked to resolve, because a README one directory
    down needs `../docs/`, and a broken pointer is the failure mode this whole
    file exists to catch.
    """

    #: README -> the chapter it must point at.
    SUBDIRECTORY_READMES = {
        "tools/README.md": "docs/11-performance-and-cost.md",
        "SLURMs/README.md": "docs/02-installation.md",
        "configs/README.md": "docs/04-configuration.md",
        "data/configs/README.md": "docs/05-data-and-partitioning.md",
    }

    def test_each_points_at_its_chapter(self) -> None:
        for readme, chapter in self.SUBDIRECTORY_READMES.items():
            with self.subTest(readme=readme):
                path = REPO_ROOT / readme
                self.assertTrue(path.is_file(), f"{readme} is missing")
                text = path.read_text(encoding="utf-8")
                self.assertIn(
                    Path(chapter).name,
                    text,
                    f"{readme} must point at {chapter}",
                )

    def test_every_relative_docs_link_resolves(self) -> None:
        """A relative docs link from a subdirectory README must resolve.

        configs/README.md is one level down and needs ../docs/;
        data/configs/README.md is two and needs ../../docs/. Getting that
        wrong produces a link that looks right in the source and 404s in a
        browser, which is what this caught on its first run.
        """

        offenders = []
        for readme in self.SUBDIRECTORY_READMES:
            path = REPO_ROOT / readme
            base = path.parent
            for link in re.findall(
                r"\]\(((?:\.\./)+docs/[\w./-]+\.md)\)", path.read_text(encoding="utf-8")
            ):
                if not (base / link).resolve().is_file():
                    offenders.append(f"{readme}  {link}")
        self.assertEqual(
            offenders,
            [],
            f"relative links that do not resolve: {offenders}",
        )

    def test_tools_readme_does_not_restate_the_chapter_table(self) -> None:
        """The script table lives in chapter 11 §7, in one place.

        It was in both, and the two would have had to stay in step: the
        chapter says which of its measured numbers came from which script,
        which is a claim the tools README cannot make.
        """

        text = (REPO_ROOT / "tools" / "README.md").read_text(encoding="utf-8")
        scripts = [path.name for path in (REPO_ROOT / "tools").glob("*.py")]
        listed = [name for name in scripts if name in text]
        self.assertEqual(
            listed,
            [],
            f"tools/README.md lists scripts again: {listed}. Chapter 11 §7 "
            "owns that table and its guard checks it against the directory.",
        )

    def test_the_config_readmes_do_not_restate_the_config_surface(self) -> None:
        """An inventory may name files; it may not become a key reference."""

        from fedbrew.core.config import _KNOWN_EXTRA_KEYS

        sections = {name for name in _KNOWN_EXTRA_KEYS if "." not in name}
        for readme in ("configs/README.md", "data/configs/README.md"):
            with self.subTest(readme=readme):
                text = (REPO_ROOT / readme).read_text(encoding="utf-8")
                dotted = re.findall(r"`((?:\w+\.)+\w+)`", text)
                offenders = sorted(
                    {
                        key
                        for key in dotted
                        if key.split(".")[0] in sections
                        and not key.endswith((".yaml", ".json", ".py", ".md"))
                    }
                )
                self.assertEqual(
                    offenders,
                    [],
                    f"{readme} names config keys: {offenders}. Chapter 04 owns "
                    "them and diffs its tables against the code.",
                )


if __name__ == "__main__":
    unittest.main()
