"""docs/04-configuration.md must list the keys the code actually accepts.

The configuration surface is the largest thing the documentation set
describes and the one a reader is least able to check by eye: roughly 150
keys, most of them living in an ``extra`` dict rather than as a named
dataclass field. A key table written by hand goes stale in one commit.

So the tables are checked against the code that defines them. The direction
matters and is bidirectional: a key the code accepts and the chapter omits
leaves a reader unable to discover it, and a key the chapter names and the
code rejects sends them to write a config that will not load. Both fail here.

What this cannot check is prose -- that a default is described correctly, that
an interaction is stated. Those claims carry a source reference in the chapter
instead, and the module-existence check below keeps those references pointing
at files that exist.
"""

from __future__ import annotations

import dataclasses
import re
import unittest
from pathlib import Path

import pytest
from docs_sections import table_after

from fedbrew.core import registry
from fedbrew.core.config import (
    _KNOWN_EXTRA_KEYS,
    _REMOVED_KEYS,
    DEFAULTS_KEYS,
    EVALUATION_MODEL_SCOPES,
    FROZEN_GRADIENT_WEIGHTINGS,
    MATMUL_PRECISIONS,
    SUPPORTED_AGGREGATION_WEIGHTING,
    UPDATE_MODES,
    ClientStatisticsConfig,
    DivergenceConfig,
)

pytestmark = pytest.mark.fast

REPO_ROOT = Path(__file__).resolve().parent.parent
CHAPTER = REPO_ROOT / "docs" / "04-configuration.md"


def _chapter_text() -> str:
    return CHAPTER.read_text(encoding="utf-8")


#: A fenced block. Stripped before tokenising, because a ``` fence is itself
#: three backticks and would shift every inline-code pairing after it -- the
#: first version of this helper silently returned whole paragraphs as "tokens"
#: and the coverage check below passed by matching nothing.
_FENCE = re.compile(r"```.*?```", re.DOTALL)


def _backticked(text: str) -> set[str]:
    """Every `token` the chapter writes as inline code."""

    return set(re.findall(r"`([^`\n]+)`", _FENCE.sub("", text)))


def _section(number: str) -> str:
    """Just the body of ``## <number>. ...``, up to the next ``## ``.

    The removed-key check below reads this rather than the whole chapter. It
    used to read the whole chapter, and a removed key was therefore "listed"
    if its name appeared anywhere at all -- including §11, which names the
    key a CLI flag writes. Both relocated keys passed that way while §10
    omitted them and §4 and §5 still called them required.
    """

    match = re.search(
        rf"^## {re.escape(number)}\..*?(?=^## |\Z)",
        _chapter_text(),
        re.DOTALL | re.MULTILINE,
    )
    if match is None:
        raise AssertionError(f"docs/04-configuration.md has no section {number}")
    return match.group(0)


def _default_cell(section: str, key: str) -> str | None:
    """The Default column of ``section``'s row for ``key``.

    Split by hand rather than by one regex: the Type column writes its
    alternatives as ``int \\| `final` \\| `never```, so an escaped pipe sits
    inside a cell and a naive column split lands in the wrong place.
    """

    for line in section.splitlines():
        if not line.startswith(f"| `{key}` |"):
            continue
        cells = [cell.strip() for cell in line.replace("\\|", "\x00").split("|")]
        if len(cells) < 4:
            return None
        return cells[3].strip("`").strip('"')
    return None


class ChapterShapeTest(unittest.TestCase):
    def test_the_chapter_is_present_and_agent_facing(self) -> None:
        self.assertTrue(CHAPTER.is_file())
        self.assertIn("\n## For agents\n", _chapter_text())


class ExtraKeyCoverageTest(unittest.TestCase):
    """Every key a section's `extra` accepts must appear in the chapter."""

    def setUp(self) -> None:
        self.tokens = _backticked(_chapter_text())

    def test_every_config_block_has_a_row(self) -> None:
        """Section 2's table against the blocks a config may actually write.

        Nothing checked this: deleting `evaluation`'s row went unnoticed.
        Section 2 is the map a reader uses to find out what a config may
        contain, so a block the loader accepts and the table omits is missing
        exactly where someone goes looking for it. Both directions, because a
        row for a block that no longer exists is the same defect inverted.

        Derived from the surface a config author types, not from
        ``FullConfig``. The two differ in both directions, and this test
        checked the wrong one: ``task`` is a ``FullConfig`` field that a
        config may not write -- it is inferred from ``model.name`` -- and was
        listed here as a **required** block, while ``defaults`` is required by
        the loader, absent from ``FullConfig`` because it is consumed at load,
        and was therefore invisible to this check and documented nowhere.
        """

        import dataclasses

        from fedbrew.core.config import _LOAD_TIME_BLOCKS, _REMOVED_BLOCKS, FullConfig

        table = table_after(_chapter_text(), "## 2. The blocks")
        listed = {
            row.strip("| ").split("|")[0].strip().strip("`")
            for row in table.splitlines()[2:]
            if row.strip().startswith("|")
        }
        listed.discard("")
        self.assertTrue(listed, "no blocks parsed from section 2's table")
        writable = (
            {field.name for field in dataclasses.fields(FullConfig)} - set(_REMOVED_BLOCKS)
        ) | set(_LOAD_TIME_BLOCKS)
        self.assertEqual(
            writable - listed,
            set(),
            f"section 2's table omits blocks a config may write: {sorted(writable - listed)}",
        )
        self.assertEqual(
            listed - writable,
            set(),
            f"section 2's table names blocks a config may not write: {sorted(listed - writable)}",
        )

    def test_no_removed_block_is_offered_as_a_writable_one(self) -> None:
        """A removed block named in section 2 reads as an instruction.

        ``task`` sat there marked **required** while the loader refused it
        outright, because the refusal was a bare ``if`` in
        ``_reject_restated_keys`` that no table could be diffed against.
        """

        from fedbrew.core.config import _REMOVED_BLOCKS

        section_ten = _section("10")
        for block in _REMOVED_BLOCKS:
            with self.subTest(block=block):
                self.assertIn(
                    block,
                    section_ten,
                    f"the {block} block is refused at load; section 10 must say so",
                )

    def test_every_accepted_extra_key_is_documented(self) -> None:
        missing: list[str] = []
        for section, keys in _KNOWN_EXTRA_KEYS.items():
            for key in keys:
                # Either spelling counts: the bare key, or the dotted form the
                # chapter uses when the same name lives in two sections.
                if key in self.tokens or f"{section}.{key}" in self.tokens:
                    continue
                missing.append(f"{section}.{key}")
        self.assertEqual(
            sorted(missing),
            [],
            "docs/04-configuration.md does not mention these accepted keys; a "
            "key nobody can discover is as good as absent",
        )

    def test_the_dataloader_block_is_exactly_four_keys(self) -> None:
        """The chapter states the number, so the number has to be right."""

        keys = _KNOWN_EXTRA_KEYS["runtime.performance.dataloader"]
        self.assertEqual(len(keys), 4)
        self.assertIn("Exactly four keys", _chapter_text())

    def test_sections_that_accept_nothing_are_not_given_keys(self) -> None:
        """A section with an empty allow-list must not be shown taking options."""

        for section in ("experiment", "data", "client_statistics", "divergence"):
            with self.subTest(section=section):
                self.assertEqual(_KNOWN_EXTRA_KEYS[section], frozenset())


class DataclassDefaultsTest(unittest.TestCase):
    """Defaults the chapter prints must be the dataclass's own."""

    def setUp(self) -> None:
        self.text = _chapter_text()

    def _defaults(self, cls: type) -> dict[str, object]:
        values = {}
        for field in dataclasses.fields(cls):
            if field.default is not dataclasses.MISSING:
                values[field.name] = field.default
        return values

    def test_client_statistics_defaults_match(self) -> None:
        for name, value in self._defaults(ClientStatisticsConfig).items():
            if name == "extra":
                continue
            with self.subTest(key=name):
                row = re.search(
                    rf"^\| `client_statistics\.{re.escape(name)}` \| `([^`]+)` \|$",
                    self.text,
                    flags=re.MULTILINE,
                )
                self.assertIsNotNone(row, f"client_statistics.{name} has no default row")
                assert row is not None
                self.assertEqual(
                    row.group(1),
                    str(value).lower() if isinstance(value, bool) else str(value),
                )

    def test_evaluation_defaults_match(self) -> None:
        """Section 8's per-split defaults against the factories that set them.

        These were hand-written beside two guarded siblings. They happened to
        be right, which is the only reason it was not already a defect: the
        real defaults live in EvaluationConfig's per-split default_factory
        calls -- 10/participating, 5/all, 10/all, 10 -- and nothing tied the
        table to them.

        Read from a default-constructed EvaluationConfig rather than by
        parsing the factories, so the values are the ones load_config
        actually produces.
        """

        from fedbrew.core.config import EvaluationConfig

        evaluation = EvaluationConfig()
        expected: dict[str, object] = {"model_scope": evaluation.model_scope}
        for split in ("train", "val", "test"):
            block = getattr(evaluation, split)
            expected[f"{split}.every"] = block.every
            expected[f"{split}.clients"] = block.clients
        expected["central_test.every"] = evaluation.central_test.every

        section_eight = _section("8")
        for key, value in expected.items():
            with self.subTest(key=key):
                documented = _default_cell(section_eight, key)
                self.assertIsNotNone(row_missing := documented, f"evaluation.{key} has no row")
                del row_missing
                self.assertEqual(
                    documented,
                    str(value),
                    f"section 8 says evaluation.{key} defaults to {documented}, "
                    f"EvaluationConfig produces {value}",
                )

    def test_the_split_dataclasses_declare_no_shadowed_default(self) -> None:
        """A field default that no code path can produce is a wrong answer.

        ``SplitEvaluationConfig.every`` read ``= 1`` and ``clients`` read
        ``= "all"``, while every split reaching ``load_config`` comes from
        ``EvaluationConfig``'s factories -- 10/participating, 5/all, 10/all --
        and ``CentralTestConfig.every`` read ``= 1`` against a factory of 10.
        Unreachable, so nothing failed; but a reader of the source who found
        the field first had the wrong number, and the chapter and the
        dataclass disagreed with no way to tell which was current.

        The three splits do not share a default, so there is no value these
        fields could carry that would be right. They carry none.
        """

        from fedbrew.core.config import CentralTestConfig, SplitEvaluationConfig

        for cls, required in (
            (SplitEvaluationConfig, {"every", "clients"}),
            (CentralTestConfig, {"every"}),
        ):
            for field_ in dataclasses.fields(cls):
                if field_.name not in required:
                    continue
                with self.subTest(cls=cls.__name__, field=field_.name):
                    self.assertIs(
                        field_.default,
                        dataclasses.MISSING,
                        f"{cls.__name__}.{field_.name} carries a default "
                        f"({field_.default!r}) that EvaluationConfig overrides "
                        "for every split, so nothing can produce it",
                    )
                    self.assertIs(field_.default_factory, dataclasses.MISSING)

    def test_divergence_defaults_match(self) -> None:
        expected = {
            "metric": '"fit_loss"',
            "non_finite": "true",
            "blowup_factor": "10.0",
            "blowup_absolute": "null",
            "patience": "null",
            "min_delta": "0.0",
        }
        actual = self._defaults(DivergenceConfig)
        # The mapping above is the chapter's YAML spelling of the dataclass
        # defaults; check it against the dataclass so it cannot drift.
        self.assertEqual(actual["metric"], "fit_loss")
        self.assertIs(actual["non_finite"], True)
        self.assertEqual(actual["blowup_factor"], 10.0)
        self.assertIsNone(actual["blowup_absolute"])
        self.assertIsNone(actual["patience"])
        self.assertEqual(actual["min_delta"], 0.0)
        for name, rendered in expected.items():
            with self.subTest(key=name):
                # re.MULTILINE, and not assertRegex, which cannot pass flags --
                # without it ^ and $ anchor to the whole file and every row
                # "fails" while dumping the chapter into the failure message.
                self.assertIsNotNone(
                    re.search(
                        rf"^\| `divergence\.{re.escape(name)}` \| "
                        rf"`{re.escape(rendered)}` \|$",
                        self.text,
                        flags=re.MULTILINE,
                    ),
                    f"divergence.{name} is not documented as {rendered}",
                )


class EnumTest(unittest.TestCase):
    """Every enumerated value the chapter offers must be a legal one."""

    def setUp(self) -> None:
        self.tokens = _backticked(_chapter_text())

    def test_enumerations_are_complete(self) -> None:
        for label, values in (
            ("MATMUL_PRECISIONS", MATMUL_PRECISIONS),
            ("UPDATE_MODES", UPDATE_MODES),
            ("FROZEN_GRADIENT_WEIGHTINGS", FROZEN_GRADIENT_WEIGHTINGS),
            ("EVALUATION_MODEL_SCOPES", EVALUATION_MODEL_SCOPES),
            ("SUPPORTED_AGGREGATION_WEIGHTING", SUPPORTED_AGGREGATION_WEIGHTING),
        ):
            for value in values:
                with self.subTest(enum=label, value=value):
                    self.assertIn(
                        value,
                        self.tokens,
                        f"{label} value {value!r} is not offered by the chapter",
                    )


class DefaultsTableTest(unittest.TestCase):
    """§2.1's key table is exactly the keys the defaults block accepts.

    Nothing checked it. The block is read at load and never stored, so the
    field- and extra-based checks above cannot see it, and when its schedule
    key was renamed no guard noticed the table still offering the old name.
    """

    def test_the_table_lists_exactly_the_accepted_keys(self) -> None:
        table = table_after(_chapter_text(), "### 2.1 `defaults`")
        listed = {
            row.strip("| ").split("|")[0].strip().strip("`")
            for row in table.splitlines()[2:]
            if row.strip().startswith("|")
        }
        self.assertEqual(listed, set(DEFAULTS_KEYS))


class WhatOneIterationIsTest(unittest.TestCase):
    """§2.1 defines one local iteration under every update_mode.

    `local_iterations` counts iterations of the local loop and `update_mode`
    decides what one is -- a step, a pass, a frozen-gradient pass. A mode the
    table does not define leaves a reader to infer how much work the count
    buys, which is what the key's old name, `local_epochs`, got wrong. A new
    mode must add its row here.
    """

    def test_every_update_mode_has_a_row_in_the_definition_table(self) -> None:
        match = re.search(r"^### 2\.1 .*?(?=^##)", _chapter_text(), re.DOTALL | re.MULTILINE)
        self.assertIsNotNone(match, "docs/04-configuration.md has no section 2.1")
        rows = [line for line in match.group(0).splitlines() if line.startswith("| `")]
        for mode in sorted(UPDATE_MODES):
            with self.subTest(update_mode=mode):
                self.assertTrue(
                    any(row.startswith(f"| `{mode}` |") for row in rows),
                    f"section 2.1 does not say what one iteration is under {mode}",
                )


class RegistryNamesTest(unittest.TestCase):
    """The strategy and update-rule lists must be the registered ones."""

    def setUp(self) -> None:
        registry.register_builtin_components()
        self.text = _chapter_text()

    def _fenced_after(self, heading: str) -> set[str]:
        start = self.text.index(heading)
        opening = self.text.index("```", start)
        closing = self.text.index("```", opening + 3)
        return set(self.text[opening + 3 : closing].split())

    def test_server_strategies_match_the_registry(self) -> None:
        self.assertEqual(
            self._fenced_after("Registered strategies"),
            set(registry.server_strategies.builtin()),
        )

    def test_client_update_rules_match_the_registry(self) -> None:
        self.assertEqual(
            self._fenced_after("Registered update rules"),
            set(registry.client_updates.builtin()),
        )


class RemovedKeysTest(unittest.TestCase):
    def test_every_removed_key_is_listed(self) -> None:
        section_ten = _section("10")
        missing = [
            f"{section}.{name}"
            for section, name in _REMOVED_KEYS
            if f"{section}.{name}" not in section_ten
        ]
        self.assertEqual(
            missing,
            [],
            "docs/04-configuration.md section 10 must list every removed key; "
            "a reader with an old config meets exactly these errors",
        )

    def test_no_removed_key_is_offered_as_a_settable_one(self) -> None:
        """The other direction, and the one that was missing.

        A removed key named in a block's key table reads as an instruction to
        write it. Both relocated keys sat in §4 and §5 marked **required** --
        the first rows a reader copies -- and every check passed, because the
        dataclass fields still exist and the removal was refused in a raise
        that ``_REMOVED_KEYS`` did not know about.
        """

        blocks = {
            "server": "4",
            "client": "5",
            "model": "6",
            "data": "7",
        }
        offenders = []
        for (section, name), _ in _REMOVED_KEYS.items():
            number = blocks.get(section)
            if number is None:
                continue
            rows = [
                line
                for line in _section(number).splitlines()
                if line.startswith("|") and f"`{name}`" in line
            ]
            offenders.extend(f"{section}.{name}  ->  {row.strip()}" for row in rows)
        self.assertEqual(
            offenders,
            [],
            f"removed keys offered as settable in their block's table: {offenders}",
        )


class CitedPathsTest(unittest.TestCase):
    def test_cited_modules_exist(self) -> None:
        cited = set(re.findall(r"`(fedbrew/[\w/]+\.py)", _chapter_text()))
        self.assertTrue(cited)
        missing = sorted(path for path in cited if not (REPO_ROOT / path).is_file())
        self.assertEqual(missing, [])

    def test_cited_tests_exist(self) -> None:
        cited = set(re.findall(r"`(tests/test_\w+\.py)`", _chapter_text()))
        self.assertTrue(cited)
        missing = sorted(path for path in cited if not (REPO_ROOT / path).is_file())
        self.assertEqual(missing, [])

    def test_cited_configs_exist(self) -> None:
        cited = set(re.findall(r"`(configs/[\w/]+\.yaml)`", _chapter_text()))
        self.assertTrue(cited)
        missing = sorted(path for path in cited if not (REPO_ROOT / path).is_file())
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
