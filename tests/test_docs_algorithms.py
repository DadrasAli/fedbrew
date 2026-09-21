"""docs/07-algorithms.md must match the registries, pairings and cost notices.

Three kinds of claim live in that chapter and only two can be checked here.

The registry listings and the enforced pairings are data: they are diffed
against register_builtin_components and against the validators that raise, so a
strategy added without a chapter section fails, and a pairing rule that stops
being enforced fails.

The communication multipliers -- SCAFFOLD 2x, FedLALR 3x -- are checked
indirectly: the preflight notices that state them to a user are asserted to
still exist, and tests/test_scaffold_fedprox_communication_cost.py measures the
volume itself. A chapter that disagreed with the notice would be caught; a
chapter that disagreed with reality while the notice also did would not, which
is why that other test exists.

The formulas are prose. They carry a module reference each, and the path check
below keeps those references pointing somewhere real.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import pytest
from docs_sections import fenced_block_after, section_of, table_after

from fedbrew.core import registry
from fedbrew.core.config import PAIRED_STRATEGIES

pytestmark = pytest.mark.fast

REPO_ROOT = Path(__file__).resolve().parent.parent
CHAPTER = REPO_ROOT / "docs" / "07-algorithms.md"

#: (strategy, rule) pairs the chapter says are enforced, and the module whose
#: source must still contain the rule that enforces them.
#: The three matched pairs, as the chapter's table lists them. The authority is
#: PAIRED_STRATEGIES in fedbrew/core/config.py and this is diffed against it,
#: so the chapter cannot list a pairing the code does not enforce or miss one
#: it does.
ENFORCED_PAIRINGS = ("centralized", "fedlalr", "scaffold")


def _chapter_text() -> str:
    return CHAPTER.read_text(encoding="utf-8")


class ChapterShapeTest(unittest.TestCase):
    def test_present_and_agent_facing(self) -> None:
        self.assertTrue(CHAPTER.is_file())
        self.assertIn("\n## For agents\n", _chapter_text())


class RegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        registry.register_builtin_components()
        self.text = _chapter_text()

    def _listing(self) -> dict[str, set[str]]:
        """The block's two entries, keyed by registry.

        Parsed per registry rather than flattened. Four names appear in both
        registries, so a flat set cannot tell `centralized` documented under
        server_strategies from `centralized` documented only under
        client_updates -- and that is exactly the error this block invites.
        """

        start = self.text.index("## 1. The two registries")
        opening = self.text.index("```", start)
        closing = self.text.index("```", opening + 3)
        block = self.text[opening + 3 : closing]

        listing: dict[str, set[str]] = {}
        current: str | None = None
        for line in block.splitlines():
            words = line.split()
            if not words:
                continue
            if words[0] in {"server_strategies", "client_updates"}:
                current = words[0]
                listing[current] = set(words[1:])
            elif current is not None:
                listing[current].update(words)
        return listing

    def test_each_registry_listing_is_that_registry(self) -> None:
        listing = self._listing()
        for attribute in ("server_strategies", "client_updates"):
            with self.subTest(registry=attribute):
                self.assertEqual(
                    listing.get(attribute, set()),
                    set(getattr(registry, attribute).builtin()),
                    f"the chapter's {attribute} line and the registry disagree",
                )

    def test_every_strategy_and_rule_is_mentioned_somewhere(self) -> None:
        """A name in a registry with no prose is a component nobody can use."""

        for label, names in (
            ("strategy", registry.server_strategies.builtin()),
            ("rule", registry.client_updates.builtin()),
        ):
            for name in names:
                with self.subTest(**{label: name}):
                    self.assertIn(
                        f"`{name}`",
                        self.text,
                        f"{name} is registered but the chapter never names it",
                    )


class PairingTest(unittest.TestCase):
    """The three enforced pairings must still be enforced, and still documented."""

    def test_the_chapter_lists_exactly_what_the_code_enforces(self) -> None:
        """Against the table, not against a proximity match in a module.

        This used to grep each module for the strategy name within 400
        characters of "requires", which passed for the wrong reason once the
        scattered raises became one table -- and would have kept passing if a
        pairing had been enforced in only one direction.
        """

        self.assertEqual(sorted(PAIRED_STRATEGIES), sorted(ENFORCED_PAIRINGS))

    def test_the_refusal_is_on_the_run_path(self) -> None:
        """validation.py reports; config.py refuses. The chapter says so."""

        source = (REPO_ROOT / "fedbrew" / "core" / "config.py").read_text(encoding="utf-8")
        self.assertIn("_validate_paired_strategies(config)", source)
        self.assertIn("PAIRED_STRATEGIES", _chapter_text())
        self.assertIn("does not gate", _chapter_text())

    def test_each_pairing_has_a_row(self) -> None:
        text = _chapter_text()
        section = text[text.index("Three algorithms need both halves") : text.index("`fedprox`, ")]
        for strategy in ENFORCED_PAIRINGS:
            with self.subTest(pairing=strategy):
                self.assertIn(f"`{strategy}` ⟷ `{strategy}`", section)

    def test_the_stated_count_matches(self) -> None:
        self.assertEqual(len(ENFORCED_PAIRINGS), 3)
        self.assertIn("Three algorithms need both halves", _chapter_text())


class CommunicationCostTest(unittest.TestCase):
    """The multipliers the chapter quotes must match the preflight notices."""

    def setUp(self) -> None:
        self.validation = (REPO_ROOT / "fedbrew" / "core" / "validation.py").read_text(
            encoding="utf-8"
        )
        self.text = _chapter_text()

    def test_the_multi_state_notices_still_exist(self) -> None:
        for name in (
            "scaffold_communication_cost",
            "fedlalr_communication_cost",
            "communicated_bytes",
        ):
            with self.subTest(notice=name):
                self.assertIn(name, self.validation)

    def test_the_chapter_states_every_multiplier(self) -> None:
        for algorithm, multiplier in (
            ("scaffold", "2"),
            ("fedlalr", "3"),
        ):
            with self.subTest(algorithm=algorithm):
                self.assertRegex(
                    self.text,
                    rf"`{algorithm}` \| \*\*{multiplier}×\*\*",
                    f"section 5 must give {algorithm} a {multiplier}x row",
                )

    def test_the_chapter_repeats_the_preflight_advice(self) -> None:
        """Both the notice and the chapter must say what to compare on."""

        self.assertIn(
            "Compare arms on communicated_bytes, not on round count alone.",
            self.validation,
        )
        self.assertIn("Compare arms on `communicated_bytes`", self.text)


class RefusedOptionTest(unittest.TestCase):
    def test_the_engine_options_match(self) -> None:
        from fedbrew.core.config import ENGINE_CLIENT_OPTIONS

        # Scoped to section 6's fenced block, which holds exactly the option
        # names. The previous form took 900 characters from the heading and
        # split the lot on whitespace, so any word anywhere in that window
        # counted -- and it only checked one direction.
        listed = set(fenced_block_after(_chapter_text(), "## 6. Options a rule refuses").split())
        gated = set(ENGINE_CLIENT_OPTIONS)
        self.assertTrue(gated, "no engine options declared; the import is broken")
        self.assertEqual(
            gated - listed,
            set(),
            f"section 6 omits engine options the factory gates: {sorted(gated - listed)}",
        )
        # The other direction, which docs/00 requires and this guard did not
        # do: a chapter promising that a rule refuses an option the code
        # honours is a promise nothing keeps.
        self.assertEqual(
            listed - gated,
            set(),
            f"section 6 names options the factory does not gate: {sorted(listed - gated)}",
        )

        # The chapter spells the count as a word, as prose does, so both forms
        # are accepted -- but only for the true value. Rewriting whichever word
        # appears into the number under test, which is what
        # section.replace("nine", ...) did, cannot fail: it turned a stale
        # "nine" into the right answer before asserting on it.
        count = len(gated)
        words = (
            "zero one two three four five six seven eight nine ten eleven twelve "
            "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty"
        ).split()
        forms = {str(count)}
        if count < len(words):
            forms |= {words[count], words[count].capitalize()}
        section = section_of(_chapter_text(), "## 6. Options a rule refuses")
        self.assertTrue(
            any(f"{form} engine keys" in section for form in sorted(forms)),
            f"section 6 must say {count} engine keys, which is len(ENGINE_CLIENT_OPTIONS)",
        )

    def test_the_per_rule_table_matches_the_code(self) -> None:
        """Every rule's row names exactly the keys that rule refuses.

        The chapter used to scope this claim to fedprox and scaffold, which is
        what let delta_sgd and fedlalr accept five keys they never
        received. The table is now per rule, so the guard has to be too.
        """

        from fedbrew.core.config import ENGINE_CLIENT_OPTIONS, UNHONOURED_CLIENT_OPTIONS

        registry.register_builtin_components()
        table = table_after(_chapter_text(), "## 6. Options a rule refuses")
        rows = [row for row in table.splitlines()[2:] if row.startswith("|")]
        self.assertTrue(rows, "section 6 has no per-rule rows")

        documented: dict[str, set[str]] = {}
        for row in rows:
            cells = [cell.strip() for cell in row.strip("|").split("|")]
            self.assertEqual(len(cells), 3, f"section 6 row is not three columns: {row}")
            named = set(re.findall(r"`([a-z_]+)`", cells[0]))
            self.assertTrue(named, f"section 6 row names no rule: {row}")
            keys = set(re.findall(r"`([a-z_]+)`", cells[2]))
            for rule in named:
                documented[rule] = keys

        self.assertEqual(
            set(documented),
            set(registry.client_updates.builtin()),
            "section 6 must give every registered update rule a row",
        )
        for rule, keys in sorted(documented.items()):
            with self.subTest(rule=rule):
                _, refused = UNHONOURED_CLIENT_OPTIONS.get(rule, ("", ()))
                self.assertEqual(
                    keys,
                    set(refused),
                    f"section 6's row for {rule} disagrees with UNHONOURED_CLIENT_OPTIONS",
                )
                # A row that names nothing must say so in words, not be blank.
                if not refused:
                    self.assertIn("none", table.lower())
                self.assertEqual(keys - set(ENGINE_CLIENT_OPTIONS), set())

    def test_delta_sgd_still_refuses_amp(self) -> None:
        source = (REPO_ROOT / "fedbrew" / "core" / "config.py").read_text(encoding="utf-8")
        self.assertIn(
            "delta_sgd is incompatible with runtime.use_amp: true",
            source,
        )
        self.assertIn("Incompatible with `runtime.use_amp: true`", _chapter_text())


class CitedPathsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.text = _chapter_text()

    def test_cited_modules_exist(self) -> None:
        cited = set(re.findall(r"`(fedbrew/[\w/]+\.py)", self.text))
        self.assertGreaterEqual(len(cited), 12)
        missing = sorted(path for path in cited if not (REPO_ROOT / path).is_file())
        self.assertEqual(missing, [])

    def test_cited_tests_exist(self) -> None:
        cited = set(re.findall(r"`(tests/test_\w+\.py)`", self.text))
        self.assertGreaterEqual(len(cited), 10)
        missing = sorted(path for path in cited if not (REPO_ROOT / path).is_file())
        self.assertEqual(missing, [])

    def test_cited_configs_exist(self) -> None:
        cited = set(re.findall(r"(configs/[\w/]+\.yaml)", self.text))
        self.assertTrue(cited)
        missing = sorted(path for path in cited if not (REPO_ROOT / path).is_file())
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
