"""A field that must be numeric says why a number arrived as a string.

`server.tau: 1e-08` is a string. PyYAML's safe loader implements YAML 1.1,
whose float resolver needs a decimal point **and a signed exponent**, so
`1e-08`, `1e-8`, `1E-08` and `1.0e8` all resolve to `str` while `1.0e-08` and
`1.e-8` resolve to `float`. The loader then refuses the non-numeric value,
which is correct -- and said nothing about the cause.

On 2026-09-20 five sweep arms (`fedadagrad`, `fedadam`, `fedyogi`, `fedlalr`,
`local_adamw`) died 11-15 seconds into scheduled SLURM jobs, before a single
point ran, each on one `1e-08`. The refusal named the field and the diagnosis
still took twenty minutes, because "must be numeric" reads as "you wrote a
word there" and the file plainly contains a number.

What is pinned here is the *cause*, not the wording: that the message names
the string it got, that it names both halves of the rule, and that it gives
the spelling to write instead. "Add a decimal point" is the wrong rule --
`1.0e8` is still a string -- so a message stating only that half would send
the reader round again, and the `1.0e8` case below is what stops it being
written that way.
"""

from __future__ import annotations

import unittest

import pytest
import yaml

from fedbrew.core.refusal import yaml_number_cause

pytestmark = pytest.mark.fast

#: Spellings PyYAML resolves to `str`, and what each should be written as.
STRINGS = {
    "1e-08": "1.0e-08",
    "1e-8": "1.0e-08",
    "1E-08": "1.0e-08",
    "1e+8": "100000000.0",
    "1.0e8": "100000000.0",
    "3e4": "30000.0",
}

#: Spellings PyYAML resolves to a number, which must produce no cause at all.
NUMBERS = ("1.0e-08", "1.e-8", ".5e-8", "0.0001", "1.0", "42", "-2.5")


class TheResolverBehavesAsTheMessageClaimsTest(unittest.TestCase):
    """The message describes PyYAML. If PyYAML changes, the message is wrong."""

    def test_the_listed_spellings_really_do_resolve_to_strings(self) -> None:
        for text in STRINGS:
            with self.subTest(written=text):
                self.assertIsInstance(yaml.safe_load(f"x: {text}")["x"], str)

    def test_a_decimal_point_alone_is_not_enough(self) -> None:
        """The half of the rule a reader is most likely to stop at."""

        self.assertIsInstance(yaml.safe_load("x: 1.0e8")["x"], str)
        self.assertIsInstance(yaml.safe_load("x: 1.0e-08")["x"], float)

    def test_the_listed_numbers_really_do_resolve_to_numbers(self) -> None:
        for text in NUMBERS:
            with self.subTest(written=text):
                self.assertIsInstance(yaml.safe_load(f"x: {text}")["x"], int | float)


class TheCauseNamesWhatHappenedTest(unittest.TestCase):
    def test_it_quotes_the_string_it_was_given(self) -> None:
        self.assertIn(repr("1e-08"), yaml_number_cause("1e-08"))

    def test_it_states_both_halves_of_the_rule(self) -> None:
        cause = yaml_number_cause("1e-08")
        self.assertIn("decimal point", cause)
        self.assertIn("signed exponent", cause)

    def test_it_names_the_case_a_decimal_point_does_not_fix(self) -> None:
        """Without this, "add a decimal point" is the rule a reader takes away."""

        self.assertIn("1.0e8", yaml_number_cause("1e-08"))

    def test_it_gives_a_spelling_that_actually_parses(self) -> None:
        for text, expected in STRINGS.items():
            with self.subTest(written=text):
                cause = yaml_number_cause(text)
                self.assertIn(expected, cause)
                parsed = yaml.safe_load(f"x: {expected}")["x"]
                self.assertIsInstance(parsed, float)
                self.assertEqual(parsed, float(text))


class ItStaysOutOfTheWayOtherwiseTest(unittest.TestCase):
    """Appended unconditionally, so every other refusal must be unchanged."""

    def test_a_value_that_is_not_a_string_gets_nothing(self) -> None:
        for value in (1e-08, 0, -3, 1.5, True, None, [], {}, object()):
            with self.subTest(value=repr(value)):
                self.assertEqual(yaml_number_cause(value), "")

    def test_a_string_that_is_not_a_number_gets_nothing(self) -> None:
        for value in ("", "auto", "1 / t", "cosine", "1e", "e-08", "1,0", "0x10"):
            with self.subTest(value=value):
                self.assertEqual(yaml_number_cause(value), "")

    def test_the_expression_field_is_untouched(self) -> None:
        """`client.tau: "1 / t"` is an expression and never reaches a coercion.

        `"1"` is the constant form of that expression, and it *is* a
        numeric-looking string -- so this pins that the helper is only ever
        appended by a numeric coercion, and that a quoted constant expression
        is not one.
        """

        self.assertEqual(yaml_number_cause("1 / t"), "")
        self.assertNotEqual(yaml_number_cause("1"), "")


class TheRefusalsCarryItTest(unittest.TestCase):
    """Every coercion that says "must be numeric" reaches the helper."""

    def test_every_site_appends_it(self) -> None:
        """Read off the source, so a new coercion added without it is visible."""

        import pathlib
        import re

        root = pathlib.Path(__file__).resolve().parent.parent / "fedbrew" / "core"
        pattern = re.compile(r'must be numeric"(?!\s*\+\s*yaml_number_cause)')
        offenders = [
            path.name
            for path in (root / "config.py", root / "factory.py", root / "validation.py")
            if pattern.search(path.read_text(encoding="utf-8"))
        ]
        self.assertEqual(
            offenders,
            [],
            "a 'must be numeric' refusal does not append yaml_number_cause, so it "
            "will name the field and not the reason",
        )


if __name__ == "__main__":
    unittest.main()
