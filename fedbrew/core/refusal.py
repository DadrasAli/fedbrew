"""The exception a run raises when it declines what it was given."""

from __future__ import annotations


def yaml_number_cause(value: object) -> str:
    """Why a field that must be numeric was handed a string, when it looks numeric.

    PyYAML's safe loader implements YAML 1.1, whose float resolver is::

        [-+]?(\\.[0-9]+|[0-9]+(\\.[0-9]*)?)([eE][-+][0-9]+)?

    -- a decimal point **and a signed exponent**. So `1e-08`, `1e-8`, `1E-08`
    and `1.0e8` all resolve to plain strings, while `1.0e-08` and `1.e-8`
    resolve to floats. "Add a decimal point" is the wrong rule and `1.0e8` is
    the case that catches someone who takes it as one; both halves are named in
    the message because only naming one sends the reader round again.

    A tolerance written in scientific notation is where this bites: the refusal
    "server.tau must be numeric" is correct and says nothing about why a number
    looked like a string. Five sweep arms died 13 seconds into scheduled jobs
    on 2026-09-20 for exactly this, and the diagnosis took twenty minutes that
    this sentence would have saved.

    Returns "" for anything that is not a numeric-looking string, so a caller
    appends it unconditionally and the message is unchanged in every other case.
    """

    if not isinstance(value, str):
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    spelled = repr(number)
    if "e" in spelled:
        mantissa, _, exponent = spelled.partition("e")
        mantissa = mantissa if "." in mantissa else mantissa + ".0"
        exponent = exponent if exponent[:1] in ("+", "-") else "+" + exponent
        spelled = f"{mantissa}e{exponent}"
    elif "." not in spelled:
        spelled += ".0"
    return (
        f"; got the string {value!r}. YAML reads an exponent as a number only with "
        "BOTH a decimal point and a signed exponent, so 1e-08, 1E-08 and 1.0e8 are "
        f"all strings while 1.0e-08 is a float. Write {spelled}"
    )


class RunRefused(ValueError):
    """A deliberate refusal, with a reason written for the person who asked.

    Raised for input a run will not accept -- a config value, a CLI flag, a
    checkpoint it will not resume from -- and never for a defect in fedbrew.
    `fedbrew run` catches exactly this type, prints the reason and exits with
    `runner.EXIT_REFUSED`; any other exception keeps its traceback. A
    ValueError, so code that catches or asserts ValueError is unaffected.
    """
