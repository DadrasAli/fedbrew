"""Generic path expansion helpers."""

from __future__ import annotations

import os
import re
from pathlib import Path

#: A $NAME or ${NAME} that expand_path left alone because the process did not
#: export it. Callers treat that as "unset" rather than writing to a literal
#: "$FL_LOCAL_SCRATCH" directory beside the checkout.
_ENV_PATTERN = re.compile(r"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?")


def has_unexpanded_env(value: str) -> bool:
    """Whether an already-expanded path still names an environment variable."""

    return bool(_ENV_PATTERN.search(value))


def expand_path(path: str | Path) -> Path:
    """Expand user and environment variables in a path."""

    return Path(os.path.expandvars(os.path.expanduser(str(path))))


def resolve_output_dir(path: str | Path) -> Path:
    """Resolve an experiment output path without changing relative semantics."""

    return expand_path(path)


def resolve_data_path(path: str | Path) -> Path:
    """Resolve a dataset path without changing relative semantics."""

    return expand_path(path)


def resolve_named_config(base_dir: str | Path, name: str) -> Path:
    """Resolve a short config name to a path, ``.yaml`` implied.

    An existing literal path always wins: a value that is already a real file
    -- relative, absolute, with or without an extension -- is returned as-is,
    never joined to `base_dir`. The lookup under `base_dir` only fires for a
    value that is not itself a path, which is what keeps this sugar and
    ``--config <path>`` from ever disagreeing about the same string.

    Raises:
        FileNotFoundError: Neither the literal path nor the resolved one under
            `base_dir` exists. The message names both, so a typo in either
            half is visible without re-deriving what was tried.
    """

    literal = Path(name)
    if literal.is_file():
        return literal

    candidate = name if name.endswith((".yaml", ".yml")) else f"{name}.yaml"
    resolved = Path(base_dir) / candidate
    if resolved.is_file():
        return resolved

    raise FileNotFoundError(
        f"no config named {name!r}: {literal} is not a file, and {resolved} does not exist either"
    )
