#!/usr/bin/env python
"""List top-level folders from a common dataset root."""

import os
import sys
from pathlib import Path

COMMON_DATASETS_ENV = "COMMON_DATASETS"


def main():
    """List the top-level directories under ``$COMMON_DATASETS``.

    Exits with an error naming the variable when it is unset or does not point
    at an existing directory. There is deliberately no fallback path: the
    shared dataset mount is a per-cluster fact, and guessing one would either
    silently list the wrong tree or fail with a confusing error about a
    directory the user never named.
    """

    root = _common_dataset_root()

    print("COMMON_DATASETS:", root)
    entries = sorted(path.name for path in root.iterdir() if path.is_dir())
    if not entries:
        print("No top-level dataset folders found.")
        return
    for name in entries:
        print(name)


def _common_dataset_root() -> Path:
    """The common dataset root, from the environment.

    Deliberately not defaulted: the shared dataset mount point is a
    per-cluster fact, not something this tool should guess at.
    """

    configured = os.environ.get(COMMON_DATASETS_ENV, "").strip()
    if not configured:
        sys.exit(
            f"{COMMON_DATASETS_ENV} is not set, so there is no common dataset "
            "root to list.\n"
            f"  export {COMMON_DATASETS_ENV}=/path/to/common-datasets"
        )
    path = Path(os.path.expandvars(os.path.expanduser(configured)))
    if not path.exists():
        sys.exit(f"{COMMON_DATASETS_ENV}={path} does not exist")
    return path


if __name__ == "__main__":
    main()
