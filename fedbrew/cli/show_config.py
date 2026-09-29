"""``fedbrew config show <config>``: a run config as the loader resolves it.

An arm file that extends a family base states only its own keys, so the file
alone no longer shows the run: its numerics, its evaluation schedule and most
of its client settings are the base's. This prints the whole of it -- the
``extends`` chain merged, and the values the loader inferred filled in and
marked -- as one flat config, after loading it exactly as ``fedbrew run``
does, so a config the run would refuse is refused here with the same message.

What it prints is itself a config: written to a file under the same
``configs/`` path, it loads to the same run. Keys that neither the chain nor
the loader sets take their defaults, which chapter 04 lists; run.json records
those too, once the run exists.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml

from fedbrew.core.config import EXTENDS_KEY, load_config, load_config_mapping, load_yaml
from fedbrew.core.paths import resolve_named_config

#: Where a short name is looked up, as ``fedbrew run`` does.
CONFIG_BASE_DIR = "configs"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="fedbrew config",
        description="Print a run config resolved: extends merged, inferred values filled.",
    )
    actions = parser.add_subparsers(dest="action", metavar="action", required=True)
    show = actions.add_parser("show", help="Print the config as one flat, resolved file.")
    show.add_argument(
        "config",
        help=f"A config path, or a short name resolved against {CONFIG_BASE_DIR}/ "
        "('examples/drift-quad/fedavg').",
    )
    return parser.parse_args(argv)


def extends_chain(path: Path) -> list[Path]:
    """The config and every base under it, the config first."""

    chain = [path]
    while True:
        base = load_yaml(chain[-1]).get(EXTENDS_KEY)
        if not isinstance(base, str) or len(chain) > 64:
            return chain
        chain.append(chain[-1].parent / base)


def resolved_text(path: Path) -> str:
    """The config at ``path`` as one flat file, with its provenance in comments."""

    config = load_config(path)
    mapping = load_config_mapping(path)
    for key in config.inferred:
        block, name = key.split(".", 1)
        mapping.setdefault(block, {})[name] = getattr(getattr(config, block), name)
    body = yaml.safe_dump(mapping, sort_keys=False, default_flow_style=False)
    lines = body.splitlines()
    for key, source in config.inferred.items():
        _mark(lines, key.split("."), f"inferred from {source}")
    chain = extends_chain(path)
    header = [f"# {path}, resolved"]
    header.extend(f"# extends {base}" for base in chain[1:])
    header.append("# One flat file: the chain merged, and the values the loader inferred.")
    return "\n".join([*header, *lines]) + "\n"


def _mark(lines: list[str], key: list[str], note: str) -> None:
    """Append ``# note`` to the line that sets ``key`` in a block-style dump."""

    start, depth = 0, 0
    for part in key:
        prefix = "  " * depth + f"{part}:"
        for index in range(start, len(lines)):
            if lines[index] == prefix or lines[index].startswith(prefix + " "):
                start, depth = index, depth + 1
                break
        else:
            return
    lines[start] = f"{lines[start]}  # {note}"


def _as_config_path(name: str) -> Path:
    return Path(resolve_named_config(CONFIG_BASE_DIR, name))


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.action == "show":
        sys.stdout.write(resolved_text(_as_config_path(args.config)))


if __name__ == "__main__":
    main()
