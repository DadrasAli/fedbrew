"""``fedbrew sweep``: several configs, those differing only in numeric hyperparameters as one group.

A sweep's configs are grouped automatically. Two configs share a group when
both load, both ask for the batched executor (``runtime.performance.executor``
left out asks for it), neither resumes, and they are equal on every key except
the ones that name a run (``runner._NOT_CONFIGURATION``) and ``VARIABLE``:
the client's learning rate, momentum, weight decay, FedProx's mu and clipping
norm, and the server's learning rate and momentum. A key one config has and the other does not is a
difference. Seeds, data, models and local iterations are therefore the same
inside a group; runs that differ in them run apart.

Each group of two or more runs in one child process (``--run-group``), its
settings trained together (``fedbrew/core/settings_group.py``, chapter 11
§10); every other config runs as ``fedbrew run --config`` would, in a child
of its own. The children run one after another, in the order of the command
line. ``--plan`` prints the grouping and runs nothing.

Exit status: 1 if any run crashed, otherwise 2 if any was refused, otherwise
0 -- a diverged or stalled run ended as a run does, as ``fedbrew run`` says.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from fedbrew.core.config import FullConfig, asked_executor, load_config
from fedbrew.core.runner import _NOT_CONFIGURATION, EXIT_REFUSED, _as_written

#: The keys a group's settings may differ in: numeric hyperparameters that
#: leave what is computed the same shape.
VARIABLE = frozenset(
    {
        "client.learning_rate",
        "client.extra.momentum",
        "client.extra.weight_decay",
        "client.extra.proximal_mu",
        "client.extra.max_grad_norm",
        "server.extra.server_learning_rate",
        "server.extra.beta1",
    }
)

#: A child that crashed, as ``python`` exits on an uncaught exception.
EXIT_CRASHED = 1

#: Kept apart from any key: the value of a key a config lacks.
_ABSENT = "<absent>"


@dataclass(slots=True)
class Planned:
    """One child of the sweep: a group of settings, or one config run as ``fedbrew run``."""

    configs: list[str]
    varies: list[str] = field(default_factory=list)
    #: Why a config runs alone; None for a group.
    alone: str | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="fedbrew sweep",
        description=(
            "Run several configs; those that differ only in numeric hyperparameters run as "
            "one group in one process, their clients trained together."
        ),
    )
    parser.add_argument("configs", nargs="+", help="config files, run in this order")
    parser.add_argument(
        "--plan", action="store_true", help="print how the configs group, and run nothing"
    )
    parser.add_argument(
        "--run-group",
        action="store_true",
        help="run the configs as one group in this process (what the sweep's children run)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.run_group:
        raise SystemExit(_run_one_group(args.configs))
    started = time.perf_counter()
    planned = plan(args.configs)
    _print_plan(planned)
    if args.plan:
        return
    codes = []
    for child in planned:
        if child.alone is None:
            command = ["sweep", "--run-group", *child.configs]
        else:
            command = ["run", "--config", child.configs[0]]
        print(f"fedbrew sweep: fedbrew {' '.join(command)}", flush=True)
        codes.append(
            subprocess.run([sys.executable, "-m", "fedbrew.cli.dispatch", *command]).returncode
        )
    for child, code in zip(planned, codes, strict=True):
        label = "group" if child.alone is None else "alone"
        print(f"fedbrew sweep: exit {code}  {label}  {' '.join(child.configs)}")
    print(f"fedbrew sweep: {len(args.configs)} configs in {time.perf_counter() - started:.1f} s")
    raise SystemExit(_worst(codes))


def plan(paths: Sequence[str | Path]) -> list[Planned]:
    """How ``paths`` group, in the order each group's first config appears."""

    signatures: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    children: list[Planned | str] = []
    for path in [str(path) for path in paths]:
        try:
            config = load_config(path)
        except Exception as error:  # run alone, where it fails as it fails today
            children.append(Planned([path], alone=f"does not load: {error}"))
            continue
        reason = _alone_reason(config)
        if reason is not None:
            children.append(Planned([path], alone=reason))
            continue
        flat = _flatten(_as_written(asdict(config)))
        signature = repr(sorted((key, value) for key, value in flat.items() if key not in _IGNORED))
        if signature not in signatures:
            signatures[signature] = []
            children.append(signature)
        signatures[signature].append((path, flat))
    planned: list[Planned] = []
    for child in children:
        if isinstance(child, Planned):
            planned.append(child)
            continue
        for members in _apart_by_directory(signatures[child]):
            if len(members) == 1:
                planned.append(
                    Planned(
                        [members[0][0]],
                        alone="no other config differs from it only in "
                        + ", ".join(sorted(VARIABLE)),
                    )
                )
            else:
                planned.append(Planned([path for path, _ in members], varies=_varies(members)))
    return planned


_IGNORED = _NOT_CONFIGURATION | VARIABLE


def _alone_reason(config: FullConfig) -> str | None:
    performance = config.runtime.extra.get("performance") or {}
    if asked_executor(performance) != "batched":
        return "runtime.performance.executor is sequential"
    if config.runtime.extra.get("resume_from") or config.runtime.extra.get("resume_latest"):
        return "it resumes, and a group is not resumed"
    return None


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """Every leaf of a config as a dotted key, a mapping's keys each their own."""

    if isinstance(value, Mapping):
        flat: dict[str, Any] = {}
        for key, inner in value.items():
            flat.update(_flatten(inner, f"{prefix}{key}."))
        if not value and prefix:
            flat[prefix[:-1]] = {}
        return flat
    return {prefix[:-1]: value}


def _apart_by_directory(
    members: list[tuple[str, dict[str, Any]]],
) -> list[list[tuple[str, dict[str, Any]]]]:
    """Configs that would write the same directory go to different groups, in turn."""

    groups: list[list[tuple[str, dict[str, Any]]]] = []
    for member in members:
        directory = _directory(member[1])
        for group in groups:
            if directory is None or all(_directory(other[1]) != directory for other in group):
                group.append(member)
                break
        else:
            groups.append([member])
    return groups


def _directory(flat: Mapping[str, Any]) -> str | None:
    if flat.get("experiment.use_run_subdir"):
        return None
    return str(Path(str(flat.get("experiment.output_dir"))).resolve())


def _varies(members: list[tuple[str, dict[str, Any]]]) -> list[str]:
    """The variable keys whose values are not the same across a group."""

    return sorted(
        key for key in VARIABLE if len({repr(flat.get(key, _ABSENT)) for _, flat in members}) > 1
    )


def _print_plan(planned: list[Planned]) -> None:
    groups = [child for child in planned if child.alone is None]
    alone = [child for child in planned if child.alone is not None]
    total = sum(len(child.configs) for child in planned)
    print(f"fedbrew sweep: {total} configs, {len(groups)} groups and {len(alone)} alone")
    for number, child in enumerate(planned, start=1):
        if child.alone is None:
            varies = ", ".join(child.varies) or "nothing"
            print(f"  {number}. group of {len(child.configs)}, varies {varies}")
            for path in child.configs:
                print(f"       {path}")
        else:
            print(f"  {number}. alone: {child.configs[0]} ({child.alone})")


def _run_one_group(paths: Sequence[str]) -> int:
    """Run ``paths`` as one group in this process; the sweep's exit status for it."""

    from fedbrew.core.settings_group import run_group

    planned = plan(paths)
    if len(planned) != 1 or planned[0].alone is not None or len(planned[0].configs) != len(paths):
        print(
            "fedbrew sweep --run-group: these configs are not one group; "
            "run them with fedbrew sweep, which groups them",
            file=sys.stderr,
        )
        return EXIT_REFUSED
    outcomes = run_group(list(paths), planned[0].varies)
    codes = []
    for outcome in outcomes:
        detail = f" ({outcome.detail})" if outcome.detail else ""
        print(f"fedbrew sweep: {outcome.status}  {outcome.config}{detail}")
        codes.append(
            EXIT_CRASHED
            if outcome.status == "crashed"
            else EXIT_REFUSED
            if outcome.status == "refused"
            else 0
        )
    return _worst(codes)


def _worst(codes: Sequence[int]) -> int:
    """1 if any child crashed, otherwise 2 if any was refused, otherwise 0."""

    if any(code not in (0, EXIT_REFUSED) for code in codes):
        return EXIT_CRASHED
    return EXIT_REFUSED if EXIT_REFUSED in codes else 0


if __name__ == "__main__":
    main()
