"""Config values the loader infers instead of requiring them.

Five keys a config had to write carry nothing the loader could not already
find out:

- ``model.input_dim`` and ``model.num_classes``: the manifest states both,
  and a model whose value disagreed with it was refused when it was built;
- ``experiment.output_dir``: 84 of the 97 shipped run configs wrote
  ``outputs/<their path under configs/>``;
- ``experiment.name``: 71 wrote ``<their directory>-<their file name>``;
- ``server.strategy``: 60 wrote the one their ``client.update_rule`` implies --
  the paired rules' own strategy (``scaffold``, ``fedlalr``, ``centralized``),
  and ``fedavg`` for the FedAvg family's rules.

Each is now inferred when a config leaves it out, and stating it stays legal.
A stated model dimension is checked against the manifest at load, where the
disagreement used to surface only when the model was built. A stated strategy
is checked against the rule as before: a paired rule's half without the other
is refused (``_validate_paired_strategies``), and a FedOpt strategy over a
FedAvg-family rule is the config's choice. A stated output
directory or name that is not the path's is the config's choice -- 13 shipped
configs write to a directory of their own -- and is kept.

An inferred value is written into the resolved config like any stated one,
and ``FullConfig.inferred`` names each inferred key and where it came from,
which run.json records with the config and the plan header prints beside the
value.

The manifest is read here without loading a shard. When it cannot be read --
data not generated yet, or staged somewhere else -- nothing is inferred from
it, a stated dimension is checked where the model is built, as before, and an
unstated one is filled there from the same metadata (``factory._model_config``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any

from fedbrew.core.refusal import RunRefused

#: The ``FullConfig.inferred`` sources.
FROM_MANIFEST = "the dataset manifest"
FROM_CONFIG_PATH = "the config path"
FROM_UPDATE_RULE = "client.update_rule"

#: The directory whose tree the path inference reads a config's place in.
CONFIGS_DIRECTORY = "configs"

#: Where an inferred output directory goes: this, then the config's path under
#: ``configs/`` without its suffix. Relative, as every shipped value is, so it
#: resolves against the working directory as a stated one does.
OUTPUTS_DIRECTORY = "outputs"


def read_manifest(data_path: str | None) -> Mapping[str, Any] | None:
    """The manifest at ``data_path``, or None when there is none to read.

    Nothing here refuses: a missing or unreadable manifest is the dataset's to
    report where it is built, and preflight's data check.
    """

    if not data_path:
        return None
    from fedbrew.core.paths import resolve_data_path

    try:
        manifest = json.loads(Path(resolve_data_path(data_path)).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return manifest if isinstance(manifest, Mapping) else None


def path_under_configs(config_path: Path) -> Path | None:
    """The config's path below its nearest ``configs`` directory, or None if it has none."""

    resolved = config_path.resolve()
    for parent in resolved.parents:
        if parent.name == CONFIGS_DIRECTORY:
            return resolved.relative_to(parent)
    return None


def inferred_output_dir(config_path: Path) -> str | None:
    """``outputs/<path under configs/ without .yaml>``, or None outside a configs tree."""

    relative = path_under_configs(config_path)
    if relative is None:
        return None
    return str(Path(OUTPUTS_DIRECTORY) / relative.with_suffix(""))


def inferred_name(config_path: Path) -> str:
    """``<directory>-<file stem>`` for a config in a directory under ``configs/``.

    The file stem alone otherwise -- for a config directly in ``configs/`` or
    outside any configs tree -- which is what every config without a name was
    called before.
    """

    relative = path_under_configs(config_path)
    if relative is None or len(relative.parts) < 2:
        return config_path.stem
    return f"{relative.parent.name}-{config_path.stem}"


def infer_experiment(
    experiment: MutableMapping[str, Any],
    config_path: Path,
    inferred: MutableMapping[str, str],
) -> None:
    """Fill ``output_dir`` and ``name`` from the config path where the mapping has none."""

    if "output_dir" not in experiment:
        output_dir = inferred_output_dir(config_path)
        if output_dir is None:
            raise RunRefused(
                f"experiment.output_dir is required: {config_path} is not under a "
                f"{CONFIGS_DIRECTORY}/ directory, the one place it is inferred from "
                f"({OUTPUTS_DIRECTORY}/<the config's path under {CONFIGS_DIRECTORY}/>)"
            )
        experiment["output_dir"] = output_dir
        inferred["experiment.output_dir"] = FROM_CONFIG_PATH
    if not experiment.get("name"):
        experiment["name"] = inferred_name(config_path)
        inferred["experiment.name"] = FROM_CONFIG_PATH


def infer_strategy(
    server: MutableMapping[str, Any],
    client: Mapping[str, Any],
    inferred: MutableMapping[str, str],
) -> None:
    """Fill ``server.strategy`` from ``client.update_rule`` where the server block has none.

    A rule that implies no strategy -- an extension's, or a missing one --
    leaves a strategy required, and the refusal says which rules imply one.
    """

    from fedbrew.core.config import (
        FEDAVG_FAMILY_CLIENT_RULES,
        PAIRED_STRATEGIES,
        implied_strategy,
    )

    if "strategy" in server:
        return
    rule = client.get("update_rule")
    strategy = implied_strategy(rule)
    if strategy is None:
        raise RunRefused(
            f"server.strategy is required: client.update_rule={rule!r} implies none. "
            f"It is inferred for {', '.join(sorted(PAIRED_STRATEGIES))} (the strategy "
            f"of the same name) and for {', '.join(sorted(FEDAVG_FAMILY_CLIENT_RULES))} "
            "(fedavg)"
        )
    server["strategy"] = strategy
    inferred["server.strategy"] = FROM_UPDATE_RULE


def infer_model_shape(config: Any, stated: Mapping[str, Any]) -> None:
    """Fill the model's shape keys from the manifest, and check the stated ones against it.

    Only the keys the model's builder is sized by (``models.register(...,
    shape_keys=...)``): a manifest can state an ``input_dim`` its model never
    reads -- pl-1d's scalar, FEMNIST's image shape under a ResNet -- and an
    inferred value nothing reads would be recorded as if it had applied.
    Changes ``config.model`` and ``config.inferred`` in place.
    """

    from fedbrew.core.factory import model_data_shape_mismatch
    from fedbrew.core.registry import model_shape_keys

    if config.data.name != "manifest_dataset":
        return
    manifest = read_manifest(config.data.path)
    if manifest is None:
        return
    stated_values = {name: stated[name] for name in ("input_dim", "num_classes") if name in stated}
    mismatch = model_data_shape_mismatch(stated_values, manifest)
    if mismatch is not None:
        raise RunRefused(mismatch)
    for name in model_shape_keys(config.model.name):
        if name in stated or name not in manifest:
            continue
        setattr(config.model, name, manifest[name])
        config.inferred[f"model.{name}"] = FROM_MANIFEST
