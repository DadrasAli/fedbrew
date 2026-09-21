"""Generate manifest-based federated datasets."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import inspect
import json
import os
import shutil
import struct
import urllib.request
import warnings
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

warnings.filterwarnings(
    "ignore",
    message="Failed to initialize NumPy:.*",
    category=UserWarning,
)
import torch
import yaml  # type: ignore[import-untyped]

from fedbrew.core.console import (
    Rail,
    Row,
    Surface,
    add_output_arguments,
    silent_rail,
    surface_from_args,
)
from fedbrew.core.download_progress import redirect_torchvision_progress
from fedbrew.core.extensions import LoadedExtension, load_extensions
from fedbrew.core.logging import print_download_progress
from fedbrew.core.paths import expand_path, resolve_named_config
from fedbrew.core.registry import GeneratorSpec, generators, register_builtin_components
from fedbrew.data.official_test_partitioning import partition_test_indices_like_train
from fedbrew.data.partitioners.dirichlet import partition_dirichlet
from fedbrew.data.partitioners.iid import partition_iid
from fedbrew.data.partitioners.label_skew import partition_label_skew
from fedbrew.data.partitioners.quantity_skew import (
    LOGNORMAL_SIGMA,
    partition_quantity_skew,
)
from fedbrew.data.stats import compute_client_stats, compute_partition_summary
from fedbrew.data.synthetic_classification import (
    TEACHER_SEED_OFFSET,
    synthetic_labels,
    synthetic_teacher,
)
from fedbrew.data.writers.manifest import save_clients_jsonl, save_manifest
from fedbrew.data.writers.torch_shards import (
    save_client_shard,
    save_split_client_shard,
)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


#: The two sections every generator reads. The sections a generator reads
#: beyond these are declared on its ``GeneratorSpec`` in the ``generators``
#: registry (fedbrew/core/registry.py). A generator config is read with
#: .get(name, default) throughout, so a misspelled section would be dropped
#: and every key inside it would take its default -- a differently
#: partitioned dataset, generated without complaint -- which is why an
#: undeclared section is refused instead.
#:
#: ``client_splits`` was a third entry here, and that exemption was the one
#: hole in the rule above: the two SFT generators take the section, validate
#: it and delete it -- their splits are cut at the conversation tree by SHA --
#: so ``client_splits.eval_ratio: 0.1`` in a shipped OASST1 config named a
#: ratio nothing applied. It is declared per generator now, like every other
#: section, so setting it where it does nothing is refused by name.
_SHARED_SECTIONS = frozenset({"dataset", "partition"})

#: The causal-LM text generator and its tiny_causal_lm sibling read one
#: section under either name.
_CAUSAL_LM_KEYS = frozenset(
    {
        "sequence_length",
        "stride",
        "append_eos_between_records",
        "pad_incomplete_window",
        "pad_incomplete",
        "padding",
        "corpus_path",
        "source_path",
        "input_path",
        "asset_manifest",
        "tokenizer_asset_manifest",
        "preparation_config",
        "tokenizer_path",
        "tokenizer_identifier",
        "tokenizer_revision",
        "vocab_size",
        "local_files_only",
        "trust_remote_code",
    }
)

#: oasst1_sft reads one section under either name.
_OASST1_SFT_KEYS = frozenset(
    {
        "sequence_length",
        "ignore_index",
        "dataset_asset_manifest",
        "source_asset_manifest",
        "dataset_preparation_config",
        "tokenizer_asset_manifest",
        "asset_manifest",
        "tokenizer_preparation_config",
        "local_files_only",
        "trust_remote_code",
    }
)

#: Keys inside a section, enumerated from the readers rather than from the
#: shipped configs. The four LLM generators need prepared Hugging Face assets
#: and so cannot be exercised here, which is a reason their behaviour is
#: untested -- not a reason a misspelled key in them should take a default. An
#: unlisted section is still left unchecked; every section a generator reads is
#: now listed.
#:
#: Several keys are aliases the readers accept for each other
#: (``pad_incomplete_window`` / ``pad_incomplete`` / ``padding``,
#: ``corpus_path`` / ``source_path`` / ``input_path``, the ``maximum_*`` /
#: ``max_*`` pairs). They are listed because the readers honour them, not
#: because a config should use them.
_GENERATOR_SECTION_KEYS: dict[str, frozenset[str]] = {
    "dataset": frozenset({"name", "output_dir", "raw_dir", "seed", "extensions"}),
    "partition": frozenset(
        {
            "strategy",
            "num_clients",
            "alpha",
            "min_size",
            "max_size",
            "sigma",
            "labels_per_client",
        }
    ),
    "client_splits": frozenset({"train_ratio", "eval_ratio", "test_ratio"}),
    "splits": frozenset({"train_ratio", "test_ratio"}),
    "synthetic": frozenset({"num_samples", "input_dim", "num_classes"}),
    "mnist": frozenset({"flatten", "normalize", "max_train_samples", "max_test_samples"}),
    "cifar10": frozenset({"normalize", "max_train_samples", "max_test_samples"}),
    "femnist": frozenset(
        {
            "source",
            "revision",
            "split",
            "writer_column",
            "image_column",
            "label_column",
            "min_samples_per_client",
        }
    ),
    "source_splits": frozenset({"train_ratio", "test_ratio"}),
    "causal_lm": _CAUSAL_LM_KEYS,
    "hf_causal_lm_text": _CAUSAL_LM_KEYS,
    "generic_sft": frozenset(
        {
            "sequence_length",
            "ignore_index",
            "prompt_template",
            "response_template",
            "client_field",
            "group_field",
            "required_fields",
            "choice",
            "system_prompt",
            "anonymize_client_ids",
            "dataset_asset_manifest",
            "dataset_preparation_config",
            "tokenizer_asset_manifest",
            "asset_manifest",
            "tokenizer_preparation_config",
            "local_files_only",
            "trust_remote_code",
        }
    ),
    "oasst1_sft": _OASST1_SFT_KEYS,
    "sft": _OASST1_SFT_KEYS,
    "caps": frozenset(
        {
            "train_fraction",
            "train_windows_per_client",
            "client_eval_windows_per_client",
            "global_test_windows",
        }
    ),
    "tree_splits": frozenset(
        {
            "train_ratio",
            "client_eval_ratio",
            "eval_ratio",
            "global_test_ratio",
            "test_ratio",
        }
    ),
    "pilot_caps": frozenset(
        {
            "maximum_generated_windows_per_split",
            "maximum_qualifying_assistant_responses_per_client",
            "max_train_responses_per_client",
            "maximum_client_evaluation_responses_per_client",
            "max_client_eval_responses_per_client",
            "maximum_global_test_responses",
            "max_global_test_responses",
        }
    ),
}

#: Blocks nested one level inside a section. The same defect lives here: a
#: misspelled `option_fields` would have produced a prompt with no options.
_GENERATOR_NESTED_KEYS: dict[tuple[str, str], frozenset[str]] = {
    ("generic_sft", "choice"): frozenset({"index_field", "option_fields", "labels"}),
    ("pilot_caps", "maximum_generated_windows_per_split"): frozenset(
        {"train_per_client", "client_eval_per_client", "global_test"}
    ),
}


def _section_keys_of(spec: GeneratorSpec, dataset_name: str) -> dict[str, frozenset[str]]:
    """Every section this generator reads, with the keys inside each.

    Two sources, merged: the keys the spec declared for its own sections,
    and ``_GENERATOR_SECTION_KEYS`` for the shared three and for any section
    declared by name alone. A section in neither is refused here, on the
    first ``generate`` against this generator, rather than passed through
    with its keys unchecked -- which is what an out-of-tree generator's own
    section was before the spec could carry keys, and the reason
    ``condition_numbr: 100.0`` once generated a dataset at the default.
    """

    keys = dict(_GENERATOR_SECTION_KEYS)
    keys.update(spec.section_keys)
    unlisted = sorted(spec.sections - set(keys))
    if unlisted:
        raise ValueError(
            f"generator {dataset_name!r} declares the section"
            + ("s " if len(unlisted) > 1 else " ")
            + ", ".join(repr(section) for section in unlisted)
            + " but not the keys inside: register it with sections={"
            + ", ".join(f"{section!r}: {{...}}" for section in unlisted)
            + "} naming every key it reads, so that a misspelled one is refused "
            "instead of taking its default."
        )
    return keys


def _sections_read_by(spec: GeneratorSpec) -> frozenset[str]:
    """Every config section this generator's path actually reads.

    ``client_splits`` is not on a ``tensors`` generator's spec because the
    generator never sees it: it returns pooled tensors, and the shared writer
    in this module cuts every client's train/eval slices from them. Declaring
    it on each of the four would be a hand-kept copy of "every tensors
    generator", and the first one added without the copy would have a working
    ``client_splits`` section refused. A ``shards`` generator writes its own
    splits, so there it is a real per-generator question and the spec answers
    it.
    """

    if spec.kind == "tensors":
        return _SHARED_SECTIONS | spec.sections | {"client_splits"}
    return _SHARED_SECTIONS | spec.sections


def _validate_generator_keys(config: Mapping[str, Any], dataset_name: str) -> None:
    """Refuse a section or key no generator reads."""

    spec = generator_spec(dataset_name)
    allowed = _sections_read_by(spec)
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(
            f"generator {dataset_name!r} does not read these sections: "
            + ", ".join(unknown)
            + ". It reads: "
            + ", ".join(sorted(allowed))
            + ". An unread section is dropped and every key in it takes its "
            "default, so a misspelling generates a different dataset."
            # client_splits is the one section a config can carry in good
            # faith: it is what every other generator cuts its slices with,
            # and this one cuts them somewhere else entirely.
            + (
                " client_splits does not apply here: this generator cuts its "
                "splits in tree_splits, at the conversation tree, so a "
                "per-client ratio has nothing to divide."
                if "client_splits" in unknown and "tree_splits" in allowed
                else ""
            )
        )

    section_keys = _section_keys_of(spec, dataset_name)
    for section in sorted(set(config) & set(section_keys)):
        value = config[section]
        if not isinstance(value, Mapping):
            continue
        known = section_keys[section]
        stray = sorted(set(value) - known)
        if stray:
            raise ValueError(
                f"generator section {section!r} does not read: "
                + ", ".join(f"{section}.{name}" for name in stray)
                + ". It reads: "
                + ", ".join(sorted(known))
                + ". An unread key takes its default instead."
            )

        for name, nested_known in _GENERATOR_NESTED_KEYS.items():
            if name[0] != section:
                continue
            nested = value.get(name[1])
            if not isinstance(nested, Mapping):
                continue
            nested_stray = sorted(set(nested) - nested_known)
            if nested_stray:
                path = f"{section}.{name[1]}"
                raise ValueError(
                    f"generator section {path!r} does not read: "
                    + ", ".join(f"{path}.{key}" for key in nested_stray)
                    + ". It reads: "
                    + ", ".join(sorted(nested_known))
                    + ". An unread key takes its default instead."
                )


def generator_spec(dataset_name: str) -> GeneratorSpec:
    """Return the registered generator for ``dataset.name``, or refuse the name.

    The generator registry is checked here rather than at run-config load,
    because a generator config is the only config that selects from it.
    """

    register_builtin_components()
    if not generators.exists(dataset_name):
        raise ValueError(
            f"unknown dataset.name={dataset_name!r}. Registered generators: " + generators.listing()
        )
    return generators.get(dataset_name)


def _accepts_progress(generate: Any) -> bool:
    """Whether a shards generator takes ``on_progress``.

    Read off the signature rather than declared in a set beside it: a
    generator that accepts the parameter and is not wired to the stage is
    instrumentation nobody sees, and one wired without accepting it is a
    TypeError at the moment someone generates. The signature cannot disagree
    with itself.
    """

    return "on_progress" in inspect.signature(generate).parameters


@contextmanager
def _staged_output(output_dir: str | Path) -> Iterator[Path]:
    """Build the dataset beside its destination, then swap it in.

    Generators used to write straight into output_dir with
    mkdir(exist_ok=True), overwrite the shards one at a time, and save
    manifest.json and clients.jsonl last. A regeneration with a different seed,
    num_clients or split ratios that died part-way -- a time limit, an OOM --
    therefore left the *previous* run's manifest.json, clients.jsonl and
    global_test.pt beside the new run's shards. Every existing check passed:
    the files are all there and the shards all have x and y. FedAvg then
    weighted each client by a count from the old file, and FEMNIST's old
    global_test.pt, which is the concatenation of the old eval splits, could
    overlap the new train splits.

    Nothing is removed from the destination until the new dataset is complete,
    so a generation that raises leaves the working dataset exactly as it was
    and takes its own directory with it. A hard kill -- SIGKILL, a dead node --
    leaves `.<name>.incomplete-<pid>` behind, which the next run of the same
    generator does not read and the one after that overwrites. The swap is two
    renames within one directory, so it is atomic as far as any reader is
    concerned; the peak cost is one extra copy of the dataset on disk.
    """

    destination = Path(output_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.incomplete-{os.getpid()}"
    previous = destination.parent / f".{destination.name}.replaced-{os.getpid()}"
    _remove_tree(staging)
    _remove_tree(previous)

    try:
        yield staging
    except BaseException:
        _remove_tree(staging)
        raise

    if not staging.is_dir():
        raise ValueError(f"the generator wrote nothing into {staging}; {destination} is unchanged")

    if destination.exists():
        os.replace(destination, previous)
    try:
        os.replace(staging, destination)
    except BaseException:
        if previous.exists():
            os.replace(previous, destination)
        _remove_tree(staging)
        raise
    _remove_tree(previous)


def _remove_tree(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def generate_from_config(config_path: str | Path, rail: Rail | None = None) -> Path:
    """Generate a federated dataset from a YAML generator config.

    `rail` is where the stages report, defaulting to one that renders nothing
    so every existing caller keeps its old signature and its old silence.
    """

    rail = silent_rail() if rail is None else rail
    config = _load_yaml(config_path)
    dataset_config = _mapping(config["dataset"])
    # Before the name is looked up: an out-of-tree generator gets its name
    # from the extension the config names, the same way a run config's
    # components do.
    loaded = load_extensions(_extensions_of(dataset_config))
    _validate_generator_keys(config, str(dataset_config["name"]))
    partition_config = _mapping(config["partition"])

    dataset_name = str(dataset_config["name"])
    output_dir = expand_path(str(dataset_config["output_dir"]))
    seed = int(dataset_config.get("seed", 42))
    client_splits = _parse_client_splits(config.get("client_splits"))

    rail.detail("config", str(config_path))
    rail.detail("seed", str(seed))
    for extension in loaded:
        rail.detail("extension", extension.entry)
    with _staged_output(output_dir) as staging:
        manifest_path = _generate_into_directory(
            config=config,
            dataset_name=dataset_name,
            write_dir=staging,
            display_dir=output_dir,
            partition_config=partition_config,
            seed=seed,
            client_splits=client_splits,
            rail=rail,
        )
        if loaded:
            _record_extensions(Path(manifest_path), loaded)
    return Path(output_dir) / Path(manifest_path).relative_to(staging)


def _extensions_of(dataset_config: Mapping[str, Any]) -> list[str]:
    """The ``dataset.extensions`` list, checked to be one."""

    extensions = dataset_config.get("extensions", [])
    if not isinstance(extensions, list) or not all(
        isinstance(entry, str) and entry.strip() for entry in extensions
    ):
        raise ValueError(
            "dataset.extensions must be a list of non-empty strings: paths ending "
            "in .py or dotted module names"
        )
    if len(set(extensions)) != len(extensions):
        raise ValueError("dataset.extensions lists an entry twice")
    return list(extensions)


def _record_extensions(manifest_path: Path, loaded: Sequence[LoadedExtension]) -> None:
    """Write which extensions generated the dataset into its manifest.

    The manifest is the description of what was generated, and a generator
    loaded from outside the package is part of that description: the run
    that reads the manifest records it in turn, so a number can be tied to
    the version of the generator that produced its data.
    """

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"{manifest_path} must hold a JSON object")
    manifest["extensions"] = [
        {"entry": extension.entry, "resolved": extension.resolved, "sha256": extension.sha256}
        for extension in loaded
    ]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _generate_into_directory(
    *,
    config: Mapping[str, Any],
    dataset_name: str,
    write_dir: Path,
    display_dir: Path,
    partition_config: Mapping[str, Any],
    seed: int,
    client_splits: Mapping[str, float],
    rail: Rail,
) -> Path:
    """Run one generator, writing into write_dir and reporting display_dir."""

    output_dir = write_dir

    spec = generator_spec(dataset_name)
    if spec.kind == "shards":
        # One opaque call, and the only stage in this command that can take
        # minutes; the oasst1_sft generator used to spend all of it silent. A
        # stage cannot see inside it, but it can say what is running
        # and what came out, which is the difference between a hung command
        # and a slow one.
        generate = spec.resolve()
        with rail.stage(f"generate {dataset_name}") as stage:
            arguments: dict[str, Any] = {
                "config": config,
                "output_dir": output_dir,
                "seed": seed,
                "client_splits": client_splits,
            }
            if _accepts_progress(generate):
                # Stage.tick throttles and advances the pulse, so a generator
                # may report as often as its loop iterates: the OASST1 one
                # calls this 24,239 times for the tokenizer pass alone.
                arguments["on_progress"] = stage.tick
            summary = generate(**arguments)
            stage.done(
                f"{summary.num_clients} clients",
                note=f"{summary.num_examples:,} train, {summary.num_test_examples:,} test rows",
            )
        _print_summary(
            rail.surface,
            display_dir,
            dataset_name,
            summary.num_clients,
            summary.num_examples,
            summary.num_test_examples,
        )
        return summary.manifest_path

    num_clients = int(partition_config["num_clients"])
    train_x, train_y, test_x, test_y, metadata = spec.resolve()(config, seed)
    strategy = str(partition_config["strategy"])
    partitions = _partition_train_indices(
        train_y=train_y,
        strategy=strategy,
        num_clients=num_clients,
        seed=seed,
        alpha=float(partition_config.get("alpha", 0.5)),
        min_size=_optional_int(partition_config.get("min_size")),
        max_size=_optional_int(partition_config.get("max_size")),
        sigma=float(partition_config.get("sigma", LOGNORMAL_SIGMA)),
        labels_per_client=_optional_int(partition_config.get("labels_per_client")),
    )
    # A result line, not a stage: every step of this path is short, so an
    # animated state would be a claim about where the time goes that is not
    # true. What is worth saying
    # is which knob was actually used -- alpha, labels_per_client and sigma
    # each belong to exactly one strategy, and a config carrying all three
    # says nothing about which one is in force.
    rail.result(
        "partition",
        f"{strategy}, {num_clients} clients",
        note=_partition_parameters(strategy, partition_config),
    )
    manifest_path = _write_torch_shard_dataset(
        output_dir=output_dir,
        dataset_name=dataset_name,
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        test_y=test_y,
        partitions=partitions,
        num_clients=num_clients,
        metadata=metadata,
        partition_strategy=str(partition_config["strategy"]),
        partition_parameters=_partition_parameter_values(strategy, partition_config),
        client_splits=client_splits,
        seed=seed,
        rail=rail,
    )
    _print_summary(rail.surface, display_dir, dataset_name, num_clients, len(train_y), len(test_y))
    return manifest_path


#: Which partition parameter each strategy actually reads. A config may carry
#: all of them; only one is in force.
_PARTITION_PARAMETERS = {
    "dirichlet": ("alpha",),
    "label_skew": ("labels_per_client",),
    "quantity_skew": ("sigma", "min_size", "max_size"),
    "iid": (),
}


def _partition_parameter_values(
    strategy: str, partition_config: Mapping[str, Any]
) -> dict[str, Any]:
    """The knobs the configured strategy read, by name, for the record.

    The manifest recorded `partition_strategy` and `num_clients` and nothing
    else about the cut, so two datasets that differ only in `alpha` or
    `labels_per_client` had byte-identical manifests -- verified on
    `synthetic_label_skew` at `labels_per_client` 1 and 2, whose partitions have
    no client in common. A regeneration under the same path with a different
    knob was undetectable downstream, which compounds finding 7 of the same
    report: regenerating into an existing directory does not clear it.

    Read from `_PARTITION_PARAMETERS` rather than a second list, so the record
    and the rail line name the same knobs, and a strategy added there is
    recorded without being remembered here.
    """

    return {
        name: partition_config[name]
        for name in _PARTITION_PARAMETERS.get(strategy, ())
        if partition_config.get(name) is not None
    }


def _partition_parameters(strategy: str, partition_config: Mapping[str, Any]) -> str | None:
    stated = [
        f"{name}={value}"
        for name, value in _partition_parameter_values(strategy, partition_config).items()
    ]
    return ", ".join(stated) if stated else None


#: Where a short name (``synthetic``) resolves to a file (``data/configs/synthetic.yaml``).
_CONFIG_BASE_DIR = "data/configs"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse generator CLI arguments."""

    parser = argparse.ArgumentParser(description="Generate federated data manifests.")
    parser.add_argument(
        "config_name",
        nargs="?",
        metavar="name",
        help=(
            f"Short name resolved against {_CONFIG_BASE_DIR}/, .yaml implied "
            "(e.g. 'synthetic'). Mutually exclusive with --config."
        ),
    )
    parser.add_argument("--config", help="Path to generator YAML config.")
    add_output_arguments(parser)
    args = parser.parse_args(argv)

    if args.config_name and args.config:
        parser.error(
            f"give a config name or --config, not both ({args.config_name!r} and {args.config!r})"
        )
    if args.config_name:
        try:
            args.config = str(resolve_named_config(_CONFIG_BASE_DIR, args.config_name))
        except FileNotFoundError as exc:
            parser.error(str(exc))
    elif not args.config:
        parser.error("the following arguments are required: name or --config")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    """Run the generator CLI."""

    args = parse_args(argv)
    surface = surface_from_args(args)
    surface.rule("GENERATE")
    generate_from_config(args.config, surface.rail(["partition", "clients", "labels"]))


def generate_synthetic_classification_tensors(
    config: Mapping[str, Any],
    seed: int,
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    """Draw the synthetic classification tensors from a seeded linear teacher."""

    synthetic_config = _mapping(config["synthetic"])
    splits_config = _mapping(config["splits"])
    num_samples = int(synthetic_config["num_samples"])
    input_dim = int(synthetic_config["input_dim"])
    num_classes = int(synthetic_config["num_classes"])

    x, y = _generate_synthetic_tensors(
        num_samples=num_samples,
        input_dim=input_dim,
        num_classes=num_classes,
        seed=seed,
    )
    train_indices, test_indices = _split_indices(
        num_samples=num_samples,
        train_ratio=float(splits_config["train_ratio"]),
        seed=seed,
    )
    metadata = {
        "input_dim": input_dim,
        "num_classes": num_classes,
        "label_rule": "linear_teacher",
    }
    return (
        x[train_indices],
        y[train_indices],
        x[test_indices],
        y[test_indices],
        metadata,
    )


def generate_mnist_tensors(
    config: Mapping[str, Any],
    seed: int,
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    """Load MNIST tensors through torchvision and prepare them for shards.

    ``seed`` is part of the tensors contract. MNIST is a fixed corpus and
    draws nothing here; the seed acts downstream, in the partition.
    """

    del seed
    dataset_config = _mapping(config["dataset"])
    mnist_config = _mapping(config.get("mnist", {}))
    raw_dir = expand_path(str(dataset_config.get("raw_dir", "data/raw/datasets/mnist")))
    source = _TORCHVISION_MNIST_SOURCE
    try:
        train_images, train_labels, test_images, test_labels = _load_mnist_torchvision(raw_dir)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MNIST generation requires torchvision. Install it with: pip install -e '.[vision]'"
        ) from exc
    except (RuntimeError, OSError) as exc:
        # Narrow on purpose. This is torchvision's could-not-fetch surface --
        # RuntimeError("Error downloading ...") once every mirror failed, and
        # OSError, which URLError subclasses, for the network and the disk. A
        # TypeError or an AttributeError means .data/.targets moved under us,
        # which the mirror cannot stand in for, so it propagates as the bug it
        # is instead of being rerouted into data that looks fine.
        warnings.warn(
            f"torchvision could not read MNIST ({type(exc).__name__}: {exc}); "
            f"reading the idx archives from {_MNIST_MIRROR_SOURCE} instead. "
            "The manifest and every run.json built from it record that mirror "
            "as the source.",
            stacklevel=2,
        )
        source = _MNIST_MIRROR_SOURCE
        train_images, train_labels, test_images, test_labels = _load_mnist_idx_files(
            raw_dir=raw_dir,
            max_train_samples=_optional_int(mnist_config.get("max_train_samples")),
            max_test_samples=_optional_int(mnist_config.get("max_test_samples")),
        )
    return _prepare_mnist_tensors(
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
        mnist_config=mnist_config,
        source=source,
    )


def _load_mnist_torchvision(raw_dir: Path) -> tuple[Any, Any, Any, Any]:
    from torchvision.datasets import MNIST  # type: ignore[import-untyped]

    with redirect_torchvision_progress(print_download_progress):
        train_dataset = MNIST(root=str(raw_dir), train=True, download=True)
        test_dataset = MNIST(root=str(raw_dir), train=False, download=True)
    return (
        train_dataset.data,
        train_dataset.targets,
        test_dataset.data,
        test_dataset.targets,
    )


#: Where MNIST came from when torchvision read it, and when it could not. Both
#: are `source` values in the manifest and in every run.json built from it.
_TORCHVISION_MNIST_SOURCE = "torchvision.datasets.MNIST"
_MNIST_MIRROR_SOURCE = "https://storage.googleapis.com/cvdf-datasets/mnist"

#: MD5 of each MNIST archive, as published in
#: ``torchvision.datasets.MNIST.resources``. Held here rather than read from
#: torchvision because this path runs exactly when torchvision did not; the two
#: tables are pinned equal by ``tests/test_mnist_source_provenance.py``.
_MNIST_ARCHIVE_MD5 = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def _load_mnist_idx_files(
    raw_dir: Path,
    max_train_samples: int | None,
    max_test_samples: int | None,
) -> tuple[Any, Any, Any, Any]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }
    for file_name in files.values():
        _download_mnist_file(raw_dir / file_name)

    train_images = _read_idx_images(raw_dir / files["train_images"], max_train_samples)
    train_labels = _read_idx_labels(raw_dir / files["train_labels"], max_train_samples)
    test_images = _read_idx_images(raw_dir / files["test_images"], max_test_samples)
    test_labels = _read_idx_labels(raw_dir / files["test_labels"], max_test_samples)
    return train_images, train_labels, test_images, test_labels


def _download_mnist_file(path: Path) -> None:
    """Put one verified MNIST archive at ``path``, downloading it if needed.

    The archive torchvision would have checked is the archive this checks: the
    digests are the ones ``torchvision.datasets.MNIST.resources`` carries, so
    the mirror is held to the same bytes rather than to nothing. Downloads land
    on a temporary name and are renamed only once they verify, so an interrupted
    download never becomes a file a later run would trust.
    """

    expected = _MNIST_ARCHIVE_MD5[path.name]
    if path.exists():
        found = _md5(path.read_bytes())
        if found == expected:
            return
        # Not a truncated download of ours -- those never get this name. Say
        # which file, and leave it where the user can look at it.
        raise ValueError(
            f"{path} does not match the MNIST archive it is named after "
            f"(md5 {found}, expected {expected}). Delete it to fetch a fresh "
            "copy, or point dataset.raw_dir somewhere else."
        )

    url = f"{_MNIST_MIRROR_SOURCE}/{path.name}"
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = response.read()
    found = _md5(payload)
    if found != expected:
        raise ValueError(
            f"{url} served {len(payload):,} bytes with md5 {found}, expected "
            f"{expected}. The mirror is not serving the MNIST archive "
            "torchvision declares, so the download is discarded rather than "
            "partitioned into shards."
        )
    partial = path.with_name(f"{path.name}.partial")
    partial.write_bytes(payload)
    partial.replace(path)


def _md5(payload: bytes) -> str:
    # Not a security boundary: these are the digests torchvision publishes for
    # the same four files, so this answers "are these the MNIST archives" in the
    # spelling that can be compared against torchvision's own table.
    return hashlib.md5(payload).hexdigest()


def _read_idx_images(path: Path, max_samples: int | None) -> Any:
    with gzip.open(path, "rb") as file:
        magic, num_images, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid MNIST image file: {path}")
        count = min(num_images, max_samples) if max_samples is not None else num_images
        data = file.read(count * rows * cols)
    tensor = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    return tensor.reshape(count, rows, cols)


def _read_idx_labels(path: Path, max_samples: int | None) -> Any:
    with gzip.open(path, "rb") as file:
        magic, num_labels = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid MNIST label file: {path}")
        count = min(num_labels, max_samples) if max_samples is not None else num_labels
        data = file.read(count)
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).to(torch.long)


def _prepare_mnist_tensors(
    train_images: Any,
    train_labels: Any,
    test_images: Any,
    test_labels: Any,
    mnist_config: Mapping[str, Any],
    source: str,
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    flatten = bool(mnist_config.get("flatten", True))
    normalize = str(mnist_config.get("normalize", "none"))
    if normalize != "none":
        raise ValueError("only mnist.normalize=none is supported")

    train_x = _prepare_mnist_images(train_images, flatten)
    test_x = _prepare_mnist_images(test_images, flatten)
    train_y = torch.as_tensor(train_labels, dtype=torch.long)
    test_y = torch.as_tensor(test_labels, dtype=torch.long)
    train_x, train_y = _limit_examples(
        train_x,
        train_y,
        _optional_int(mnist_config.get("max_train_samples")),
    )
    test_x, test_y = _limit_examples(
        test_x,
        test_y,
        _optional_int(mnist_config.get("max_test_samples")),
    )
    metadata = {
        "input_dim": 784 if flatten else [1, 28, 28],
        "num_classes": 10,
        # Which reader produced these tensors, not which one was tried first.
        # `source` is a run.json provenance key, so a run trained on mirror
        # bytes says so.
        "source": source,
        "max_train_samples": _optional_int(mnist_config.get("max_train_samples")),
        "max_test_samples": _optional_int(mnist_config.get("max_test_samples")),
    }
    return train_x, train_y, test_x, test_y, metadata


def _prepare_mnist_images(images: Any, flatten: bool) -> Any:
    x = torch.as_tensor(images, dtype=torch.float32) / 255.0
    if flatten:
        return x.reshape(x.shape[0], 28 * 28)
    return x.unsqueeze(1)


def generate_cifar10_tensors(
    config: Mapping[str, Any],
    seed: int,
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    """Load CIFAR-10 tensors through torchvision and prepare them for shards.

    ``seed`` is part of the tensors contract. CIFAR-10 is a fixed corpus and
    draws nothing here; the seed acts downstream, in the partition.
    """

    del seed
    dataset_config = _mapping(config["dataset"])
    cifar_config = _mapping(config.get("cifar10", {}))
    raw_dir = expand_path(str(dataset_config.get("raw_dir", "data/raw/datasets/cifar10")))
    normalize = str(cifar_config.get("normalize", "standard"))
    try:
        train_images, train_labels, test_images, test_labels = _load_cifar10_torchvision(raw_dir)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "CIFAR-10 generation requires torchvision. Install it with: pip install -e '.[vision]'"
        ) from exc

    train_x = _prepare_cifar10_images(train_images, normalize)
    test_x = _prepare_cifar10_images(test_images, normalize)
    train_y = torch.as_tensor(train_labels, dtype=torch.long)
    test_y = torch.as_tensor(test_labels, dtype=torch.long)
    max_train_samples = _optional_int(cifar_config.get("max_train_samples"))
    max_test_samples = _optional_int(cifar_config.get("max_test_samples"))
    train_x, train_y = _limit_examples(train_x, train_y, max_train_samples)
    test_x, test_y = _limit_examples(test_x, test_y, max_test_samples)
    metadata = {
        "input_shape": [3, 32, 32],
        "num_classes": 10,
        "source": "torchvision.datasets.CIFAR10",
        "normalize": normalize,
        "max_train_samples": max_train_samples,
        "max_test_samples": max_test_samples,
    }
    return train_x, train_y, test_x, test_y, metadata


def _load_cifar10_torchvision(raw_dir: Path) -> tuple[Any, Any, Any, Any]:
    from torchvision.datasets import CIFAR10  # type: ignore[import-untyped]

    dataset_root = _resolve_cifar10_root(raw_dir)
    download = not _has_cifar10_files(dataset_root)
    with redirect_torchvision_progress(print_download_progress):
        train_dataset = CIFAR10(root=str(dataset_root), train=True, download=download)
        test_dataset = CIFAR10(root=str(dataset_root), train=False, download=download)
    return (
        train_dataset.data,
        train_dataset.targets,
        test_dataset.data,
        test_dataset.targets,
    )


def _resolve_cifar10_root(raw_dir: Path) -> Path:
    for candidate in _cifar10_root_candidates(raw_dir):
        if _has_cifar10_files(candidate):
            return candidate
    return raw_dir


def _cifar10_root_candidates(raw_dir: Path) -> list[Path]:
    candidates = [raw_dir]
    common_datasets = os.environ.get("COMMON_DATASETS")
    if common_datasets:
        common_root = expand_path(common_datasets)
        candidates.extend([common_root / "CIFAR", common_root / "cifar10"])

    unique_candidates: list[Path] = []
    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)
    return unique_candidates


def _has_cifar10_files(root: Path) -> bool:
    data_dir = root / "cifar-10-batches-py"
    return (data_dir / "data_batch_1").is_file() and (data_dir / "test_batch").is_file()


def _prepare_cifar10_images(images: Any, normalize: str) -> Any:
    x = torch.as_tensor(images, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    if normalize == "none":
        return x.contiguous()
    if normalize == "standard":
        mean = torch.tensor(CIFAR10_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(CIFAR10_STD, dtype=torch.float32).view(1, 3, 1, 1)
        return ((x - mean) / std).contiguous()
    raise ValueError("cifar10.normalize must be one of: none, standard")


def _limit_examples(x: Any, y: Any, max_samples: int | None) -> tuple[Any, Any]:
    if max_samples is None:
        return x, y
    if max_samples < 0:
        raise ValueError("max sample limits must be non-negative")
    return x[:max_samples], y[:max_samples]


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("max sample limits must be integers or null")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    raise ValueError("max sample limits must be integers or null")


def _write_torch_shard_dataset(
    output_dir: Path,
    dataset_name: str,
    train_x: Any,
    train_y: Any,
    test_x: Any,
    test_y: Any,
    partitions: Mapping[str, list[int]],
    num_clients: int,
    metadata: Mapping[str, Any],
    partition_strategy: str,
    client_splits: Mapping[str, float],
    seed: int,
    partition_parameters: Mapping[str, Any] | None = None,
    rail: Rail | None = None,
) -> Path:
    # Optional for the same reason the public entry points make it optional:
    # a caller that wants the files and not the commentary should not have to
    # construct a renderer to say so.
    rail = silent_rail() if rail is None else rail
    output_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    clients_metadata = []
    client_stats = []
    train_labels = [int(label) for label in train_y]
    test_labels = [int(label) for label in test_y]
    test_partitions = partition_test_indices_like_train(
        test_labels=test_labels,
        train_labels=train_labels,
        train_partitions=partitions,
        strategy=partition_strategy,
        seed=seed + 2003,
    )
    train_ratio = float(client_splits["train_ratio"])
    eval_ratio = float(client_splits["eval_ratio"])
    for client_position, client_id in enumerate(sorted(partitions)):
        local_indices = partitions[client_id]
        if not local_indices:
            # The postcondition asserted where the shard is written, not only
            # inside the partitioners that used to break it: split_client_indices
            # maps [] to ([], []) without complaint, so an empty partition
            # becomes a zero-row shard that generation reports as a success and
            # the run rejects hours later at evaluation. Every partitioner
            # already refuses to produce this -- fill_empty_clients for the two
            # label-skewing ones, min_size >= 1 for quantity_skew -- and this is
            # what stops the next one from reintroducing it quietly.
            raise ValueError(
                f"client {client_id!r} received no examples from the "
                f"{partition_strategy} partition; a zero-row shard is not a client"
            )
        shard = f"shards/{client_id}.pt"
        train_indices, eval_indices = split_client_indices(
            local_indices,
            train_ratio=train_ratio,
            eval_ratio=eval_ratio,
            seed=seed + client_position + 1009,
        )
        test_indices = test_partitions[client_id]
        save_split_client_shard(
            shards_dir / f"{client_id}.pt",
            train_x[train_indices],
            train_y[train_indices],
            train_x[eval_indices],
            train_y[eval_indices],
            test_x[test_indices],
            test_y[test_indices],
        )
        stats = compute_client_stats(client_id, local_indices, train_labels)
        train_label_counts = compute_label_counts_for_indices(train_indices, train_labels)
        eval_label_counts = compute_label_counts_for_indices(eval_indices, train_labels)
        test_label_counts = compute_label_counts_for_indices(test_indices, test_labels)
        stats.update(
            {
                "num_train_examples": len(train_indices),
                "num_eval_examples": len(eval_indices),
                "num_test_examples": len(test_indices),
                "train_label_counts": train_label_counts,
                "test_label_counts": test_label_counts,
                "eval_label_counts": eval_label_counts,
            }
        )
        client_stats.append(stats)
        clients_metadata.append(
            {
                "client_id": client_id,
                "split": "train",
                # Every split. local_indices is only train + eval -- the
                # client's slice of the training corpus -- while its test rows
                # come from the official test set, so this used to mean
                # something different here than in femnist.py, which counts all
                # three. manifest_validation checks the sum.
                "num_examples": len(train_indices) + len(eval_indices) + len(test_indices),
                "num_train_examples": len(train_indices),
                "num_eval_examples": len(eval_indices),
                "num_test_examples": len(test_indices),
                "shard": shard,
                "label_counts": stats["label_counts"],
                "train_label_counts": train_label_counts,
                "eval_label_counts": eval_label_counts,
                "test_label_counts": test_label_counts,
                "num_labels": stats["num_labels"],
                "dominant_label": stats["dominant_label"],
                "dominant_label_fraction": stats["dominant_label_fraction"],
            }
        )

    save_client_shard(shards_dir / "global_test.pt", test_x, test_y)
    summary = _write_partition_stats(
        output_dir=output_dir,
        dataset_name=dataset_name,
        partition_strategy=partition_strategy,
        client_stats=client_stats,
        client_splits=client_splits,
        partition_parameters=dict(partition_parameters or {}),
        seed=seed,
    )
    _report_partition_summary(rail, summary)
    manifest = _build_manifest(
        dataset_name=dataset_name,
        num_clients=num_clients,
        metadata=metadata,
        partition_strategy=partition_strategy,
        partition_parameters=dict(partition_parameters or {}),
        client_splits=client_splits,
        seed=seed,
        input_claims=_input_claims_of(train_x, test_x),
    )
    manifest_path = save_manifest(output_dir, manifest)
    save_clients_jsonl(output_dir, clients_metadata)
    return manifest_path


def _input_claims_of(train_x: Any, test_x: Any) -> dict[str, Any]:
    """`input_dtype` and `input_range`, read off the tensors being written.

    `input_range` is a bound, not a domain: the check in
    `manifest_validation._validate_input_claims` fails a shard holding values
    *outside* it. Taken over the pooled train and test features here, so every
    client shard cut from them satisfies it -- and a shard left behind by an
    earlier generation into the same directory, which `generate` does not
    clear, does not.

    `input_dtype` is exact, and it is the half that catches the case worth
    catching: a FEMNIST-style manifest declaring uint8 over [0, 255] against
    shards regenerated as [0, 1] floats fails on the dtype, where the range
    alone would not, since [0, 1] sits inside [0, 255].
    """

    minimum = min(float(train_x.min()), float(test_x.min()))
    maximum = max(float(train_x.max()), float(test_x.max()))
    return {
        "input_dtype": str(train_x.dtype).removeprefix("torch."),
        "input_range": [minimum, maximum],
    }


def _build_manifest(
    dataset_name: str,
    num_clients: int,
    metadata: Mapping[str, Any],
    partition_strategy: str,
    partition_parameters: Mapping[str, Any],
    client_splits: Mapping[str, float],
    seed: int,
    input_claims: Mapping[str, Any],
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "dataset_name": dataset_name,
        "format": "torch_shards",
        "num_clients": num_clients,
        "num_classes": metadata["num_classes"],
        "clients_file": "clients.jsonl",
        "global_test": "shards/global_test.pt",
        "shards_dir": "shards",
        "partition_strategy": partition_strategy,
        # What the strategy was given, and the seed every random choice in the
        # partition came from. Without these two the manifest described the
        # shape of the cut and none of its inputs: two datasets differing only
        # in alpha or labels_per_client were byte-identical here, so a
        # regeneration under the same path with a different knob left nothing
        # downstream able to tell. `seed` is spelled as generic_sft already
        # spells it.
        "partition_parameters": dict(partition_parameters),
        "seed": seed,
        # What the feature tensors are, computed from the ones just written.
        # femnist.py declared these two and nothing read them; the difference
        # they exist to record is real -- FEMNIST shards are uint8 over
        # [0, 255] and these are float32 over [0, 1] -- and it was stated for
        # one of the two. manifest_validation holds every shard to them now,
        # so the pair is a description of the data rather than decoration.
        **input_claims,
        "partition_stats_file": "partition_stats.json",
        "client_stats_file": "client_stats.csv",
        "client_shard_format": "split_v2",
        "client_test_source": "partitioned_global_test",
        "client_splits": {
            "train_ratio": float(client_splits["train_ratio"]),
            "eval_ratio": float(client_splits["eval_ratio"]),
        },
    }
    if "input_dim" in metadata:
        manifest["input_dim"] = metadata["input_dim"]
    if "input_shape" in metadata:
        manifest["input_shape"] = metadata["input_shape"]
    if "label_rule" in metadata:
        # Say how the labels were produced. The previous rule -- draw them
        # independently of the features -- left the manifest indistinguishable
        # from one describing a learnable dataset.
        manifest["label_rule"] = metadata["label_rule"]
    if dataset_name in {"mnist", "cifar10"}:
        manifest.update(
            {
                "source": metadata.get("source"),
                "normalize": metadata.get("normalize"),
                "max_train_samples": metadata.get("max_train_samples"),
                "max_test_samples": metadata.get("max_test_samples"),
            }
        )
    return manifest


def _generate_synthetic_tensors(
    num_samples: int,
    input_dim: int,
    num_classes: int,
    seed: int,
) -> tuple[Any, Any]:
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(num_samples, input_dim, generator=generator)
    teacher = synthetic_teacher(input_dim, num_classes, seed + TEACHER_SEED_OFFSET)
    targets = synthetic_labels(features, teacher, generator)
    return features, targets


def _split_indices(
    num_samples: int,
    train_ratio: float,
    seed: int,
) -> tuple[Any, Any]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")
    generator = torch.Generator().manual_seed(seed + 1)
    shuffled = torch.randperm(num_samples, generator=generator)
    train_size = int(num_samples * train_ratio)
    return shuffled[:train_size], shuffled[train_size:]


def split_client_indices(
    indices: Sequence[int],
    train_ratio: float,
    eval_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Split one client's local indices into deterministic train/eval sets."""

    _validate_client_split_ratios(train_ratio, eval_ratio)
    values = list(indices)
    if not values:
        return [], []
    if eval_ratio == 0.0 or len(values) == 1:
        return sorted(values), []

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(values), generator=generator).tolist()
    shuffled = [values[index] for index in order]
    eval_size = max(1, int(round(len(values) * eval_ratio)))
    eval_size = min(eval_size, len(values) - 1)
    train_values = shuffled[eval_size:]
    eval_values = shuffled[:eval_size]
    return sorted(train_values), sorted(eval_values)


def compute_label_counts_for_indices(
    indices: Sequence[int],
    labels: Sequence[Any],
) -> dict[str, int]:
    """Count how many of ``indices`` carry each label.

    Args:
        indices: Positions into ``labels`` -- one client's slice of the pooled
            training set. Every entry must be a valid index.
        labels: The pooled label sequence. Entries must be int-coercible.

    Returns:
        Label (stringified integer) to count, ordered by numeric label value
        rather than lexicographically, so "10" sorts after "9". Labels absent
        from ``indices`` are omitted rather than mapped to 0, so a client's
        entry only names the classes it actually holds -- which is what makes
        the partition statistics show label skew directly.
    """

    counts: dict[str, int] = {}
    for index in indices:
        key = str(int(labels[index]))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def _parse_client_splits(value: object) -> dict[str, float]:
    """train/eval/test fractions of each client's own examples.

    test_ratio defaults to 0: generators whose test data comes from somewhere
    else (MNIST partitions the official test set) do not carve a third slice.
    A generator whose only source of test data is the client itself has to set
    it, or its "test" metric is measured on data it also selects on.
    """

    if value is None:
        train_ratio = 1.0
        eval_ratio = 0.0
        test_ratio = 0.0
    else:
        config = _mapping(value)
        train_ratio = float(config.get("train_ratio", 1.0))
        eval_ratio = float(config.get("eval_ratio", 0.0))
        test_ratio = float(config.get("test_ratio", 0.0))
    _validate_client_split_ratios(train_ratio, eval_ratio, test_ratio)
    return {
        "train_ratio": train_ratio,
        "eval_ratio": eval_ratio,
        "test_ratio": test_ratio,
    }


def _validate_client_split_ratios(
    train_ratio: float, eval_ratio: float, test_ratio: float = 0.0
) -> None:
    if train_ratio <= 0.0:
        raise ValueError("client_splits.train_ratio must be > 0")
    if eval_ratio < 0.0:
        raise ValueError("client_splits.eval_ratio must be >= 0")
    if test_ratio < 0.0:
        raise ValueError("client_splits.test_ratio must be >= 0")
    if abs((train_ratio + eval_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("client_splits train_ratio + eval_ratio + test_ratio must equal 1.0")


def _partition_train_indices(
    train_y: Any,
    strategy: str,
    num_clients: int,
    seed: int,
    alpha: float,
    min_size: int | None,
    max_size: int | None,
    sigma: float,
    labels_per_client: int | None,
) -> dict[str, list[int]]:
    local_indices = list(range(len(train_y)))
    if strategy == "iid":
        return partition_iid(local_indices, num_clients, seed)
    if strategy == "dirichlet":
        labels = [int(label) for label in train_y]
        return partition_dirichlet(labels, num_clients, alpha, seed)
    if strategy == "quantity_skew":
        if min_size is None or max_size is None:
            raise ValueError("quantity_skew requires min_size and max_size")
        return partition_quantity_skew(
            local_indices,
            num_clients=num_clients,
            min_size=min_size,
            max_size=max_size,
            seed=seed,
            sigma=sigma,
        )
    if strategy == "label_skew":
        if labels_per_client is None:
            raise ValueError("label_skew requires labels_per_client")
        labels = [int(label) for label in train_y]
        return partition_label_skew(
            labels,
            num_clients=num_clients,
            labels_per_client=labels_per_client,
            seed=seed,
        )
    raise ValueError(f"Unknown partition strategy: {strategy}")


def _report_partition_summary(rail: Rail, summary: Mapping[str, object]) -> None:
    """Say out loud what partition_stats.json already knew.

    These are the headline facts of the whole command -- how many examples,
    how unevenly they landed, how many labels -- and until now they were
    computed on every run and written only to a file. "The smallest client has
    9 examples" is the sentence a reader needs before training on the result,
    and finding it meant opening a JSON file they did not know existed.
    """

    total = summary.get("total_examples")
    smallest = summary.get("min_examples_per_client")
    largest = summary.get("max_examples_per_client")
    mean = summary.get("mean_examples_per_client")
    if isinstance(mean, (int, float)):
        spread = f"{smallest}-{largest} each, mean {float(mean):.1f}"
    else:  # pragma: no cover - compute_partition_summary always returns it.
        spread = f"{smallest}-{largest} each"
    rail.result(
        "clients", f"{total:,} examples" if isinstance(total, int) else str(total), note=spread
    )

    labels = summary.get("global_label_counts")
    if isinstance(labels, Mapping) and labels:
        counts = [int(count) for count in labels.values()]
        rail.result(
            "labels",
            f"{len(labels)} classes",
            note=f"{min(counts):,}-{max(counts):,} examples each",
        )

    # The one line that is a warning rather than a result. A client with no val
    # slice is legal and reports zero examples for it, so val_* is an average
    # over fewer clients than num_clients -- a fact the reader has to have
    # before comparing that curve with a run whose clients all had one.
    without_eval = summary.get("clients_without_eval_split")
    if isinstance(without_eval, int) and without_eval > 0:
        rail.warn(
            "val split",
            f"{without_eval:,} of {summary['num_clients']:,} clients have none",
            note="too few examples to hold one out; excluded from every val_* average",
        )


def _write_partition_stats(
    output_dir: Path,
    dataset_name: str,
    partition_strategy: str,
    client_stats: list[dict[str, object]],
    client_splits: Mapping[str, float],
    partition_parameters: Mapping[str, Any],
    seed: int,
) -> dict[str, object]:
    summary = compute_partition_summary(client_stats)
    # A client too small to hold out an eval slice gets none: split_client_indices
    # returns ([x], []) for a one-example partition, and fill_empty_clients
    # manufactures exactly those. The runtime allows it on purpose -- the
    # per-split evaluate() reports zero examples and drops the client from the
    # val aggregate rather than failing the round -- so generation may not
    # refuse it. What it may not do either is stay quiet: the shipped
    # mnist_dirichlet.yaml (1000 clients, alpha 0.1) leaves 29 clients with no
    # val split, so val_* is an average over 971 of the 1000 the config names,
    # and nothing anywhere said so.
    summary["clients_without_eval_split"] = (
        sum(1 for stats in client_stats if not stats["num_eval_examples"])
        if float(client_splits["eval_ratio"]) > 0.0
        else 0
    )
    payload = {
        "dataset_name": dataset_name,
        "partition_strategy": partition_strategy,
        # The same two the manifest records. This file is the one a reader
        # opens to ask how the partition landed, and it could not say what was
        # asked for.
        "partition_parameters": dict(partition_parameters),
        "seed": seed,
        "num_clients": summary["num_clients"],
        "total_examples": summary["total_examples"],
        "min_examples_per_client": summary["min_examples_per_client"],
        "max_examples_per_client": summary["max_examples_per_client"],
        "mean_examples_per_client": summary["mean_examples_per_client"],
        "clients_without_eval_split": summary["clients_without_eval_split"],
        "labels": summary["labels"],
        "global_label_counts": summary["global_label_counts"],
        "client_splits": {
            "train_ratio": float(client_splits["train_ratio"]),
            "eval_ratio": float(client_splits["eval_ratio"]),
        },
        "clients": client_stats,
    }
    (output_dir / "partition_stats.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_client_stats_csv(output_dir / "client_stats.csv", client_stats, summary)
    return summary


def _write_client_stats_csv(
    path: Path,
    client_stats: list[dict[str, object]],
    summary: dict[str, object],
) -> None:
    raw_labels = summary.get("labels", [])
    labels = [str(label) for label in raw_labels] if isinstance(raw_labels, list) else []
    fieldnames = [
        "client_id",
        "num_examples",
        "num_train_examples",
        "num_eval_examples",
        "num_labels",
        "dominant_label",
        "num_test_examples",
        "dominant_label_fraction",
        *[f"label_{label}" for label in labels],
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for stats in client_stats:
            counts = stats.get("label_counts", {})
            if not isinstance(counts, dict):
                counts = {}
            row: dict[str, object] = {
                "client_id": stats["client_id"],
                "num_examples": stats["num_examples"],
                "num_train_examples": stats.get("num_train_examples", stats["num_examples"]),
                "num_test_examples": stats.get("num_test_examples", 0),
                "num_eval_examples": stats.get("num_eval_examples", 0),
                "num_labels": stats["num_labels"],
                "dominant_label": stats["dominant_label"],
                "dominant_label_fraction": stats["dominant_label_fraction"],
            }
            for label in labels:
                row[f"label_{label}"] = counts.get(label, 0)
            writer.writerow(row)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("generator config must be a YAML mapping")
    return cast(dict[str, Any], data)


def _mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("expected config section to be a mapping")
    return cast(Mapping[str, Any], value)


def _print_summary(
    surface: Surface,
    output_dir: Path,
    dataset_name: str,
    num_clients: int,
    num_train: int,
    num_test: int,
) -> None:
    surface.rows(
        [
            Row("Generated", dataset_name),
            Row("Clients", str(num_clients)),
            Row("Train rows", f"{num_train:,}"),
            Row("Test rows", f"{num_test:,}"),
        ]
    )
    # The output path is the one thing the next command needs, so it is the
    # line --quiet leaves behind.
    surface.final(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
