"""Registry helpers for discovering benchmark components."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib import import_module
from types import MappingProxyType
from typing import Any, Generic, TypeVar

from fedbrew.clients.base import ClientUpdate
from fedbrew.core.refusal import RunRefused
from fedbrew.data.dataset import FederatedDataset
from fedbrew.servers.base import ServerStrategy
from fedbrew.tasks.base import TaskAdapter

T = TypeVar("T")
Model = Any
ServerFactory = Callable[..., ServerStrategy]
ClientFactory = Callable[..., ClientUpdate]
TaskFactory = Callable[..., TaskAdapter]
DatasetFactory = Callable[..., FederatedDataset]
ModelFactory = Callable[..., Model]

#: The origin every built-in name is registered under. Any other origin is the
#: config entry that loaded an extension -- see fedbrew/core/extensions.py --
#: and it is what an "unknown name" message and a duplicate-registration error
#: print beside the name, so a reader can tell where a component came from.
BUILTIN = "builtin"

#: The origin in force, innermost last. ``registering_from`` pushes one for the
#: duration of an extension's ``register()`` so that the extension's own calls
#: -- ``registry.tasks.register(name, factory)``, with no origin argument --
#: are attributed to the file the config named rather than to the package.
#: Each entry carries the list its registrations are recorded into.
_ORIGINS: list[tuple[str, list[tuple[str, str]]]] = []


@contextmanager
def registering_from(origin: str) -> Iterator[list[tuple[str, str]]]:
    """Attribute every registration made inside the block to ``origin``.

    Yields the list the block's registrations are appended to, as
    ``(registry label, name)`` pairs, which is how a loader learns what an
    extension added without the extension having to report it.
    """

    if not isinstance(origin, str) or not origin.strip():
        raise ValueError("an origin must be a non-empty string")
    recorded: list[tuple[str, str]] = []
    _ORIGINS.append((origin, recorded))
    try:
        yield recorded
    finally:
        _ORIGINS.pop()


def _current_origin() -> str:
    return _ORIGINS[-1][0] if _ORIGINS else BUILTIN


def _record(label: str, name: str) -> None:
    if _ORIGINS:
        _ORIGINS[-1][1].append((label, name))


def describe_origin(origin: str) -> str:
    """The words a message uses for an origin: the package, or the entry."""

    return "the package" if origin == BUILTIN else origin


def _config_key_set(config_keys: Iterable[str]) -> frozenset[str]:
    if isinstance(config_keys, str):
        raise ValueError("config_keys must be a collection of key names, not one string")
    keys = frozenset(config_keys)
    for key in keys:
        if not isinstance(key, str) or not key.strip():
            raise ValueError("every declared config key must be a non-empty string")
    return keys


class Registry(Generic[T]):
    """Minimal typed registry for named benchmark components.

    Every name carries the origin it was registered under: ``BUILTIN`` for
    the package's own components, otherwise the config entry that loaded the
    extension. The origin is what lets the documentation guards diff a
    chapter against the built-in set alone while a run that loaded an
    extension still sees every name, and what a duplicate-registration error
    names on both sides.
    """

    def __init__(self, label: str, *, config_section: str | None = None) -> None:
        """Start empty. Names are registered once and never overwritten.

        Args:
            label: What this registry holds, as messages say it --
                ``"tasks"``, ``"models"`` -- and as ``registering_from``
                records it.
            config_section: The run-config block whose free keys a component
                registered here may extend -- ``"server"`` for a strategy,
                ``"client"`` for an update rule, ``"data"`` for a dataset
                backend. None for a registry whose components read no block
                beyond its named fields; ``config_keys`` is then refused, so
                a declaration nothing would read cannot be made.
        """

        self.label = label
        self.config_section = config_section
        self._items: dict[str, T] = {}
        self._origins: dict[str, str] = {}
        self._config_keys: dict[str, frozenset[str]] = {}

    def register(
        self,
        name: str,
        obj: T,
        *,
        origin: str | None = None,
        config_keys: Iterable[str] = (),
    ) -> None:
        """Register an object under a unique name.

        Args:
            name: The name a config selects the object by.
            obj: The factory or specification registered under it.
            origin: Where the registration comes from. Defaults to the
                origin in force -- ``BUILTIN`` outside ``registering_from``.
            config_keys: The keys this component reads from its config
                block's free keys, beyond the ones every component may use.
                Accepted at load only while this component is the one in
                force, and forwarded to its factory as keyword arguments, so
                for an extension "declared" and "forwarded" are one act.

        Raises:
            ValueError: If the name is already registered -- the message
                names both origins, because a second registration is a
                conflict between two sources and a reader needs to know which
                two -- or if keys are declared on a registry with no config
                section.
        """

        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{self.label}: a registered name must be a non-empty string")
        keys = _config_key_set(config_keys)
        if keys and self.config_section is None:
            raise ValueError(
                f"{self.label}: {name!r} declares config keys "
                f"{', '.join(sorted(keys))}, but a component in this registry "
                "reads no config block beyond its named fields, so nothing "
                "would forward them."
            )
        resolved = _current_origin() if origin is None else origin
        if name in self._items:
            raise ValueError(
                f"{self.label} already holds {name!r}, registered by "
                f"{describe_origin(self._origins[name])}; refusing a second "
                f"registration from {describe_origin(resolved)}. Names are "
                "registered once and never overwritten."
            )
        self._items[name] = obj
        self._origins[name] = resolved
        self._config_keys[name] = keys
        _record(self.label, name)

    def config_keys(self, name: str) -> frozenset[str]:
        """The config keys a registered component declared; empty for a built-in."""

        if name not in self._items:
            raise KeyError(f"Object is not registered: {name}")
        return self._config_keys[name]

    def get(self, name: str) -> T:
        """Return a registered object by name."""
        if name not in self._items:
            raise KeyError(f"Object is not registered: {name}")
        return self._items[name]

    def list(self) -> list[str]:
        """List every registered name in insertion order, extensions included."""
        return [*self._items]

    def builtin(self) -> list[str]:
        """List the names the package itself registered, in insertion order.

        The set the documentation chapters describe and their guards diff
        against. An extension loaded earlier in the same process is in
        ``list()`` and not here.
        """

        return [name for name, origin in self._origins.items() if origin == BUILTIN]

    def origin(self, name: str) -> str:
        """Return the origin a name was registered under."""
        if name not in self._origins:
            raise KeyError(f"Object is not registered: {name}")
        return self._origins[name]

    def exists(self, name: str) -> bool:
        """Return whether a name is registered."""
        return name in self._items

    def listing(self) -> str:
        """Every name, built-ins first and each extension's under its origin.

        The form every "unknown name" message uses, so that a reader who
        mistyped sees what there is and where each part of it came from.
        """

        parts = [", ".join(sorted(self.builtin()))]
        by_origin: dict[str, list[str]] = {}
        for name, origin in self._origins.items():
            if origin != BUILTIN:
                by_origin.setdefault(origin, []).append(name)
        for origin, names in by_origin.items():
            parts.append(f"from {origin}: " + ", ".join(sorted(names)))
        return "; ".join(part for part in parts if part)


#: Which task adapter each model needs. Filled by ``ModelRegistry.register``,
#: which requires ``task=``: a model only ever works with one, so the
#: registration is the single source of truth and configs do not restate it.
#: Read through ``task_for_model``, which registers the built-ins first.
MODEL_TASKS: dict[str, str] = {}


class ModelRegistry(Registry[ModelFactory]):
    """The model registry, which also records the task each model needs."""

    def register(  # type: ignore[override]
        self,
        name: str,
        obj: ModelFactory,
        *,
        task: str,
        origin: str | None = None,
    ) -> None:
        """Register a model builder together with the task adapter it needs.

        Args:
            name: The ``model.name`` a config selects it by.
            obj: The builder, called with the model config mapping.
            task: The ``task.name`` this model trains under. Required: a
                model with no task cannot be run, and a config cannot supply
                one because the pairing is a fact about the model.
            origin: As for ``Registry.register``.
        """

        if not isinstance(task, str) or not task.strip():
            raise ValueError(f"models: {name!r} must name the task adapter it needs (task=...)")
        super().register(name, obj, origin=origin)
        MODEL_TASKS[name] = task


def task_for_model(model_name: str) -> str:
    """Return the task adapter a model requires."""

    register_builtin_components()
    if model_name not in MODEL_TASKS:
        raise RunRefused(
            f"Unknown model: {model_name}. Known models: "
            + ", ".join(sorted(MODEL_TASKS))
            + ". A model is registered together with the task it needs "
            "(models.register(name, builder, task=...)). A model defined "
            "outside the package is registered when the config names its "
            "module in experiment.extensions, so a name missing from this "
            "list is usually a missing entry there rather than a missing "
            "model (chapter 12)."
        )
    return MODEL_TASKS[model_name]


#: The two shapes a dataset generator can take; see ``GeneratorSpec``.
GENERATOR_KINDS = frozenset({"shards", "tensors"})


def _frozen_key_set(section: str, names: Iterable[str]) -> frozenset[str]:
    """The keys declared for one section, refused if that is nothing."""

    if isinstance(names, str) or not isinstance(names, Iterable):
        raise ValueError(
            f"generator section {section!r} must declare its keys as a set of names, got {names!r}"
        )
    keys = frozenset(names)
    if not keys or any(not isinstance(key, str) or not key.strip() for key in keys):
        raise ValueError(
            f"generator section {section!r} declares no keys; a section a generator "
            "reads has at least one, and declaring it by name alone is the built-in "
            "form, whose keys live in fedbrew/data/generate.py"
        )
    return keys


@dataclass(frozen=True, slots=True)
class GeneratorSpec:
    """One dataset generator, as ``fedbrew generate`` dispatches it.

    Two kinds. A ``shards`` generator writes the whole dataset itself --
    the shards, ``clients.jsonl`` and ``manifest.json`` -- through
    ``generate_<name>_from_config(config, output_dir, seed, client_splits)``,
    returning a summary that names the manifest and counts the clients and
    rows. A ``tensors`` generator returns one pooled
    ``(train_x, train_y, test_x, test_y, metadata)`` from ``(config, seed)``
    and leaves the partitioning and the writing to the shared path in
    ``fedbrew/data/generate.py``.

    ``target`` is the callable, or a ``"module:function"`` string naming it,
    resolved on first use so that registering the built-ins imports none of
    them. ``sections`` are the config sections the generator reads beyond
    ``dataset`` and ``partition``, which every generator shares; a section not
    declared here is refused at generate time, because an unread section would
    be dropped and every key in it would take its default. ``client_splits``
    is declared like any other section by a ``shards`` generator -- the two
    SFT ones cut their splits at the conversation tree instead and do not
    declare it -- and is implicit for a ``tensors`` generator, whose splits
    are cut for it by the shared writer.

    ``section_keys`` are the keys inside those sections, for the sections
    whose keys the spec itself declares. The same argument applies one level
    down -- a generator config is read with ``.get(key, default)``, so a
    misspelled key would take its default as silently as a misspelled
    section -- and the built-ins keep their key lists in
    ``fedbrew/data/generate.py`` beside the readers. A section declared by
    name alone therefore means "its keys are listed there"; a section listed
    nowhere is refused at generate time rather than left unchecked, which is
    what an out-of-tree generator's own section would otherwise be.
    """

    target: str | Callable[..., Any]
    sections: frozenset[str]
    kind: str = "shards"
    section_keys: Mapping[str, frozenset[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Refuse a kind the generator cannot dispatch, and freeze the sections."""

        if self.kind not in GENERATOR_KINDS:
            raise ValueError(
                f"a generator's kind must be one of {', '.join(sorted(GENERATOR_KINDS))}, "
                f"got {self.kind!r}"
            )
        if not callable(self.target) and (
            not isinstance(self.target, str) or self.target.count(":") != 1
        ):
            raise ValueError(
                "a generator's target is the callable or a 'module:function' string naming it"
            )
        keys = {
            section: _frozen_key_set(section, names) for section, names in self.section_keys.items()
        }
        object.__setattr__(self, "section_keys", MappingProxyType(keys))
        object.__setattr__(self, "sections", frozenset(self.sections) | frozenset(keys))

    def resolve(self) -> Callable[..., Any]:
        """Return the generator callable, importing it if it was named."""

        if callable(self.target):
            return self.target
        module_name, _, attribute = self.target.partition(":")
        target = getattr(import_module(module_name), attribute)
        if not callable(target):
            raise TypeError(f"{self.target} is not callable")
        return target


class GeneratorRegistry(Registry[GeneratorSpec]):
    """The generator registry, which builds the spec from the parts.

    ``register(name, generate, sections={"problem": {"dim", "kappa"}})`` is
    the whole registration a generator needs: the sections it reads, and the
    keys inside each. The package's own eight write the shorter
    ``sections={"mnist"}``, naming the section only, because their key lists
    live in ``fedbrew/data/generate.py`` beside the readers -- with the other
    difference that a built-in names its target as a ``"module:function"``
    string, which is what keeps registering the eight of them from importing
    torchvision, transformers or any generator module.
    """

    def register(  # type: ignore[override]
        self,
        name: str,
        obj: GeneratorSpec | str | Callable[..., Any],
        *,
        sections: Iterable[str] | Mapping[str, Iterable[str]] = (),
        kind: str = "shards",
        origin: str | None = None,
        config_keys: Iterable[str] = (),
    ) -> None:
        """Register one generator.

        Args:
            name: The ``dataset.name`` a generator config selects it by.
            obj: The generator callable, a ``"module:function"`` string naming
                it, or a ready ``GeneratorSpec``.
            sections: The config sections it reads beyond ``dataset`` and
                ``partition``, ``client_splits`` included when this generator
                cuts by it. A mapping declares each section's keys with it; a
                plain set of names declares the sections only, and is the form
                for a section whose keys ``fedbrew/data/generate.py`` already
                lists. Refused if ``obj`` is already a spec, which carries its
                own.
            kind: ``"shards"`` for a generator that writes its own manifest,
                ``"tensors"`` for one returning a pooled tensor pair.
            origin: As for ``Registry.register``.
            config_keys: Refused. A generator reads a generator config, whose
                sections are declared above; it never reads a run config.
        """

        if isinstance(obj, GeneratorSpec):
            if sections or kind != "shards":
                raise ValueError(
                    f"generators: {name!r} was given a GeneratorSpec and "
                    "sections/kind as well; the spec already carries them"
                )
            spec = obj
        elif isinstance(sections, Mapping):
            spec = GeneratorSpec(obj, frozenset(), kind, section_keys=dict(sections))
        else:
            spec = GeneratorSpec(obj, frozenset(sections), kind)
        super().register(name, spec, origin=origin, config_keys=config_keys)


server_strategies: Registry[ServerFactory] = Registry("server_strategies", config_section="server")
client_updates: Registry[ClientFactory] = Registry("client_updates", config_section="client")
tasks: Registry[TaskFactory] = Registry("tasks")
datasets: Registry[DatasetFactory] = Registry("datasets", config_section="data")
models: ModelRegistry = ModelRegistry("models")
generators: GeneratorRegistry = GeneratorRegistry("generators")


def register_builtin_components() -> None:
    """Register built-in component factories once."""

    _register_once(server_strategies, "fedavg", _build_fedavg_server)
    _register_once(server_strategies, "fedavgm", _build_fedavgm_server)
    _register_once(server_strategies, "fedadam", _build_fedadam_server)
    _register_once(server_strategies, "fedyogi", _build_fedyogi_server)
    _register_once(server_strategies, "fedadagrad", _build_fedadagrad_server)
    _register_once(server_strategies, "fedopt", _build_fedopt_server)
    _register_once(server_strategies, "scaffold", _build_scaffold_server)
    _register_once(server_strategies, "fedlalr", _build_fedlalr_server)
    # The centralized baseline runs the FedAvg server and client unchanged over a
    # single pooled client (see fedbrew/data/centralized_dataset.py). Averaging one
    # result is the identity, so the baseline differs from a FedAvg run only in how
    # the data is partitioned -- the local-update modes cannot drift apart.
    _register_once(server_strategies, "centralized", _build_fedavg_server)
    _register_once(client_updates, "local_sgd", _build_torch_sgd_client)
    _register_once(client_updates, "fedavg", _build_fedavg_client)
    _register_once(client_updates, "centralized", _build_fedavg_client)
    _register_once(client_updates, "local_adamw", _build_torch_adamw_client)
    _register_once(client_updates, "fedprox", _build_torch_fedprox_client)
    _register_once(client_updates, "scaffold", _build_torch_scaffold_client)
    _register_once(client_updates, "delta_sgd", _build_delta_sgd_client)
    _register_once(client_updates, "fedlalr", _build_fedlalr_client)
    _register_once(client_updates, "fedavg_ft", _build_fedavg_ft_client)
    _register_once(datasets, "synthetic_classification", _build_synthetic_dataset)
    _register_once(datasets, "manifest_dataset", _build_manifest_dataset)
    _register_once(models, "mlp", _build_torch_mlp, task="classification")
    _register_once(models, "cnn", _build_torch_cnn, task="classification")
    _register_once(models, "small_cnn", _build_torch_small_cnn, task="classification")
    _register_once(models, "femnist_resnet18", _build_femnist_resnet18, task="classification")
    _register_once(
        models, "openimage_shufflenet", _build_openimage_shufflenet, task="classification"
    )
    _register_once(models, "tiny_gpt2", _build_tiny_gpt2, task="causal_lm")
    _register_once(models, "hf_causal_lm", _build_hf_causal_lm, task="causal_lm")
    _register_once(models, "hf_causal_lm_lora", _build_hf_causal_lm_lora, task="causal_lm")
    _register_once(tasks, "classification", _build_torch_classification_task)
    _register_once(tasks, "causal_lm", _build_torch_causal_lm_task)
    # Generators by name, resolved on first use: registering them imports no
    # torchvision, no transformers and no generator module.
    _register_once(
        generators,
        "synthetic_classification",
        "fedbrew.data.generate:generate_synthetic_classification_tensors",
        sections={"synthetic", "splits"},
        kind="tensors",
    )
    _register_once(
        generators,
        "mnist",
        "fedbrew.data.generate:generate_mnist_tensors",
        sections={"mnist"},
        kind="tensors",
    )
    _register_once(
        generators,
        "cifar10",
        "fedbrew.data.generate:generate_cifar10_tensors",
        sections={"cifar10"},
        kind="tensors",
    )
    _register_once(
        generators,
        "femnist",
        "fedbrew.data.femnist:generate_femnist_from_config",
        sections={"femnist", "client_splits"},
    )
    _register_once(
        generators,
        "tiny_causal_lm",
        "fedbrew.data.tiny_causal_lm:generate_tiny_causal_lm_from_config",
        sections={"causal_lm", "splits", "client_splits"},
    )
    _register_once(
        generators,
        "generic_sft",
        "fedbrew.data.generic_sft:generate_generic_sft_from_config",
        sections={"generic_sft", "caps", "tree_splits"},
    )
    _register_once(
        generators,
        "hf_causal_lm_text",
        "fedbrew.data.hf_causal_lm_text:generate_hf_causal_lm_text_from_config",
        sections={
            "hf_causal_lm_text",
            "causal_lm",
            "splits",
            "source_splits",
            "client_splits",
        },
    )
    _register_once(
        generators,
        "oasst1_sft",
        "fedbrew.data.oasst1_sft:generate_oasst1_sft_from_config",
        sections={"oasst1_sft", "sft", "pilot_caps", "tree_splits", "splits"},
    )


def _build_fedavg_server(*args: Any, **kwargs: Any) -> ServerStrategy:
    with _suppress_torch_numpy_warning():
        from fedbrew.servers.fedavg import FedAvgServer

    return FedAvgServer(*args, **kwargs)


def _build_fedavgm_server(*args: Any, **kwargs: Any) -> ServerStrategy:
    kwargs = dict(kwargs)
    # The one place this alias's optimizer is named. `_build_server` passes
    # `server_optimizer` only for the `fedopt` strategy, where the config
    # supplies it, so this default is what a named strategy runs on rather
    # than dead code beside a value the factory had already decided. P10-F33.
    kwargs.setdefault("server_optimizer", "fedavgm")
    return _build_fedopt_server(*args, **kwargs)


def _build_fedadam_server(*args: Any, **kwargs: Any) -> ServerStrategy:
    kwargs = dict(kwargs)
    # The one place this alias's optimizer is named. `_build_server` passes
    # `server_optimizer` only for the `fedopt` strategy, where the config
    # supplies it, so this default is what a named strategy runs on rather
    # than dead code beside a value the factory had already decided. P10-F33.
    kwargs.setdefault("server_optimizer", "fedadam")
    return _build_fedopt_server(*args, **kwargs)


def _build_fedyogi_server(*args: Any, **kwargs: Any) -> ServerStrategy:
    kwargs = dict(kwargs)
    # The one place this alias's optimizer is named. `_build_server` passes
    # `server_optimizer` only for the `fedopt` strategy, where the config
    # supplies it, so this default is what a named strategy runs on rather
    # than dead code beside a value the factory had already decided. P10-F33.
    kwargs.setdefault("server_optimizer", "fedyogi")
    return _build_fedopt_server(*args, **kwargs)


def _build_fedadagrad_server(*args: Any, **kwargs: Any) -> ServerStrategy:
    kwargs = dict(kwargs)
    # The one place this alias's optimizer is named. `_build_server` passes
    # `server_optimizer` only for the `fedopt` strategy, where the config
    # supplies it, so this default is what a named strategy runs on rather
    # than dead code beside a value the factory had already decided. P10-F33.
    kwargs.setdefault("server_optimizer", "fedadagrad")
    return _build_fedopt_server(*args, **kwargs)


def _build_fedopt_server(*args: Any, **kwargs: Any) -> ServerStrategy:
    with _suppress_torch_numpy_warning():
        from fedbrew.servers.fedopt import FedOptServer

    return FedOptServer(*args, **kwargs)


def _build_scaffold_server(*args: Any, **kwargs: Any) -> ServerStrategy:
    with _suppress_torch_numpy_warning():
        from fedbrew.servers.scaffold import ScaffoldServer

    return ScaffoldServer(*args, **kwargs)


def _build_torch_sgd_client(*args: Any, **kwargs: Any) -> ClientUpdate:
    with _suppress_torch_numpy_warning():
        from fedbrew.clients.torch_sgd_client import TorchSGDClient

    return TorchSGDClient(*args, **kwargs)


def _build_fedavg_client(*args: Any, **kwargs: Any) -> ClientUpdate:
    with _suppress_torch_numpy_warning():
        from fedbrew.clients.fedavg_client import FedAvgClient

    return FedAvgClient(*args, **kwargs)


def _build_torch_adamw_client(*args: Any, **kwargs: Any) -> ClientUpdate:
    with _suppress_torch_numpy_warning():
        from fedbrew.clients.torch_adamw_client import TorchAdamWClient

    return TorchAdamWClient(*args, **kwargs)


def _build_torch_fedprox_client(*args: Any, **kwargs: Any) -> ClientUpdate:
    with _suppress_torch_numpy_warning():
        from fedbrew.clients.torch_fedprox_client import TorchFedProxClient

    return TorchFedProxClient(*args, **kwargs)


def _build_torch_scaffold_client(*args: Any, **kwargs: Any) -> ClientUpdate:
    with _suppress_torch_numpy_warning():
        from fedbrew.clients.torch_scaffold_client import TorchScaffoldClient

    return TorchScaffoldClient(*args, **kwargs)


def _build_synthetic_dataset(*args: Any, **kwargs: Any) -> FederatedDataset:
    with _suppress_torch_numpy_warning():
        from fedbrew.data.synthetic_classification import SyntheticClassificationDataset

    return SyntheticClassificationDataset(*args, **kwargs)


def _build_manifest_dataset(*args: Any, **kwargs: Any) -> FederatedDataset:
    with _suppress_torch_numpy_warning():
        from fedbrew.data.manifest_dataset import ManifestFederatedDataset

    return ManifestFederatedDataset(*args, **kwargs)


def _build_torch_mlp(*args: Any, **kwargs: Any) -> Model:
    with _suppress_torch_numpy_warning():
        from fedbrew.models.torch_mlp import build_torch_mlp

    return build_torch_mlp(*args, **kwargs)


def _build_torch_cnn(*args: Any, **kwargs: Any) -> Model:
    with _suppress_torch_numpy_warning():
        from fedbrew.models.torch_cnn import build_torch_cnn

    return build_torch_cnn(*args, **kwargs)


def _build_torch_small_cnn(*args: Any, **kwargs: Any) -> Model:
    with _suppress_torch_numpy_warning():
        from fedbrew.models.torch_cnn import build_torch_small_cnn

    return build_torch_small_cnn(*args, **kwargs)


def _build_femnist_resnet18(*args: Any, **kwargs: Any) -> Model:
    with _suppress_torch_numpy_warning():
        from fedbrew.models.femnist_resnet import build_femnist_resnet18

    return build_femnist_resnet18(*args, **kwargs)


def _build_openimage_shufflenet(*args: Any, **kwargs: Any) -> Model:
    with _suppress_torch_numpy_warning():
        from fedbrew.models.openimage_shufflenet import build_openimage_shufflenet

    return build_openimage_shufflenet(*args, **kwargs)


def _build_tiny_gpt2(*args: Any, **kwargs: Any) -> Model:
    with _suppress_torch_numpy_warning():
        from fedbrew.models.tiny_gpt2 import build_tiny_gpt2

    return build_tiny_gpt2(*args, **kwargs)


def _build_hf_causal_lm(*args: Any, **kwargs: Any) -> Model:
    with _suppress_torch_numpy_warning():
        from fedbrew.models.hf_causal_lm import build_hf_causal_lm

    return build_hf_causal_lm(*args, **kwargs)


def _build_hf_causal_lm_lora(*args: Any, **kwargs: Any) -> Model:
    with _suppress_torch_numpy_warning():
        from fedbrew.models.hf_causal_lm_lora import build_hf_causal_lm_lora

    return build_hf_causal_lm_lora(*args, **kwargs)


def _build_torch_classification_task(*args: Any, **kwargs: Any) -> TaskAdapter:
    with _suppress_torch_numpy_warning():
        from fedbrew.tasks.classification.torch_classification import TorchClassificationTask

    return TorchClassificationTask(*args, **kwargs)


def _build_torch_causal_lm_task(*args: Any, **kwargs: Any) -> TaskAdapter:
    with _suppress_torch_numpy_warning():
        from fedbrew.tasks.causal_lm.torch_causal_lm import TorchCausalLMTask

    return TorchCausalLMTask(*args, **kwargs)


def _build_delta_sgd_client(*args: Any, **kwargs: Any) -> ClientUpdate:
    with _suppress_torch_numpy_warning():
        from fedbrew.clients.torch_delta_sgd_client import TorchDeltaSGDClient

    return TorchDeltaSGDClient(*args, **kwargs)


def _build_fedavg_ft_client(*args: Any, **kwargs: Any) -> ClientUpdate:
    with _suppress_torch_numpy_warning():
        from fedbrew.clients.fedavg_ft_client import FedAvgFTClient

    return FedAvgFTClient(*args, **kwargs)


def _build_fedlalr_client(*args: Any, **kwargs: Any) -> ClientUpdate:
    with _suppress_torch_numpy_warning():
        from fedbrew.clients.torch_fedlalr_client import TorchFedLALRClient

    return TorchFedLALRClient(*args, **kwargs)


def _build_fedlalr_server(*args: Any, **kwargs: Any) -> ServerStrategy:
    with _suppress_torch_numpy_warning():
        from fedbrew.servers.fedlalr import FedLALRServer

    return FedLALRServer(*args, **kwargs)


@contextmanager
def _suppress_torch_numpy_warning() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Failed to initialize NumPy:.*",
            category=UserWarning,
        )
        yield


def _register_once(registry: Registry[T], name: str, obj: T, **metadata: Any) -> None:
    """Register a built-in, unless this exact registration is already in place.

    `register_builtin_components()` is called from a dozen entry points and has
    to be idempotent, which is the whole reason this wrapper exists. But
    `if not registry.exists(name)` bought that by skipping *any* existing
    registration, including one made by something else -- and extensions are
    loaded at `load_config` time, before the builtins are registered. So an
    extension registering ``"fedavg"`` silently kept it and the package's own
    strategy never appeared, under a `Registry.register` whose contract is
    "names are registered once and never overwritten" and whose duplicate
    message names both origins.

    Idempotence needs only the narrower skip: a name this same function already
    registered, which ``origin(name) == BUILTIN`` says exactly. Anything else
    falls through to `register`, which raises with the message that was always
    meant for this. See FINDINGS.csv P10-F33.

    Not ``registry.get(name) is obj`` as well, which was the first attempt:
    `GeneratorRegistry.register` wraps what it is given in a fresh
    `GeneratorSpec`, so the stored object is never the argument and all seven
    generators re-registered and raised on the second call. The origin is the
    whole question; identity answered a different one.
    """

    if registry.exists(name) and registry.origin(name) == BUILTIN:
        return
    registry.register(name, obj, **metadata)
