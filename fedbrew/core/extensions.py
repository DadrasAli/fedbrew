"""Load the out-of-tree components a config names.

A run config names its extensions under ``experiment.extensions`` and a
generator config under ``dataset.extensions``. Each entry is either a path to
a ``.py`` file, resolved the way ``data.path`` is -- ``~`` and environment
variables expanded, relative to the working directory -- or the dotted name
of an importable module. Both forms go through one loader and one protocol:
the module defines ``register()``, which calls the registries in
``fedbrew.core.registry`` for whatever it adds. Nothing registers at import,
so importing an extension for its other definitions has no side effect.

An extension is loaded once per process per resolved location. A second
config naming the same file gets the same record back and ``register()`` is
not called again; a different file registering the same name is refused by
the registry, which names both origins.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import ModuleType

from fedbrew.core import registry
from fedbrew.core.paths import expand_path
from fedbrew.core.refusal import RunRefused

#: The attribute an extension module must define.
REGISTER_HOOK = "register"

#: What a dotted module name looks like. Anything else is taken to be a path,
#: which must then end in ``.py``.
_MODULE_NAME = re.compile(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)*$")

#: The package synthetic module names are placed under in ``sys.modules``, so
#: that two files with the same basename cannot shadow one another and neither
#: can shadow an installed module.
_FILE_MODULE_PREFIX = "fedbrew_extensions"


@dataclass(frozen=True, slots=True)
class LoadedExtension:
    """One loaded extension: what the config wrote, and what it turned into."""

    #: The entry as written in the config, which is also the origin every
    #: name it registered carries.
    entry: str
    #: The absolute path of the file, or the module's name.
    resolved: str
    #: SHA-256 of the file the module was loaded from. The package's git
    #: state does not cover an out-of-tree file, so this is what run.json can
    #: record about which version of the extension ran. None only for a module
    #: with no file, such as a namespace package.
    sha256: str | None
    #: ``(registry label, name)`` for everything ``register()`` added.
    registered: tuple[tuple[str, str], ...]


#: Resolved location -> the record of its one load.
_loaded: dict[str, LoadedExtension] = {}


def load_extensions(entries: Sequence[str]) -> list[LoadedExtension]:
    """Import every entry, call its ``register()`` once, and return the records.

    Args:
        entries: The ``extensions`` list from a config, in order. Each is a
            path ending in ``.py`` or a dotted module name.

    Returns:
        One record per entry, in the order given. An entry already loaded in
        this process returns its existing record.

    Raises:
        RunRefused: If an entry is not a string, names a file or a module that
            does not exist, is neither a ``.py`` path nor a module name, or
            resolves to a module with no callable ``register``.
        ImportError: If an extension's own imports fail while it loads.
            Propagated with its own message, which names the real problem.
    """

    if isinstance(entries, str) or not isinstance(entries, Sequence):
        raise RunRefused("extensions must be a list of paths or module names")
    records = []
    for entry in entries:
        records.append(_load_one(entry))
    return records


def loaded_extensions() -> list[LoadedExtension]:
    """Every extension loaded so far in this process, in load order."""

    return list(_loaded.values())


def _load_one(entry: object) -> LoadedExtension:
    if not isinstance(entry, str) or not entry.strip():
        raise RunRefused(
            f"an extensions entry must be a path ending in .py or a module name, got {entry!r}"
        )
    entry = entry.strip()
    if entry.endswith(".py"):
        path = expand_path(entry).resolve()
        if not path.is_file():
            raise RunRefused(
                f"extensions entry {entry!r} does not exist: {path}. A path is "
                "resolved like data.path, relative to the working directory."
            )
        resolved = str(path)
        if resolved in _loaded:
            return _loaded[resolved]
        module = _import_file(path)
        sha256: str | None = _sha256(path)
    elif _MODULE_NAME.match(entry):
        resolved = entry
        if resolved in _loaded:
            return _loaded[resolved]
        module = _import_named(entry)
        module_file = getattr(module, "__file__", None)
        sha256 = _sha256(Path(module_file)) if module_file else None
    else:
        raise RunRefused(
            f"extensions entry {entry!r} is neither a path ending in .py nor a dotted module name"
        )

    hook = getattr(module, REGISTER_HOOK, None)
    if not callable(hook):
        raise RunRefused(
            f"extension {entry} defines no {REGISTER_HOOK}() function. An "
            "extension module registers its components from a register() "
            "function that calls fedbrew.core.registry; nothing is registered "
            "at import."
        )
    with registry.registering_from(entry) as registered:
        hook()
    record = LoadedExtension(
        entry=entry,
        resolved=resolved,
        sha256=sha256,
        registered=tuple(registered),
    )
    _loaded[resolved] = record
    return record


def _import_named(entry: str) -> ModuleType:
    """Import a dotted extensions entry, refusing one that names no module.

    Only the caller can tell the two failures apart. A ModuleNotFoundError whose
    missing name is the entry, or a package the entry sits in, means the config
    named something that is not there. Any other ImportError was raised while
    the extension imported its own dependencies, and keeps its traceback.
    """

    try:
        return import_module(entry)
    except ModuleNotFoundError as error:
        missing = error.name
        if missing is None or not (entry == missing or entry.startswith(f"{missing}.")):
            raise
        raise RunRefused(
            f"extensions entry {entry!r} names a module that cannot be imported: no "
            f"module named {missing!r}. A dotted entry is imported by name from this "
            "environment; a file is named by a path ending in .py."
        ) from error


def _import_file(path: Path) -> ModuleType:
    """Import one file under a synthetic, collision-free module name."""

    name = f"{_FILE_MODULE_PREFIX}.{_sha256(path)[:16]}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RunRefused(f"cannot import {path} as a module")
    module = importlib.util.module_from_spec(spec)
    # Registered before execution, as import does, so that the module's own
    # dataclasses and pickles resolve their module by name.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
