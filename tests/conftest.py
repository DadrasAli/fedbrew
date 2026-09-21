"""What the whole suite shares: one thread per process, the `fast` mark's
guard, and each shipped dataset validated once per process.

Measured on 2026-09-18 on the CPU login node chapter 13 names, before any of
this: the default suite took 38.6 minutes serially, and 20 of its 2,102 tests
took 33 of them. None of those 20 trains. Each preflights a shipped FEMNIST or
MNIST config, and preflight validates the dataset the config reads, which is
3,597 shards and 702 MB under `data/generated/femnist_natural` -- about 13 s a
call at one thread, and the slowest test made 13 calls. CI has no
`data/generated`, which is why the same tests cost nothing there. Chapter 13
section 2.2 has the numbers after.

**One thread per process.** torch starts 64 intra-op and 128 inter-op threads
in a fresh process on that node: 191 tasks against a per-user limit of 2,000,
so 16 xdist workers at the default would need about 3,056 and abort (the
OpenBLAS `pthread_create` failure). One thread was also faster for the work
that dominates: a FEMNIST preflight took 24-28 s at the default and 13.4 s at
one thread, because the default oversubscribes a 32-core quota. The
variables are set before anything imports torch or numpy, and every
subprocess a test starts inherits them.

**The `fast` mark.** `python -m pytest -m fast` is the gate for a change that
touches only text, configs or docs: the tests that read files and configs and
nothing heavier. A test marked `fast` must not start a subprocess, run autograd
backward, step an optimizer, iterate a DataLoader, or call `torch.load` or
`torch.save`, and this file fails it at teardown if it does. It checks in every
run, so the full gate and CI both enforce it. The marks were set from what
every test did with and without the `llm` extra installed, and in a clean export
with no generated data, which is what CI sees, so a test counts as fast only
if it did none of that anywhere. A test whose work lands in a class's
`setUpClass` is not marked on its own, because that work is charged to
whichever test of the class a worker runs first.

**Shipped datasets, validated once.** `validate_manifest` is memoized, for this
process only, when the manifest lives under `data/generated/`. The key is the
arguments, the working directory, and the size and modification time of every
file in the dataset's directory, so every shipped dataset is still read in
full the first time a process asks, and any change to it is read again. Data a
test writes for itself lives anywhere else and is never cached, so a test that
corrupts a shard and expects a refusal still gets one. `fedbrew` itself is
unchanged: this replaces the module attribute that `_validate_data` looks up
on each call, and restores it when the session ends.
"""

from __future__ import annotations

import collections
import copy
import functools
import os
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
for _name in THREAD_VARIABLES:
    os.environ[_name] = "1"

REPO_ROOT = Path(__file__).resolve().parent.parent
SHIPPED_DATA = (REPO_ROOT / "data" / "generated").resolve()

#: What a test marked `fast` must not do, in the words its failure uses.
HEAVY_WORK = {
    "subprocess": "started a subprocess",
    "backward": "ran autograd backward",
    "optimizer_step": "stepped an optimizer",
    "dataloader": "iterated a DataLoader",
    "torch_load": "called torch.load",
    "torch_save": "called torch.save",
}

#: What the running test has done so far, or None between tests.
_work: collections.Counter[str] | None = None


def _did(kind: str) -> None:
    if _work is not None:
        _work[kind] += 1


def _counting(kind: str, function: Any) -> Any:
    @functools.wraps(function)
    def counted(*args: Any, **kwargs: Any) -> Any:
        _did(kind)
        return function(*args, **kwargs)

    return counted


def pytest_configure(config: pytest.Config) -> None:
    import torch
    import torch.utils.data
    from torch.optim.optimizer import register_optimizer_step_pre_hook

    subprocess.Popen.__init__ = _counting("subprocess", subprocess.Popen.__init__)  # type: ignore[method-assign]
    torch.autograd.backward = _counting("backward", torch.autograd.backward)
    register_optimizer_step_pre_hook(lambda *args, **kwargs: _did("optimizer_step"))
    torch.utils.data.DataLoader.__iter__ = _counting(  # type: ignore[method-assign]
        "dataloader", torch.utils.data.DataLoader.__iter__
    )
    torch.load = _counting("torch_load", torch.load)
    torch.save = _counting("torch_save", torch.save)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None) -> Iterator[None]:
    global _work
    _work = collections.Counter()
    try:
        return (yield)
    finally:
        _work = None


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]) -> Iterator[Any]:
    report = yield
    if call.when == "teardown" and _work and item.get_closest_marker("fast") is not None:
        done = ", ".join(f"{HEAVY_WORK[kind]} ({count}x)" for kind, count in sorted(_work.items()))
        report.outcome = "failed"
        report.longrepr = (
            f"{item.nodeid} is marked fast, and it {done}. The fast gate "
            "(python -m pytest -m fast) holds only tests that read text and "
            "configs. Take the mark off this test: if it comes from the "
            "module's pytestmark or the class decorator, move the mark down "
            "to the neighbours that still qualify. Chapter 13 section 2.2."
        )
    return report


def _dataset_state(root: Path) -> tuple[tuple[str, int, int], ...]:
    state = []
    for directory, subdirectories, files in os.walk(root):
        subdirectories.sort()
        for name in sorted(files):
            status = os.stat(os.path.join(directory, name))
            state.append((os.path.join(directory, name), status.st_size, status.st_mtime_ns))
    return tuple(state)


@pytest.fixture(scope="session", autouse=True)
def _shipped_datasets_validated_once() -> Iterator[None]:
    from fedbrew.data import manifest_validation

    real = manifest_validation.validate_manifest
    results: dict[tuple[Any, ...], list[Any]] = {}

    @functools.wraps(real)
    def validate_manifest(
        manifest_path: str | Path,
        *,
        require_client_test: bool = False,
        require_global_test: bool = False,
    ) -> list[Any]:
        path = Path(manifest_path).resolve()
        if not path.is_file() or not path.is_relative_to(SHIPPED_DATA):
            return real(
                manifest_path,
                require_client_test=require_client_test,
                require_global_test=require_global_test,
            )
        key = (
            os.fspath(manifest_path),
            os.getcwd(),
            require_client_test,
            require_global_test,
            _dataset_state(path.parent),
        )
        if key not in results:
            results[key] = real(
                manifest_path,
                require_client_test=require_client_test,
                require_global_test=require_global_test,
            )
        return [copy.copy(issue) for issue in results[key]]

    manifest_validation.validate_manifest = validate_manifest
    try:
        yield
    finally:
        manifest_validation.validate_manifest = real
