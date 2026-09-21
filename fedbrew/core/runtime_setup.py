"""Runtime configuration helpers for local and HPC execution."""

from __future__ import annotations

import os
import random
from collections.abc import Mapping
from typing import Any

from fedbrew.core.config import FullConfig

# Re-exported: the derivation itself lives in the leaf module fedbrew.core.seeding
# so fedbrew/servers/ and fedbrew/data/ can use it without importing the config layer.
from fedbrew.core.seeding import (  # noqa: F401
    client_seed,
    dataloader_seed,
    derive_seed,
)

_UINT32_MODULUS = 2**32
_CUBLAS_WORKSPACE_CONFIG = "CUBLAS_WORKSPACE_CONFIG"
_DEFAULT_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def configure_deterministic_environment(deterministic: bool) -> str | None:
    """Set process environment required by deterministic CUDA operations."""

    if deterministic:
        os.environ.setdefault(
            _CUBLAS_WORKSPACE_CONFIG,
            _DEFAULT_CUBLAS_WORKSPACE_CONFIG,
        )
    return os.environ.get(_CUBLAS_WORKSPACE_CONFIG)


def seed_everything(
    seed: int | None,
    deterministic: bool = False,
    warn_only: bool = False,
) -> dict[str, object]:
    """Seed available RNGs and return reproducibility metadata.

    warn_only defaults to False, so `deterministic=True` means the run is
    deterministic rather than that it warns once on stderr when it is not.
    Under warn_only=True torch prints "Memory Efficient attention defaults to a
    non-deterministic algorithm" and carries on; the warning is emitted once per
    process by Python's default filter, goes to logs_and_errs/*.err, and is not
    recorded anywhere. Measured on an A100-40GB (torch 2.5.1, CUDA 11.8, August
    2026), Qwen2.5-0.5B at the shipped shape gave 2 distinct gradient digests
    over 6 identical passes with warn_only=True and 1 with warn_only=False.
    """

    cublas_workspace_config = configure_deterministic_environment(deterministic)
    metadata = _base_seed_metadata(seed, deterministic, warn_only)
    metadata["cublas_workspace_config"] = cublas_workspace_config
    normalized_seed = _normalize_seed(seed) if seed is not None else None

    if seed is not None:
        random.seed(int(seed))

    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - depends on local environment.
        metadata["numpy_error"] = str(exc)
    else:
        metadata["numpy_version"] = getattr(np, "__version__", None)
        if normalized_seed is not None:
            np.random.seed(normalized_seed)

    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on local environment.
        metadata["torch_error"] = str(exc)
        return metadata

    metadata.update(_torch_runtime_metadata(torch))
    if normalized_seed is not None:
        torch.manual_seed(normalized_seed)
        if hasattr(torch, "cuda"):
            torch.cuda.manual_seed(normalized_seed)
            torch.cuda.manual_seed_all(normalized_seed)

    if deterministic:
        _enable_torch_determinism(torch, warn_only=warn_only)

    metadata.update(_torch_runtime_metadata(torch))
    return metadata


def capture_rng_state() -> dict[str, Any]:
    """Snapshot every process-wide RNG a round can consume.

    nn.Dropout draws from the process-wide default generator, not from any
    per-client stream, so a run's position in that stream is part of its state
    just as much as its weights are. Restoring weights, server state and client
    state but not this leaves a resumed run internally deterministic and
    disagreeing with the uninterrupted run it claims to continue.

    Same optional-dependency handling as seed_everything: numpy and torch may
    legitimately be absent, and a checkpoint is still worth writing without
    them.
    """

    state: dict[str, Any] = {"python": random.getstate()}
    try:
        import numpy as np
    except Exception:  # pragma: no cover - depends on local environment.
        pass
    else:
        state["numpy"] = np.random.get_state()

    try:
        import torch
    except Exception:  # pragma: no cover - depends on local environment.
        return state

    state["torch"] = torch.get_rng_state()
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        # Every visible device: a run can move between them across a requeue,
        # and get_rng_state_all/set_rng_state_all is the pair that round-trips.
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> list[str]:
    """Restore what capture_rng_state saved; return the streams restored.

    Missing entries are skipped rather than raising: a checkpoint written on a
    CUDA node and resumed on a CPU one is a normal requeue outcome, and a
    partially restored stream still beats a freshly seeded one.
    """

    restored: list[str] = []
    python_state = state.get("python")
    if python_state is not None:
        random.setstate(tuple(python_state) if isinstance(python_state, list) else python_state)
        restored.append("python")

    numpy_state = state.get("numpy")
    if numpy_state is not None:
        try:
            import numpy as np
        except Exception:  # pragma: no cover - depends on local environment.
            pass
        else:
            np.random.set_state(
                tuple(numpy_state) if isinstance(numpy_state, list) else numpy_state
            )
            restored.append("numpy")

    try:
        import torch
    except Exception:  # pragma: no cover - depends on local environment.
        return restored

    torch_state = state.get("torch")
    if torch_state is not None:
        torch.set_rng_state(torch_state.cpu().to(torch.uint8))
        restored.append("torch")

    cuda_state = state.get("torch_cuda")
    if (
        cuda_state
        and hasattr(torch, "cuda")
        and torch.cuda.is_available()
        and len(cuda_state) == torch.cuda.device_count()
    ):
        torch.cuda.set_rng_state_all([t.cpu().to(torch.uint8) for t in cuda_state])
        restored.append("torch_cuda")
    return restored


def configure_runtime(config: FullConfig, deterministic: bool) -> dict[str, object]:
    """Apply optional runtime performance settings and return runtime info.

    ``deterministic`` is passed in rather than read from ``config`` here. It
    used to be read in both places, each supplying its own ``False`` default.

    The two readers did apply different rules -- the caller's goes through a
    bool check that raises on ``deterministic: "false"``, this one took any
    truthy value -- but that difference was not reachable: ``validate_config``
    runs ``_validate_extra_bools`` over both determinism keys before either
    reader sees them, so a non-bool never arrives from ``load_config``. It
    would take a ``FullConfig`` built in code to tell the two apart.

    The reason to have one read is the default, not the parsing. Two
    independently defaulted reads of one key agree until one of them changes,
    and nothing marks the moment it does.
    """

    requested_device = config.runtime.device
    resolved_device = requested_device
    torch_available = False
    cuda_available = None
    cuda_device_name = None
    performance = config.runtime.extra.get("performance", {})
    if not isinstance(performance, Mapping):
        performance = {}

    # torch is the one optional part, and the only thing this catches. The
    # `try` used to span the CUDA probe and all three performance settings as
    # well, under a bare `except Exception` whose handler returned early: a
    # `torch_num_threads` that int() could not read left cudnn_benchmark and
    # matmul_precision unapplied and absent from the returned record, so the
    # run trained at torch's default precision while the config asked for
    # another and nothing on disk said which had happened. Measured on the
    # pre-fix tree, `matmul_precision: high` beside `torch_num_threads:
    # "not-an-int"`, `0`, or `-4`: torch held "highest" in all three, and the
    # record carried a `runtime_setup_error` string nothing reads. P04-F07.
    try:
        import torch

        from fedbrew.core.torch_utils import resolve_torch_device
    except ImportError as exc:
        return {
            "requested_device": requested_device,
            "resolved_device": resolved_device,
            "torch_available": torch_available,
            "cuda_available": cuda_available,
            "cuda_device_name": cuda_device_name,
            "runtime_setup_error": str(exc),
        }

    torch_available = True
    cuda_available, cuda_device_name, probe_error = _probe_cuda(torch)
    if requested_device == "auto":
        # resolve_torch_device probes CUDA itself, so it is asked only when
        # the probe above found some: a driver that cannot answer
        # `is_available` answers once rather than twice, and `auto` on such a
        # node resolves to the CPU rather than raising out of a metadata call.
        resolved_device = resolve_torch_device(requested_device) if cuda_available else "cpu"

    # Applied, not attempted. Each of these three is a value validate_config
    # has already checked -- _validate_performance_values covers all three --
    # so a failure here is a fault in the environment rather than a typo in
    # the config, and a run that cannot apply its own numerics settings must
    # not continue as though it had.
    torch_num_threads = performance.get("torch_num_threads")
    if torch_num_threads is not None:
        torch.set_num_threads(int(torch_num_threads))

    cudnn_benchmark = performance.get("cudnn_benchmark")
    if cudnn_benchmark is not None and not bool(deterministic) and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)

    matmul_precision = performance.get("matmul_precision")
    if matmul_precision and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(str(matmul_precision))

    torch_metadata = _torch_runtime_metadata(torch)
    record: dict[str, object] = {
        "requested_device": requested_device,
        "resolved_device": resolved_device,
        "torch_available": torch_available,
        "cuda_available": cuda_available,
        "cuda_device_name": cuda_device_name,
        "torch_num_threads": performance.get("torch_num_threads"),
        # What torch actually holds, not what the config asked for -- the two
        # differ when the key is unset (torch defaults to "highest") and
        # matmul_precision is the one performance key that changes the
        # numbers, so run.json has to record the fact rather than the request.
        # Same rule cudnn_benchmark below already follows.
        "matmul_precision": torch_metadata["matmul_precision"],
        "cudnn_benchmark": torch_metadata["cudnn_benchmark"],
        "cudnn_deterministic": torch_metadata["cudnn_deterministic"],
        "torch_deterministic_algorithms": torch_metadata["torch_deterministic_algorithms"],
    }
    if probe_error is not None:
        record["cuda_probe_error"] = probe_error
    return record


def _probe_cuda(torch: Any) -> tuple[bool | None, str | None, str | None]:
    """What CUDA this node has, as provenance rather than as a setting.

    Guarded, unlike the three settings above it: a driver that cannot answer
    `get_device_name` is a fact about the node worth recording, and a CPU run
    on such a node is still a run. The reason is returned rather than dropped,
    so `run.json` carries it.

    Args:
        torch: The imported module.

    Returns:
        ``(cuda_available, cuda_device_name, probe_error)``. The first is None
        only when the probe itself failed.
    """

    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:
        return None, None, str(exc)
    if not cuda_available:
        return False, None, None
    try:
        return True, str(torch.cuda.get_device_name(0)), None
    except Exception as exc:
        return True, None, str(exc)


def _base_seed_metadata(
    seed: int | None,
    deterministic: bool,
    warn_only: bool,
) -> dict[str, object]:
    return {
        "seed": seed,
        "deterministic": deterministic,
        "deterministic_warn_only": warn_only,
        "cudnn_deterministic": None,
        "cudnn_benchmark": None,
        "torch_deterministic_algorithms": None,
        "matmul_precision": None,
        "torch_version": None,
        "cuda_available": None,
        "cuda_version": None,
        "cudnn_version": None,
        "numpy_version": None,
    }


def _normalize_seed(seed: int | None) -> int:
    if seed is None:
        raise ValueError("seed cannot be None")
    return int(seed) % _UINT32_MODULUS


def _enable_torch_determinism(torch_module: Any, warn_only: bool) -> None:
    use_deterministic = getattr(torch_module, "use_deterministic_algorithms", None)
    if callable(use_deterministic):
        try:
            use_deterministic(True, warn_only=warn_only)
        except TypeError:  # pragma: no cover - for older torch signatures.
            use_deterministic(True)

    backends = getattr(torch_module, "backends", None)
    cudnn = getattr(backends, "cudnn", None)
    if cudnn is not None:
        if hasattr(cudnn, "benchmark"):
            cudnn.benchmark = False
        if hasattr(cudnn, "deterministic"):
            cudnn.deterministic = True


def _torch_runtime_metadata(torch_module: Any) -> dict[str, object]:
    cuda_available: bool | None = False
    if hasattr(torch_module, "cuda"):
        try:
            cuda_available = bool(torch_module.cuda.is_available())
        except Exception:
            # Provenance, not a setting. A driver that cannot answer is a
            # fact about the node and not a reason to stop a CPU run; the
            # reason itself is recorded by _probe_cuda, and seed_everything
            # calls this before any probe has run. P04-F07.
            cuda_available = None

    cuda_version = getattr(getattr(torch_module, "version", None), "cuda", None)
    cudnn_version = None
    cudnn_benchmark = None
    cudnn_deterministic = None
    backends = getattr(torch_module, "backends", None)
    cudnn = getattr(backends, "cudnn", None)
    if cudnn is not None:
        if hasattr(cudnn, "benchmark"):
            cudnn_benchmark = bool(cudnn.benchmark)
        if hasattr(cudnn, "deterministic"):
            cudnn_deterministic = bool(cudnn.deterministic)
        version = getattr(cudnn, "version", None)
        if callable(version):
            cudnn_version = version()

    deterministic_algorithms = None
    are_enabled = getattr(torch_module, "are_deterministic_algorithms_enabled", None)
    if callable(are_enabled):
        deterministic_algorithms = bool(are_enabled())

    matmul_precision = None
    get_precision = getattr(torch_module, "get_float32_matmul_precision", None)
    if callable(get_precision):
        matmul_precision = str(get_precision())

    return {
        "matmul_precision": matmul_precision,
        "torch_version": getattr(torch_module, "__version__", None),
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "cudnn_version": cudnn_version,
        "cudnn_deterministic": cudnn_deterministic,
        "cudnn_benchmark": cudnn_benchmark,
        "torch_deterministic_algorithms": deterministic_algorithms,
    }
