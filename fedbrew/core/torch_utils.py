"""Utilities for PyTorch model state handling."""

from __future__ import annotations

import copy
import math
import random
import warnings
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor, nn

StateDict = dict[str, Any]


@runtime_checkable
class OptimizerLike(Protocol):
    """The two methods a task's `train_step` calls on the optimizer it is given.

    Four update rules hand `train_step` an object that wraps a real optimizer
    rather than being one: `_ClippingOptimizer`, `_GradientOnlyOptimizer`,
    `_ScaffoldCorrectingOptimizer` and `_FedProxCorrectingOptimizer`. That is a
    deliberate shape -- SCAFFOLD's correction and FedProx's proximal term only
    ever mutate `.grad`, so wrapping `.step()` keeps every caller on the plain
    `task.train_step(model, batch, optimizer)` signature instead of growing
    hook plumbing into the task adapters that two of nine rules need.

    Typed as `Optimizer`, that shape was five mypy errors reporting one
    decision five times, and two of the suppressions written for them sat on
    the call line while mypy blames the argument line, so they suppressed
    nothing and `warn_unused_ignores = false` kept that invisible. A protocol
    says the same thing as a checked claim: these objects are usable here
    because they answer `zero_grad` and `step`, which is exactly what the two
    `train_step` implementations call.

    `param_groups` is deliberately not required. Three of the four wrappers
    expose it and `_GradientOnlyOptimizer` does not, because nothing in a
    training step reads it -- requiring it here would exclude a caller that
    works.
    """

    def zero_grad(self, set_to_none: bool = ...) -> None:
        """Clear the gradients accumulated on the wrapped parameters."""

    def step(self, closure: Any | None = ...) -> Any:
        """Apply one update. The return value is unused by every caller."""


class SeedWorker:
    """Seed a DataLoader worker's Python and NumPy RNGs, per epoch.

    A DataLoader spawns its workers afresh for each iterator, and gives each
    one ``base_seed + worker_id`` as its torch seed, where ``base_seed`` is
    drawn from the loader's ``generator`` whenever a new iterator is made --
    every epoch, unless ``persistent_workers`` keeps the workers, and their
    seeds, from one epoch to the next. That is what makes a seeded loader
    reproducible *and* different each epoch. torch's own worker loop then seeds
    ``random`` from that value, and NumPy from ``_generate_state(base_seed,
    worker_id)`` when NumPy is importable (``torch/utils/data/_utils/worker.py``),
    so the per-worker, per-epoch variation in those two RNGs is torch's doing
    and not this hook's.

    What the hook adds is that the rule is stated here rather than inherited.
    Both RNGs are seeded from ``torch.initial_seed()``, one definition for both
    task adapters, so a run's in-worker streams are defined by this repository
    and do not move if torch changes its recipe. The NumPy seed it sets is
    derived from the same value as ``random``'s, rather than from torch's
    separate spread.

    The two copies this replaces -- one per task adapter, identical -- took a
    ``base_seed`` fixed at construction and set every worker RNG to
    ``base_seed + worker_id``, so every epoch of a round replayed the same
    in-worker stream. Measured at ``num_workers=2`` over three epochs of a
    dataset drawing ``random.random()`` in ``__getitem__``: all three epochs
    returned byte-identical values, where torch's own recipe returned three
    different ones and still reproduced exactly under a fixed generator seed.
    Nothing in a shipped dataset draws randomness today, so this was dormant
    rather than wrong -- and it was unreachable besides, until
    ``runtime.num_workers`` was wired through to the loader. Enabling workers
    for throughput is the change that wakes it. P09-F08(c).

    ``torch.manual_seed`` is deliberately *not* called here. torch has already
    seeded the worker with the full 64-bit ``base_seed + worker_id``; setting
    it again to that value truncated to 32 bits would discard entropy and
    replace a per-epoch seed with a narrower one.

    Stateless, and a class rather than a closure only because a
    ``worker_init_fn`` must survive pickling to reach a worker process.
    """

    def __call__(self, worker_id: int) -> None:
        """Seed this worker from the per-epoch value torch gave it.

        Args:
            worker_id: The worker's index. Unused: torch has already folded it
                into ``initial_seed()``, and adding it a second time would
                make worker *w* of epoch *n* collide with worker *w+1* of an
                epoch whose base seed happened to be one lower.
        """

        worker_seed = torch.initial_seed() % (2**32)
        random.seed(worker_seed)
        try:
            import numpy as np
        except Exception:  # pragma: no cover - depends on optional numpy import.
            pass
        else:
            np.random.seed(worker_seed)


def resolve_torch_device(device: str) -> str:
    """Resolve cpu/cuda/auto to a usable PyTorch device string.

    Args:
        device: ``"cpu"``, ``"cuda"`` or ``"auto"``. Anything else is
            returned unchanged, for a caller naming a device index.

    Returns:
        The device string. ``"auto"`` becomes ``"cuda"`` only when a CUDA
        allocation actually succeeds.

    Warns:
        RuntimeWarning: When CUDA is present but the allocation fails, so
            ``auto`` falls back to the CPU. A node with no CUDA at all is not
            surprising and says nothing; a node that *has* a GPU and cannot
            hand one out is the case that used to fall back without a word,
            leaving a run an order of magnitude slower with nothing but
            `resolved_device` in `run.json` to say why. P04-F07.
    """

    if device != "auto":
        return device
    if not torch.cuda.is_available():
        return "cpu"
    try:
        torch.empty(1, device="cuda")
    except Exception as exc:
        warnings.warn(
            f"device=auto found CUDA but could not allocate on it ({exc}); "
            "running on the CPU. Set runtime.device explicitly to make this "
            "a choice rather than a fallback.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "cpu"
    return "cuda"


def get_model_state(model: nn.Module) -> StateDict:
    """Return a cloned CPU copy of a model state dict."""

    return clone_model_state(model.state_dict())


# Attribute stamped on a model instance naming the state dict object currently
# resident in its weights, or absent/None when that is unknown. Read by the
# client evaluation path to skip reloading a state the model already holds.
RESIDENT_STATE_ATTR = "_fl_resident_model_state"


def forget_resident_state(model: nn.Module) -> None:
    """Drop any claim about which state dict is resident in ``model``.

    Called from every function that overwrites model weights. Local training
    then mutates those weights further, so "unknown" is the only safe answer
    after a load, and whoever knows better re-stamps it afterwards.
    """

    setattr(model, RESIDENT_STATE_ATTR, None)


def load_model_state(model: nn.Module, state: Mapping[str, Any]) -> None:
    """Load a cloned state dict into a model."""

    forget_resident_state(model)
    model.load_state_dict(clone_model_state(state))


def tied_state_aliases(model: nn.Module) -> dict[str, str]:
    """Map each duplicate state key to the key whose storage it shares.

    Weight tying makes one tensor appear under several state-dict keys — Qwen
    ties ``lm_head.weight`` to ``model.embed_tokens.weight``, for example. Those
    duplicates are the same parameter, so federating them all would transmit and
    average identical values more than once.
    """

    aliases: dict[str, str] = {}
    seen: dict[tuple[Any, ...], str] = {}
    for name, value in model.state_dict().items():
        if not isinstance(value, Tensor) or value.data_ptr() == 0:
            continue
        identity = (value.data_ptr(), tuple(value.shape), value.dtype)
        if identity in seen:
            aliases[name] = seen[identity]
        else:
            seen[identity] = name
    return aliases


def persistent_buffer_keys(model: nn.Module) -> list[str]:
    """State-dict keys that are neither parameters nor tied aliases of one.

    What is left after those two exclusions is registered buffer state that
    ``load_state_dict`` expects: BatchNorm's ``running_mean``, ``running_var``
    and ``num_batches_tracked``, and anything a custom module registers with
    ``persistent=True``.

    Both exclusions are load-bearing, and measured rather than assumed:

    - ``named_buffers()`` alone is the wrong question. It reports GPT-2's
      ``attn.bias`` causal mask and Qwen2's ``rotary_emb.inv_freq``, neither of
      which is persistent, so neither is in the state dict and neither is ever
      federated. A check built on it would refuse both LLM models.
    - The tied-alias exclusion is what keeps GPT-2 out. ``lm_head.weight`` is
      in the state dict and is not in ``named_parameters()``, because it shares
      storage with ``transformer.wte.weight`` -- it is one parameter under two
      names, which is :func:`tied_state_aliases`' whole subject.

    With both, every model this repository ships returns an empty list: the
    five classification builders, tiny GPT-2, and a Qwen2 built from config.
    """

    parameters = set(dict(model.named_parameters()))
    aliases = set(tied_state_aliases(model))
    return sorted(
        name for name in model.state_dict() if name not in parameters and name not in aliases
    )


def get_untied_model_state(model: nn.Module) -> StateDict:
    """Return the model state with tied duplicates removed."""

    aliases = tied_state_aliases(model)
    return clone_model_state(
        {name: value for name, value in model.state_dict().items() if name not in aliases}
    )


def load_untied_model_state(model: nn.Module, state: Mapping[str, Any]) -> None:
    """Load a state whose tied duplicates were removed, restoring them first."""

    forget_resident_state(model)
    restored = clone_model_state(state)
    for duplicate, source in tied_state_aliases(model).items():
        if duplicate in restored:
            continue
        if source not in restored:
            raise ValueError(
                f"model state is missing {source!r}, needed to restore the tied "
                f"parameter {duplicate!r}"
            )
        restored[duplicate] = restored[source]
    model.load_state_dict(restored)


def clone_model_state(state: Mapping[str, Any]) -> StateDict:
    """Clone state values without sharing tensor references."""

    cloned: StateDict = {}
    for key, value in state.items():
        cloned[key] = _clone_value(value)
    return cloned


class NonFiniteStateError(ValueError):
    """A client sent a model state that is not finite.

    Its own class because the loop turns it into a recorded "diverged" outcome
    rather than a traceback: a run whose model blew up is a result, not a crash.
    """


def refuse_non_finite_state(state: Mapping[str, Any], what: str) -> None:
    """Raise :class:`NonFiniteStateError` if a floating tensor in ``state`` is not finite.

    The one finiteness check every aggregation path uses: the accumulator
    applies it to each client state it folds, and a strategy applies it to
    auxiliary state it sums outside the accumulator -- SCAFFOLD's control
    deltas and the control variate they produce (FINDINGS.csv POST-F27).
    Integer tensors are skipped, as the accumulator skips them.

    Args:
        state: Name to tensor.
        what: What the state is, for the message: ``"client state"``.
    """

    for key, value in state.items():
        tensor = value.detach() if isinstance(value, Tensor) else as_cpu_tensor(key, value)
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise NonFiniteStateError(f"{what} tensor {key!r} contains non-finite values")


#: Low-precision dtypes the running sum is promoted out of, and what to.
#: float32 and float64 accumulate in themselves; see _accumulation_dtype.
_PROMOTED_ACCUMULATION_DTYPES: dict[torch.dtype, torch.dtype] = {
    torch.bfloat16: torch.float32,
    torch.float16: torch.float32,
}


def _accumulation_dtype(dtype: torch.dtype) -> torch.dtype:
    """The dtype a running weighted sum over that input should be kept in.

    The accumulator took its precision from whatever the client sent, which is
    correct for float32 and float64 and destroys a low-precision state. A
    weighted sum over a federation is thousands of additions into one running
    total whose magnitude grows the whole time, so the terms stop being
    representable long before the mean is computed. Measured through this
    class, FEMNIST-shaped weights (16-525 examples) and 3597 clients:

        dtype       median rel. error   max rel. error
        float32              9.6e-07          4.0e-04
        bfloat16             6.0e-02          8.0e+00
        float16              6.3e-03          6.4e+00

    and float16 does not merely lose precision. On a coordinate whose sign
    does not change -- a bias, a post-ReLU weight column, any real parameter,
    since a random zero-mean draw hides this -- the running total reaches
    65504 and becomes inf: after 3290 clients at |value| 0.05, 791 at 0.2, and
    153 at 1.0.

    float32 is left alone deliberately. Its worst case above is a relative
    error on a coordinate whose true mean is near zero, so the denominator is
    what is small; promoting it to float64 would double the one buffer this
    class exists to keep at one model state, to fix nothing that has been
    shown to matter.

    Nothing today sends bf16 or fp16 -- hf_causal_lm loads at torch's default
    float32. The obvious next edit does: torch_dtype=torch.bfloat16 to fit a
    Qwen round in less memory, or a bf16 LoRA adapter.
    """

    return _PROMOTED_ACCUMULATION_DTYPES.get(dtype, dtype)


class WeightedStateAccumulator:
    """Accumulate a running weighted mean of model states one client at a time.

    Buffering every client state before averaging costs one full model copy per
    participating client, which is prohibitive for cross-device rounds with
    thousands of clients. This accumulator keeps a running sum instead, so what
    it retains is two model states -- the sum, and the reference copy below --
    regardless of participation. Measured at 4, 16 and 64 clients by
    tests/test_aggregation_peak_memory.py, which is also what would catch a
    change back to buffering: the averaged result is identical either way.
    """

    def __init__(self) -> None:
        """Start empty; the key set and dtypes are fixed by the first state.

        Sums are kept on the CPU regardless of where the incoming tensors live,
        so a round's peak GPU memory does not grow with participation.
        """

        self._totals: StateDict = {}
        self._reference: dict[str, Tensor] = {}
        self._total_weight = 0.0
        self._count = 0

    def add(self, state: Mapping[str, Any], weight: float) -> None:
        """Add one weighted client state to the running sum.

        Args:
            state: One client's model state -- parameter name to Tensor. Every
                value must be a Tensor, and after the first call the key set
                must match the first state's exactly.
            weight: This client's aggregation weight, in whatever units the
                strategy uses (example counts for ``examples`` weighting, 1.0
                for ``uniform``). Must be finite. Only the ratio between
                weights matters, since
                :meth:`result` divides by their total.

        Raises:
            NonFiniteStateError: If ``weight`` is NaN or infinite. Caught here
                rather than at the end, so the offending client is identifiable.
            ValueError: If the key set differs from the first state's -- which
                would otherwise silently average a subset of the model.
            TypeError: If any value is not a Tensor.
        """

        weight = float(weight)
        # Every strategy aggregates through this class, so a non-finite weight
        # is refused here once rather than in each server.
        if not math.isfinite(weight):
            raise NonFiniteStateError(f"client weight must be finite, got {weight}")
        if self._count and set(state.keys()) != set(self._reference.keys()):
            raise ValueError("all states must have the same keys")

        for key, value in state.items():
            if not isinstance(value, Tensor):
                raise TypeError(f"state value for {key} is not a tensor")
            value_cpu = value.detach().cpu()
            reference = self._reference.get(key)
            if reference is None:
                self._reference[key] = value_cpu
                if value_cpu.is_floating_point():
                    self._totals[key] = torch.zeros_like(
                        value_cpu, dtype=_accumulation_dtype(value_cpu.dtype)
                    )
            elif value_cpu.dtype != reference.dtype or value_cpu.shape != reference.shape:
                raise ValueError(
                    f"state tensor {key!r} must have matching dtype and shape across clients"
                )

            if value_cpu.is_floating_point():
                # Caught here rather than downstream because downstream is
                # forever: FedOpt's v <- b2*v + (1-b2)*d^2, FedLALR's second
                # moment and SCAFFOLD's control variates all keep a NaN
                # permanently (measured: still NaN after 20 clean rounds), and
                # save_state writes it into latest.pt, so --resume-latest picks
                # a dead run back up. One poisoned client would otherwise take
                # the whole round's weighted mean with it.
                refuse_non_finite_state({key: value_cpu}, "client state")
                self._totals[key].add_(value_cpu, alpha=weight)
            elif self._count and not torch.equal(value_cpu, self._reference[key]):
                raise ValueError(f"non-floating state tensor {key!r} differs between clients")

        self._total_weight += weight
        self._count += 1

    def result(self) -> StateDict:
        """Return the weighted mean and reset the accumulator."""

        if not self._count:
            raise ValueError("states must not be empty")
        if self._total_weight == 0.0:
            raise ValueError("weights sum must not be zero")

        averaged: StateDict = {}
        for key, reference in self._reference.items():
            if reference.is_floating_point():
                # Back to the model's dtype: the accumulation dtype is this
                # class's business and the state's is the model's contract.
                averaged[key] = self._totals[key].div_(self._total_weight).to(reference.dtype)
            else:
                averaged[key] = reference.clone()

        self._totals = {}
        self._reference = {}
        self._total_weight = 0.0
        self._count = 0
        return averaged


def subtract_model_states(a: Mapping[str, Any], b: Mapping[str, Any]) -> StateDict:
    """Return a - b for matching tensor-only model states."""

    return _binary_tensor_state(a, b, lambda left, right: left - right)


def add_model_states(a: Mapping[str, Any], b: Mapping[str, Any]) -> StateDict:
    """Return a + b for matching tensor-only model states."""

    return _binary_tensor_state(a, b, lambda left, right: left + right)


def scale_model_state(state: Mapping[str, Any], scale: float) -> StateDict:
    """Return state scaled by a scalar without mutating the input."""

    scaled: StateDict = {}
    for key, value in state.items():
        tensor = as_cpu_tensor(key, value)
        scaled[key] = tensor * float(scale)
    return scaled


def zeros_like_model_state(state: Mapping[str, Any]) -> StateDict:
    """Return a CPU zero tensor state matching the input tensor shapes."""

    zeros: StateDict = {}
    for key, value in state.items():
        tensor = as_cpu_tensor(key, value)
        zeros[key] = torch.zeros_like(tensor)
    return zeros


def sqrt_model_state(state: Mapping[str, Any], eps: float = 0.0) -> StateDict:
    """Return tensor-wise sqrt(state + eps) on CPU."""

    rooted: StateDict = {}
    for key, value in state.items():
        tensor = as_cpu_tensor(key, value)
        rooted[key] = torch.sqrt(tensor + float(eps))
    return rooted


def divide_model_states(
    numerator: Mapping[str, Any],
    denominator: Mapping[str, Any],
    eps: float = 0.0,
) -> StateDict:
    """Return numerator / (denominator + eps) for matching tensor states."""

    return _binary_tensor_state(
        numerator,
        denominator,
        lambda left, right: left / (right + float(eps)),
    )


def model_state_is_all_zeros(state: Mapping[str, Any]) -> bool:
    """Whether every tensor in a state is zero, an empty state included.

    Cheaper than `squared_l2_norm_model_state(state) == 0.0`, which sums every
    coordinate in float and compares a float to zero; this stops at the first
    non-zero it finds and asks an exact question. Non-tensor values are
    ignored, so a state carrying scalars beside its tensors answers about its
    tensors.
    """

    return not any(
        bool(torch.any(value != 0)) for value in state.values() if isinstance(value, torch.Tensor)
    )


def squared_l2_norm_model_state(state: Mapping[str, Any]) -> float:
    """Return the squared L2 norm across tensor state values."""

    total = 0.0
    for key, value in state.items():
        tensor = as_cpu_tensor(key, value).float()
        total += float(torch.sum(tensor * tensor).item())
    return total


def _clone_value(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    return copy.deepcopy(value)


def _binary_tensor_state(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    operation: Any,
) -> StateDict:
    validate_matching_keys(a, b)
    result: StateDict = {}
    for key in a:
        left = as_cpu_tensor(key, a[key])
        right = as_cpu_tensor(key, b[key])
        result[key] = operation(left, right)
    return result


def validate_matching_keys(*states: Mapping[str, Any]) -> None:
    """Refuse model states that do not describe the same parameters.

    Public because it is the package's shared precondition for every
    elementwise state operation, and servers/fedopt.py had its own copy of it
    -- taking two states rather than any number, and otherwise identical. Two
    copies of a precondition is one copy that can be strengthened while the
    other is not, which is what a `_`-prefixed name invites a second module to
    do. Ditto :func:`as_cpu_tensor`.

    Raises:
        ValueError: If no states are given, or if their key sets differ.
    """

    if not states:
        raise ValueError("states must not be empty")
    state_keys = set(states[0].keys())
    for state in states[1:]:
        if set(state.keys()) != state_keys:
            raise ValueError("all states must have the same keys")


def as_cpu_tensor(key: str, value: Any) -> Tensor:
    """Return one state value as a detached CPU tensor.

    The single door every elementwise state operation reads a value through,
    so a dtype or device policy has exactly one place to go. See
    :func:`validate_matching_keys` for why it is public.

    Raises:
        TypeError: If the value is not a Tensor. ``key`` is only ever used to
            say which one.
    """

    if not isinstance(value, Tensor):
        raise TypeError(f"state value for {key} is not a tensor")
    return value.detach().cpu()
