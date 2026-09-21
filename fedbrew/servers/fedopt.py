"""FedOpt server strategies for adaptive server-side optimization."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from typing import Any, cast

import torch

from fedbrew.core.checkpointing import refuse_a_reconfigured_resume
from fedbrew.core.metrics import filter_metrics
from fedbrew.core.protocol import FitResult, RoundInfo
from fedbrew.core.torch_utils import (
    StateDict,
    add_model_states,
    as_cpu_tensor,
    clone_model_state,
    divide_model_states,
    scale_model_state,
    sqrt_model_state,
    subtract_model_states,
    validate_matching_keys,
    zeros_like_model_state,
)
from fedbrew.servers.fedavg import FedAvgServer
from fedbrew.tasks.base import TaskAdapter

SUPPORTED_FEDOPT_OPTIMIZERS = {"fedavgm", "fedadam", "fedyogi", "fedadagrad"}

#: The four hyperparameters a FedOpt config sets, in the order it lists them.
FEDOPT_HYPERPARAMETERS: tuple[str, ...] = (
    "server_learning_rate",
    "beta1",
    "beta2",
    "tau",
)

#: Which of those four each optimizer's update never reads, as
#: optimizer -> (what its update is, the hyperparameters no line of it
#: consumes). An optimizer that reads all four has no row.
#:
#: ``_fedavgm_update`` is one momentum buffer and one scale. It allocates no
#: second moment, so it never reaches ``_initial_v`` and never floors a
#: denominator: neither ``beta2`` nor ``tau`` has anywhere to enter.
#: ``_fedadagrad_update`` accumulates ``v <- v + delta**2`` with no decay --
#: that missing decay is exactly what makes it Adagrad rather than Adam -- so
#: it reads ``tau`` twice and ``beta2`` not at all.
#:
#: Declared here, beside the updates that decide it, and read by
#: ``core/config.py``, so a config is judged against the code rather than
#: against a second copy of this table. The rule is the one
#: ``UNHONOURED_CLIENT_OPTIONS`` already applies on the client side: a key its
#: component never receives is refused, not accepted and dropped. P01-F07.
UNREAD_FEDOPT_HYPERPARAMETERS: dict[str, tuple[str, tuple[str, ...]]] = {
    "fedavgm": (
        "accumulates one momentum buffer, m <- beta1*m + delta, and steps by "
        "server_learning_rate*m, so it has no second moment and no denominator",
        ("beta2", "tau"),
    ),
    "fedadagrad": (
        "accumulates its second moment without decay, v <- v + delta**2, "
        "which is what makes it Adagrad rather than Adam",
        ("beta2",),
    ),
}


#: What each hyperparameter must be, as name -> (predicate, the bound in
#: words). One definition, read by `_validate_hyperparameters` here and by
#: `validate_config` on the run path, and reported by `--validate-only`.
#:
#: The bounds lived in `core/validation.py` alone, so `validate_config` --
#: which every run reaches through `load_config` -- checked only that the four
#: were numeric. Three of the four were caught late anyway, by the constructor
#: below. `tau: 0` was caught nowhere: it is non-negative, so the constructor
#: passed it, and it makes `v_{-1} = tau**2` zero, so a coordinate whose delta
#: is exactly zero computes `0 / (sqrt(0) + 0)`. Measured on the pre-fix tree,
#: one round at `tau = 0` against a zero delta: fedadagrad, fedadam and
#: fedyogi each returned `[nan, nan, nan, nan]`, and a NaN in the server's
#: model state is permanent. An exactly-zero delta is not exotic -- it is
#: what a parameter that receives no gradient returns from every client.
#: P04-F05.
#:
#: `beta1` and `beta2` are half-open at 1.0, which is the bound preflight
#: always stated and the constructor did not: at `beta = 1` the moment never
#: takes up the delta, so `m` stays at zeros and the server never moves. A
#: silent no-op is the failure this bound exists to refuse.
FEDOPT_BOUNDS: dict[str, tuple[Callable[[float], bool], str]] = {
    "server_learning_rate": (lambda value: value > 0.0, "> 0"),
    "beta1": (lambda value: 0.0 <= value < 1.0, "in [0, 1)"),
    "beta2": (lambda value: 0.0 <= value < 1.0, "in [0, 1)"),
    "tau": (lambda value: value > 0.0, "> 0"),
}


def fedopt_bound_violation(name: str, value: float) -> str | None:
    """Why `value` is not a legal `name`, if it is not.

    Args:
        name: One of :data:`FEDOPT_HYPERPARAMETERS`.
        value: The configured value.

    Returns:
        A message naming the bound, or None when the value is inside it. A
        name with no bound returns None rather than raising: the caller is
        asking about a value, not about the name.
    """

    entry = FEDOPT_BOUNDS.get(name)
    if entry is None:
        return None
    predicate, bound = entry
    if not math.isfinite(value) or not predicate(value):
        return f"FedOpt {name} must be {bound}, got {value!r}"
    return None


def unread_fedopt_hyperparameters(optimizer: str) -> tuple[str, tuple[str, ...]]:
    """What ``optimizer`` does, and which hyperparameters it never reads.

    Args:
        optimizer: A supported optimizer name, or any string. An unknown name
            is answered as though it read all four, which leaves every
            hyperparameter required and lets the name itself be the error.

    Returns:
        ``(reason, unread)`` from :data:`UNREAD_FEDOPT_HYPERPARAMETERS`, or
        ``("", ())`` for an optimizer with no row.
    """

    return UNREAD_FEDOPT_HYPERPARAMETERS.get(optimizer.strip().lower(), ("", ()))


class FedOptServer(FedAvgServer):
    """Server-side FedOpt strategy using FedAvg round orchestration.

    Sampling, broadcast and client-state averaging are FedAvg's, unchanged.
    What differs is the last step: instead of adopting the averaged client
    state directly, the server treats ``average - current`` as a pseudo-gradient
    and applies an adaptive optimizer to it, following Reddi et al.
    (arXiv:2003.00295).
    """

    def __init__(
        self,
        server_optimizer: str,
        server_learning_rate: float,
        beta1: float,
        beta2: float | None,
        tau: float | None,
        participation_rate: float | None,
        seed: int,
        task: TaskAdapter | None = None,
        model_config: Mapping[str, Any] | None = None,
        metrics: list[str] | None = None,
        aggregation_weighting: str = "examples",
        participation_probability: float | None = None,
    ) -> None:
        """Configure the server optimizer on top of FedAvg's round handling.

        Args:
            server_optimizer: One of ``fedavgm``, ``fedadam``, ``fedyogi``,
                ``fedadagrad``. Case- and whitespace-insensitive.
            server_learning_rate: Step size applied to the server update, in
                model-parameter units per round. Must be positive. This is
                *not* the client learning rate; the two are independent.
            beta1: First-moment decay in [0, 1). Used by every optimizer
                here, including ``fedavgm``, where it is the momentum
                coefficient. Half-open at 1.0: the moment would never take up
                the delta, so the server would never move.
            beta2: Second-moment decay in [0, 1), or ``None`` for an
                optimizer that has no second moment to decay. Required by
                ``fedadam`` and ``fedyogi``; must be ``None`` for ``fedavgm``,
                which keeps no ``v``, and for ``fedadagrad``, whose ``v``
                accumulates undecayed.
            tau: Adaptivity floor, in the same units as the pseudo-gradient's
                magnitude, or ``None`` for an optimizer with no denominator to
                floor. Appears as ``sqrt(v) + tau`` there, so a larger value
                makes the update more like plain averaging. Must be positive:
                at zero the floor is not a floor, and a coordinate whose delta
                is exactly zero divides ``0`` by ``sqrt(0) + 0``. Required by
                every optimizer but ``fedavgm``, which must pass ``None``.
            participation_rate: As FedAvgServer.
            seed: As FedAvgServer.
            task: As FedAvgServer.
            model_config: As FedAvgServer.
            metrics: As FedAvgServer.
            aggregation_weighting: As FedAvgServer -- applied to the client
                average that forms the pseudo-gradient.
            participation_probability: As FedAvgServer. A round that selects
                no client is not aggregated, so the moments do not decay on it.

        Raises:
            ValueError: If the optimizer name is unsupported, a hyperparameter
                falls outside :data:`FEDOPT_BOUNDS`, one this optimizer reads
                is ``None``, or one it never reads is not -- see
                :data:`UNREAD_FEDOPT_HYPERPARAMETERS`.

        The second-moment accumulator starts at ``tau ** 2`` rather than zero,
        the smallest value Algorithm 2 of the paper allows (it requires
        ``v_{-1} >= tau^2``), so the first rounds' denominator is not pure ``tau``.
        """

        super().__init__(
            task=task,
            model_config=model_config,
            participation_rate=participation_rate,
            seed=seed,
            metrics=metrics,
            aggregation_weighting=aggregation_weighting,
            participation_probability=participation_probability,
        )
        self.server_optimizer = _normalize_server_optimizer(server_optimizer)
        self.server_learning_rate = float(server_learning_rate)
        self.beta1 = float(beta1)
        self.beta2 = None if beta2 is None else float(beta2)
        self.tau = None if tau is None else float(tau)
        self._validate_hyperparameters()
        self._m: StateDict | None = None
        self._v: StateDict | None = None
        self._update_step = 0

    def aggregate_stream(
        self,
        round_info: RoundInfo,
        results: Iterable[FitResult],
    ) -> dict[str, Any]:
        """Aggregate client states and apply the configured FedOpt update.

        Args:
            round_info: The round being aggregated. Its ``metrics`` mapping is
                updated in place with this round's filtered client metrics.
            results: Client fit results, consumed as a stream -- each is folded
                into a running weighted sum and dropped, so peak memory is one
                model state regardless of participation.

        Returns:
            The broadcast payload for the next round, carrying the updated
            global model state and this round's metrics.

        Raises:
            ValueError: If the model state is still uninitialised after
                :meth:`initialize`.
        """

        if self._model_state is None:
            self.initialize()
        if self._model_state is None:
            raise ValueError("model state was not initialized")

        averaged_client_state, raw_metrics = self._accumulate_fit_results(results)
        delta = subtract_model_states(averaged_client_state, self._model_state)
        self._model_state = self._apply_fedopt_update(delta)

        metrics = filter_metrics(raw_metrics, self.metrics)
        round_info.metrics.update(metrics)
        self._round_metrics.append(metrics)
        return self._federated_payload(metrics=metrics)

    def save_state(self) -> dict[str, Any]:
        """Return a checkpointable snapshot of server optimizer state.

        Returns:
            FedAvg's state plus the optimizer name, the hyperparameters this
            optimizer reads, deep copies of the first- and second-moment
            accumulators ``m`` and ``v``, and ``update_step``. The moments are
            cloned, so mutating the returned dict cannot corrupt the live
            server. A hyperparameter this optimizer never reads is absent
            rather than ``None``: a checkpoint records what ran.
        """

        state = super().save_state()
        state.update(
            {
                "server_optimizer": self.server_optimizer,
                **self._configured_hyperparameters(),
                "m": clone_model_state(self._m or {}),
                "v": clone_model_state(self._v or {}),
                "update_step": self._update_step,
            }
        )
        return state

    def _configured_hyperparameters(self) -> dict[str, float]:
        """The hyperparameters this optimizer reads, keyed as state keys."""

        _, unread = unread_fedopt_hyperparameters(self.server_optimizer)
        return {
            name: cast(float, getattr(self, name))
            for name in FEDOPT_HYPERPARAMETERS
            if name not in unread
        }

    def load_state(self, state: Mapping[str, Any]) -> None:
        """Restore model, metrics, and server optimizer state.

        Args:
            state: A mapping as produced by :meth:`save_state`. Missing
                optimizer keys leave the corresponding attribute at its current
                value, so a checkpoint from an older layout still loads; an
                absent or empty ``m`` / ``v`` resets that moment to None and it
                is re-initialised on the next update. A checkpoint written
                before ``beta2``/``tau`` became optional carries a value for
                one this optimizer never reads; it is neither compared nor
                restored, because nothing would have used it either way.

        Raises:
            ValueError: If the restored hyperparameters fail validation.
        """

        refuse_a_reconfigured_resume(
            "fedopt server",
            state,
            {
                "server_optimizer": self.server_optimizer,
                **self._configured_hyperparameters(),
            },
        )
        super().load_state(state)
        optimizer = state.get("server_optimizer")
        if isinstance(optimizer, str):
            self.server_optimizer = _normalize_server_optimizer(optimizer)
        _, unread = unread_fedopt_hyperparameters(self.server_optimizer)
        for name in FEDOPT_HYPERPARAMETERS:
            if name in unread or name not in state:
                continue
            setattr(self, name, float(state[name]))
        self._validate_hyperparameters()

        m = state.get("m")
        self._m = clone_model_state(m) if isinstance(m, dict) and m else None
        v = state.get("v")
        self._v = clone_model_state(v) if isinstance(v, dict) and v else None
        self._update_step = int(state.get("update_step", self._update_step))

    def _apply_fedopt_update(self, delta: Mapping[str, Any]) -> StateDict:
        if self._model_state is None:
            raise ValueError("model state was not initialized")
        if self._m is None:
            self._m = zeros_like_model_state(delta)

        if self.server_optimizer == "fedavgm":
            update = self._fedavgm_update(delta)
        elif self.server_optimizer == "fedadam":
            update = self._fedadam_update(delta)
        elif self.server_optimizer == "fedyogi":
            update = self._fedyogi_update(delta)
        elif self.server_optimizer == "fedadagrad":
            update = self._fedadagrad_update(delta)
        else:  # pragma: no cover - guarded by constructor/load validation.
            raise ValueError(f"unsupported server optimizer: {self.server_optimizer}")

        self._update_step += 1
        return add_model_states(self._model_state, update)

    def _fedavgm_update(self, delta: Mapping[str, Any]) -> StateDict:
        m = cast(Mapping[str, Any], self._m)
        self._m = add_model_states(
            scale_model_state(m, self.beta1),
            delta,
        )
        return scale_model_state(self._m, self.server_learning_rate)

    def _initial_v(self, delta: Mapping[str, Any]) -> StateDict:
        """Return the second-moment accumulator's initial value.

        Algorithm 2 of Reddi et al. (arXiv:2003.00295) requires ``v_{-1} >= tau^2``
        rather than zeros, which keeps the ``sqrt(v) + tau`` denominator away from
        pure ``tau`` on the first rounds.
        """

        tau = cast(float, self.tau)
        zeros = zeros_like_model_state(delta)
        return {key: value + tau**2 for key, value in zeros.items()}

    def _fedadagrad_update(self, delta: Mapping[str, Any]) -> StateDict:
        if self._v is None:
            self._v = self._initial_v(delta)
        tau = cast(float, self.tau)
        m = cast(Mapping[str, Any], self._m)
        v = cast(Mapping[str, Any], self._v)
        self._m = add_model_states(
            scale_model_state(m, self.beta1),
            scale_model_state(delta, 1.0 - self.beta1),
        )
        # FedAdagrad accumulates without decay: v <- v + delta^2.
        self._v = add_model_states(v, _square_model_state(delta))
        adaptive_step = divide_model_states(
            self._m,
            sqrt_model_state(self._v),
            eps=tau,
        )
        return scale_model_state(adaptive_step, self.server_learning_rate)

    def _fedadam_update(self, delta: Mapping[str, Any]) -> StateDict:
        if self._v is None:
            self._v = self._initial_v(delta)
        beta2 = cast(float, self.beta2)
        tau = cast(float, self.tau)
        m = cast(Mapping[str, Any], self._m)
        v = cast(Mapping[str, Any], self._v)
        delta_squared = _square_model_state(delta)
        self._m = add_model_states(
            scale_model_state(m, self.beta1),
            scale_model_state(delta, 1.0 - self.beta1),
        )
        self._v = add_model_states(
            scale_model_state(v, beta2),
            scale_model_state(delta_squared, 1.0 - beta2),
        )
        adaptive_step = divide_model_states(
            self._m,
            sqrt_model_state(self._v),
            eps=tau,
        )
        return scale_model_state(adaptive_step, self.server_learning_rate)

    def _fedyogi_update(self, delta: Mapping[str, Any]) -> StateDict:
        if self._v is None:
            self._v = self._initial_v(delta)
        beta2 = cast(float, self.beta2)
        tau = cast(float, self.tau)
        m = cast(Mapping[str, Any], self._m)
        v = cast(Mapping[str, Any], self._v)
        delta_squared = _square_model_state(delta)
        self._m = add_model_states(
            scale_model_state(m, self.beta1),
            scale_model_state(delta, 1.0 - self.beta1),
        )
        signed_delta_squared = _multiply_model_states(
            delta_squared,
            _sign_model_state(subtract_model_states(v, delta_squared)),
        )
        self._v = subtract_model_states(
            v,
            scale_model_state(signed_delta_squared, 1.0 - beta2),
        )
        adaptive_step = divide_model_states(
            self._m,
            sqrt_model_state(self._v),
            eps=tau,
        )
        return scale_model_state(adaptive_step, self.server_learning_rate)

    def _validate_hyperparameters(self) -> None:
        reason, unread = unread_fedopt_hyperparameters(self.server_optimizer)
        for name in FEDOPT_HYPERPARAMETERS:
            value = getattr(self, name)
            if name in unread:
                if value is not None:
                    raise ValueError(
                        f"{self.server_optimizer} {reason}, so it never reads {name}; pass None"
                    )
            elif value is None:
                raise ValueError(f"{self.server_optimizer} requires {name}")
        for name in FEDOPT_HYPERPARAMETERS:
            value = getattr(self, name)
            if value is None:
                continue
            violation = fedopt_bound_violation(name, value)
            if violation is not None:
                raise ValueError(violation)


def _normalize_server_optimizer(value: str) -> str:
    normalized = value.lower().strip()
    if normalized not in SUPPORTED_FEDOPT_OPTIMIZERS:
        raise ValueError(
            "server_optimizer must be one of: " + ", ".join(sorted(SUPPORTED_FEDOPT_OPTIMIZERS))
        )
    return normalized


def _square_model_state(state: Mapping[str, Any]) -> StateDict:
    squared: StateDict = {}
    for key, value in state.items():
        tensor = as_cpu_tensor(key, value)
        squared[key] = tensor * tensor
    return squared


def _sign_model_state(state: Mapping[str, Any]) -> StateDict:
    signed: StateDict = {}
    for key, value in state.items():
        signed[key] = torch.sign(as_cpu_tensor(key, value))
    return signed


def _multiply_model_states(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
) -> StateDict:
    validate_matching_keys(a, b)
    multiplied: StateDict = {}
    for key in a:
        multiplied[key] = as_cpu_tensor(key, a[key]) * as_cpu_tensor(key, b[key])
    return multiplied
