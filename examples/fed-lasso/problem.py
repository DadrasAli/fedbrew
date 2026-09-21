"""Federated lasso: least squares plus an L1 term, over a planted sparse signal.

The problem
-----------
Client `i` holds an `m x d` design `H` and its own targets `y_i`, and optimises
the composite objective::

    f_i(x) = (1/2m) ||H x - y_i||^2 + lam ||x||_1

`H` is the `d x d` Sylvester-Hadamard matrix with `m = d` rows, so
`H^T H = m I` **exactly** -- the design is orthonormal after the `1/m`, every
client shares it, and the smooth part reduces, exactly, to::

    (1/2m) ||H x - y_i||^2 = (1/2) ||x - theta_i||^2      theta_i = H^T y_i / m

`theta_i` is client `i`'s ordinary-least-squares solution. The targets are built
as `y_i = H theta_i` from `theta_i = x_true + w + u_i`, where `x_true` is the
planted sparse signal, `w` is a pooled perturbation (the ``noise`` dial, 0 by
default) and the `u_i` are per-client offsets laid out in exact +/- pairs, so
they sum to zero in floating point and the pooled OLS solution is::

    theta_bar = x_true + w        (= x_true exactly, at noise = 0)

The federated objective is therefore, exactly::

    F(x) = (1/2) ||x - theta_bar||^2 + lam ||x||_1 + zeta^2 / 2

with `zeta^2 = mean_i ||u_i||^2` the heterogeneity residual that no `x` removes.
Because the design is orthonormal the minimiser is coordinate-wise soft
thresholding, in closed form::

    x* = S_lam(theta_bar)         S_lam(v)_j = sign(v_j) max(0, |v_j| - lam)

so `x*`, its support, and `F* = F(x*)` are all known before the run starts.

The penalty is a dial
---------------------
``problem.penalty`` selects which term `lam` multiplies, and nothing else about
the problem moves -- the design, the planted signal, the offsets and the stored
targets are identical under both::

    l1   f_i(x) = (1/2m) ||H x - y_i||^2 + lam ||x||_1        (the default)
    l2   f_i(x) = (1/2m) (||H x - y_i||^2 + lam ||x||^2)

The L2 form is the ridge objective under the same `1/m` as the residual, so the
minimiser is the textbook estimator and, under `H^T H = m I`, a uniform
shrinkage::

    x* = (H^T H + lam I)^-1 H^T y_bar = (m + lam)^-1 H^T y_bar
       = m theta_bar / (m + lam)

`F*` follows from it the same way, and both are written into the manifest by
the same code path. What the dial changes is the *kind* of problem: the L1 term
is non-smooth and the L2 term is not, so the `O(eta lam)` subgradient floor that
every arm sits on disappears and every arm converges. What it does not change is
the sparsity story, in the opposite direction from the one a reader expects:
ridge sets no coordinate to zero at any `lam`, so `x*` is `theta_bar` scaled and
support recovery is not something the L2 problem asks for. At ``noise: 0.0``,
`theta_bar` is `x_true` and its thirteen zeros are *inherited* by `x*` rather
than found -- which makes the reference row's `support_f1` read 1.00 under both
penalties and mean something under only one. README, "The smooth control".

`lam` is the same dial in both, kept under the name ``penalty_strength`` because that
is what every shipped config already spells, but it is not on the same *scale*:
the L2 term carries the `1/m` and the L1 term does not, so `lam = 0.05` shrinks
by `lam sqrt(s) = 0.0866` under L1 and by a factor of `m / (m + lam)` under L2.

What the ground truth is, and what it is not
--------------------------------------------
Two different vectors, and conflating them is the first thing this example is
for:

``x_true``
    The planted signal. `sparsity` non-zeros with geometrically decreasing
    magnitudes and alternating signs, at evenly spaced coordinates.

``x*``
    The minimiser of `F`, which is `x_true` **shrunk by lam** on its support and
    exactly zero off it (whenever `lam` is below the smallest planted magnitude,
    which :meth:`ProblemSpec.support_recoverable` checks). At noise 0 the
    distance between them is `lam * sqrt(sparsity)`, in closed form.

So `||x* - x_true||` is not zero, and an arm can end below it only by stopping
away from `x*`; an arm that drove `||x - x_true||` to zero would be *worse* on
the objective it was given. :meth:`ProblemSpec.truth_distance_floor` is that
number, and the README reports it beside the column so the column cannot be
read as an optimiser score.

Support recovery is the second thing that needs saying twice. `x*` has exactly
`sparsity` non-zeros. No shipped algorithm produces an exact zero at all --
there is no proximal step anywhere in the repository -- so "the support of the
iterate" is a *thresholding decision*, and ``model.support_tolerance`` is the
free parameter that makes it. The ``exact_zeros`` metric is the unthresholded
version: the count of coordinates that are bit-for-bit 0.0. It is `d` at
initialisation and 0 from round 1 onward, on every arm.

What this file registers
------------------------
Three names, at ``register()`` time, and nothing at import:

    generators  "fed_lasso"      writes the shards and the manifest
    tasks       "fed_lasso"      FedLassoTask
    models      "lasso_vector"   LassoModel, a d-vector plus its penalty

There is no dataset backend. The design and the targets are *generated* --
from the spec in a generator config -- and a run reads them through the
shipped ``manifest_dataset`` like every other run in the repository. The
reference optimum travels with the data: `x_true`, `x*`, `F*` and every floor
they imply are properties of the generated shards, so the generator writes
them into the manifest under ``reference``, the task reads them from
``dataset_metadata``, and ``run.json`` records them.

`lam` is on the model as well as in the data, because it *is* part of the
objective the client descends and ``model.extra`` is where a run config
carries it; :class:`FedLassoTask` is the one object handed both the model
block and the manifest, so it is where the two are cross-checked. So is
``penalty``, for the same reason and with the same consequence if it drifted:
a client descending the ridge objective against lasso data would fail on
nothing and be scored against the wrong optimum.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn, optim

from fedbrew.data.manifest_validation import IDENTICAL_TO_TRAIN
from fedbrew.data.writers.manifest import save_clients_jsonl, save_manifest
from fedbrew.data.writers.torch_shards import save_client_shard, save_split_client_shard
from fedbrew.models.config_keys import reject_unknown_model_keys
from fedbrew.tasks.base import TaskAdapter

#: Every tensor here is float64. The objective's floor is set by `lam` and sits
#: around 1e-3, far above float32's precision, so this is not about resolving
#: the floor -- it is so that `exact_zeros` means what it says, and so that the
#: closed-form optimality check in `_self_check` can use a 1e-14 tolerance
#: rather than one loose enough to pass by accident.
DTYPE = torch.float64


# ---------------------------------------------------------------------------
# The objective
# ---------------------------------------------------------------------------


def soft_threshold(values: Tensor, level: float) -> Tensor:
    """`S_t(v)_j = sign(v_j) max(0, |v_j| - t)`, the L1 proximal operator.

    The one operator this problem needs and the repository does not have. It is
    here to *define* the reference optimum, not to run: nothing under
    ``fedbrew/clients/`` or ``fedbrew/servers/`` calls anything like it, and
    this example does not add a client rule that would.
    """

    return torch.sign(values) * torch.clamp(values.abs() - level, min=0.0)


def ridge_shrink(values: Tensor, level: float, rows: int) -> Tensor:
    """`R_t(v) = m v / (m + t)`, the L2 analogue of :func:`soft_threshold`.

    Not a proximal operator and not called one: it is the *minimiser map* of
    `(1/2)||x - v||^2 + t ||x||^2 / 2m`, in closed form because the penalty is
    quadratic. Uniform shrinkage toward the origin by a single factor, which is
    the whole difference from soft thresholding: no coordinate is ever set to
    zero, however small it is.
    """

    return values * (rows / (rows + level))


def penalty_of(x: Tensor, level: float, penalty: str, rows: int) -> Tensor:
    """The penalty term, as a differentiable scalar tensor.

    ``l1``
        `lam ||x||_1`, added to the `1/m`-normalised residual at full strength.

    ``l2``
        `lam ||x||^2 / 2m`, which is `(lam/2) ||x||^2` under the *same* `1/m`
        as the residual -- so `f_i` is `(1/2m)(||H x - y_i||^2 + lam ||x||^2)`
        and the minimiser is the textbook ridge estimator
        `(H^T H + lam I)^-1 H^T y = (m + lam)^-1 H^T y`. The `1/m` is what
        makes that formula the one that holds; it is also why `lam` is *not*
        on the same scale in the two arms, which the README states beside the
        numbers rather than leaving to be inferred.
    """

    if penalty == "l1":
        return level * x.abs().sum()
    return level * (x @ x) / (2.0 * rows)


def penalty_gradient(x: Tensor, level: float, penalty: str, rows: int) -> Tensor:
    """`lam sign(x)`, or `lam x / m`: the penalty's (sub)gradient."""

    if penalty == "l1":
        return level * torch.sign(x)
    return (level / rows) * x


def objective(
    x: Tensor,
    features: Tensor,
    targets: Tensor,
    penalty_strength: float,
    penalty: str = "l1",
    rows: int | None = None,
) -> float:
    """`f_i(x) = (1/2m) ||H x - y_i||^2 + P(x)`, as a plain float.

    ``rows`` is `m`, which the L2 penalty is normalised by; it defaults to the
    number of targets, which is `m` for one client's rows.
    """

    residual = features @ x - targets
    smooth = 0.5 * float(residual @ residual) / len(targets)
    scale = len(targets) if rows is None else rows
    return smooth + float(penalty_of(x, penalty_strength, penalty, scale))


def gradient(
    x: Tensor,
    features: Tensor,
    targets: Tensor,
    penalty_strength: float,
    penalty: str = "l1",
    rows: int | None = None,
) -> Tensor:
    """`(1/m) H^T (H x - y_i) + grad P(x)`, the analytic (sub)gradient.

    A *sub*gradient under ``l1``, not a gradient: `||x||_1` is not
    differentiable at 0, and `sign(0) = 0` picks the minimum-norm element of
    the subdifferential there. That choice is torch's too, so autograd on the
    closed form returns exactly this and no custom ``autograd.Function`` is
    needed -- unlike ``examples/pl-1d``, whose objective has a transcendental
    term. ``_self_check`` verifies the agreement rather than assuming it, on
    both penalties. Under ``l2`` the term is differentiable everywhere and the
    word subgradient stops applying, which is the point of the control.
    """

    residual = features @ x - targets
    scale = len(targets) if rows is None else rows
    return features.T @ residual / len(targets) + penalty_gradient(
        x, penalty_strength, penalty, scale
    )


def support_of(values: Tensor, tolerance: float) -> set[int]:
    """The coordinates a run would call non-zero at `tolerance`."""

    return {index for index, value in enumerate(values.tolist()) if abs(value) > tolerance}


def support_f1(predicted: set[int], actual: set[int]) -> float:
    """F1 of a predicted support against the planted one; 0.0 if nothing hit."""

    hits = len(predicted & actual)
    if not hits:
        return 0.0
    precision = hits / len(predicted)
    recall = hits / len(actual)
    return 2.0 * precision * recall / (precision + recall)


def _hadamard_row(index: int, dim: int) -> list[float]:
    """Row `index` of the Sylvester-Hadamard matrix, as +/-1 entries.

    `H[k][j] = (-1)^popcount(k & j)`, symmetric, and for a power-of-two `dim`
    orthogonal: `H^T H = dim I` in exact integer arithmetic, which float64
    represents without error at these sizes. That is what makes the design
    orthonormal and the lasso solution a soft threshold.
    """

    return [-1.0 if bin(index & column).count("1") % 2 else 1.0 for column in range(dim)]


def hadamard_matrix(dim: int) -> Tensor:
    """The `dim x dim` Sylvester-Hadamard matrix."""

    return torch.tensor([_hadamard_row(row, dim) for row in range(dim)], dtype=DTYPE)


# ---------------------------------------------------------------------------
# The problem
# ---------------------------------------------------------------------------

_MODEL_KEYS = ("penalty_strength", "penalty", "support_tolerance", "x_init")

#: The two forms the penalty term may take. `lam` is the same dial in both --
#: ``problem.penalty_strength``, kept under that name because renaming it would move
#: every published number's config out from under it -- but it is not on the
#: same *scale* in the two, because the L2 term carries the residual's `1/m`
#: and the L1 term does not. README, "The smooth control".
PENALTIES = ("l1", "l2")


@dataclass(frozen=True, slots=True)
class ProblemSpec:
    """The whole problem, as six numbers, plus every reference it implies."""

    num_clients: int = 8
    #: `d`, and also `m`: the design is the full `d x d` Hadamard matrix. A
    #: power of two, so the design is orthogonal.
    dim: int = 16
    #: Number of planted non-zeros in `x_true`.
    sparsity: int = 3
    #: `lam`. Below the smallest planted magnitude, the exact minimiser keeps
    #: the whole support; above it, the problem itself drops a coefficient and
    #: support recovery stops being possible for anyone. Under ``penalty:
    #: l2`` neither sentence applies: ridge keeps every coordinate at every
    #: `lam`, and the name is kept only because it is the dial the shipped
    #: configs already spell.
    penalty_strength: float = 0.05
    #: Which penalty `lam` multiplies: ``l1`` for `lam ||x||_1`, the lasso this
    #: example is named for, or ``l2`` for `lam ||x||^2 / 2m`, the smooth
    #: control. A dial and not a second example, because everything else --
    #: the design, the planted signal, the offsets, the data on disk -- is
    #: unchanged and only the term added to it moves.
    penalty: str = "l1"
    #: Sup-norm of the pooled perturbation `w` of the OLS solution. 0.0 makes
    #: the pooled problem noiseless, so `x*` differs from `x_true` only by the
    #: L1 shrinkage and the distance between them is exactly `lam sqrt(s)`.
    noise: float = 0.0
    #: `zeta`: the per-client spread of `theta_i` about `theta_bar`. The mean is
    #: exact regardless, so this moves the clients apart without moving `x*`.
    heterogeneity: float = 0.4

    def __post_init__(self) -> None:
        """Refuse a spec that cannot express what it claims to."""

        if self.num_clients < 2:
            raise ValueError("partition.num_clients must be at least 2")
        if self.dim < 2 or self.dim & (self.dim - 1):
            raise ValueError("problem.dim must be a power of two of at least 2")
        if not 1 <= self.sparsity <= self.dim:
            raise ValueError("problem.sparsity must be between 1 and problem.dim")
        if self.penalty_strength < 0.0:
            raise ValueError("problem.penalty_strength must be non-negative")
        if self.penalty not in PENALTIES:
            raise ValueError(f"problem.penalty must be one of {PENALTIES}, not {self.penalty!r}")
        if self.noise < 0.0:
            raise ValueError("problem.noise must be non-negative")
        if self.heterogeneity < 0.0:
            raise ValueError("problem.heterogeneity must be non-negative")
        if self.num_clients // 2 > self.dim - 1:
            raise ValueError(
                f"{self.num_clients} clients need {self.num_clients // 2} offset "
                f"directions, and d={self.dim} supplies {self.dim - 1}."
            )

    # -- the ground truth ---------------------------------------------------

    def truth(self) -> Tensor:
        """`x_true`: the planted signal, `sparsity` non-zeros in `d` coordinates.

        Magnitudes halve along the support and the signs alternate, so the
        smallest planted coefficient is `2^-(s-1)` and the support spans a
        factor of `2^(s-1)` in size. A method that only finds the largest
        coefficient and a method that finds all of them are then distinguishable
        on ``support_f1`` rather than on the objective alone.
        """

        values = torch.zeros(self.dim, dtype=DTYPE)
        for order in range(self.sparsity):
            index = (order * self.dim) // self.sparsity
            values[index] = (-1.0) ** order * 0.5**order
        return values

    def truth_support(self) -> set[int]:
        """The planted support, as coordinate indices."""

        return {(order * self.dim) // self.sparsity for order in range(self.sparsity)}

    def smallest_planted_magnitude(self) -> float:
        """`min_j |x_true_j|` over the support: `2^-(s-1)`."""

        return 0.5 ** (self.sparsity - 1)

    # -- the data -----------------------------------------------------------

    def design(self) -> Tensor:
        """`H`, shared by every client. `H^T H = m I` exactly."""

        return hadamard_matrix(self.dim)

    def pooled_perturbation(self) -> Tensor:
        """`w`: the pooled displacement of the OLS solution, sup-norm `noise`.

        Spread over every coordinate with alternating signs rather than
        concentrated, so raising ``noise`` past ``penalty_strength`` turns *all* the
        off-support coordinates into false positives at once instead of one at
        a time. That makes the transition legible at the cost of making it
        abrupt; the default is 0.
        """

        signs = torch.tensor([(-1.0) ** index for index in range(self.dim)], dtype=DTYPE)
        return self.noise * signs

    def client_offsets(self) -> Tensor:
        """The `u_i`, shape `(n, d)`, in exact +/- pairs summing to zero.

        Hadamard rows again, scaled to norm ``heterogeneity``: the pairing is
        what makes `sum_i u_i` exactly zero in floating point, and that is what
        keeps `theta_bar` -- and so `x*`, its support and `F*` -- exact for the
        federated objective rather than approximately right.
        """

        scale = self.heterogeneity / (self.dim**0.5)
        rows: list[list[float]] = []
        for pair in range(self.num_clients // 2):
            direction = [scale * sign for sign in _hadamard_row(pair + 1, self.dim)]
            rows.append(direction)
            rows.append([-value for value in direction])
        if self.num_clients % 2:
            rows.append([0.0] * self.dim)
        return torch.tensor(rows, dtype=DTYPE)

    def client_ols(self) -> Tensor:
        """The `theta_i = x_true + w + u_i`, shape `(n, d)`."""

        return self.truth() + self.pooled_perturbation() + self.client_offsets()

    def pooled_ols(self) -> Tensor:
        """`theta_bar = x_true + w`, exactly, because the `u_i` sum to zero."""

        return self.truth() + self.pooled_perturbation()

    def client_targets(self) -> Tensor:
        """The `y_i = H theta_i`, shape `(n, m)`. This is the stored data."""

        return self.client_ols() @ self.design().T

    # -- the reference optimum ----------------------------------------------

    def optimum(self) -> Tensor:
        """The exact minimiser of `F`, in closed form under either penalty.

        ``l1``
            `x* = S_lam(theta_bar)`: coordinate-wise soft thresholding, exact
            because the design is orthonormal.

        ``l2``
            `x* = (m + lam)^-1 H^T y_bar = m theta_bar / (m + lam)`: the ridge
            estimator, exact for the same reason. `H^T y_bar = m theta_bar`
            under `H^T H = m I`, so the two spellings are the same vector.
        """

        if self.penalty == "l1":
            return soft_threshold(self.pooled_ols(), self.penalty_strength)
        return ridge_shrink(self.pooled_ols(), self.penalty_strength, self.dim)

    def client_optima(self) -> Tensor:
        """`S_lam(theta_i)`: each client's own lasso solution, shape `(n, d)`.

        Worth looking at: at the default dials every off-support coordinate of
        `theta_i` sits at `heterogeneity / sqrt(d) = 0.1`, above `lam = 0.05`,
        so *every client's own solution is dense* while the federated one has
        three non-zeros. No client can see the true support; only the average
        can. :meth:`client_support_sizes` is that sentence as a number.
        """

        if self.penalty == "l1":
            return soft_threshold(torch.as_tensor(self.client_ols()), self.penalty_strength)
        return ridge_shrink(torch.as_tensor(self.client_ols()), self.penalty_strength, self.dim)

    def client_support_sizes(self) -> list[int]:
        """How many non-zeros each client's own lasso solution has."""

        return [int((row != 0.0).sum()) for row in self.client_optima()]

    def objective_at(self, x: Tensor) -> float:
        """`F(x)`, computed from the stored data rather than the reduced form.

        The reduced form `(1/2)||x - theta_bar||^2 + lam ||x||_1 + zeta^2/2` is
        exact in real arithmetic and differs from this by rounding in the
        materialised `y_i`. This is the objective the run actually descends, so
        it is the one `F*` and every gap are measured in; ``_self_check``
        pins the two together to 1e-13.
        """

        features = self.design()
        targets = self.client_targets()
        return sum(
            objective(x, features, row, self.penalty_strength, self.penalty, self.dim)
            for row in targets
        ) / len(targets)

    def optimal_objective(self) -> float:
        """`F* = F(x*)`."""

        return self.objective_at(self.optimum())

    def heterogeneity_residual(self) -> float:
        """`zeta^2 / 2`: the part of `F*` no `x` can remove."""

        offsets = self.client_offsets()
        return 0.5 * float((offsets * offsets).sum(dim=1).mean())

    def truth_distance_floor(self) -> float:
        """`||x* - x_true||`: how far the *answer* is from the planted signal.

        `lam sqrt(s)` at noise 0. An arm below it has not beaten anything: it
        has stopped away from the minimiser of the objective it was given.
        """

        return float(torch.linalg.vector_norm(self.optimum() - self.truth()))

    def support_recoverable(self) -> bool:
        """Whether `x*` has exactly the planted support.

        False when `lam` reaches the smallest planted magnitude, or when
        ``noise`` reaches `lam` -- in either case the *problem* has dropped or
        added a coordinate and no algorithm can be blamed for it.

        Under ``penalty: l2`` the flag is true at the default dials and means
        nothing. Ridge multiplies every coordinate of `theta_bar` by the same
        factor `m / (m + lam)`, so `x*` has a zero exactly where `theta_bar`
        has one and the penalty selects nothing at all; at ``noise: 0.0``,
        `theta_bar` is `x_true`, so the planted support is inherited rather
        than recovered. Turn ``noise`` up by any amount and the L1 `x*` still
        has `sparsity` non-zeros while the L2 one has `dim`. The README says
        this beside the table, because a `support_f1` of 1.00 on the reference
        row reads the same under both penalties and is an achievement under
        only one.
        """

        return support_of(self.optimum(), 0.0) == self.truth_support()


# ---------------------------------------------------------------------------
# The model: one d-vector, and the penalty it is scored with
# ---------------------------------------------------------------------------


class LassoModel(nn.Module):  # type: ignore[misc]
    """The iterate `x`, as a `d`-element `nn.Parameter`.

    ``penalty_strength`` and ``support_tolerance`` are plain float attributes, not
    buffers: a buffer is in ``state_dict``, and ``torch_utils.get_model_state``
    is ``state_dict``, so either would be uploaded, averaged and counted into
    ``communicated_parameters`` every round. They are on the model rather than
    on the task because ``model.extra`` is the channel a run config carries
    them in.
    """

    def __init__(
        self,
        dim: int,
        penalty_strength: float = 0.05,
        penalty: str = "l1",
        support_tolerance: float = 1.0e-3,
        x_init: float = 0.0,
    ) -> None:
        """Place the iterate at `x_init` in every coordinate.

        Args:
            dim: Problem dimension `d`, from ``model.input_dim``. Also `m`,
                the row count the L2 penalty is normalised by.
            penalty_strength: `lam`. Part of the objective, so it must match the
                data's; :class:`FedLassoTask` checks that against the
                manifest's ``reference``.
            penalty: Which term `lam` multiplies, ``l1`` or ``l2``. Part of
                the objective too, and cross-checked the same way: a model
                block descending the ridge objective against lasso data would
                be scored against the wrong optimum and fail on nothing.
            support_tolerance: The threshold the support metrics use. *Not*
                part of the objective -- it changes what a run reports and
                nothing about what it optimises.
            x_init: Starting value. 0.0 by default, which is the canonical
                lasso start and also the sparsest possible one: the iterate
                begins with `d` exact zeros and has none after a single
                subgradient step, which is the shortest statement of what is
                missing here.
        """

        super().__init__()
        if penalty not in PENALTIES:
            raise ValueError(f"model.penalty must be one of {PENALTIES}, not {penalty!r}")
        self.x = nn.Parameter(torch.full((dim,), float(x_init), dtype=DTYPE))
        self.penalty_strength = float(penalty_strength)
        self.penalty_form = str(penalty)
        self.support_tolerance = float(support_tolerance)

    def forward(self, features: Tensor) -> Tensor:
        """Return the predictions `H_B x` for a batch of rows."""

        return features @ self.x

    def penalty(self) -> Tensor:
        """`lam ||x||_1`, or `lam ||x||^2 / 2m`, as a scalar tensor.

        A function of the *parameters*, not of the batch, which is why it
        cannot travel through a `criterion(outputs, targets)`. README, "A
        composite objective does not fit `criterion(outputs, targets)`". That
        is true of the L2 form as well: a smooth parameter penalty is no more
        expressible through the two-argument signature than a non-smooth one,
        which is why the control does not close that section.

        `m = d` here, and the vector's own length is where it is read from, so
        the model needs no second dial to state it.
        """

        return penalty_of(self.x, self.penalty_strength, self.penalty_form, self.x.numel())

    @property
    def iterate(self) -> Tensor:
        """The vector the whole example is about, detached."""

        return self.x.detach().clone()


def build_lasso_vector(config: Mapping[str, Any] | None = None) -> LassoModel:
    """Registry builder for `model.name: lasso_vector`.

    Builds from the model block alone. Whether `d` and `lam` agree with the
    data is checked by :class:`FedLassoTask`, the one object handed both the
    model block and the manifest.
    """

    values = dict(config or {})
    reject_unknown_model_keys(values, _MODEL_KEYS, "lasso_vector")
    dim = values.get("input_dim")
    if dim is None:
        raise ValueError("model.input_dim is required for lasso_vector: it is the problem's d")
    return LassoModel(
        dim=int(dim),
        penalty_strength=float(values.get("penalty_strength", 0.05)),
        penalty=str(values.get("penalty", "l1")),
        support_tolerance=float(values.get("support_tolerance", 1.0e-3)),
        x_init=float(values.get("x_init", 0.0)),
    )


# ---------------------------------------------------------------------------
# The generator: one regression per client, written as shards a run reads
# ---------------------------------------------------------------------------

#: The name of this problem's generator and task, as configs write it.
DATASET_NAME = "fed_lasso"


@dataclass(frozen=True, slots=True)
class GenerationSummary:
    """What ``fedbrew generate`` prints when this generator finishes."""

    manifest_path: Path
    num_clients: int
    num_examples: int
    num_test_examples: int


def _spec_from_config(config: Mapping[str, Any]) -> ProblemSpec:
    """Read the spec out of a generator config's ``problem`` section.

    ``partition.num_clients`` rather than a ``problem.clients`` of our own:
    the client count is the one dial every generator states in the same
    place, and stating it twice is how two spellings of one number drift
    apart.
    """

    problem = dict(config.get("problem", {}))
    partition = dict(config.get("partition", {}))
    if "num_clients" not in partition:
        raise ValueError("fed_lasso needs partition.num_clients: it is the number of regressions")
    return ProblemSpec(
        num_clients=int(partition["num_clients"]),
        dim=int(problem.get("dim", 16)),
        sparsity=int(problem.get("sparsity", 3)),
        penalty_strength=float(problem.get("penalty_strength", 0.05)),
        penalty=str(problem.get("penalty", "l1")),
        noise=float(problem.get("noise", 0.0)),
        heterogeneity=float(problem.get("heterogeneity", 0.4)),
    )


def _spec_from_reference(reference: Mapping[str, Any]) -> ProblemSpec:
    """Rebuild the spec from the dials the manifest's ``reference`` records.

    The generator and the task are the same module, and the spec is a
    deterministic function of its dials, so the task rebuilds exactly what
    the generator wrote -- the design, the targets, `x*` -- rather than
    carrying `n x m` floats through JSON to get the same tensors back.
    """

    problem = reference["problem"]
    if "penalty_strength" not in problem and "l1_penalty" in problem:
        # The dial was called `l1_penalty` before it was renamed to say what
        # it is. Named here rather than left as a KeyError three lines down,
        # because the fix is a command and not a code change.
        raise ValueError(
            "this manifest records the penalty strength under the old name "
            "`l1_penalty`; it is `penalty_strength` now. Regenerate the data: "
            "fedbrew generate --config data/configs/examples/<setting>.yaml"
        )
    return ProblemSpec(
        num_clients=int(problem["clients"]),
        dim=int(problem["dim"]),
        sparsity=int(problem["sparsity"]),
        penalty_strength=float(problem["penalty_strength"]),
        # Defaulted rather than required: every manifest written before the
        # penalty dial existed is an L1 one, and reading it back has to keep
        # scoring those runs against the optimum they were scored against.
        penalty=str(problem.get("penalty", "l1")),
        noise=float(problem["noise"]),
        heterogeneity=float(problem["heterogeneity"]),
    )


def reference_of(spec: ProblemSpec) -> dict[str, Any]:
    """Everything a run on this data is scored against, in closed form.

    Written into the manifest, copied into ``run.json`` by
    ``run_metadata.build_dataset_provenance``, and read back by
    :class:`FedLassoTask`. It carries the dials as well as the optimum
    because the floors are only meaningful beside the dials that set them,
    and because the task rebuilds the spec from them.
    """

    optimum = spec.optimum()
    return {
        "problem": {
            "clients": spec.num_clients,
            "dim": spec.dim,
            "sparsity": spec.sparsity,
            "penalty_strength": spec.penalty_strength,
            "penalty": spec.penalty,
            "noise": spec.noise,
            "heterogeneity": spec.heterogeneity,
        },
        "rows_per_client": spec.dim,
        "x_true": spec.truth().tolist(),
        "x_star": optimum.tolist(),
        "f_star": spec.optimal_objective(),
        "truth_support": sorted(spec.truth_support()),
        "optimum_support": sorted(support_of(optimum, 0.0)),
        "support_recoverable": spec.support_recoverable(),
        "client_support_sizes": spec.client_support_sizes(),
        "heterogeneity_residual": spec.heterogeneity_residual(),
        "truth_distance_floor": spec.truth_distance_floor(),
        "smallest_planted_magnitude": spec.smallest_planted_magnitude(),
    }


def generate_fed_lasso_from_config(
    config: Mapping[str, Any],
    output_dir: Path,
    seed: int,
    client_splits: Mapping[str, float],
) -> GenerationSummary:
    """Write one shard per client, plus the manifest a run reads.

    The four arguments are the generator contract (chapter 12 §4). Two of
    them do nothing here, and say so rather than being quietly dropped:

    ``seed``
        The targets are a deterministic function of the spec -- a planted
        signal and Hadamard rows in exact +/- pairs -- so there is no draw
        for a seed to fix. Recorded in the manifest anyway, because every
        other dataset's provenance says what seed it was made at.

    ``client_splits``
        The ratios describe a cut, and there is nothing to cut: `f_i` is
        defined over all `m` of a client's rows, and holding rows out would
        change the objective rather than estimate it. All three splits hold
        the same `m` rows, which the manifest declares as
        ``client_test_source: identical_to_train`` so that preflight tells a
        reader their ``test_*`` and ``central_test_*`` numbers are training
        numbers.
    """

    del client_splits
    spec = _spec_from_config(config)
    features = spec.design()
    targets = spec.client_targets()
    output_dir = Path(output_dir)
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    clients: list[dict[str, Any]] = []
    for index, client_targets in enumerate(targets):
        client_id = f"client_{index}"
        # Every client carries the same design. Written into each shard rather
        # than once, because a shard is what a run reads and a shard is
        # self-contained everywhere else; at d = 16 it is 2 KB.
        x = features.clone()
        y = client_targets.clone()
        save_split_client_shard(shards_dir / f"{client_id}.pt", x, y, x, y, x, y)
        clients.append(
            {
                "client_id": client_id,
                "shard": f"shards/{client_id}.pt",
                "num_examples": 3 * spec.dim,
                "num_train_examples": spec.dim,
                "num_eval_examples": spec.dim,
                "num_test_examples": spec.dim,
                "own_support_size": spec.client_support_sizes()[index],
            }
        )

    # The server's central pass evaluates F in one go, so the pooled shard is
    # every client's rows stacked -- exactly the federated objective, because
    # every client has the same m rows and so the same weight.
    save_client_shard(
        shards_dir / "global_test.pt",
        features.repeat(len(targets), 1),
        targets.reshape(-1).clone(),
    )
    _write_partition_stats(output_dir, spec, clients)

    manifest = {
        "dataset_name": DATASET_NAME,
        "format": "torch_shards",
        "client_shard_format": "split_v2",
        "client_test_source": IDENTICAL_TO_TRAIN,
        "num_clients": len(clients),
        "input_dim": spec.dim,
        "clients_file": "clients.jsonl",
        "shards_dir": "shards",
        "global_test": "shards/global_test.pt",
        "partition_stats_file": "partition_stats.json",
        "client_stats_file": "client_stats.csv",
        "partition_strategy": "analytic",
        "partition_key": "offset_index",
        "seed": seed,
        "reference": reference_of(spec),
    }
    manifest_path = save_manifest(output_dir, manifest)
    save_clients_jsonl(output_dir, clients)
    return GenerationSummary(
        manifest_path=manifest_path,
        num_clients=len(clients),
        num_examples=len(clients) * spec.dim,
        num_test_examples=len(clients) * spec.dim,
    )


def _write_partition_stats(
    output_dir: Path,
    spec: ProblemSpec,
    clients: Sequence[Mapping[str, Any]],
) -> None:
    """Write the two files ``fedbrew inspect-data`` reads beside the manifest.

    No label counts: a regression has no labels, and an empty
    ``global_label_counts`` would read as a partition with none rather than a
    problem with none.
    """

    payload = {
        "dataset_name": DATASET_NAME,
        "partition_strategy": "analytic",
        "num_clients": len(clients),
        "total_examples": sum(int(client["num_examples"]) for client in clients),
        "min_examples_per_client": 3 * spec.dim,
        "max_examples_per_client": 3 * spec.dim,
        "mean_examples_per_client": 3.0 * spec.dim,
        "penalty_strength": spec.penalty_strength,
        "penalty": spec.penalty,
        "heterogeneity": spec.heterogeneity,
        "heterogeneity_residual": spec.heterogeneity_residual(),
        "truth_distance_floor": spec.truth_distance_floor(),
        "clients": [dict(client) for client in clients],
    }
    # allow_nan=False: every float here is measured, and Python's json writes
    # a non-finite one as the bare token NaN, which is not JSON and which the
    # next reader chokes on rather than noticing.
    (output_dir / "partition_stats.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "client_stats.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["client_id", "num_examples", "own_support_size"])
        for client in clients:
            writer.writerow(
                [client["client_id"], client["num_examples"], client["own_support_size"]]
            )


# ---------------------------------------------------------------------------
# The task adapter
# ---------------------------------------------------------------------------


class FedLassoTask(TaskAdapter):
    """Bridge between the composite objective and the generic FL orchestration.

    Three of the seven metrics are measured against `x*`, `x_true` and `F*`,
    which are properties of the problem rather than of any batch. They arrive
    in the manifest's ``reference``, written by the generator, and the task
    rebuilds the spec from the dials recorded there.
    """

    def __init__(
        self,
        model_config: Mapping[str, Any] | None = None,
        dataset_metadata: Mapping[str, Any] | None = None,
        device: str = "cpu",
        **unused: Any,
    ) -> None:
        """Build the adapter, and refuse a run whose two halves disagree.

        The arguments are the task contract (chapter 12 §1.4); the ones this
        problem does not use -- ``batch_size``, ``eval_batch_size``,
        ``dataloader_config``, ``reuse_model`` -- arrive in ``unused`` and are
        ignored, because the loader is built per call.

        Args:
            model_config: The resolved ``model`` block. Carries `d` and `lam`.
            dataset_metadata: The manifest, as ``manifest_dataset`` reports
                it. Carries ``reference``, written by the generator.
            device: The resolved ``runtime.device``.
            **unused: The rest of the contract.

        Raises:
            ValueError: If the model block and the generated data describe
                different problems. `lam` travels in ``model.extra`` because
                that is where a run config can carry it, and the targets
                travel in the shards; a `lam` mismatch would fail on nothing
                at all -- the run would descend one objective and be scored
                against another's optimum.
        """

        del unused
        self.device = torch.device(device)
        self.reference = dict((dataset_metadata or {}).get("reference") or {})
        if "problem" not in self.reference:
            raise ValueError(
                "fed_lasso task needs a manifest written by its own generator: the "
                "reference optimum lives in the manifest's `reference` and this data "
                "has none."
            )
        spec = _spec_from_reference(self.reference)
        _check_model_against_reference(model_config or {}, spec)
        self.spec = spec
        self._truth = spec.truth().to(self.device)
        self._optimum = spec.optimum().to(self.device)
        self._optimal_objective = spec.optimal_objective()
        self._truth_support = spec.truth_support()
        # Three arguments, not the two `criterion(outputs, targets)` shape that
        # `TorchClassificationTask` uses: the L1 term is a function of the
        # parameters. README, §"A composite objective does not fit
        # `criterion(outputs, targets)`".
        self._criterion = _composite_loss
        # Read as `getattr(task, "_scaler", None)` by four client rules, to
        # decide whether to refuse `runtime.use_amp: true`.
        self._scaler: Any = None

    # -- TaskAdapter --------------------------------------------------------

    def build_model(self, config: Mapping[str, Any] | None = None) -> LassoModel:
        """Build the vector model named by the `model` config block."""

        from fedbrew.core.registry import models, register_builtin_components

        register_builtin_components()
        values = dict(config or {})
        factory = models.get(str(values.get("name", "lasso_vector")))
        model = factory(values)
        return model.to(self.device)

    def build_dataloader(
        self,
        data: Any,
        config: Mapping[str, Any] | bool | None = None,
    ) -> list[tuple[Tensor, Tensor]]:
        """Cut one client's rows into batches.

        The rows arrive as a shard's ``x`` and ``y``, because that is what
        ``manifest_dataset`` hands over and what every other dataset in the
        repository stores.

        A list, not a generator: ``single_batch`` re-iterates the loader when it
        runs out, so the loader has to be re-iterable. Unlike the scalar
        examples this one has more rows than one, so ``batch_size``,
        ``shuffle`` and ``drop_last`` all reach something.
        """

        loader_config = _loader_config(config)
        features, targets = _rows_of(data)
        features = features.to(self.device)
        targets = targets.to(self.device)
        rows = len(targets)
        batch_size = max(1, int(loader_config.get("batch_size", rows) or rows))
        if bool(loader_config.get("shuffle", False)):
            order = _permutation(rows, loader_config.get("seed"))
            features, targets = features[order], targets[order]
        batches = [
            (features[start : start + batch_size], targets[start : start + batch_size])
            for start in range(0, rows, batch_size)
        ]
        if bool(loader_config.get("drop_last", False)) and len(batches) > 1:
            batches = [batch for batch in batches if len(batch[1]) == batch_size]
        return batches

    def train_step(
        self,
        model: LassoModel,
        batch: Any,
        optimizer: optim.Optimizer | None = None,
    ) -> dict[str, float]:
        """Take one subgradient step on the batch's composite objective.

        The L1 term enters at full strength in every batch, which is the
        standard minibatch form of a composite objective: the smooth part is an
        unbiased estimate of its full-data value and the penalty is exact.
        """

        if optimizer is None:
            optimizer = optim.SGD(model.parameters(), lr=0.01)
        model.train()
        features, targets = self._move_batch(batch)
        optimizer.zero_grad(set_to_none=True)
        loss = self._criterion(model, model(features), targets)
        loss.backward()
        optimizer.step()
        return {"loss": float(loss.detach())}

    def eval_step(self, model: LassoModel, batch: Any) -> dict[str, float]:
        """Measure the batch's objective, and six properties of the iterate."""

        model.eval()
        features, targets = self._move_batch(batch)
        with torch.no_grad():
            loss = self._criterion(model, model(features), targets)
            iterate = model.iterate
            tolerance = model.support_tolerance
            found = support_of(iterate, tolerance)
        return {
            "loss": float(loss),
            "total": float(len(targets)),
            # Carried per batch because compute_metrics is handed the outputs
            # and nothing else, and six of the seven numbers it returns are
            # functions of the iterate and the problem rather than of the batch.
            "optimality_gap": self.spec.objective_at(iterate) - self._optimal_objective,
            "distance_to_optimum": float(torch.linalg.vector_norm(iterate - self._optimum)),
            "distance_to_truth": float(torch.linalg.vector_norm(iterate - self._truth)),
            "support_size": float(len(found)),
            "support_f1": support_f1(found, self._truth_support),
            "exact_zeros": float(int((iterate == 0.0).sum())),
        }

    def compute_metrics(self, outputs: Sequence[Any]) -> dict[str, float]:
        """Fold eval-step outputs into the seven numbers this task reports.

        ``loss``
            The example-weighted mean of the composite objective over whatever
            was evaluated. On the central pass that is every client's rows, so
            it *is* `F(x)` -- but `F* != 0` here, so unlike ``examples/pl-1d``
            and ``examples/drift-quad`` it is **not** the optimality gap. `F*`
            carries the heterogeneity residual `zeta^2/2` that no `x` removes.

        ``optimality_gap``
            `F(x) - F*`, against the closed-form `x*`. The one column that says
            how well the run solved the problem it was given.

        ``distance_to_optimum`` / ``distance_to_truth``
            `||x - x*||` and `||x - x_true||`. The second has a floor of
            `||x* - x_true||`, which is `lam sqrt(s)` at noise 0, so it is not
            an optimiser score and the README reports the floor beside it.

        ``support_size`` / ``support_f1``
            Measured at ``model.support_tolerance``, which is a reporting
            choice and not part of the objective.

        ``exact_zeros``
            Coordinates that are bit-for-bit 0.0. `d` at initialisation, 0 from
            round 1 on every shipped arm, because none of them applies a
            proximal operator.
        """

        records = [record for record in outputs if isinstance(record, Mapping)]
        names = (
            "loss",
            "optimality_gap",
            "distance_to_optimum",
            "distance_to_truth",
            "support_size",
            "support_f1",
            "exact_zeros",
        )
        if not records:
            return dict.fromkeys(names, 0.0)

        weights = [float(record.get("total", 0.0)) for record in records]
        total = sum(weights)

        def pooled(name: str) -> float:
            values = [float(record.get(name, 0.0)) for record in records]
            if total:
                return sum(v * w for v, w in zip(values, weights, strict=True)) / total
            return sum(values) / len(values)

        return {name: pooled(name) for name in names}

    # -- the server's central pass -----------------------------------------

    def evaluate_model(self, model: LassoModel, data: Any) -> dict[str, float]:
        """Measure the global model on every client's rows at once.

        ``FedAvgServer.evaluate_global`` prefixes what this returns with
        ``global_`` and hands it to ``loop._evaluate_central_test_set``, which
        passes through every finite numeric key it is given -- so all seven
        arrive as ``central_test_<name>``.
        """

        outputs = [self.eval_step(model, batch) for batch in self.build_dataloader(data, None)]
        return self.compute_metrics(outputs)

    # -- narrowing a split to given positions --------------------------------

    def count_examples(self, data: Any) -> int:
        """The design rows `data` holds, which are what `select_examples` indexes."""

        return len(_rows_of(data)[1])

    def select_examples(self, data: Any, indices: Tensor) -> dict[str, Tensor]:
        """The design rows and targets of `data` at `indices`, in that order.

        Returned as a shard's ``x`` and ``y``, so `build_dataloader` reads it as
        it reads any split: at ``batch_size=len(indices)`` and no shuffle it
        yields exactly one batch holding those rows in that order. A client
        that picks its own batches by position calls this instead of asking
        the loader for a permutation it did not choose.
        """

        features, targets = _rows_of(data)
        return {"x": features[indices], "y": targets[indices]}

    # -- this task's own batch splitter -------------------------------------

    def _move_batch(self, batch: Any) -> tuple[Tensor, Tensor]:
        """Split a batch into (features, targets), both on the device."""

        features, targets = batch
        return features.to(self.device), targets.to(self.device)


def _composite_loss(model: LassoModel, outputs: Tensor, targets: Tensor) -> Tensor:
    """`(1/2|B|) ||H_B x - y_B||^2 + lam ||x||_1`, as a scalar tensor."""

    residual = outputs - targets
    return 0.5 * (residual * residual).mean() + model.penalty()


def _rows_of(data: Any) -> tuple[Tensor, Tensor]:
    """The design rows and targets in one split of a shard."""

    if isinstance(data, Mapping):
        features = data.get("x", data.get("X", data.get("features")))
        targets = data.get("y", data.get("Y", data.get("targets")))
        if isinstance(features, Tensor) and isinstance(targets, Tensor):
            return features.to(DTYPE), targets.to(DTYPE)
    raise ValueError("fed_lasso data must be a shard mapping carrying 'x' and 'y' tensors")


def _check_model_against_reference(model_config: Mapping[str, Any], spec: ProblemSpec) -> None:
    """Refuse a model block that describes a different problem than the data."""

    dim = int(model_config.get("input_dim", 0))
    penalty_strength = float(model_config.get("penalty_strength", 0.05))
    penalty = str(model_config.get("penalty", "l1"))
    if dim == spec.dim and penalty_strength == spec.penalty_strength and penalty == spec.penalty:
        return
    raise ValueError(
        f"model config describes d={dim}, lam={penalty_strength}, penalty={penalty!r}, but "
        f"the generated data is d={spec.dim}, lam={spec.penalty_strength}, "
        f"penalty={spec.penalty!r} (manifest reference). The targets come from the "
        "shards and the penalty from the model block, so a disagreement scores the "
        "run against the wrong optimum."
    )


def _loader_config(config: Mapping[str, Any] | bool | None) -> dict[str, Any]:
    if isinstance(config, bool):
        return {"shuffle": config}
    return dict(config or {})


def _permutation(count: int, seed: Any) -> Tensor:
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    return torch.randperm(count, generator=generator)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

#: The three registry names this example adds. ``DATASET_NAME`` above is the
#: generator's; the task shares the name, because a run config's
#: ``dataset.name`` and its task are the same problem seen from two commands.
TASK_NAME = "fed_lasso"
MODEL_NAME = "lasso_vector"


def register() -> None:
    """Register the generator, the task and the model.

    Called once by ``fedbrew.core.extensions``, for the entry a config names
    under ``experiment.extensions`` or ``dataset.extensions``. Nothing is
    registered at import, so this module can be imported for
    :class:`ProblemSpec` alone, without touching the registries.
    """

    from fedbrew.core import registry

    registry.generators.register(
        DATASET_NAME,
        generate_fed_lasso_from_config,
        sections={
            "problem": {"dim", "sparsity", "penalty_strength", "penalty", "noise", "heterogeneity"}
        },
    )
    registry.tasks.register(TASK_NAME, lambda **kwargs: FedLassoTask(**kwargs))
    registry.models.register(MODEL_NAME, build_lasso_vector, task=TASK_NAME)


def _self_check() -> None:
    """Assert the six claims this module's docstring makes about the problem.

    Cheap at import: a 16x16 matmul and one backward pass. Each is load-bearing.
    If the design is not orthonormal the lasso solution is not a soft threshold
    and `x*` is wrong. If the offsets do not sum to zero, `theta_bar` -- and so
    `x*`, its support and `F*` -- is wrong. If `x*` does not satisfy the lasso
    optimality conditions it is not the optimum and every gap is measured from
    the wrong place. If the stored targets do not reproduce `theta_i` the data
    is not the problem the module documents. If the spec the task rebuilds
    from the manifest is not the spec the generator wrote, the run is scored
    against a different problem. And if autograd disagrees with
    :func:`gradient` the run is descending something else.
    """

    spec = ProblemSpec()
    design = spec.design()
    rows = spec.dim

    gram = design.T @ design
    if not torch.equal(gram, rows * torch.eye(rows, dtype=DTYPE)):
        raise AssertionError("the design is not orthogonal: H^T H != m I")

    offsets = spec.client_offsets()
    if float(offsets.sum(dim=0).abs().max()) != 0.0:
        raise AssertionError("client offsets must sum to exactly zero")

    recovered = spec.client_targets() @ design / rows
    if float((recovered - spec.client_ols()).abs().max()) > 1e-14:
        raise AssertionError("stored targets do not reproduce the OLS solutions")

    # Lasso optimality, coordinate by coordinate: on the support the gradient of
    # the smooth part balances lam exactly; off it, the smooth part's gradient
    # is inside the subdifferential ball of radius lam.
    optimum, pooled = spec.optimum(), spec.pooled_ols()
    for index in range(spec.dim):
        smooth = float(optimum[index] - pooled[index])
        if optimum[index] != 0.0:
            residual = smooth + spec.penalty_strength * float(torch.sign(optimum[index]))
            if abs(residual) > 1e-14:
                raise AssertionError("x* violates the lasso stationarity condition")
        elif abs(smooth) > spec.penalty_strength + 1e-14:
            raise AssertionError("x* violates the lasso subdifferential condition")

    if not spec.support_recoverable():
        raise AssertionError("the default spec should have a recoverable support")

    # The L2 dial, on the same claims. Its optimum is stationary rather than
    # subdifferentially optimal -- the objective is differentiable everywhere,
    # which is the whole of what the control removes -- and the closed form
    # written as the textbook ridge estimator has to be the same vector as the
    # shrinkage the reduced problem gives, or one of the two spellings in the
    # README is wrong.
    ridge = ProblemSpec(penalty="l2")
    ridge_optimum = ridge.optimum()
    stationarity = (ridge_optimum - ridge.pooled_ols()) + (
        ridge.penalty_strength / ridge.dim
    ) * ridge_optimum
    if float(stationarity.abs().max()) > 1e-15:
        raise AssertionError("the ridge x* violates the stationarity condition")
    pooled_targets = ridge.client_targets().mean(dim=0)
    textbook = design.T @ pooled_targets / (ridge.dim + ridge.penalty_strength)
    if float((textbook - ridge_optimum).abs().max()) > 1e-14:
        raise AssertionError("(m + lam)^-1 H^T y_bar is not the ridge x*")
    # Ridge selects nothing: `x*` has a zero exactly where `theta_bar` has one,
    # because every coordinate is multiplied by the same factor. At noise 0
    # `theta_bar` *is* `x_true`, so the ridge `x*` inherits its thirteen zeros
    # and `support_recoverable()` reads True -- which is the trap the README
    # names, and the reason this check is on the mechanism rather than on the
    # flag. The second pair below is the same claim where the two penalties
    # visibly part: under a pooled perturbation smaller than `lam`, soft
    # thresholding still returns the planted support and ridge returns all `d`
    # coordinates.
    if support_of(ridge_optimum, 0.0) != support_of(ridge.pooled_ols(), 0.0):
        raise AssertionError("ridge shrinkage changed the support of theta_bar")
    perturbed = {
        form: ProblemSpec(penalty=form, noise=0.4 * spec.penalty_strength) for form in PENALTIES
    }
    if support_of(perturbed["l1"].optimum(), 0.0) != spec.truth_support():
        raise AssertionError("soft thresholding should still recover the support below lam")
    if len(support_of(perturbed["l2"].optimum(), 0.0)) != spec.dim:
        raise AssertionError("ridge should leave every coordinate non-zero under any noise")

    for candidate in (spec, ridge):
        if _spec_from_reference(reference_of(candidate)) != candidate:
            raise AssertionError("the spec does not survive the round trip through the manifest")

        model = LassoModel(
            dim=candidate.dim,
            penalty_strength=candidate.penalty_strength,
            penalty=candidate.penalty,
            x_init=0.3,
        )
        features, targets = design, candidate.client_targets()[0]
        _composite_loss(model, model(features), targets).backward()
        measured = model.x.grad
        if measured is None:
            raise AssertionError("the backward pass left no gradient on the iterate")
        expected = gradient(
            model.iterate,
            features,
            targets,
            candidate.penalty_strength,
            candidate.penalty,
            candidate.dim,
        )
        if float((measured - expected).abs().max()) > 1e-15:
            raise AssertionError("autograd disagrees with the analytic subgradient")


_self_check()
