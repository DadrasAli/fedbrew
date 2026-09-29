"""Linear classifiers with a regularizer, federated, on planted and real corpora.

Four problems: l1-regularized and ridge logistic regression, logistic
regression with a nonconvex regularizer, and the sigmoid loss with a ridge
term. The first is written out below; the others swap the loss or the penalty
(:data:`LOSSES`, :data:`PENALTIES`, :data:`PROBLEMS`).

The problem
-----------
Rows `(a_i, b_i)` with `a_i` in `R^d` and `b_i` in `{-1, +1}`. The objective is::

    F(x) = (1/n) sum_{i=1}^n log(1 + exp(-b_i x' a_i)) + lam ||x||_1

over `x` in `R^d`, with **no constraint set**: `log(1 + exp(.))` is bounded
below by 0 and `lam ||x||_1 -> infinity`, so `F` is coercive and a minimiser
exists without a box. Writing `l(x)` for the first term::

    grad l(x) = -(1/n) sum_i b_i sigma(-b_i x' a_i) a_i
    F = l + lam ||.||_1,   l smooth and convex, the penalty neither.

The loss and the penalty are named (``problem.loss``, ``problem.penalty`` and
the model's keys of the same names), and a problem is the pair of them with
`lam`: :data:`LOSSES` and :data:`PENALTIES` hold the forms this file defines.

How it is federated
-------------------
Clients `c = 1..N` hold disjoint index sets `I_c` with `m_c = |I_c|` rows each,
and client `c` optimises::

    F_c(x) = (1/m_c) sum_{i in I_c} log(1 + exp(-b_i x' a_i)) + lam ||x||_1

with the **whole** penalty in every client, as ``examples/fed-lasso`` does.
Every client holds the same `m = rows_per_client` rows, so uniform and
example-weighted aggregation are the same average and both reproduce `F`
exactly. ``_self_check`` asserts both spellings against the stored rows.

The data
--------
No draw anywhere. The design and the labels come from low-discrepancy
sequences:

``design``
    `a_ij = Phi^-1(vdc(i, p_j))`, the Halton sequence through the normal
    quantile, with `p_j` the `j`-th prime.

``labels``
    `b_i = +1` if `sigma(x_true . a_i) > vdc(i, q)` else `-1`, with
    `q = p_d` -- **the first prime the design does not use**. A threshold
    sequence in a base a column already uses is a function of the same index
    in the same base, and correlates with that column: base 3 puts a false
    positive into the support at coordinate 1, and base 2 corrupts `x*` while
    the support still matches (measured 2026-09-20).

``x_true``
    ``sparsity`` non-zeros at evenly spaced coordinates, magnitudes halving,
    signs alternating -- fed-lasso's planting rule -- scaled by
    ``signal_scale``.

``clients``
    The rows sorted by their margin `x_true . a_i` (a stable sort) and dealt in
    blocks of ``partition_block`` rows, round robin: client `c` holds blocks
    `c, c + N, c + 2N, ...`. 1 is a stratified deal and `rows_per_client` one
    contiguous band of margins per client; the block is the heterogeneity dial.

Corpora, and the certified optima
---------------------------------
The generator writes a *corpus*: the rows, the labels and the deal, one
dataset shared by every problem posed on it, and its content digest
(:func:`corpus_digest`) into the manifest. The problem -- loss, penalty,
`lam` -- is the run's model block. A convex problem's `x*` has no closed form:
it is solved once (FISTA to identify an L1 support, then Newton on it; damped
Newton for the squared L2 penalty), certified by its KKT residual on the full
vector, and kept in one table (:data:`OPTIMA_TABLE`) keyed by the corpus's
digest, the loss, the penalty and `lam` (:func:`certify`, ``certify.py``). The
task looks `F*` up there and refuses a convex problem without an entry
(:func:`find_optimum`): the one structural difference from fed-lasso, whose
task re-derives its closed form.

What this file registers
------------------------
Three names, at ``register()`` time, and nothing at import::

    generators  "fed_logistic_l1"    writes the shards and the manifest
    tasks       "fed_logistic_l1"    FedLogisticL1Task
    models      "logistic_vector"    LogisticModel, a d-vector plus its penalty

The problem's `lam`, loss and penalty are on the model, because they are the
objective the client descends and ``model.extra`` is where a run config
carries them; the task is the one object handed both them and the corpus, so
it is where the optimum is looked up.
"""

from __future__ import annotations

import bz2
import csv
import ctypes
import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn, optim

from fedbrew.data.manifest_validation import IDENTICAL_TO_TRAIN
from fedbrew.data.writers.manifest import save_clients_jsonl, save_manifest
from fedbrew.data.writers.torch_shards import (
    load_client_shard,
    save_client_shard,
    save_split_client_shard,
)
from fedbrew.models.config_keys import reject_unknown_model_keys
from fedbrew.tasks.base import (
    LoaderOrder,
    ReportedMetrics,
    TaskAdapter,
    batch_row_numbers,
    listed_loader_order,
    row_count,
    row_mean,
    row_numbers,
)

#: Every tensor here is float64: so that ``exact_zeros`` means what it says, and
#: so that the reference solve reaches a KKT residual near 1e-17 rather than
#: near float32's epsilon.
DTYPE = torch.float64


# ---------------------------------------------------------------------------
# Deterministic sequences: the design and the labels, with no draw
# ---------------------------------------------------------------------------


def primes(count: int) -> list[int]:
    """The first `count` primes, by trial division. `count` is `d + 1` here."""

    found: list[int] = []
    candidate = 2
    while len(found) < count:
        if all(candidate % prime for prime in found):
            found.append(candidate)
        candidate += 1
    return found


def van_der_corput(count: int, base: int) -> Tensor:
    """`vdc(i, base)` for `i = 1..count`, as a float64 tensor in `(0, 1)`.

    The radical inverse, digits accumulated lowest first, so the order of the
    float64 additions is fixed by the sequence and not by how the loop is
    written. Indices start at 1 because the normal quantile of 0 is `-inf`.
    """

    index = torch.arange(1, count + 1, dtype=torch.int64)
    value = torch.zeros(count, dtype=DTYPE)
    denominator = 1.0
    while bool(index.any()):
        denominator *= base
        value += (index % base).to(DTYPE) / denominator
        index = index // base
    return value


def halton_normal_design(rows: int, dim: int) -> Tensor:
    """`a_ij = Phi^-1(vdc(i, p_j))`: the design, as a pure function of `(n, d)`."""

    columns = [van_der_corput(rows, prime) for prime in primes(dim)]
    return torch.special.ndtri(torch.stack(columns, dim=1))


def label_base(dim: int) -> int:
    """`p_d`: the first prime the design does not use (see the module docstring)."""

    return primes(dim + 1)[dim]


#: The threads the conditioning QR runs on (:func:`conditioned`).
QR_THREADS = 1


def conditioned(design: Tensor, condition_number: float) -> Tensor:
    """The design's centred columns orthonormalised and rescaled to a given conditioning.

    A reduced QR of the centred design, its columns scaled by the square roots
    of `kappa^(-j/(d-1))`, `j = 0..d-1`, and by `sqrt(n)`: the pooled Gram
    `A'A/n` of the result has eigenvalues from 1 down to `1/kappa`, so its
    condition number is `kappa` and `lambda_max = 1` whatever `kappa` is. The
    QR makes the design depend on the LAPACK it runs on, to about 1e-14, which
    the manifest records (:func:`lapack_record`), and on the threads it runs
    on: one thread and many give different last bits (measured 2026-09-28), so
    it runs on one, and a machine's core count does not change the data.
    """

    rows, dim = design.shape
    threads = torch.get_num_threads()
    torch.set_num_threads(QR_THREADS)
    try:
        basis, _ = torch.linalg.qr(design - design.mean(0), mode="reduced")
    finally:
        torch.set_num_threads(threads)
    scale = torch.logspace(0.0, -0.5 * math.log10(condition_number), dim, dtype=DTYPE)
    return math.sqrt(rows) * basis * scale


def lapack_record() -> dict[str, Any]:
    """What a conditioned design depends on: torch, its BLAS and LAPACK, and the CPU."""

    config = torch.__config__.show()

    def found(pattern: str) -> str | None:
        match = re.search(pattern, config)
        return match.group(1) if match else None

    cpu = None
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            cpu = next(
                (line.split(":", 1)[1].strip() for line in handle if line.startswith("model name")),
                None,
            )
    except OSError:
        pass
    return {
        "torch": torch.__version__,
        "blas": found(r"BLAS_INFO=(\w+)"),
        "lapack": found(r"LAPACK_INFO=(\w+)"),
        "cpu_capability": found(r"CPU capability usage: (\w+)"),
        "cpu": cpu,
        "qr_threads": QR_THREADS,
    }


def deal(key: Tensor, clients: int, rows_per_client: int, block: int) -> list[Tensor]:
    """Rows sorted by `key` (stable), dealt in blocks of `block`, round robin."""

    order = torch.argsort(key, stable=True)
    return [
        torch.cat(
            [
                order[(group * clients + client) * block : (group * clients + client + 1) * block]
                for group in range(rows_per_client // block)
            ]
        )
        for client in range(clients)
    ]


# ---------------------------------------------------------------------------
# The objective: a loss of the margin, and a penalty of the iterate
# ---------------------------------------------------------------------------


def _logistic(signed: Tensor, mask: Tensor | None = None) -> Tensor:
    """`mean log(1 + exp(z))` over the rows, `z = -b a.x`, through ``softplus``."""

    return row_mean(torch.nn.functional.softplus(signed), mask)


def _logistic_weights(signed: Tensor) -> Tensor:
    """`d loss_i / d z_i`: `sigma(z)`."""

    return torch.sigmoid(signed)


def _tanh(signed: Tensor, mask: Tensor | None = None) -> Tensor:
    """`1 + mean tanh(z)`, the sigmoid loss `1 - tanh(b a.x)`: the constant added once."""

    return 1.0 + row_mean(torch.tanh(signed), mask)


def _tanh_weights(signed: Tensor) -> Tensor:
    """`d loss_i / d z_i`: `1 - tanh(z)^2`."""

    return 1.0 - torch.tanh(signed).square()


#: The losses of the margin, each as (the mean over a batch's rows of `z =
#: -b a.x`, its derivative in `z` per row, whether it is convex, and the bound
#: on its second derivative that makes `grad l` Lipschitz with `L = bound *
#: ||A||^2 / n`).
LOSSES: dict[str, tuple[Callable[..., Tensor], Callable[[Tensor], Tensor], bool, float]] = {
    "logistic": (_logistic, _logistic_weights, True, 0.25),
    # |tanh''| <= 4 / (3 sqrt 3), attained where tanh(z)^2 = 1/3.
    "tanh": (_tanh, _tanh_weights, False, 4.0 / (3.0 * math.sqrt(3.0))),
}


def _l1(x: Tensor, lam: float) -> Tensor:
    return lam * x.abs().sum()


def _l1_gradient(x: Tensor, lam: float) -> Tensor:
    # sign(0) = 0: the minimum-norm subgradient, and what autograd gives |x|.
    return lam * torch.sign(x)


def _l2sq(x: Tensor, lam: float) -> Tensor:
    return 0.5 * lam * (x * x).sum()


def _l2sq_gradient(x: Tensor, lam: float) -> Tensor:
    return lam * x


def _nonconvex(x: Tensor, lam: float) -> Tensor:
    squared = x * x
    return lam * (squared / (1.0 + squared)).sum()


def _nonconvex_gradient(x: Tensor, lam: float) -> Tensor:
    # 2 lam x / (1 + x^2)^2: bounded by lam 3 sqrt(3) / 8, and |r''| <= 2 lam.
    return x / (1.0 + x * x).square() * (2.0 * lam)


#: The penalties, each as (its value, its (sub)gradient, whether it is convex,
#: and the bound on its second derivative, None where it has none).
PENALTIES: dict[
    str, tuple[Callable[[Tensor, float], Tensor], Callable[[Tensor, float], Tensor], bool, Any]
] = {
    "l1": (_l1, _l1_gradient, True, None),
    "l2sq": (_l2sq, _l2sq_gradient, True, 1.0),
    "nonconvex": (_nonconvex, _nonconvex_gradient, False, 2.0),
}


#: The problems this example poses, as (loss, penalty) pairs: each has
#: generator configs and an arm, and is held by the tests batched against
#: sequential.
PROBLEMS: tuple[tuple[str, str], ...] = (
    ("logistic", "l1"),
    ("logistic", "l2sq"),
    ("logistic", "nonconvex"),
    ("tanh", "l2sq"),
)


def convex(loss: str, penalty: str) -> bool:
    """Whether a problem is convex, and so has a certified `F*`."""

    return LOSSES[loss][2] and PENALTIES[penalty][2]


def penalty_value(x: Tensor, lam: float, penalty: str = "l1") -> Tensor:
    """The penalty at `x`, as a scalar tensor."""

    return PENALTIES[penalty][0](x, lam)


def mean_loss(
    outputs: Tensor, labels: Tensor, loss: str = "logistic", mask: Tensor | None = None
) -> Tensor:
    """The mean loss of a batch's margins `outputs = A_B x`."""

    return LOSSES[loss][0](-labels * outputs, mask)


def smooth_gradient(x: Tensor, features: Tensor, labels: Tensor, loss: str = "logistic") -> Tensor:
    """`grad l(x) = -(1/n) A' (b * loss'(z))`, the analytic gradient of the mean loss."""

    weights = LOSSES[loss][1](-labels * (features @ x))
    return features.T @ (-labels * weights) / len(labels)


def objective(
    x: Tensor,
    features: Tensor,
    labels: Tensor,
    penalty_strength: float,
    loss: str = "logistic",
    penalty: str = "l1",
) -> float:
    """`F(x)` over the rows given, as a plain float."""

    return float(mean_loss(features @ x, labels, loss)) + float(
        penalty_value(x, penalty_strength, penalty)
    )


def gradient(
    x: Tensor,
    features: Tensor,
    labels: Tensor,
    penalty_strength: float,
    loss: str = "logistic",
    penalty: str = "l1",
) -> Tensor:
    """The analytic (sub)gradient of `F`: the smooth part plus the penalty's."""

    return smooth_gradient(x, features, labels, loss) + PENALTIES[penalty][1](x, penalty_strength)


def kkt_residual(
    x: Tensor, features: Tensor, labels: Tensor, penalty_strength: float, penalty: str = "l1"
) -> float:
    """How far `x` is from the optimality conditions of the logistic problem.

    For the L1 penalty, per coordinate: `|grad l(x)_k + lam sign(x_k)|` where
    `x_k != 0`, and `max(|grad l(x)_k| - lam, 0)` where it is 0. Zero exactly at
    the minimiser, and with `l` convex a point at residual `r` has
    `F(x) - F* <= r ||x - x*||_1`: the whole certificate. For the squared L2
    penalty `F` is smooth, and the residual is `max_k |grad F(x)_k|`.
    """

    smooth = smooth_gradient(x, features, labels)
    if penalty == "l2sq":
        return float((smooth + penalty_strength * x).abs().max())
    if penalty == "l1":
        violations = torch.where(
            x != 0.0,
            (smooth + penalty_strength * torch.sign(x)).abs(),
            torch.clamp(smooth.abs() - penalty_strength, min=0.0),
        )
        return float(violations.max())
    raise ValueError(f"no certificate for the {penalty!r} penalty")


def soft_threshold(values: Tensor, level: float) -> Tensor:
    """`S_t(v)_j = sign(v_j) max(0, |v_j| - t)`, the L1 proximal operator.

    Here to *compute* the reference solution and for nothing else: nothing
    under ``fedbrew/clients/`` or ``fedbrew/servers/`` applies it.
    """

    return torch.sign(values) * torch.clamp(values.abs() - level, min=0.0)


def support_of(values: Tensor, tolerance: float) -> set[int]:
    """The coordinates a run would call non-zero at `tolerance`."""

    return {index for index, value in enumerate(values.tolist()) if abs(value) > tolerance}


def lipschitz_of(features: Tensor) -> float:
    """`||A||_2^2 / 4n`: the Lipschitz constant of the logistic loss's gradient."""

    return float(torch.linalg.matrix_norm(features, 2)) ** 2 / (4.0 * features.shape[0])


# ---------------------------------------------------------------------------
# Real data: a LIBSVM file, pinned by digest and never downloaded
# ---------------------------------------------------------------------------

#: Where the LIBSVM binary collection lives, for the message a missing file
#: raises. Written out rather than fetched: see :func:`load_source`.
LIBSVM_BINARY = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary"

#: The partition keys: the margin as the design's product with the reference
#: solution (``float``), or each row's sum taken exactly against that solution
#: rounded to ten significant digits (``exact``), which is the same key on any
#: machine whatever its BLAS does with the order.
PARTITION_KEYS = ("float", "exact")

#: The significant digits ``exact`` rounds the reference solution to.
EXACT_KEY_DIGITS = 10


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """A LIBSVM file on disk, identified by its content and not by its name.

    Frozen and hashable, so :func:`load_source` can cache the parse against it.
    """

    #: Where the file is, relative to the directory the generator runs in.
    path: str
    #: The SHA-256 the file must have, checked before a byte is parsed.
    sha256: str
    #: Where a reader can fetch it: part of the refusal message and the
    #: manifest. Nothing here requests it.
    url: str = ""
    #: The only format this reads, stated so a second would be a config change.
    format: str = "libsvm"
    #: Scale every row to unit L2 norm.
    row_normalize: bool = True

    def __post_init__(self) -> None:
        if self.format != "libsvm":
            raise ValueError(f"source.format must be 'libsvm', not {self.format!r}")
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256):
            raise ValueError("source.sha256 must be 64 lowercase hex digits")


def digest_of(path: Path) -> str:
    """The file's SHA-256, read a megabyte at a time."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _libsvm_row(line: str, dim: int, where: str) -> tuple[float, list[int], list[float]]:
    """One LIBSVM line as `(label, 0-based columns, values)`; every refusal names the line."""

    fields = line.split()
    if not fields:
        raise ValueError(f"{where}: an empty row")
    label = float(fields[0])
    if label not in (1.0, -1.0):
        raise ValueError(f"{where}: label {label:g}, and this problem's labels are -1 or +1")
    columns, values = [], []
    for field in fields[1:]:
        key, separator, value = field.partition(":")
        if not separator:
            raise ValueError(f"{where}: {field!r} is not index:value")
        column = int(key)
        if not 1 <= column <= dim:
            raise ValueError(f"{where}: feature index {column} is outside 1..{dim}")
        columns.append(column - 1)
        values.append(float(value))
    return label, columns, values


def read_libsvm(path: Path, dim: int, rows: int) -> tuple[Tensor, Tensor]:
    """The **first** `rows` rows of a LIBSVM file, dense, as `(features, labels)`.

    First in file order, and the rest of the file is not read: the trim is how
    `n = N m` is hit exactly. A file with fewer rows is refused.
    """

    row_ids: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    labels: list[float] = []
    opener = bz2.open if path.suffix == ".bz2" else open
    with opener(path, "rt", encoding="ascii") as handle:  # type: ignore[operator]
        for line in handle:
            if len(labels) == rows:
                break
            label, row_columns, row_values = _libsvm_row(line, dim, f"{path}:{len(labels) + 1}")
            row_ids.extend([len(labels)] * len(row_columns))
            columns.extend(row_columns)
            values.extend(row_values)
            labels.append(label)
    if len(labels) < rows:
        raise ValueError(f"{path} holds {len(labels)} rows, and the dials ask for {rows}")
    features = torch.zeros((rows, dim), dtype=DTYPE)
    features[torch.tensor(row_ids), torch.tensor(columns)] = torch.tensor(values, dtype=DTYPE)
    return features, torch.tensor(labels, dtype=DTYPE)


@lru_cache(maxsize=2)
def load_source(source: SourceSpec, dim: int, rows: int) -> tuple[Tensor, Tensor]:
    """The trimmed, optionally row-normalised design and labels of a source.

    **This never downloads.** The file is fetched once, by hand, and what pins
    the data is its digest; a missing file is refused with the command that
    fetches it, and a file with another digest is refused rather than
    generating a different problem under the same name.
    """

    path = Path(source.path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing, and this generator never downloads. Fetch it once:\n"
            f"  mkdir -p {path.parent} && curl -L -o {path} {source.url or LIBSVM_BINARY}\n"
            f"Generation then checks its SHA-256 against the {source.sha256} the config pins."
        )
    found = digest_of(path)
    if found != source.sha256:
        raise ValueError(
            f"{path} has SHA-256 {found}, and the config pins {source.sha256}. The file is not "
            "the one this dataset was defined on: re-fetch it, or change the pin deliberately."
        )
    features, labels = read_libsvm(path, dim, rows)
    if not source.row_normalize:
        return features, labels
    norms = torch.linalg.vector_norm(features, dim=1, keepdim=True)
    if float(norms.min()) == 0.0:
        raise ValueError(f"{path}: a row of the trim is all zeros, so it has no unit-norm form")
    return features / norms, labels


def exact_margins(features: Tensor, weights: Tensor, digits: int = EXACT_KEY_DIGITS) -> Tensor:
    """`a_i . w` with `w` rounded to `digits` significant digits, each row summed exactly.

    Each product is one IEEE multiplication and ``math.fsum`` rounds the sum
    once, so the key does not depend on the order a BLAS sums in, and two
    reference solves that agree to more than `digits` digits deal the same
    partition.
    """

    rounded = torch.tensor(
        [float(f"{value:.{digits - 1}e}") for value in weights.tolist()], dtype=DTYPE
    )
    products = (features * rounded).tolist()
    return torch.tensor([math.fsum(row) for row in products], dtype=DTYPE)


# ---------------------------------------------------------------------------
# The reference solver: FISTA to find the support, Newton on it to finish
# ---------------------------------------------------------------------------

#: What a certified solve must reach on the full vector. Generation refuses a
#: reference above it: `x*` is what every optimality gap on the data is
#: measured against, and a wrong one is a different problem.
CERTIFICATE = 1.0e-12

#: How still the iterate must be, beside an unchanged support, before FISTA
#: hands over to the polish in a certified solve. Loose on purpose: FISTA is
#: there to identify the support, and the certificate on the full vector is
#: what makes the answer right.
SETTLED_MOVE = 1.0e-3

#: A certified solve's settle counters: the support must hold for this many
#: iterations, inside a budget of this many per block. Short blocks, because
#: each ends in a polish and the polish is where a coordinate leaves the
#: support (on Gisette at `lam = 5e-5`, blocks of 5,000 certify in three).
SETTLE_ITERATIONS = 1_000
CERTIFIED_FISTA_ITERATIONS = 5_000


def solve_reference(
    features: Tensor,
    labels: Tensor,
    penalty_strength: float,
    iterations: int = 20_000,
    polish_steps: int = 60,
    settle: int | None = None,
    rounds: int = 12,
) -> tuple[Tensor, float]:
    """`(x*, kkt_residual(x*))` for the pooled L1-logistic problem.

    **FISTA** at the step `1/L` identifies the support -- the proximal step
    sets coordinates to exactly zero -- and **Newton on the support**, with its
    signs fixed, takes the residual to machine precision. The residual is
    computed on the full vector, so the off-support condition is checked
    rather than assumed.

    ``settle`` makes it a *certified* solve: FISTA stops once the support has
    held still, and the pair repeats, up to ``rounds`` times, while the
    residual is above :data:`CERTIFICATE` -- the next FISTA block is what lets
    a violating coordinate back in. The polish is a side branch off FISTA's own
    iterate, never a point FISTA resumes from: a point fitted under the wrong
    support is a worse start for a proximal method than the unpolished one.
    """

    dim = features.shape[1]
    step = 1.0 / lipschitz_of(features)
    robust = settle is not None
    current = torch.zeros(dim, dtype=DTYPE)
    polished, residual = current, float("inf")
    for _ in range(rounds if robust else 1):
        current = _fista(features, labels, penalty_strength, step, iterations, current, settle)
        polished = _polish(
            features, labels, penalty_strength, current.clone(), polish_steps, robust
        )
        residual = kkt_residual(polished, features, labels, penalty_strength)
        if not robust or residual <= CERTIFICATE:
            break
    return polished, residual


def _fista(
    features: Tensor,
    labels: Tensor,
    penalty_strength: float,
    step: float,
    iterations: int,
    start: Tensor,
    settle: int | None = None,
) -> Tensor:
    """FISTA from `start`, for `iterations` steps or until the support settles.

    With ``settle`` it also restarts adaptively (O'Donoghue and Candes): when
    the momentum has carried the iterate uphill the weight is reset to 1.
    """

    current = start.clone()
    momentum = current.clone()
    weight = 1.0
    unchanged, moved = 0, 0.0
    for _ in range(iterations):
        candidate = momentum - step * smooth_gradient(momentum, features, labels)
        following = soft_threshold(candidate, step * penalty_strength)
        if settle is not None and float(((momentum - following) * (following - current)).sum()) > 0:
            weight = 1.0
        next_weight = 0.5 * (1.0 + (1.0 + 4.0 * weight * weight) ** 0.5)
        momentum = following + ((weight - 1.0) / next_weight) * (following - current)
        if settle is not None:
            same = bool(torch.equal(following != 0.0, current != 0.0))
            unchanged = unchanged + 1 if same else 0
            moved = float((following - current).abs().max())
        current, weight = following, next_weight
        if settle is not None and unchanged >= settle and moved < SETTLED_MOVE:
            break
    return current


def _polish(
    features: Tensor,
    labels: Tensor,
    penalty_strength: float,
    current: Tensor,
    polish_steps: int,
    robust: bool = False,
) -> Tensor:
    """Newton on the identified support, with its signs held fixed.

    ``robust`` is for data whose support Hessian can be singular -- a corpus
    may carry a column twice, and Gisette does: the step is then taken in the
    Hessian's range (:func:`_newton_step`), and a coordinate the step pushes
    across its own sign leaves the support (:func:`_drop_flipped`).
    """

    rows = features.shape[0]
    support = torch.nonzero(current, as_tuple=False).ravel().tolist()
    if not support:
        return current
    signs = torch.sign(current[support])
    for _ in range(polish_steps):
        residual = smooth_gradient(current, features, labels)[support]
        residual = residual + penalty_strength * signs
        if float(residual.abs().max()) < 1e-18:
            break
        scores = features[:, support]
        probabilities = torch.sigmoid(-labels * (features @ current))
        curvature = probabilities * (1.0 - probabilities)
        hessian = (scores * curvature.unsqueeze(1)).T @ scores / rows
        current[support] -= _newton_step(hessian, residual, robust)
        if robust:
            support, signs = _drop_flipped(current, support, signs)
            if not support:
                break
    return current


def _drop_flipped(current: Tensor, support: list[int], signs: Tensor) -> tuple[list[int], Tensor]:
    """Zero every coordinate the last Newton step pushed across its own sign."""

    index = torch.tensor(support)
    flipped = torch.sign(current[index]) != signs
    if not bool(flipped.any()):
        return support, signs
    current[index[flipped]] = 0.0
    return index[~flipped].tolist(), signs[~flipped]


def _newton_step(hessian: Tensor, residual: Tensor, robust: bool) -> Tensor:
    """`H^-1 r`, or its minimum-norm reading when `H` is singular."""

    if not robust:
        return torch.linalg.solve(hessian, residual)
    values, vectors = torch.linalg.eigh(hessian)
    largest = float(values.max())
    if largest <= 0.0:
        return torch.zeros_like(residual)
    inverted = torch.where(values > largest * 1.0e-12, 1.0 / values, torch.zeros_like(values))
    return vectors @ (inverted * (vectors.T @ residual))


def certified_optimum(
    features: Tensor, labels: Tensor, penalty_strength: float
) -> tuple[Tensor, float]:
    """`(x*, residual)` of the L1 problem, run until it is certified."""

    return solve_reference(
        features,
        labels,
        penalty_strength,
        iterations=CERTIFIED_FISTA_ITERATIONS,
        settle=SETTLE_ITERATIONS,
    )


#: The Newton solve's iteration budget: from 0 it certifies in about ten.
NEWTON_ITERATIONS = 100


def solve_l2sq(features: Tensor, labels: Tensor, penalty_strength: float) -> tuple[Tensor, float]:
    """`(x*, residual)` of the logistic problem with the squared L2 penalty.

    Damped Newton from 0 on the smooth, strongly convex `F`: the Hessian
    `A' diag(s (1 - s)) A / n + lam I`, `s = sigma(-b a.x)`, and a backtracking
    (Armijo) line search, until `max |grad F|` is under :data:`CERTIFICATE`.
    Near machine precision the objective stops decreasing before the residual
    does, and the full step is then taken and the residual left to decide.
    """

    rows = features.shape[0]
    current = torch.zeros(features.shape[1], dtype=DTYPE)
    value = objective(current, features, labels, penalty_strength, penalty="l2sq")
    for _ in range(NEWTON_ITERATIONS):
        grad = gradient(current, features, labels, penalty_strength, penalty="l2sq")
        residual = float(grad.abs().max())
        if residual < CERTIFICATE:
            return current, residual
        probabilities = torch.sigmoid(-labels * (features @ current))
        curvature = probabilities * (1.0 - probabilities)
        hessian = (features * curvature.unsqueeze(1)).T @ features / rows
        hessian.diagonal().add_(penalty_strength)
        step = torch.linalg.solve(hessian, grad)
        current, value = _armijo(features, labels, penalty_strength, current, value, grad, step)
    return current, kkt_residual(current, features, labels, penalty_strength, "l2sq")


def _armijo(
    features: Tensor,
    labels: Tensor,
    penalty_strength: float,
    current: Tensor,
    value: float,
    grad: Tensor,
    step: Tensor,
) -> tuple[Tensor, float]:
    """The Newton step, halved until it decreases `F` enough, or taken whole at the floor."""

    size = 1.0
    decrease = float(grad @ step)
    while True:
        candidate = current - size * step
        found = objective(candidate, features, labels, penalty_strength, penalty="l2sq")
        if found <= value - 1e-4 * size * decrease or size < 1e-12:
            break
        size *= 0.5
    if found >= value and float(grad.abs().max()) < 1e3 * CERTIFICATE:
        candidate = current - step
        found = objective(candidate, features, labels, penalty_strength, penalty="l2sq")
    return candidate, found


@lru_cache(maxsize=2)
def partition_reference(source: SourceSpec, dim: int, rows: int, lam: float) -> Tensor:
    """`w_ref`: the certified L1 `x*` at the lam a real dataset's deal sorts by."""

    features, labels = load_source(source, dim, rows)
    return certified_optimum(features, labels, lam)[0]


# ---------------------------------------------------------------------------
# The problem
# ---------------------------------------------------------------------------

_MODEL_KEYS = ("penalty_strength", "loss", "penalty", "support_tolerance", "x_init", "optima")


@dataclass(frozen=True, slots=True)
class ProblemSpec:
    """A corpus, as its dials, and a problem posed on it: the loss, the penalty, `lam`.

    The corpus dials are what the generator writes; the problem is the run's,
    from its model block. Everything here but :meth:`optimum` is closed form
    and cheap. The optimum is a solve, done once per corpus and problem by
    ``certify.py`` and kept in the optima table (:data:`OPTIMA_TABLE`).
    """

    num_clients: int = 32
    #: `d`. Picks how many primes the design uses.
    dim: int = 32
    #: `m`, the rows each client holds; `n = num_clients * rows_per_client`.
    rows_per_client: int = 64
    #: Non-zeros in `x_true`, at evenly spaced coordinates.
    sparsity: int = 3
    #: What the halving magnitudes are multiplied by. At 1.0 the smallest
    #: planted coefficient is 0.25 and the logistic loss shrinks it to 0.005 at
    #: the widest recovering `lam`; at 2.0 the recovering window is `lam` in
    #: `[0.02, 0.07]` (measured 2026-09-20).
    signal_scale: float = 2.0
    #: `lam`. A problem dial, not a corpus one.
    penalty_strength: float = 0.03
    #: How the margin-sorted rows are dealt: blocks of this many, round robin.
    partition_block: int = 32
    #: The loss of the margin (:data:`LOSSES`).
    loss: str = "logistic"
    #: The penalty `lam` multiplies (:data:`PENALTIES`).
    penalty: str = "l1"
    #: `kappa`: the condition number of the pooled Gram `A'A/n`, by rescaling
    #: the design's orthonormalised columns (:func:`conditioned`). None keeps
    #: the Halton design as it is.
    condition_number: float | None = None
    #: A LIBSVM file to take the rows from instead of planting them. With one
    #: set, ``sparsity`` and ``signal_scale`` describe nothing, and a config
    #: stating them beside a source is refused.
    source: SourceSpec | None = None
    #: The `lam` whose certified L1 `x*` orders a real dataset's rows before the
    #: deal: there is no planted `x_true` to sort by. Its own dial, so settings
    #: that differ in the problem share one partition.
    partition_reference_lambda: float | None = None
    #: How that margin is taken (:data:`PARTITION_KEYS`).
    partition_key: str = "float"
    #: The corpus's name, as the optima table records it beside its digest.
    corpus: str = ""

    def __post_init__(self) -> None:
        """Refuse a spec that cannot express what it claims to."""

        if self.num_clients < 2:
            raise ValueError("partition.num_clients must be at least 2")
        if self.dim < 2:
            raise ValueError("problem.dim must be at least 2")
        if self.rows_per_client < 1:
            raise ValueError("problem.rows_per_client must be at least 1")
        if not 1 <= self.sparsity <= self.dim:
            raise ValueError("problem.sparsity must be between 1 and problem.dim")
        if self.signal_scale <= 0.0:
            raise ValueError("problem.signal_scale must be positive")
        if self.penalty_strength < 0.0:
            raise ValueError("problem.penalty_strength must be non-negative")
        self._check_forms()
        if self.partition_block < 1 or self.rows_per_client % self.partition_block:
            raise ValueError(
                f"problem.partition_block must divide problem.rows_per_client: "
                f"{self.partition_block} does not divide {self.rows_per_client}. The deal "
                "is whole blocks, so a remainder would give some clients fewer rows and "
                "break the equal m_c that makes uniform aggregation exactly F."
            )

    def _check_forms(self) -> None:
        """Refuse a form this file does not define, or dials that cannot go together."""

        if self.source is not None and not self.partition_reference_lambda:
            raise ValueError(
                "problem.partition_reference_lambda is required with a source: real rows carry "
                "no planted signal, so the deal sorts by the margin against a solved x*"
            )
        if self.source is not None and self.condition_number is not None:
            raise ValueError("problem.condition_number reconditions the synthetic design only")
        if self.partition_key not in PARTITION_KEYS:
            raise ValueError(f"problem.partition_key must be one of {PARTITION_KEYS}")
        if self.condition_number is not None and self.condition_number < 1.0:
            raise ValueError("problem.condition_number must be at least 1")
        if self.loss not in LOSSES:
            raise ValueError(f"problem.loss must be one of {sorted(LOSSES)}, not {self.loss!r}")
        if self.penalty not in PENALTIES:
            raise ValueError(
                f"problem.penalty must be one of {sorted(PENALTIES)}, not {self.penalty!r}"
            )

    @property
    def rows(self) -> int:
        """`n = N m`, the pooled row count."""

        return self.num_clients * self.rows_per_client

    @property
    def certified(self) -> bool:
        """Whether this problem has a certified `F*`: a convex loss and penalty."""

        return convex(self.loss, self.penalty)

    # -- the ground truth ---------------------------------------------------

    def truth(self) -> Tensor:
        """`x_true`: `sparsity` non-zeros, evenly spaced, magnitudes halving."""

        if self.source is not None:
            raise ValueError("a source has no planted signal: x_true is not defined on real data")
        values = torch.zeros(self.dim, dtype=DTYPE)
        for order in range(self.sparsity):
            index = (order * self.dim) // self.sparsity
            values[index] = self.signal_scale * (-1.0) ** order * 0.5**order
        return values

    def truth_support(self) -> set[int]:
        """The planted support, as coordinate indices."""

        return {(order * self.dim) // self.sparsity for order in range(self.sparsity)}

    # -- the data -----------------------------------------------------------

    def design(self) -> Tensor:
        """`A`, shape `(n, d)`, in source order: Halton, conditioned when asked, or the file's."""

        if self.source is not None:
            return load_source(self.source, self.dim, self.rows)[0]
        design = halton_normal_design(self.rows, self.dim)
        if self.condition_number is None:
            return design
        return conditioned(design, self.condition_number)

    def labels(self) -> Tensor:
        """`b`, shape `(n,)`, in `{-1, +1}`: a deterministic Bernoulli draw, or the file's."""

        if self.source is not None:
            return load_source(self.source, self.dim, self.rows)[1]
        probability = torch.sigmoid(self.design() @ self.truth())
        threshold = van_der_corput(self.rows, label_base(self.dim))
        return torch.where(probability > threshold, 1.0, -1.0).to(DTYPE)

    def margin_key(self) -> Tensor:
        """The per-row score the deal sorts by: `x_true . a_i`, or `w_ref . a_i` on real data."""

        if self.source is None:
            return self.design() @ self.truth()
        reference = partition_reference(
            self.source, self.dim, self.rows, float(self.partition_reference_lambda or 0.0)
        )
        if self.partition_key == "exact":
            return exact_margins(self.design(), reference)
        return self.design() @ reference

    def client_indices(self) -> list[Tensor]:
        """Which rows each client holds: blocks of the margin order, round robin."""

        return deal(self.margin_key(), self.num_clients, self.rows_per_client, self.partition_block)

    # -- the reference optimum ----------------------------------------------

    def optimum(self, iterations: int = 20_000) -> tuple[Tensor, float]:
        """`(x*, its KKT residual)`. **A solve, not a formula.**

        The squared L2 problem by damped Newton (:func:`solve_l2sq`). The L1
        problem by the fixed-budget solve first and, where it stops short of
        :data:`CERTIFICATE` or the rows are real, the certified one.
        """

        features, labels = self.design(), self.labels()
        if self.penalty == "l2sq":
            return solve_l2sq(features, labels, self.penalty_strength)
        if self.source is not None:
            return certified_optimum(features, labels, self.penalty_strength)
        optimum, residual = solve_reference(
            features, labels, self.penalty_strength, iterations=iterations
        )
        if residual > CERTIFICATE:
            optimum, residual = certified_optimum(features, labels, self.penalty_strength)
        return optimum, residual

    def objective_at(self, x: Tensor) -> float:
        """`F(x)`, over every row. Rebuilds the data; the task caches instead."""

        return objective(
            x, self.design(), self.labels(), self.penalty_strength, self.loss, self.penalty
        )

    def lipschitz(self) -> float:
        """`L = ||A||_2^2 / 4n`, the Lipschitz constant of the logistic `grad l`."""

        return lipschitz_of(self.design())

    def lambda_max(self) -> float:
        """The smallest L1 `lam` whose minimiser is `x* = 0`: `||A' b||_inf / 2n`."""

        return float((self.design().T @ self.labels()).abs().max()) / (2.0 * self.rows)

    def client_objective_at(self, x: Tensor) -> list[float]:
        """`F_c(x)` for each client, from the rows that client holds."""

        features, labels = self.design(), self.labels()
        return [
            objective(
                x, features[index], labels[index], self.penalty_strength, self.loss, self.penalty
            )
            for index in self.client_indices()
        ]

    def client_label_balance(self) -> list[float]:
        """Each client's fraction of `+1` labels: what the deal's block moves."""

        labels = self.labels()
        return [float((labels[index] > 0).to(DTYPE).mean()) for index in self.client_indices()]


# ---------------------------------------------------------------------------
# The model: one d-vector, and the penalty it is scored with
# ---------------------------------------------------------------------------


class LogisticModel(nn.Module):  # type: ignore[misc]
    """The iterate `x`, as a `d`-element `nn.Parameter`.

    `lam`, the loss and the penalty are plain attributes, not buffers: a
    buffer is in ``state_dict``, and would be uploaded and averaged every round.
    """

    def __init__(
        self,
        dim: int,
        penalty_strength: float = 0.03,
        support_tolerance: float = 1.0e-3,
        x_init: float = 0.0,
        loss: str = "logistic",
        penalty: str = "l1",
    ) -> None:
        """Place the iterate at `x_init` in every coordinate.

        Args:
            dim: Problem dimension `d`, from ``model.input_dim``.
            penalty_strength: `lam`.
            support_tolerance: The threshold the support metrics use. Not part
                of the objective.
            x_init: Starting value in every coordinate.
            loss: The loss of the margin.
            penalty: The penalty `lam` multiplies.
        """

        super().__init__()
        if loss not in LOSSES:
            raise ValueError(f"model.loss must be one of {sorted(LOSSES)}, not {loss!r}")
        if penalty not in PENALTIES:
            raise ValueError(f"model.penalty must be one of {sorted(PENALTIES)}, not {penalty!r}")
        self.x = nn.Parameter(torch.full((dim,), float(x_init), dtype=DTYPE))
        self.penalty_strength = float(penalty_strength)
        self.support_tolerance = float(support_tolerance)
        self.loss_form = str(loss)
        self.penalty_form = str(penalty)

    def forward(self, features: Tensor) -> Tensor:
        """The margins `A_B x` for a batch of rows."""

        return features @ self.x

    def penalty(self) -> Tensor:
        """The penalty at the iterate, as a scalar tensor."""

        return penalty_value(self.x, self.penalty_strength, self.penalty_form)

    @property
    def iterate(self) -> Tensor:
        """The vector the whole example is about, detached."""

        return self.x.detach().clone()


def build_logistic_vector(config: Mapping[str, Any] | None = None) -> LogisticModel:
    """Registry builder for `model.name: logistic_vector`."""

    values = dict(config or {})
    reject_unknown_model_keys(values, _MODEL_KEYS, "logistic_vector")
    dim = values.get("input_dim")
    if dim is None:
        raise ValueError("model.input_dim is required for logistic_vector: it is the problem's d")
    return LogisticModel(
        dim=int(dim),
        penalty_strength=float(values.get("penalty_strength", 0.03)),
        support_tolerance=float(values.get("support_tolerance", 1.0e-3)),
        x_init=float(values.get("x_init", 0.0)),
        loss=str(values.get("loss", "logistic")),
        penalty=str(values.get("penalty", "l1")),
    )


# ---------------------------------------------------------------------------
# The generator: one dataset per corpus
# ---------------------------------------------------------------------------

#: The name of this problem's generator and task, as configs write it.
DATASET_NAME = "fed_logistic_l1"

#: The ``problem`` keys a generator config may state: the corpus's dials. The
#: loss, the penalty and `lam` are not among them -- they are the run's, and
#: every problem on a corpus shares its one generated dataset.
PROBLEM_KEYS = {
    "corpus",
    "dim",
    "rows_per_client",
    "sparsity",
    "signal_scale",
    "partition_block",
    "condition_number",
    "partition_reference_lambda",
    "partition_key",
}

#: The ``source`` keys a generator config may state.
SOURCE_KEYS = {"format", "path", "sha256", "url", "row_normalize"}

#: The synthetic dials, which a real-data config may not state: they describe a
#: planted signal, and a file has none.
_PLANTED_KEYS = ("sparsity", "signal_scale", "condition_number")


@dataclass(frozen=True, slots=True)
class GenerationSummary:
    """What ``fedbrew generate`` prints when this generator finishes."""

    manifest_path: Path
    num_clients: int
    num_examples: int
    num_test_examples: int


def _spec_from_config(config: Mapping[str, Any]) -> ProblemSpec:
    """Read the corpus out of a generator config's ``problem``, ``partition`` and ``source``."""

    problem = dict(config.get("problem", {}))
    partition = dict(config.get("partition", {}))
    if "num_clients" not in partition:
        raise ValueError(
            "fed_logistic_l1 needs partition.num_clients: it is how many ways the rows are dealt"
        )
    return ProblemSpec(
        num_clients=int(partition["num_clients"]),
        dim=int(problem.get("dim", 32)),
        rows_per_client=int(problem.get("rows_per_client", 64)),
        sparsity=int(problem.get("sparsity", 3)),
        signal_scale=float(problem.get("signal_scale", 2.0)),
        partition_block=int(problem.get("partition_block", 32)),
        condition_number=_optional_float(problem.get("condition_number")),
        source=_source_from_config(config),
        partition_reference_lambda=_optional_float(problem.get("partition_reference_lambda")),
        partition_key=str(problem.get("partition_key", "float")),
        corpus=str(problem.get("corpus", "")),
    )


def _source_from_config(config: Mapping[str, Any]) -> SourceSpec | None:
    """Read the optional ``source`` section, and refuse it beside planted dials."""

    section = config.get("source")
    if section is None:
        return None
    values = dict(section)
    stated = [key for key in _PLANTED_KEYS if key in dict(config.get("problem", {}))]
    if stated:
        raise ValueError(
            f"fed_logistic_l1 was given a source and the planted dials {stated}. Real rows carry "
            "no planted signal: drop them, or drop the source."
        )
    missing = [key for key in ("path", "sha256") if key not in values]
    if missing:
        raise ValueError(f"source needs {missing}: the file, and the digest that pins it")
    return _source_of(values)


def _source_of(values: Mapping[str, Any]) -> SourceSpec:
    return SourceSpec(
        path=str(values["path"]),
        sha256=str(values["sha256"]),
        url=str(values.get("url", "")),
        format=str(values.get("format", "libsvm")),
        row_normalize=bool(values.get("row_normalize", True)),
    )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _spec_from_reference(
    reference: Mapping[str, Any], model_config: Mapping[str, Any] | None = None
) -> ProblemSpec:
    """The corpus from the manifest's ``reference``, and the problem from the model block."""

    corpus = reference["problem"]
    model = dict(model_config or {})
    return ProblemSpec(
        num_clients=int(corpus["clients"]),
        dim=int(corpus["dim"]),
        rows_per_client=int(corpus["rows_per_client"]),
        sparsity=int(corpus.get("sparsity", 3)),
        signal_scale=float(corpus.get("signal_scale", 2.0)),
        penalty_strength=float(model.get("penalty_strength", 0.03)),
        partition_block=int(corpus["partition_block"]),
        loss=str(model.get("loss", "logistic")),
        penalty=str(model.get("penalty", "l1")),
        condition_number=_optional_float(corpus.get("condition_number")),
        source=None if corpus.get("source") is None else _source_of(corpus["source"]),
        partition_reference_lambda=_optional_float(corpus.get("partition_reference_lambda")),
        partition_key=str(corpus.get("partition_key", "float")),
        corpus=str(corpus.get("corpus", "")),
    )


def _corpus_record(spec: ProblemSpec) -> dict[str, Any]:
    """The corpus dials, as the manifest's ``reference.problem`` records them."""

    record: dict[str, Any] = {
        "corpus": spec.corpus,
        "clients": spec.num_clients,
        "dim": spec.dim,
        "rows_per_client": spec.rows_per_client,
        "sparsity": spec.sparsity,
        "signal_scale": spec.signal_scale,
        "partition_block": spec.partition_block,
    }
    if spec.condition_number is not None:
        record["condition_number"] = spec.condition_number
    if spec.source is not None:
        for key in ("sparsity", "signal_scale"):
            del record[key]
        source = spec.source
        record["source"] = {
            "format": source.format,
            "path": source.path,
            "sha256": source.sha256,
            "url": source.url,
            "row_normalize": source.row_normalize,
            "rows_kept": spec.rows,
        }
        record["partition_reference_lambda"] = spec.partition_reference_lambda
        record["partition_key"] = spec.partition_key
    return record


def corpus_digest(features: Tensor, labels: Tensor, clients: int) -> str:
    """The SHA-256 of a corpus as its global shard holds it: every row, dealt, and its label.

    Over the float64 bytes of the stacked rows and labels, in the order the
    global shard stacks them, after a header naming the client count and the
    shape. It is what the optima table keys `F*` by: a certified optimum
    belongs to exactly these rows, and rows that differ in one bit -- another
    LAPACK's QR, another file -- are another corpus.
    """

    digest = hashlib.sha256(
        f"fed_logistic_l1 corpus: {clients} clients, rows {tuple(features.shape)}\n".encode()
    )
    for tensor in (features, labels):
        block = tensor.detach().to(device="cpu", dtype=DTYPE).contiguous()
        digest.update(ctypes.string_at(block.data_ptr(), block.numel() * block.element_size()))
    return digest.hexdigest()


def corpus_reference(spec: ProblemSpec) -> dict[str, Any]:
    """What the manifest records of a corpus: its dials, its digest, and what they determine.

    Copied into ``run.json`` and read back by :class:`FedLogisticL1Task`. No
    optimum: that belongs to a problem posed on the corpus, and lives in the
    optima table under this reference's ``corpus_digest``.
    """

    features, labels = spec.design(), spec.labels()
    stacked = torch.cat(spec.client_indices())
    reference: dict[str, Any] = {
        "problem": _corpus_record(spec),
        "corpus_digest": corpus_digest(features[stacked], labels[stacked], spec.num_clients),
        "rows": spec.rows,
        "rows_per_client": spec.rows_per_client,
        "lipschitz": spec.lipschitz(),
        "client_label_balance": spec.client_label_balance(),
    }
    if spec.source is None:
        reference.update(
            label_base=label_base(spec.dim),
            x_true=spec.truth().tolist(),
            truth_support=sorted(spec.truth_support()),
        )
    else:
        reference["lambda_max"] = spec.lambda_max()
    if spec.condition_number is not None:
        reference.update(_gram_record(features))
    return reference


def _sparse(values: Tensor) -> dict[str, list[Any]]:
    """A vector as its non-zeros: `{"indices": [...], "values": [...]}`."""

    indices = torch.nonzero(values, as_tuple=False).ravel()
    return {"indices": indices.tolist(), "values": values[indices].tolist()}


def _x_star_of(entry: Mapping[str, Any], dim: int) -> Tensor:
    """A stored `x*`, dense or by its non-zeros, as a `d`-vector."""

    stored = entry["x_star"]
    if not isinstance(stored, Mapping):
        return torch.tensor(stored, dtype=DTYPE)
    optimum = torch.zeros(dim, dtype=DTYPE)
    indices = torch.tensor([int(index) for index in stored["indices"]], dtype=torch.long)
    optimum[indices] = torch.tensor([float(value) for value in stored["values"]], dtype=DTYPE)
    return optimum


def _gram_record(features: Tensor) -> dict[str, Any]:
    """The pooled Gram's spectrum as built, and what the build depended on."""

    eigenvalues = torch.linalg.eigvalsh(features.T @ features / features.shape[0])
    return {
        "gram_condition": float(eigenvalues[-1] / eigenvalues[0]),
        "gram_lambda_max": float(eigenvalues[-1]),
        "gram_lambda_min": float(eigenvalues[0]),
        "built_with": lapack_record(),
    }


def generate_fed_logistic_l1_from_config(
    config: Mapping[str, Any],
    output_dir: Path,
    seed: int,
    client_splits: Mapping[str, float],
) -> GenerationSummary:
    """Write one shard per client, plus the manifest a run reads.

    One dataset per corpus: every problem posed on it runs on these shards.
    ``seed`` fixes no draw -- the data is a deterministic function of the dials
    -- and is recorded because every dataset records the seed it was made at.
    ``client_splits`` describe a cut and there is nothing to cut: `F_c` is
    defined over all `m` of a client's rows, so all three splits hold them,
    declared as ``client_test_source: identical_to_train``.
    """

    del client_splits
    spec = _spec_from_config(config)
    features = spec.design()
    labels = spec.labels()
    partition = spec.client_indices()
    output_dir = Path(output_dir)
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    reference = corpus_reference(spec)

    clients: list[dict[str, Any]] = []
    for index, rows in enumerate(partition):
        client_id = f"client_{index}"
        x = features[rows].clone()
        y = labels[rows].clone()
        save_split_client_shard(shards_dir / f"{client_id}.pt", x, y, x, y, x, y)
        clients.append(
            {
                "client_id": client_id,
                "shard": f"shards/{client_id}.pt",
                "num_examples": 3 * spec.rows_per_client,
                "num_train_examples": spec.rows_per_client,
                "num_eval_examples": spec.rows_per_client,
                "num_test_examples": spec.rows_per_client,
                "positive_label_fraction": reference["client_label_balance"][index],
            }
        )

    # The server's central pass evaluates F in one go, so the pooled shard is
    # every client's rows stacked: exactly the federated objective, because
    # every client holds the same m rows and so the same weight.
    stacked = torch.cat(partition)
    save_client_shard(
        shards_dir / "global_test.pt", features[stacked].clone(), labels[stacked].clone()
    )
    _write_partition_stats(output_dir, spec, clients, reference)

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
        "partition_key": "margin_block",
        "seed": seed,
        "reference": reference,
    }
    manifest_path = save_manifest(output_dir, manifest)
    save_clients_jsonl(output_dir, clients)
    return GenerationSummary(
        manifest_path=manifest_path,
        num_clients=len(clients),
        num_examples=spec.rows,
        num_test_examples=spec.rows,
    )


def _write_partition_stats(
    output_dir: Path,
    spec: ProblemSpec,
    clients: Sequence[Mapping[str, Any]],
    reference: Mapping[str, Any],
) -> None:
    """Write the two files ``fedbrew inspect-data`` reads beside the manifest."""

    balance = list(reference["client_label_balance"])
    positives = int(round(sum(balance) * spec.rows_per_client))
    payload = {
        "dataset_name": DATASET_NAME,
        "partition_strategy": "analytic",
        "num_clients": len(clients),
        "total_examples": sum(int(client["num_examples"]) for client in clients),
        "min_examples_per_client": 3 * spec.rows_per_client,
        "max_examples_per_client": 3 * spec.rows_per_client,
        "mean_examples_per_client": 3.0 * spec.rows_per_client,
        "global_label_counts": {"-1": spec.rows - positives, "1": positives},
        "partition_block": spec.partition_block,
        "min_positive_label_fraction": min(balance),
        "max_positive_label_fraction": max(balance),
        "clients": [dict(client) for client in clients],
    }
    # allow_nan=False: every float here is measured, and a non-finite one is
    # not JSON.
    (output_dir / "partition_stats.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    columns = ["client_id", "num_examples", "positive_label_fraction"]
    with (output_dir / "client_stats.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for client in clients:
            writer.writerow([client[column] for column in columns])


# ---------------------------------------------------------------------------
# The certified optima: one table, keyed by corpus digest, loss, penalty, lam
# ---------------------------------------------------------------------------

#: The shipped table of certified optima, beside this file. A run config may
#: name another with ``model.optima``.
OPTIMA_TABLE = Path(__file__).resolve().parent / "optima.json"


def certify(spec: ProblemSpec) -> dict[str, Any]:
    """Solve a convex problem on its corpus, certified, as the table's entry for it.

    `x*` by :meth:`ProblemSpec.optimum`, on the rows in source order; `F*` is
    `F` at it over the rows as the global shard stacks them, which is the order
    the task's gap is computed in. Refused above :data:`CERTIFICATE`.
    """

    if not spec.certified:
        raise ValueError(f"{spec.loss}+{spec.penalty} is not convex: it has no certified F*")
    features, labels = spec.design(), spec.labels()
    stacked = torch.cat(spec.client_indices())
    optimum, residual = spec.optimum()
    if residual > CERTIFICATE:
        raise ValueError(
            f"the reference solve reached a KKT residual of {residual:.3e}, above the "
            f"{CERTIFICATE:g} a table entry is certified to"
        )
    return {
        "corpus": spec.corpus,
        "digest": corpus_digest(features[stacked], labels[stacked], spec.num_clients),
        "loss": spec.loss,
        "penalty": spec.penalty,
        "lam": spec.penalty_strength,
        "f_star": objective(
            optimum,
            features[stacked],
            labels[stacked],
            spec.penalty_strength,
            spec.loss,
            spec.penalty,
        ),
        "kkt_residual": residual,
        "optimum_support": sorted(support_of(optimum, 0.0)),
        "x_star": _sparse(optimum),
    }


def read_optima(path: Path) -> list[dict[str, Any]]:
    """The entries of an optima table; an absent file is an empty table."""

    if not path.is_file():
        return []
    return list(json.loads(path.read_text(encoding="utf-8"))["optima"])


def write_optimum(path: Path, entry: Mapping[str, Any]) -> None:
    """Add an entry to a table, replacing the one with the same key, sorted by corpus."""

    key = _key_of(entry)
    entries = [old for old in read_optima(path) if _key_of(old) != key] + [dict(entry)]
    entries.sort(key=lambda item: (item["corpus"], item["loss"], item["penalty"], item["lam"]))
    payload = {
        "about": (
            "Certified optima of examples/fed-logistic-l1's convex problems, keyed by the "
            "corpus's content digest (corpus_digest), the loss, the penalty and lam. "
            "Written by examples/fed-logistic-l1/certify.py."
        ),
        "optima": entries,
    }
    path.write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n", encoding="utf-8")


def _key_of(entry: Mapping[str, Any]) -> tuple[str, str, str, float]:
    return (str(entry["digest"]), str(entry["loss"]), str(entry["penalty"]), float(entry["lam"]))


def find_optimum(
    entries: Sequence[Mapping[str, Any]], digest: str, spec: ProblemSpec
) -> Mapping[str, Any]:
    """The table's entry for this corpus digest and problem, or a refusal saying why not.

    A convex setting without one is refused rather than run without a gap: an
    entry for the same corpus name at another digest means the rows are not
    the ones `F*` was solved on, and no entry means it was never certified.
    """

    wanted = (digest, spec.loss, spec.penalty, float(spec.penalty_strength))
    for entry in entries:
        if _key_of(entry) == wanted:
            return entry
    problem = f"{spec.loss}+{spec.penalty} at lam={spec.penalty_strength}"
    for entry in entries:
        if entry["corpus"] == spec.corpus and _key_of(entry)[1:] == wanted[1:]:
            raise ValueError(
                f"the optima table certifies {problem} on corpus {spec.corpus!r} at digest "
                f"{entry['digest']}, and this data's digest is {digest}: F* was solved on other "
                "rows, so it is not this data's optimum. Regenerate the corpus where the table "
                "was certified, or certify it here with examples/fed-logistic-l1/certify.py."
            )
    raise ValueError(
        f"the optima table has no certified F* for {problem} on corpus {spec.corpus!r} "
        f"(digest {digest}), and the problem is convex, so every run on it reports a gap. "
        "Certify it with examples/fed-logistic-l1/certify.py."
    )


# ---------------------------------------------------------------------------
# The task adapter
# ---------------------------------------------------------------------------


class FedLogisticL1Task(TaskAdapter):
    """Bridge between the composite objective and the generic FL orchestration.

    Looks `x*` and `F*` up in the optima table by the corpus's digest and the
    problem, where the problem is convex: there is no closed form to rebuild
    them from.
    """

    #: What compute_metrics and the central pass can report, and which side of
    #: each is better (TaskAdapter.METRICS). Which of them a run reports depends
    #: on its problem and its corpus (:func:`reported_metrics`): the gap and the
    #: distance to `x*` only where `F*` is certified, the distance to `x_true`
    #: only on a planted corpus, and the gap only on the central pass.
    #: support_size and exact_zeros are read against a support, not as smaller
    #: or larger.
    METRICS = {
        "loss": "min",
        "optimality_gap": "min",
        "distance_to_optimum": "min",
        "distance_to_truth": "min",
        "support_size": "none",
        "support_f1": "max",
        "exact_zeros": "none",
    }

    #: What each metric measures, for the plan header's glosses (TaskAdapter.METRIC_GLOSSES).
    METRIC_GLOSSES = {
        "loss": (
            "client objective (1/m) Σ_i ℓ(b_i a_iᵀx) + λ r(x), ℓ the logistic or the sigmoid "
            "loss and r the l1, the ridge or the nonconvex regularizer"
        ),
        "optimality_gap": "optimality gap F(x) − F*, F* certified in the optima table",
        "distance_to_optimum": "distance ‖x − x*‖₂ to the certified optimum",
        "distance_to_truth": "distance ‖x − x_true‖₂ to the planted vector",
        "support_size": "count of coordinates with |x_j| above model.support_tolerance",
        "support_f1": (
            "F1 of the support above model.support_tolerance against the planted one, or "
            "against x*'s on a corpus with nothing planted"
        ),
        "exact_zeros": "count of coordinates exactly 0.0",
    }

    #: What its grad_norm_sq measures (TaskAdapter.GRAD_NORM_GLOSS).
    GRAD_NORM_GLOSS = (
        "squared norm of the gradient of the federated objective F(x) = (1/n) Σ_i "
        "ℓ(b_i a_iᵀx) + λ r(x) at the global model -- ridge or l1-regularized logistic "
        "regression, logistic regression with a nonconvex regularizer, or the sigmoid loss "
        "with a ridge term; under the l1 regularizer, of F's minimum-norm subgradient, whose "
        "coordinates at x_j = 0 are the smooth gradient's soft-thresholded at λ"
    )

    def __init__(
        self,
        model_config: Mapping[str, Any] | None = None,
        dataset_metadata: Mapping[str, Any] | None = None,
        device: str = "cpu",
        **unused: Any,
    ) -> None:
        """Build the adapter, and refuse a run whose two halves disagree.

        Args:
            model_config: The resolved ``model`` block: `d`, and the problem --
                the loss, the penalty and `lam` -- and optionally ``optima``,
                a table of certified optima other than the shipped one.
            dataset_metadata: The manifest, as ``manifest_dataset`` reports it.
                Carries the corpus's ``reference``, written by the generator.
            device: The resolved ``runtime.device``.
            **unused: The rest of the task contract, ignored because the loader
                is built per call.

        Raises:
            ValueError: If the data carries no corpus reference, its rows are
                not the digest it records, the model block's `d` is not the
                data's, or a convex problem has no certified optimum for this
                corpus in the table (:func:`find_optimum`).
        """

        del unused
        self.device = torch.device(device)
        metadata = dataset_metadata or {}
        self.reference = dict(metadata.get("reference") or {})
        if "corpus_digest" not in self.reference:
            raise ValueError(
                "fed_logistic_l1 task needs a manifest written by its own generator: the "
                "corpus and its digest live in the manifest's `reference`, and this data has "
                "none. Regenerate it with this example's generator."
            )
        model_config = model_config or {}
        spec = _spec_from_reference(self.reference, model_config)
        _check_model_against_reference(model_config, spec)
        self.spec = spec
        pooled = _pooled_rows(spec, metadata)
        digest = corpus_digest(pooled[0], pooled[1], spec.num_clients)
        if digest != self.reference["corpus_digest"]:
            raise ValueError(
                f"the rows on disk have digest {digest}, and the manifest records "
                f"{self.reference['corpus_digest']}: the shards are not the corpus it describes."
            )
        self._optimum: Tensor | None = None
        self._optimal_objective: float | None = None
        entry: Mapping[str, Any] = {}
        if spec.certified:
            table = Path(str(model_config.get("optima") or OPTIMA_TABLE))
            entry = find_optimum(read_optima(table), digest, spec)
            self._optimum = _x_star_of(entry, spec.dim).to(self.device)
            self._optimal_objective = float(entry["f_star"])
        self.optimum_entry = dict(entry)
        # The support a run's is scored against: the planted one, or on real
        # data -- nothing planted -- x*'s own, where the problem has one.
        self._truth: Tensor | None = None
        self._truth_support: set[int] = set(entry.get("optimum_support", ()))
        if spec.source is None:
            self._truth = spec.truth().to(self.device)
            self._truth_support = spec.truth_support()
        self._pooled_features = pooled[0].to(self.device)
        self._pooled_labels = pooled[1].to(self.device)
        self._support_mask = torch.tensor(
            [index in self._truth_support for index in range(spec.dim)], device=self.device
        )
        # What a client evaluation reports, and what the central pass adds to
        # it. `optimality_gap` is F over every row, the same in every client's
        # copy, so it is measured once a round by `evaluate_model`.
        reported = _reported(spec, entry)
        self._names = reported.client
        self._central_names = reported.central
        # Read as `getattr(task, "_scaler", None)` by four client rules, to
        # decide whether to refuse `runtime.use_amp: true`.
        self._scaler: Any = None

    def pooled_objective(self, x: Tensor) -> float:
        """`F(x)` over every row, from the cached pooled rows."""

        spec = self.spec
        return objective(
            x,
            self._pooled_features,
            self._pooled_labels,
            spec.penalty_strength,
            spec.loss,
            spec.penalty,
        )

    # -- TaskAdapter --------------------------------------------------------

    def build_model(self, config: Mapping[str, Any] | None = None) -> LogisticModel:
        """Build the vector model named by the `model` config block."""

        from fedbrew.core.registry import models, register_builtin_components

        register_builtin_components()
        values = dict(config or {})
        factory = models.get(str(values.get("name", "logistic_vector")))
        model = factory(values)
        return model.to(self.device)

    def build_dataloader(
        self,
        data: Any,
        config: Mapping[str, Any] | bool | None = None,
    ) -> list[tuple[Tensor, Tensor]]:
        """Cut one client's rows into batches: a list, so it is re-iterable."""

        loader_config = _loader_config(config)
        features, labels = _rows_of(data)
        features = features.to(self.device)
        labels = labels.to(self.device)
        rows = len(labels)
        batch_size = max(1, int(loader_config.get("batch_size", rows) or rows))
        if bool(loader_config.get("shuffle", False)):
            order = _permutation(rows, loader_config.get("seed"))
            features, labels = features[order], labels[order]
        batches = [
            (features[start : start + batch_size], labels[start : start + batch_size])
            for start in range(0, rows, batch_size)
        ]
        if bool(loader_config.get("drop_last", False)) and len(batches) > 1:
            batches = [batch for batch in batches if len(batch[1]) == batch_size]
        return batches

    def train_step(
        self,
        model: LogisticModel,
        batch: Any,
        optimizer: optim.Optimizer | None = None,
    ) -> dict[str, float]:
        """Take one (sub)gradient step on the batch's composite objective.

        The penalty enters at full strength in every batch: the smooth part is
        an unbiased estimate of its full-data value and the penalty is exact.
        """

        if optimizer is None:
            optimizer = optim.SGD(model.parameters(), lr=0.01)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = self.functional_loss(model, None, None, self._move_batch(batch))
        loss.backward()
        optimizer.step()
        return {"loss": float(loss.detach())}

    def evaluation_total(self, batch: Any) -> float | None:
        """eval_step's "total", the batch's rows, without evaluating."""

        _, labels = self._move_batch(batch)
        return float(len(labels))

    def eval_step(self, model: LogisticModel, batch: Any) -> dict[str, float]:
        """Measure the batch's objective, and the properties of the iterate."""

        model.eval()
        with torch.no_grad():
            outputs = self.functional_eval(model, None, None, self._move_batch(batch))
        return {name: float(value) for name, value in outputs.items()}

    def objective_loss(self, model: Any, batch: Any) -> tuple[Tensor, float]:
        """The batch's objective, as ``train_step`` takes it, and its rows (TaskAdapter)."""

        loss, _ = self.functional_loss(model, None, None, self._move_batch(batch))
        return loss, float(self.evaluation_total(batch) or 0.0)

    def objective_l1(self, model: Any) -> dict[str, float]:
        """``lam ||x||_1`` on ``x`` under the l1 regularizer; nothing under the smooth ones."""

        return {"x": model.penalty_strength} if model.penalty_form == "l1" else {}

    # -- the batched executor (fedbrew.tasks.base.BatchableTask) --------------

    def loader_order(
        self, data: Any, config: Mapping[str, Any] | bool | None = None
    ) -> LoaderOrder:
        """What ``build_dataloader(data, config)`` yields, declared (``LoaderOrder``)."""

        return listed_loader_order(len(_rows_of(data)[1]), config)

    def split_rows(self, data: Any) -> tuple[Tensor, Tensor]:
        """A split's rows and labels, as ``build_dataloader`` slices them."""

        features, labels = _rows_of(data)
        return features.to(self.device), labels.to(self.device)

    def row_batches(
        self, data: Any, config: Mapping[str, Any] | bool | None = None
    ) -> list[Tensor]:
        """``build_dataloader(data, config)``'s batches as row indices, by the same code."""

        numbers = row_numbers(len(_rows_of(data)[1]))
        batches = self.build_dataloader({"x": numbers.unsqueeze(1), "y": numbers}, config)
        return [batch_row_numbers(numbered) for _, numbered in batches]

    def functional_loss(
        self,
        model: LogisticModel,
        params: Mapping[str, Tensor] | None,
        buffers: Mapping[str, Tensor] | None,
        batch: tuple[Tensor, ...],
        mask: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """The batch's composite objective at ``params``, or at the model's own when None."""

        features, labels = batch
        iterate = model.x if params is None else params["x"]
        outputs = (
            model(features)
            if params is None
            else torch.func.functional_call(model, (dict(params), dict(buffers or {})), (features,))
        )
        loss = mean_loss(outputs, labels, model.loss_form, mask) + penalty_value(
            iterate, model.penalty_strength, model.penalty_form
        )
        return loss, {"loss": loss.detach()}

    def functional_eval(
        self,
        model: LogisticModel,
        params: Mapping[str, Tensor] | None,
        buffers: Mapping[str, Tensor] | None,
        batch: tuple[Tensor, ...],
        mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """The batch's objective, and the properties of the iterate, as tensors."""

        loss, _ = self.functional_loss(model, params, buffers, batch, mask)
        iterate = (model.x if params is None else params["x"]).detach()
        found = iterate.abs() > model.support_tolerance
        # Carried per batch because compute_metrics is handed the outputs and
        # nothing else, and all but the loss are functions of the iterate.
        measured = {"loss": loss.detach(), "total": row_count(batch[1], mask)}
        if self._optimum is not None:
            measured["distance_to_optimum"] = torch.linalg.vector_norm(iterate - self._optimum)
        if self._truth is not None:
            measured["distance_to_truth"] = torch.linalg.vector_norm(iterate - self._truth)
        measured["support_size"] = found.sum().to(DTYPE)
        if self._truth_support:
            measured["support_f1"] = self._support_f1(found)
        measured["exact_zeros"] = (iterate == 0.0).sum().to(DTYPE)
        return measured

    def _support_f1(self, found: Tensor) -> Tensor:
        """F1 of the support at the tolerance against the planted one; 0 if nothing hit."""

        hits = (found & self._support_mask).sum().to(DTYPE)
        precision = hits / found.sum().to(DTYPE)
        recall = hits / len(self._truth_support)
        f1 = 2.0 * precision * recall / (precision + recall)
        return torch.where(hits > 0, f1, torch.zeros_like(f1))

    def compute_metrics(self, outputs: Sequence[Any]) -> dict[str, float]:
        """Fold eval-step outputs into the numbers this task reports.

        ``loss`` is the example-weighted mean of the composite objective over
        what was evaluated -- `F(x)` on the central pass, and not the gap, since
        `F* != 0`. ``distance_to_optimum`` and ``distance_to_truth`` are
        `||x - x*||` (where the problem has a certified `x*`) and
        `||x - x_true||`, whose floor is `||x* - x_true||`. The support columns
        are measured at ``model.support_tolerance``, and ``exact_zeros`` counts
        coordinates that are bit-for-bit 0.0: `d` at initialisation and 0 from
        round 1 on every shipped arm, since none applies a proximal operator.
        """

        records = [record for record in outputs if isinstance(record, Mapping)]
        names = self._names
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

    def evaluate_model(self, model: LogisticModel, data: Any) -> dict[str, float]:
        """Measure the global model on every client's rows at once.

        Returned keys arrive as ``central_test_<name>``. ``optimality_gap`` is
        measured here and only here: `F(x)` is the same number for every
        client and batch.
        """

        outputs = [self.eval_step(model, batch) for batch in self.build_dataloader(data, None)]
        measured = self.compute_metrics(outputs)
        if self._optimal_objective is not None:
            measured["optimality_gap"] = (
                self.pooled_objective(model.iterate) - self._optimal_objective
            )
        return {name: measured[name] for name in self._central_names}

    # -- narrowing a split to given positions --------------------------------

    def count_examples(self, data: Any) -> int:
        """The rows `data` holds, which are what `select_examples` indexes."""

        return len(_rows_of(data)[1])

    def select_examples(self, data: Any, indices: Tensor) -> dict[str, Tensor]:
        """The rows of `data` at `indices`, in that order, as a shard's ``x`` and ``y``."""

        features, labels = _rows_of(data)
        return {"x": features[indices], "y": labels[indices]}

    # -- this task's own batch splitter -------------------------------------

    def _move_batch(self, batch: Any) -> tuple[Tensor, Tensor]:
        """Split a batch into (features, labels), both on the device."""

        features, labels = batch
        return features.to(self.device), labels.to(self.device)


def _composite_loss(model: LogisticModel, outputs: Tensor, labels: Tensor) -> Tensor:
    """The batch's mean loss plus the penalty at the model's iterate."""

    return mean_loss(outputs, labels, model.loss_form) + model.penalty()


def _rows_of(data: Any) -> tuple[Tensor, Tensor]:
    """The feature rows and labels in one split of a shard."""

    if isinstance(data, Mapping):
        features = data.get("x", data.get("X", data.get("features")))
        labels = data.get("y", data.get("Y", data.get("targets")))
        if isinstance(features, Tensor) and isinstance(labels, Tensor):
            return features.to(DTYPE), labels.to(DTYPE)
    raise ValueError("fed_logistic_l1 data must be a shard mapping carrying 'x' and 'y' tensors")


def _pooled_rows(spec: ProblemSpec, metadata: Mapping[str, Any]) -> tuple[Tensor, Tensor]:
    """Every row of the dataset, in the order the global shard stacks them.

    Read from the global shard beside the manifest when there is one -- it is
    the dealt data itself -- and rebuilt from the dials otherwise.
    """

    manifest_path = str(metadata.get("manifest_path") or "")
    global_test = str(metadata.get("global_test") or "")
    if manifest_path and global_test:
        path = Path(manifest_path).parent / global_test
        if path.is_file():
            return _rows_of(load_client_shard(path))
    stacked = torch.cat(spec.client_indices())
    return spec.design()[stacked], spec.labels()[stacked]


def _check_model_against_reference(model_config: Mapping[str, Any], spec: ProblemSpec) -> None:
    """Refuse a model block sized for other data than the corpus."""

    dim = int(model_config.get("input_dim", 0))
    if dim != spec.dim:
        raise ValueError(
            f"model config describes d={dim}, but the corpus is d={spec.dim} (manifest "
            "reference): the model is sized for other data."
        )


def _reported(spec: ProblemSpec, entry: Mapping[str, Any]) -> ReportedMetrics:
    """What a run on ``spec``'s corpus and problem reports, ``entry`` its optimum or empty.

    The distance to `x*` where the problem is certified, the distance to
    `x_true` where the corpus is planted, the support's F1 where there is a
    support to score against -- the planted one, or `x*`'s -- and the gap on
    the central pass alone, which measures `F` over every row.
    """

    certified = bool(entry)
    planted = spec.source is None
    support = spec.truth_support() if planted else set(entry.get("optimum_support", ()))
    client = (
        "loss",
        *(("distance_to_optimum",) if certified else ()),
        *(("distance_to_truth",) if planted else ()),
        "support_size",
        *(("support_f1",) if support else ()),
        "exact_zeros",
    )
    central = ("loss", *(("optimality_gap",) if certified else ()), *client[1:])
    return ReportedMetrics(client=client, central=central)


def reported_metrics(config: Any) -> ReportedMetrics | None:
    """Which of ``METRICS`` a run of ``config`` reports (``registry.tasks.register(reported=)``).

    Read from the corpus the manifest describes and the problem the model block
    poses, as the task decides it when built; None -- every declared name --
    where the data has not been generated, or the problem has no optimum this
    run could use, which the task then refuses when it is built.
    """

    from fedbrew.core import inferred

    manifest = inferred.read_manifest(config.data.path)
    reference = dict((manifest or {}).get("reference") or {})
    if "problem" not in reference or "corpus_digest" not in reference:
        return None
    model_config = {**config.model.extra, "input_dim": config.model.input_dim}
    try:
        spec = _spec_from_reference(reference, model_config)
        entry: Mapping[str, Any] = {}
        if spec.certified:
            table = Path(str(model_config.get("optima") or OPTIMA_TABLE))
            entry = find_optimum(read_optima(table), str(reference["corpus_digest"]), spec)
    except (KeyError, TypeError, ValueError):
        return None
    return _reported(spec, entry)


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

TASK_NAME = "fed_logistic_l1"
MODEL_NAME = "logistic_vector"


def register() -> None:
    """Register the generator, the task and the model.

    Called once by ``fedbrew.core.extensions``, for the entry a config names
    under ``experiment.extensions`` or ``dataset.extensions``. Nothing is
    registered at import, so this module can be imported for
    :class:`ProblemSpec` alone.
    """

    from fedbrew.core import registry

    registry.generators.register(
        DATASET_NAME,
        generate_fed_logistic_l1_from_config,
        sections={"problem": PROBLEM_KEYS, "source": SOURCE_KEYS},
    )
    registry.tasks.register(
        TASK_NAME,
        lambda **kwargs: FedLogisticL1Task(**kwargs),
        metrics=FedLogisticL1Task.METRICS,
        glosses=FedLogisticL1Task.METRIC_GLOSSES,
        grad_norm=FedLogisticL1Task.GRAD_NORM_GLOSS,
        reported=reported_metrics,
    )
    registry.models.register(MODEL_NAME, build_logistic_vector, task=TASK_NAME)


def _self_check() -> None:
    """Assert the claims this module's docstring makes, on a small instance.

    d = 8, 4 clients of 16 rows: cheap at import, and every mechanism. If the
    label base collided with a feature column's, `x*` would be corrupted. If
    the federated decomposition were not exact, every gap would be measured
    against the wrong objective. If the partition dropped or duplicated a row,
    `F` would not be the pooled objective. If the certificate were not checked
    on the full vector, a polish off the support would look clean. If autograd
    disagreed with :func:`gradient`, the run would descend something else. If
    the spec did not survive the manifest, the task would rebuild a different
    corpus than the generator wrote. And if a certified entry's digest were not
    the generator's, no run could find its `F*`.
    """

    spec = ProblemSpec(num_clients=4, dim=8, rows_per_client=16, partition_block=4)
    for check in (_check_data, _check_decomposition, _check_solve, _check_gradients):
        failure = check(spec)
        if failure is not None:
            raise AssertionError(failure)


def _check_data(spec: ProblemSpec) -> str | None:
    """The rows are finite, the labels +/-1, and the deal a partition into equal shares."""

    features, labels = spec.design(), spec.labels()
    if not bool(torch.isfinite(features).all()):
        return "the design holds a non-finite entry"
    if set(labels.tolist()) - {-1.0, 1.0}:
        return "labels must be exactly -1 or +1"
    if label_base(spec.dim) in primes(spec.dim):
        return "the label base is one the design already uses"
    partition = spec.client_indices()
    if sorted(torch.cat(partition).tolist()) != list(range(spec.rows)):
        return "the client partition is not a partition of the rows"
    if {len(index) for index in partition} != {spec.rows_per_client}:
        return "clients do not hold equal shares"
    return None


def _check_decomposition(spec: ProblemSpec) -> str | None:
    """Both aggregation weightings of the client objectives are the pooled `F`."""

    partition = spec.client_indices()
    probe = torch.linspace(-0.7, 0.9, spec.dim, dtype=DTYPE)
    per_client = spec.client_objective_at(probe)
    uniform = sum(per_client) / len(per_client)
    weighted = (
        sum(value * len(index) for value, index in zip(per_client, partition, strict=True))
        / spec.rows
    )
    pooled = spec.objective_at(probe)
    if abs(uniform - pooled) > 1e-14 or abs(weighted - pooled) > 1e-14:
        return "the client objectives do not average to the pooled objective"
    return None


def _check_solve(spec: ProblemSpec) -> str | None:
    """The reference solve is certified on the full vector, and survives the manifest."""

    features, labels = spec.design(), spec.labels()
    probe = torch.linspace(-0.7, 0.9, spec.dim, dtype=DTYPE)
    optimum, residual = solve_reference(features, labels, spec.penalty_strength, iterations=800)
    if residual > CERTIFICATE:
        return f"the reference solve is not certified: KKT residual {residual}"
    if kkt_residual(optimum + 0.1, features, labels, spec.penalty_strength) <= residual:
        return "the KKT residual does not increase away from the optimum"
    if spec.objective_at(optimum) > spec.objective_at(probe):
        return "the reference solve is not the better of two points"
    reference = corpus_reference(spec)
    model_config = {"penalty_strength": spec.penalty_strength, "loss": "logistic", "penalty": "l1"}
    if _spec_from_reference(reference, model_config) != spec:
        return "the spec does not survive the round trip through the manifest"
    _, residual = solve_l2sq(features, labels, spec.penalty_strength)
    if residual > CERTIFICATE:
        return f"the squared-L2 Newton solve is not certified: residual {residual}"
    entry = certify(spec)
    if entry["digest"] != reference["corpus_digest"]:
        return "the table's digest is not the one the generator records"
    stored = _x_star_of(entry, spec.dim)
    stacked = torch.cat(spec.client_indices())
    at_stored = objective(stored, features[stacked], labels[stacked], spec.penalty_strength)
    if entry["f_star"] != at_stored:
        return "the stored f_star is not F at the stored x_star"
    return None


def _check_gradients(spec: ProblemSpec) -> str | None:
    """Autograd on the composite loss is the analytic gradient, for every loss and penalty."""

    features, labels = spec.design(), spec.labels()
    for loss in LOSSES:
        for penalty in PENALTIES:
            model = LogisticModel(
                dim=spec.dim,
                penalty_strength=spec.penalty_strength,
                x_init=0.3,
                loss=loss,
                penalty=penalty,
            )
            _composite_loss(model, model(features), labels).backward()
            measured = model.x.grad
            if measured is None:
                return "the backward pass left no gradient on the iterate"
            expected = gradient(
                model.iterate, features, labels, spec.penalty_strength, loss, penalty
            )
            if float((measured - expected).abs().max()) > 1e-15:
                return f"autograd disagrees with the analytic {loss}+{penalty}"
    return None


_self_check()
