# examples/nonconvex-simplex — a non-convex quadratic over the simplex

*One of the problems in [`examples/`](../README.md); that index says what each one is for.*

**No shipped algorithm can solve this one, every arm diverges, and all eight
runs report `status: completed`.** The losses at round 150 run from −2.4e+02 to
−2.7e+117 and no divergence detector fires. That is the example.

```
minimise    F(x) = mean_i −½ xᵀA_i x = −½ xᵀAx      subject to    x ∈ Δ
```

`A` is a graph's adjacency matrix and each client holds a noisy view `A_i`; the
perturbations are symmetric, zero-diagonal and laid out in exact ± pairs, so
they sum to zero and the mean is the true graph exactly. No client sees the
graph; only the average is it. `A` is symmetric with zero trace and is not the
zero matrix, so it has eigenvalues of both signs and `F` is genuinely
non-convex — and the constrained value is known in closed form by the
**Motzkin-Straus theorem** (Canad. J. Math. 1965):

```
max_{x ∈ Δ} xᵀAx = 1 − 1/ω(G)        F* = −½(1 − 1/ω)        x* = uniform on a maximum clique
```

The graph is a `K₅` disjoint from a star `K₁,₂₅` on 31 vertices, so `ω = 5`,
`F* = −0.4`, and `x*` is `1/5` on the clique's five vertices. The star holds a
second set of local optima, at `−0.25`: any `x` with half its mass on the hub
and the other half spread over the leaves is worth `−0.25`, and no nearby
feasible point does better. That set is one connected face containing the
star's 25 edges, not 25 isolated optima, and its value is in closed form rather
than asserted.

## Why a star, and not a second clique

Because the leading eigenvector must not point at the answer. `K₅`'s adjacency
has spectral radius 4; `K₁,₂₅`'s is `√25 = 5`. **The clique number is the K₅'s
and the spectral radius is the star's.** A method that follows the top
eigenvector — which is what unconstrained gradient ascent on `xᵀAx` does — goes
to the wrong component of the graph. Two disjoint cliques would put the spectral
radius and the clique number on the same vertices, and every failure below would
accidentally look like a success. `ProblemSpec.__post_init__` refuses a
`star_leaves` that does not exceed `(clique_size − 1)²`, so the decoy cannot
stop being one by accident.

## Running it

Generate the data, then run an arm. Both are the ordinary commands; nothing
here is specific to this example except the two config paths.

```bash
fedbrew generate --config data/configs/examples/nonconvex-simplex.yaml
fedbrew inspect-data data/generated/examples/nonconvex-simplex/manifest.json

fedbrew run --config configs/examples/nonconvex-simplex/fedavg.yaml
fedbrew run --config configs/examples/nonconvex-simplex/scaffold.yaml --validate-only
```

The problem is defined outside the package, in [`problem.py`](problem.py), and
reaches the CLI because each config names it: `dataset.extensions` in the
generator config, `experiment.extensions` in every arm config. Chapter 12 is
the general form and [`drift-quad`](../drift-quad/) is its worked case.

`run.py` is a convenience over those commands and nothing more — it runs the
eight arm configs in order through the `fedbrew` CLI, then builds the
comparison table by reading `outputs/`:

```bash
python examples/nonconvex-simplex/run.py               # 8 arms, CPU
python examples/nonconvex-simplex/run.py --arm fedavgm
python examples/nonconvex-simplex/run.py --table-only  # re-table what is on disk
```

Each arm writes `outputs/examples/nonconvex-simplex/<arm>/` with the usual four
artifacts, and each one starts a fresh interpreter that imports torch.

**There is no control config, and the reason is a retraction.** This section
used to offer `--all --star-leaves 4` as "the decoy removed". That command
never worked: `ProblemSpec.__post_init__` refuses any `star_leaves` at or below
`(clique_size − 1)² = 16`, because a star that does not out-rank the clique
spectrally is not a decoy and the example would stop testing anything. So the
documented control raised `ValueError` rather than running, and it did so from
the first line of `run_algorithm`, before any round. Removing the decoy and
keeping the guard are contradictory; the guard is the one worth keeping, and a
graph without a decoy is a different example rather than a control on this one.

## Which shipped algorithms can solve it

None, for the reason [`simplex-lsq`](../simplex-lsq/) gives — there is no
projection, no linear minimisation oracle and no mirror map anywhere in fedbrew,
and no hook one could attach to — and then one more, which makes this the louder
of the two failures:

**The unconstrained problem is unbounded below.** `A` has a positive eigenvalue,
so `−½xᵀAx → −∞` along it and there is no minimum to converge to. In
`simplex-lsq` the arms converge to a well-defined point that happens to be
infeasible. Here there is no point to converge to at all, and every arm runs off
to infinity at `(1 + ηλ_max)` per local step.

So a run demonstrates three things.

1. **Every arm diverges**, at a rate the config sets and nothing else bounds.
2. **Nothing stops it, and nothing says so** — see the next section.
3. **Projecting afterwards makes it worse than not running.** `simplex-lsq` ends
   with `Π_Δ` of its iterate landing exactly on `x*`, which is a property of that
   problem's geometry rather than a method. Here the iterate diverges along the
   star's eigenvector, whose largest coordinate is the hub, and the projection of
   a large vector concentrates on its largest coordinate — so `Π_Δ(x)` is the
   single hub vertex, which spans no edge and scores `F = 0`. The projected gap
   ends at **+0.4**: worse than the barycentre the run started from, and the
   worst value any feasible point can have.

## Nothing stops it

`divergence` cannot see this run, and the reason is worth reading off the code
rather than inferring:

| Detector | Why it does not fire |
| --- | --- |
| `blowup_factor` | `divergence._check_blowup` anchors on `self._first_value`, which `_observe` sets from **the first strictly positive observation** of the monitored metric. `fit_loss` here is negative from round 1, so the anchor is never set and the detector never arms. The comment beside it reasons about a metric that starts at 0; a metric that is never positive falls in the same hole. |
| `blowup_absolute` | An **upper** ceiling: `value > ceiling`. The value is heading to −∞. |
| `patience` | Not enabled by default, and it would not help: the monitored metric improves monotonically and spectacularly, every single round. |
| `non_finite` | The only one that can fire, and only once the arithmetic overflows. At the shipped rate FedAvg ends round 150 at −8.9e+85; run on with `--rounds 600`, it overflows at round 531. |

All eight arms therefore finish, and `run.json` records `status: completed`,
`final_round: 150`, `termination: null`. Nothing in the artifacts distinguishes
this from a healthy run except the values themselves.

This is not an argument that `divergence` is broken. It is watching for a loss
that goes *up*, which is what a divergence looks like on every objective in the
repository, all of which are non-negative losses. It is an argument that a
maximisation cast as a minimisation — which is what any objective of the form
`−(something good)` is — leaves the guard with nothing to anchor on, and that
the guard says nothing about it either way.

### A finite loss is not a representable variance

At `client.learning_rate: 0.1` rather than the shipped `0.05` this example does
not finish, and not because anything detected the divergence. It raises

```
OverflowError: integer division result too large for a float
```

from `loop._client_distribution_statistics`, which computes
`statistics.pstdev(values)` over the per-client `test_loss`. Python's
`statistics` works in exact rationals: the values are around 1e157 and finite,
their variance is around 1e314 and is not a float, and `_convert` raises on the
way back. The line reads `statistics.pstdev(values) if finite else math.nan` —
so it checks that the *values* are finite, which they are, and overflows on a
statistic of them anyway.

Like the cosine-to-zero crash this example's sibling turned up
(`FINDINGS.csv` `POST-F01`), it lands in the artifact path
after the compute is spent: 148 of 150 rounds on disk, `run.json` reading
`status: running`, `error: null`.

**Fixed, and recorded as `POST-F02`.** `loop._overflow_safe` guards the
statistic instead of the data and answers the NaN the caller already reserves
for a distribution nothing can summarise. `--all` at `0.1` now completes every
arm at 150 rounds; `test_loss_std` reads `nan` on 2 rounds of `fedavg`,
`fedprox` and `scaffold` and on 21 of `fedavgm`, and `test_loss_avg` — well
inside float range — on none. The defect was version-dependent besides: CPython
3.11 rewrote `pstdev` to take the square root before converting, so this same
run died on 3.10 and completed on 3.12.

The shipped rate stays `0.05`. It was picked as the largest at which the run
completed, and re-picking it now that `0.1` also completes would be choosing a
setting after seeing what it produces, which is the thing this repository is
about. The table below is the `0.05` run.

## What the shipped strategies do on it

150 rounds, 31 vertices, 8 clients, all participating, 3 local steps per round,
`heterogeneity` 0.1, seed 42. `F* = −0.4`; the star's local optima are worth
`−0.25`; the barycentre the run starts from is worth `−0.03642`. All of that
travels with the data: the generator writes `F*`, `x*`, both spectral radii and
the maximal-clique count into the manifest under `reference`, so `run.json`
records that the leading eigenvector points away from the answer.

| arm | loss @ 150 | gap @ 150 | **feasible gap @ 150** | `‖x‖` @ 150 | Σx | mass on clique | run status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fedavg | −8.95e+85 | −8.95e+85 | **+0.4000** | 5.98e+42 | 2.53e+43 | 0.0000 | completed 150/150 |
| fedprox | −8.36e+85 | −8.36e+85 | **+0.4000** | 5.78e+42 | 2.45e+43 | 0.0000 | completed 150/150 |
| fedavgm | −2.73e+117 | −2.73e+117 | **+0.4000** | 3.31e+58 | 1.40e+59 | 0.0000 | completed 150/150 |
| fedadam | −2.81e+04 | −2.81e+04 | +0.3636 | 1.58e+02 | 8.78e+02 | 0.1613 | completed 150/150 |
| fedyogi | −2.37e+04 | −2.37e+04 | +0.3636 | 1.45e+02 | 8.06e+02 | 0.1613 | completed 150/150 |
| fedadagrad | −2.41e+02 | −2.41e+02 | +0.3636 | 1.46e+01 | 8.14e+01 | 0.1613 | completed 150/150 |
| scaffold | −8.19e+85 | −8.19e+85 | **+0.4000** | 5.72e+42 | 2.43e+43 | 0.0000 | completed 150/150 |
| fedlalr | −1.04e+04 | −1.04e+04 | +0.3386 | 9.53e+01 | 5.31e+02 | 0.2439 | completed 150/150 |
| **`x*`** *(not an arm)* | **−0.4000** | **0** | **0** | 0.4472 | **1.0000** | **1.0000** | — |

Two readings, and neither is a ranking.

**The per-coordinate arms diverge more slowly, and land on the starting point.**
`fedadam`, `fedyogi` and `fedadagrad` normalise each coordinate by its own
gradient scale, which caps the step and leaves them at `‖x‖ ≈ 1e2` instead of
`1e42`. Their projected point has `mass on clique` of exactly 0.1613, which is
`5/31` — the barycentre. Their feasible gap of 0.3636 is exactly `F(1/d) − F*`:
after 150 rounds their projection is worth precisely what the initialisation was
worth. Diverging less is not the same as getting anywhere.

**The best feasible point any arm ever visits is at round 2.** Projecting at
every round and taking the best gives:

| arm | best feasible gap | at round | mass on clique there |
| --- | --- | --- | --- |
| fedavg / fedprox / scaffold | 0.2338 | 2 | 0.2005 |
| fedavgm | 0.2311 | 2 | 0.2305 |
| fedlalr | 0.3370 | 15 | 0.2585 |
| fedadam / fedyogi / fedadagrad | 0.3636 | 150 | 0.1613 |

The best any of them manages, at any round, with a projection applied for free,
is 0.23 short of `F*` on a problem whose whole range is 0.4 — and it happens in
round 2, before the divergence has taken over, after which every arm gets
monotonically worse forever. There is no stopping rule in the repository that
would find round 2, and no metric in the round record that is falling at the
time.

## There is nothing here to tune

Every scalar this problem offers is optimised by diverging faster:

* `optimality_gap` is unbounded below, so a tuner scored on it prefers the
  largest learning rate that does not overflow. That is not a search over
  algorithms, it is a search over how far past the answer the run can get.
* `feasible_gap` is minimised by *not* diverging, so a tuner scored on it prefers
  a learning rate of zero, which reports the barycentre.
* `mass_on_clique` has the same defect and the same fixed point.

So there is no tuning grid anywhere for this example, and the learning rates in
the arm configs are one value per family chosen as the largest at which the run
finishes. Each config says so at the line that sets it. Reporting a tuned winner
would be reporting the tuner's preference for a failure mode.

## Where the abstraction fit, and where it did not

[`simplex-lsq`](../simplex-lsq/)'s README covers what a constrained problem meets
that the unconstrained ones do not — no hook for a projection, and feasibility
carried as a metric rather than as a validity condition. Two things are this
example's own.

### The guard's premise is that losses are positive

Stated above, and it is the general form of the `POST-F01` finding rather than a
second instance of it: `divergence`, `metrics`, and the terminal round block all
assume the monitored quantity is a non-negative loss that should go down and
whose going *up* is the alarm. Every objective in the repository is one. An
objective that is negative, or unbounded below, or being maximised, passes
through all of them without contradiction — and `blowup_factor`, the detector
whose whole job is scale-free divergence detection, silently does not arm.

### `d` is derived from the problem, not configured

`drift-quad` and `fed-lasso` state the dimension in `model.input_dim` and
cross-check it against the manifest. Here `d = clique_size + 1 + star_leaves`
is a property of the graph, so the generator config states the graph and each
arm config states the `d` that follows from it — 31 — with the task
cross-checking the two. That is the right way round: a dimension that can
disagree with the problem is a dimension that will. It does mean `input_dim` in
an arm config is a derived number written by hand, and a reader who edits it
there rather than in the generator config gets the task's error naming both
values rather than a silently different graph.

This is the one place the migration left a number in two files. The
alternative — deriving `input_dim` inside the task and refusing it in the model
block — would make this model unlike every other model in the repository, whose
dimension is a config key. The cross-check is the cheaper half of that trade.

## Files

| File | What it is |
| --- | --- |
| `problem.py` | the whole extension: the objective, its analytic gradient, the graph and its Motzkin-Straus value, `project_simplex` — used only to measure — the closed-form `x*`, the decoy's spectral radius and the maximal-clique count, the generator that writes the shards and the manifest, the model, the task adapter, and `register()` |
| `run.py` | a convenience over the CLI: runs the eight arm configs, then tables `outputs/`. Registers nothing, composes no config |
| `README.md` | this file |

And outside the example, where every other dataset and arm keeps theirs:

| Path | What it is |
| --- | --- |
| `data/configs/examples/nonconvex-simplex.yaml` | the generator config: the graph, and the extension that reads it |
| `configs/examples/nonconvex-simplex/*.yaml` | one run config per arm. Not tuned, and each says why |
| `data/generated/examples/nonconvex-simplex/` | the shards, `clients.jsonl` and the manifest — including `reference`, the Motzkin-Straus value this run is scored against |
| `outputs/examples/nonconvex-simplex/<arm>/` | what a run wrote |
