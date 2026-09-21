# examples/simplex-lsq — least squares over the probability simplex

*One of the problems in [`examples/`](../README.md); that index says what each one is for.*

**No shipped algorithm can solve this one, and every arm ends with a negative
optimality gap.** Read as a column of numbers, all eight beat the optimum. They
beat it by leaving the feasible set, and a table that reports the objective
without reporting the constraint says the opposite of what happened. That is the
whole example.

```
minimise    F(x) = mean_i (1/2m)‖Hx − y_i‖²      subject to    x ∈ Δ = {x ≥ 0, Σx = 1}
```

`H` is the `d × d` Sylvester-Hadamard matrix with `m = d` rows, so `HᵀH = mI`
exactly, the smooth part reduces to `½‖x − θ_i‖²`, and with the per-client
offsets in exact ± pairs

```
F(x) = ½‖x − θ̄‖² + ζ²/2        θ̄ = x_true + displacement
```

The unconstrained minimiser is `θ̄`, and the `infeasibility` dial puts it
**outside the simplex**: at the default it has eight coordinates at −0.09375 and
sums to 1.5. Because the design is orthonormal the constrained minimiser is then
exactly a Euclidean projection, which `project_simplex` computes by sorting —
finite and exact, not an iteration:

```
x* = Π_Δ(θ̄)        F* = 0.21281        F(θ̄) − F* = −0.13281
```

At the default dials `x*` is exactly `x_true`, the planted point, so this
example has one ground truth where [`fed-lasso`](../fed-lasso/) has two. That is
by construction — the displacement is constant on the even coordinates and the
planted support is even, so the projection's threshold lands on exactly that
constant — and `problem._self_check` asserts it rather than leaving it to this
paragraph.

## Running it

Generate the data, then run an arm. Both are the ordinary commands; nothing
here is specific to this example except the two config paths.

```bash
fedbrew generate --config data/configs/examples/simplex-lsq.yaml
fedbrew inspect-data data/generated/examples/simplex-lsq/manifest.json

fedbrew run --config configs/examples/simplex-lsq/fedavg.yaml
fedbrew run --config configs/examples/simplex-lsq/scaffold.yaml --validate-only
```

The problem is defined outside the package, in [`problem.py`](problem.py), and
reaches the CLI because each config names it: `dataset.extensions` in the
generator config, `experiment.extensions` in every arm config. Chapter 12 is
the general form and [`drift-quad`](../drift-quad/) is its worked case.

The feasible control has its own generator config, because `infeasibility` is a
property of the data and of what the data is scored against:

```bash
fedbrew generate --config data/configs/examples/simplex-lsq-feasible.yaml
fedbrew run --config configs/examples/simplex-lsq-feasible/fedavg.yaml
```

`run.py` is a convenience over those commands and nothing more — it runs a
directory of arm configs in order through the `fedbrew` CLI, then builds the
comparison table by reading `outputs/`:

```bash
python examples/simplex-lsq/run.py                       # 8 arms, CPU
python examples/simplex-lsq/run.py --setting simplex-lsq-feasible
python examples/simplex-lsq/run.py --table-only          # re-table what is on disk
```

Each arm writes `outputs/examples/simplex-lsq/<arm>/` with the usual four
artifacts, and each one starts a fresh interpreter that imports torch — the
honest cost of every arm being a real `fedbrew run`.

## Which shipped algorithms can solve it

None, and the reason is a whole category rather than a line of code. Minimising
over a constraint set needs one of a projection (projected SGD), a linear
minimisation oracle (Frank-Wolfe), or a mirror map (entropic mirror descent /
exponentiated gradient, which is the natural one for a simplex). **None of the
three exists anywhere in the tree.** There is no projection step anywhere under
`fedbrew/clients/` or `fedbrew/servers/`, `TaskAdapter.train_step` is handed an
optimizer and returns a loss, and a feasible-set projection is neither of those
things — there is no hook it could be attached to without adding one.

So every arm minimises `F` over all of `R^d`. What follows is not slow
convergence and not divergence: the objective is strongly convex, every arm
converges cleanly and quickly, and it converges to `θ̄`, which is not a
solution. The failure is that the answer is in the wrong set.

A run therefore demonstrates three things, and the third is the one worth
having:

1. Unconstrained descent on a constrained problem converges, and converges
   fast — `fedavg`'s gap is already negative at round 1 and at the floor by
   round 5.
2. The iterate leaves the feasible set immediately and never returns.
   `constraint_violation` is exactly 0 at initialisation, because the
   barycentre `1/d` sums to exactly 1.0 at a power-of-two `d`, and it is 0.5154
   from round 5 on.
3. **The objective column inverts.** `optimality_gap` starts at +0.2656, crosses
   zero, and settles at −0.1328 — which is `−½‖x* − θ̄‖²`, known in closed form
   before the run. Any ranking taken from that column ranks arms by how
   thoroughly they broke the constraint.

## What the shipped strategies do on it

150 rounds, d = 16, 8 clients, all participating, m = 16 rows each, batch 4, 3
local iterations of one pass each, `infeasibility` 0.5, ζ = 0.4, seed 42. `F* = 0.21281`, of which
`ζ²/2 = 0.08` is the heterogeneity residual no `x` removes.
`F(x₀) − F* = +0.2656` at the barycentre. All of that travels with the data:
the generator writes `F*`, `x*`, `negative_gap_floor` and both readings of the
unconstrained optimum into the manifest under `reference`, so `run.json`
records the floor the gap column was always going to reach.

| arm | gap @ 150 | first round gap < 0 | feasible gap @ 150 | violation | Σx | min x_j | negative mass | `‖x − x*‖` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fedavg | **−0.1328** | 1 | −2.8e-17 | 0.5154 | 1.5000 | −0.0938 | 0.7500 | 0.5154 |
| fedprox | **−0.1328** | 4 | 6.2e-13 | 0.5154 | 1.5000 | −0.0938 | 0.7500 | 0.5154 |
| fedavgm | **−0.1328** | 4 | 9.8e-11 | 0.5154 | 1.5000 | −0.0937 | 0.7500 | 0.5154 |
| fedadam | **−0.1328** | 2 | 6.8e-09 | 0.5153 | 1.5001 | −0.0937 | 0.7497 | 0.5153 |
| fedyogi | **−0.1328** | 6 | 1.6e-08 | 0.5154 | 1.5000 | −0.0938 | 0.7500 | 0.5154 |
| fedadagrad | **−0.1328** | 4 | 3.4e-09 | 0.5155 | 1.4997 | −0.0938 | 0.7503 | 0.5155 |
| scaffold | **−0.1328** | 4 | 5.7e-09 | 0.5154 | 1.5000 | −0.0940 | 0.7500 | 0.5154 |
| fedlalr | **−0.1328** | 4 | 7.2e-08 | 0.5154 | 1.5000 | −0.0941 | 0.7500 | 0.5154 |
| **`x*`** *(not an arm)* | **0** | — | 0 | **0** | **1** | **0** | **0** | **0** |

Every arm lands on the same point to four decimal places. There is nothing to
rank: they all solve the same unconstrained problem, they all solve it, and the
answer they agree on has three quarters of a unit of probability mass sitting
below zero. The one column that separates them at all — `feasible gap` — spans
1e-17 to 7e-08, which is the difference between converging to `θ̄` to eleven
digits and to eight.

## The projected column is a property of the problem, not a fix

`feasible_gap` projects the iterate onto the simplex and then measures. It
reaches machine zero on every arm, which looks like a recipe: *run anything,
project at the end.* It is not one, and the reason it works here is worth being
explicit about, because the number is otherwise misleading.

Every arm converges to `θ̄`. The projected value is therefore `F(Π_Δ(θ̄)) − F*`,
and `Π_Δ(θ̄)` **is** `x*` — but only because `HᵀH = mI`. With an identity Gram
the constrained problem is literally "project the unconstrained solution", so
projecting at the end is exact. Change the design so the Gram is not a multiple
of the identity and it stops being exact immediately: the constrained optimum is
no longer the projection of the unconstrained one, and post-hoc projection
becomes an arbitrary feasible point with no guarantee at all.

So this example cannot be used to argue that projecting at the end is a
substitute for a projected method. It is the one geometry where it happens to
be, and the column is here so a reader knows which of the two they are looking
at. Below about 1e-16 the column's sign is arithmetic — `fedavg` reports
−2.8e-17 — not an arm beating `F*` twice over.

## The control: with the optimum inside the simplex, every arm is right

`simplex-lsq-feasible` leaves `θ̄` at the planted point, which is already
feasible, so the constraint binds on nothing and the constrained and
unconstrained problems coincide. Three arms, one per family, because "every arm
is right" is a claim about more than one:

| arm | gap @ 150, `infeasibility 0` |
| --- | --- |
| fedavg | **0.0** (exactly) |
| fedadam | 6.9e-09 |
| scaffold | 1.4e-07 |

FedAvg reaches the optimum exactly. Nothing about the design, the heterogeneity,
the participation or the loop is responsible for the table above; the constraint
is, and it is responsible for all of it.

## How the arms were tuned, and why the grid barely matters

One grid — client learning rate over 0.0125 … 0.4 for the SGD-family arms,
crossed with the server learning rate over 0.01 … 1.0 for the FedOpt arms —
scored on `central_test_feasible_gap` at the final round. Single seed, and the
selection metric is the reported metric.

`feasible_gap` is the least-bad scalar available, and it is not a good one.
There is **no scalar this problem can rank the shipped arms on**:

* `optimality_gap` is minimised by being *more* infeasible. A tuner scored on it
  selects for the defect, and would prefer the arm that leaves the simplex
  fastest.
* `constraint_violation` is minimised by not moving. It is 0 at the barycentre,
  so a tuner scored on it prefers a learning rate of zero.
* `feasible_gap` avoids both, and then bottoms out at machine precision for
  every arm at almost every step size, so the grid separates nothing.

The grid is run and reported anyway, so that no arm's result can be blamed on a
bad step size. It is not a ranking, and the winners differ from their neighbours
by rounding.

## Where the abstraction fit, and where it did not

The three earlier examples' READMEs cover what this one meets again unchanged.
Two things are specific to a constrained problem.

### The problem is data, and the data says what the answer is

**Closed.** The design and the targets used to reach the run through a closure
— `problem.register(spec)` bound a `ProblemSpec` into a dataset factory —
because `factory._build_source_dataset` called a third backend's factory with
no arguments, and `SimplexLSQTask` read `x*` and `F*` off the same closure and
raised if it was built before `register()`. Nothing in a config could state the
problem, and nothing in `run.json` could record it.

They are now *generated*. `generate_simplex_lsq_from_config` writes one shard
per client and a manifest, `fedbrew generate` runs it like any other generator,
and a run reads it through the shipped `manifest_dataset`. The reference
optimum travels with the data under the manifest's `reference` key, and the
task rebuilds its `ProblemSpec` from the dials recorded there — `_self_check`
asserts the round trip is exact. `infeasibility` is a property of the data
rather than a flag, which is why the control is a second generator config.

### There is no hook a projection could attach to

This is the honest statement of what is missing. A proximal step, which
[`fed-lasso`](../fed-lasso/) wants, at least has an obvious home: it composes
with an optimizer, the way `torch_scaffold_client._ScaffoldCorrectingOptimizer`
and `local_update_modes._ClippingOptimizer` already wrap one. A feasible-set
projection is the same shape — apply it after `optimizer.step()` — and so is a
mirror map, in the same place.

What has no home is *where the set is stated*. The feasible set is a property of
the problem, like the reference optimum, and the problem is expressed across a
dataset closure and `model.extra`; a client rule that projected would need to be
told what to project onto, and there is no channel that carries it.
`TaskAdapter` has five abstract methods and none of them is `project`, and
adding one would break the twelve task doubles in `tests/` the way a sixth
abstract method would — the argument `tasks.base.SupportsDatasetEvaluation`
makes for `evaluate_model` being a protocol rather than a method applies
verbatim. A `SupportsProjection` protocol beside it is the shape this would
take, and it does not exist.

### A withdrawn claim about thread count

An earlier version of this section said two `--all` runs of this example
disagreed in the last one to three significant digits of several columns —
`central_test_feasible_gap` at `−8.3e-17` on one run and `−2.8e-17` on the next
— and blamed the thread count, on the reasoning that a threaded reduction adds
its pieces in whatever order the threads finish. Every example was pinned to
`torch_num_threads: 1` on that basis.

**The diagnosis was wrong, and it is retracted.** Re-measured 2026-09-04, on
this example and on `nonconvex-simplex`, at 1, 8 and 64 threads, single-arm and
`--all`: every non-timing column of `round_metrics.csv`, `client_metrics.csv`
and `client_update_metrics.csv` is bit-identical across all of them. The
unpinned runs land on `−2.7755575615628914e-17` every time — the same value the
retracted note recorded for one of its two runs.

It could not have been thread count. Torch runs an intra-op region on a single
thread below a grain size of 32768 elements; these problems are 16- and
31-dimensional, so no reduction here is ever split, at any setting. The pin is
inert by construction, which is the check that should have been run before the
claim was written.

What did move between those two original runs was not established and now
cannot be — most likely two different states of this example during
development, since the metric in question is a cancelling difference whose true
value is `0` and whose observed value is one to three ulps of rounding noise. In
that column "the last one to three significant digits" was *all* of the
significant digits, which is the reading that should have prompted the check.

The pin stays, on the smaller claim it can support: it costs nothing here, and
a config that states its thread count is one fewer thing inherited from the
machine. It is not load-bearing, and no table in these READMEs depends on it.

`THROUGHPUT_ONLY_PERFORMANCE_KEYS` in `config.py` classifies
`torch_num_threads` as throughput-only, and this episode is not a reason to
move it.

### A constraint has no metric route of its own

`compute_metrics` is free-form on the fit and central paths, so the four
feasibility readings arrive as `fit_*` and `central_test_*` without any
widening — that part fits. What does not is that nothing in the loop knows any
of them are constraints. `divergence` watches `fit_loss` and would not fire on a
run whose loss is falling beautifully into an infeasible region; the terminal
round block classifies `central_test_constraint_violation` by its prefix and
prints it beside the losses with no indication that a non-zero value invalidates
every other number in the block. The feasibility columns are carried, and they
are carried as metrics rather than as a validity condition, so it remains
entirely up to the reader to notice.

## Files

| File | What it is |
| --- | --- |
| `problem.py` | the whole extension: the objective, its analytic gradient, `project_simplex` — the operator that defines `x*` and that no arm runs — the planted point, the closed-form constrained optimum and the negative floor it implies, the generator that writes the shards and the manifest, the model, the task adapter, and `register()` |
| `run.py` | a convenience over the CLI: runs a directory of arm configs, then tables `outputs/`. Registers nothing, composes no config |
| `README.md` | this file |

And outside the example, where every other dataset and arm keeps theirs:

| Path | What it is |
| --- | --- |
| `data/configs/examples/simplex-lsq.yaml` | the generator config: the dials, and the extension that reads them. `simplex-lsq-feasible.yaml` beside it is the control |
| `configs/examples/simplex-lsq/*.yaml` | one run config per arm, tuned. `configs/examples/simplex-lsq-feasible/` holds the control's three |
| `data/generated/examples/simplex-lsq/` | the shards, `clients.jsonl` and the manifest — including `reference`, the constrained optimum and the negative floor this run is scored against |
| `outputs/examples/simplex-lsq/<arm>/` | what a run wrote |
