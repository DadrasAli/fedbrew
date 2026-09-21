# examples/drift-quad — separable strongly convex quadratics, with two dials

*One of the problems in [`examples/`](../README.md); that index says what each one is for.*

A federated problem with a known answer, a closed-form gradient, and two knobs
that turn independently. It exists to ask one question of the algorithms rather
than of the framework: **when a federated method beats FedAvg, which property
of the problem is it beating, and can you turn that property off and watch the
win disappear?**

Every client optimises a quadratic over `x ∈ R^d` with the *same* diagonal
curvature and its *own* linear offset:

```
f_i(x) = ½ xᵀA x − b_iᵀx        ∇f_i(x) = A x − b_i        A = diag(κ^(j/(d−1)))
```

`a_0 = 1` and `a_{d−1} = κ` exactly, so `μ = 1`, `L = κ`, and the condition
number of the problem *is* the `condition_number` dial. The Hessian is diagonal:
the `d` coordinates are `d` independent scalar quadratics that never mix, and
every claim below can be checked one coordinate at a time. The offsets are laid
out in exact ± pairs of Hadamard rows, so they sum to zero in floating point and

```
F(x) = ½ xᵀA x        ∇F(x) = A x        x* = 0        F* = 0
```

are exact for the *federated* objective, not only for one client. Client `i`'s
own minimiser is `x_i* = A⁻¹b_i`, and every `‖b_i‖` equals the `dissimilarity`
dial, so that dial is the gradient-dissimilarity constant itself:

```
ζ² = (1/n) Σ_i ‖∇f_i(x*) − ∇F(x*)‖² = (1/n) Σ_i ‖b_i‖²
```

The two dials are the two things a federated method can fix, and they are as
separable as the coordinates. **κ sets the rate**: local SGD needs `η < 2/L` to
be stable in the stiffest coordinate, and the softest then contracts by
`(1 − η)^K` per round, so κ divides the progress of every SGD arm. **ζ sets the
floor**: under partial participation the sampled mean offset is not zero, the
round's fixed point is displaced by `A⁻¹b̄_S`, and no client learning rate
removes it. Server preconditioning attacks the first. Control variates attack
the second. Neither attacks the other, and the tables below are that sentence,
measured.

## Running it

Generate the data, then run an arm. Both are the ordinary commands; nothing
here is specific to this example except the two config paths.

```bash
fedbrew generate --config data/configs/examples/drift-quad.yaml
fedbrew inspect-data data/generated/examples/drift-quad/manifest.json

fedbrew run --config configs/examples/drift-quad/fedavg.yaml
fedbrew run --config configs/examples/drift-quad/scaffold.yaml --validate-only
```

The problem is defined outside the package, in
[`problem.py`](problem.py), and reaches the CLI because each config names it:
`dataset.extensions` in the generator config, `experiment.extensions` in every
arm config. Chapter 12 is the general form; this example is its worked case.

The two dials each have their own generator config, because a dial changes the
data rather than the run:

```bash
fedbrew generate --config data/configs/examples/drift-quad-rate.yaml   # ζ = 0
fedbrew generate --config data/configs/examples/drift-quad-floor.yaml  # κ = 1
fedbrew run --config configs/examples/drift-quad-rate/fedadagrad.yaml
```

`run.py` is a convenience over those commands and nothing more — it runs a
directory of arm configs in order through the `fedbrew` CLI, then builds the
comparison table by reading `outputs/`:

```bash
python examples/drift-quad/run.py                        # 8 arms, CPU
python examples/drift-quad/run.py --setting drift-quad-rate
python examples/drift-quad/run.py --table-only           # re-table what is on disk
```

Each arm writes `outputs/examples/<setting>/<arm>/` with the usual four
artifacts, and each one starts a fresh interpreter that imports torch — the
honest cost of every arm being a real `fedbrew run` rather than a loop inside
one process. Runs are bit-reproducible: two sweeps produced identical values in
every non-timing column of every `round_metrics.csv`, on all eight arms
(verified 2026-09-04). The configs also pin `torch_num_threads: 1`, which is
**not** what makes that true — see [`simplex-lsq`](../simplex-lsq/)'s README for
the claim that was retracted.

## Which shipped algorithms can solve it, and what a run demonstrates

All eight. The problem is smooth, unconstrained and strongly convex, `∇f_i` is
exact and cheap, and every shipped client rule is a first-order method that
applies to it unmodified. Nothing here is disqualified, no arm is a stand-in for
an algorithm the repository does not have, and the comparison is real — which
is why this is the example the other four are measured against.

That is not the same as saying every arm reaches `x*`. Under partial
participation exactly one does:

| arm | mechanism it brings | fixes the κ dial | fixes the ζ dial |
| --- | --- | --- | --- |
| `fedavg` | — | no | no |
| `fedprox` | proximal term damps the local step | no | no — on the same floor as FedAvg |
| `fedavgm` | server heavy ball | yes | no |
| `fedadam` / `fedyogi` / `fedadagrad` | per-coordinate server preconditioner | yes | no |
| `fedlalr` | per-coordinate local AMSGrad | yes | no |
| `scaffold` | control variates | no | **yes, exactly** |

SCAFFOLD is exact here for a reason the problem makes visible in one line: with
a shared `A`, `∇f_i(x) − ∇F(x) = b̄ − b_i` is a **constant**, independent of `x`.
So the correction that removes it is a constant too. SCAFFOLD's control
variates estimate that correction from each client's local steps, and once the
local iterates settle on `x*` the estimate is exact: `c − c_i` cancels the whole
heterogeneity term, and the corrected local gradient is `∇F` evaluated at the
local point. Everything else in the table descends toward a floor set by ζ and
then oscillates on it.

So a run demonstrates: a ten-order-of-magnitude separation with a stated cause,
a control that removes the cause and collapses the separation, and — for every
arm except SCAFFOLD — a final-round number that is a floor rather than a result.

## What the shipped strategies do on it

200 rounds, d = 16, 8 clients, 4 sampled per round, 10 local steps, κ = 100,
ζ = 1, seed 42. `F(x_0) − F* = 8.0` exactly, `‖x_0‖ = 1.938`, and client drift
can carry a local iterate at most `max_i‖x_i*‖ = 0.369` from `x*`. Every arm is
the best of the grid in **How the arms were tuned**. `gap` is
`central_test_loss`, which for this problem *is* `F(x) − F*` for the aggregated
iterate.

| arm | gap @ 200 | median gap, 181–200 | best gap | first < 1e-2 | first < 1e-6 | `fit_distance_to_optimum` | `fit_distance_to_client_optimum` | params/round |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fedavg | 5.69e-04 | 7.18e-04 | 2.48e-04 | 103 | — | 2.30e-02 | 3.61e-01 | 16 |
| fedprox | 5.69e-04 | 7.18e-04 | 2.48e-04 | 103 | — | 2.30e-02 | 3.61e-01 | 16 |
| fedavgm | 1.54e-04 | 1.52e-04 | 1.11e-04 | 105 | — | 1.06e-01 | 2.80e-01 | 16 |
| fedadam | 1.78e-03 | 1.19e-03 | 3.90e-04 | 68 | — | 2.74e-02 | 3.56e-01 | 16 |
| fedyogi | 4.59e-04 | 4.56e-04 | 1.84e-04 | 73 | — | 2.18e-02 | 3.54e-01 | 16 |
| fedadagrad | 3.49e-04 | 3.43e-04 | 1.49e-04 | 77 | — | 2.00e-02 | 3.55e-01 | 16 |
| **scaffold** | **8.52e-15** | **1.85e-13** | **7.67e-15** | **26** | **82** | **1.06e-07** | 3.69e-01 | 32 |
| fedlalr | 8.04e-04 | 7.52e-04 | 1.31e-04 | 58 | — | 1.84e-02 | 3.61e-01 | 48 |

**Read only one column difference as a result.** SCAFFOLD's eleven orders are a
separation; nothing else in the table is. Over rounds 181–200 the FedAvg arm's
gap swings between 2.48e-04 and 1.75e-03 — a factor of 7.1 — while the entire
spread from FedAvg to the best non-SCAFFOLD arm is a factor of 3.7. Every arm
but SCAFFOLD is sitting on the same ζ floor, and which of them prints the
smallest number at round 200 is a question about the phase of an oscillation.
The `median gap, 181–200` column is there so that is visible rather than
inferred; `best gap` is there because it is the number a less careful table
would report.

Two columns are worth reading against each other. `fit_distance_to_optimum` is
`‖x − x*‖` for each selected client's **post-local-training** iterate: 0.023 on
FedAvg, 1.06e-07 on SCAFFOLD. That is client drift, measured. The next column is
`‖x − x_i*‖`, the distance from the same iterate to *that client's own*
minimiser, and it sits at 0.36 for every arm including SCAFFOLD — near the drift
radius. So the cartoon of drift, where local training runs away to `x_i*`, is not
what happens at `K = 10`: the local iterates barely move toward their own optima,
and the FedAvg floor is the sampled subset's non-zero mean offset, not
clients solving their own problems. The last column prices the mechanisms:
SCAFFOLD moves two model-shaped states per direction per round and FedLALR
three, against FedAvg's one.

`central_test_loss` and `test_loss_sample_weighted_avg` are two independent
routes to `F(x)` — one batched server pass, one per-client pass pooled over
clients — and they agree to ≤ 1.8e-15 on every arm, which checks that the
per-client weighting really is uniform.

## The two dials, turned separately

Same protocol, same grid, re-tuned per setting, so each column is each arm's
own best and not the base setting's hyperparameters carried somewhere they do
not belong. A dial changes the *data*, so each setting is its own generated
dataset with its own directory of tuned arm configs:

```bash
fedbrew generate --config data/configs/examples/drift-quad-rate.yaml    # ζ = 0
python examples/drift-quad/run.py --setting drift-quad-rate

fedbrew generate --config data/configs/examples/drift-quad-floor.yaml   # κ = 1
python examples/drift-quad/run.py --setting drift-quad-floor
```

**ζ = 0, κ = 100 — the pure conditioning problem.** No heterogeneity, so no
floor: every arm converges linearly and the column is a rate.

| arm | tuned setting | gap @ 200 | first < 1e-6 | first < 1e-12 |
| --- | --- | --- | --- | --- |
| fedavg | η 0.019 | 2.37e-34 | 35 | 71 |
| fedprox | η 0.019, μ 0.01 | 2.55e-34 | 35 | 71 |
| fedavgm | η 0.019, server 1.0, β₁ 0.5 | 3.54e-60 | 22 | 42 |
| fedadam | η 0.008, server 1.0, β₁ 0.5 | 1.13e-56 | 32 | 53 |
| fedyogi | η 0.019, server 0.3, β₁ 0.0 | 1.60e-215 | 8 | 13 |
| **fedadagrad** | η 0.019, server 3.0, β₁ 0.0 | **2.73e-221** | **8** | **13** |
| scaffold | η 0.008 | 1.26e-14 | 84 | 172 |
| fedlalr | η 0.01 | 9.32e-92 | 15 | 28 |

Read the round counts, not the exponents: an arm below 1e-30 has solved the
problem, and the mantissa at round 200 only measures how fast it kept going
afterwards. FedAdagrad and FedYogi with `β₁ = 0` reach 1e-12 in 13 rounds
against FedAvg's 71. The mechanism is visible in the update rule: Adagrad's `v`
accumulates and never decays, so `√v` freezes at each coordinate's own historical
gradient scale and `lr·Δ/√v` is a per-coordinate Newton-like step —
`FedOptServer._fedadagrad_update` is a diagonal preconditioner, and a diagonal
preconditioner is exactly what a diagonal Hessian needs. FedAvgM's server
momentum takes 42 rounds against 71, at `β₁ = 0.5`. FedLALR's per-coordinate
local AMSGrad takes 28.

And **SCAFFOLD is the worst arm here** — 172 rounds, worse than plain FedAvg by
101. Its control variates are not zero when there is nothing to correct: a
sampled client refreshes `c_i` while the others keep stale values, so `c − c_i`
injects an error even with identical clients. That the error is staleness and
nothing else is checkable — running the `drift-quad-rate` arms at
`--participation-rate 1.0`, SCAFFOLD and FedAvg produce the same
`central_test_loss` to the last digit (5.568e-15 at η 0.008), because no
client's control variate is ever stale. The method that
wins the base table by eleven orders loses this one.

**κ = 1, ζ = 1 — the pure heterogeneity problem.** Perfectly conditioned, so
there is no rate to fix and only the floor is left.

| arm | tuned setting | gap @ 200 | first < 1e-6 | first < 1e-12 |
| --- | --- | --- | --- | --- |
| fedavg | η 1.9 | 3.78e-02 | — | — |
| fedprox | η 0.1, μ 1.0 | 2.90e-02 | — | — |
| fedavgm | η 0.8, server 0.03, β₁ 0.0 | 3.25e-04 | — | — |
| fedadam | η 1.9, server 0.01, β₁ 0.9 | 6.11e-04 | — | — |
| fedyogi | η 1.9, server 0.01, β₁ 0.9 | 3.72e-04 | — | — |
| fedadagrad | η 0.8, server 0.1, β₁ 0.5 | 2.83e-04 | — | — |
| **scaffold** | η 0.2 | **3.08e-33** | **23** | **64** |
| fedlalr | η 0.001 | 2.77e-04 | — | — |

Thirty-one orders, and every other arm — including all four FedOpt members —
stalls between 3e-04 and 4e-02. A server-side step size damps the drift the
round injects and so buys two orders over FedAvg; it does not cancel it, and no
shipped arm but SCAFFOLD does.

## The control: with full participation, the ζ dial does nothing

With a shared `A` the local map is affine, so averaging `K` local steps over
*every* client gives

```
mean_i [ x_i* + (I − ηA)^K (x − x_i*) ] = (I − ηA)^K x        when  mean_i x_i* = 0
```

and the offsets have vanished. FedAvg is exactly unbiased however large ζ is.
This control is the one table here with no config directory of its own,
because it differs from the base arms only by two flags `fedbrew run` already
takes — a config per cell would be nine files stating what a flag states:

```bash
fedbrew run --config configs/examples/drift-quad/fedavg.yaml \
  --participation-rate 1.0 --lr 0.008 --output-dir outputs/examples/control/fedavg-p1-0.008
```

Measured that way, base dials, ζ = 1:

| η | FedAvg, ζ=1, p=1.0 | SCAFFOLD, ζ=1, p=1.0 | FedAvg, ζ=0, p=0.5 |
| --- | --- | --- | --- |
| 0.002 | 1.759e-04 | 1.759e-04 | 1.759e-04 |
| 0.008 | 5.568e-15 | 5.568e-15 | 5.568e-15 |
| 0.016 | 4.773e-29 | 4.654e-29 | 4.779e-29 |

FedAvg and SCAFFOLD print the same digits at η 0.002 and 0.008 and differ only
in round-off at 0.016 (4.773e-29 against 4.654e-29), and both agree with the run
that has no heterogeneity at all. Client drift on this problem is a statement
about partial participation: at `participation_rate: 1.0`, a SCAFFOLD win here
would be a difference that is not there.

## The starting point is part of the comparison

`x_0` was `x_init` in every coordinate while this example was being built, which
is the obvious choice and the wrong one. A per-coordinate normalised server step
— FedAdam with `β₁ = 0`, and any sign-like rule — moves every coordinate by the
same amount, so from a start where every coordinate is the same distance from 0
it can land on `x*` in a single round no matter what κ is. Measured on that
start: FedAdam reached `central_test_loss` 2.7e-12 **after round 1** and
3.3e-136 at round 200, against FedAvg's 2.4e-34. A table would have printed it
as a hundred-order win for adaptivity, and it was a fact about `x_0`.

`QuadraticModel` now starts at `x_0[j] = x_init/√a_j` — equal objective energy in
every coordinate, so `F(x_0) − F* = d·x_init²/2 = 8.0` whatever κ is, and the
coordinates start a factor `√κ` apart in distance. The dial then measures the
condition number instead of the start.

## How the arms were tuned

One grid, run identically at each of the three dial settings, scored on
`central_test_loss` at the final round; the winner is what the table reports and
what each arm config ships as its `client.learning_rate`. The grid is here,
because a grid is neither a run config nor a generator config and there is
nowhere in the tree it belongs:

```yaml
client.learning_rate:        # log-uniform; the upper end is the stability
  range: [1.0e-3, 2.0e-2]    # bound 2/L = 2/kappa, which is 0.02 at kappa=100
  grid: [0.001, 0.002, 0.004, 0.008, 0.016, 0.019]
  applies_to: [fedavg, fedprox, fedavgm, fedadam, fedyogi, fedadagrad, scaffold]
server.server_learning_rate:
  range: [1.0e-2, 4.0]
  grid: [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
  applies_to: [fedavgm, fedadam, fedyogi, fedadagrad]
server.beta1:
  range: [0.0, 0.99]
  grid: [0.0, 0.5, 0.9]
  applies_to: [fedavgm, fedadam, fedyogi, fedadagrad]
client.proximal_mu:
  range: [1.0e-3, 1.0e1]
  grid: [0.01, 0.1, 1.0]
  applies_to: [fedprox]
client.learning_rate@fedlalr:  # local AMSGrad's alpha is not bounded by 2/L,
  range: [1.0e-4, 1.0e-1]      # so it gets its own band
  grid: [0.0003, 0.001, 0.002, 0.005, 0.01, 0.03]
  applies_to: [fedlalr]
```

| knob | grid |
| --- | --- |
| `client.learning_rate` (SGD-family arms) | 5%, 10%, 20%, 40%, 80%, 95% of the stability bound `2/L` |
| `server.server_learning_rate` | 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0 |
| `server.beta1` | 0.0, 0.5, 0.9 |
| `client.proximal_mu` | 0.01, 0.1, 1.0 |
| `client.learning_rate` (fedlalr) | 3e-4, 1e-3, 3e-3, 0.01, 0.03 |

Three things this protocol does not give you, all of which the tables above are
weaker for:

* **One seed, and the selection metric is the reported metric.** These are
  best-of-grid numbers, not estimates of anything. On the floor-limited arms the
  grid is choosing between points whose separation is smaller than the
  oscillation those same runs show, which is why the base table above says not
  to read them as a ranking.
* **`beta1: 0.0` is in the grid, and it wins twice.** At `β₁ = 0`,
  `FedOptServer._fedavgm_update` is `update = server_lr · delta` — FedAvg with a
  server learning rate, under `server.strategy: fedavgm`. Nothing in
  `validation._validate_fedopt` objects, and nothing in the run record says the
  momentum is off. The name in the strategy column stops describing the
  algorithm, and only the arm's own config says so.
* **61 of the 861 grid points tripped the divergence guard** and were scored as
  failures rather than as slow runs. That is `divergence` doing its job — the
  stability bound `2/L` is real here, and a server learning rate of 3.0 on top
  of it leaves the basin in ten rounds — but it means the grid's edges are
  defined by a guard's thresholds as well as by the problem.

FedProx deserves its own line. On the base dials the tuner picks the smallest
penalty in the grid, μ 0.01, and the arm prints FedAvg's digits: at `ηK = 0.02`
the local iterate barely leaves the broadcast point, so a term that pulls it
back toward the broadcast point has almost nothing to pull. At ζ = 0 it picks
μ 0.01 again. At κ = 1 the tuned rate is η 0.1, so `ηK = 1`, and the tuner picks
the largest penalty, μ 1.0: the gap ends 23% below FedAvg's, on the same floor
and 50 to 100 times the FedOpt arms' gaps. FedProx is shipped here as FedAvg's
control, and on two of the three settings that is all it measures.

## Where the abstraction fit, and where it did not

This example used to open this section by listing five gaps it had met. Four
are closed, and the fourth commit of the extension-hook work is what closed
them; the fifth is still here. What follows is what a reader meets today,
with a note on what each replaced, because "this used to be a problem" is
worth knowing when the shape of the code around it still looks like the
workaround.

### Nothing about this example is a special case any more

**Closed.** There was no way for an out-of-tree component to reach the CLI:
`register_builtin_components` was the only caller of `Registry.register` a run
could get to, so this example shipped a `run.py` that imported `problem.py`,
composed a run config in Python and called `fedbrew.core.runner.run` directly
— past `--validate-only`, past the plan header, past every load-time guard,
and writing a `materialised_config.yaml` no other run in the repository has.

A config now names the components it is built from. `experiment.extensions`
in an arm config and `dataset.extensions` in the generator config each name
[`problem.py`](problem.py); the loader imports it and calls its `register()`
before any name in the config is looked up. `run.json` records the file's
SHA-256 beside the commit, and the plan header prints an amber `Extensions`
row, because some of what the run is built from is not the package.

The test of that is this README's four tables: every number in them was
produced by `fedbrew run` against a shipped config, and every one is
identical to what the old private runner produced.

### The problem is data, and the data says what the answer is

**Closed, and this is the part that changed most.** The offsets used to reach
the run through a closure — `problem.register(spec)` bound a `ProblemSpec`
into a dataset factory — because `factory._build_source_dataset` called a
third backend's factory with no arguments. Nothing in a config could state
the problem, and nothing in `run.json` could record it.

The offsets are now *generated*. `generate_drift_quad_from_config` writes one
shard per client and a manifest, `fedbrew generate` runs it like any other
generator, and a run reads it through the shipped `manifest_dataset`. The
one-way rule holds unchanged: the run reads the manifest and never
regenerates, and nothing at run time can see a `ProblemSpec`.

The reference optimum travels with the data, under the manifest's `reference`
key — `x*`, `F* = 0`, the dials, `2/L`, the drift radius, ζ as realised, and
every client's own optimum. It is a property of the shards, so it is written
beside them; `run_metadata.build_dataset_provenance` copies it into
`run.json`, which is how a run finally says what it was scored against. The
task reads it from `dataset_metadata` rather than closing over it.

An analytic problem fits the generator path with one thing left over. The
generator contract passes `seed` and `client_splits`, and this generator uses
neither: the offsets are Hadamard rows in exact ± pairs, so there is no draw
for a seed to fix, and the split ratios describe a cut of a corpus when there
is no corpus. Both are documented at the point they are ignored rather than
dropped silently. The generator config omits `client_splits` entirely, which
is the honest form — and `fedbrew inspect-data` then prints
`Client splits  train=unknown, eval=unknown`, which reads like a missing value
rather than an absent question. That is a small wart in a shipped tool, and
inventing ratios to remove it would be worse.

### Three splits over data that has none, now declared

**Closed as a silence, still true as a fact.** All three splits hold the same
single row, because `f_i` is not estimated from samples — it *is* the client.
The manifest declares that as `client_test_source: identical_to_train`, and
preflight prints a note on every run against this data:

```
❖ data.test_is_training_data  this dataset's client test split is its training
  data (client_test_source=identical_to_train): test_* and central_test_* are
  training numbers
```

Before, a reader had to reach this README to learn it.

### The problem is split across two config channels, and the task joins them

**Still true, and now checked in the right place.** The offsets reach the run
through the shards and the curvature through `model.extra`, because that is
the one free-form channel a run config has for a model. So `d` and `κ` are
stated twice, and a run whose two halves describe different problems is
expressible: a `d` mismatch would at least fail on a shape, and a `κ` mismatch
would fail on nothing at all and draw a plausible curve for a condition number
the config does not name.

`build_quad_vector` used to hold that check, against the registered spec. It
cannot any more, and should not have: a model builder is handed the model
block and nothing else. `DriftQuadTask.__init__` holds it instead, because the
task contract hands it both `model_config` and `dataset_metadata` — it is the
one object in the run that sees both channels. The check is still this
example's own; any out-of-tree problem whose definition spans two channels
needs its own.

### `model.input_dim` is the one dimension that needs no `extra`

`config.ModelConfig` names `input_dim`, `hidden_dim` and `num_classes` as
fields, `factory._model_config` injects whichever are set, and
`config_keys._INJECTED_KEYS` makes `reject_unknown_model_keys` accept them
without the builder declaring them. So a problem dimension travels as
`model.input_dim` and is validated as a config key; `condition_number` and
`x_init` travel as `model.extra`, which the loader does not check and the
builder does.

### A constant that is not state has to say so

`A` is the same on every client and never updated, so it is not federated state
— but `torch_utils.get_model_state` *is* `state_dict()`, and a plain
`register_buffer` is in `state_dict()`. A persistent buffer would have been
uploaded, averaged and counted into `communicated_parameters` every round:
numerically harmless, since every client holds the same `A`, and wrong in the
one column whose job is to say what a round costs — 32 parameters per round for
a 16-parameter model. `register_buffer(..., persistent=False)` is the fix, and
the `params/round` column in the table above is the check.

### The metric surface, from a second task

`compute_metrics` returns four numbers and all four reach `round_metrics.csv`
twice — as `fit_*` from the client fit path and as `central_test_*` from
`loop._evaluate_central_test_set`, which passes through every finite numeric key
`evaluate_global` returns. Both routes are open, and this task needed no
widening to use them.

The two routes pl-1d found closed are still closed, and the shape of what they
cost is clearer with a task that has a metric worth aggregating per client:
`config.CLIENT_METRIC_BASES` is still exactly `("loss", "accuracy")`, and
`loop._aggregate_client_split_metrics` iterates it, so
`distance_to_client_optimum` — the one metric here whose per-client
*spread* would say something, since it is a different number for every client —
arrives only as a mean over the fit path and never as `test_distance_to_client_optimum_std`.
`artifacts._CLIENT_EVALUATION_FIELDS` is still a fixed 13-column schema.

There is one new cost of running through the shipped surface, and it is
cosmetic rather than numerical: the plan header's metrics block predicts the
columns a run will write from `client_metric_names`, which is name-driven and
does not know this task reports no accuracy. So preflight lists
`central_test_accuracy`, `test_accuracy_avg` and four more that no round of
this example ever writes. The CSV is correct; the prediction of it is not.

### Four client rules still read `task._scaler`

**Still here**, unchanged, and the one gap from pl-1d's list that the hook did
not touch. Delta-SGD, FedLALR, SCAFFOLD and FedProx read `task._scaler` to
decide whether to refuse a run under `runtime.use_amp: true` — a private name
standing in for a question the interface has no place for. The read is
absence-tolerant, so a task that never defines it is refused nothing;
`DriftQuadTask` sets it to `None` explicitly only to put the answer where a
reader will look for it.

## Files

| File | What it is |
| --- | --- |
| `problem.py` | the whole extension: the objective, the analytic gradient, the closed-form structure (`A`, `b_i`, `x*`, `x_i*`, ζ, the drift radius, `2/L`), the generator that writes the shards and the manifest, the model, the task adapter, and `register()` |
| `run.py` | a convenience over the CLI: runs a directory of arm configs, then tables `outputs/`. Registers nothing, composes no config |
| `README.md` | this file |

And outside the example, where every other dataset and arm keeps theirs:

| Path | What it is |
| --- | --- |
| `data/configs/examples/drift-quad.yaml` | the generator config: the dials, and the extension that reads them. Two siblings turn one dial each |
| `configs/examples/drift-quad/*.yaml` | one run config per arm, tuned. Two sibling directories, one per dial setting |
| `data/generated/examples/drift-quad/` | the shards, `clients.jsonl` and the manifest — including `reference`, the closed-form optimum this run is scored against |
| `outputs/examples/drift-quad/<arm>/` | what a run wrote |
