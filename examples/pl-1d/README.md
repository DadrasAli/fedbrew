# examples/pl-1d — a scalar Polyak-Łojasiewicz objective

*One of the problems in [`examples/`](../README.md); that index says what each one is for.*

A federated problem with no data, no model to speak of, and a known answer.
It exists to ask one question of the framework: **does an analytic per-client
objective fit fedbrew's task / client / server abstraction additively?**

The objective is the non-convex function that satisfies the PL inequality in
Karimi, Nutini, Schmidt, [arXiv:1608.04636](https://arxiv.org/abs/1608.04636) §2.2:

```
F(x) = x² + 3 sin²(x)        F'(x) = 2x + 3 sin(2x)        x* = 0    F* = 0
```

`F` is not convex — `F''(x) = 2 + 6 cos(2x)` goes negative — but `F'` has a
single zero, and `F` satisfies `½|F'(x)|² ≥ μ(F(x) − F*)` with `μ = 1/32`.
Client `i` optimises `f_i(x) = F(x) + s_i·x`; the shifts `s_i` are laid out in
exact ± pairs so they sum to zero and the uniform mean over clients is `F`
itself. Client `i`'s own minimiser solves `F'(x) = −s_i`, so it is not at 0 —
the clients genuinely disagree, and `shift_scale` is how much.

## Running it

Generate the data, then run an arm. Both are the ordinary commands; nothing
here is specific to this example except the two config paths.

```bash
fedbrew generate --config data/configs/examples/pl-1d.yaml
fedbrew inspect-data data/generated/examples/pl-1d/manifest.json

fedbrew run --config configs/examples/pl-1d/fedavg.yaml
fedbrew run --config configs/examples/pl-1d/scaffold.yaml --validate-only
```

The problem is defined outside the package, in
[`problem.py`](problem.py), and reaches the CLI because each config names it:
`dataset.extensions` in the generator config, `experiment.extensions` in every
arm config. Chapter 12 is the general form; [`drift-quad`](../drift-quad/) is
its worked case, and this example is the same shape with fewer dials.

`run.py` is a convenience over those commands and nothing more — it runs the
seven arm configs in order through the `fedbrew` CLI, then builds the
comparison table by reading `outputs/`:

```bash
python examples/pl-1d/run.py                 # 7 arms, CPU, no downloads
python examples/pl-1d/run.py --arm scaffold
python examples/pl-1d/run.py --table-only    # re-table what is on disk
```

Each arm writes `outputs/examples/pl-1d/<arm>/` with the usual four artifacts,
and each one starts a fresh interpreter that imports torch, which is the honest
cost of every arm being a real `fedbrew run` rather than a loop inside one
process.

## What the shipped strategies do on it

200 rounds, 8 clients, 4 sampled per round, 5 local steps, `shift_scale: 2.0`,
`x_init: 2.5`, seed 42. `gap` is `central_test_loss`, which for this problem
*is* `F(x) − F*` for the aggregated iterate.

| arm | gap @ 200 | median gap, rounds 181–200 | best gap | first round < 1e-6 | first < 1e-12 | `fit_distance_to_optimum` @ 200 |
| --- | --- | --- | --- | --- | --- | --- |
| fedavg | 2.29e-02 | 1.19e-02 | 1.64e-07 | 39 | — | 1.61e-01 |
| fedavgm | 7.91e-07 | 5.88e-02 | 7.91e-07 | 200 | — | 1.55e-01 |
| fedadam | 7.64e-03 | 7.65e-03 | 3.72e-07 | 78 | — | 1.61e-01 |
| fedyogi | 3.02e-03 | 7.84e-03 | 5.31e-08 | 95 | — | 1.58e-01 |
| fedadagrad | 3.23e-01 | 3.91e-01 | 3.23e-01 | — | — | 1.48e-01 |
| **scaffold** | **0.00e+00** | **0.00e+00** | **0.00e+00** | **23** | **59** | **1.80e-17** |
| fedlalr | 6.64e-03 | 1.10e-02 | 2.50e-08 | 93 | — | 7.58e-02 |

The separation is the textbook one, and it is legible precisely because the
problem is otherwise noiseless: local training has no minibatch noise, so the
*only* stochasticity is which four clients the round drew. Every arm but
SCAFFOLD descends to a floor set by the sampled subset's non-zero mean shift
and then oscillates on it. SCAFFOLD's control variates cancel exactly that
term, so it keeps contracting to machine zero. The last column is the tell: on
every other arm the selected clients' *local* iterates end 0.15 or so from `x*`
— sitting at their own tilted minimisers, which is client drift, measured — and
on SCAFFOLD they end at 1.8e-17, still on the global optimum.

`central_test_loss` and `test_loss_sample_weighted_avg` are two independent
routes to `F(x)` — one batched server pass, one per-client pass pooled over
clients — and they agree to ≤ 2e-15 on every arm. That is a useful check that
the per-client weighting really is uniform.

## How the arms are set

The rates are the ones this example has always shipped, and they were not
picked from a grid: `0.05` for every SGD-family arm, which is a fifth of the
`2/F''(0) = 0.25` above which plain gradient descent on `F` overshoots. The
bands a tuner would search are recorded here so the numbers above are
identifiable as points inside a band rather than as constants; nothing in the
tree reads them.

```yaml
client.learning_rate:          # log-uniform. The upper end is where a 1-step
  scale: log                   # FedAvg arm starts to overshoot: F''(0) = 8, so
  range: [1.0e-3, 2.5e-1]      # plain GD diverges above 2/8 = 0.25.
server.server_learning_rate:
  scale: log
  range: [1.0e-2, 2.0]
  applies_to: [fedavgm, fedadam, fedyogi, fedadagrad]
client.local_iterations:
  scale: int
  range: [1, 20]
problem.shift_scale:
  scale: linear
  range: [0.0, 4.0]
```

## Where the abstraction fit, and where it did not

It fit further than expected: `ServerStrategy`, the six protocol dataclasses,
`aggregate_stream`, the checkpoint/resume path and `round_metrics.csv` needed
nothing. All seven shipped strategies ran unmodified. This section used to
open by listing five places the example had to work against a class designed
for a dataset-backed model. Two of the five are closed — the extension-hook
work closed them, and [`drift-quad`](../drift-quad/)'s README tells that story
in full — one is closed as a silence, and two are still here. Each is kept
with a note on what it replaced, because "this used to be a problem" is worth
knowing when the shape of the code around it still looks like the workaround.

### §1 Nothing imported an out-of-tree component

**Closed.** `fedbrew run` built from `register_builtin_components()` and
nothing else could reach `Registry.register`, so this example shipped a
`run.py` that imported `problem.py`, composed a run config in Python and
called `fedbrew.core.runner.run` directly — past `--validate-only`, past the
plan header, past every load-time guard.

A config now names the components it is built from. `experiment.extensions`
in an arm config and `dataset.extensions` in the generator config each name
[`problem.py`](problem.py); the loader imports it and calls its `register()`
before any name in the config is looked up. `run.json` records the file's
SHA-256 beside the commit, and the plan header prints an amber `Extensions`
row, because some of what the run is built from is not the package. The seven
arm configs under `configs/examples/pl-1d/` are ordinary run configs, and
`run.py` composes nothing.

### §2 A third dataset backend and a third task received no configuration

**Closed, and the shape changed.** `factory._build_source_dataset` used to
call a third backend's factory with no arguments and `_build_task` a third
task's with none, so `PL1DDataset` was configured through a closure —
`problem.register(spec)` — and there was no way for a config to state the
problem or for `run.json` to record it.

There is no `PL1DDataset` any more. The shifts are *generated*:
`generate_pl_1d_from_config` writes one shard per client and a manifest,
`fedbrew generate` runs it like any other generator, and a run reads the
result through the shipped `manifest_dataset`. The reference optimum travels
with the data under the manifest's `reference` key — `x*`, `F*`, `μ` and the
shifts themselves — and `run_metadata.build_dataset_provenance` copies it into
`run.json`, which is how a run says what it was scored against. The task is
built from the documented contract (`factory.EXTENSION_TASK_KEYS`) and reads
`reference` off the `dataset_metadata` it is handed.

The `model:` block was the one free-form channel and still is: `x_init`
travels as `model.extra`, forwarded verbatim to the builder and checked by
`reject_unknown_model_keys`. It is the only model key, and it says nothing
about the problem — `x*` and `F*` are constants of the objective — so this is
the one example with nothing for the task to cross-check between the model
block and the manifest. `drift-quad` has two halves that must agree;
this one has one.

`MODEL_TASKS` **was** a closed dict, so `pl_scalar` had no entry and
`experiment.task: pl_1d` was the documented way past that. A model is now
registered with the task it needs — `registry.models.register(MODEL_NAME,
build_pl_scalar, task=TASK_NAME)`, which is the line in `problem.py` —
`MODEL_TASKS` is filled from those registrations, and `experiment.task` is
refused at load with the redirect.

### §3 Client rules reached through the adapter into three private attributes

Two of the three reaches are gone; the third is still here.

**Gone.** SCAFFOLD and FedProx used to skip `task.train_step` and run the step
themselves — `task._move_batch(batch)`, then `model(features)`, then
`task._criterion(outputs, targets)` and `.backward()` — reaching through the
adapter into two privates that only `TorchClassificationTask` has. Both now
wrap the optimizer instead: `torch_scaffold_client._ScaffoldCorrectingOptimizer`
and `torch_fedprox_client._FedProxCorrectingOptimizer` correct `.grad` inside a
`step()` that delegates to a real `torch.optim.SGD`, and the rule calls
`task.train_step(model, batch, optimizer)` like every other rule. Nothing under
`fedbrew/clients/` names `_move_batch` or `_criterion` any more.

For a task author that means the five abstract methods are now enough.
`PL1DTask` still has `_move_batch` and `_criterion`, but as its own helpers —
its `train_step` and `eval_step` call them — not as an interface a client rule
requires. Same for the batch shape: batches here still carry a dummy target
tensor nothing reads, but that is now this example's own choice and could be
dropped, rather than a shape every rule imposes by destructuring into two.

**Still here.** Four rules — delta-SGD, FedLALR, SCAFFOLD and FedProx — read
`task._scaler` to decide whether to refuse a run under `runtime.use_amp: true`.
That is a private name standing in for a question the interface has no place
for: whether a task supports mixed precision. The read is absence-tolerant
(`getattr(task, "_scaler", None)`), so a task that never defines it is refused
nothing; `PL1DTask` sets it to `None` explicitly only to put the answer where a
reader will look for it.

### §4 The metric surface was a closed vocabulary in four places; two are open now

`round_metrics.csv` and `client_update_metrics.csv` are fully additive — their
columns are the union of the names actually present. Everything upstream of
them is not:

| Route into the round record | Widening needed for a new name |
| --- | --- |
| fit path (`compute_metrics` → `fit_*`) | **none** — free-form, this is how the two new metrics arrive |
| per-client split aggregate | `config.CLIENT_METRIC_BASES` — `loop._aggregate_client_split_metrics` iterates it, so it is still exactly `loss` and `accuracy` |
| server central pass | **none, now** — `loop._evaluate_central_test_set` passes through every finite numeric key `evaluate_global` returns, as `central_test_<name>` |
| `client_metrics.csv` | `artifacts._CLIENT_EVALUATION_FIELDS` is a fixed 13-column schema |
| the terminal round block | `metrics.FIXED_METRIC_GLOSSES`, **or the `central_test_` prefix** — `logging.classify_metric` accepts the prefix as an open class, so a task-supplied central metric is classified rather than printed under "unclassified" |

`optimality_gap` and `distance_to_optimum` reach `round_metrics.csv` as
`fit_optimality_gap` and `fit_distance_to_optimum`, measured on each selected
client's **post-local-training** model and averaged over clients. The same two
numbers for the **aggregated** model now arrive too, as
`central_test_optimality_gap` and `central_test_distance_to_optimum` — a run of
this example emits all four columns. Before the central pass was widened they
did not arrive at all, and this problem was the one place that cost nothing,
because `F* = 0` makes `central_test_loss` *equal* to the global optimality
gap. That was a coincidence of this objective, not a general fit.

Two of the three routes are still closed, and they are different kinds of
closed. `CLIENT_METRIC_BASES` is a genuine vocabulary: a new base name has to
be added there and to `_client_distribution_statistics` together.
`_CLIENT_EVALUATION_FIELDS` is a CSV schema, so widening it is a file-format
change, not a lookup.

`accuracy` is no longer required anywhere, and this task no longer reports one.
It used to have to: `loop._validate_client_evaluation` demanded
`{split}_accuracy` from every evaluated client, `_aggregate_client_split_metrics`
indexed it unconditionally, and `_evaluate_central_test_set` refused a central
pass without a numeric one. A scalar objective has no accuracy, so
`compute_metrics` returned a hit indicator — `1.0` if the iterate was inside a
`tolerance` ball around `x*` — which was the most honest number that fitted
through a hole shaped like that.

All three now treat it as optional; `loss` is what stays required, and a base
metric absent from any client counted into a split is skipped rather than
averaged from whichever clients happened to report it. So the indicator is
gone, and with it `model.tolerance`, which existed only to feed it. This task
reports three numbers — `loss`, `optimality_gap`, `distance_to_optimum` — and
every one of them is a property of the problem.

That is the point of the shim being removable rather than merely unnecessary:
a task that carries a fake metric because the framework demands one will keep
carrying it, and a reader cannot tell the fake from the real.

What the plan header *predicts* is the one place this still shows. The header
lists the columns a run of this shape usually produces, from the metric names
rather than from the task, so a run of this example is promised
`central_test_accuracy` and the `test_accuracy_*` aggregates before it starts,
and its `loss` columns are glossed as cross-entropy. The run record is right
and the prediction is wrong.

### §5 Three splits over data that has none, now declared

**Closed as a silence, still true as a fact.** The loop needs a non-empty
`train` and `test` split from every client it evaluates. `f_i` is not
estimated from samples — it *is* the client — so all three splits hold the
same single row, and a number read off `test_*` here is a training number.
`evaluation.train` and `evaluation.val` are `never` for that reason; `test` is
kept only because the per-client aggregate is the one place client spread is
visible.

What changed is that the data now says so. The manifest carries
`client_test_source: identical_to_train`, and preflight prints
`data.test_is_training_data` on every run against it, so the fact is on the
terminal rather than in this paragraph alone. The generator config omits
`client_splits` entirely — the ratios describe a cut and there is none — and
`fedbrew inspect-data` then prints `Client splits  train=unknown, eval=unknown`,
which reads like a missing value rather than an absent question. Inventing
ratios to remove that would be worse.

## Files

| File | What it is |
| --- | --- |
| `problem.py` | the whole extension: the objective, the analytic gradient, the closed-form optimum, the generator that writes the shards and the manifest, the model, the task adapter, and `register()` |
| `run.py` | a convenience over the CLI: runs the seven arm configs, then tables `outputs/`. Registers nothing, composes no config |
| `README.md` | this file |

And outside the example, where every other dataset and arm keeps theirs:

| Path | What it is |
| --- | --- |
| `data/configs/examples/pl-1d.yaml` | the generator config: the dials, and the extension that reads them |
| `configs/examples/pl-1d/*.yaml` | one run config per arm |
| `data/generated/examples/pl-1d/` | the shards, `clients.jsonl` and the manifest — including `reference`, the closed-form optimum this run is scored against |
| `outputs/examples/pl-1d/<arm>/` | what a run wrote |
