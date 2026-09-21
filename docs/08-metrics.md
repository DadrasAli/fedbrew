# 08 — Metrics

Every number a run reports, where it comes from, what it is called on disk, and
how it was combined across clients.

This chapter is derived from the emitting code, not from prose. Each entry
names the module that produces it. If a claim here and the code disagree, the
code is right and this chapter is a bug.

## 1. How a metric reaches disk

A round produces metrics along two paths that never mix.

**The fit path** — what the local models did during training.

```
task.train_step / eval_step        per batch, on the client
  -> task.compute_metrics          folded to one number per client
  -> filter_metrics(client.metrics)  client-side filter, prefixed "fit_"
  -> FitResult.metrics             one record per selected client
  -> client_update_metrics.csv     one row per client per round; needs
                                   client_statistics.per_client_csv
  -> WeightedMetricAccumulator     example-weighted mean across clients
  -> filter_metrics(server.metrics)  server-side filter
  -> round_metrics.csv             one column per surviving name
```

**The evaluation path** — what the aggregated model does on held-out data.

```
task.eval_step                     per batch, on the client
  -> task.compute_metrics          one number per client per split
  -> EvalResult.metrics            keyed "{split}_{metric}"
  -> client_metrics.csv            per client, if client_statistics.per_client_csv
  -> _aggregate_client_split_metrics   the aggregate columns
  -> round_metrics.csv
```

The two paths answer different questions. `fit_loss` is the loss of *each
client's own locally-trained model* on *its own* training data, before
averaging. `train_loss_sample_weighted_avg` is the loss of *the aggregated
global model* on *every* client's training data, after averaging. They are
measured on the same examples and are not comparable.

Sources: `fedbrew/core/loop.py`, `WeightedMetricAccumulator`
(`fedbrew/servers/fedavg.py`), `TorchSGDClient._evaluate_model`
(`fedbrew/clients/torch_sgd_client.py`), `fedbrew/core/metrics.py`.

The plan header printed at run start and by `--validate-only`
(`fedbrew/core/logging.py`) lists these columns with a one-line gloss each. It
asks `client_metric_names` for them, so it tracks the `client_statistics`
toggles and the configured `worst_percent` rather than assuming the defaults.
By default it lists a curated subset; `--verbose` lists every column the run
will write. The gloss text is `fedbrew/core/metrics.py`, composed from the base
metric and the suffix — see §5.

## 2. The two base metrics

Everything in this chapter is built from exactly two per-client quantities.
`CLIENT_METRIC_BASES` (`fedbrew/core/config.py`) fixes the set:

```python
CLIENT_METRIC_BASES = ("loss", "accuracy")
```

Both are produced by the task. The two tasks define them differently.

### 2.1 `classification`

`fedbrew/tasks/classification/torch_classification.py`.

**loss** — mean cross-entropy, `nn.CrossEntropyLoss()` with its default
`reduction="mean"` (line 71). Per batch *b* with logits *z* and labels *y*:

```
L_b = -(1/n_b) * sum_i log softmax(z_i)[y_i]
n_b = targets.numel()
```

Batches are combined by example count, not by batch count
(`compute_metrics`, lines 258-271):

```
loss = sum_b (L_b * n_b) / sum_b n_b
```

**accuracy** — top-1 over `argmax(dim=1)` (lines 226-232):

```
correct_b  = |{ i : argmax(z_i) = y_i }|
accuracy   = sum_b correct_b / sum_b n_b
```

### 2.2 `causal_lm`

`fedbrew/tasks/causal_lm/torch_causal_lm.py`, `_loss_and_counts` at line 461.

Both metrics are computed over **active tokens only**. A token is active when
its target is neither `ignore_index` (default `-100`, covering padding and, for
SFT, the prompt) nor `pad_token_id` when one is set:

```
active = (target != ignore_index) & (target != pad_token_id)
total  = |active|
```

The second term matches by token value, so a `pad_token_id` equal to the
dataset's EOS id would remove every real end-of-document target from both
metrics and from `active_target_tokens`. That combination is refused at task
construction rather than measured — `docs/06` §3.3.

**loss** — cross-entropy over the active positions, mean-reduced:

```
L_b = -(1/total_b) * sum_{i in active} log softmax(z_i)[y_i]
```

Combined across batches by **token** count, not example count:

```
loss = sum_b (L_b * total_b) / sum_b total_b
```

This is a per-token mean cross-entropy in nats. Perplexity is `exp(loss)`; the
codebase does not compute or store it.

**accuracy** — next-token top-1 over active positions:

```
correct_b = |{ i in active : argmax(z_i) = y_i }|
accuracy  = sum_b correct_b / sum_b total_b
```

### 2.3 Degenerate inputs

Both tasks fall back identically (`compute_metrics`, classification
lines 254-262, causal_lm lines 317-324):

| Condition | `loss` | `accuracy` |
| --- | --- | --- |
| no eval records at all | `0.0` | `0.0` |
| records present, total count is 0 | unweighted mean of the per-batch losses | `0.0` |

A zero accuracy from this path is indistinguishable from a genuinely zero
accuracy. Chapter 05 covers the split-emptiness checks that make the second row
unreachable for a correctly generated dataset.

One batch of `causal_lm` can be degenerate on its own: every target position
inactive, so there is nothing to take a cross-entropy over. That batch's loss
is `0.0` and its `total` is `0`, so it drops out of the token-weighted mean
above and contributes nothing (`_loss_and_counts` line 494). The zero is built
by summing an empty slice of the logits rather than by scaling the whole logit
tensor to zero — it has to stay grad-connected, because the caller runs
`backward()` and `step()` on it like any other batch, and a scaled sum turns
`nan` the moment one logit in the block is non-finite.

The SFT packer drops windows with no active target
(`oasst1_sft.pack_sft_examples`), so no shipped generator emits such a batch.
The fallback is what a custom dataset, or a direct `compute_metrics(logits,
targets)` call, lands on.

## 3. Naming grammar

Every metric name on disk is built from these pieces.

| Piece | Meaning | Set by |
| --- | --- | --- |
| `fit_` prefix | measured on the client's **local post-training model** | `TorchSGDClient.fit` (`fedbrew/clients/torch_sgd_client.py`), `prefix="fit_"` |
| `{split}_` | `train`, `val`, `test` — the client's own split, measured with the **aggregated** model | `TorchSGDClient._split_metrics_and_counts` (`fedbrew/clients/torch_sgd_client.py`) |
| `personal_` prefix | the personalized pass; a split-name prefix, so `personal_val` behaves as a fourth split | `PERSONAL_SPLIT_PREFIX` (`fedbrew/core/config.py`), `_scope_split_names` (`fedbrew/core/loop.py`) |
| `central_test_` | the server's pooled test shard, one batched pass | `_evaluate_central_test_set` (`fedbrew/core/loop.py`) |
| `_{suffix}` | the cross-client aggregation rule — §5 | `client_metric_names` (`fedbrew/core/config.py`) |

The personalized pass is a *split prefix*, not a metric suffix, deliberately:
everything downstream keys off `f"{split}_{metric}"`, so `personal_val_accuracy`
flows through the aggregation, dispersion statistics and worst-percent summary
with no special handling — `_scope_split_names` (`fedbrew/core/loop.py`).

## 4. Fit-phase metrics

One value per selected client per round, then an example-weighted mean.

### 4.1 Aggregation rule

`WeightedMetricAccumulator` (`fedbrew/servers/fedavg.py`). For metric *m*:

```
m_round = sum_c (m_c * n_c) / sum_c n_c
```

where *n_c* is `FitResult.num_examples` for client *c* — the size of its train
split for classification, not the rows its round trained on; chapter 07 §3.1
defines it per task — and the sums run **only over clients that reported that
name**. Each metric keeps its own weight total,
so a metric only some clients report is averaged over just those clients rather
than diluted by the ones that never sent it. A metric whose weights sum to zero
is omitted from the round entirely rather than raising.

This weighting is **not** affected by `server.aggregation_weighting`. That key
changes how much a client's *parameters* count in the model average; metrics
stay example-weighted in every mode, because that is what makes the reported
number a population mean — `FedAvgServer._accumulate_fit_results`
(`fedbrew/servers/fedavg.py`).

### 4.2 Metrics every update rule reports

| Name | Definition | Emitted by |
| --- | --- | --- |
| `fit_loss` | §2 loss of the client's post-training local model on its own train split | `TorchSGDClient.fit` (`fedbrew/clients/torch_sgd_client.py`) |
| `fit_accuracy` | §2 accuracy, same model, same data | same |
| `optimizer_steps` | count of optimizer `.step()` calls this round | `TorchSGDClient.fit` (`fedbrew/clients/torch_sgd_client.py`) |
| `active_target_tokens` | sum of `total` over training outputs — examples for classification, active tokens for causal_lm | `TorchSGDClient.fit` (`fedbrew/clients/torch_sgd_client.py`) |
| `trainable_parameters` | `sum(p.numel() for p in model.parameters() if p.requires_grad)` | `TorchSGDClient.fit` (`fedbrew/clients/torch_sgd_client.py`) |
| `communicated_parameters` | tensor elements in the state the client uploads | `model_state_size` (`fedbrew/core/federated_state.py`) |
| `communicated_bytes` | `sum(t.numel() * t.element_size())` over that state, in bytes | same |

`communicated_bytes` counts **one direction, one client, one round**: the
upload. Chapter 07 gives the per-algorithm multipliers; SCAFFOLD and FedLALR
add their auxiliary state to this number at the client
in `TorchScaffoldClient.fit` (`fedbrew/clients/torch_scaffold_client.py`) and
`TorchFedLALRClient.fit` (`fedbrew/clients/torch_fedlalr_client.py`), so it is
already the true upload volume for those arms.

### 4.3 Which fit metrics survive to `round_metrics.csv`

Two filters, both `filter_metrics` (`fedbrew/core/metrics.py`), whose rule
is: **an empty list means keep everything**; a non-empty list keeps only the
named metrics that exist, and silently drops names that do not.

| Filter | Config key | Applies to |
| --- | --- | --- |
| Client-side | `client.metrics` | the `fit_`-prefixed task metrics only, before the algorithm extras are added |
| Server-side | `server.metrics` | the aggregated client metrics, before they reach `round_metrics.csv` |

The client-side filter runs on the **prefixed** names, so a config asking for
`fit_loss` selects the number the task reports as `loss`
— `TorchSGDClient._evaluate_model` (`fedbrew/clients/torch_sgd_client.py`).

**Each filter governs measured metrics and nothing else — but only its own
side.** The algorithm extras in §4.2 and §7 are added *after* the client filter
in `TorchSGDClient.fit` (`fedbrew/clients/torch_sgd_client.py`), and the two servers that produce
diagnostics add theirs *after* the server filter (§7.2, §7.3). Neither
list can remove a diagnostic added on *its* side, which is deliberate: a
metrics list that could drop `server_control_norm` could also drop a column
`checkpointing.best_metric` or `divergence.metric` names, and the run would
then fail mid-flight on a key it never asked to lose. The rule is stated once,
in `filter_metrics`' docstring (`fedbrew/core/metrics.py`).

**A client extra is exempt from `client.metrics` and not from
`server.metrics`.** The two filters run in series, so an extra the client adds
after its own filter still meets the server's. The five names in
`CLIENT_UNFILTERED_FIT_METRICS` — `optimizer_steps`, `communicated_bytes`,
`communicated_parameters`, `trainable_parameters`, `active_target_tokens`, plus
`client_learning_rate` on `fedavg`, `centralized` and `fedavg_ft` — reach
`round_metrics.csv` **only if `server.metrics` lists them or is empty.**

**More generally, naming a metric under `client.metrics` does not by itself
produce a column**, for any rule: the server filters the aggregated dict
against a non-empty `server.metrics`, so a name the client list carries and the
server list omits is emitted and then dropped. Fourteen shipped configs are in
that shape on purpose: every FedAvg-family arm lists `optimizer_steps` and
`client_learning_rate` under `client.metrics` and neither under
`server.metrics`, and `delta_sgd.yaml` and `fedlalr.yaml` keep `client_eta_0`
and `client_alpha` the same way. Nothing is malfunctioning — both lists do
exactly what the table above says — but the config reads as though it asked for
columns it does not get, so the plan header names every such name before the
run starts:

```
not written   client_learning_rate, optimizer_steps -- named in client.metrics
              and emitted, then removed by server.metrics. Add them there to
              keep them.
```

The shipped arms are left as they are rather than edited: adding the names to
`server.metrics` would make an arm's `round_metrics.csv` two columns wider than
the runs it has already written, which is a comparability problem in exchange
for a diagnostic nothing currently reads. A run that wants the
columns adds them to `server.metrics` in its own config.

**The communication columns are named in both lists.** Preflight tells a reader
to compare arms on `communicated_bytes` (chapter 07 §4.4), so every shipped arm
that reports the volume — `fedprox`, `scaffold`, `delta_sgd` and `fedlalr` on
FEMNIST, `scaffold` on MNIST — lists `communicated_parameters` and
`communicated_bytes` under `server.metrics` as well as `client.metrics`. For
those four rules both are needed: they filter their whole fit result against
`client.metrics` and have no row in `CLIENT_UNFILTERED_FIT_METRICS`, so a name
listed only under `server.metrics` never reaches the server. The five used to list them under `client.metrics` alone,
and the header's row, which then consulted only the FedAvg family's exempt
extras, said nothing. `tests/test_planned_columns_are_written.py` runs the
header-against-CSV round trip for every registered rule, and checks that each
shipped arm asking for the volume writes it.

**The evaluation path is deliberately outside both.** The loop sends
`metrics: ["loss", "accuracy"]` in the evaluation request built by
`_evaluate_models_on_clients` (`fedbrew/core/loop.py`),
which overrides `client.metrics`, and the split aggregates are written straight
into `round_info.metrics` without passing through `server.metrics`. Setting
`server.metrics: [fit_loss]` does **not** remove `test_accuracy_avg` from the
CSV.

This is a design choice, not an oversight, and the reasoning is worth stating
because the alternative looks tidier than it is:

| | Effect |
| --- | --- |
| **What controls the evaluation columns instead** | `client_statistics` (§5) plus `evaluation.splits` and `evaluation.model_scope`. That axis is *per statistic* — drop `_std`, drop `_variance`, change `worst_percent` — which is the axis anyone actually wants. Naming 36 columns individually is not. |
| **Why `server.metrics` is not extended to cover it** | It would silently drop columns other keys depend on: `checkpointing.best_metric` and `divergence.metric` both name evaluation columns, and neither is consulted when the filter runs. It would also change the output of every existing config that sets a non-empty `server.metrics`, including the shipped ones. |
| **Why there is no third `evaluation.metrics` key** | The config surface is the thing this documentation set is trying to shrink, and a third filter key with a third scope is the shape of surface that produces the drift in the first place. |

The consequence to know: `server.metrics` shortens `round_metrics.csv` on the
fit side only. The evaluation columns come with the splits you evaluate.

## 5. Client-evaluation aggregates

The largest group. `_aggregate_client_split_metrics` (`fedbrew/core/loop.py`) and
`_client_distribution_statistics` (`fedbrew/core/loop.py`).

**Column name:** `{split}_{metric}_{suffix}`, where `{split}` is one of `train`,
`val`, `test`, each optionally carrying the `personal_` prefix, and `{metric}`
is `loss` or `accuracy` -- `accuracy` conditionally; §5.4.

**Which clients count:** clients reporting zero examples for that split are
excluded before anything is computed, in `_aggregate_client_split_metrics`
(`fedbrew/core/loop.py`). Let *C* be the set of
clients with `n_c > 0` on that split, *m_c* the client's value, and
*N = sum_c n_c*.

| Suffix | Formula | Gloss | Switched by |
| --- | --- | --- | --- |
| `_sample_weighted_avg` | `sum_c (m_c * n_c) / N` | pooled over examples — the largest clients move it most | always on |
| `_avg` | `(1/\|C\|) * sum_c m_c` | averaged over clients — a 9-example client counts as much as a 900-example one | always on |
| `_std` | `sqrt( (1/\|C\|) * sum_c (m_c - mean)^2 )` — `statistics.pstdev`, **population**, divides by \|C\| | spread across clients (population standard deviation) | `client_statistics.std` (default `true`) |
| `_variance` | `(1/\|C\|) * sum_c (m_c - mean)^2` — `statistics.pvariance`, **population** | spread across clients, before the square root (population variance) | `client_statistics.variance` (default `false`) |
| `_min` | `min_c m_c` | the single lowest client value | `client_statistics.min` (default `true`) |
| `_max` | `max_c m_c` | the single highest client value | `client_statistics.max` (default `true`) |
| `_worst{P}` | mean of the *k* worst, `k = max(1, ceil(\|C\| * P / 100))` | the mean over the worst {P}% of clients — the tail, not the average | `client_statistics.worst_percent` (default `10.0`) |

`{split}_num_clients` is emitted beside these and is not one of them: it is
`|C|` itself, the client count every formula above divides by or sums over.

It is there because the aggregate names do not say what population they are
over. `evaluation.{split}.clients` takes `all`, `participating`, `sample:N` or
`resample:N`, and every setting produces the same column names — so
`val_accuracy_sample_weighted_avg` from a `sample:200` run and from an `all`
run over 3,597 clients are one column name over two populations, differing in
standard error by `sqrt(3597/200) = 4.2x`. `configs/femnist/fedavg_ft.yaml` is
the one shipped arm where this bites: it samples 200 writers, the other ten
FEMNIST arms use all of them, and it selects `best.pt` off the sampled column.
Its own config says so at the `val` block. With `{split}_num_clients` in the
CSV, a reader comparing two runs' columns can see it without opening either
config.

Selecting `best.pt` on it is refused: `validate_selection_metric` wants a
direction word, and a client count has none — there is no better or worse
number of clients.

The **Gloss** column is not written for this chapter. It is
`METRIC_SUFFIX_GLOSSES` in `fedbrew/core/metrics.py`, quoted verbatim — the
same sentences the plan header prints beside each column before a run starts,
composed with the base metric's own gloss:

```
loss     -> Cross-entropy
accuracy -> Top-1 accuracy
```

so `test_accuracy_avg` reads as "`Top-1 accuracy on client test data, averaged over clients — a 9-example client counts as much as a 900-example one.`"
`tests/test_docs_metric_names.py` diffs this table against that dictionary, so
a gloss cannot be improved in one place and left stale in the other.

Note that `_avg` and `_sample_weighted_avg` are the same number only when every
client has the same split size. On cross-device FEMNIST they differ
substantially, and which one a result quotes matters. That pair is the reason
the glosses have to differ in *words*: the two column names are one token apart
and the two numbers are both plausible, so a reader checking which one a table
quotes has nothing else to go on.

### 5.1 The worst-percent column

`_client_distribution_statistics` (`fedbrew/core/loop.py`). "Worst" is the
direction that is bad *for that metric*, not a
fixed end of the sorted list:

```python
ordered = sorted(values, reverse=metric != "accuracy")
computed[f"{prefix}_worst{label}"] = statistics.fmean(ordered[:count])
```

- For `accuracy`: ascending, so the *k* **lowest** accuracies.
- For `loss`: descending, so the *k* **highest** losses.

The `{P}` in the name comes from `worst_percent_label` (`fedbrew/core/config.py`):
`f"{float(p):g}".replace(".", "p")`. So `10` gives `worst10`, `2.5` gives
`worst2p5`, `5` gives `worst5`.

### 5.2 Full column list at the default configuration

With `client_statistics` at its defaults (`std` and `min` and `max` on,
`variance` off, `worst_percent` 10.0) and `evaluation.model_scope: global`,
each evaluated split contributes thirteen columns:

```
{split}_loss_sample_weighted_avg      {split}_accuracy_sample_weighted_avg
{split}_loss_avg                      {split}_accuracy_avg
{split}_loss_std                      {split}_accuracy_std
{split}_loss_min                      {split}_accuracy_min
{split}_loss_max                      {split}_accuracy_max
{split}_loss_worst10                  {split}_accuracy_worst10
{split}_num_clients
```

Three splits gives 39 columns; `model_scope: both` doubles that to 78.
`client_metric_names(split, statistics)` (`fedbrew/core/config.py`) returns exactly
this set for one split and is the authority — the guard in §13 diffs this
chapter against it.

### 5.3 When a split produces no columns

If no client reports a non-empty split, the loop raises rather than emitting
zeros — `_aggregate_client_split_metrics` (`fedbrew/core/loop.py`):

```
no client reported a non-empty {split} split; the dataset has no
{split} data and cannot be used for this evaluation
```

`val` is the exception at the *client* level: a client too small to hold out a
validation split reports zero and is dropped from the val aggregate instead of
failing the round — `TorchSGDClient._split_metrics_and_counts` (`fedbrew/clients/torch_sgd_client.py`). The
round still fails if
*every* client does this.

### 5.4 When a base metric produces no columns

`accuracy` is optional, unlike `loss`: `_validate_client_evaluation` only
requires `{split}_loss` from an evaluated client — `_validate_client_evaluation`
(`fedbrew/core/loop.py`) — and
`_aggregate_client_split_metrics` emits a base metric's twelve-or-fewer
columns for a split only if **every** client counted into that split
reported it — `_aggregate_client_split_metrics` (`fedbrew/core/loop.py`):

```python
for metric in CLIENT_METRIC_BASES:
    if not all(f"{split}_{metric}" in result.metrics for result in qualifying):
        continue
```

A task with no notion of correct/incorrect -- a scalar regression objective,
say -- reports `{split}_loss` and nothing named `accuracy` at all, on every
client, every round; no `{split}_accuracy_*` column ever appears for it, the
same way a column no `client_statistics` toggle turns on never appears.
`fit_accuracy` is unaffected: the fit path is free-form (§4.3) and was never
gated on this.

This is deliberately all-or-nothing rather than an average over whichever
clients happened to report the metric. Those are two different defects --
"this task has no accuracy" and "this task has accuracy and one client's rule
forgot to report it" -- and the aggregate cannot tell them apart from here, so
both get the same safe answer: the column is absent for that round rather
than computed from a subset the row's own schema does not say was partial.

## 6. Central test metrics

`_evaluate_central_test_set` (`fedbrew/core/loop.py`). One batched pass over the
server's pooled test shard, not a per-client aggregate.

| Column | Definition |
| --- | --- |
| `central_test_loss` | §2 loss of the global model over the whole `global_test` shard |
| `central_test_accuracy` | §2 accuracy over the same shard |
| `central_test_{name}` | any other finite numeric key the server's `evaluate_global` reports, unmodified — a task's own central-pass diagnostic |

**Aggregation:** none across clients — the shard is evaluated as one dataset, so
the batch-combining rule in §2 is the whole story. For classification that is
`total_correct / total_examples`; for causal_lm, `total_correct / total_tokens`.

**Switched by:** `evaluation.central_test.every` (default `10` in the
`EvaluationConfig` default factory, `fedbrew/core/config.py`). See chapter 04 for the
schedule grammar.

The server reports these under a `global_` prefix
(`FedAvgServer._result_weight`, `fedbrew/servers/fedavg.py`), and the loop
accepts any of `central_test_{name}`,
`global_test_{name}`, `global_{name}`, `test_{name}`, `{name}` — first match
wins — then always writes the column as `central_test_{name}`
(`_evaluate_central_test_set`, `fedbrew/core/loop.py`). That resolution is
required for `loss` only: a missing or
non-numeric one fails the round. `accuracy` uses the same five-spelling
resolution but is optional — a central pass with no notion of correct/incorrect
reports no `central_test_accuracy` rather than failing (§5.4 is the same rule
on the per-client aggregates). Every other key `evaluate_global` returns
reaches `round_metrics.csv` the same way, bare name first, `central_test_`
written back on — also optional, in `_evaluate_central_test_set`
(`fedbrew/core/loop.py`): a non-numeric or
non-finite value is dropped rather than failing the round, and that round's
cell is blank instead of the column
disappearing. Only the last spelling appears on disk in either case, and a
name a task invents gets a derived gloss and console grouping the same way
the two fixed ones do (`metrics.py`'s `metric_gloss`, `logging.py`'s
`classify_metric`) rather than one written by hand.

On datasets where the client test splits partition this shard, `central_test_*`
is a cheaper second view of `test_*_sample_weighted_avg` and should agree with
it closely. On the LLM corpora, which hold out by conversation tree, it is the
only measure of generalisation beyond the clients' own data.

## 7. Algorithm-specific metrics

Emitted only by the arms that produce them. Each is a real column in
`round_metrics.csv` for that arm and absent for every other.

### 7.1 FedProx — `TorchFedProxClient.fit` (`fedbrew/clients/torch_fedprox_client.py`)

| Name | Definition |
| --- | --- |
| `fit_proximal_loss` | `(mu/2) * ||w_local - w_round_start||^2` after local training |
| `fit_total_loss` | `fit_loss + fit_proximal_loss` — the objective actually minimised |

Both are example-weighted across clients by §4.1.

### 7.2 SCAFFOLD — client `TorchScaffoldClient.fit` (`fedbrew/clients/torch_scaffold_client.py`), server `ScaffoldServer.aggregate_stream` (`fedbrew/servers/scaffold.py`)

| Name | Definition | Side |
| --- | --- | --- |
| `control_delta_norm` | `||c_i_new - c_i_old||_2` — L2 norm of this client's control-variate change | client |
| `client_control_norm` | `||c_i||_2` after the update | client |
| `local_steps` | optimizer steps taken locally this round | client |
| `server_control_norm` | `||c||_2` of the server control variate after the round | server |
| `mean_client_control_delta_norm` | `(1/\|R\|) * sum_i ||c_i_new - c_i_old||_2` over participating clients | server |

The two server metrics are added **after** `filter_metrics`
(`ScaffoldServer.aggregate_stream`, `fedbrew/servers/scaffold.py`), so they
appear whether or not `server.metrics` lists
them. `configs/femnist/scaffold.yaml` lists them anyway; the list is not what
makes them appear.

### 7.3 FedLALR — client `TorchFedLALRClient.fit` (`fedbrew/clients/torch_fedlalr_client.py`), server `FedLALRServer.aggregate_stream` (`fedbrew/servers/fedlalr.py`)

| Name | Definition | Side |
| --- | --- | --- |
| `local_steps`, `optimizer_steps` | local step count, reported under both names | client |
| `client_alpha` | the client's configured base learning rate | client |
| `momentum_norm` | `||m||_2` of the synchronized momentum state | server |
| `second_moment_norm` | `||v_hat||_2` of the synchronized second-moment state | server |

The learning-rate diagnostics, one name per quantity. A client's
per-coordinate rate is `r_j = alpha / sqrt(v_hat_j)` over every coordinate *j*
of every floating tensor in its second moment after the round's last step --
no epsilon in the denominator, as in the update itself, because `v_hat` never
falls below `epsilon^2` (`_learning_rate_metrics`,
`fedbrew/clients/torch_fedlalr_client.py`):

| Name | Side | Population | Over coordinates | Denominator | Across clients, into `round_metrics.csv` |
| --- | --- | --- | --- | --- | --- |
| `effective_learning_rate_coordinate_mean` | client | one client's coordinates | mean | the client's coordinate count | example-weighted mean over the clients that report it (§4.1) |
| `effective_learning_rate_coordinate_min` | client | one client's coordinates | minimum | -- | example-weighted mean of the clients' minima (§4.1), **not** a minimum over clients |
| `effective_learning_rate_coordinate_max` | client | one client's coordinates | maximum | -- | example-weighted mean of the clients' maxima (§4.1), **not** a maximum over clients |
| `effective_learning_rate_across_clients_mean` | server | each participating client's `effective_learning_rate_coordinate_mean` | -- | the number of those clients | unweighted mean |
| `effective_learning_rate_across_clients_std` | server | the same | -- | the number of those clients (population) | `sqrt( (1/n) * sum (x - mean)^2 )`, unweighted |
| `effective_learning_rate_across_clients_min` | server | the same | -- | -- | minimum |
| `effective_learning_rate_across_clients_max` | server | the same | -- | -- | maximum |

The three `coordinate` names are per-client values: each client's row in
`client_update_metrics.csv` holds its own, and a round column exists only
when `server.metrics` passes it, as for every client metric. The four
`across_clients` names are the server's, computed in `_dispersion_metrics`
(`fedbrew/servers/fedlalr.py`) with each client counted once whatever its
example count, matching §5's population variance.

Both families were once `client_effective_learning_rate_{mean,min,max}`, on
the client and the server alike: different quantities under one name. In one
config shape -- a `client.metrics` naming the client's `_min` and not `_mean`
-- the round record carried the clients' averaged minimum under the name
defined as the server's minimum across clients (`FINDINGS.md`, `POST-F14`).
The four old names are retired (`RETIRED_METRIC_NAMES`,
`fedbrew/core/metrics.py`). A config naming one in `server.metrics`,
`client.metrics`, `divergence.metric` or `best_metric` is refused at load with
the replacement. A resume onto CSVs whose header carries one is refused
before anything is written, so no file holds both spellings.

`momentum_norm`, `second_moment_norm` and the four `across_clients` names are
added **after** `filter_metrics`, in `FedLALRServer.aggregate_stream`
(`fedbrew/servers/fedlalr.py`), as SCAFFOLD's are, so they appear whether or
not `server.metrics` lists them, with one condition. The four spread names
are computed from each client's `effective_learning_rate_coordinate_mean`, so
they exist only when the clients report it: a non-empty `client.metrics`
without it filters it out, and the server has nothing to spread.
`SERVER_DIAGNOSTIC_SOURCES` (`fedbrew/core/metrics.py`) records that
dependency. The plan header promises the four only when it holds, and config
load refuses a config that asks for one of them without it (§12,
`POST-F12`).

### 7.4 Delta-SGD — `_step_size_metrics` (`fedbrew/clients/torch_delta_sgd_client.py`)

The step-size trace is the algorithm, so it is emitted every round.

| Name | Definition |
| --- | --- |
| `client_eta_0` | the configured initial step size, echoed per round |
| `client_step_size_mean` | `(1/T) * sum_t eta_t` over the round's *T* local steps |
| `client_step_size_min`, `_max` | extremes of the same trace |
| `client_step_size_final` | `eta_T`, the last step size of the round |
| `step_size_clamp_fraction` | clamped steps / *T* — how often `eta_max` bound the step |
| `undefined_curvature_fraction` | steps with undefined curvature / *T* |

A run whose `client_step_size_mean` never leaves `eta_0` is one where the
auto-tuner did nothing. The client raises if the round produced no local steps.

### 7.5 FedAvg, FedOpt family, centralized

No server-side diagnostics. `fedopt` and its four aliases (`fedadam`,
`fedyogi`, `fedadagrad`, `fedavgm`) emit only the weighted client metrics
(`FedOptServer.aggregate_stream`, `fedbrew/servers/fedopt.py`).

## 8. Timing and count columns

Appended after the metric columns so adding a timing never shifts an existing
column's position — `_ROUND_TIMING_FIELDS` (`fedbrew/core/artifacts.py`).

| Column | Seconds spent in | Source |
| --- | --- | --- |
| `duration_sec` | the whole round, wall clock | `RoundTimings.total` |
| `fit_sec` | client `fit()` calls only, summed | `RoundTimings.fit` |
| `aggregate_sec` | the server's own aggregation — `fit_phase_seconds - fit_seconds`, floored at 0 | `RoundTimings.aggregate` |
| `client_eval_sec` | the per-client evaluation pass | `RoundTimings.client_eval` |
| `global_eval_sec` | the central-test pass | `RoundTimings.global_eval` |
| `checkpoint_sec` | building and writing checkpoints | `RoundTimings.checkpoint` |

All are `time.perf_counter()` deltas, rounded to 4 decimal places on write
(`save_round_metrics_csv`, `fedbrew/core/artifacts.py`). The server consumes fit
results as they stream in, so
`aggregate_sec` is a subtraction rather than a direct measurement
(`run_fl_loop`, `fedbrew/core/loop.py`).

A round replayed from an artifact written before timings existed leaves these
**blank**, not zero — `save_round_metrics_csv` (`fedbrew/core/artifacts.py`).

Two count columns sit before the metrics:

| Column | Definition |
| --- | --- |
| `round_id` | 1-based round number |
| `num_clients` | number of clients the server selected this round; 0 when `participation_probability` drew none, and such a round has no fit or server-diagnostic columns |
| `num_examples` | `sum_c FitResult.num_examples` over selected clients |

## 9. Where each metric lands

### 9.1 `round_metrics.csv`

One row per round. Rewritten in full every round, atomically via a `.tmp`
sibling and `os.replace` — `_atomic_text_writer` (`fedbrew/core/artifacts.py`).

Columns, in order — `save_round_metrics_csv` (`fedbrew/core/artifacts.py`):

```
round_id, num_clients, num_examples,
<every metric name seen in any round, sorted>,
duration_sec, fit_sec, aggregate_sec, client_eval_sec, global_eval_sec, checkpoint_sec
```

The metric column set is the **union over the whole history**, sorted
(`_round_metric_names`, `fedbrew/core/artifacts.py`). A metric first emitted at
round 40 gets a column for
rounds 1-39 too, left empty. A metric absent from a given round is written as
the empty string, not `0`.

### 9.2 `client_metrics.csv`

One row per client per round. **Off by default** — written only when
`client_statistics.per_client_csv` is `true`, which also switches on §9.3. At `clients: all` on a 500-round
FEMNIST run this is ~1.8M rows (`ClientStatisticsConfig`, `fedbrew/core/config.py`).

Fixed thirteen-column schema — `_CLIENT_EVALUATION_FIELDS`
(`fedbrew/core/artifacts.py`):

```
round_id, client_id, participated,
train_num_examples, val_num_examples, test_num_examples,
global_model_train_loss, global_model_val_loss, global_model_test_loss,
global_model_train_accuracy, global_model_val_accuracy, global_model_test_accuracy,
model_scope
```

| Column | Definition |
| --- | --- |
| `participated` | whether this client was **selected for training** this round, not whether it was evaluated |
| `{split}_num_examples` | the client's example count for that split, or `0` if that split was not evaluated this round |
| `global_model_{split}_{metric}` | the client's raw value for that split; **empty** when that split's count is 0 |
| `model_scope` | which model the row's numbers came from: `global` or `personal` |

The `global_model_*` names predate `evaluation.model_scope` and were not
renamed, because renaming them would break every existing reader. `model_scope`
is what disambiguates them:

| `evaluation.model_scope` | What the row holds | `model_scope` column |
| --- | --- | --- |
| `global` | the aggregated model's numbers | `global` |
| `personal` | the personalized model's numbers | `personal` |
| `both` | the **global** pass's numbers | `global` |

Under `both` only the global pass is recorded here, which is what these columns
have always held; the personalized numbers reach `round_metrics.csv` through
the `personal_` aggregates either way.

### 9.3 `client_update_metrics.csv`

One row per selected client per round, from the fit path. **Off by default** —
`client_statistics.per_client_csv` gates *both* per-client CSVs, not just
`client_metrics.csv` — `flush_round_artifacts` (`fedbrew/core/artifacts.py`) and
`_artifact_file_names` (`fedbrew/core/runner.py`). **Appended**
rather than rewritten — `flush_client_csvs` (`fedbrew/core/artifacts.py`): they
used to be rewritten in
full on every round that wrote a checkpoint, which `save_last` makes every
round, so turning the switch on meant rewriting every earlier round's rows on
every round.

```
round_id, client_id, phase, num_examples, <every fit metric name, sorted>
```

`phase` is always `"fit"` — it is the only value the loop constructs
(`_build_client_metric_record`, `fedbrew/core/loop.py`). `num_examples` is the
client's aggregation weight.

The metric columns are the client's `FitResult.metrics` after `client.metrics`
filtering and after the algorithm extras were added — so this file carries the
per-client detail that `round_metrics.csv` only has the mean of, and is not
subject to `server.metrics`.

### 9.4 `run.json`

| Key | Content |
| --- | --- |
| `results.final_metrics` | the last round's complete `metrics` dict, key-sorted (`save_run_json`, `fedbrew/core/artifacts.py`) |
| `timing` | the timing summary derived from the round history |
| `termination` | `null` for a normal run; otherwise the divergence verdict — §12 |
| `status` | `completed`, or the early-stop status |

`results.final_metrics` is the final round's numbers, not the best round's.
Best-round selection lives in the checkpoint tracker (§11).

## 10. Non-finite values

A diverged run's loss is `inf` or `nan`. Three mechanisms handle it.

**In the aggregates.** If *any* client's value for a split is non-finite, the
whole dispersion block short-circuits to `NaN` — `_std`, `_variance`, `_min`,
`_max` and `_worst{P}` all become `NaN` together —
`_client_distribution_statistics` (`fedbrew/core/loop.py`). The
column set is deliberately unchanged: a round that dropped columns instead of
reporting `NaN` would change the CSV schema partway through a run. The two
averages are not short-circuited; they propagate the non-finite value through
ordinary arithmetic.

The reason is that the helpers are unsafe in two different ways.
`statistics.pstdev` and `pvariance` use exact rational arithmetic and raise
outright on a non-finite value (on CPython 3.10, `ValueError` for NaN and
`OverflowError` for an infinity), which would kill the process mid-round before the
divergence monitor could record the blow-up. `min`, `max` and `sorted` are
worse: NaN comparisons are all `False`, so they return whichever element the
NaN happened to sit next to, with no sign that anything went wrong.

**On write to JSON.** `json_safe` (`fedbrew/core/metrics.py`) replaces
every non-finite float with `null`, recursively through mappings and lists,
before `run.json` is serialised. `json.dumps` is then called with
`allow_nan=False` in `save_run_json` (`fedbrew/core/artifacts.py`) so a field
that slipped past `json_safe`
is a loud failure rather than an invalid file. Without this, Python would emit
the bare tokens `NaN` and `Infinity`, which RFC 8259 does not define: Python
and `pandas.read_json` (since pandas 1.0) read them back, `jq` 1.6 turns `NaN`
into `null` and `Infinity` into the largest double, and Go, `serde_json` and
`JSON.parse` refuse the file outright.

**In the CSVs.** No conversion. A non-finite value is written as Python's
`repr` — `nan`, `inf`, `-inf`. `pandas.read_csv` parses all three.

## 11. Metrics that select checkpoints

`checkpointing.best_metric` names a round-level column.
`fedbrew/core/checkpointing.py`.

**Direction is derived from the name, never configured** (lines 24-40). The
name is split on `_` and the words checked against two sets:

```python
_MINIMIZED_METRIC_WORDS = frozenset({"loss", "error", "perplexity"})
_MAXIMIZED_METRIC_WORDS = frozenset({"accuracy", "acc", "f1", "auc"})
```

A name matching neither, or both, raises at config load. So `val_loss_avg`
minimises and `val_accuracy_worst10` maximises, with nothing to configure.

**Only validation metrics are accepted** (lines 45-70). The name must start with
`val_` or `personal_val_`; selecting `best.pt` on a test metric makes the
reported test score optimistically biased, so it is refused at config load with
a message listing the available names.

Chapter 04 covers the `checkpointing` block; chapter 09 covers what `best.pt`
contains.

## 12. Metrics that stop a run

`divergence` watches exactly one round-level metric.
`DivergenceConfig` (`fedbrew/core/config.py`), `fedbrew/core/divergence.py`.

| Key | Default | What it watches |
| --- | --- | --- |
| `divergence.metric` | `"fit_loss"` | the metric name to monitor |
| `divergence.non_finite` | `true` | fires on the round the metric becomes `NaN` or `Inf` |
| `divergence.blowup_factor` | `10.0` | fires when the metric exceeds this multiple of its **first observed** value |
| `divergence.blowup_absolute` | `null` | absolute ceiling; the backstop for a run already pathological at round 1 |
| `divergence.patience` | `null` | rounds without improvement **against the best so far** before the run is called stalled |
| `divergence.min_delta` | `0.0` | relative improvement required to reset the patience counter |

The default metric is `fit_loss` because it is produced every round and is free.
`train_loss_sample_weighted_avg` only exists on `evaluation.train`'s schedule
(default: every 10 rounds), so watching it would burn up to 10 rounds on an
already-dead run (`DivergenceConfig`, `fedbrew/core/config.py`).

For a cross-entropy loss the natural reference for `blowup_absolute` is the
random-guess value `ln(num_classes)` — 2.30 for MNIST's 10 classes, 4.13 for
FEMNIST's 62.

`patience` is off by default and is the only detector that can be wrong: at
participation rates of 1-2% the metric is measured on a different client subset
each round, so a rise can be the draw rather than the run. The counter that
turns consecutive non-improvements into a `stalled` verdict is
`_check_patience` (`fedbrew/core/divergence.py`); `min_delta` gates what counts as
an improvement at all, so a noise-sized gain does not reset it.

**How often that fires on noise alone is a derivation, and nothing here checks
it.** For exchangeable noise `P(k consecutive increases) = 1/(k+1)!`, which
puts a 500-round run at roughly 21 false alarms at `k=3`. No test simulates
that, no measurement is recorded, and the arithmetic assumes an exchangeability
the participation schedule does not guarantee — treat it as the order of
magnitude that motivates leaving `patience` off, not as a figure this
repository stands behind.

**`divergence: null` disables everything.** There is no `enabled` key — it was
removed because the same state had two spellings. The `active` property
(`DivergenceConfig.active`, `fedbrew/core/config.py`) is true when any detector
is on, and every detector off
*is* off.

**Every detector reads the same one name**, so a `metric` no round emits
silences all of them — `non_finite` included — for the whole run, with no
error. The loop notices at the end and prints
`divergence.metric=... was never present in any round's metrics`
(`run_fl_loop`, `fedbrew/core/loop.py`), which on a 500-round arm is several
GPU-hours late. Two
preflight checks cover the ways a name goes missing, and the first covers
**both** filters on the path from the task to the round record:

| Check | Refuses |
| --- | --- |
| `_validate_divergence_metric_is_reachable` | a `metric` that a non-empty `client.metrics` would drop before the `FitResult` is built, or that a non-empty `server.metrics` would drop before it reaches the round record. Evaluation columns, `central_test_*` and the strategy's own diagnostics are exempt from both, because §4.3's filter never reaches them. |
| `_validate_checkpoint_metric_is_emitted` | the same defect for checkpoint selection: a `val_` name `client_metric_names` never produces. |

**A strategy diagnostic built from a client metric is exempt only while that
metric survives.** FedLALR's four `effective_learning_rate_across_clients_*`
columns exist only while `client.metrics` keeps
`effective_learning_rate_coordinate_mean` (§7.3). The exemption is
`server_diagnostic_metrics`, which drops them when it does not, and
`_require_derived_diagnostic_sources` refuses a `divergence.metric` or either
metrics list naming one of the four under a `client.metrics` without it,
naming the metric to add. The check once exempted every strategy diagnostic
whole, so that monitor loaded and watched a column no round carried
(`FINDINGS.md`, `POST-F12`).

**The client filter is the earlier and the stricter of the two.** Every rule
runs its fit metrics through `client.metrics` before the `FitResult` exists
(`TorchSGDClient._evaluate_model`, `fedbrew/clients/torch_sgd_client.py`), so a list omitting `fit_loss` — the
default
`divergence.metric` — means no client ever reports it and `server.metrics` has
nothing left to keep. The two lists also have different exemptions: five rules
add `optimizer_steps`, `communicated_*`, `trainable_parameters` and
`active_target_tokens` *after* the client filter, so those survive any
`client.metrics` (`CLIENT_UNFILTERED_FIT_METRICS`), while `fedprox`,
`scaffold`, `delta_sgd` and `fedlalr` filter the lot and have no
exemption at all. `server.metrics` governs every one of those names for every
rule.

Both live in `validate_config` (`fedbrew/core/config.py`), which `load_config`
calls, so both stop a run. `fedbrew/core/validation.py` is reached only under
`--validate-only`; a check there reports and does not gate. Chapter 14 §6 and
its invariant 5 cover the distinction.

Neither can catch a plain typo in an evaluation column name that
`client_metric_names` *does* produce for some other configuration; the
end-of-run warning is the backstop for that.

When a detector fires, `run.json` gets `status` set and a `termination` block
(`DivergenceVerdict`, `fedbrew/core/divergence.py`):

```json
{ "detector": "...", "round_id": 0, "metric": "...", "value": 0.0,
  "threshold": null, "reason": "..." }
```

`threshold` is the blow-up ceiling for `blowup`, the best value so far for
`patience`, and `null` for `non_finite`, which has nothing to compare against.
Divergence and stagnation are reported as **different statuses**: a run whose
loss went to `NaN` is a different claim from one that merely stopped improving.

## 13. Config keys that change the metric set

Every key that adds, removes or renames a column.

| Key | Default | Effect on metrics |
| --- | --- | --- |
| `server.metrics` | required | Filters fit-phase and server-diagnostic columns in `round_metrics.csv`. Empty list keeps everything. Does not touch evaluation aggregates. |
| `client.metrics` | required | Filters the `fit_`-prefixed task metrics at the client, before algorithm extras are added. Does not touch the evaluation pass. |
| `client_statistics.per_client_csv` | `false` | Writes **both** `client_metrics.csv` and `client_update_metrics.csv`. With it off, a run's only metric artifacts are `round_metrics.csv` and `run.json`. |
| `client_statistics.std` | `true` | Adds `{split}_{metric}_std`. |
| `client_statistics.variance` | `false` | Adds `{split}_{metric}_variance`. |
| `client_statistics.min` | `true` | Adds `{split}_{metric}_min`. |
| `client_statistics.max` | `true` | Adds `{split}_{metric}_max`. |
| `client_statistics.worst_percent` | `10.0` | Adds `{split}_{metric}_worst{P}`; `null` or `0` removes it. Changing `P` **renames** the column. |
| `evaluation.model_scope` | `"global"` | `personal` replaces every split name with its `personal_` form; `both` emits both sets. |
| `evaluation.{train,val,test}.every` | `10`, `5`, `10` | Which rounds have values in that split's columns. The columns exist for the whole run either way. |
| `evaluation.central_test.every` | `10` | Same, for `central_test_loss` and `central_test_accuracy`. |
| `evaluation.{train,val,test}.clients` | `participating`, `all`, `all` | Which clients enter the aggregate — changes the numbers, not the column set. |
| `divergence.metric` | `"fit_loss"` | Requires that metric to be present every round. `validate_config` refuses a name a non-empty `server.metrics` would filter out (`config.py`, `_validate_divergence_metric_is_reachable`). |
| `checkpointing.best_metric` | — | Requires that column to exist; validated against `client_metric_names` at config load. |

A `clients` scope of `all` rather than `selected` changes what
`{split}_*_avg` means: it becomes a mean over the whole population rather than
over the sampled subset. Chapter 11 covers what it costs.

## For agents

### Paths

| Path | What it owns |
| --- | --- |
| `fedbrew/core/config.py` | `CLIENT_METRIC_BASES` — the two base metric names |
| `fedbrew/core/config.py` | `worst_percent_label` — the `worst10` / `worst2p5` spelling |
| `fedbrew/core/config.py` | `client_metric_names` — **the authority on aggregate column names** |
| `fedbrew/core/config.py` | `ClientStatisticsConfig` — the suffix toggles and their defaults |
| `fedbrew/core/config.py` | `DivergenceConfig` — the watched metric and detectors |
| `fedbrew/core/loop.py` | `_aggregate_client_split_metrics` — the two averages |
| `fedbrew/core/loop.py` | `_client_distribution_statistics` — std/variance/min/max/worst |
| `fedbrew/core/loop.py` | `_evaluate_central_test_set` — the `central_test_*` columns |
| `fedbrew/core/loop.py` | `_scope_split_names` — the `personal_` prefix |
| `fedbrew/core/metrics.py` | `json_safe` and `filter_metrics` |
| `fedbrew/core/artifacts.py` | `_CLIENT_EVALUATION_FIELDS`, `_ROUND_TIMING_FIELDS` — fixed column schemas |
| `fedbrew/core/checkpointing.py` | `selection_mode_for_metric` and `validate_selection_metric` — metric direction and the validation-metric rule |
| `fedbrew/servers/fedavg.py` | `WeightedMetricAccumulator` — the fit-phase aggregation |
| `fedbrew/tasks/classification/torch_classification.py` | `TorchClassificationTask.compute_metrics` — classification loss and accuracy |
| `fedbrew/tasks/causal_lm/torch_causal_lm.py` | `TorchCausalLMTask.compute_metrics` and `TorchCausalLMTask._loss_and_counts` — causal_lm loss and accuracy over active tokens |
| `fedbrew/clients/torch_sgd_client.py` | `TorchSGDClient._evaluate_model` — the `fit_` prefix and the client-side filter |

### Commands

```bash
# Prove the column names in this chapter match what the code emits.
python -m pytest tests/test_docs_metric_names.py -v

# The behavioural tests behind the formulas here.
python -m pytest tests/test_global_evaluation.py \
                 tests/test_evaluation_schedule.py \
                 tests/test_evaluation_model_scope.py \
                 tests/test_aggregation_weighting.py \
                 tests/test_non_finite_aggregation.py \
                 tests/test_communication_cost_metrics.py \
                 tests/test_client_communication_cost.py \
                 tests/test_divergence.py

# See the real column set for a config without training.
fedbrew run --config configs/dev/smoke.yaml --validate-only

# Read the columns a finished run actually produced.
head -1 <output_dir>/round_metrics.csv | tr ',' '\n'
```

### Invariants

1. **`client_metric_names()` is the authority on aggregate column names.** Any
   new suffix must be added there *and* in `_client_distribution_statistics`,
   or checkpoint validation will accept a metric no run produces. The two read
   the same `CLIENT_METRIC_BASES` tuple for exactly this reason.
2. **`CLIENT_METRIC_BASES` is `("loss", "accuracy")`.** Adding a third base
   metric changes every aggregate column set and the `client_metrics.csv`
   schema. It is not a local change.
3. **The column set must not change partway through a run.** A non-finite round
   emits `NaN` in every dispersion column rather than dropping the columns.
   Preserve this in any new statistic.
4. **`round_metrics.csv` columns are the union over the whole history, sorted.**
   A metric that appears late gets empty cells for earlier rounds. Never write
   `0` for an absent metric.
5. **Only `val_`-prefixed metrics may select checkpoints.** Selecting on a test
   metric biases the reported test score. Enforced at config load.
6. **Metric direction is derived from the name, never configured.** A new
   metric whose name contains neither a minimised nor a maximised word cannot
   be used for checkpoint selection until a word is added to one of the two
   frozensets in `checkpointing.py`.
7. **Fit metrics are always example-weighted**, regardless of
   `server.aggregation_weighting`. That key governs parameter aggregation only.
8. **`json_safe` must run on every record that reaches JSON**, paired with
   `allow_nan=False` at the `json.dumps` call.
9. **A metrics list filters measured metrics and nothing else.** A server that
   adds diagnostics of its own adds them *after* `filter_metrics`, as both
   that do already agree (§7.2, §7.3); a client adds its algorithm extras after
   `client.metrics`. Filtering a framework diagnostic would let a list drop a
   column another key depends on.
10. **The evaluation path stays outside both filters.** Its columns are
    controlled by `client_statistics`, `evaluation.splits` and
    `evaluation.model_scope`, and `fedbrew/core/loop.py` does not import
    `filter_metrics` at all. §4.3 gives the reasoning.

### Tests that guard this chapter

| Test | Claim |
| --- | --- |
| `tests/test_docs_metric_names.py` | Every aggregate column name in §5.2 equals `client_metric_names()`; the fixed schemas in §9 equal `_CLIENT_EVALUATION_FIELDS` and `_ROUND_TIMING_FIELDS`; the base metrics equal `CLIENT_METRIC_BASES`. |
| `tests/test_global_evaluation.py` | `test_accuracy_worst10` is computed as the mean of the worst 10%. |
| `tests/test_evaluation_schedule.py` | Suffix set follows the `client_statistics` toggles, including fractional `worst_percent`. |
| `tests/test_evaluation_model_scope.py` | `personal_` prefixing, and that checkpoint selection rejects a `worst` percentage other than the configured one. |
| `tests/test_aggregation_weighting.py` | Metrics stay example-weighted under uniform parameter weighting. |
| `tests/test_non_finite_aggregation.py` | A non-finite client value produces `NaN` in every dispersion column without changing the column set. |
| `tests/test_communication_cost_metrics.py` | A client's cost metrics survive `client.metrics` into the CSV. |
| `tests/test_client_communication_cost.py` | `communicated_bytes` equals every model-shaped state in the payload, for every rule. |
| `tests/test_scaffold_fedprox_communication_cost.py` | SCAFFOLD's 2x per-round volume. |
| `tests/test_divergence.py` | Detector thresholds and the `termination` block. |
| `tests/test_fedlalr_diagnostics.py` | §7.3: each FedLALR learning-rate column equals its hand-computed estimand for two clients with known rates, no name is both a coordinate and an across-clients statistic, a retired name is refused in every config place that names a metric, and a resume onto CSVs carrying one is refused and changes nothing (`POST-F14`). |
| `tests/test_divergence_metric_reachable.py` | §12's cross-check on both filters: it fires on a name either `client.metrics` or `server.metrics` would drop, stays quiet for evaluation columns, `central_test_*`, the selected strategy's diagnostics and the extras a rule adds after the client filter, derives `CLIENT_UNFILTERED_FIT_METRICS` from every rule's `fit`, no shipped config trips it, and the check stays in `validate_config` rather than the preflight module. |
| `tests/test_client_history_summary.py` | The running totals `client_update_metrics.csv` column names come from. |
| `tests/test_metric_filter_scope.py` | §4.3, §7.2 and §7.3: both servers add diagnostics after the filter, the loop never filters, and §4.3 states the choice as one. Fails if a third server starts emitting diagnostics. |

### Known failure modes

- **`fit_loss` read as the global model's training loss.** It is the mean over
  clients of each client's *own local* post-training model on its *own* data.
  The global model's training loss is `train_loss_sample_weighted_avg`, and it
  is only measured on `evaluation.train`'s schedule.
- **`_avg` and `_sample_weighted_avg` treated as interchangeable.** They are
  equal only when every client's split is the same size. On FEMNIST they are
  not, and quoting the wrong one changes the headline number.
- **Expecting `server.metrics` to shorten the evaluation columns.** It governs
  the fit side only (§4.3). Use `evaluation.splits`, `evaluation.model_scope`
  and `client_statistics`.
- **A non-empty `server.metrics` that omits `divergence.metric`.** The monitor
  then watches a name no round emits and every detector — `non_finite`
  included — stays silent, with the loop's warning arriving after the last
  round. `validate_config` refuses it, so the config fails to load and no run
  starts. Both defaults
  avoided it anyway: an empty `server.metrics` keeps everything, and `fit_loss`
  is emitted by every arm.
- **`server.metrics` expected to filter evaluation columns.** It does not. The
  evaluation path writes into `round_info.metrics` directly, and the loop
  overrides `client.metrics` with `["loss", "accuracy"]` for the eval request.
- **Changing `worst_percent` mid-sweep.** It renames the column
  (`worst10` to `worst5`), so `round_metrics.csv` files from the two halves of
  the sweep no longer share that column, and a `best_metric` naming the old
  spelling fails config validation.
- **Reading `client_metrics.csv` without checking `model_scope`.** The columns
  are named `global_model_*` under every scope; the `model_scope` column says
  whether they hold the global or the personalized model's numbers. Under
  `both` they hold the global pass, and the personalized numbers are in the
  round-level `personal_*` aggregates only.
- **Taking the plan header's default column list as the whole set.** At its
  default verbosity the header (`fedbrew/core/logging.py`) lists a *curated
  subset* — fifteen names at the shipped defaults, where a three-split run
  writes 36 — chosen so it explains the round output without drowning it. The
  same set is what each round prints. Every name in it is real
  (it is derived from `client_metric_names`), but it is not the full set. Run
  with `--verbose`, read the CSV header, or call `client_metric_names`
  directly.
- **Assuming `phase` in `client_update_metrics.csv` has more than one value.**
  It is always `"fit"`.
- **Looking for the per-client CSVs in a default run.** Both are gated on
  `client_statistics.per_client_csv`, which is `false`. A default run writes
  `round_metrics.csv`, `run.json` and `checkpoints/` only —
  `DEFAULT_ARTIFACT_FILES` names all four possible files, but
  `runner._artifact_file_names` decides which a given config produces.
- **Reading a CSV's last row after a kill.** `client_update_metrics.csv` is
  appended, not atomically replaced, so a process killed inside the write can
  leave a short final row. Both readers drop an unparseable final row.
