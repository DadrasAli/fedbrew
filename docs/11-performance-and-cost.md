# 11 — Performance and cost

Where a run's time and memory go, what the tunable settings actually do, and
which numbers are measured rather than derived.

**Measured numbers in this chapter are labelled and dated.** They are not
guarded by tests and they will drift with hardware and dataset shape. Treat
them as orders of magnitude and remeasure with the tools in §7 before relying
on one.

## 1. Where a round's time goes

Six phases, timed separately and written as columns in `round_metrics.csv`.
Chapter 08 §8 gives the exact definitions.

| Column | Phase |
| --- | --- |
| `fit_sec` | client local training |
| `aggregate_sec` | the server's own aggregation |
| `client_eval_sec` | per-client evaluation |
| `global_eval_sec` | the central test pass |
| `checkpoint_sec` | building and writing checkpoints |
| `duration_sec` | the whole round |

Read these before tuning anything. The distribution is not what most people
expect: on cross-device FEMNIST, `client_eval_sec` can rival `fit_sec`, because
evaluation touches every client while training touches a sampled fraction.

## 2. Clients run one at a time, on purpose

A cross-device round is a small amount of GPU maths wrapped in a large amount
of Python, so concurrency adds contention without filling GPU gaps.

**The measurement behind that is historical.** Concurrency across processes and
across threads was implemented during development and measured against the
serial loop, and both came out slower. That was before the mechanisms, and the
code that compared them, were removed, so nothing in this tree can re-run the
comparison, and its figures are not quoted.

The consequence for tuning: a round's wall clock is dominated by per-client
Python overhead, not by matrix multiplication. Settings that reduce the number
of client visits help; settings that make each visit's arithmetic faster mostly
do not. The exception is `runtime.performance.executor: batched`, which removes
the per-client overhead instead, by training the sampled clients together
(§9).

**Each worker builds its optimizer once.** Every local update used to construct
its own `torch.optim` optimizer, 86 µs per client per round, 86 ms a round at
1000 clients. `reused_optimizer` (`fedbrew/clients/local_update_modes.py`)
keeps one per class per worker and hands it out bound to the update's
parameters with no state, so it steps exactly as a new one would; it is
rebuilt when the hyperparameters change, once a round under a cosine rate.
`tests/test_optimizer_is_reused.py` pins the trajectory against one optimizer
per update, for every rule that steps one.

## 3. Memory

**Peak model memory does not scale with participation rate.**
`configure_round` broadcasts one shared read-only state, and `aggregate_stream`
folds each result into a running weighted sum, so a client's model state
becomes unreachable as soon as it is aggregated, in `_stream_fit_results`
(`fedbrew/core/loop.py`). That holds for any `ClientExecutor`: the seam's
contract is a generator the `Aggregator` pulls from (chapter 01 §2).

Measured rather than reasoned: the accumulator retains **two model states** —
the running sum and one reference copy — identically at 4, 16 and 64 clients
(`tests/test_aggregation_peak_memory.py`). The chapter used to say one; two is
what it holds, and the number that matters is that it is the same number at
every participation rate. Buffering client states before averaging would be one
copy per participant, which is the cost `WeightedStateAccumulator` exists to
avoid. On CPU the reference copy shares storage with the *first* client's
state, so that one client's state does outlive its fold; the other N-1 do
not. Beside them it keeps two scalars per client per tensor, the minimum and
maximum it names a non-finite client from (chapter 07 §3.3).

**Two model states, except in one dtype each.** The running sum is kept in
float32 when the clients send bfloat16 or float16, so for those the sum is
twice the size of a client state and the pair costs three halves of what the
sentence above says. That is the price of the sum being usable at all: a
weighted mean over 3597 FEMNIST-shaped clients in a float16 running total
reaches 65504 and becomes `inf` on any coordinate whose sign does not change,
and in bfloat16 it lands 6% off the true mean in the median coordinate. In
float32 the sum stays float32 and nothing changes — which is every model this
repository ships. `torch_utils._accumulation_dtype` is the policy, and
`tests/test_accumulator_precision.py` is the measurement.

**Peak data memory scales with `participation_rate × client count.** The loop
deliberately does *not* release clients after fitting: evaluation runs
immediately afterwards over an overlapping client set, so releasing would force
every client to be rebuilt moments later. The cost is that a round holds
every participating client's data at once — `_stream_fit_results`
(`fedbrew/core/loop.py`).

**A manifest client stays built while its shard is cached.** The pool used to
release every client after its evaluation and rebuild it for its next fit —
constructor, setup, and a `load_state` of its own snapshot, per client per
round, 20% of an MNIST MLP round at 1000 clients. `LazyClientPool` now keeps a
client built while `touch_shard` finds its shard in the cache, and the cache's
eviction releases it (`on_shard_evicted`, `fedbrew/data/manifest_dataset.py`),
so a client object never holds a shard the cache has let go:
`shard_cache_bytes` (§4.5) stays the bound on client data in memory, and a
built client adds only its own small state beside it. With the cache off,
every client is released after its evaluation, as before.
`tests/test_resident_clients.py` pins the trajectory against released clients
and the bound at a small cap.

**The two per-client histories are held in memory for the whole run.** One
record per client per round. At thousands of clients with `clients: all` over
hundreds of rounds this is the dominant memory cost, and it is not streamed.
Their *CPU* cost is gone — `ClientHistorySummary` maintains running totals
rather than re-scanning — but the memory point stands.

## 4. The settings that change cost

### 4.1 `evaluation.*.clients` — the largest lever

Which clients an evaluation pass visits. Chapter 04 §8 gives the grammar.

| Scope | Visits |
| --- | --- |
| `participating` | only the clients trained this round |
| `all` | every client, every scheduled round |
| `sample:<N>` | a fixed *N*, the same set each time |
| `resample:<N>` | *N* redrawn each evaluation |

`all` is the honest population measure and the expensive one. At cross-device
scale it multiplies client visits by `1 / participation_rate`. If a
population-level number is not required, `sample:<N>` gives a stable estimate
at a fixed cost — and being *fixed* rather than resampled means round-to-round
movement is signal rather than a changing denominator.

Each split draws its own sample, so `val: sample:N` and `test: sample:N` cost
`2N` client visits between them and measure different clients — chapter 04 §8
says why that matters. Sharing one draw would be cheaper only in cache terms,
never in visits.

`evaluation.*.every` is the other half: a split evaluated every 10 rounds costs
a tenth of one evaluated every round, and the columns exist either way.

### 4.2 `client_statistics.per_client_csv`

Off by default, and it gates **both** per-client CSVs. Turning it on at
FEMNIST scale with `clients: all` produces roughly **1.8M rows over 500
rounds** (recorded on `ClientStatisticsConfig`, `fedbrew/core/config.py`).

The write cost used to be far worse: the files were rewritten in full on every
round that wrote a checkpoint, which `save_last` makes every round, so each
round rewrote every earlier round's rows and **the bytes written grew with the
square of the round count** while the data grew linearly. They are appended
now, and each row is written once.

Two related reductions: the `.jsonl` twins of these files, which held
byte-identical records in a larger format, are gone, and a checkpoint no longer
stores the model twice.

### 4.3 `runtime.performance.dataloader`

The `dataloader` block holds four keys, and two of them are unreachable in
every shipped config.

| Key | Default | Effect |
| --- | --- | --- |
| `num_workers` | `0` | worker processes; `--num-workers` also sets it |
| `pin_memory` | `false` | pinned host memory for faster H2D copies. **Inert while `num_workers == 0` and `fast_batching` is on** |
| `persistent_workers` | torch's | **only read when `num_workers > 0`** |
| `prefetch_factor` | torch's | **same gate** |

`num_workers` is `0` in every shipped config, so the last two are unreachable
as shipped and `pin_memory` is inert. Raising `num_workers` on a cross-device
run is usually a loss: each client's split is small, so worker startup is paid
per client per round against very little loading work.

Raising it stays reproducible. torch seeds each worker's Python and NumPy RNGs
itself, per worker and per epoch, from a base seed the loader draws from its own
generator afresh for every iterator unless `persistent_workers` keeps the
workers alive. `SeedWorker` (`fedbrew/core/torch_utils.py`) states that rule
here instead of inheriting it: both RNGs are seeded from `torch.initial_seed()`,
one definition for both task adapters, so the in-worker streams are this
repository's to define. The two copies it replaces derived the worker seed from
a value fixed when the loader was built, so every epoch of a round replayed one
in-worker stream. Nothing in a shipped dataset draws randomness in
`__getitem__`, so that changed no number — but adding an augmentation while
raising `num_workers` is one change, and it would have silently made every
epoch identical.

### 4.4 `runtime.performance.torch_num_threads`

`null` means **torch decides**, which on a shared node usually means it takes
every core it can see. On a cluster that allocates you a subset, setting this
to the allocation is the difference between using your share and
oversubscribing it. Nothing here reads SLURM's CPU count for you.

### 4.5 `runtime.performance.shard_cache_bytes`

4 GiB by default. Bounds the client-shard cache for `manifest_dataset`, which
reads shards from disk on demand. Forced to `0` under the `centralized`
strategy, whose pooled view keeps the whole concatenation resident anyway.

Raise it when the working set fits and storage is slow; it only ever avoids
repeat reads of shards nothing is allowed to edit, so it cannot change results.
Chapter 5 §8 covers what holds that -- half prevented, half refused.

### 4.6 `reuse_model` and `fast_batching`

`runtime.performance.reuse_model` and `runtime.performance.fast_batching` both
default `true`, and both are throughput-only. `fast_batching` reproduces
the `DataLoader` RNG protocol exactly, so a seeded run yields the same batch
order epoch for epoch — `tests/test_fast_batching.py` pins that against the
real `DataLoader`. Chapter 10.

### 4.7 `runtime.flush_every` and `evaluation.fit.every`

Both default to 1, every round, and neither changes a number.
`runtime.flush_every: N` writes the CSV rows, `run.json` and `latest.pt`, each
with one `fsync`, every N rounds instead of every round; an `fsync` on `/proj`'s
NFS costs ~7 ms against ~0.7 on `/tmp`, and a round does five. A kill loses at
most the rounds since the last flush. Chapter 09 §2.
`evaluation.fit.every: n` runs the post-fit pass behind the `fit_` metrics on
scheduled rounds only; it was 18% of an MNIST MLP round at 1000 clients. The
divergence monitor watching `fit_loss` then reacts up to *n* − 1 rounds later.
Chapter 04 §8.

## 5. Data staging

`runtime.data_staging` copies a manifest dataset to node-local scratch for the
duration of a job, for when shared storage is the bottleneck.

| Key | Default | Effect |
| --- | --- | --- |
| `enabled` | `false` | also driven by `--staging` / `--no-staging` |
| `local_root` | `null` | **the only way to point staging at a specific scratch path** |

`local_root` accepts an environment-variable path such as
`$FL_LOCAL_SCRATCH`. An unset value and one naming a variable the job did not
export are the same answer — no usable scratch root — so nothing is staged and
a line says so, rather than a literal `$FL_LOCAL_SCRATCH` directory being
created beside the checkout.

The destination is `<local_root>/<basename>-<digest>`, the digest taken over
the source directory's absolute path — `staged_directory_name`
(`fedbrew/core/data_staging.py`). The basename alone was the key, so two
datasets whose directories share one, such as
`$FL_DATA_ROOT/oasst1_qwen05b_4clients` and
`data/generated/oasst1_qwen05b_4clients`, staged into the same place and left
a tree holding one manifest beside both sets of shards. Two runs of the *same*
dataset still share one copy, which is the point of staging on a packed node.

The copy is whole or absent: it lands in a private temporary sibling and is
renamed into place once a `.fedbrew_staged.json` marker naming the source and
its file count is written. A tree without a valid marker is a copy that died,
not a dataset, and the next run replaces it. So a second run packed on the
same node cannot read a tree that is still being written.

**`run.json` does not record that a run was staged.** Its dataset provenance
names the source manifest, and `data_staging.enabled` in the config echo is the
request, so a staged run, an unstaged one and one whose staging was skipped
look the same on disk; the only record of a skip is its line on stdout. Open
by decision — `FINDINGS.md`, `POST-F11`.

Staging only helps when reads dominate and the copy amortises. It is a whole
dataset copy at job start; on a short job that copy is the cost. **The runtime
does not delete staged data** — node-local cleanup is the cluster's job — and
never write final outputs there.

## 6. Communication cost

Chapter 07 §5 has the per-algorithm table. The short version: FedAvg, FedProx
and the FedOpt family move one model-shaped state per direction per round,
SCAFFOLD two, FedLALR three, and LoRA far less than one, on the rules that
train it (chapter 07 §5.1).

`communicated_bytes` is emitted per client per round and already includes the
auxiliary state, so it is the true upload volume. **Compare arms on
`communicated_bytes`, not on round count alone** — a 100-round SCAFFOLD arm has
moved what a 200-round FedAvg arm did.

## 7. Measuring instead of assuming

`tools/` holds the scripts that produced the measured numbers here. They are
run by path, are not part of the installed package, and each takes `--help`.

| Script | Answers |
| --- | --- |
| `tools/profile_client_eval.py` | where per-client evaluation time goes, under cProfile against the real runner |
| `tools/bench_client_scope_report.py` | what `client_scope: all` costs against `selected`, per `local_iterations` |
| `tools/bench_compare_runs.py` | whether an optimisation changed only wall clock — **any** difference in `central_test_accuracy` between the two roots means it altered results and must be rejected, however good the speedup |
| `tools/bench_resolution_ratio.py` | the real forward/backward cost ratio between input resolutions, which is below the pixel-count bound because small models at small resolutions are launch-bound rather than FLOP-bound |
| `tools/generate_openimage_shaped_synthetic.py` | round cost at OpenImage's *shape* — 13,771 clients, ~94 examples each, 596 classes — with noise pixels, so a projected round time becomes a measured one. Not a dataset to train on; chapter 05 §2 |
| `tools/strip_checkpoint_client_states.py` | reclaims disk by dropping per-client state from `best.pt`, which never needs it |

The peak-memory claim in §3 is the exception to the labelling rule above: it is
not a dated measurement but a guarded one, re-measured by
`tests/test_aggregation_peak_memory.py` on every run of the suite.

`bench_compare_runs.py` encodes the rule worth stating on its own: **a
performance change is only a performance change if the numbers are identical.**
The one setting that changes them by summation order alone, the batched
executor, is held to a stated tolerance instead, and to identity where nothing
is summed differently (§9).

## 8. Before a long run

1. Run one `--rounds 1` job and read the six timing columns. Tune the phase
   that dominates, not the one you expected to.
2. Decide `evaluation.*.clients` deliberately — it is usually the largest cost
   and it changes what the numbers mean, not just what they cost.
3. Leave `per_client_csv` off unless per-client analysis is the point.
4. Set `torch_num_threads` to your CPU allocation.
5. Consider `--staging` only if the job is long and storage is slow.
6. Choose a checkpoint interval; `save_every_round` on a large model fills a
   quota quickly.

## 9. The batched executor

`runtime.performance.executor: batched` trains a round's sampled clients
together (`fedbrew/core/batched_executor.py`). Every client's parameters are
stacked on a leading client dimension, and each local step is one
`torch.func.vmap(torch.func.grad(...))` of the task's `functional_loss` over
the stack, followed by the rule's step over the same dimension
(`fedbrew/clients/batched_update.py`). The post-fit pass that gives `fit_*`
and the aggregation weight is one `vmap` of `functional_eval`, and the results
are folded in one weighted reduction per tensor (chapter 07 §3.3).
`sequential`, the default, is the reference it is held to.

| Key | Type | Default | Effect |
| --- | --- | --- | --- |
| `executor` | `sequential` \| `batched` | `sequential` | How a round's sampled clients are run. |
| `executor_chunk_bytes` | int > 0 | `1073741824` (1 GiB) | The memory one chunk of clients may take. Read only by `batched`. |

**What it computes.** Per client, what the sequential executor computes:

- the same batches in the same order, because each client's rule replays its
  own loader iteration on the row numbers of its split to name them
  (`sgd_mode_updates`, `own_loop_updates`), drawing what the loader draws;
- the same records, weights and state, because each client's rule builds its
  `FitResult` from its share of the stack with the code its own `fit` ends
  with (`batched_result`);
- the same arithmetic, to summation order: `functional_loss` is the loss
  `train_step` backpropagates, and the step is `torch.optim.SGD`'s and
  `AdamW`'s single-tensor step operation for operation. Across a stack,
  batched matrix products and reductions round in another order; a chunk of
  one client is not vmapped at all.

`tests/test_batched_executor_tolerance.py` runs each configuration through
both executors and compares every round's model, every persistent client
state and every non-timing cell: within `1e-12` relative in float64 (a tensor
against its own largest element, since an element whose exact value is 0
holds rounding residue), the identity and count columns equal, and bit for bit
when every chunk holds one client or a round samples one. The classification
task is compared at that bound in float64 -- the shipped task with its rows
and model widened -- and as it ships, in float32, at `1e-4`: float32 rounds at
`6e-8`, and three rounds of the MNIST MLP at 1000 clients differed by at most
`4.0e-5` (one client's `fit_loss`) and `1.9e-5` of a tensor's scale (measured
2026-09-27). A client state derived from a difference of models, SCAFFOLD's
`c_i`, is measured at least at its model tensor's scale: it tends to 0 as
clients agree, and its error is the models'. Two things the tolerance cannot
cover, both about values whose exact answer is 0:

- **a count of exact zeros** -- fed-lasso's `exact_zeros` -- is a function of
  summation order itself: a coordinate that cancels to 0 comes out `0.0` or
  `1e-19` by the order of the sum. It is compared bit for bit where the
  executors must agree, and not otherwise;
- **a kink** -- fed-lasso's L1 term, whose subgradient `sign(x)` is 0 at
  exactly 0 and ±1 a rounding error away -- turns that residue into a step
  `lam` apart, and from there the two runs part by more than rounding: under
  AdamW, 1.3e-3 in one coordinate at round 2, and on an A100 the shipped
  FedAvg arm parts from round 1 (measured 2026-09-27). The run was on the
  kink; the executor computed what it was given. The rules are compared on
  the smooth control, `fed-lasso-l2`, which agrees on the A100 to 2e-16 of
  the model's scale over 150 rounds.

**What is batchable.** A run is batched when all of these hold; otherwise it
runs sequentially, the plan header says `Executor: sequential; batched falls
back: <reason>` in amber, and `run.json` records the reason
(`reproducibility.executor`, chapter 09 §3.3):

| Part | Requirement |
| --- | --- |
| Task | implements `BatchableTask` (`fedbrew/tasks/base.py`): `split_rows`, `row_batches`, `functional_loss`, `functional_eval`, with a loss that averages over rows. The classification task does (the MLP and the CNNs), and so do the five linear examples; the causal-LM task does not. |
| Model | its federated state is exactly its parameters, all trainable; no dropout at `p > 0`, which draws from the process-wide generator; no batch normalisation, whose statistics are state |
| Rule | declares a batched update on its own class: `fedavg` in every update mode, with or without `max_grad_norm`; `local_sgd` with momentum, Nesterov, weight decay and a cosine rate; `local_adamw`, its step cap included; `fedprox`; `scaffold`, whose `c_i` is gathered from each client and its new value kept there. The last four under both of their modes. `fedlalr`, `delta_sgd`, `fedavg_ft` and `centralized` run sequentially. |
| Runtime | `experiment.seed` set, so every loader draws from its own generator; `use_amp: false`; CPU or CUDA; not the `centralized` strategy |

**Buckets and chunks.** Clients whose updates have the same shape -- the same
number of updates, each over the same number of batches -- are stepped
together. Within a bucket a batch shorter than the others is padded with one
of the client's own rows and masked, and the task's functions take the mean
over the real rows. Consecutive clients join a chunk while its estimated
memory fits `executor_chunk_bytes`: per client, its parameters times three
(parameters, gradient, sum or update) plus its optimizer's slots and any
persistent state, plus its split's rows and two of its longest batch. A
chunk's results are yielded in request order before the next chunk is
stacked, so what is held at once is one chunk's stack, and the previous one's
while the aggregator still holds that chunk's last result, as the sequential
executor's previous client state is held while the next client fits.
`run.json` records the most clients one chunk held (`largest_chunk_clients`).
A bucket's rows are held padded to its longest split and stacked, so a batch
that is every client's whole split, as the post-fit pass usually is, is used
in place. When a round is one chunk its stacked rows are kept for the next
round and reused for the same clients' data, the same objects unedited; at
full participation that is every round, and what is held between rounds is
what the chunk holds anyway.

**Evaluation.** A batched run is measured by the batched evaluator
(`fedbrew/core/batched_evaluator.py`), on the cadences of `evaluation.*`,
which it does not change. The clients due for evaluation in a round are
measured together: each requested split held as the task's rows, stacked per
split name, and batch `k` of every split measured in one `vmap` of
`functional_eval` at the broadcast model, which all of them share. Each
client's batches are its own evaluation loader's, shuffled or not, and its
result is built by its rule from its share with the code its own `evaluate`
ends with, so a missing `val` split is reported as zero examples and a
missing `test` split is refused in the same words, for the same client.
Splits are chunked by `executor_chunk_bytes`; when a round's evaluation fits
one chunk its rows are kept for the next evaluation round, and a train
split's rows are the ones the executor already holds. The central pass keeps
one model and copies the server's state into it in place, rather than
building a model and cloning the state into it each time, and reads the
global test shard once, refusing it if it has been edited since
(`CachedPayload`). The tolerance is training's: every run in
`tests/test_batched_executor_tolerance.py` evaluates every split every round,
and `tests/test_batched_evaluator.py` adds splits of different lengths, a
shuffled evaluation loader, a client without a `val` or a `test` split, and
the central pass's kept model and shard.

**What it keeps.** The no-training-batches refusal comes before any client
runs, in the rule's own words; the non-finite refusal names the client and
tensor the sequential run names; permuting the sampled clients permutes the
results, and a NaN in one client's data leaves every other client's result
bit-identical; neither executor draws from the process-wide generator on a
seeded run. `tests/test_batched_executor.py` pins each, and
`tests/test_batched_executor_tolerance.py` that two batched runs are
identical.

## For agents

### Paths

| Path | What it owns |
| --- | --- |
| `fedbrew/core/loop.py` | `_stream_fit_results` — streaming fit results, and why clients are not released after fitting |
| `fedbrew/core/state.py` | `ClientHistorySummary`, the running totals |
| `fedbrew/core/artifacts.py` | `flush_round_metrics_csv`, `flush_client_csvs` and `_append_csv_rows` — the CSV write volume, and the append path that replaced rewriting each file in full |
| `fedbrew/core/factory.py` | `_shard_cache_bytes`, including the centralized zero |
| `fedbrew/core/runtime_setup.py` | `configure_runtime` — thread count, cudnn, matmul |
| `fedbrew/core/data_staging.py` | staging, and the unresolvable-root path |
| `fedbrew/data/manifest_dataset.py` | on-demand shard reads and the cache |
| `fedbrew/clients/lazy_pool.py` | building clients on demand |
| `fedbrew/core/batched_executor.py` | §9: `BatchedExecutor`, its buckets and chunks, and `select_executor`'s fallback |
| `fedbrew/clients/batched_update.py` | §9: a rule's update as steps over a stack, and the batches its loop draws |
| `fedbrew/core/batched_evaluator.py` | §9: the due clients' splits measured together, and the central pass's kept model and shard |
| `tools/` | the benchmark and profiling scripts |

### Commands

```bash
# Prove this chapter's keys, defaults and tool list are current.
python -m pytest tests/test_docs_performance.py -v

# Where the time went, from a finished run.
python -c "import csv;rows=list(csv.DictReader(open('<output_dir>/round_metrics.csv')));\
print({k:round(sum(float(r[k]) for r in rows),1) for k in \
['fit_sec','aggregate_sec','client_eval_sec','global_eval_sec','checkpoint_sec']})"

# The benchmarks behind the measured numbers.
python tools/profile_client_eval.py --help
python tools/bench_client_scope_report.py --help
python tools/bench_compare_runs.py --help
```

### Invariants

1. **A performance change must not change the numbers.** `bench_compare_runs.py`
   exists to check that; any `central_test_accuracy` difference rejects the
   change regardless of the speedup. The batched executor changes them by
   summation order only, and is held to `1e-12` relative in float64, and to
   identity with one client per chunk (§9).
2. **Aggregation stays streaming.** Two model states, whatever the
   participation rate — never one per participant.
3. **Clients are released after evaluation, never after fitting.** Releasing
   after fitting forces every client to be rebuilt for the evaluation that
   follows.
4. **`shard_cache_bytes` is forced to `0` for `centralized`.** The pooled view
   already holds the concatenation.
5. **Staged data is never a destination for final outputs**, and the runtime
   does not clean it up.
6. **Measured numbers carry a source.** Either the code comment that records
   them or the tool that reproduces them. Never quote a figure with neither.

### Tests that guard this chapter

| Test | Claim |
| --- | --- |
| `tests/test_docs_performance.py` | The dataloader keys, their gates, the staging keys and the tool list here match the code. |
| `tests/test_client_csv_append.py` | The per-client CSVs append rather than rewrite. |
| `tests/test_round_metrics_are_appended.py` | `round_metrics.csv` appends rather than rewrites. |
| `tests/test_resident_clients.py` | §3: a client stays built while its shard is cached, at the released trajectory and within the cache's budget. |
| `tests/test_finiteness_is_checked_on_the_aggregate.py` | One finiteness check per round on the average, still naming the client. |
| `tests/test_optimizer_is_reused.py` | §2: one optimizer per worker, reset for each update, and the trajectory of one per update. |
| `tests/test_evaluation_cadence.py` | `evaluation.fit.every` skips the post-fit forward pass and changes no training number. |
| `tests/test_flush_cadence.py` | `runtime.flush_every` writes and fsyncs every N rounds and changes no number. |
| `tests/test_checkpoint_no_duplicate_model.py` | A checkpoint stores the model once. |
| `tests/test_checkpoint_size_is_constant_in_rounds.py` | A checkpoint does not grow with the round count. |
| `tests/test_client_history_summary.py` | The running totals replace the re-scan. |
| `tests/test_run_json_timing_is_running.py` | run.json's timing block is kept running too, bit for bit. |
| `tests/test_aggregation_peak_memory.py` | §3's peak model memory, measured at three participation counts. |
| `tests/test_shard_cache.py` | The cache is bounded, and serves without handing over what it keeps. |
| `tests/test_data_staging.py` | An unresolvable root stages nothing. |
| `tests/test_lazy_client_pool_selection.py` | The lazy pool follows from the dataset. |
| `tests/test_fast_batching.py` | `fast_batching` matches the real `DataLoader`. |
| `tests/test_evaluation_client_scope.py` | The four client scopes. |
| `tests/test_round_timing.py` | The six phase timings. |
| `tests/test_report_run_size.py` | Reported run size. |
| `tests/test_batched_executor_tolerance.py` | §9: both executors agree on every model, client state and cell, to `1e-12`, and bit for bit with one client per chunk. |
| `tests/test_batched_executor.py` | §9: the keys, the fallback and its record, client isolation, the refusals, the generator, and one chunk at a time. |
| `tests/test_batched_evaluator.py` | §9: ragged, shuffled and missing evaluation splits through both evaluators, the refusal's words, and the central pass's kept model and shard. |
| `tests/test_stacked_fold.py` | A stack's rows fold to their mean, and one row exactly. |

### Known failure modes

- **Tuning `fit_sec` when `client_eval_sec` dominates.** Read the columns
  first; on cross-device runs evaluation visits every client and training does
  not.
- **Raising `num_workers` on a cross-device run.** Worker startup is paid per
  client per round against very little loading work.
- **Setting `pin_memory` and expecting an effect.** Inert at
  `num_workers: 0` with `fast_batching` on, which is every shipped config.
- **Enabling `--staging` for a short job.** The dataset copy at job start is
  the cost you were trying to avoid.
- **Leaving `torch_num_threads` unset on a shared node.** Torch takes what it
  can see, not what you were allocated.
- **Turning on `per_client_csv` at scale without meaning to.** 1.8M rows over
  a 500-round FEMNIST run.
- **Quoting a number from this chapter as current.** They are dated
  measurements. Remeasure with `tools/`.
- **Asking for `batched` and not reading the plan header.** A configuration
  it cannot batch runs sequentially, as fast as it always did, and the header
  and `run.json` say why.
