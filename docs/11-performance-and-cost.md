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
| `checkpoint_sec` | building the checkpoints' snapshot; the writer writes them behind the loop |
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
do not. The exception is the batched executor, every batchable run's default,
which removes the per-client overhead instead, by training the sampled clients
together (§9).

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

**What lives for the whole run is frozen out of the garbage collector.** Every
full collection traverses every tracked object, and resident clients are many:
about 276,000 objects at 1000 MNIST clients, 27 ms of every round (measured on
2026-09-27). `run_fl_loop` therefore collects once and calls `gc.freeze()` when
its first round is done, and `gc.unfreeze()` when the run ends, by any exit, for
either executor (`_LongLivedObjects`, `fedbrew/core/loop.py`). Later
collections traverse only what later rounds build. The trade-off: a reference
cycle among frozen objects — a client the shard cache evicts, whose objects
refer to each other — is not collected until the run ends; everything outside
a cycle is still freed by reference counting as it was. A process that froze
objects itself, or turned the collector off, is left alone; what the
interpreter froze at its own start -- CPython 3.12 freezes 375 tuples, earlier
releases none -- does not count as the process's.
`tests/test_long_lived_objects_are_frozen.py`.

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

### 4.2 `reporting.per_client_csv`

Off by default, and it gates **both** per-client CSVs. Turning it on at
FEMNIST scale with `clients: all` produces roughly **1.8M rows over 500
rounds** (recorded on `ReportingConfig`, `fedbrew/core/config.py`).

The write cost used to be far worse: the files were rewritten in full on every
round that wrote a checkpoint, which `save_last` makes every round, so each
round rewrote every earlier round's rows and **the bytes written grew with the
square of the round count** while the data grew linearly. They are appended
now, and each row is written once.

Two related reductions: the `.jsonl` twins of these files, which held
byte-identical records in a larger format, are gone, and a checkpoint no longer
stores the model twice.

Off, the run builds no per-client record at all: those files are the records'
only reader, so the histories keep their summary -- the clients, the counts
and the metric names run.json's `scale` block reports -- and hold no record
(`run_fl_loop(client_records=...)`, `_AppendOnlyHistory.keeps`,
`fedbrew/core/state.py`), and the resident round builds none of a stack's
(`extend_stacked`). A caller of `run_fl_loop` keeps them by default. Figure
1's rounds spent 0.05 ms building 32 records each on 2026-10-04.

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
NFS costs ~7 ms against ~0.7 on `/tmp`, and a round does five. Every write's
I/O runs on one writer thread behind the loop, so the next round trains while a
flush is written; its Python half, formatting and serialising, runs on the
loop, where it does not contend for the GIL (chapter 09 §2). A kill loses at most the rounds since the last flush the
writer finished. Chapter 09 §2.
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
| `tools/drift_diagnostic.py` | the average drift at a model, ρ̂ = ‖mean_c (w − w_c^(H))/(ηH)‖ (Wang et al., arXiv:2206.04723), over one full-participation round with no update: where a dataset sits on the δ\* axis, `(η(H − 1)/2)‖δ*‖` to first order. Not a timing tool; listed here because every script in `tools/` is |

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

The batched executor trains a round's sampled clients
together (`fedbrew/core/batched_executor.py`). Every client's parameters are
stacked on a leading client dimension, and each local step is one
`torch.func.vmap(torch.func.grad(...))` of the task's `functional_loss` over
the stack, followed by the rule's step over the same dimension
(`fedbrew/clients/batched_update.py`). The post-fit pass that gives `fit_*`
and the aggregation weight is one `vmap` of `functional_eval`, and the results
are folded in one weighted reduction per tensor (chapter 07 §3.3).
It is the default: a run that states no `executor` takes it wherever its task
and rule can be batched, and the sequential executor where they cannot, which
the plan header and `run.json` record as the default, with the reason, not as
a fallback. That run is the sequential run: the client built to ask whether
its rule can be batched is not counted as one the run used until it fits or
evaluates it (`LazyClientPool.ask`), so its checkpoints hold the client
states the stated sequential run's hold. `sequential`, stated, is the
reference it is held to, and every test holds the batched paths to it on
autograd.

| Key | Type | Default | Effect |
| --- | --- | --- | --- |
| `executor` | `sequential` \| `batched` | unset: `batched` where it can run, `sequential` otherwise | How a round's sampled clients are run. `batched` stated on a run it cannot batch is a fallback, in amber. |
| `executor_chunk_bytes` | int > 0 \| `auto` | `1073741824` (1 GiB) | The memory one chunk of clients may take. Read only by `batched`. `auto`: half the device's free memory at the start of the run, below. |
| `cuda_graphs` | `on` \| `off` | `off` | Replays a resident round's training from a CUDA graph recorded once per round shape (§9.1). Read only by `batched`, on CUDA. |
| `gradient_form` | `closed_form` \| `autograd` \| `vmap_grad` \| `summed` | unset: `closed_form` where the task gives one, `autograd` otherwise | How a client's gradients are taken. `closed_form` takes no autograd at all: the task's formula, for every client of a stack at once, and in the sequential executor for its one client (`closed_form_train_step`). `autograd` is the form the task declares (`batched_gradient`), and `loss.backward()` in the sequential executor; `vmap_grad` and `summed` name the batched executor's two: `summed` walks the stack once forward and once back, and is the faster for a small model on the CPU. |

**What it computes.** Per client, what the sequential executor computes:

- the same batches in the same order. Each client's rule declares the loop
  its update takes over its loader, and the task declares what that loader
  yields (`LoaderOrder`: its batching, and how its generator draws the
  order); every client's batches for the round are then planned together
  (`fedbrew/clients/batch_orders.py`): the loaders' seeds derived as
  `dataloader_seed` derives them, the hash of the fields all clients share
  taken once; each shuffled loader's permutations drawn with the calls the
  loader makes, on a generator seeded as it seeds its own, so they are its
  permutations by construction; and every client's batch at every step cut
  from them as row indices for all clients at once, where replaying each
  loader collected them in Python. `tests/test_batch_orders.py` holds every order to the loader's
  own, iterated as each rule's loop iterates it (`sgd_mode_updates`,
  `own_loop_updates`), for every update mode, shuffled and not, with and
  without `drop_last` and `max_local_steps`. A task that declares no order
  has each client's loader replayed on the row numbers of its split
  (`row_batches`), which draws what the loader draws;
- the same records, weights and state, because each client's rule builds its
  `FitResult` from its share of the stack with the code its own `fit` ends
  with (`batched_result`). A rule whose result is that code unchanged --
  `fedavg`, `local_sgd`, `local_adamw`, on a task that weighs a client by the
  examples its post-fit pass counts -- builds the whole chunk's results at
  once instead (`batched_stacked_results`), from the chunk's columns: the
  post-fit metrics the task folded, each training step's `total`, and the
  same arithmetic per client. They reach the aggregator as one
  `StackedFitResults` a chunk (the seam's stacked path, chapter 01 §2), and a
  run with every client in one bucket is bit-identical to the same run one
  result at a time. `fedprox` and `scaffold` build their results client by
  client;
- the same arithmetic, to summation order: `functional_loss` is the loss
  `train_step` backpropagates, and the step is `torch.optim.SGD`'s and
  `AdamW`'s single-tensor step operation for operation. Across a stack,
  batched matrix products and reductions round in another order; a chunk of
  one client is not vmapped at all.

`tests/test_batched_executor_tolerance.py` runs each configuration through
both executors and compares every round's model, every persistent client
state and every non-timing cell: within `1e-12` relative in float64 (a tensor
against its own largest element, since an element whose exact value is 0
holds rounding residue; and a cell that is a difference of the row's other
quantities against their scale -- `constraint_violation` against 1, a
`_optimality_gap` or `_feasible_gap` against the row's `_loss`, a `_std`
against the row's `_avg`, `_min` and `_max` -- since near the answer it
holds its terms' rounding, `cell_scale` in the test), the identity and count
columns equal, and bit for bit
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

**Planned ahead.** A round's orders depend on the roster and the round's
number, never on what a round trains: FedAvg samples from
`derive_seed(seed, "participation", round)` (`sampled_client_ids`,
`fedbrew/servers/fedavg.py`, which `sample_clients` itself calls; SCAFFOLD's
server samples through it too), and each loader is seeded from the round and
the client. So a batched run plans them
from the roster read once at the start (`fedbrew/core/round_planner.py`),
with the calls `plan_round` makes, and the round adopts them as `plan_round`
would (`adopt_orders`) after checking they are its plans' -- the same clients,
loaders, loops and seeds; a round whose plans are not falls back to
`plan_round`, recorded. On a CUDA run worker processes plan a few rounds
ahead of the loop, as many as the job's CPUs less two, at most four, and
write each round into slots of shared memory made once, at the start, so only
a few numbers cross the queue per round (handing a round's tensors over one
by one, each through a file descriptor, cost about 6 ms of the loop's round
on an A100 node, measured 2026-09-29); on a CPU run, whose training
occupies those cores, the loop plans each round itself. A worker is a
spawned process that imports torch before it can plan, which took 3.4 to
4.2 s of the first round's wait on an A100 node (measured 2026-09-29), so
until one has said it started the loop plans each round it asks for itself,
with the same function, and does not wait; the workers take over from the
round after the first one asked for once one has. A worker that dies,
raises or does not answer within two minutes leaves the planning to the
loop, with the same function. `run.json` records `executor.planner`: `used`,
`workers`, the rounds the loop planned itself rather than a worker
(`in_process`), the seconds it waited on them (`waited_sec`), and any
`fallback` or `mismatch`; `used: off` names why a run's rounds could not be
planned ahead (another sampler, a rule that plans its update its own way, a
task without a loader order). A settings group plans its rounds together
(§10) and plans none ahead. `tests/test_round_planner.py` holds the planned
orders to `plan_round`'s, tensor for tensor, for every update mode, sampling
scheme and loader setting, and runs planned in the loop, in this process
and by two workers to each other, bit for bit.

**What is batchable.** A run is batched when all of these hold; otherwise it
runs sequentially, the plan header says `Executor: sequential; batched falls
back: <reason>` in amber, and `run.json` records the reason
(`reproducibility.executor`, chapter 09 §3.3):

| Part | Requirement |
| --- | --- |
| Task | implements `BatchableTask` (`fedbrew/tasks/base.py`): `split_rows`, `row_batches`, `functional_loss`, `functional_eval`, with a loss that averages over rows; optionally `loader_order`, which lets the round's batch orders be computed together rather than each client's loader replayed. The classification task does all five (the MLP and the CNNs), and so do the six linear examples; the causal-LM task does not. |
| Model | its federated state is exactly its parameters, all trainable; no dropout at `p > 0`, which draws from the process-wide generator; no batch normalisation, whose statistics are state |
| Rule | declares a batched update on its own class: `fedavg` in every update mode, with or without `max_grad_norm`; `local_sgd` with momentum, Nesterov, weight decay and a cosine rate; `local_adamw`, its step cap included; `fedprox`; `scaffold`, whose `c_i` is gathered from each client and its new value kept there. The last four under both of their modes. `fedlalr`, `delta_sgd`, `fedavg_ft` and `centralized` run sequentially. |
| Runtime | `experiment.seed` set, so every loader draws from its own generator; `use_amp: false`; CPU or CUDA; not the `centralized` strategy |

**Buckets and chunks.** Clients whose updates have the same shape -- the same
number of updates, each over the same number of batches, and a step with the
same terms: momentum, weight decay, FedProx's correction and clipping each
present or not -- are stepped together. Their values need not match: the
learning rate, momentum, weight decay, `mu` and clipping norm are each
client's own tensors (`ProgramValues`), and on the CPU each scalar form
torch's optimizers use has a tensor form that rounds the same (`add(b,
alpha=s)` is one fused multiply-add, as `addcmul(a, b, s)` is), so a client
stepped beside clients with other values is the client stepped alone, bit for
bit. On CUDA the pairs part: on an A100 under torch 2.5.1, `addcmul` differs
from `add(alpha)` in the last bit of about one element in ten and
`addcdiv(a, b * s, d)` from `addcdiv(value=s)` in about one in six, in both
widths. There a batched client matches the sequential one to the executor's
tolerance rather than bit for bit, while each setting of a group still
matched its own batched run alone bit for bit (the MNIST MLP at 1,000
clients, eight learning rates; measured on 2026-09-28). Within a bucket a batch shorter than the others is padded with one
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
`executor_chunk_bytes: auto` measures the free memory of the model's device
once, when the executor is chosen (after the dataset and model are built):
CUDA's free memory, or on the host the smaller of the kernel's `MemAvailable`
and the headroom of every memory cgroup the process is in, which is a batch
job's limit. A chunk gets half (`AUTO_CHUNK_FRACTION`): the estimate leaves
out a forward pass's activations and the allocator's slack, and the previous
chunk's stack can be alive while the next is built. `run.json` records
`chunk_bytes: {asked: auto, used, free, fraction}`, and the plan header prints
it; where free memory cannot be read the run takes the 1 GiB default and
records why. A settings group (§10) takes its first setting's budget for
every setting. Since the budget decides the chunks, and they the order of the
same sums, two `auto` runs on machines with different free memory agree to
the executor's tolerance, not bit for bit; setting the recorded `used` as the
budget repeats a run.
A bucket's rows are held padded to its longest split and stacked, so a batch
that is every client's whole split, as the post-fit pass usually is, is used
in place. When a round is one chunk its stacked rows are kept for the next
round and reused for the same clients' data, the same objects unedited; at
full participation that is every round, and what is held between rounds is
what the chunk holds anyway.

**The gradient.** A stack's gradients are taken in one of two forms, the
one the task declares (`batched_gradient`): `vmap(grad(functional_loss))`,
or one vmapped forward and one backward through the sum of the per-client
losses. Client `c`'s loss depends on client `c`'s parameters alone, so the
derivative of the sum with respect to them is its own gradient -- every other
term's is exactly zero -- and the two forms differ in how the stack is walked,
not in what each client gets, to summation order. In the summed form an
unclipped per-batch step is applied to the stacked tensors directly: it is
elementwise, so that is the vmapped arithmetic. Which is faster depends on the
model, so it is chosen per task by measurement (one CPU thread, 2026-09-27):

| Task | `vmap_grad`, ms a round | `summed`, ms a round | Declared |
| --- | ---: | ---: | --- |
| fed-lasso (FedAvg arm, central pass only, 150 rounds) | 7.82 | 5.98 | `summed` |
| drift-quad | 10.66 | 8.54 | `summed` |
| simplex-lsq | 6.81 | 5.21 | `summed` |
| nonconvex-simplex | 4.53 | 3.84 | `summed` |
| pl-1d | 5.90 | 4.86 | `summed` |
| classification, MNIST MLP at 1000 clients (local steps only) | 101 | 141 | `vmap_grad` |

On an A100 the MLP's local steps took 2.7 and 2.6 ms, no reason to change.
`tests/test_batched_executor_tolerance.py` holds each form, forced, to the
tolerance over every rule, the combining modes, ragged clients and the
float64 MLP; a stack of one client is not vmapped and takes the sequential
gradient either way.

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

**Records.** Per client, what reaches the host is floats: every bucket's
training and post-fit outputs, widened to float64 on the device, cross to the
host in one copy per chunk (`host_floats`), and every evaluation round's in
one copy. Where the task folds eval-step outputs into its metrics for a whole
stack at once (`stacked_metrics`, which the classification task implements),
each client's post-fit and evaluation metrics are computed on the device, and
what is copied is one dict of numbers per client and split rather than one
per batch; `tests/test_batched_executor.py` holds it to `compute_metrics`
split by split. A stack's rows are checked against the server's state once
per aggregation rather than once per row, a row's communicated size is
measured once per stack, and a client kept built checks its cached shard's
tensors for in-place edits by their counters alone
(`CachedPayload.check`), as it does on every access.

**What it keeps.** The no-training-batches refusal comes before any client
runs, in the rule's own words; the non-finite refusal names the client and
tensor the sequential run names; permuting the sampled clients permutes the
results, and a NaN in one client's data leaves every other client's result
bit-identical; neither executor draws from the process-wide generator on a
seeded run. `tests/test_batched_executor.py` pins each, and
`tests/test_batched_executor_tolerance.py` that two batched runs are
identical.

### 9.1 The resident round

A batched run whose rounds allow it holds them on its device for the whole
run (`fedbrew/core/resident.py`), and computes what the per-round path above
computes, bit for bit on the same device:

- **Rows.** Every client's training rows are stacked once, at the start. A
  bucket's rows are its clients' rows gathered from that stack, padded to the
  bucket's longest split: the tensors the round's own stacking gives, so the
  step reads the same values in the same layout.
- **Model.** The server's state is the round's mean, where the fold left it.
  The next round's clients start from it without a copy, and the host holds
  it only as the checkpoints and the evaluation read it. A server that
  updates from its fold -- FedOpt's four optimizers -- hands the mean to its
  own `update_from_fold` there (chapter 07 §3.2): the model it returns is the
  round's, and the moments it carries stay on the device. The update is
  elementwise arithmetic, written without a move to the CPU, so it rounds as
  the per-round path's does on the CPU; on CUDA it runs on the device where the
  per-round path runs it on the host, so the two agree there to the executor's
  float32 tolerance, as a batched run agrees with a sequential one (§9), while the
  eager and the graph-replayed resident rounds are the same kernels and agree
  exactly. The fold's finiteness is what a round is judged on, as the per-round path judges its aggregate. The flush reads
  each round's model and moments back, and the server adopts them
  (`adopt_update`), so a checkpoint holds the moments and the update count the
  per-round path's holds.
- **The fold.** Each bucket's `weights @ rows` is added into sums on the
  device, and the sums are divided there by the total weight as a device
  tensor. Float addition rounds the same on either device, and so does that
  division: a division by a CPU scalar is not the same on CUDA, which
  multiplies by the reciprocal -- 12,111 of the MNIST MLP's 50,176 first-layer
  weights differed that way, and none by a device tensor (measured on an A100,
  2026-09-28). A round with a bucket of one client is folded on the CPU, as
  the accumulator adds one client's row.
- **Plans and records.** A round's clients and orders are the planner's
  (§9, planned ahead), and its program the first client's rule's, which every
  client shares; no per-client object exists while the round trains. Each
  round's per-client outputs wait on the device until the flush, which reads
  every round since the last back in one copy and records each, in order,
  with the code the per-round path runs: the rule's `batched_stacked_results`,
  the loop's observer, the server's metric sums, and then the round's
  evaluation, verdict, checkpoints and flush, as `run_fl_loop` does them.
  A client is taken from the pool when the round would first touch it, so a
  checkpoint lists the clients it lists on the per-round path.
- **SCAFFOLD's controls.** Every client's `c_i` is a row of one table on the
  device, filled from its rule the round it first trains, and `c` a state
  beside the model. A bucket's clients read their rows and the shared `c`,
  and their new rows are written back as Option II computes them,
  `c_i - c + (x - y_i) / (K * learning_rate)`; the control deltas are summed
  a client at a time in the round's order, from zeros, into `c + sum / N`, as
  `ScaffoldServer` sums them -- elementwise arithmetic, the same on either
  device. The server is handed its results one by one, so its models are
  folded as its accumulator folds them: each run of consecutive clients of one
  bucket together, a run of one on the CPU. Each round's trained states wait
  for the flush, where every client's result is its rule's own
  `batched_result` -- its control update, kept in the rule, and its norms --
  and `c`'s update the server's own code (`ScaffoldServer.aggregate_folded`),
  so the host's `c_i` and `c` are what the per-round path's checkpoint of that
  round holds. Its rounds run eagerly.
- **Evaluation.** The round's due client splits are measured at the round,
  on the device, at the mean the fold left: every split's clients in the
  batched evaluator's order and chunks, their rows gathered from stacks held
  for the run, `measure_splits` and the task's folding as that evaluator runs
  them (`fedbrew/core/resident_evaluation.py`). The central pass runs
  `functional_eval` over the global test rows in `evaluate_model`'s batches,
  where the server's `evaluate_global` is the task's `evaluate_model` (FedAvg's
  and SCAFFOLD's, `central_pass_is_the_tasks`) and that is the classification
  task's, or is a `CentralPassInParts` task's (`fedbrew/tasks/base.py`) -- its
  steps, and its terms of the model alone measured beside them, as
  fed-logistic-l1's pooled objective is; heterogeneous-quadratic's has steps
  alone. Where the global rows are every client's train rows in order,
  unpadded, the pass reads the stack the round trains on rather than a copy
  of its own (`ResidentRounds._share_central`, compared once at set-up): the
  stacked rows themselves, or, where the steps take the closed form, the
  prepared stack through the task's `closed_form_central`, steps and terms
  from one set of rows (Figure 1's convex runs read three copies of the
  rows a round before, the training stack, the pass's rows and the gap's,
  where the AdaFed code reads one). On a round that measures `grad_norm_sq` too, where that pass is F --
  its rows every client's train rows -- the gradient is taken through it, one
  pass for both (`FusedPass`, `evaluation.grad_norm.fused`, chapter 08 §6.1),
  in the task's closed form where it gives one (`closed_form_eval`,
  `evaluation.grad_norm.gradient_form`): Figure 1's smooth-nonconvex rounds
  spent 0.83 ms of 1.98 on autograd's fused pass on 2026-10-04.
  Both are staged with the round's other values and read back
  in its one copy; the flush builds each client's `EvalResult` with its
  rule's `batched_evaluation_result`, and the central metrics with the task's
  `compute_metrics` -- or its `central_metrics` of the steps and the terms --
  as the evaluator does after its own copy. A rule the
  batched evaluator does not measure, and any other task's central pass, are
  evaluated at the flush by the evaluator itself.
- **The flush** (`fedbrew/core/resident_flush.py`). Nothing a round does
  waits on the device: its uploads are staged in pinned memory and copied
  without blocking (`uploaded`), and its finiteness flag joins its staged
  values. At a flush round the next round is queued first, so the device has
  work while the host records; then the window's values and models come to
  the host in one copy, made on a stream of its own behind the window's last
  round, and the host waits for that copy alone -- one wait per flush. The
  flush's writes run on the writer thread every run's loop writes through,
  in its order: the staged checkpoints to their temporary files, the CSV
  rows, run.json, and only then the checkpoints' commit (POST-F24). The loop
  hands over a flush only once the last one is written, so a kill loses at
  most the rounds since the last flush the writer finished, and a resume from that flush's
  `latest.pt` continues the run bit for bit. A round's timings are its
  phases on the device's own timeline, from events read back at the flush.
- **CUDA graphs** (`cuda_graphs: on`, `fedbrew/core/resident_graphs.py`). A
  round is computed in two halves: the host's (`_prepare`: its chunks and
  buckets, and every tensor its steps and fold read -- batch row indices and
  lengths, gathered rows' indices, fold weights and total), and the
  device's (`_execute`), which reads nothing else. The second time a round of
  one shape comes -- the same buckets, clients' positions, batch widths,
  sliced and masked steps, program and post-fit pass, all in the shape's
  key -- its device half is recorded as a CUDA graph, and every later round of
  that shape copies its host tensors and starting model into the graph's
  buffers and replays it: the same kernels on the same inputs, in the same
  order, bit for bit the eager round. The client and central evaluations run
  eagerly after it. A round folded on the CPU, one with a bucket of one
  client, a combined-gradient update and SCAFFOLD's run eagerly; so does every round
  after a capture or a replay fails, with a notice and a record
  (`executor.cuda_graphs`: `used`, `captured`, `replayed`, or `fallback`),
  and a run that is not resident or not on CUDA records `used: off` and why.
- **Stops.** A divergence verdict or a refusal inside a flush window ends the
  run at that round, exactly as the per-round loop ends it; the rounds
  trained after it are dropped. An aggregate that is not finite is run once
  more through the per-round path from the model before it, which refuses it
  naming the client and tensor it names; a client with no training batches is
  refused after the rounds before it are recorded.

It applies when the run is batched with a planner (§9);
uses FedAvg's server, fold and payload, FedOpt's (its update from the fold),
or SCAFFOLD's with its rule, the
streaming aggregator and the batched evaluator at the global scope; trains
`fedavg`, `local_sgd`, `local_adamw` or `scaffold`; holds every client's rows
-- and, for SCAFFOLD, the control table and a flush window's trained states --
in a quarter of the device's free memory; and has a client pool that cannot release a client
behind the round's back -- built clients, or a manifest dataset whose shard
cache holds every client's shard. Otherwise the per-round path runs it.
`run.json` records which: `executor.rounds`, `used: resident` or
`used: per_round` with the reason. `tests/test_resident_round.py` runs each
arm both ways and compares every CSV cell but the timings, every checkpoint
file -- model, server and client states, metrics and RNG state -- and how the
run ended: full and Bernoulli participation, clients of different sizes and
several buckets (some of one client), the own-loop rules, uniform weighting,
a post-fit pass on some rounds, a flush every third round, a manifest
dataset, a stall and a non-finite aggregate inside a flush window, and a run
stopped at a flush and between two and resumed from its `latest.pt`; SCAFFOLD
in one bucket and several, with runs of one client, under both of its modes,
with its norms reported, on a float64 example, with a control that stops
being finite, and resumed from inside a flush window; FedOpt's four
optimizers at full and Bernoulli participation, with the moments a checkpoint
holds, stopped inside a window and resumed; and of
the evaluation, that it is measured on the device, under mixed schedules and
client scopes, a shuffled evaluation loader, a client without a `val` split,
and a missing `test` split refused in the per-round path's words.

## 10. Settings run as one group

Runs whose configs differ only in numeric hyperparameters -- a learning-rate
grid, say -- can run as the settings of one group, in one process
(`fedbrew/core/settings_group.py`, `run_group`).

```bash
fedbrew sweep configs/sweep/lr-*.yaml          # groups them, then runs each group and the rest
fedbrew sweep --plan configs/sweep/lr-*.yaml   # prints the grouping, runs nothing
```

`fedbrew sweep` (`fedbrew/core/sweep.py`) groups its configs itself. Two
share a group when both load, both ask for `runtime.performance.executor:
batched`, neither resumes, and they are equal on every key but those that
name a run (`experiment.output_dir`, `run_id`, ...) and these: the client's
`learning_rate`, `momentum`, `weight_decay`, `proximal_mu` and
`max_grad_norm`, and the server's `server_learning_rate` and `beta1`. A key
one config has and the other lacks is a difference, so seeds, data, models
and local iterations are always the same inside a group; configs that would
write the same directory go to different groups. Each group of two or more
runs as one child process (`fedbrew sweep --run-group`), every other config
as `fedbrew run --config` would, one child after another in command-line
order. The sweep exits 1 if any run crashed, otherwise 2 if any was refused,
otherwise 0.

Each setting is
`runner.run` of its own config on its own thread, writing its own run
directory as it would alone; `run.json` records the group
(`reproducibility.group`, chapter 09 §3.3). What the settings share:

- the process: the imports, and the first `torch.func` call, which imports
  `torch._dynamo`, are paid once rather than once a run;
- the dataset, built by the first setting and handed to the others when their
  process-wide generators stand where the first's stood before it built it;
  they are then moved to where the first's stood after, so each draws on as
  it would have after building its own;
- each round's batch orders, planned once when every setting's plans are
  drawn alike (the same clients, seeds, loops and loader orders);
- the batched executor: each round, every setting's clients are trained
  together, the settings a second batch axis beside the clients. Each
  setting plans its round as its executor alone would, into chunks of its
  own (its units); every setting's first units, then every second, are
  packed into combined chunks under the same `executor_chunk_bytes`, and a
  bucket takes the rows of several settings when they are the same unit of
  the same orders, their values each client's own (§9, `ProgramValues`).

Exactly one setting runs at a time. A setting hands over when it waits for
the others -- to plan their round, or to take their share of a trained chunk
-- and when its run ends, always to the next live setting in group order,
so the interleaving is the same every run; at every hand-over the
process-wide generators (torch's, CUDA's, Python's, numpy's) are saved and
the next setting's restored, so each draws what it draws alone and its
checkpoints record what they record alone. A setting that ends -- completed,
diverged, stalled, refused or raised -- leaves the group, what it had not
taken is dropped, and the others go on. Its console lines are its own,
prefixed with its place in the group.

**Each setting is its run alone.** The step arithmetic is the same per row
(§9), and what a setting reads back is cut out as its executor alone would
hold it: a state stack shared by several settings is copied into each
setting's own tensors, which it folds as its own stack; each setting's
post-fit pass is measured on its own rows, since a task's evaluation may
multiply the stacked parameters by a tensor of its own -- fed-lasso's
objective does, by its design matrix -- which `vmap` makes one matrix product
over every row, and such a product rounds by how many rows it has (on the
CPU, a design of 40 × 20 against 16, 24 and 48 stacked iterates gave three
different last bits for the same iterate; measured 2026-09-28). The batched
training steps take each client's rows through products of its own. On the
CPU, every setting of the groups in `tests/test_settings_group.py` -- the
FedAvg, FedProx and FedAvgM arms of fed-lasso, and the MLP -- came out bit
for bit its run alone; what the tests hold is:

| What | Held to |
| --- | --- |
| a group of one | its config run alone, bit for bit: every non-timing CSV cell, every round's checkpoint, the generator state it records |
| each setting of a group | its run alone, within §9's tolerance; the generator state in its checkpoints, exactly |
| a setting that diverges or is refused | stops alone; the others bit-identical to the group without it |

A combined chunk's stacked rows are kept for the next chunk that trains the
same tensors: with the shared dataset every setting is served its own
mappings over the same shard tensors, so at the default budget, where one
setting's round of the MNIST MLP at 1000 clients fills a chunk alone, each
setting's chunk reuses the rows the first stacked, and each setting's
evaluator is handed them as its own. The evaluator is each setting's own:
evaluation is not batched across settings. A setting's timing columns
(`fit_sec`, `aggregate_sec`, `duration_sec`) are its own wall clock, which
includes the other settings' turns: its fold waits for the round's other
settings to plan. A group is not resumed as a group.

## 11. Modes of the batched step

Two opt-in keys change how the batched executor's training step runs
(`StepContext`, `fedbrew/core/batched_executor.py`): `runtime.performance.compile`,
and `numerics.precision`, which is in the numerics block because it changes
the numbers (chapter 04 §7.5). Both are refused beside
`runtime.performance.executor: sequential`, and the defaults leave the step as
§9 describes it. They change the training step only: the update arithmetic
stays at the model's precision, and the post-fit pass and every evaluation
run eagerly at the model's own precision, so what a run reports is measured
the way the reference measures it. `run.json` records what ran
(`reproducibility.executor`, chapter 09 §3.3), and the plan header prints it,
in amber when it is not what was asked for.

**`compile: on`** compiles a stack's whole round of local steps as one
function (`local_loop`, through `StepContext.loop`): every step's batches
gathered from the stack's rows by the round's row indices, the rule's
per-client step -- a batch's gradient and update, a pass's gradient sum, a
combined update -- vmapped over the clients, and the combination's weights,
all in one graph. Its inputs are the round's tensors; its shape (`LoopShape`:
the updates' batch counts, each step's width and whether it is full, the
clients, the corrections' dimensions) is a constant it is specialised to, so
one shape is one graph that every later round of that shape reuses, and
dynamo's guards and wrappers run once a round rather than once a step. The
step functions are made once per task, model structure, program shape and
gradient form for the run. The summed gradient form (§9) is compiled as
`vmap(grad)`; the closed form is compiled as the task's closed form of one
client, vmapped. Resident runs (§9.1) take it, their compiled rounds run
eagerly rather than replayed from CUDA graphs. Compiling costs once per shape
-- 12 s for ten steps of fed-logistic-l1's 32 clients on a loaded login node,
where the round then took 0.77 ms against 5.7 eager (measured 2026-09-30) --
and a run whose rounds keep changing shape (Bernoulli participation over
clients of different sizes) reaches the recompilation limit. A loop that
does not compile -- no C++ compiler for inductor, an operation dynamo does
not trace, more recompilations than the limit -- leaves that round and
every later one to the eager steps; the run prints why on stderr and records it
(`compile: {used: off, fallback: ...}`), and is then the reference run bit for
bit. Inductor needs a C++ compiler: on Berzelius the `g++` first on `PATH` is
a wrapper that refuses to run without a build-environment module, so set
`CXX=/usr/bin/g++`. On CUDA it also needs Triton, which the conda build of
torch 2.5.1 the repository's environments use does not bring: there a
compiled run falls back, and records `RuntimeError: Cannot find a working
triton installation` (measured on an A100, 2026-09-28).

**`precision`** trains at a lower precision:

| Mode | What the step does | Applies to |
| --- | --- | --- |
| `reference` | the model's own precision | everything |
| `f32_f64` | steps a float64 model in float32: parameters, data rows, masks and values cast once per round, the trained stack cast back | a float64 model (the linear examples) |
| `tf32` | float32 matmuls and convolutions on TensorFloat32 (`allow_tf32`), for the step only | a float32 model on CUDA |
| `bf16` | the loss under bfloat16 autocast | a float32 model |

A mode that does not apply runs the reference and says why.

**How far each is from the reference.** Measured against the batched
reference run of the same config over four rounds (2026-09-28, CPU), and
held by a test at the bound given:

| Mode | Measured, worst model tensor / worst cell | Held to | Test |
| --- | --- | --- | --- |
| `compile`, fed-lasso (float64) | 9.6e-18 / 4.2e-16 | `1e-12`, §9's | `tests/test_compile_mode.py` |
| `compile`, the MLP (float32) | 2.9e-7 / 1.6e-6 | `1e-4`, §9's float32 bound | same |
| `f32_f64`, fed-lasso-l2, simplex-lsq, pl-1d | at most 5.6e-7 / 6.0e-6 | `1e-4` | `tests/test_precision_modes.py` |
| `f32_f64`, fed-lasso | 1.1e-3 / 1.4e-4 | not held: the L1 kink (§9) turns float32 rounding into steps `lam` apart | — |
| `bf16`, the MLP | 9.5e-3 / 9.1e-3 in a loss spread | `5e-2`, model and loss cells | `tests/test_precision_modes.py` |
| `tf32`, the MLP on CUDA | 9.1e-5 / 2.8e-4, on an A100 | `1e-3`, model and loss cells | same, marked `cuda` |

Accuracy cells under `bf16` move by whole examples and are not held.

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
| `fedbrew/clients/batched_update.py` | §9: a rule's update as steps over a stack, the plan a rule declares, and the round's planning |
| `fedbrew/clients/batch_orders.py` | §9: every client's batch order for a round, from the loaders' declarations: seeds, permutations, batches |
| `fedbrew/core/round_planner.py` | §9: each round's sampled clients and orders planned from the roster, ahead of the loop, by worker processes |
| `fedbrew/core/resident.py` | §9.1: the resident round: rows, model and records held on the device, recorded at the flush |
| `fedbrew/core/resident_evaluation.py` | §9.1: the resident round's client splits and central pass, measured on the device at the round |
| `fedbrew/core/resident_flush.py` | §9.1: the flush's one copy to the host, its writer thread, and the round's device timings |
| `fedbrew/core/resident_graphs.py` | §9.1: `cuda_graphs`: a resident round's training recorded once per shape and replayed |
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
| `tests/test_resident_fold_update.py` | §9.1: FedOpt's four optimizers resident are the per-round path bit for bit: every CSV cell and checkpoint, the moments and update count included, at full and Bernoulli participation (a round that selects none), uniform weighting, a flush window, a float64 example, a stall and a non-finite aggregate inside a window, and a resume; a server that updates its own way runs per round and says why; on CUDA, the resident round against the per-round path to the float32 tolerance, and eager against replayed bit for bit. |
| `tests/test_resident_clients.py` | §3: a client stays built while its shard is cached, at the released trajectory and within the cache's budget. |
| `tests/test_long_lived_objects_are_frozen.py` | §3: what the first round built is frozen out of the collector from the second round to the run's end, by any exit, and a process's own freeze or disabled collector is left alone. |
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
| `tests/test_lazy_client_pool_ask.py` | A client built to be asked about (`LazyClientPool.ask`) is built once, and is in the checkpoints' client states only once the run uses it. |
| `tests/test_fast_batching.py` | `fast_batching` matches the real `DataLoader`. |
| `tests/test_evaluation_client_scope.py` | The four client scopes. |
| `tests/test_round_timing.py` | The six phase timings. |
| `tests/test_report_run_size.py` | Reported run size. |
| `tests/test_batched_executor_tolerance.py` | §9: both executors agree on every model, client state and cell, to `1e-12`, and bit for bit with one client per chunk. |
| `tests/test_batched_executor.py` | §9: the keys, the fallback and its record, client isolation, the refusals, the generator, and one chunk at a time. |
| `tests/test_default_executor.py` | §9: a run that states neither key trains batched on the task's closed form, or its declared autograd form without one; one it cannot batch runs sequentially with the reason, not a fallback, and keeps the stated sequential run's client states; stated `batched` it is a fallback; `sequential` stated takes the closed form unless `autograd` is asked for; the plan header's rows; what the config refuses. |
| `tests/test_closed_form_gradients.py` | §9: every example's closed form is autograd's within `1e-12`; each example's default run, batched and sequential, is the sequential run on autograd within the tolerance; the sequential closed-form step under every rule and update mode is the autograd step within it. |
| `tests/test_batch_orders.py` | §9: every planned order is its loader's own, for every task, update mode, shuffle, `drop_last` and `max_local_steps`, 520 clients at once included; the bulk seeds are `dataloader_seed`'s. |
| `tests/test_batched_evaluator.py` | §9: ragged, shuffled and missing evaluation splits through both evaluators, the refusal's words, and the central pass's kept model and shard. |
| `tests/test_compile_mode.py` | §11: a compiled loop is held to §9's bounds on fed-lasso (FedAvg, local_sgd in both modes, local_adamw, SCAFFOLD) and the MLP, dynamo having made its graphs; a resident run makes one compiled call a round, under `vmap_grad` and `closed_form`, within the executor's tolerance; a loop that does not compile runs eagerly, bit for bit the reference, and run.json says why; the key needs the batched executor. |
| `tests/test_precision_modes.py` | §11: `f32_f64` within `1e-4` on the smooth examples, and in a group of settings each as alone; `bf16` within `5e-2` on the MLP; `tf32` within `1e-3` on CUDA; each mode trained otherwise than the reference; a mode that does not apply runs the reference bit for bit and says why; the key needs the batched executor. |
| `tests/test_program_values.py` | §9: a bucket shares its program's shape, not its values; a client stepped beside clients with other values is the client stepped alone, bit for bit, in every shape and both float widths, and a client alone is `torch.optim`'s SGD and AdamW step. |
| `tests/test_sweep.py` | §10: `fedbrew sweep` groups the configs equal but for the run's name and the numeric hyperparameters it lists, runs any other config alone and one that does not load alone, keeps two configs that would write one directory apart, `--plan` runs nothing, `--run-group` refuses configs that are not one group, and a sweep of a group and a config alone writes both, the group recorded; the exit status. |
| `tests/test_settings_group.py` | §10: a group of one is its run alone bit for bit; each setting matches its run alone within the tolerance, its checkpoints' generator state exactly, under the stacked and the per-client paths, server settings, a budget that splits rounds, and the MLP; a diverging or refused setting stops alone, the others bit-identical to the group without it; run.json's `group`; how units are packed. |
| `tests/test_stacked_fold.py` | A stack's rows fold to their mean, and one row exactly. |
| `tests/test_stacked_results.py` | §9: the stacked path is each client's result exactly: the same metrics to the bit, the same refusals naming the same client, the model to `1e-12`; one bucket bit-identical to one result at a time. |

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
