# 01 — Architecture

What runs where: the round loop, the six registries that decide which pieces
it assembles, the four protocol objects those pieces exchange, and which module
owns which decision.

Read this to find where a change belongs. The chapters that follow describe the
pieces; this one describes the seams.

## 1. The shape

One loop, four pluggable positions.

```
        +--------------------------------------------------+
        |  runner.run()                                     |
        |    load config -> build -> loop -> write          |
        +--------------------------------------------------+
                              |
                    +---------v---------+
                    |  run_fl_loop()    |   fedbrew/core/loop.py
                    +---------+---------+
                              |
      +-----------+-----------+-----------+-------------+
      |           |           |           |             |
  ServerStrategy  ClientUpdate  TaskAdapter  FederatedDataset
  servers/        clients/      tasks/       data/
      |           |           |           |
  aggregates   trains one   forward,     supplies each
  the round    client       loss, state  client's shards
```

A run is: build the four, then repeat the round below `global_rounds` times,
writing artifacts after every round.

**What this is not.** It simulates federated learning and does not deploy it:
one process, one device per run, no network layer, no client daemon and no
secure aggregation. Clients are Python objects the server calls in sequence
(chapter 11 §2 says why they run one at a time), and "communication" is a tensor
handed from one object to another, so its cost is measured and reported —
`communicated_bytes`, chapter 08 §4.2 — not incurred. There is no sweep runner —
a sweep is a grid of generated configs submitted as independent jobs, and
`SLURMs/` shows two ways to do it — and no plotting: a run writes CSVs and
`run.json`, and a figure is something you build from them.

## 2. One round

`run_fl_loop` (`fedbrew/core/loop.py`), in order. Each phase is timed
separately, and those timings are columns in `round_metrics.csv` — chapter 08 §8.

| # | Phase | Who does it | Timed as |
| --- | --- | --- | --- |
| 1 | Select clients for this round | `server.configure_round` | — |
| 2 | Train each selected client locally | `client.fit` per client | `fit_sec` |
| 3 | Fold each result into the global model as it arrives; skipped when the round selected no client, which leaves the model and server state unchanged | `server.aggregate_stream` | `aggregate_sec` |
| 4 | Evaluate on each due client's own splits | `client.evaluate` per client | `client_eval_sec` |
| 5 | Evaluate on the server's pooled test shard | `server.evaluate_global` | `global_eval_sec` |
| 6 | Write checkpoints if due, staged: complete on disk as `.tmp`, not yet visible | `checkpointing` | `checkpoint_sec` |
| 7 | Flush artifacts: the CSV rows, then `run.json` | `artifacts.flush_round_artifacts`, the runner's writer | — |
| 8 | Commit the staged checkpoints, then prune to `keep_last` | `loop._commit_checkpoints` | — |

A round's checkpoints become visible last, so a kill anywhere in a round leaves
a checkpoint whose round the metric history already holds, which a resume can
continue (`POST-F24`).

Three properties of this loop are load-bearing.

**Aggregation is streaming.** `aggregate_stream` consumes fit results from a
generator and folds each into a running weighted sum, so peak memory is a
constant number of model states regardless of how many clients participate
— `FedAvgServer` (`fedbrew/servers/fedavg.py`). The server never holds a list
of client models.
Measured: **two model states** retained at 4, 16 and 64 clients alike — the
running sum, and one reference copy the accumulator keeps to fix the key set,
dtypes and shapes. Buffering would be one per participant.
`tests/test_aggregation_peak_memory.py` measures it.
A strategy that needs all results at once must say so; the four in `servers/`
do not.

**Artifacts are written every round, not at the end.** A run that is
cancelled, preempted or hits its wall clock never reaches the save in
`runner.run()`. Before this, such a run left a checkpoint at round *N* beside
no metrics at all, so `--resume-latest` had nothing to replay and the resumed
run's CSV started mid-experiment, in `run_fl_loop` (`fedbrew/core/loop.py`).

**Each client is visited once per round even when due for several splits.**
The evaluation plan groups by client, not by split, so a client due for both
`val` and `test` is loaded once — `_round_evaluation_plan`
(`fedbrew/core/loop.py`). Chapter 11 explains why
that matters at FEMNIST scale.

## 3. The six registries

`fedbrew/core/registry.py`. Names are registered once and never overwritten;
registering a name twice raises, naming both origins. `register_builtin_components()`
is the whole built-in set; an extension a config names registers into the
same six, and every name carries where it came from (chapter 12).

| Registry | Config key that selects from it | Count |
| --- | --- | --- |
| `server_strategies` | `server.strategy` | 9 |
| `client_updates` | `client.update_rule` | 9 |
| `models` | `model.name` | 8 |
| `datasets` | `data.name` | 2 |
| `tasks` | `task.name` | 2 |
| `generators` | `dataset.name` | 8 |

`dataset.name` is a generator config's key, not a run config's: the first five
are selected by `fedbrew run` and the sixth by `fedbrew generate`.

Registered names:

```
server_strategies  fedavg fedavgm fedadam fedyogi fedadagrad fedopt
                   scaffold fedlalr centralized
client_updates     local_sgd fedavg centralized local_adamw fedprox
                   scaffold delta_sgd fedlalr fedavg_ft
models             mlp cnn small_cnn femnist_resnet18 openimage_shufflenet
                   tiny_gpt2 hf_causal_lm hf_causal_lm_lora
datasets           synthetic_classification manifest_dataset
tasks              classification causal_lm
generators         synthetic_classification mnist cifar10 femnist tiny_causal_lm
                   generic_sft hf_causal_lm_text oasst1_sft
```

A name appears in two registries when the same word means a different thing on
each side: `fedavg` is both a server strategy and a client update rule, and
`centralized` and `scaffold` likewise. They are separate objects; the config
selects each independently. Chapter 07 covers which pairings are legal — several
are enforced, because either half alone silently changes what runs.

`fedbrew/core/factory.py` turns a validated config into the four built objects.
It is where most `extra`-key defaults are applied, so a default that is not in
a dataclass is usually there.

## 4. The protocol

Six dataclasses, all in `fedbrew/core/protocol.py`, all `slots=True`. They are
the only things servers and clients exchange.

| Object | Direction | Carries |
| --- | --- | --- |
| `ClientInfo` | dataset → server | `client_id`, `num_examples`, `payload` |
| `RoundInfo` | loop → both | `round_id`, `total_rounds` (T, set by the loop), `payload`, and the round's accumulated `metrics` |
| `FitRequest` | server → client | `round_id`, `client_id`, `total_rounds` (copied from `RoundInfo` by the server), `payload` |
| `FitResult` | client → server | `round_id`, `client_id`, `num_examples`, `payload`, `metrics` |
| `EvalRequest` | server → client | `round_id`, `client_id`, `payload` |
| `EvalResult` | client → server | same shape as `FitResult` |

Everything algorithm-specific travels in `payload`. A SCAFFOLD client returns
its control-variate delta there; a FedLALR client returns momentum and
second-moment state; a plain FedAvg client returns only `model_state`. The
protocol objects themselves do not change when an algorithm is added, which is
the point of them. The exception is context every round has: `total_rounds`
travels on `RoundInfo` and `FitRequest` so that a client whose local step
depends on T, the rounds in the run, receives it with the request rather than
from its constructor. An object built outside a run carries `None` there.

`RoundInfo.metrics` is mutated in place by both the server and the loop. That
is how the fit-phase metrics and the evaluation aggregates end up in one dict —
and it is why `server.metrics` filters one group and not the other, since only
the server's own path passes through `filter_metrics`. Chapter 08 §4.3.

## 5. The four abstract bases

| Base | File | Must implement |
| --- | --- | --- |
| `ServerStrategy` | `fedbrew/servers/base.py` | `initialize`, `configure_round`, `aggregate`, `aggregate_stream`, `evaluate`, `save_state`, `load_state` |
| `ClientUpdate` | `fedbrew/clients/base.py` | `setup`, `fit`, `evaluate`, `get_state`, `load_state` |
| `TaskAdapter` | `fedbrew/tasks/base.py` | `build_model`, `build_dataloader`, `train_step`, `eval_step`, `compute_metrics`, plus the federated-state group below |
| `FederatedDataset` | `fedbrew/data/dataset.py` | `list_clients`, `get_client_data`, `get_client_metadata`, `get_global_data`, `get_metadata` |

`save_state` / `load_state` on both the server and the client are what make
checkpoints resumable: a run resumed from `latest.pt` restores the server's
optimizer state and each client's own state, not just the model. Chapter 09.

### 5.1 The federated-state group

`TaskAdapter` carries four methods that exist because **what gets federated is
not always the whole model** — `TaskAdapter` (`fedbrew/tasks/base.py`):

| Method | Answers |
| --- | --- |
| `get_federated_model_state` | which tensors leave the client |
| `load_federated_model_state` | how they are applied on arrival |
| `federated_model_state_metadata` | what scope they are — `full`, or an adapter |
| `federated_aggregation_weight` | this client's weight in the average |

Under LoRA only the adapter tensors move. `model_state_scope` rides along in
the metadata, and every built-in server -- `fedavg`, the FedOpt family,
`scaffold` and `fedlalr` -- runs one check on every fit result before folding
anything (`_compatible_model_state`, `fedbrew/servers/fedavg.py`): the scope
must equal its own, an adapter's base model, revision, adapter name and LoRA
config must equal its own (`validate_federated_state_metadata`), and the keys
and tensor shapes must equal its own state's (`validate_state_matches`). So a
full-model client cannot be aggregated into an adapter-scoped run, nor the
reverse, and a refused result leaves the server's state as it was. `scaffold`
and `fedlalr` once folded results without it (FINDINGS.csv `POST-F28`).
Chapter 06 covers the scopes; chapter 07 covers what each algorithm adds on
top.

## 6. Module ownership

Where a decision lives. This is the table to consult before adding code.

| Decision | Module |
| --- | --- |
| What keys a config may carry, and their defaults | `fedbrew/core/config.py` |
| Turning a config into built objects | `fedbrew/core/factory.py` |
| Which names exist | `fedbrew/core/registry.py` |
| The round order, and what is timed | `fedbrew/core/loop.py` |
| Cross-client metric aggregation | `_aggregate_client_split_metrics` (`fedbrew/core/loop.py`) |
| Process setup: seeds, determinism, device, threads | `fedbrew/core/runtime_setup.py`, `fedbrew/core/seeding.py` |
| CLI flags and their overrides | `fedbrew/core/runner.py` |
| What a run refuses, and how a refusal is reported | `fedbrew/core/refusal.py` for the type; `main` in `fedbrew/core/runner.py`, the one place it is caught |
| Preflight validation | `fedbrew/core/validation.py` |
| What is written, and in what format | `fedbrew/core/artifacts.py` |
| Checkpoint policy and best-metric direction | `fedbrew/core/checkpointing.py` |
| Early stopping | `fedbrew/core/divergence.py` |
| In-run history and running totals | `fedbrew/core/state.py` |
| Terminal output: what a line says | `fedbrew/core/logging.py` |
| Terminal output: what a line looks like | `fedbrew/core/console.py` |
| Run identity and provenance | `fedbrew/core/run_metadata.py` |
| Non-finite handling and metric filtering | `fedbrew/core/metrics.py` |
| Node-local staging | `fedbrew/core/data_staging.py` |
| Tensor helpers, norms, weighted accumulation | `fedbrew/core/torch_utils.py`, `fedbrew/core/federated_state.py` |

`fedbrew/core/metrics.py` is deliberately a leaf: it imports nothing from the artifact
layer, so a CLI can use `json_safe` without pulling in the writers.

`fedbrew/core/console.py` is the other deliberate leaf, and the only module
allowed to import rich, name a colour or emit an escape sequence. Everything
else -- the run reporter, preflight, `inspect-data` -- composes rows and tones
and hands them over, so the palette, the Dingbat marker set and the single
`isatty()` gate have one definition rather than one per command.
`tests/test_console_is_the_only_renderer.py` enforces it.

## 7. Client pooling

At FEMNIST scale there are thousands of clients and a round touches a few
dozen. Building every client object up front would be the dominant cost, so a
manifest dataset gets `LazyClientPool` (`fedbrew/clients/lazy_pool.py`): a
`Mapping` that constructs a client on first access and can release it again.

Which pool is used **follows from the dataset**, not from a config key: a
manifest dataset reads thousands of shards from disk and gets the lazy pool;
the in-process synthetic dataset is already in memory and does not. A
`client.lazy_clients` key existed and was removed for that reason — chapter 04
§10.

The loop releases each client it evaluates — `_release_client`
(`fedbrew/core/loop.py`) —
but deliberately does **not** release after fitting: evaluation runs
immediately afterwards and would rebuild what it just dropped.

## 8. What a run produces

Four files, written every round. Chapter 09 covers them in full.

```
<output_dir>/
  round_metrics.csv          one row per round, rewritten atomically
  client_metrics.csv         per-client evaluation; off unless per_client_csv
  client_update_metrics.csv  per-client fit diagnostics, appended
  run.json                   identity, config echo, results, provenance
  checkpoints/               numbered, plus latest.pt / best.pt if enabled
```

## For agents

### Paths

| Path | What it is |
| --- | --- |
| `fedbrew/core/loop.py` | the round loop; `run_fl_loop` at line 96 |
| `fedbrew/core/runner.py` | `run()`, the CLI, and the override application |
| `fedbrew/core/factory.py` | config → built objects |
| `fedbrew/core/registry.py` | the six registries; `register_builtin_components` is the whole built-in set |
| `fedbrew/core/protocol.py` | the six exchanged dataclasses |
| `fedbrew/servers/base.py` | `ServerStrategy` |
| `fedbrew/clients/base.py` | `ClientUpdate` |
| `fedbrew/tasks/base.py` | `TaskAdapter`, including the federated-state group |
| `fedbrew/data/dataset.py` | `FederatedDataset` |
| `fedbrew/clients/lazy_pool.py` | `LazyClientPool` |
| `fedbrew/core/state.py` | `ExperimentState` and the running history summaries |

### Commands

```bash
# Prove the registry lists and module paths in this chapter are current.
python -m pytest tests/test_docs_architecture.py -v

# Build every registered component without training, on the smoke config.
fedbrew run --config configs/dev/smoke.yaml --validate-only

# The seams themselves.
python -m pytest tests/test_lazy_client_pool_selection.py \
                 tests/test_auxiliary_state_aggregation.py \
                 tests/test_tied_weight_federation.py \
                 tests/test_round_timing.py
```

### Invariants

1. **Aggregation stays streaming.** `aggregate_stream` must consume its
   iterable exactly once and hold a constant number of model states — two,
   measured — never one per participant. A strategy that buffers every
   client's parameters changes the memory profile of every run that uses it.
2. **Artifacts are flushed every round.** Never move the write to the end of
   the run; a preempted job must leave replayable metrics.
3. **Registry names are registered once.** `register` raises on a duplicate.
   Do not add a name that already exists in the same registry.
4. **Algorithm-specific data travels in `payload`.** Do not add fields to the
   protocol dataclasses for one algorithm's state.
5. **`model_state_scope` must accompany every state that moves.** Every
   server refuses a scope mismatch through `_compatible_model_state`; a
   strategy that folds results without calling it lets a full-model client be
   averaged into an adapter run.
6. **The evaluation plan groups by client, not by split.** A client due for two
   splits is visited once. Preserve that when changing the schedule logic.
7. **`core/metrics.py` imports nothing from the artifact layer.** It is a leaf
   so CLIs can use it cheaply.

### Tests that guard this chapter

| Test | Claim |
| --- | --- |
| `tests/test_docs_architecture.py` | The registry name lists and every module path named here exist and match the code. |
| `tests/test_lazy_client_pool_selection.py` | The pool follows from the dataset, not a config key. |
| `tests/test_auxiliary_state_aggregation.py` | Algorithm state travels in `payload` and survives aggregation. |
| `tests/test_aggregation_peak_memory.py` | Aggregation retains the same bytes at 4, 16 and 64 clients, and the fit phase is a generator. |
| `tests/test_tied_weight_federation.py` | The federated-state group handles tied weights. |
| `tests/test_lora_adapter_federation.py` | Adapter-scoped state federates without the full model. |
| `tests/test_federated_state_compatibility.py` | §5: FedAvg, SCAFFOLD and FedLALR each refuse a result of the wrong scope, adapter identity, keys or shapes, and leave their state unchanged (`POST-F28`). |
| `tests/test_round_timing.py` | The phase timings match the phases described here. |
| `tests/test_checkpoint_no_duplicate_model.py` | `save_state` does not store the model twice. |

### Known failure modes

- **Adding a strategy that needs all fit results at once.** `aggregate_stream`
  hands over a generator. Materialising it works and silently multiplies peak
  memory by the participant count.
- **Assuming `fedavg` means one thing.** It is a server strategy *and* a client
  update rule, and a config sets them separately. Chapter 07 lists the pairings
  that are enforced.
- **Looking for a default in the dataclass and not finding it.** Most `extra`
  defaults are applied in `core/factory.py` at the `.get()` call, not in
  `core/config.py`.
- **Releasing a client after `fit`.** Evaluation runs next and would rebuild
  it. The loop releases after evaluation only.
- **Expecting `run.json` to hold a model.** It holds identity, config and
  results. Model state is in `checkpoints/`.
