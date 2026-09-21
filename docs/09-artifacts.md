# 09 — Artifacts

What a run writes, the `run.json` schema, checkpoints, the global run index,
and what a resume replays from.

Everything here is written **every round**, not at the end. A run that is
cancelled, preempted or hits its wall clock never reaches the final save, and
before this it left a checkpoint at round *N* beside no metrics at all — so
`--resume-latest` had nothing to replay and the resumed run's CSV started
mid-experiment.

## 1. The files

```
<output_dir>/
  round_metrics.csv          always
  client_metrics.csv         only with client_statistics.per_client_csv
  client_update_metrics.csv  only with client_statistics.per_client_csv
  run.json                   always
  checkpoints/               unless checkpointing.enabled is false
```

`DEFAULT_ARTIFACT_FILES` (`fedbrew/core/artifacts.py`) names all four possible metric
files; `runner._artifact_file_names` decides which a given config produces. A
default run writes **two**, because `per_client_csv` is `false` and gates both
per-client files together.

The `.jsonl` twins of the two per-client CSVs were removed: they held
byte-identical records in a larger format, and `results.json` held a third
copy. The readers still exist, so older runs stay loadable.

Chapter 08 documents every column in all three CSVs. This chapter covers how
they are written and what else the run leaves behind.

`client_metrics.csv` has a **fixed thirteen-column schema**, which is why a
resume compares against its header: a run whose column set changed produces a
fresh file rather than one whose columns mean different things in different
halves.

## 2. How the files are written

Two different strategies, for two different sizes.

**`round_metrics.csv` is rewritten in full every round, atomically.** It is one
row per round, so a full rewrite is cheap. The write goes to a sibling `.tmp`
and is `os.replace`d into position by `_atomic_text_writer`
(`fedbrew/core/artifacts.py`), so the visible file
is always either the previous complete version or the new complete one. That
matters because these files are now rewritten every round: a process killed
mid-write is no longer rare, it is how a preempted run normally ends.

**`run.json` is first written after the first completed round.** There is no
initial `run.json`: nothing is written before round 1, so a run that fails
while building its components or inside round 1 leaves none. After each
completed round the loop calls the runner's writer (`runner._run_json_writer`),
after the round's CSV rows and before its checkpoints commit, and it rewrites
the whole file with `status: "running"` and `final_round` at that round. When
the loop returns, `runner.run` writes the final record over it, with the
status the run ended in. `tests/test_run_json_is_written_after_each_round.py`
pins all three.

**`run.json` and every checkpoint are replaced the same way.** `run.json` goes
through `_atomic_text_writer`, and `round_NNN.pt`, `latest.pt` and `best.pt`
through `_save_atomically` (`fedbrew/core/checkpointing.py`): a sibling
`<name>.tmp`, flushed and `fsync`ed, then `os.replace`d over the target. Both
were written in place until `POST-F23`, and `torch.save` truncates first: on
2026-09-20 a time limit's kill landed inside a `latest.pt` write and left it at
zero bytes, 7235 rounds into a 10000-round run whose resume could then only
start again from round 1. A write that fails removes its temp file; one killed
outright leaves it, `clear_stale_temp_files` sweeps it at the next start, and
no checkpoint glob matches a `.tmp` name in between.

**A round's checkpoints are committed after its CSV rows and `run.json`.** A
resume replays `round_metrics.csv` up to the checkpoint's round, so a
checkpoint visible for a round the CSV does not hold cannot be continued. The
loop used to write the checkpoints first, and late in a long run the CSV
rewrite is most of each round, so a time limit's kill landed between the two
almost every time: on 2026-09-20 all ten SCAFFOLD points a 12 h limit stopped
had `latest.pt` one round ahead of their CSV. The checkpoints are now staged in
place — written, flushed and `fsync`ed as `.tmp`, so `checkpoint_sec` still
times the write — and `StagedCheckpoints.commit` renames them into place after
`run.json` (`POST-F24`). A kill before the commit leaves the previous round's
checkpoint beside a history that reaches it or one row past it; a resume drops
the rows after the checkpoint's round and recomputes them.

**The two per-client CSVs are appended.** At FEMNIST scale with `clients: all`
they are ~1.8M rows over 500 rounds. They used to be rewritten in full on
every round that wrote a checkpoint — which `save_last` makes every round — so
turning `per_client_csv` on meant bytes written that grew with the square of
the round count, for data that grew linearly.

The append is one buffered `write()` followed by `fsync`, so the exposure to a
kill is the width of that single call. Unlike the rename it is **not atomic**,
so the last row can be short if the process dies inside it. Both readers drop
an unparseable final row for exactly that reason.

## 3. `run.json`

Twenty top-level keys, in a deliberate order — `sort_keys` is off, because
the section order *is* the organisation.

| Key | Content |
| --- | --- |
| `run_id` | this run's identity |
| `created_at`, `started_at`, `finished_at` | UTC timestamps |
| `duration_sec` | across **every** attempt |
| `attempt_duration_sec` | this process only |
| `attempts` | how many processes have contributed |
| `resumed` | whether this run **continued** an earlier attempt |
| `resume_from` | which checkpoint it was told to continue from, taken or not |
| `num_rounds` | records present |
| `first_round`, `final_round` | the range those records cover |
| `status` | `completed`, or an early-stop status |
| `termination` | `null`, or the divergence verdict — chapter 08 §12 |
| `results` | `final_metrics`, key-sorted |
| `timing` | per-phase summary derived from the round history |
| `scale` | what the run actually did — §3.2 |
| `artifacts` | which files and checkpoints exist |
| `reproducibility` | §3.3 |
| `config` | the config echo |

**`num_rounds` counts records, which equals rounds run only when
`first_round` is 1.** A resume whose history cannot be read is refused rather
than taken, so on a run this version completed the two agree. A `run.json`
written before `POST-F10` was fixed can still show them disagreeing, which meant
that resume went ahead without its round history.

**`resumed` is what happened, not what was asked for.** It used to be
`bool(resume_from)`, which is a config value, so a run that asked to resume and
could not — checkpoint rejected, artifacts discarded, restarted from round 1 —
recorded `resumed: true`. Nothing on disk disagreed: `first_round` is 1 for a
taken resume too, because a taken resume replays `round_metrics.csv` from round
1 (`POST-F05`). A resume that cannot be taken is now refused before anything is
written (§5, `POST-F25`), so no `run.json` records one, and the
`resume_restart` key that described the restart is gone.

**A run that stopped early is not a failed run.** Without `status`, a run cut
short by divergence or a walltime kill would be indistinguishable from one that
finished, short of comparing `final_round` to `global_rounds`.

### 3.1 `results.final_metrics`

The **last** round's complete metrics dict, key-sorted — not the best round's.
Best-round selection lives in `artifacts.checkpoints.best_round_id`.

Non-finite values become `null` here. `json_safe` walks the record replacing
them, and `json.dumps` is called with `allow_nan=False` so anything that slipped
past is a loud failure rather than an invalid file. Python would otherwise emit
the bare tokens `NaN` and `Infinity`, which RFC 8259 does not define: Python
and `pandas.read_json` (since pandas 1.0) read them back, `jq` 1.6 turns `NaN`
into `null` and `Infinity` into the largest double, and Go, `serde_json` and
`JSON.parse` refuse the file outright. Chapter 08 §10.

### 3.2 `scale`

What the run did, which is the honest denominator for any cost claim.

```json
"scale": {
  "total_client_fits": 10,
  "total_client_evaluations": 10,
  "total_client_update_metric_records": 10,
  "client_update_metric_phase_counts": {"fit": 10},
  "total_client_train_examples_evaluated": 194,
  "total_client_test_examples_evaluated": 60,
  "total_client_examples_processed": 448,
  "unique_clients": 5
}
```

`unique_clients` is a **count, not the list**. At OpenImage's 13,771 clients the
list alone would outweigh every other section of this file, and the per-client
CSVs already hold the identities.

These are maintained as running totals by `ClientHistorySummary`
(`fedbrew/core/state.py`) rather than recomputed. `run.json` is rewritten every
round, and every helper here used to re-scan the whole history each time —
O(records so far) per round, so quadratic over a run.

### 3.3 `reproducibility`

Four to seven sub-blocks, whichever the run actually had. **Empty entries are
dropped rather than written as `null`**, so a missing key means the run never
had one, not that a lookup returned nothing.

| Block | Content |
| --- | --- |
| `code_state` | `git_commit`, `git_dirty`, `commit_source` — `"git"` in a checkout, `"archive"` for a release exported by `git archive`, where the commit comes from the stamp in `fedbrew/_build_info.py` and `git_dirty` is `null` because an extracted tree has nothing to diff against. `git_dirty` is `git status --porcelain --untracked-files=no`: whether a *tracked* file differs from the commit, not what the difference is. Below the table: what this identifies |
| `dataset` | which manifest the run read, and what it declares — §3.4 |
| `seeding` | `seed`, `deterministic`, `deterministic_warn_only`, `cublas_workspace_config`, and the torch/CUDA/cuDNN/numpy versions |
| `runtime` | what torch **actually held**: resolved device, `matmul_precision`, `cudnn_benchmark`, the determinism flags |
| `llm` | model and adapter provenance, when there is one |
| `federated_model_state` | `model_state_scope` and the parameter counts |
| `extensions` | one entry per `experiment.extensions` item: the path it resolved to, the SHA-256 of the file that was imported, and the names it registered. `code_state` covers the package's commit and nothing outside it, so this is what ties a run to the version of an out-of-tree component |

**What the code record identifies.** The commit, a tracked-dirty flag and the
SHA-256 of each extension entry file (`extensions`). That identifies the code
exactly only for a clean tracked checkout — `commit_source: "git"`,
`git_dirty: false` — and, for an extension, only the file that was imported.
It does not record the content of a dirty tracked file (`git_dirty: true` says
the code differs from the commit, not how), any untracked file, including an
untracked module the run imported, anything an extension file imports in turn,
or an edit to an extracted archive (`git_dirty: null`). A run meant to be
reproduced from its record starts from a clean checkout.

**`seeding` and `runtime` are deduplicated, and `runtime` wins.**
`seed_everything` reads torch's flags at seeding time, which is *before*
`configure_runtime` applies the performance block — so its `matmul_precision`
says `highest` on a run that trains at `high`. Keeping both would put a stale
value beside the effective one for the single key that changes every fp32
matmul. Any key present in both is removed from `seeding`
in `save_run_json` (`fedbrew/core/artifacts.py`), and only a key `runtime`
actually reports is given
up, so `configure_runtime`'s exception path cannot leave a flag recorded
nowhere.

`artifacts.checkpoints` carries **results, not policy** — which checkpoints
exist and which round won. The policy fields live in
`config.runtime.extra.checkpointing`, and `run.json`'s contract is that nothing
appears in it twice.

### 3.4 `reproducibility.dataset`

Written for every `manifest_dataset` run that is not `causal_lm` — the LLM runs
record the same ground under `llm`, down to `corpus_hash`. Built by
`build_dataset_provenance` (`fedbrew/core/run_metadata.py`).

| Key | What it settles |
| --- | --- |
| `manifest_path`, `manifest_resolved_path` | what the config asked for, and what that resolved to |
| `manifest_sha256` | the SHA-256 of the manifest file's bytes, including every question this table does not anticipate. It identifies the manifest, not the shards — below |
| `client_shard_format` | `split_v2` (train + eval + test per client) or, for an older shard set, `null` |
| `client_test_source` | where a client's test slice comes from — chapter 05 §4 |
| `client_splits` | the ratios the shards were cut at |
| `dataset_name`, `partition_strategy`, `partition_key`, `num_clients`, `num_classes`, `global_test` | what was partitioned, and how |
| `partition_parameters`, `seed` | what the strategy was *given*, and the seed it drew with. Without these two a run on a dataset regenerated at a different `alpha` under the same path recorded provenance identical to the first — chapter 05 §5.1 |
| `source`, `source_revision`, `source_split`, `source_num_clients`, `min_samples_per_client` | the corpus behind it. `source` is the reader that produced the tensors, not the one that was tried first: MNIST falls back from torchvision to a mirror of the idx archives when torchvision cannot fetch them, and a run on those bytes records the mirror |
| `reference` | what the generator said a run on this data is scored against — a reference optimum and the dials that produced it — copied verbatim; `null` for data with no known answer |
| `extensions` | which extensions generated the data, with each file's SHA-256; `null` for a built-in generator |

**The config recorded the manifest's path and nothing about its content**, and
the generated data is not in the repository. The same path holds a different
dataset before and after a regeneration, so a FEMNIST accuracy could not be
shown to have come from shards whose per-client test slice is disjoint from the
eval slice `best.pt` was selected on, rather than from an older set where
`global_test.pt` was a copy of that eval data. Those are the difference between
a test number and the validation number under another name.

**What this block identifies, and what it does not.** Four things, of
different strength:

| What | Recorded as | Identifies |
| --- | --- | --- |
| how the partition was generated | `dataset_name`, `partition_strategy`, `partition_parameters`, `seed`, `client_splits`, `source*` | what the generator was asked for; the shipped generators' partition tests check what it produces from that (chapter 05, its tests table) |
| the manifest | `manifest_sha256` | the manifest file exactly |
| what the manifest declares | the remaining keys | the client count, split semantics and test source the manifest states; `--validate-only` and `inspect-data` check the shards' structure against them (chapter 05 §6) |
| the shard bytes | nothing | — |

No digest covers the client roster or the shard files. A shard edited or
replaced in place, at the same row count, leaves `manifest_sha256` and every
key here unchanged, and no structural check compares samples, so a sample
copied into two clients' shards by hand is not detected either. The record
says which dataset the run was pointed at and what that dataset claims; that
the bytes are the ones the generator wrote rests on the generated data not
being edited afterwards. The LLM runs' `llm` block stands the same way: its
`corpus_hash` is a digest of the source corpus the generator read, not of the
token shards it wrote.

**A key the manifest does not declare is written as `null`, not omitted.** That
is the opposite of the rule for whole blocks above, and deliberately: a
manifest predating `client_shard_format` *is* a `split_v1` shard set, so "the
manifest did not say" has to be distinguishable from "nobody looked". A lookup
that failed outright is `manifest_error` instead, and is never fatal — a
provenance read must not lose a run that already holds a GPU.

`tests/test_dataset_provenance.py` guards it against a manifest the real
generator writes.

## 4. Checkpoints

```
checkpoints/
  round_001.pt  round_002.pt  ...   numbered, per checkpointing.interval
  latest.pt                          with save_last
  best.pt                            with save_best
```

Chapter 04 §7.3 covers the policy keys — including that the defaults invert
when the whole `checkpointing` block is absent.

**A checkpoint no longer stores the model twice.** It used to carry the model
state both at the top level and inside the server state; removing the duplicate
roughly halves a checkpoint that carries no client states.
`tests/test_checkpoint_no_duplicate_model.py`
guards it.

A checkpoint carries enough to continue, not just to evaluate:

| Contents | Why |
| --- | --- |
| model state | the weights |
| server state | optimizer moments, control variates — otherwise a resumed FedAdam restarts its second moment |
| client states | per-client state for the rules that keep one |
| RNG state | Python, numpy, torch and per-device CUDA — §5 |
| round metrics | so the resumed run's history is continuous |

`best.pt` is selected on `checkpointing.best_metric`, which must be a `val_` or
`personal_val_` metric — selecting on a test metric makes the reported test
score optimistically biased, and it is refused at config load. Direction is
derived from the metric's name, never configured. Chapter 08 §11.

**A round whose selection metric is not finite is skipped, and says so once.**
NaN used to be *accepted* as a first observation — nothing had been seen, so
anything won — and then froze `best.pt` for the rest of the run, because every
later comparison is `x < nan` or `x > nan` and both are False in `min` mode and
`max` mode alike. The run reported `best_metric_value: nan`, `best_round_id:
1`, and a `best.pt` from the round that produced the NaN. That input is a
designed output elsewhere: `_overflow_safe` answers NaN for a statistic that
cannot fit (chapter 08 §7, `POST-F02`), and `val_loss_avg` goes through it. A
NaN arriving after a real value was always rejected correctly, so it was the
first observation alone. Infinities are refused the same way, since `+inf`
beats every later accuracy and `-inf` every later loss.

## 5. Resume

Two flags: `--resume-from <path>` and `--resume-latest`, which picks
`checkpoints/latest.pt` or the highest-numbered checkpoint.

**Resume restores RNG position, not just weights.** The checkpoint carries the
Python, numpy, torch and per-device CUDA RNG states, so a resumed run continues
the stream the interrupted one was in rather than restarting it.

A checkpoint written before `rng_state` existed makes the run **warn** rather
than fail, in `_initialize_or_resume` (`fedbrew/core/loop.py`): continuing on a
fresh stream is a real
divergence from the interrupted run, and it is said out loud rather than
inferred later.

**Resuming SCAFFOLD from `best.pt` is refused.** SCAFFOLD's server control
variate is *defined* as the mean of the clients' — `c = (1/N) Σ cᵢ` — and every
local step is corrected by `c - cᵢ`. `best.pt` is written without client state
on purpose (§4, and `tools/strip_checkpoint_client_states.py` does the same to
files already on disk), so resuming from one would restore `c` while every `cᵢ`
resets to zero. The correction becomes `+c` for every client, both sides then
move by the same per-round increments, and the gap never closes. Measured on a
four-client run checkpointed at round 3 and resumed for three more: the
residual `‖c - mean(cᵢ)‖` was `4.8e-08` from `latest.pt` and `0.4668` from
`best.pt` — the whole of `‖c‖` — and the final models differed by 7.4% of the
model norm. The run completed and reported success either way, which is why
`_refuse_a_half_restored_resume` raises instead.

The refusal is narrow. It fires only for a strategy that declares
`coupled_client_state` (SCAFFOLD is the only one) and only when the checkpoint
really carries a non-zero value under one of those keys, so a FedAvg resume
from `best.pt` is unaffected and so is a round-1 SCAFFOLD checkpoint whose `c`
is still zeros. Resume from `latest.pt`, or `--resume-latest`.

**A resume refuses a changed hyperparameter.** Every `load_state` used to
write the checkpoint's value over the one the config had just built —
`self.beta1 = float(state.get("beta1", self.beta1))` and a dozen like it — so a
run continued at the old value while `run.json` recorded the new one, and its
own record of itself was false. The packed SLURM scripts pass `--resume-latest`
whenever a `latest.pt` exists, so editing a config and resubmitting into the
same directory is one command.

`refuse_a_reconfigured_resume` compares the two and raises, naming every key
that disagrees and both values. It covers **every configured setting a
checkpoint carries**, not only the ones something restores. That reaches only
as far as what `save_state` and `get_state` write, so for every shipped server
strategy and client update rule the settings it is built with are held to being
written as well:

| | Keys | Was |
| --- | --- | --- |
| Restored | `aggregation_weighting`; FedOpt's five; FedLALR's `epsilon`; FedProx's `proximal_mu`; `local_adamw`'s and `fedlalr`'s betas and epsilon; `delta_sgd`'s five; `fedavg_ft`'s two; `update_mode`, `frozen_gradient_weighting`; `base_seed`, `train_shuffle`, `eval_shuffle`, `drop_last`, `max_local_steps` | The checkpoint silently outranked the config (`P10-F14`) |
| Checkpointed, restored by nothing | `learning_rate`, `local_iterations`, `batch_size`, `eval_batch_size`, `momentum`, `weight_decay`, `nesterov`, `learning_rate_schedule`, `min_learning_rate`, `client_id` | The edited config silently won from the resumed round on (`POST-F04`) |
| Checkpointed since `POST-F07` and `POST-F08`, restored by nothing | `participation_rate`, `seed` — every server strategy; `max_grad_norm` — `FedAvgClient`, `TorchDeltaSGDClient` | Never checkpointed, so no comparison could see them: an edited value was taken silently, and `run.json` recorded it for the whole run |
| Exempt | `num_examples`, `metrics` | Measured rather than configured; and which columns a run writes, which the CSV cursor already handles |

Both halves are the same contract — this chapter's, and chapter 10's — read in
opposite directions, which is why refusing only the first half would have been
worse than refusing neither: it reads as coverage.

`total_rounds` is the one conditional. It is an input to `_round_learning_rate`
and to nothing else, and that method returns before reading it when the
schedule is `constant` — so extending a constant-schedule run changes no number
a client computes, while `global_rounds: 500` → `1000` under `cosine`
re-anneals every remaining round. Compared under `cosine`, not under
`constant`.

Refused rather than resolved in either direction, because neither is supported:
picking the config would change an experiment mid-run, and picking the
checkpoint is the other half of the same thing. It is the call
`_refuse_a_foreign_seed` already makes for `experiment.seed`. State a run
*learns* — SCAFFOLD's control variate, FedOpt's moments, a client's
`num_examples` — still restores, which is what a resume is for.

The message names both ways forward, and names ones that work. Continuing the
same experiment in a new directory means **copying the run directory** and
resuming inside the copy: `--resume-from` pointed at a checkpoint beside an
empty `output_dir` finds no `round_metrics.csv`, and is refused. Running the new settings means starting from round 1 — there is
no warm start, and no flag loads a checkpoint's weights under a different
configuration.

**A resume that cannot be taken is refused, and nothing on disk changes.** A
resume replays `round_metrics.csv` up to the checkpoint's round;
`round_metrics_gap` (`fedbrew/core/artifacts.py`) describes why it cannot, or
returns `None`. When it cannot, the loop used to delete the attempt —
checkpoints, the CSVs, `run.json` — print one line, and start again from round
1, which in a batch job turned a stopped 10000-round run into a fresh one with
nothing left to decide from. It now raises before anything is written, stale
`.tmp` files included, and names the two ways forward: resume from an earlier
checkpoint the CSV reaches, or move the directory aside and start over
(`POST-F25`).

**A resume replays `round_metrics.csv`.** The
per-client CSVs are resumed by cursor, and the cursor is invalidated if the
file's header no longer matches the column set the run would write — so a
config change between attempts produces a fresh file rather than a CSV whose
columns mean different things in different halves. A per-client CSV that cannot
be read refuses the resume, naming the file and the line, because the resumed
run rewrites both files from the rounds it replays and would otherwise delete
every earlier round's rows (`POST-F10`). A torn final row is the one exception:
an interrupted append leaves exactly that, so it is dropped with a warning.

**A run refuses a foreign seed.** If `output_dir` already holds a `run.json`
with a different `experiment.seed`, the run stops
(`runner._refuse_a_foreign_seed`). Replicate seeds are the only way to put a
dispersion on a comparison, and nothing in the output path distinguishes them
unless the caller puts it there — otherwise two seeds of one arm resolve to one
directory, the second overwrites the first's `run.json`, appends to its
`round_metrics.csv`, and if the checkpoint is picked up continues the first
seed's model while reporting the second seed's config. The result is a
directory whose contents cannot be attributed to either run, and the "3-seed
spread" computed from it is not one.

It is an error rather than a silent path rewrite: appending `seed_N`
automatically would move every existing run's output location.

**A fresh start refuses to replace a finished run of another config**
(`runner._refuse_to_replace_a_finished_run`, `POST-F22`). The seed check
guards the seed and nothing else, so at the same seed a fresh start replaced
whatever finished run `output_dir` held — `run.json`, `round_metrics.csv`, the
checkpoints — with a run of different settings, and nothing on disk said it
had: a directory named for one learning rate held another's curve. If
`run.json` says `completed`, `diverged` or `stalled` and its recorded config
differs from this run's, the run stops and names every differing key with both
values. `runner.config_differences` makes the comparison, the way `run.json`
writes a config:

- A key recorded on one side only is not compared, as a checkpoint written
  before a setting existed still resumes.
- The keys that name a run or say how it was launched are not configuration:
  `experiment.run_id`, `name`, `tags`, `notes` and `output_dir`, and
  `runtime.extra`'s `quiet`, `verbose`, `no_rich`, `print_every`,
  `resume_from` and `resume_latest`.
- Under data staging, `data.path` names a per-job copy of the same manifest and
  is not compared either.

The same config reruns in place, replacing a run with the same experiment. A
resume is not a fresh start and has its own check, above. A `run.json` still
saying `running` — what a crash leaves, as well as a live run — restarts in
place as before.

## 6. The global run index

`<root_output_dir>/runs_index.jsonl` — one line per run, appended, so a sweep
has a single file to read rather than *N* directories to walk.
`tests/test_runs_index_json_validity.py` guards that every line stays valid
JSON.

## 7. Reading the artifacts

```bash
# Columns a run produced.
head -1 <output_dir>/round_metrics.csv | tr ',' '\n'

# Status, termination and final metrics.
python -c "import json;r=json.load(open('<output_dir>/run.json'));\
print(r['status'], r['termination'], len(r['results']['final_metrics']))"

# A whole sweep.
cat <root_output_dir>/runs_index.jsonl | jq -s 'map({run_id, status})'

# Size and status reports.
fedbrew report --run-dir <output_dir>
```

`round_metrics.csv` loads directly with `pandas.read_csv`; non-finite values are
written as `nan` / `inf`, which pandas parses. `run.json` uses `null` instead,
so it stays valid JSON for any reader in any language.

## For agents

### Paths

| Path | What it owns |
| --- | --- |
| `fedbrew/core/artifacts.py` | every writer and reader |
| `fedbrew/core/artifacts.py` | `DEFAULT_ARTIFACT_FILES` |
| `fedbrew/core/artifacts.py` | `_CLIENT_EVALUATION_FIELDS` and `_ROUND_TIMING_FIELDS` — the two fixed column schemas |
| `fedbrew/core/artifacts.py` | `_atomic_text_writer`, `clear_stale_temp_files` |
| `fedbrew/core/artifacts.py` | `_append_csv_rows` — the append path, and why it is not atomic |
| `fedbrew/core/artifacts.py` | `round_metrics_gap` |
| `fedbrew/core/state.py` | `ExperimentState`, `ClientHistorySummary` |
| `fedbrew/core/checkpointing.py` | policy, pruning, best-metric direction, `_save_atomically`, `StagedCheckpoints` |
| `fedbrew/core/run_metadata.py` | provenance gathering |
| `fedbrew/core/runner.py` | `_refuse_a_foreign_seed`, `_refuse_to_replace_a_finished_run`, `config_differences` |
| `fedbrew/core/runner.py` | `_artifact_file_names` |
| `fedbrew/core/metrics.py` | `json_safe` |

### Commands

```bash
# Prove this chapter's schemas and file lists are current.
python -m pytest tests/test_docs_artifacts.py -v

# The artifact and resume guards.
python -m pytest tests/test_run_provenance.py \
                 tests/test_dataset_provenance.py \
                 tests/test_runs_index_json_validity.py \
                 tests/test_run_json_resume_accounting.py \
                 tests/test_resume_metrics_continuity.py \
                 tests/test_resume_rng_state.py \
                 tests/test_client_csv_append.py \
                 tests/test_checkpoint_no_duplicate_model.py \
                 tests/test_resume_is_all_or_nothing.py \
                 tests/test_best_checkpoint_selection_is_not_frozen.py \
                 tests/test_resume_refuses_a_changed_hyperparameter.py \
                 tests/test_resume_provenance_is_what_happened.py \
                 tests/test_checkpoint_writes_survive_a_kill.py \
                 tests/test_a_round_commits_its_checkpoint_last.py \
                 tests/test_a_refused_resume_changes_nothing.py \
                 tests/test_generator_atomic_write.py
```

### Invariants

1. **Artifacts are flushed every round.** Never move a write to the end of the
   run.
2. **`round_metrics.csv`, `run.json` and every checkpoint are replaced
   atomically.** `.tmp` plus `os.replace`; never `open("w")` or `torch.save`
   over the live file. **A round's checkpoints are committed last**, after its
   CSV rows and `run.json`.
3. **The per-client CSVs are appended, and their last row may be short.**
   Readers must tolerate an unparseable final row.
4. **`json_safe` runs on every record reaching JSON**, paired with
   `allow_nan=False`.
5. **Nothing appears twice in `run.json`.** Policy lives in the config echo;
   `artifacts.checkpoints` carries results only; `seeding` gives up any key
   `runtime` also reports.
6. **`unique_clients` is a count.** Do not write the client list.
7. **A checkpoint stores the model once.**
   **`best.pt` is never selected on a value that cannot be compared.** A
   non-finite candidate is skipped; accepting one freezes the selection.
8. **Resume restores RNG state**, and warns loudly when a checkpoint has none.
   **A resume that cannot be taken is refused, and changes nothing on disk.**
   Never delete an attempt to start over; that is the reader's decision.
9. **A resume restores both halves of a coupled state or neither.** SCAFFOLD's
   `c` and the clients' `cᵢ` are defined in terms of each other; a checkpoint
   carrying one without the other is refused, not half-restored.
   **A restored hyperparameter that disagrees with the config is refused too.**
   The checkpoint must never quietly outrank `run.json`.
10. **A foreign seed is refused, not renamed.** Silently rewriting the path
    would move every existing run's output location.

### Tests that guard this chapter

| Test | Claim |
| --- | --- |
| `tests/test_docs_artifacts.py` | The file list, `run.json` key set and fixed CSV schemas here match the writers. |
| `tests/test_run_provenance.py` | The `reproducibility` blocks are recorded. |
| `tests/test_dataset_provenance.py` | §3.4: a run records the manifest it read, and the FEMNIST configs describe the format the generator writes. |
| `tests/test_runs_index_json_validity.py` | Every index line is valid JSON. |
| `tests/test_run_json_resume_accounting.py` | `attempts`, `resumed`, `first_round` across attempts. |
| `tests/test_run_json_is_written_after_each_round.py` | §2: no `run.json` exists before round 1 or after a run that fails inside it; it is rewritten with `status: "running"` after each completed round and replaced by the final record at the end. |
| `tests/test_resume_metrics_continuity.py` | A resumed run's history is continuous. |
| `tests/test_resume_rng_state.py` | RNG position is restored. |
| `tests/test_client_csv_append.py` | The per-client CSVs append rather than rewrite. |
| `tests/test_checkpoint_no_duplicate_model.py` | The model is stored once. |
| `tests/test_resume_is_all_or_nothing.py` | §5: a SCAFFOLD resume that would strand the control variate is refused, and the narrower cases are not. |
| `tests/test_best_checkpoint_selection_is_not_frozen.py` | §4: a non-finite selection metric is skipped rather than made the run's best. |
| `tests/test_resume_refuses_a_changed_hyperparameter.py` | §5: a resume that would change a restored hyperparameter is refused, and learned state still restores. |
| `tests/test_resume_provenance_is_what_happened.py` | §3.1: `resumed` is what the run did, and a refused resume says so under its own key. |
| `tests/test_checkpoint_writes_survive_a_kill.py` | §2: a write killed before its rename leaves the previous checkpoint or `run.json` readable. |
| `tests/test_a_round_commits_its_checkpoint_last.py` | §2: a round's checkpoints are staged while its CSV rows and `run.json` are written, and a kill in either write is resumable. |
| `tests/test_a_refused_resume_changes_nothing.py` | §5: a resume the history cannot back is refused with every file byte-identical, and names the ways forward. |
| `tests/test_client_history_summary.py` | The running totals behind `scale`. |
| `tests/test_report_run_size.py`, `tests/test_report_run_status.py` | `fedbrew report`. |
| `tests/test_cleanup_runtime_artifacts.py` | `fedbrew cleanup` preserves what it is told to. |
| `tests/test_non_finite_aggregation.py` | Non-finite values reach disk as `null`. |

### Known failure modes

- **Expecting four metric files.** Two, unless `per_client_csv` is on.
- **Reading `results.final_metrics` as the best round.** It is the last round.
  `artifacts.checkpoints.best_round_id` is the best.
- **Trusting `num_rounds` as rounds run.** Equal only when `first_round` is 1.
  A resume without its history is refused now, so a disagreement appears only in
  a `run.json` written before `POST-F10` was fixed.
- **Parsing `run.json` with a strict JSON reader and hitting `NaN`.** It cannot
  happen — that is what `json_safe` is for. The CSVs *do* carry `nan`/`inf`.
- **Leaving `experiment.output_dir` empty.** Refused at config load
  (`_validate_output_dir`, `fedbrew/core/config.py`). `Path("")` is `Path(".")`,
  so the whole artifact set would land in the working directory — for a batch
  job, whatever the submit script last `cd`'d to. Write `"."` if that is
  actually what you want.
- **Pointing two seeds of one arm at one `output_dir`.** Refused. Use
  `--use-run-subdir` or a distinct directory.
- **Resuming across a config change.** Refused: every configured setting the
  checkpoint carries is compared, restored or not (§5). The per-client CSV
  cursor is invalidated when the header no longer matches. What is still
  unguarded is *code*, and settings no checkpoint carries — the `data`,
  `model` and `evaluation` blocks among them. Do not change code or config
  between a run and its resume.
- **Reading a stale `matmul_precision` from `seeding`.** It is removed from
  that block when `runtime` reports it; `runtime` is the effective value.
