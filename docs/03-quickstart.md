# 03 — Quickstart

Generate a federated dataset, inspect it, validate a config, train, and read
the result. About a minute on CPU, no network.

The synthetic generator builds its data from a seeded linear teacher, so
nothing is downloaded. Everything below was run to produce the output shown.

## 1. Install

```bash
pip install -e .
```

Chapter 02 covers extras and offline installation. Nothing here needs them.

If `fedbrew` is not on your `PATH` — a checkout that has not been
`pip install`ed, which is a normal state for a working copy — every command
below also works as:

```bash
python -m fedbrew.cli.dispatch <subcommand> ...
```

That is the same entry point the console script calls, so the two are
interchangeable.

## 2. The four commands

```bash
# 1. Generate a small label-skewed federated dataset.
fedbrew generate --config data/configs/synthetic_label_skew.yaml

# 2. Check the manifest is valid and see the per-client split sizes.
fedbrew inspect-data data/generated/synthetic_label_skew/manifest.json

# 3. Validate the experiment config without training.
fedbrew run --config configs/dev/synthetic_label_skew.yaml --validate-only

# 4. Train for two rounds.
fedbrew run --config configs/dev/synthetic_label_skew.yaml --rounds 2
```

Each takes about ten seconds, nearly all of it importing torch. The training
step is the longest only because its first round pays a one-time warm-up; see
step 4.

### Step 1 — generate

```
─────────────────────────────────── GENERATE ───────────────────────────────────
❙ ❖ partition  label_skew, 5 clients  labels_per_client=2
❙ ❖ clients    120 examples  22-31 each, mean 24.0
❙ ❖ labels     3 classes  28-47 examples each
Generated   synthetic_classification
Clients     5
Train rows  120
Test rows   30
Output: data/generated/synthetic_label_skew
```

150 examples: 120 partitioned across five clients, 30 held back as the
server's pooled test shard. Chapter 05 covers the manifest this writes and the
four partition strategies.

The three lines above the summary are the contents of `partition_stats.json`,
which this command has always written and never shown. **22-31 examples each**
is the label skew, and it is the number to check before training on a
generated dataset: a partition that left a client with nine examples is a
different experiment from the one the config describes. The `partition` line
names the parameter actually in force — a config may carry `alpha`,
`labels_per_client` and `sigma` together, and only the one belonging to the
configured strategy does anything.

### Step 2 — inspect

```
─────────────────────────── FEDERATED DATASET REPORT ───────────────────────────
Manifest  data/generated/synthetic_label_skew/manifest.json
❖ VALID DATASET
0 errors  0 warnings

Dataset               synthetic_classification
Partition             label_skew
Clients               5
Total examples        120
Input dimension       6
Client splits         train=0.8, eval=0.2
Train examples        total=97, min=18, max=25, mean=19.40
Eval examples         total=23, min=4, max=6, mean=4.60
Label counts          {'0': 47, '1': 45, '2': 28}
Partition statistics  data/generated/synthetic_label_skew/partition_stats.json
Client statistics     data/generated/synthetic_label_skew/client_stats.csv
Client shards         data/generated/synthetic_label_skew/shards
──────────────────────── DATASET READY FOR EXPERIMENTS ─────────────────────────
```

The 120 client examples split 80/20 into 97 train and 23 validation. The
per-client spread — 18 to 25 train examples — is the label skew: clients do not
hold equal shares. That `min`/`max` gap is what makes the two averages in
chapter 08 differ.

### Step 3 — validate

```
────────────────────────────────── PREFLIGHT ───────────────────────────────────
❙ ❖ load config              configs/dev/synthetic_label_skew.yaml
❙ ❖ components
❙ ❖ experiment
❙ ❖ server
❙ ❖ client
❙ ❖ task
❙ ❖ data
❙ ❖ model
❙ ❖ evaluation
❙ ❖ runtime
❙ ❖ algorithm compatibility
...                                 (the plan header, as in step 4)
───────────────────────────────── READY TO RUN ─────────────────────────────────
0 errors  0 warnings
```

Each check settles as it lands, and the plan header — the same one step 4
prints — is what a clean preflight settles into: it is the result the checks
just proved. A failing check withholds it, because a plan printed under an
error describes something that will not happen.

That is a clean checkout. Run step 3 again *after* step 4 and the verdict
changes, because the output directory is no longer empty:

```
────────────────────────────────── PREFLIGHT ───────────────────────────────────
❙ ❖ load config              configs/dev/synthetic_label_skew.yaml
❙ ❖ components
❙ ❢ experiment               1 warning
  ❢ experiment.output_dir_not_empty  Output directory already contains files:
                                     outputs/synthetic_label_skew
      Use --use-run-subdir or choose a fresh output directory for long runs.
...                                 (the remaining checks, all clean)
...                                 (the plan header)
─────────────────────────── READY — REVIEW WARNINGS ────────────────────────────
0 errors  1 warning
```

Both verdicts are a pass. What matters is `0 errors` and a `READY` line; the
warning is telling you that a second run into the same directory would mix two
experiments' artifacts. `--use-run-subdir` writes under `output_dir/run_id`
instead.

When a check does fail, every remaining check still runs — preflight exists so
you can fix everything in one pass — and the verdict at the bottom repeats every
error, so you do not have to scroll back through the rail to collect them.

`--validate-only` loads the config, builds every component, runs the whole
preflight, and exits without training. It is the cheapest way to check a config
change, and worth running before any long job, because step 4 does not run it:
an ordinary run validates the config and stops at the first refusal, and prints
none of the warnings above (chapter 04 §1).

### Step 4 — train

The run prints its plan before it does anything:

```
─────────────────────────────── EXPERIMENT PLAN ────────────────────────────────
Experiment                          synthetic_label_skew
Run                                 20260903_104830_synthetic_label_skew_seed42
Output                              outputs/synthetic_label_skew
Device                              cpu
Seed                                42
Determinism                         on
Matmul precision                    highest

data
Dataset                             manifest_dataset
Manifest                            data/generated/synthetic_label_skew/manifest.json
Clients                             5

federation
Strategy                            fedavg
Rounds                              2
Participation rate                  1
Clients per round                   5

algorithm
Update rule                         local_sgd
Task                                classification
Model                               mlp
Local iterations                    1
Batch size                          32
Learning rate                       0.01
Learning rate schedule              constant
Momentum                            0
Weight decay                        0

metrics
evaluation.train                    every 10 rounds, participating clients
evaluation.val                      every 5 rounds, all clients
evaluation.test                     every 10 rounds, all clients
evaluation.central_test             every 10 rounds
evaluation.model_scope              global
checkpoint selects on               val_accuracy_sample_weighted_avg
divergence watches                  fit_loss
Columns                             15 of 43 listed; --verbose lists them all
train_loss_sample_weighted_avg      Cross-entropy on the selected clients' train
                                    data, pooled over examples — the largest
                                    clients move it most.
central_test_loss                   Average loss of the global model on the
                                    complete global test set.
...                                 (thirteen more columns, each glossed)
```

Every value there is the *resolved* one — after the config's defaults, after
any CLI override, and after `device: auto` became something concrete. The
`metrics` block names the columns this run will write and glosses each one;
`--verbose` lists all 43 instead of the fifteen shown. Amber marks the five
things a reader would otherwise assume were not the case: a `matmul_precision`
other than `highest`, `deterministic_warn_only`, an output directory that
already holds files, a resumed run, and components loaded from outside the
package through `experiment.extensions`. This run has none of them.

Then the rounds. The last line is live -- one line, redrawn in place, that
never scrolls; every block above it is a round that finished, with its bar
frozen where it landed:

```
─────────────────────────────────── TRAINING ───────────────────────────────────
  ━━━━━━━━─────────   ✦ round 1/2   5/5 clients
  ━━━━━━━━─────────     round 1/2   5/5 clients

  loss
    train      —                       1.0873
    train      sample_weighted_avg     1.0922
    validation sample_weighted_avg     1.1829
    validation avg                     1.2144
    test       sample_weighted_avg     1.1921
    test       avg                     1.1966
    central    —                       1.1921

  accuracy
    train      —                       38.14%
    train      sample_weighted_avg     38.14%
    validation sample_weighted_avg     30.43% ✦
    validation avg                     26.00%
    test       sample_weighted_avg     23.33%
    test       avg                     26.57%
    central    —                       23.33%

  client spread
    test       accuracy_std            17.83%
    test       accuracy_min             0.00%
    test       accuracy_worst10         0.00%

23 more columns written; --verbose shows them

  ━━━━━━━━━━━━━━━━━   ✦ round 2/2   5/5 clients   4.34s/round
  ━━━━━━━━━━━━━━━━━     round 2/2   5/5 clients

  loss
    train      —                       1.0865
    train      sample_weighted_avg     1.0914
    validation sample_weighted_avg     1.1823
    validation avg                     1.2137
    test       sample_weighted_avg     1.1914
    test       avg                     1.1960
    central    —                       1.1914

  accuracy
    train      —                       38.14%
    train      sample_weighted_avg     38.14%
    validation sample_weighted_avg     30.43% ✦
    validation avg                     26.00%
    test       sample_weighted_avg     23.33%
    test       avg                     26.57%
    central    —                       23.33%

  client spread
    test       accuracy_std            17.83%
    test       accuracy_min             0.00%
    test       accuracy_worst10         0.00%

23 more columns written; --verbose shows them

───────────────────────────── EXPERIMENT COMPLETE ──────────────────────────────
Finished successfully.
Rounds completed  2
Wall clock        4.43s over 2 timed rounds (2.22s/round)
Metrics           outputs/synthetic_label_skew/round_metrics.csv
Artifacts         outputs/synthetic_label_skew
```

**Two rounds of a 3-class MLP on 97 training examples learns nothing.** Accuracy
near 23% on three classes is chance. The point is that the pipeline runs end to
end and produces the artifacts below.

**Four groups, coloured by split.** A round is a list rather than a table, so
`--verbose` grows it in place — the same headings in the same order, with
every column under each instead of the curated few. `loss` and `accuracy` hold
the two averages of each split; `client spread` holds the statistics that
describe how clients *differ* rather than how they did, which is why
`accuracy_std` sits there and not under `accuracy`; `algorithm` holds the
per-round diagnostics, and appears only when a run emits some. The split label
is what a reader scans down for, so it carries the colour; the qualifier is
what the group heading does not already say, and a column with nothing further
to say — `fit_loss` under `loss` — shows an em dash rather than an empty cell.

**The gold ✦ marks the column `checkpointing.best_metric` selects on.** One per
block, on the number that decides whether this round became `best.pt`.

**The bar is frozen where each round landed**, so scrolling back through a long
run shows how far in each block was written. The last line is the live one: the
same bar, plus the rate and the time remaining, redrawn in place. The client
count does not animate — `5/5` is a property of the round, decided when the
round was selected — so liveness comes from the pulse and the clock, which keep
moving on an LLM config where a round takes minutes and the bar advances once.
`--verbose` adds the within-phase count back, which matters at
`client_scope: all` on FEMNIST: 3,500 clients in one pass.

**Redirect it and the same information arrives flat.** No bar, no colour, no
redraw, no footer — and because there is no footer to carry them, the header
line picks up the examples, the duration and the estimate:
`round 1/2   5/5 clients   97 examples   4.86s   ETA 4.86s`.

**Round 1 is far more expensive than round 2, and that is warm-up, not work.**
Both rounds do the same arithmetic on the same five clients; round 1 is almost
entirely `fit_sec` paid once, as one-time warm-up. So `Wall clock`, which sums the
per-round timings, reports a mean of 2.22s/round for a steady-state round that
costs a small fraction of that, and the `4.34s/round` the footer showed at
round 2 is the same warm-up seen from the other side. Never extrapolate a job
budget from the mean of a two-round run; chapter 11 covers the per-round cost
that actually scales. The absolute values here are one CPU login node on
2026-09-03 and will not match yours.

**Only evaluation rounds report.** Both rounds above did, because every
schedule pins round 1 and the final round; a 500-round run at the default
`every: 10` prints 50 blocks rather than 500, and the rounds it skips are the
ones whose evaluation numbers would be the previous round's unchanged. Which
rounds those are is read from `evaluation.*.every`, not from which columns a
round happened to produce. `--verbose` reports every round, and
`--print-every N` reports rounds 1, N, 2N, … and the final round instead of
the evaluation rounds. Neither changes what is recorded: every round is still
evaluated on its schedule and written to `round_metrics.csv`.

Each round prints the same curated set the plan header glossed, not the full
column set — the trailing line says how many more went to the CSV. The set is
derived from your `client_statistics` settings, so `client spread` carries
`accuracy_worst10` at the default `worst_percent: 10` and `accuracy_worst2p5`
if you set 2.5.

Nothing about those labels is written down: the split, the group and the
qualifier are derived from the column name, against the same two definitions
that decide which columns exist at all. A name neither of them knows prints
under an `unclassified` heading in a colour used nowhere else, rather than
under a plausible one — `tests/test_round_block.py` fails if any column a run
can actually produce lands there.

While a round is running, an interactive terminal shows one line, redrawn in
place. A redirected stream does not get it: a carriage-return redraw in a log
file is one line per update, and the round block that follows carries the same
facts settled — including the duration and the estimate, which move onto the
block header when there is no footer to hold them.

`--quiet` replaces all of it with one line when the run ends:

```
completed 2 rounds in 4.43s -> outputs/synthetic_label_skew
```

which names the outcome because the exit code cannot — a run stopped by the
divergence monitor exits zero on purpose, and its line says `diverged at round
47` instead.

A run that declines to start or continue — a config refused by validation, a
flag out of range, a dataset file it cannot read, a checkpoint it
will not resume from, a second seed in one output directory, a fresh start over
a finished run of another config — prints `RUN REFUSED` and the reason on stderr, whatever
`--quiet` says, and exits 2, the status argparse already uses when `fedbrew run`
refuses its own arguments. `--validate-only` exits 1 on a failing check instead,
a config that will not load included, and the two statuses mean different
things: 1 is preflight reporting problems in a config it was asked to inspect,
and 2 is a run that was asked to start and would not.

## 3. What the run wrote

```
outputs/
  runs_index.jsonl              <- one line appended per run, across all runs
  synthetic_label_skew/
    round_metrics.csv
    run.json
    checkpoints/
      best.pt  latest.pt  round_001.pt  round_002.pt
```

`runs_index.jsonl` is written beside the run directory, not inside it: it is
the index over every run under `outputs/`, so a second experiment appends to
the same file rather than creating its own. Chapter 09 owns its schema.

**Two metric files, not four.** `client_metrics.csv` and
`client_update_metrics.csv` are both gated on
`client_statistics.per_client_csv`, which is `false` by default. Chapter 08 §9.

### `round_metrics.csv`

One row per round, 52 columns:

```
round_id, num_clients, num_examples,          <- 3 identity columns
central_test_accuracy ... val_num_clients,    <- 43 metric columns, sorted
duration_sec, fit_sec, aggregate_sec,         <- 6 timing columns, appended
client_eval_sec, global_eval_sec, checkpoint_sec
```

The 43 metric columns are: 13 per split × 3 splits (`train`, `val`, `test`),
plus `central_test_{loss,accuracy}` and `fit_{loss,accuracy}`. Twelve of the
thirteen are aggregates; the thirteenth is `{split}_num_clients`, how many
clients they are over. Chapter 08 gives each one's formula.

`num_clients` in the identity block and `{split}_num_clients` are different
counts and both are wanted: the first is how many clients the server *selected*
to train this round, the second how many reported a non-empty split when it was
*evaluated*. Two configs can agree on the first and differ on the second.

### `run.json`

Twenty top-level keys. The five worth checking after any run:

| Check | Expect |
| --- | --- |
| `status` | `"completed"` |
| `termination` | `null` — a divergence detector would fill it |
| `results.final_metrics` | present and finite: no `NaN`, no infinity |
| `artifacts.checkpoints` | names `latest_checkpoint` and `best_checkpoint` |
| `reproducibility.code_state.git_commit` | recorded, with `git_dirty` and `commit_source` |

From the run above:

```json
"status": "completed",
"termination": null,
"artifacts": {
  "checkpoints": {
    "latest_checkpoint": "checkpoints/latest.pt",
    "best_checkpoint": "checkpoints/best.pt",
    "best_metric_value": 0.30434782608695654,
    "best_round_id": 1
  }
},
"reproducibility": {
  "code_state": {"git_commit": "<your checkout's commit>", "git_dirty": false, "commit_source": "git"},
  "seeding": {"seed": 42, "deterministic": true, "matmul_precision": "highest"}
}
```

`best_round_id: 1` with two rounds run is expected here: the model is not
learning, so round 1's validation accuracy was never beaten. Selection uses
`val_accuracy_sample_weighted_avg` — a validation metric, never a test one.
Chapter 08 §11.

`scale` records what the run actually did, which is the honest denominator for
any cost claim:

```json
"scale": {
  "total_client_fits": 10,
  "total_client_evaluations": 10,
  "total_client_examples_processed": 448,
  "unique_clients": 5
}
```

Ten fits over two rounds is five clients at `participation_rate: 1`.

## 4. Reading the results

```bash
# Columns this run produced.
head -1 outputs/synthetic_label_skew/round_metrics.csv | tr ',' '\n'

# The headline number per round.
cut -d, -f1,11 outputs/synthetic_label_skew/round_metrics.csv

# Status and final metrics.
python -c "import json;r=json.load(open('outputs/synthetic_label_skew/run.json'));print(r['status'],r['results']['final_metrics'])"
```

`round_metrics.csv` loads directly with `pandas.read_csv`. Non-finite values
are written as `nan` / `inf`, which pandas parses; `run.json` converts them to
`null` instead, so it stays valid JSON for any reader.

## 5. Next

| To | Go to |
| --- | --- |
| Understand what just ran | chapter 01 |
| Change a setting | chapter 04 |
| Use a real dataset | chapter 05 |
| Interpret a column | chapter 08 |
| Run on a cluster | chapter 02, chapter 11 |

`configs/mnist/fedavg.yaml` runs the same loop on real data.
`fedbrew generate --config data/configs/mnist_one_label.yaml` downloads MNIST
through torchvision if it is not already cached — which needs the `vision`
extra, `pip install -e ".[vision]"`, because torchvision is not in the core
install. Read chapter 02 §1 first if you installed a torch build matched to
your CUDA driver: that extra will replace it. Without network access, the
synthetic workflow above is the fallback.

It is not the same *size*: that config ships `global_rounds: 1000` over 1000
clients at `participation_rate: 1`, and measured 8.3s/round on CPU
(2026-09-02), so running it as shipped is a multi-hour job. Pass `--rounds 2`
first, the way step 4 does, and read chapter 11 before removing the flag.

## For agents

### Paths

| Path | What it is |
| --- | --- |
| `data/configs/synthetic_label_skew.yaml` | the generator config step 1 reads |
| `configs/dev/synthetic_label_skew.yaml` | the experiment config steps 3 and 4 read |
| `data/generated/synthetic_label_skew/` | what step 1 writes; gitignored |
| `outputs/synthetic_label_skew/` | what step 4 writes; gitignored |
| `configs/dev/smoke.yaml` | a smaller config for validation-only checks |
| `fedbrew/cli/dispatch.py` | the entry point behind both invocation forms |

### Commands

```bash
# Prove this chapter's commands and configs are real.
python -m pytest tests/test_docs_quickstart.py -v

# Run the whole quickstart end to end and check its artifacts.
# Marked, because it trains: excluded from the default suite.
python -m pytest -m quickstart

# The four steps, from a clean checkout.
pip install -e .
fedbrew generate --config data/configs/synthetic_label_skew.yaml
fedbrew inspect-data data/generated/synthetic_label_skew/manifest.json
fedbrew run --config configs/dev/synthetic_label_skew.yaml --validate-only
fedbrew run --config configs/dev/synthetic_label_skew.yaml --rounds 2
```

### Invariants

1. **The four commands must work from a clean checkout with no network.** That
   is the whole point of the chapter; `-m quickstart` runs them.
2. **`python -m fedbrew.cli.dispatch` is equivalent to `fedbrew`.** The console
   script is a thin entry point; do not add behaviour to one and not the other.
3. **A default run writes two metric files**, not four. Both per-client CSVs
   are gated on `client_statistics.per_client_csv`.
4. **Step 4 must stay short.** It is `--rounds 2` on 97 examples; do not make
   the quickstart depend on a config that takes minutes.
5. **The output shown here is real.** Regenerate it by running the commands,
   never by editing the chapter to match what you expect.

### Tests that guard this chapter

| Test | Claim |
| --- | --- |
| `tests/test_docs_quickstart.py` | Every command is a real subcommand with real flags, every config and path named exists, the artifact list matches `_artifact_file_names`, and the plan block's column counts are the ones the plan computes. |
| `tests/test_docs_quickstart_e2e.py` | The four steps run end to end, produce the artifacts described, and print the plan block's column count. Marked `quickstart`; not in the default suite. |
| `tests/test_cli_commands_exist.py` | The commands exist. |
| `tests/test_cli_flags_exist.py` | `--validate-only` and `--rounds` exist on `run`. |
| `tests/test_validation_commands.py` | `--validate-only` runs the preflight and exits. |
| `tests/test_synthetic_label_signal.py` | The synthetic generator produces a learnable signal. |

### Known failure modes

- **`fedbrew: command not found` after cloning.** The console script arrives
  with `pip install -e .`. Until then use `python -m fedbrew.cli.dispatch`.
- **Expecting the two rounds to learn.** Chance on three classes is ~33%; the
  run reaches ~23%. Two rounds on 97 examples is a pipeline check.
- **Looking for `client_metrics.csv`.** Off by default. Set
  `client_statistics.per_client_csv: true`.
- **Re-running step 4 into the same `output_dir`.** The run refuses a config
  whose seed differs from the one already recorded there, rather than mixing
  two experiments in one directory. Once that run has finished, it also refuses
  a fresh start with any other setting changed, rather than replacing it; the
  same config reruns in place.
- **Reading the per-round table as the column list.** It is the curated set the
  plan header glossed; the CSV has twelve columns per split. The trailing line
  under each round says how many more were written, and `--verbose` prints them
  all.
