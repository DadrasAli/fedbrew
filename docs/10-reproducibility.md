# 10 — Reproducibility

What one seed fixes, what it does not, which settings change the numbers, and
what a resumed run inherits.

## Reproducibility

Every run records what it did in `run.json` under `reproducibility`: the git
commit and whether a tracked file was dirty (which identifies the code exactly
only for a clean checkout — chapter 09 §3.3), the seed, the deterministic flags, the torch, CUDA,
cuDNN and numpy versions, and — separately from what the config asked for —
what torch **actually held** for `matmul_precision`, `cudnn_benchmark`,
`cudnn_deterministic` and `torch_deterministic_algorithms`. Chapter 09 §3.3.

### What is guaranteed

**One seed fixes everything.** `experiment.seed` seeds Python, numpy and torch
(CPU and every CUDA device). Per-client and per-dataloader streams are derived
from it:

```
client stream      derive_seed(seed, "round", round_id, "client", client_id)
dataloader stream  derive_seed(seed, "round", round_id, "client", client_id,
                               "dataloader", phase)
```

So a client's batch order depends on **which client and which round it is**,
not on how many clients were selected or in what order. A change to
`participation_rate` or `participation_probability` does not change any
surviving client's batches.

`phase` is what separates two passes over the same client's same split in the
same round. There are three — `fit`, `eval`, `finetune` — and the third exists
because it was missing: `fedavg_ft`'s fine-tuning pass ran under `fit`, so it
drew the identical batch order the fit pass had just used, from the identical
starting weights. Two passes sharing a phase name share a stream, which is the
one way this derivation can be got wrong from a call site rather than from the
derivation itself. `tests/test_finetune_draws_its_own_order.py`.

The derivation **hashes rather than adds**, and that is not a stylistic choice.
Adding a base seed to an index makes neighbouring seeds the same sequence read
from different offsets: `Random(42 + 2)` and `Random(43 + 1)` are one
generator. Replicates seeded 42/43/44 would then share their draws shifted by a
round, and the "spread" across them would be an artefact.

**`runtime.deterministic: true` is strict by default.**
`deterministic_warn_only` defaults to `false`, so a nondeterministic kernel
*raises* rather than warning once on stderr and carrying on. It also sets
`CUBLAS_WORKSPACE_CONFIG=:4096:8` (only if not already set), disables
`cudnn.benchmark` and enables `cudnn.deterministic`.

**Resume restores RNG position, not just weights.** A checkpoint carries the
Python, numpy, torch and per-device CUDA RNG states, so a resumed run continues
the stream the interrupted one was in. A checkpoint written before `rng_state`
existed makes the run warn rather than fail — continuing on a fresh stream is a
real divergence and is said out loud. Chapter 09 §5.

**A resumed run is accounted honestly.** `run.json` records `resumed`,
`resume_from`, `attempts`, `first_round`, `duration_sec` across every attempt,
and `attempt_duration_sec` for the current process.

**A partition is reproducible independently of the run.** `dataset.seed` fixes
the partition; `experiment.seed` fixes training. Two runs at different training
seeds share one partition, which is what makes a seed spread measure training
variance rather than partition variance. Chapter 05 §3.6.

**These settings are throughput-only and leave results unchanged.** The split
is `THROUGHPUT_ONLY_PERFORMANCE_KEYS` in `fedbrew/core/config.py`, derived as
the complement of `NUMERICS_PERFORMANCE_KEYS` beside it, so this table and
chapter 11's are checked against the code rather than against each other:

| Setting | Why it is safe |
| --- | --- |
| `runtime.performance.fast_batching` | reproduces the `DataLoader` RNG protocol exactly, so a seeded run yields the same batch order epoch for epoch — `tests/test_fast_batching.py` pins this against the real `DataLoader` |
| `runtime.performance.reuse_model` | every caller overwrites the model's weights with a received state before use, and builds a fresh optimizer per fit |
| `runtime.performance.shard_cache_bytes` | only avoids repeat reads of shards nothing is allowed to edit: a served payload's mappings are the caller's own, and an in-place edit of its tensors is refused on the next serve. Chapter 5 §8 |
| `runtime.performance.torch_num_threads` | how many CPU threads torch may use. Measured bit-identical at 1, 8 and 64 threads on `examples/simplex-lsq` and `examples/nonconvex-simplex`, every non-timing column (2026-09-04), and on this repository's fixtures before that. The reason, which is the condition under which this row would stop holding: torch keeps an intra-op region on **one** thread below a grain size of 32768 elements (`at::internal::GRAIN_SIZE`), so a reduction smaller than that is summed in one order whatever the count. Above it a reduction is split across threads and float addition is not associative, so two thread counts can differ in the last ulps of a sum — real, and below the precision any table here reports, which is what throughput-only means. Chapter 11 §4.4 covers why you would set it |
| `runtime.performance.pin_memory`, `num_workers` | transfer and loading only |
| `runtime.performance.persistent_workers`, `prefetch_factor` | worker lifetime and queue depth; both inert while `num_workers` is `0` |

**The torch release is not one of the guaranteed things.** Everything above
fixes a run against *itself* — same seed, same machine, same build. It does not
survive a torch upgrade. Any model with a stochastic layer consumes torch's
global RNG stream, so a release that changes how that stream is drawn moves the
whole trajectory from round 1: upgrading 2.5.1 to 2.13.0 moved
`tests/test_regression_baseline.py`'s frozen `fit_loss` at round 1 by 0.118,
and its `fit_accuracy`, `val_accuracy_sample_weighted_avg` and
`central_test_accuracy` at later rounds by up to 0.083 (measured 2026-09-02).
Nothing is wrong when that happens, and nothing here can prevent it — but two
numbers you intend to compare must come from one torch release, and `run.json`
records the version precisely so you can check. Within a release, the CPU and
CUDA wheels agree: 2.13.0+cpu and 2.13.0+cu130 produce that fixture's numbers
bit for bit.

### These settings **do** change numerics

They are opt-in, and a run that sets one is not comparable to a run that does
not.

- **`runtime.performance.matmul_precision`.** `highest` keeps fp32 matmuls in
  fp32 (24 mantissa bits). `high` puts them on TensorFloat32 (10 stored
  mantissa bits) or a pair of bfloat16 values (~16); `medium` on bfloat16 — in
  every forward and backward pass, on any GPU that supports it. All FEMNIST
  configs set `high`, so arms within a sweep are comparable to each other; a run
  at `high` and a run at `highest` are not.

  **Absent means `highest`, not "unset".** That is torch's own default, so a
  config omitting the key is running different arithmetic from one setting
  `high` — which is why every shipped run config states it explicitly and
  `tests/test_shipped_config_explicitness.py` keeps it doing so.

  `run.json` records the precision torch actually held, not the one requested.
  `torch.set_float32_matmul_precision` does not reject an unknown value — it
  warns and keeps the current setting — so `validate_config` refuses one, and a
  typo cannot leave a run training at `highest` while `run.json` claims
  otherwise.

- **`runtime.use_amp: true`** trains in float16 where autocast allows it.

- **`client.eval_batch_size`** changes floating-point summation order in metric
  aggregation. Metrics stay example-weighted, so differences are last-bit only,
  but they are not bit-identical to a run at the training batch size.

- **`runtime.deterministic: false`** permits nondeterministic kernels. The
  difference is usually last-bit and it accumulates over rounds.

### What is not guaranteed

**Attention models are not bit-reproducible.** Every config using
`hf_causal_lm`, `hf_causal_lm_lora` or `tiny_gpt2` sets
`deterministic_warn_only: true`, which relaxes strict determinism to a warning.
This is deliberate and it is not free: `scaled_dot_product_attention`'s flash
and memory-efficient backends have no deterministic implementation, so strict
mode would *raise* rather than fall back to a slower deterministic kernel.

The measured cost: Qwen2.5-0.5B at the shipped shape gave **2 distinct gradient
digests over 6 identical passes** with `warn_only=True`, against 1 with
`warn_only=False` (A100-40GB, torch 2.5.1, CUDA 11.8, August 2026). The warning is emitted once per process by Python's
default filter, goes to stderr, and is **not** recorded in `run.json`. If an
LLM run has to be reproducible, set `deterministic_warn_only: false` and pin
the math attention backend.

**Across hardware, driver or library versions.** Determinism is within one
stack. `run.json` records the versions so a mismatch is visible, but nothing
makes an A100 and an H100 agree bit for bit.

**`cudnn_benchmark: true`** selects kernels by autotuning, which can vary run to
run. It is ignored when `runtime.deterministic` is true, so a config setting
both reads as though both apply when only one does.

**Wall-clock timings** are not reproducible and are not meant to be.

### Before a long run

1. Set `experiment.seed`.
2. Decide `runtime.deterministic`, and state it in the config.
3. Run a short same-seed reproducibility check.
4. Generate data once, up front.
5. Validate the manifest with `inspect-data` and the config with
   `run --validate-only`.
6. Choose a checkpoint interval.
7. Run one `--rounds 1` job before submitting the full one.

**Do not change code or config between a run and its resume.** A changed
learning rate mid-run produces a curve that is not any single experiment. A
resume compares the checkpoint against the settings the server strategy and the
client update rule were built with, and refuses a disagreement, naming both
values and both ways forward (chapter 09 §5). An extension's own settings are
compared only if its state methods write them. It is not the whole config: the
`data`, `model` and `evaluation` blocks are not checkpointed, so nothing
compares them, and code is unguarded entirely.

## For agents

### Paths

| Path | What it owns |
| --- | --- |
| `fedbrew/core/seeding.py` | `derive_seed`, `dataloader_seed` — the stream derivation |
| `fedbrew/core/runtime_setup.py` | `seed_everything`, and the metadata it returns |
| `fedbrew/core/runtime_setup.py` | `configure_deterministic_environment` |
| `fedbrew/core/runtime_setup.py` | `configure_runtime` — what the performance block actually sets |
| `fedbrew/core/loop.py` | `_initialize_or_resume` — RNG restore, and the missing-`rng_state` warning |
| `fedbrew/core/run_metadata.py` | what reaches `run.json`'s `reproducibility` |
| `fedbrew/core/artifacts.py` | `save_run_json` — the `seeding`/`runtime` deduplication |
| `fedbrew/core/config.py` | `MATMUL_PRECISIONS` and the value check |
| `fedbrew/core/runner.py` | `_refuse_a_foreign_seed` |

### Commands

```bash
# Prove this chapter's claims about seeding and precision.
python -m pytest tests/test_docs_reproducibility.py -v

# The determinism guards.
python -m pytest tests/test_reproducibility.py \
                 tests/test_strict_determinism_default.py \
                 tests/test_resume_rng_state.py \
                 tests/test_participation_sampling.py \
                 tests/test_matmul_precision.py \
                 tests/test_fast_batching.py \
                 tests/test_regression_baseline.py

# A same-seed check: run twice into different directories and diff.
fedbrew run --config <config> --rounds 3 --output-dir /tmp/a
fedbrew run --config <config> --rounds 3 --output-dir /tmp/b
diff /tmp/a/round_metrics.csv /tmp/b/round_metrics.csv
```

### Invariants

1. **Seed derivation hashes, never adds.** Adding makes neighbouring seeds one
   generator read at different offsets, which silently destroys a seed spread.
2. **A stream is keyed by `(seed, round_id, client_id[, phase])`** — never by
   selection order or index within a round. Two passes over the same split in
   the same round need **different phase names**, or they are one stream.
3. **`deterministic_warn_only` defaults to `false`.** Strict means strict.
4. **`CUBLAS_WORKSPACE_CONFIG` is set with `setdefault`**, never over an
   existing value.
5. **`matmul_precision` is validated at config load**, because torch warns
   rather than raising and a typo would otherwise run at `highest` while
   `run.json` recorded the typo.
6. **`run.json` records what torch held, not what the config asked**, and
   `runtime` wins over `seeding` for any key both report.
7. **`dataset.seed` and `experiment.seed` are separate.** Never derive one from
   the other.
8. **A missing `rng_state` on resume warns**; it must never pass silently.

### Tests that guard this chapter

| Test | Claim |
| --- | --- |
| `tests/test_docs_reproducibility.py` | This chapter keeps saying `matmul_precision` changes numerics, under the headings a reader would look for, and its seeding claims match `seeding.py`. |
| `tests/test_matmul_precision.py` | The value check, and that the documentation says so. |
| `tests/test_shipped_config_explicitness.py` | Every shipped config states `matmul_precision` and `deterministic`. |
| `tests/test_reproducibility.py` | Two same-seed runs agree. |
| `tests/test_strict_determinism_default.py` | `warn_only` defaults to `false`. |
| `tests/test_participation_sampling.py` | `derive_seed` field separation, and that client selection follows from the seed. |
| `tests/test_resume_rng_state.py` | RNG position survives a resume. |
| `tests/test_fast_batching.py` | `fast_batching` matches the real `DataLoader` batch for batch. |
| `tests/test_regression_baseline.py` | A pinned run still produces its pinned numbers, on the torch release they were frozen under; its round and client counts, and both resume equivalences, on every release. |
| `tests/test_runtime_setup.py` | `CUBLAS_WORKSPACE_CONFIG` and the determinism flags. |

### Known failure modes

- **Comparing a run that omits `matmul_precision` with one that sets `high`.**
  Absent means `highest`. They are not comparable.
- **Using consecutive integers as seeds and assuming independence.** They are
  independent *here*, because the derivation hashes — but only here. Code that
  adds a seed to an index does not have that property.
- **Comparing numbers produced under different torch releases.** Reproducible
  does not mean portable. Check `reproducibility.environment` in each
  `run.json` before putting two runs in one table.
- **Reading a skipped frozen-baseline test as a pass.** On any torch other than
  the one `FROZEN_TORCH_RELEASE` names, `tests/test_regression_baseline.py`
  skips its frozen-number assertions and says so. The suite is then checking
  self-consistency but not the *correct answer*. CI pins torch to that release
  so the frozen half always runs somewhere, and a guard in the same file fails
  if the pin and the constant drift apart.
- **Reading `seeding.matmul_precision` from an old `run.json`.** It was
  captured before the performance block was applied. `runtime` is effective.
- **Expecting an LLM run to be bit-reproducible.** It is not, by choice, and
  the warning is not recorded. Set `deterministic_warn_only: false` if you need
  it.
- **Setting `cudnn_benchmark: true` with `deterministic: true`.** The benchmark
  setting is skipped entirely.
- **Changing `dataset.seed` between arms.** That is a different partition, not
  a different replicate.
- **Resuming after editing the config.** Nothing prevents it and the result is
  not any single experiment.
