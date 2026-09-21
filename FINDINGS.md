# `FINDINGS.csv` — the audit finding manifest

`FINDINGS.csv` is the classified findings of the code audit, one row each, as
data rather than prose. It is the census: every finding that carries a severity
class is in it, including the ones nothing was done about. The audit filed
**112**; the census here is **109**, because three rows were removed with the
component they were about — *Rows removed with a component* below says which
and why.

The file has also outgrown the census. See *Post-census rows* below, which
counts by how much: a finding raised after the audit closed is carried in the
same file, under the same columns and the same guards, and counted separately.

The audit reports themselves are local-only and gitignored (`AUDIT/`), so they
are not in a clone and this file cites no path inside them. Everything a reader
needs — the finding's own words, where it lives, whether it is closed, and the
guard that came with its fix — is carried in the CSV.

## Columns

| Column | What it holds |
|---|---|
| `id` | `P<pass>-F<finding>`, e.g. `P10-F15`. See below. Post-census rows are `POST-F<finding>`. |
| `pass` | The two-digit audit-protocol step the finding came from, or `post` for a row outside the census. |
| `pass_title` | That step's title. Constant within a pass; repeated per row so a row stands alone. |
| `severity` | One of `wrong-results`, `silent-degradation`, `fragile`, `style`. The class the finding was filed under. |
| `component` | Area of the tree, derived from `location`: `clients`, `servers`, `core`, `data`, `tasks`, `models`, `cli`, `configs`, `tests`, `docs`, `slurm`, `reports`, `figures`. |
| `location` | One source path. See *Locations* below. |
| `summary` | The finding's own heading, verbatim except that markdown backticks are stripped. Not a paraphrase, except for the rows named under *Rows removed with a component* and *Summaries edited before publication*. |
| `confidence` | The leading token of the finding's confidence note: `certain` (102), `likely` (1), `needs-runtime-check` (1), empty (5). Qualifiers after that token are dropped. |
| `fix_commit` | Empty on every row. It held the short hashes of the commits that named each finding as fixed; *Why `fix_commit` is empty* below says why it no longer does. |
| `regression_test` | Space-separated `tests/` paths for the guard that came with the fix. Empty for 38 of the 139 rows. |
| `status` | One of three values, counted and defined in *Coverage* below. The file's only record of whether a finding was fixed. |

## How an id is formed

`P` + the pass number + `-F` + the finding's number within that pass, zero-padded
to two digits: `P01-F01` … `P13-F03`. Finding numbers restart at 1 in every pass,
so the pass number is part of the key and neither half is unique alone. The
numbering is the audit's own — findings are numbered in the order they appear in
each report, which is roughly severity order — so ids are stable but not
contiguous across the file.

## Which passes are in the census, and which are not

Twelve passes produced classified findings and make up the census:

| Pass | wrong-results | silent-degradation | fragile | style | Total |
|---|---:|---:|---:|---:|---:|
| 01 FL algorithm implementations vs. their papers | — | 2 | 2 | 4 | 8 |
| 02 Server aggregation | 1 | — | 2 | 1 | 4 |
| 03 Client-side local training | — | 1 | 5 | 1 | 7 |
| 04 Numerical stability | 1 | 3 | 4 | — | 8 |
| 06 Data generation and client partitioning | 1 | 3 | 4 | 2 | 10 |
| 07 Data leakage and evaluation protocol | 2 | 1 | 4 | 2 | 9 |
| 08 Experiment scripts and reported statistics | 2 | 3 | 3 | 1 | 9 |
| 09 Randomness and seeding | — | 5 | 4 | 1 | 10 |
| 10 Silent failures (whole codebase) | 2 | 11 | 17 | 5 | 35 |
| 11 Namespace and naming conflicts | — | — | 1 | 1 | 2 |
| 12 Performance and memory | — | 3 | — | 1 | 4 |
| 13 Reported communication and compute cost metrics | — | 2 | 1 | — | 3 |
| **Total** | **9** | **34** | **47** | **19** | **109** |

Passes 05 and 17–22 have no report; the numbers are protocol step numbers, not
report indices, and a step that produced commits instead of a report leaves a
hole by construction.

**Passes that are not in the census, and why.** Passes 14 (secrets) and 15
(third-party provenance and licensing) ran and raised no classified finding.
Passes **16, 23b, 24, 25, 26 and 28** — adversarial review, config surface,
fresh clone, the torchvision/torch pin, stage inventory, config consistency —
used numbered *sections* rather than severity-classed findings, so nothing they
raised is counted here. They are not empty: pass 16 alone raises nine
substantive sections, and they drove real fixes, including the FEMNIST held-out
split rewrite in chapter 05 and the narrowing of the README's ownership claims.
Any sentence of the form "the audit found 112 defects" drops all of that. This
file counts the severity-classed census and nothing else.

## Post-census rows

The audit is closed. Work since then still finds things, and a finding with a
severity, a location and a fix belongs in the same file as the census — a second
manifest would be a second place to look and a second thing to fall out of date.
So such a row goes in `FINDINGS.csv` with

- `pass` = `post`, which is what takes it out of every count,
- `pass_title` = `Post-census: found after the audit closed`,
- `id` = `POST-F<nn>`, numbered in the order they are added.

Everything else is the same: the same eleven columns, the same four severity
classes, the same `status` vocabulary, the same empty `fix_commit`, and — for a
row that claims a fix — the same requirement that `location` opens and
`regression_test` names a real module. A post-census row is not obliged to be
fixed. One that is still open carries an empty `status`, which is what the
census's own 5 open rows carry, and an empty `regression_test` with it.

The census tables in this file — the per-pass split, *Coverage* and its status
table — describe the **109**, and `tests/test_findings_manifest.py` computes
them over the census rows alone; the empty-cell counts describe all 139 rows.
The two numbers a reader might want:

| | Rows |
|---|---:|
| Census (passes 01–13) | 109 |
| Post-census (`pass: post`) | 30 |
| **File** | **139** |

The post-census rows, in full:

| id | severity | location | Summary | Fix |
|---|---|---|---|---|
| `POST-F01` | `wrong-results` | `fedbrew/clients/torch_sgd_client.py` | `learning_rate_schedule: cosine` with `min_learning_rate: 0.0` raises on the final round, leaving a complete-looking shorter run on disk: `global_rounds - 1` rows in `round_metrics.csv`, `run.json` still saying `status: running`, and the second-to-last round's checkpoint. Found while building `examples/fed-lasso`, where annealing to zero is the natural request. | fixed, guarded by `tests/test_learning_rate_schedule_floor.py` |
| `POST-F02` | `wrong-results` | `fedbrew/core/loop.py` | `statistics.pstdev` raises `OverflowError` on per-client losses that are every one of them a finite float: it sums the squared deviations as exact rationals, so a loss near 1e155 has a variance near 1e310 that no float can hold. The non-finite guard beside it passes them through, correctly — the condition is the statistic's, not the data's. Same artifacts as `POST-F01`, and version-dependent besides: CPython 3.11 rewrote `pstdev` to take the square root before converting, so one run dies on 3.10 and completes on 3.12, both of which CI builds. Found while building `examples/nonconvex-simplex`. | fixed, guarded by `tests/test_client_statistics_overflow.py` |
| `POST-F03` | `fragile` | `fedbrew/servers/fedopt.py` | FedOpt's first- and second-moment accumulators `m` and `v` are allocated by `zeros_like_model_state(delta)` and updated with the same `scale_model_state` / `add_model_states` helpers, none of which promotes: the moments inherit the model state's dtype and keep it for the whole run. `WeightedStateAccumulator` was fixed for the same defect under `P02-F03`, but its sum lives for one round and these live for all of them. Measured on identical bf16 deltas, moments held in bf16 against moments held in float32: `v` off by 3.2e-2 to 1.2e-1 relative, and the model state the server actually broadcasts off by 1.8e-2 to 3.8e-2, across all four optimizers at both |delta| = 1e-3 and 1e-2. Unreachable for the same reason as `P02-F03` — no shipped model loads below float32 — and reachable by the same one edit. Found while fixing `P02-F03`. | open |
| `POST-F04` | `silent-degradation` | `fedbrew/clients/torch_sgd_client.py` | `get_state` checkpoints fifteen configured client settings and `load_state` reads five of them back. `P10-F14` fixed those five — the checkpoint silently outranking the config. The other ten (`learning_rate`, `local_iterations` (then `local_epochs`), `batch_size`, `eval_batch_size`, `momentum`, `weight_decay`, `nesterov`, `learning_rate_schedule`, `min_learning_rate`, `client_id`) are the same defect facing the other way: nothing restores them, so an edited config silently wins from the resumed round on, and `run.json` records the new value as though the whole run used it. Chapter 10 states one contract covering all fifteen, so a refusal covering five under a chapter a reader takes as covering all of them reads as coverage it does not have. Found while fixing `P10-F14`. | fixed, guarded by `tests/test_resume_refuses_a_changed_hyperparameter.py` |
| `POST-F05` | `wrong-results` | `fedbrew/core/runner.py` | `run.json`'s `resumed` was `bool(resume_from)`, and `resume_from` is a config value. A run that asked to resume and could not — `round_metrics_gap` rejects the checkpoint, `discard_run_artifacts` deletes the previous attempt, and the run starts at round 1 — therefore recorded `resumed: true` and named a checkpoint it had not used. Nothing else on disk disagreed: measured on the pre-fix tree, a taken resume and a refused one produced identical `resumed`, `first_round`, `num_rounds` and `final_round`. `first_round` is not the tell it looks like — a taken resume replays `round_metrics.csv` from round 1, so it is 1 for both. Every number such a run reports is correct; what is false is the record's claim about which run produced them. Found while measuring `POST-F04`'s refusal message, which named a recovery path that turned out to restart rather than continue. | fixed, guarded by `tests/test_resume_provenance_is_what_happened.py` |
| `POST-F06` | `fragile` | `fedbrew/models/tiny_gpt2.py` | `model.pad_token_id` is the one key in this builder's table the factory writes rather than the author: `_add_causal_manifest_metadata` copies the manifest's `padding_token_id` over it, and both SFT generators write that key as `null`. `build_tiny_gpt2` read it through `int()`, so a legal pairing -- `tiny_gpt2` is a `causal_lm` model and the SFT manifests are `causal_lm` data -- died with `TypeError: int() argument must be ... not 'NoneType'` raised from inside a model factory, naming neither the key, the config, nor the manifest it came from. `hf_causal_lm` has accepted `None` for the same key all along (`_validate_padding_token`) and `GPT2Config` takes it as "no padding token", so the fix is the two builders agreeing. No shipped config makes the pairing, so nothing failed and nothing said it was unsupported. Found while closing `P10-F16`, whose measurement passed an explicit `null` through this builder. | fixed, guarded by `tests/test_tiny_gpt2_accepts_a_null_padding_token.py` |
| `POST-F07` | `silent-degradation` | `fedbrew/servers/fedavg.py` | Every server strategy is built with `participation_rate` and `seed`, and `save_state` wrote neither. `refuse_a_reconfigured_resume` compares the keys a checkpoint carries, so no comparison could see them. Measured: rounds 1–3 of a four-client SCAFFOLD run at `participation_rate: 1`, resumed from `latest.pt` for rounds 4–6 at `0.5`, exited 0, sampled 4 clients a round and then 2, and wrote `participation_rate: 0.5`, `resumed: true`, `status: completed` to `run.json`. `POST-F04`'s contract broken from the other side: there the comparison missed keys the checkpoint carried, here the checkpoint never carried the setting. It escaped the guard `POST-F04` added because that guard derives its keys from `get_state()`, the side that left the setting out, so the fix derives what every shipped strategy must save from its constructor instead. `seed` was otherwise guarded only at the runner, from `run.json`, which is skipped when `run.json` is absent. Chapter 10 said every setting in the `server` and `client` blocks was enforced. Found in the same investigation as `POST-F08`, while reproducing a SCAFFOLD resume refusal that printed a raw traceback. | fixed, guarded by `tests/test_resume_refuses_a_changed_hyperparameter.py` |
| `POST-F08` | `silent-degradation` | `fedbrew/clients/fedavg_client.py` | `FedAvgClient` (the `fedavg`, `centralized` and `fedavg_ft` rules) and `TorchDeltaSGDClient` are built with `max_grad_norm`, the bound on the gradient each applied update uses, and `get_state` did not write it. So no resume could compare it: a resume across an edited clipping threshold was taken silently, and `run.json` recorded the new value for the whole run. Measured: a four-client FedAvg run, two rounds at `max_grad_norm: 1.0`, resumed from `latest.pt` to round 4 at `0.05`, exited 0 and wrote `max_grad_norm: 0.05`, `resumed: true`, `status: completed` to `run.json`; its final model was 4.5% of the model's norm away from the same resume at `1.0`. The same defect as `POST-F07` at a different location, found in the same investigation by deriving every shipped component's constructor arguments against the keys it saves, and filed as its own row because a row holds one location. The three shipped configs that set it are LLM arms. | fixed, guarded by `tests/test_resume_refuses_a_changed_hyperparameter.py` |
| `POST-F09` | `fragile` | `fedbrew/core/runner.py` | Nothing between the console script and `run()` caught an exception, so a deliberate refusal reached the reader as a raw Python traceback whose last line happened to be the reason, and exited 1, the status a crash exits with: neither a reader nor a script could tell a run that declined its input from a defect in fedbrew. Measured through `fedbrew run` on a three-round SCAFFOLD fixture: of thirteen refusals on the resume path, twelve printed 16 to 30 lines of traceback, resuming SCAFFOLD from `best.pt` among them, and the thirteenth, a second seed in one output directory, left cleanly only because it raised `SystemExit` itself; a config refused by validation printed a traceback too. It changed what a reader saw rather than any number: every refusal still stopped the run it refused. Every refusal raised by config loading and validation, the factory, checkpoint policy, the runner's flags, the resume path, the extension loader, the model-name lookup, the loop's central- and split-evaluation checks, the LLM trace's model-config and data-manifest checks, and the dataset and prepared-asset files a run reads is now `RunRefused`, which `runner.main` alone catches, printing `RUN REFUSED` and the reason on stderr and exiting 2; any other exception keeps its traceback. A dataset manifest, client roster or shard that is missing or cannot be decoded is refused where it is read, naming the file. Refusals the client, server, model and task code raise keep their tracebacks by decision; *What is left, and why* records them. `fedbrew generate` and the `prepare-*` commands catch nothing, so their refusals, partition checks among them, still print a traceback as well; converting those is a separate decision. Found while writing the paper, from a SCAFFOLD resume refusal that printed a traceback; the investigation that found the class also found `POST-F07` and `POST-F08`. | fixed, guarded by `tests/test_a_refusal_is_a_message_not_a_traceback.py` |
| `POST-F10` | `wrong-results` | `fedbrew/core/loop.py` | A resume replays the per-client CSVs into its history, and the first flush afterwards rewrites both files from that history. `_load_existing_metric_history` caught any error loading them, printed one warning line and continued with that history empty. So a single corrupt row -- a field that does not parse, or a short row before the last -- in `client_metrics.csv` or `client_update_metrics.csv` made the resumed run rewrite both files with the resumed rounds only: every earlier round's client rows deleted, exit 0, and `run.json` saying `resumed: true` and `status: completed`. The three loads shared one handler, so a corrupt `client_metrics.csv` emptied `client_update_metrics.csv` as well, which was intact. Measured through `fedbrew run` on a five-client synthetic run with `client_statistics.per_client_csv: true`, two rounds resumed to a third: an intact resume ends with 15 rows in each file, a corrupted one with 5, all of them round 3. `load_client_metrics_csv`'s docstring names this truncation as what the reader exists to prevent, and `_refuse_or_drop_short_row` already raised on corruption before the last row; the caller swallowed it. The files are written only when `per_client_csv` is on, which is off by default and on in every shipped example config. A history that cannot be read now refuses the resume, naming the file and the line, and leaves both files as they were; a torn final row is still dropped with a warning. Found while measuring the refusals `POST-F09`'s core family converted. | fixed, guarded by `tests/test_resume_metrics_continuity.py` |
| `POST-F11` | `fragile` | `fedbrew/core/runner.py` | `maybe_stage_manifest_dataset` swaps `config.data.path` for the staged copy before the run builds its components, but the dataset provenance in `run.json` is built from the config as it was resolved before that swap, so `manifest_path` and `manifest_resolved_path` name the source whether or not the run read the copy. The only other trace is `config.runtime.extra.data_staging.enabled`, which records the request: a run whose staging was skipped -- no usable scratch root, a copy that failed -- prints one line to stdout and writes the same `run.json` as a run that staged. No number is affected, because a completed copy holds the source's files, but nothing on disk says which files a run read. Found on 2026-09-16 by a CPU rehearsal of the GPU smoke test, whose staged run's dataset provenance matched the unstaged runs' exactly. Open by decision: provenance work was out of scope on the day of the smoke test. | open |
| `POST-F12` | `silent-degradation` | `fedbrew/core/config.py` | `_validate_divergence_metric_is_reachable` exempts every diagnostic the selected strategy adds from both metrics filters: `_metrics_no_filter_can_remove` takes `SERVER_DIAGNOSTIC_METRICS` whole. FedLALR's four `client_effective_learning_rate_{mean,std,min,max}` round columns are built from each client's `client_effective_learning_rate_mean`, so a non-empty `client.metrics` that omits it leaves the server nothing to spread. A config whose `divergence.metric` names one of the four then loads, no round carries the column, and every detector -- `non_finite` included -- stays silent for the run, with the loop's warning arriving after the last round. Measured on 2026-09-17 with one FedLALR round on the synthetic base, `client.metrics` and `server.metrics` both `[fit_loss]`: the check accepted all four names and the run wrote none of them. No shipped config is in that shape. The plan header made the same assumption and no longer does, because `server_diagnostic_metrics` applies `SERVER_DIAGNOSTIC_SOURCES` there; the check does not call it. Found while fixing the header. Left open by decision on 2026-09-17. Fixed on 2026-09-21: the exemption is now `server_diagnostic_metrics`, which drops a diagnostic whose source `client.metrics` filters out, and `_require_derived_diagnostic_sources` refuses a `divergence.metric`, `server.metrics` or `client.metrics` naming one of the four spread columns under a non-empty `client.metrics` without their source, naming the metric to add. The columns are now `effective_learning_rate_across_clients_{mean,std,min,max}` and their source `effective_learning_rate_coordinate_mean` (`POST-F14`). | fixed, guarded by `tests/test_divergence_metric_reachable.py` |
| `POST-F13` | `fragile` | `tests/test_divergence_metric_reachable.py` | `ItStaysQuietWhenTheMetricSurvivesTest.test_the_strategy_s_own_diagnostics_are_never_filtered` cannot fail. It sets `server.strategy: fedlalr` on the smoke base and leaves `client.update_rule: local_sgd`, so `validate_config` refuses the pairing before the divergence check runs. `_refuses` returns False for any refusal that does not name `divergence.metric`, and the test asserts False. Measured on 2026-09-17: with the strategy diagnostics removed from the check's exemption, the test still passes. Its neighbour `test_another_rule_s_extras_are_not_borrowed` asserts a load the same way, on a `fedavg` rule missing `update_mode`. That refusal comes after the check, so the neighbour does fail when the client exemption is removed, but only because of the order `validate_config` runs its checks in. Chapter 14 §6.6 records the class, a guard that cannot fail. Left open by decision on 2026-09-17. Fixed on 2026-09-21: both tests now build a complete configuration -- the first a valid FedLALR pair, the second a `fedavg` client with its `update_mode` and weighting -- and assert that nothing at all is refused, rather than that no refusal names `divergence.metric`. With the strategy diagnostics removed from the check's exemption, the first now fails. | fixed, guarded by `tests/test_divergence_metric_reachable.py` |
| `POST-F14` | `wrong-results` | `fedbrew/servers/fedlalr.py` | FedLALR's clients and server both write `client_effective_learning_rate_min` and `_max`, as different quantities: the client's are the extremes of its own per-coordinate rates, the server's the extremes of the clients' means. `FedLALRServer.aggregate_stream` averages the clients' metrics, weighted by examples, into the round record, then overwrites the four spread names with its own values, but only when the clients reported `client_effective_learning_rate_mean`. So a `client.metrics` that names the client's `_min` or `_max` and not `_mean` puts the clients' averaged value in `round_metrics.csv`, under the name chapter 08 §7.3 defines as the server's spread, and the chapter said the two never occupy the same column. Measured on 2026-09-17 with one FedLALR round on the synthetic base: `client_effective_learning_rate_min` was 0.637 under `client.metrics: [fit_loss, client_effective_learning_rate_min]` and 81.9 under `[fit_loss, client_effective_learning_rate_mean]`. No shipped config is in that shape. Left open by decision on 2026-09-17. Fixed on 2026-09-21 by renaming both families, with no computation changed: the client's are `effective_learning_rate_coordinate_{mean,min,max}` and the server's `effective_learning_rate_across_clients_{mean,std,min,max}`, so no name is both a statistic over one client's coordinates and one across clients. The four old names are refused (`RETIRED_METRIC_NAMES`, `fedbrew/core/metrics.py`) in `server.metrics`, `client.metrics`, `divergence.metric` and `best_metric`, naming the replacement, and a resume onto CSVs whose header carries one is refused before anything is written, so no file mixes the two. `configs/femnist/fedlalr.yaml` uses the new names. Chapter 08 §7.3 defines each column's population, coordinate reduction, denominator and reduction across clients, and the guard checks all seven against hand-computed values for two clients. | fixed, guarded by `tests/test_fedlalr_diagnostics.py` |
| `POST-F15` | `fragile` | `fedbrew/clients/local_update_modes.py` | `defaults.local_iterations` counts iterations of the local loop, and `update_mode` decides what one iteration is (chapter 04 §2.1). K iterations are K parameter updates per client per round under `single_batch`, `frozen_batch_gradients` and `full_gradient`, and K × B under `sequential_epoch`, where B is the client's batch count. Every client rule takes an `update_mode`; `sequential_epoch` is also what one that leaves it unset runs, which `fedprox`, `scaffold`, `fedlalr`, `local_sgd`, `local_adamw` and `delta_sgd` may. So arms that differ in update shape do different amounts of local work at one `local_iterations` value, and a comparison between them at equal `local_iterations` is not like-for-like: it compares K steps against K × B. One shipped family is in that shape: in `configs/mnist/`, `fedavg.yaml` and `centralized.yaml` run `single_batch` and `scaffold.yaml` runs full passes, all at `local_iterations: 1`. Every other family's arms share one shape. Recorded on 2026-09-18 with the rename of `local_epochs` to `local_iterations`, which made the count's meaning explicit and left the modes as they were. Open by decision: the modes are not changed. | open |
| `POST-F16` | `silent-degradation` | `fedbrew/core/config.py` | `load_config` read `defaults.global_rounds` and `defaults.local_iterations` and checked no other key in the block, so `defaults.bogus_key: 7` loaded: measured on 2026-09-18 on `configs/dev/smoke.yaml`. Every other block refuses a key no reader consumes (`_validate_known_keys`), but that check walks the `extra` dicts `FullConfig` stores, and `defaults` is read at load and never stored, so it never saw the block. A key beside the two real ones was therefore dropped without a word: a misspelling such as `local_iteration: 20`, a key that reads as a default for every client such as `learning_rate: 0.1`, or, once the schedule key was renamed, a stale `local_epochs` next to `local_iterations`. A misspelling on its own was caught only because both real keys are required. No shipped config wrote any other key. The block now accepts exactly `DEFAULTS_KEYS` and refuses anything else at load with the message every other block gives, naming the two it accepts; the old spelling `defaults.local_epochs` is refused first, by name. Chapter 04 §2.1 says the block is closed, and `tests/test_docs_config_keys.py` now checks §2.1's table against `DEFAULTS_KEYS`, which no guard did: when the schedule key was renamed, nothing noticed the table still offering the old name. Found on 2026-09-18 while planning that rename. | fixed, guarded by `tests/test_unknown_config_keys.py` |
| `POST-F17` | `fragile` | `tests/test_client_splits_is_declared.py` | `NoShippedConfigSetsItWhereItIsInertTests.test_every_shipped_generator_config_validates` skipped any config whose `dataset.name` no registered generator answered to, and it never loaded the config's `dataset.extensions`, which is where the ten example configs get their generators. Run alone it checked 14 of the 24 shipped generator configs and passed; it checked all 24 only when an earlier test in the same process had loaded the example extensions. Measured on 2026-09-18 while making the suite run under pytest-xdist: its subtest count moved between 14 and 24 with the order the tests ran in, and it passed either way. Its sibling `GeneratorConfigTest.test_every_shipped_generator_config_still_passes` in `tests/test_unknown_config_keys.py` missed the same step without the skip, so it failed its ten example subtests when run alone or on an xdist worker that had not loaded them, and passed serially only because `RunConfigTest` ran first. Both now check each config the way `generate_from_config` does, with the extensions loaded before the name is looked up, and a name no generator answers to fails. With the load removed, each fails its ten example subtests. Chapter 14 §6.6 records the class, a guard that cannot fail; counting that section's eight and `POST-F13`, this is the tenth found in this project. | fixed, guarded by `tests/test_client_splits_is_declared.py` |
| `POST-F18` | `silent-degradation` | `fedbrew/core/config.py` | `client.frozen_gradient_weighting` is required of every `fedavg`, `centralized` and `fedavg_ft` config and checked against `examples`, `uniform` and `sum`, but only `update_mode: frozen_batch_gradients` reads it. Under `single_batch` and `sequential_epoch` any of the three values loads and changes nothing: measured on 2026-09-18 on `configs/mnist/fedavg.yaml`, all six combinations load. A resume compares it all the same (`FedAvgClient.get_state`), so editing a value that affects nothing refuses the resume. All 61 shipped configs that take an `update_mode` carry `frozen_gradient_weighting: examples` (57 `fedavg`, 2 `centralized`, 1 `fedavg_ft`, 1 `delta_sgd`, which checks the key only when it is present), and none runs the frozen mode. Found on 2026-09-18 while designing `update_mode: full_gradient`. Open by decision: requiring the key only where it is read, and refusing it elsewhere, touches all 61 configs. Its engine half, `run_sgd_update_mode` requiring the argument under every mode, touches no config and is `POST-F21`, fixed. | open |
| `POST-F19` | `fragile` | `fedbrew/clients/local_update_modes.py` | `frozen_gradient_weighting: examples` weights each batch gradient by the batch's example count, and `tests/test_frozen_gradient_weighting.py` defines the result as the example-weighted mean gradient over the train split. That holds for a loss that is a mean over examples. The causal-LM loss is a mean over active target tokens (`TorchCausalLMTask._loss_and_counts`), so each batch gradient is a token mean, and weighting them by sequence count gives the gradient of the pass only when every batch holds the same number of active tokens. Measured on 2026-09-18 on `tiny_gpt2` with the `llm` extra installed: for two batches of two sequences holding 2 and 8 active tokens, the combined update is 53.5% (relative) from the gradient of one batch holding all four sequences, and weighting each batch by the `total` its `train_step` reports instead gives 8.6e-7. The guard checks a least-squares task only. `fragile`, not `wrong-results`, because no shipped arm reaches it: no shipped config sets `update_mode: frozen_batch_gradients` (58 set `sequential_epoch` and 3 `single_batch`), no CLI flag sets the mode, and the six LLM run configs use `sequential_epoch` under `fedavg` (4) or `local_adamw` (2). Found on 2026-09-18 while designing `update_mode: full_gradient`. Fixed on 2026-09-19 by refusing the mode on a task whose loss is not an example mean, under every weighting: at load for the built-in tasks in `NON_EXAMPLE_MEAN_TASKS` (`causal_lm`), through `fedbrew/core/config.py`'s `_refuse_frozen_off_examples`, and in both engines, `run_sgd_update_mode` and `run_delta_sgd_update_mode`, for any task whose class overrides `TaskAdapter.train_loss_denominator`, before the first gradient. The second is the backstop for an extension task, whose class is not known until it is built. The guard derives the set from the built-in task classes, so the two cannot drift. Rejected: changing what `examples` means, which would alter a shipped mode's arithmetic now that `full_gradient` computes the exact gradient on every task; and leaving the row open, which leaves the pairing configurable. | fixed, guarded by `tests/test_frozen_gradient_weighting.py` |
| `POST-F20` | `fragile` | `examples/drift-quad/README.md` | The tuning section reports that 61 of 861 grid points tripped the divergence guard. The grid the same section quotes is six client rates for `fedavg`, three `proximal_mu` values over those for `fedprox`, four optimizers x six x seven x three for the FedOpt family, six for `scaffold` and six for `fedlalr` -- 540 points per dial setting, 1620 across the three. 861 is neither, and no other reading of the section gives it. The sweep that produced it wrote nothing the tree keeps, so which of the two numbers is wrong cannot be settled from here: either the grid changed after the sweep or the count covers something the section does not describe. Left open rather than guessed. | open |
| `POST-F21` | `fragile` | `fedbrew/clients/local_update_modes.py` | `run_sgd_update_mode` took `frozen_gradient_weighting` as a required argument and checked it against `examples`, `uniform` and `sum` under every `update_mode`, but only `frozen_batch_gradients` reads it. A caller running `single_batch`, `sequential_epoch` or `full_gradient` had to state a weighting that changes nothing, and one that passed none failed inside `normalize_choice` with `AttributeError: 'NoneType' object has no attribute 'lower'` rather than a refusal: measured on 2026-09-19. `FedAvgFTClient`'s fine-tuning pass, which always runs `sequential_epoch`, passed its configured weighting for that reason alone. It is `POST-F18`'s engine half: that row is the config requiring the key, this one the engine requiring the argument, and this one can be fixed without touching a config, because `FedAvgClient` still passes the value its config requires. `fragile`, not `silent-degradation`: every caller passed a valid value, so no run changes. Found on 2026-09-19 while separating which client rules take an `update_mode` from which take a fixed learning rate. Fixed on 2026-09-19: the argument defaults to none and is required, by a `ValueError` naming the choices, only before `frozen_batch_gradients` runs; a value given under another mode is still checked, and the fine-tuning pass passes none. | fixed, guarded by `tests/test_frozen_gradient_weighting.py` |
| `POST-F22` | `wrong-results` | `fedbrew/core/runner.py` | A fresh start into an `output_dir` holding a finished run of a different config replaced that run -- `run.json`, `round_metrics.csv` and the checkpoints -- with nothing refused and nothing on disk saying the settings had changed. `_refuse_a_foreign_seed` compared the seed and nothing else, and a resume compares the checkpoint, so a rerun at the same seed with an edited learning rate, or a config whose hand-written `output_dir` named settings it no longer had, silently left one run's curve under another's path: measured on 2026-09-19, where three reruns of one arm into one directory left three `runs_index.jsonl` lines and one `run.json`. Found on 2026-09-19 while deciding what a sweep may do with a directory a hand-launched run already wrote. Fixed on 2026-09-19: if `run.json` says `completed`, `diverged` or `stalled` and its recorded config differs from this run's, `fedbrew run` refuses and names every differing key with both values. Labels and launch flags are not compared, nor a key recorded on one side only, nor `data.path` under data staging; the same config still reruns in place, a resume keeps its own check, and a `run.json` still saying `running` restarts in place. | fixed, guarded by `tests/test_run_json_resume_accounting.py tests/test_resume_refuses_a_changed_hyperparameter.py` |
| `POST-F23` | `fragile` | `fedbrew/core/checkpointing.py` | `save_checkpoint`, `save_latest_checkpoint` and `save_best_checkpoint` called `torch.save` on the target path, and `save_run_json` wrote `run.json` with `write_text`. Both truncate the file before writing it, so a process killed inside the write left a short or empty file where the previous complete one had been. Under `save_last`, `latest.pt` is rewritten every round and is the one file `--resume-latest` reads; `run.json` is rewritten every round too, a resume reads it for `attempts` and the time already spent and falls back to one attempt and no time on a file it cannot parse. Found on 2026-09-21: a SLURM time limit ended a SCAFFOLD sweep job on 2026-09-20 at 23:30 with eleven points still running, and one of them was left with a zero-byte `latest.pt`, written 0.3 s after its last `round_metrics.csv`, at round 7235 of 10000, so resuming it can only restart from round 1; the other ten checkpoints loaded. Fixed on 2026-09-21: all four writers go to a sibling `<name>.tmp`, flush and `fsync` it, and `os.replace` it over the target, so the path holds the previous complete file or the new complete one. A write that fails removes its temp file; one killed outright leaves it for `clear_stale_temp_files`, which now also sweeps `checkpoints/*.pt.tmp` at the next start, and no checkpoint glob matches the `.tmp` name. The per-client CSVs are unchanged: they are appended, a kill can only shorten their last row, and both readers drop it. | fixed, guarded by `tests/test_checkpoint_writes_survive_a_kill.py` |
| `POST-F24` | `fragile` | `fedbrew/core/loop.py` | Each round wrote its checkpoints (`_update_checkpoints`) before its `round_metrics.csv` row and `run.json` (`flush_round_artifacts`, then the runner's writer). A resume replays `round_metrics.csv` up to the checkpoint's round, so a kill between the two left a checkpoint one round ahead of a history it could not be continued onto, and `--resume-latest` then restarted from round 1 and deleted the attempt. Late in a long run the full CSV rewrite is most of each round, so a time limit lands in that window almost every time. Found on 2026-09-21 by resuming a copy of one of the ten SCAFFOLD points a 12 h limit stopped on 2026-09-20: all ten had `latest.pt` at round N beside a CSV ending at N - 1, and a job cancelled on 2026-09-21 left its two points the same way. Fixed on 2026-09-21: the round's checkpoints are staged where they were written -- complete on disk as `.tmp`, so `checkpoint_sec` still times the write -- and `StagedCheckpoints.commit` renames them into place after the CSV rows and `run.json`, then `keep_last` prunes. A kill before the commit leaves the previous round's checkpoint beside a history that reaches it or one row past it, and a resume drops the rows after the checkpoint's round and recomputes them. | fixed, guarded by `tests/test_a_round_commits_its_checkpoint_last.py` |
| `POST-F25` | `fragile` | `fedbrew/core/loop.py` | A resume replays `round_metrics.csv` up to the checkpoint's round. When the CSV did not reach it, `_initialize_or_resume` called `discard_run_artifacts` -- deleting the checkpoints, the CSVs and `run.json` -- printed one line and started again from round 1, recording why under `run.json`'s `resume_restart`. In a batch job nobody reads that line: a 10000-round run stopped by a time limit came back as a fresh run, the rounds it had done were gone, and so was the evidence of why. Found on 2026-09-21 by resuming a copy of one of ten SCAFFOLD points a 12 h limit had stopped with `latest.pt` a round ahead of its CSV (`POST-F24`): `Cannot resume from round 9690 ... Restarting from round 1. Deleted the previous attempt's checkpoints, round_metrics.csv, client_metrics.csv, client_update_metrics.csv, run.json (7.8 MB)`. Resubmitting the sweep would have done that to all ten. Fixed on 2026-09-21: the resume raises `RunRefused` before anything is written -- the stale-temp sweep now runs after the decision, so not even a `.tmp` goes -- naming the checkpoint, its round, what the CSV is missing, and the two ways forward: an earlier checkpoint the CSV reaches, or moving the directory aside to start over. `discard_run_artifacts` and `resume_restart` went with the restart they served. | fixed, guarded by `tests/test_a_refused_resume_changes_nothing.py` |
| `POST-F26` | `silent-degradation` | `fedbrew/core/config.py` | `load_config` read the top-level blocks it knew by name -- `common["experiment"]`, `common.get("evaluation", {})` and the rest -- and never looked at the other keys, so a misspelled optional block loaded and the run took that block's defaults. Measured on 2026-09-21 on `configs/synthetic/fedavg.yaml`: `evaluaton:` loaded and moved train evaluation from `clients: all` to the default `participating`, `divergance:` loaded and ran with no `blowup_absolute` where the config says 23.0, and `client_statstics:` and an unrelated `bogus_root: 1` loaded as well; `--validate-only` reads the config through the same function and passed each. Every block was already closed (`_validate_known_keys`, and `POST-F16` for `defaults`), but the root was not a block. No shipped config writes a top-level key outside the ten blocks. Found on 2026-09-21 by a pre-publication review. Fixed on 2026-09-21: `load_config` refuses, before any block is read, a top-level key that is not in `root_config_keys()` -- the blocks `FullConfig` stores less the removed `task`, plus `defaults` and the older `server_config` / `client_config` path spellings -- naming the key, the closest block, and every key the root accepts. The removed `task` block keeps its own refusal and reason. The run CLI's overrides write named fields of existing blocks, so none can add a top-level key, and a misspelled flag is refused by the parser. The guard runs the three misspellings through `load_config`, `fedbrew run` and `--validate-only`, and loads every shipped run config in `configs/` and `examples/`, the component-path spelling, and an extension-declared key. | fixed, guarded by `tests/test_unknown_config_keys.py` |
| `POST-F27` | `silent-degradation` | `fedbrew/servers/scaffold.py` | `ScaffoldServer.aggregate_stream` folded each client's model state through `WeightedStateAccumulator`, which refuses NaN and Inf (`P04-F02`), and summed its `control_delta` beside it with `add_model_states`, which checks nothing. A finite model beside a non-finite delta therefore aggregated normally and put the NaN into `server_control`, where `c <- c + (1/N) sum(dc_i)` keeps it for good. Measured on 2026-09-21: in-memory, a NaN, +Inf or -Inf delta each returned normally and `save_state` carried it; and through `fedbrew run` on a two-client SCAFFOLD smoke config with one client's round-3 delta set to NaN, round 3 was recorded as healthy, `latest.pt` was written at round 3 with a non-finite `server_control`, and the run stopped a round later as `diverged` at round 4 on a non-finite *model* state, since every client's corrected step had gone NaN. So the verdict named the wrong round and the wrong state, and the last checkpoint, the one `--resume-latest` reads, was the poisoned one. Found on 2026-09-21 by a pre-publication review. Fixed on 2026-09-21: each incoming `control_delta`, their sum and the updated `server_control` go through `refuse_non_finite_state` (`fedbrew/core/torch_utils.py`), the check the accumulator itself now calls, and the new model and control variate are computed first and assigned together only after all three pass. A refused round leaves both bit-for-bit as they were, finite deltas that overflow the sum or `c` are refused too, and the loop records the round as the `non_finite_client_state` divergence a non-finite model produces, so the last checkpoint is the last healthy round and resumes. | fixed, guarded by `tests/test_scaffold_control_state_is_finite.py` |
| `POST-F28` | `fragile` | `fedbrew/servers/scaffold.py` | `FedAvgServer._accumulate_fit_results` checks every fit result's `model_state_scope` and, for an adapter, its base model, revision, adapter name and LoRA config against the server's own (`validate_federated_state_metadata`) before folding it. `ScaffoldServer.aggregate_stream` and `FedLALRServer.aggregate_stream` (`fedbrew/servers/fedlalr.py`) override the fold and did not call it. Measured on 2026-09-21: an adapter-scoped result with the server's tensor shapes was refused by FedAvg and averaged into a full-state server by both of the others. `fragile`, not `wrong-results`: the server and every client are built from one config and one task, so no shipped config produces a mismatched result. What was missing is the backstop against a client built from another task or an extension, which FedAvg had and these two did not. Found on 2026-09-21 by a pre-publication review. Fixed on 2026-09-21: the check is one method, `FedAvgServer._compatible_model_state`, which FedAvg, the FedOpt family, SCAFFOLD and FedLALR all call on every result before anything is accumulated. It also holds the result's keys and tensor shapes to the server's own state (`validate_state_matches`, `fedbrew/core/federated_state.py`), which nothing checked on any server: the accumulator compared clients only with each other. A refused result leaves the model, SCAFFOLD's control variate, FedLALR's moments and the round record unchanged, even when it arrives after a good result has been folded. | fixed, guarded by `tests/test_federated_state_compatibility.py` |
| `POST-F29` | `fragile` | `fedbrew/core/config.py` | Chapter 07's cost table listed LoRA under "any rule". Three rules cannot train adapter-only state: `fedprox` (`fedbrew/clients/torch_fedprox_client.py`) and `scaffold` load the broadcast into the whole model with `load_model_state` and return `get_model_state`, not the task's federated state, and `fedlalr` looks its AMSGrad moments up by `named_parameters()` names, which under a PEFT adapter carry the adapter name the federated state's keys do not. Config load accepted all three with `hf_causal_lm_lora`, and so did `--validate-only`. Measured on 2026-09-21 with one-round runs on a tiny GPT-2 LoRA fixture, one per combination config load allows: `fedprox` with each of the five FedAvg and FedOpt servers and `scaffold` failed in `load_state_dict`, and `fedlalr` with `missing FedLALR optimizer state for parameter: ...lora_A.probe.weight`, each in round 1 after the model was built. The other 26 combinations completed and moved the adapter. `fragile`, not `silent-degradation`: every failure was loud, and no shipped config makes these pairings. Found on 2026-09-21 by a pre-publication review, which named `fedprox` and `scaffold`; the measurement added `fedlalr`. Fixed on 2026-09-21 by refusing, not by adding support, which would mean re-keying each rule's state onto the task's hooks. The refusal comes at config load for an adapter-scoped model the package builds (`ADAPTER_SCOPED_MODELS`), before anything is built or written, and in each client before its first update when the task reports adapter scope, an extension's included (`refuse_adapter_state`, `fedbrew/core/federated_state.py`). Both messages name the rule, why, and the rules that can. `ADAPTER_STATE_CLIENT_RULES` and `FULL_STATE_ONLY_CLIENT_RULES` partition the built-in rules, and a new rule has to join one. Chapter 07 §5.1 lists the 26 combinations, each of which the guard runs for a round and requires the adapter to move. | fixed, guarded by `tests/test_adapter_state_support.py` |
| `POST-F30` | `silent-degradation` | `fedbrew/core/config.py` | `model.active_target_weighting` decides a causal-LM client's aggregation weight, and its only reader is `TorchCausalLMTask.federated_aggregation_weight`. `fedprox` (`fedbrew/clients/torch_fedprox_client.py`) and `scaffold` never call that hook: both report the count of their post-fit evaluation pass, which for the causal-LM task is the active target tokens of the whole train split. So `active_target_weighting: true`, or the `causal_lm_sft` default that turns it on, loaded under either rule, was copied into `run.json` with the config, and changed nothing. No shipped config pairs either rule with a causal-LM model. Found on 2026-09-21 while defining the aggregation weight for chapter 07 §3.1, which first recorded it as a caveat. Fixed on 2026-09-21 by refusing the combination: the key at config load, through `fedbrew run` and `--validate-only`, and the SFT default where each path reads the manifest, `factory._model_config` and preflight's `data` check (`active_target_weighting_refusal`, `fedbrew/core/federated_state.py`). The message names the rules that honour it and says `false` gives these two the weight they already use. `AGGREGATION_WEIGHT_HOOK_CLIENT_RULES` and `AGGREGATION_WEIGHT_HOOK_BYPASS_RULES` partition the built-in rules, and the guard checks each against what the rule's `fit` does with a task whose hook returns a sentinel. | fixed, guarded by `tests/test_active_target_weighting_is_honoured_or_refused.py` |

### Two corrections that are not rows

Migrating the five `examples/` onto `experiment.extensions` re-ran
each of them from the configs that now ship with it and re-measured every
published table. Every number reproduced bit-identically except two, and both of
those were wrong before the migration touched them.

Neither is a framework defect. The code did what it documents in both cases —
one is arithmetic done by hand at the wrong precision, the other a command that
was never run — so neither is a row here: `FINDINGS.csv` is a census of defects
in `fedbrew/`, and its worth is that it is a clean dataset of one kind. That is
the same call chapter 14 §6.7 records for the `torch_num_threads` retraction in
`examples/simplex-lsq/README.md`. They carry no `POST-F` id for the same reason:
an id in this file is a row's key, and there is no row.

They are written down anyway, because they are the failure the census exists to
catch — a published number that is not true — and because what caught them is
worth naming. Not a reader and not a guard: re-running the example from the
config that ships beside it.

| Where | Published | True | What it was |
|---|---|---|---|
| `examples/fed-lasso/README.md`, the η curve | `7.70e-05` at η = 0.008 | `7.69e-05`, from `7.694928…e-05` | A rounding slip in the original table. Every other cell of that curve, and every cell of every other table in that README, came back identical. |
| `examples/nonconvex-simplex/README.md`, the control | `--all --star-leaves 4`, offered as "the decoy removed" | there is no such control | The command never ran. `ProblemSpec.__post_init__` refuses any `star_leaves` at or below `(clique_size − 1)² = 16`, so it raised before the first round every time it was invoked. Removing the decoy and keeping the guard are contradictory; the guard is the half worth keeping. |

Both READMEs carry the correction in place, next to the number rather than in a
changelog, and both say the migration is what found it. The second is a
retraction rather than a fix: the example ships no control config, because the
control as documented cannot exist.

### Two severities the census got wrong

`fragile` means *correct today*: the code does the right thing, and one
plausible edit away it would not. Fixing the deferred fragile and style
findings turned up two rows where that was not true. Both are recorded here and
**neither `severity` is edited in `FINDINGS.csv`** — the file is the record of
what the audit filed, and a row rewritten to agree with a later reading is a row
that can no longer be checked against the report it came from.

| Finding | Filed | What it was | Which kind of wrong |
|---|---|---|---|
| `P04-F06` | `fragile` | A NaN first observation of the selection metric is accepted as the run's best and freezes `best.pt`. | Correct when filed, then overtaken. At census time a NaN in `val_loss_avg` needed a client to report a NaN loss — a data or model pathology. Five days later `POST-F02`'s fix made `_overflow_safe` answer NaN for a statistic that cannot fit, which is the framework's own designed output on a diverging run. The finding did not change; the reachability did. |
| `P07-F07` | `fragile` | `val: sample:N` and `test: sample:N` drew the identical clients, so the reported test score measured exactly the clients `best.pt` was selected on. | Wrong when filed. Two shipped configs were already in that shape — `configs/openimage/fedavg.yaml` at `sample:2000` and `configs/reference_evaluation.yaml` at `sample:40` — and the audit's own evidence table names both. It filed `fragile` on the grounds that neither had been run, which is a statement about the output directory rather than about the code: a config that ships wrong is not correct today, it is wrong and not yet executed. `reference_evaluation.yaml` stated the property it lacked, in a comment on the block that lacked it. |

The distinction is worth keeping because the two say different things about the
census. The first is what a census cannot help: it describes a tree, the tree
moves, and a severity is only ever true on a date. The second is a
misclassification, and the rule it broke is the one that makes `fragile` usable
at all — *is this correct today* has to be asked of the shipped configuration,
not of what has been run so far.

## Rows removed with a component

The audit filed **112** findings. The census carries **109**, because three rows
were removed together with the component they were about, when that component
was removed from the tree:

| id | pass | severity |
|---|---|---|
| `P01-F08` | 01 | `style` |
| `P07-F05` | 07 | `fragile` |
| `P08-F06` | 08 | `fragile` |

Each was about that component and nothing else, so with it gone the row would
have cited a path, a guard or a subject nobody could open or interpret. The ids
are retired: no later row reuses one. This is the one place the file does not
keep what the audit filed, and every census count here — the per-pass split,
*Coverage*, the locations — describes the 109; the empty-cell counts describe
all 139 rows of the file.

`P07-F05`'s fix did not depend on its row. It turned `save_best` on for the MNIST
baseline, which is still what `configs/mnist/fedavg.yaml` ships and what
`tests/test_comparison_arms_agree_on_save_best.py` still checks for every family.

Two rows that named the component beside others stay, and moved:

- **Locations**, by the rule under *Locations*: each now names the next path the
  finding cites that still exists — `P13-F01` `fedbrew/clients/torch_delta_sgd_client.py`,
  `P13-F03` `fedbrew/clients/torch_fedlalr_client.py`.
- **One summary.** `P13-F01`'s heading named the component beside Delta-SGD; it
  now names Delta-SGD alone. It was the first `summary` in the file not to be the
  audit's heading verbatim; *Summaries edited before publication* names the
  rest. Keeping it verbatim would have kept the removed component's name in the
  tree, and a convention that requires that has to give.
  `P13-F03`'s heading names no component and is unchanged; its "five client
  classes" is the audit's count, made when there were five.

## Summaries edited before publication

Ten more `summary` cells are not the audit's heading verbatim. Nine of them
stated what unpublished material showed: the audit read experiment write-ups,
figures and sweep outputs that were never published, and a heading that says
what those showed publishes it. Those rows now state the defect and leave the
result out. The tenth, `P01-F04`, said "the paper averages uniformly", which is
true of both papers' algorithms and not of the FedOpt paper's experiments, which
weight by example count. Nothing else in these rows changed: `id`, `severity`
and `status` are as filed, and so is `location`, except for the two paths
generalised under *Locations*.

| id | What the edit took out |
|---|---|
| `P01-F04` | "the paper", for "the papers' algorithms" |
| `P07-F01` | that the selection covered every *reported* hyperparameter |
| `P07-F02` | that arm selection on MNIST was made |
| `P07-F10` | that every run path the figures name is gone |
| `P08-F01` | the report the three-seed spreads came from |
| `P08-F02` | that the differences drawn were smaller than a run's own fluctuation |
| `P08-F07` | the "best arm" the comparison named |
| `P08-F09` | the report's name, and the claims its statistics backed |
| `P10-F01` | how many on-disk CSVs were shifted, and by how much |
| `P10-F02` | the round count of one resumed run on disk |

That gives up, for these rows, what the verbatim rule is for: checking a row
against the report it came from. The reports are local-only, as the top of this
file says, so no reader of the file can make that check for any row.

## Coverage: what is closed and what is not

| | Findings | Closed |
|---|---:|---:|
| wrong-results | 9 | **9** |
| silent-degradation | 34 | **34** |
| fragile | 47 | **47** |
| style | 19 | 14 |
| **Total** | **109** | **104** |

The second column is every row whose `status` is set — `fixed` or `obsolete`,
the two closing values of the *status* table below.

**Every result-affecting finding is closed** — all 43 of the wrong-results plus
silent-degradation rows. All **5** rows still open are `style`, and every one
of them is a decision — `fragile` is closed end to end, and `style` is the only
class with anything left in it. *What is left, and why* below names all five.

`status` takes one of three values over the 109, and two of them mean *closed*:

| `status` | Census rows | What it records |
|---|---:|---|
| `fixed` | 95 | The finding was fixed |
| `obsolete` | 9 | The code the finding names is gone, so there is nothing left to fix |
| empty | 5 | No record either way |

`fixed` does not mean the fix has been re-verified against the current tree. No
independent status record covers these 109, and a row saying `fixed` records a
closure as it was established at the time, not what is true today.

`fixed` absorbed two values that said how a closure was attributed to a commit;
*Why `fix_commit` is empty* says why. Twelve of its rows came from them: the
four findings of pass 12 and the three of pass 13, which were addressed report
by report rather than finding by finding, and five closed by a change aimed at
something else. Those five are `P03-F02` (`local_adamw` could not be
constructed) and `P03-F03` (an unvalidated loader block reaching the evaluation
pass), which went with changes that describe the same defect without citing the
audit; `P01-F06` (nothing pinned the SCAFFOLD and FedProx arithmetic), closed
when those tests were written; `P10-F28` (a round-table legend naming a column
no run emits), closed when the legend was derived from the emitter; and
`P06-F07` (the aggregation weight counting evaluation examples), closed when the
weight stopped being read from `clients.jsonl` — the only one of the five
settled by measurement rather than by reading, and the measurement is a client
whose shard declares 48 examples across a 37-row train split and an 11-row eval
split returning 37. `regression_test` is empty on all five: the column means
*the guard that came with the fix*, and a guard found afterwards by searching
would be a different fact under the same column name.

### `obsolete`

`obsolete` was added when the fragile and style findings — the ones the fix
pass deliberately skipped — were re-checked against the tree weeks later. Some
had closed because their subject was deleted, and recording that needed a value
that was neither `fixed` nor an empty cell someone forgot to fill.

Eight of the nine name code that is gone: the figure tooling (`P07-F10`,
`P10-F29`, `P10-F30`, `P10-F34`), the sweep script (`P08-F07`), a results
report (`P08-F09`), the post-run checklist (`P08-F10`), and the `mnist_ten_label`
generator config (`P06-F08`), whose IID arm is now `mnist_iid.yaml` at
`strategy: iid` — the change the finding asked for. The ninth, `P08-F08`, is the
one whose *claim* rather than whose code was removed: it says "rounds to
target" is promised and never computed, and the only thing that promised it was
that results report. Nothing computes it and nothing claims it, so there is no defect
left to fix.

## What is left, and why

**Every result-affecting row is closed** — 9 of 9 `wrong-results` and 34 of 34
`silent-degradation` — and so is every `fragile` row: 104 of the 109 close, and
the 5 below are the whole remainder. **Every one of them is open by decision.**
That is a claim about the record, not a mood, so this section names each one
and its reason; a row that is merely unfinished does not belong here and would
have to be written down as such.

For four of the five the audit itself measured the defect as unable to fire on
this tree, and the reason below is the report's own rather than a later
rationalisation; the fifth is a group of minor items triaged together:

| Row | What it is | Why it stays open |
|---|---|---|
| `P01-F09` | FedLALR skips the whole AMSGrad step for a parameter with no gradient, where the paper still decays `m` and `v` and moves `x` | No model here leaves a trainable parameter out of the forward graph, so the branch cannot be reached |
| `P06-F09` | `partition_iid` is the only partitioner returning unsorted indices; `dirichlet` iterates classes in first-appearance order | Verified harmless: `split_client_indices` sorts both halves before anything is written, so no shard ever differs. Consistency warts, not defects |
| `P06-F10` | Per-client split seeds collide with the test-partition seed at client position 994 — every dataset with ≥ 995 clients, which is all three shipped MNIST sets | Verified harmless: the two seeds feed different RNG types drawing different quantities, so no stream is shared. Consecutive per-client seeds were also measured as uncorrelated when the row was triaged |
| `P09-F10` | `seed_everything` normalises the seed to 32 bits for numpy and torch but not for `random` | `experiment.seed` is validated non-negative and every config uses 42; the paths diverge only above 2^32, where neither number is the correct one |
| `P10-F35` | Four generator minor items: `splitlines()` keeping blank lines, a float-image heuristic on a path nothing takes, column validation skipped when `column_names` is `None`, a relative-path check running before `$VAR`/`~` expansion | Four unrelated minor items kept as one row and triaged as a group. The report marks the float-image heuristic as unused by the path the generator takes, and gives no confidence note for any of the four — this row is one of the five whose `confidence` cell is empty |

Leaving them is a judgement that can be revisited; it is not a claim that they
are not defects. `P06-F09` and `P06-F10` both carry a `suggestion` block in
their report, so the shape of the fix is already written down.

Seven more things are unfinished without being open rows, and none shows up in
the counts above:

- `POST-F03` is open, and is not one of the 109. The FedOpt moments `m` and `v`
  inherit the model state's dtype and are carried for the whole run; under a
  `bf16` model the second moment holds ~10% relative error against the same
  deltas accumulated in `float32`. Post-census rows are excluded from
  *Coverage* by construction, so an open one is invisible there.
- `POST-F11` is open, by decision, and is not one of the 109 either. A staged
  run's `run.json` records the source manifest and the staging *request*, so it
  cannot say whether the run read the staged copy or whether staging was
  skipped.
- `POST-F15` is open, by decision, and is not one of the 109 either. One
  `local_iterations` value is K updates under `single_batch`,
  `frozen_batch_gradients` and `full_gradient` and K × B under
  `sequential_epoch`, which a rule that leaves `update_mode` unset also runs, so
  arms of different update shape are not comparable at equal
  `local_iterations`. The modes are left as they are.
- `POST-F18` is open, by decision, and is not one of the 109 either.
  `client.frozen_gradient_weighting` is required under every `update_mode` and
  read under one, so all 61 shipped configs that take a mode carry a value that
  changes nothing. Fixing it touches every one of them. Its engine half is
  `POST-F21`, which is fixed.
- `POST-F20` is open, and is not one of the 109 either. drift-quad's README
  reports 61 of 861 grid points tripping the divergence guard, and the grid it
  quotes gives 540 per dial setting and 1620 in all; the sweep's artifacts are
  not in the tree, so which number is wrong cannot be settled.
- `P10-F33`'s row is closed, but only two of its three clauses were
  fixed. The third — `cudnn_benchmark: true` silently ignored under
  `deterministic: true` — was closed by argument rather than by a change:
  `run.json` records the flags torch ends up holding rather than the ones the
  config asked for, so the divergence is in the record. A clause closed that
  way is a decision like the three above that say so, and the row's `status` cannot say so.
- `POST-F09`'s row is closed, but the conversion it describes stops short of
  the tree, by decision. The client, server, model and task code refuse from
  inside a running experiment, where whoever reads the error is already reading
  Python, so those raises keep their tracebacks. Counted as raise sites whose
  class is `ValueError` or `FileNotFoundError` and not `RunRefused`, every one
  reachable from `fedbrew run`: 89 in `fedbrew/clients` (10 modules), 31 in
  `fedbrew/servers` (4), 60 in `fedbrew/models` (8) and 15 in `fedbrew/tasks`
  (2), none of them yet sorted into refusals and defects. In `fedbrew/core`, 44
  remain, and each was read. Thirty are contracts with fedbrew's own code or
  recorded outcomes rather than refusals: the loop's evaluation contracts and
  round-count guard, the registry's registration API, `torch_utils`'
  aggregation checks and `NonFiniteStateError` among them. Then
  `default_override_namespace`; `build_dataset_provenance`'s internal raise;
  the four metrics-CSV loaders, whose callers catch them; the registry's
  duplicate-name refusal, which is input when two configured extensions claim
  one name and a bug otherwise, where refusing only the first is deferred; and
  `federated_state`'s seven checks, which run on checkpoint state at resume and
  on client payloads in aggregation alike. Their decided shape is to refuse at
  the server's checkpoint readers, `FedAvgServer.initialize` and `load_state`,
  not at the raise sites, and it is deferred with the server code. `fedbrew
  generate` and the `prepare-*` commands catch nothing, so the 209 such sites
  under `fedbrew/data` that a run does not import still print a traceback
  there, which is a separate decision. The scan in
  `tests/test_a_refusal_is_a_message_not_a_traceback.py` covers only the
  converted modules, so nothing checks a new raise anywhere else.

## Empty cells

An empty cell means the value could not be recovered, with one exception:
`fix_commit` is empty by decision. None has been filled with a plausible
substitute. These counts are over all 139 rows, census and post-census together,
because they describe the CSV's columns; *Coverage* above counts the 109 census
rows alone.

- `confidence` — 5 rows (`P10-F31` … `P10-F35`). Those findings carry no
  confidence note.
- `fix_commit` — all 139 rows; see *Why `fix_commit` is empty*.
- `regression_test` — 38 rows. Five are the open post-census rows
  (`POST-F03`, `POST-F11`, `POST-F15`, `POST-F18`, `POST-F20`), which have no fix and so no guard. The
  other 33 are census rows, and 26 of them had no fix commit to take a guard
  from. The other 7 had one: five (`P08-F03`, `P08-F04`, `P10-F01`, `P10-F11`,
  `P10-F12`) were guarded by figure-tooling tests that were deleted along with
  the code they guarded, and `P07-F02` and `P11-F02` were fixed by commits that
  added and changed no test at all.
- `location`, `component`, `severity`, `summary` — never empty.

`regression_test` was attributed at commit granularity, while the history that
held the fix commits was available: it is the test module the fix commit's own
`Tests:` note named, or the single `tests/` file that commit added when it had
no such note, checked against the module's docstring. Where one commit fixed two
findings, both rows carry that commit's guard.

## Locations

`location` names **one** path — the first path the finding cites that still
exists in this tree. Findings usually cite several; the CSV keeps one, and the
`summary` says what the defect is well enough to find the rest.

Line numbers are deliberately omitted. Every one in the audit predates the
package reorganisation and would point at the wrong line today.

The audit ran against the pre-reorganisation layout, so its paths were
translated forward by these prefix rules before being written into the CSV:

| Audit-time prefix | Current |
|---|---|
| `fl_framework/` | `fedbrew/` |
| `data/scripts/` | `fedbrew/data/` |
| `clients/` | `fedbrew/clients/` |
| `servers/` | `fedbrew/servers/` |
| `models/` | `fedbrew/models/` |

That move was the reorganisation that put `clients/`, `servers/`, `models/`,
`data/scripts/` and `fl_framework/` into a single `fedbrew/` package and renamed
`fl_framework`/`fl-codebase` to `fedbrew` throughout.

96 of the 109 translate to a file that exists. The other **13 cite code that has
since been deleted, not renamed**, and are marked `removed:` followed by the
audit-time path — for example `removed:figspec.py`. They are not broken paths;
they are the record of where the defect was. The removals, as path patterns:

| Path pattern | What went |
|---|---|
| `figspec.py`, `csv_plots/**`, `docs/experiment_checklist*` | The experiment write-ups, figure tooling and site-specific scripts |
| `csv_plotter.py` | The already-retired CSV plotter, dropped on the way past by `P10-F08`'s fix |
| `SLURMs/fedopt_femnist_packed.sh` | A sweep script, when `SLURMs/` was reduced to two portable, documented example sweeps |
| `reports/*` | The two measured-results reports, untracked |

The table names what went rather than the commit that took it; *Why
`fix_commit` is empty* says why.

Two of those locations named their file in full, and the names described work
that is not published: a results report for an LLM experiment, and a comparison
figure config whose filename carried the name of the method it plotted beside
the baselines. Both are generalised to the kind of file they were —
`reports/results-report.md` and
`csv_plots/csv_configs/femnist/comparison-figure.yaml` — which keeps the record
of where each defect was without publishing what was being worked on. `P07-F10`,
`P08-F09` and `P10-F12` carry them.

## Why `fix_commit` is empty

`fix_commit` held, for each finding, the short hashes of the commits that named
it as fixed. **That mapping existed in a development history that is not
published.** None of those hashes resolves in the published repository or in any
clone of it, and a hash that resolves against nothing reads as checkable while
checking nothing. So the column is empty on every row, and what the file records
about a fix is what `status` says.

Three things went with the hashes, because each meant something only while the
hashes did:

- **Two `status` values.** `fixed-report-level` said a commit cited the finding's
  report as a whole rather than the finding; `fixed-unattributed` said the
  mechanism was gone and no commit cited the finding. Both described how a
  closure was attributed to a commit, and with no commit to show, that is a
  distinction nobody can act on. Both are now `fixed`.
- **The commit column of the removals table** under *Locations*. Every
  `removed:` path is still covered by a pattern there, which is what stops the
  marker being applied to anything; the table says what went instead of which
  commit took it.
- **The commits the location column was translated through.** The prefix rules
  under *Locations* are the translation, and they stand without them.

What is lost is the link from a finding to the change that fixed it. What is
kept is the guard that came with the fix, in `regression_test`, which is in the
tree and runs.

## The guard

`tests/test_findings_manifest.py` checks that the file parses with exactly the
header above, that the census is 109 rows split 9 / 34 / 47 / 19 — post-census
rows excluded, so later work cannot inflate it, and no id retired under *Rows
removed with a component* back in it — that every `pass` is two digits or
exactly `post` and every post-census row says so in its id and its title, that
every severity is one of the four classes, that every non-empty `location`
either exists or carries the `removed:` marker with a pattern in the removals
table covering it, and that every non-empty `regression_test` names a file that
exists. It checks `status` the same way: every value is one of the three above,
the counts in the *Coverage* table are the file's own, and every sentence that
states how many census rows are open agrees with the CSV, as does the table that
names them. It checks that `fix_commit` is empty on every row, so a hash cannot
come back. It also reads the corrections table above — the one section here
that quotes another file rather than the CSV — and checks that each row names a
README in the tree and that every value it quotes is still in that README, since
nothing else holds the two files together.
