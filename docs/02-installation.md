# 02 — Installation

What to install, which extras you need for which datasets, how to install where
there is no network, and how to check the result.

## 1. Requirements

| Requirement | Value | Source |
| --- | --- | --- |
| Python | `>=3.10` | `pyproject.toml` |
| Core dependencies | `PyYAML`, `rich`, `torch` | same |

`torch` is unpinned, and the core install will not move one you already have:
install the build that matches your CUDA driver first, then install fedbrew.

**That promise stops at the `vision` extra**, and the reason is worth knowing
before you install rather than after. `torchvision` pins `torch` to one exact
version — release 0.28.0 requires `torch (==2.13.0)` — so anything that pulls
torchvision drags torch to the release torchvision names, uninstalling yours
and adding the default PyPI CUDA build with it. pip resolves that by moving
torch, not by choosing an older torchvision.

A local build tag survives, a different release does not: `==2.13.0` matches
`2.13.0+cpu` and `2.13.0+cu130`, but not `2.14.0+cpu` and not `2.12.1+cu130`.
So installing the *exact* release torchvision pins keeps your build; asking an
index for its newest torch usually does not, because an index's newest is
usually ahead of what torchvision pins.

On some indexes there is no version that works. Newest torch per index,
read 2026-09-02:

| Index | Newest torch | Carries torchvision 0.28.0's `2.13.0`? |
| --- | --- | --- |
| `cu118` | 2.7.1 | **no** |
| `cu121` | 2.5.1 | **no** |
| `cu128` | 2.11.0 | **no** |
| `cu126`, `cu130`, `cpu` | 2.14.0 | yes |

If your driver needs `cu118`, `cu121` or `cu128`, there is no torch that both
matches it and survives `pip install -e ".[vision]"`. Generate the image
datasets in a separate environment and point a run at the shards, or accept the
generic build for the generation step only. Nothing warns you at install time;
pip prints its usual lines and exits 0.

This is why torchvision is an extra and not a core dependency: it is the only
thing in the graph that pins torch to one release, and the core install, the synthetic
problem, the quickstart, the LLM arms and the whole test suite all run without
it.

## 2. Install

```bash
pip install -e .
```

That installs the core dependencies and the single `fedbrew` console script.
It is enough for the synthetic problem, MNIST and the quickstart in chapter 03.

Budget **several GB of disk** for it. `torch`'s default PyPI build carries the
CUDA runtime as a set of `nvidia-*` wheels whether or not the machine has a
GPU, and those are most of the weight. One measurement, on one machine
(2026-09-02, Linux x86-64, Python 3.11, a fresh `venv`, `torch` resolving to
2.13.0): **4.8 GB** of environment, of which 2.7 GB was `nvidia-*` and 1.2 GB
`torch` itself. Treat that as an order of magnitude, not a figure to plan a
quota against — it moves with the platform, the Python version and whatever
pip resolves on the day, and a CPU-only build of the same release measured
1.1 GB. Installing from the CPU wheel index first does not by itself avoid the
CUDA wheels; see the known failure modes below. The `dev` extra adds a few MB;
the `llm` extra is the one that adds model-sized downloads later.

### 2.1 Extras

Three groups and one alias. Install only what an arm needs — the LLM group is
large, and `vision` is the one that will move your `torch` (§1). Version bounds
live in `pyproject.toml`; they are not restated here, where they would drift.

| Extra | Command | Pulls | Needed for |
| --- | --- | --- | --- |
| `dev` | `pip install -e ".[dev]"` | `pytest`, `pytest-xdist`, `hypothesis`, `ruff`, `mypy`, `numpy` | running the test suite, across processes with `-n` (chapter 13 §2.2), and the two lint gates; `mypy` is installed but does not gate (chapter 13 §1) |
| `vision` | `pip install -e ".[vision]"` | `torchvision`, `datasets`, `huggingface-hub`, `Pillow` | generating MNIST, CIFAR-10 and FEMNIST |
| `llm` | `pip install -e ".[llm]"` | `transformers`, `datasets`, `huggingface-hub`, `peft` | the causal-LM arms, LoRA, MedMCQA, OASST1 |
| `femnist` | `pip install -e ".[femnist]"` | `fedbrew[vision]` | an alias for `vision`, kept so existing scripts keep working |

One extra covers all three image datasets rather than one per dataset: they
need the same packages, so splitting them would be a choice with nothing behind
it. `femnist` remains as an alias; new work should say `vision`.

Combine them with commas: `pip install -e ".[dev,vision]"`.

The property-based aggregation tests skip when `hypothesis` is absent rather
than failing, so the suite runs without it — it just checks less.

`numpy` is in `dev` for one `tools/` script, not because the package or the
suite needs it: every use in `fedbrew/` is a guarded lazy import that records
"not available" and carries on, and `torch` does not require it either. It was
declared nowhere at all until recently and arrived through `torchvision`, which
is why CI now installs with no extras and runs the suite -- the only way to
notice a dependency that is only ever satisfied by accident.

### 2.2 Offline and HPC installation

Compute nodes usually have no outbound network. Install on a login node, into
an environment on shared storage that the compute nodes can read:

```bash
conda create -p /path/to/shared/envs/fl python=3.11
conda activate /path/to/shared/envs/fl
pip install torch --index-url <the wheel index for your CUDA version>
pip install -e ".[dev]"
```

That torch survives, because nothing in the core dependencies or the `dev`
extra constrains it. Adding `vision` to the last line is what would replace
it — §1. If the arm you are running needs image datasets, generate the shards
in a throwaway environment and let the job read them: generation and training
are separate steps precisely so they need not share an environment.

The LLM arms additionally need their model and dataset snapshots on disk before
the job starts, because the job itself cannot download. Prepare them on the
login node:

```bash
fedbrew prepare-llm --config configs/llm_assets/<model>.yaml
fedbrew prepare-oasst1 --config configs/llm_assets/<dataset>.yaml
```

Chapter 05 covers what those write and how a config points at it.

A missing snapshot does **not** cause a download or a hang: `hf_causal_lm`
requires `local_files_only`, requires `model.asset_manifest`, and refuses to
build when either asset directory is absent, naming the `prepare-llm` command
to run. `fedbrew run --validate-only` checks the same thing before the job
starts, so a typo in the path is caught at preflight rather than minutes in.

`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are still worth setting — they
stop any library reaching for the network on its own — and both are recorded in
`run.json` so a run says whether it was offline.

## 3. Environment variables

None are required. All are read, never written — the one exception is
`CUBLAS_WORKSPACE_CONFIG`.

| Variable | Read by | Meaning |
| --- | --- | --- |
| `FL_DATA_ROOT` | `inspect-data`, `check-hpc` | Where generated manifests and shards live. Used to resolve a manifest path. |
| `FL_OUTPUT_ROOT` | `check-hpc` | Where run outputs are written. |
| `FL_CACHE_ROOT` | `check-hpc` | Downloaded/raw dataset cache. |
| `FL_LOCAL_SCRATCH` | `check-hpc` | Node-local scratch, for job-local staging only. |
| `COMMON_DATASETS` | `list-common-datasets`, CIFAR-10 root resolution in `fedbrew/data/generate.py` | Optional site-wide read-only dataset directory. |
| `CUBLAS_WORKSPACE_CONFIG` | `fedbrew/core/runtime_setup.py` | **Set** to `:4096:8` when `runtime.deterministic` is true, and only if not already set. Required for deterministic CUDA matmuls. |
| `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` | `fedbrew/core/run_metadata.py` | Not written by the package — read and recorded in `run.json`. |

Config paths run through `os.path.expandvars` and `os.path.expanduser`
(`fedbrew/core/paths.py`), so `output_dir: $FL_OUTPUT_ROOT/my_run` works in any
config.

A path naming a variable the process did not export is treated as **unset**,
not as a literal directory: `has_unexpanded_env` catches a surviving `$NAME`
after expansion, so nothing creates a directory called `$FL_LOCAL_SCRATCH`
beside the checkout.

Two rules worth stating: write generated data and final outputs to project
storage, not `$HOME`; and use node-local scratch only for job-local staging,
never as the destination for final outputs. The runtime does not delete staged
data — node-local cleanup is the cluster's job.

## 4. Verifying the install

Four checks, cheapest first.

```bash
# 1. The console script resolves and lists its eleven subcommands.
fedbrew --help

# 2. The environment a compute node actually sees: torch, CUDA, the FL_* roots.
fedbrew check-hpc

# 3. Config loading, component construction and preflight, without training.
fedbrew run --config configs/dev/smoke.yaml --validate-only

# 4. The full suite. Needs the dev extra.
python -m pytest
```

`fedbrew check-hpc` prints the SLURM variables, `CUDA_VISIBLE_DEVICES`, torch's
view of CUDA, and the status of each `FL_*` path that is set. Run it **inside**
a job, not on the login node — the point is what the compute node sees.

## 5. The command surface

One console script with eleven subcommands (`fedbrew/cli/dispatch.py`).

| Subcommand | Does |
| --- | --- |
| `run` | Run an experiment. |
| `generate` | Generate federated data manifests. |
| `report` | Create Markdown reports from a run directory. |
| `inspect-data` | Validate and inspect one generated dataset. |
| `cleanup` | Remove runtime artifacts, preserving selected paths. |
| `check-hpc` | Print local runtime and path information. |
| `prepare-llm` | Prepare a pinned Hugging Face causal-LM for offline use. |
| `prepare-oasst1` | Prepare pinned OASST1 snapshots for offline use. |
| `eval-medmcqa` | Multiple-choice accuracy on held-out MedMCQA, from a checkpoint. |
| `eval-base-model` | Evaluate the frozen pretrained base model on a run's global test set. |
| `list-common-datasets` | List top-level folders under `$COMMON_DATASETS`. |

Each dispatches to its target module's own `main()` with the remaining argv
untouched, so `fedbrew run --help` is the runner's own help, not a wrapper's.

## For agents

### Paths

| Path | What it is |
| --- | --- |
| `pyproject.toml` | Python floor, core dependencies, the three extras, pytest and ruff config |
| `fedbrew/cli/dispatch.py` | `COMMANDS` — the authority on which subcommands exist |
| `fedbrew/core/paths.py` | `expand_path`, `has_unexpanded_env`, and the two resolvers |
| `fedbrew/core/runtime_setup.py` | `configure_deterministic_environment` — the only environment variable the package writes, `CUBLAS_WORKSPACE_CONFIG` |
| `fedbrew/core/run_metadata.py` | `build_hf_causal_lm_trace` — the offline flags recorded into `run.json` |
| `fedbrew/cli/check_hpc_environment.py` | what `check-hpc` reports |
| `configs/llm_assets/` | asset-preparation configs, a different schema with no `runtime` block |
| `SLURMs/` | two portable example submit scripts |

### Commands

```bash
# Prove this chapter's requirement and command tables match the code.
python -m pytest tests/test_docs_installation.py -v

# Install, including the test extra.
pip install -e ".[dev]"

# The four verification steps, cheapest first.
fedbrew --help
fedbrew check-hpc
fedbrew run --config configs/dev/smoke.yaml --validate-only
python -m pytest
```

### Invariants

1. **`COMMANDS` in `fedbrew/cli/dispatch.py` is the authority on subcommands.**
   Any `fedbrew ...` written anywhere in the tree is checked against it by
   `tests/test_cli_commands_exist.py`, and its flags by
   `tests/test_cli_flags_exist.py`.
2. **`CUBLAS_WORKSPACE_CONFIG` is the only variable the package writes**, only under
   `runtime.deterministic: true`, and only when not already set. Everything
   else is read.
3. **An unexpanded `$NAME` in a path means unset.** Never let a literal
   `$VARIABLE` become a directory name.
4. **The extras stay minimal.** `torch` is unpinned so a site can match its
   own CUDA build; do not pin it.
5. **The dev extra is optional for the suite.** `hypothesis` absence skips the
   property tests rather than failing them.

### Tests that guard this chapter

| Test | Claim |
| --- | --- |
| `tests/test_docs_installation.py` | The Python floor, dependency lists, extras and subcommand table match `pyproject.toml` and `dispatch.py`. |
| `tests/test_cli_commands_exist.py` | Every `fedbrew ...` in the tree is a real subcommand. |
| `tests/test_cli_flags_exist.py` | Every documented flag is one argparse defines. |
| `tests/test_runtime_setup.py` | `CUBLAS_WORKSPACE_CONFIG` is set only under determinism, and not over an existing value. |
| `tests/test_data_staging.py` | An unexpanded environment path resolves to "no staging". |
| `tests/test_validation_commands.py` | `--validate-only` runs the preflight and exits. |

### Known failure modes

- **Installing the package before torch on a CUDA machine.** pip resolves a
  default torch build that may not match the driver. Install torch first.
- **Expecting a hand-matched torch to survive the `vision` extra.** It does
  not, unless it is the exact release torchvision pins. §1 has the versions
  and the three indexes where no such release exists.
- **Assuming `pip install -e .` can generate MNIST.** It cannot: `torchvision`
  moved to the `vision` extra, because as a core dependency it replaced the
  torch build the reader had just chosen. `fedbrew generate` says so and names
  the extra.
- **Running `check-hpc` on the login node.** It reports what *that* machine
  sees. The GPU and the SLURM variables you care about exist only in the job.
- **Expecting the LLM arms to download at run time.** Compute nodes are
  usually offline. Prepare snapshots on the login node with `prepare-llm` and
  `prepare-oasst1` first.
- **Setting `FL_PROJECT_ROOT`.** Nothing reads it — not `fedbrew/`, not the
  example submit scripts, which stopped using it when they were reduced to two
  portable examples. It is not in the table above for that reason.
- **Pointing `output_dir` at node-local scratch.** It is cleaned by the
  cluster after the job, and the runtime does not copy results back.
- **Assuming `pip install -e .` gives you the test suite.** `pytest` is in the
  `dev` extra.
