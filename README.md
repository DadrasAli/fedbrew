<p align="center">
  <img src="assets/fedbrew-logo.svg" alt="fedbrew" width="360">
</p>

# fedbrew

A modular federated-learning benchmark framework: a single-process simulator
that runs a federated training loop over a partitioned dataset, records what
each round did, and writes a self-describing run artifact.

**Documentation: [docs](docs/00-index.md).** Sixteen chapters, one
per area, each self-contained and each ending with a section written for
automated agents. This file is an overview and a pointer; every fact about the
system lives in a chapter, where a test keeps it true.

## What this implements

One federated round is: sample clients, broadcast the global state, run each
selected client's local update in turn, fold the results into a new global
state, evaluate, checkpoint. The pieces are pluggable through a registry, and
the loop itself does not know which algorithm it is running.

| Layer | Contract | Where |
| --- | --- | --- |
| Server strategy | `configure_round`, `aggregate` / `aggregate_stream` | `fedbrew/servers/` |
| Client update | one local update from a received state | `fedbrew/clients/` |
| Task adapter | model/loss/metrics for a modality | `fedbrew/tasks/` |
| Dataset | per-client train/val/test shards | `fedbrew/data/` |
| Model | architecture builders | `fedbrew/models/` |

Three properties shape everything else:

- **Clients run one at a time, on purpose.** A round is a little GPU
  arithmetic wrapped in a lot of Python, so parallel clients contend rather
  than fill gaps. Concurrency across processes and across threads was tried
  during development and measured slower than serial in both cases, before
  that code was removed. Chapter 11 §2.
- **Peak model memory does not grow with participation rate**, because
  results are folded into a running weighted sum as they arrive: two model
  states, never one per participant.
- **Data generation is a separate step from training.** A generator writes
  immutable shards plus a manifest; a run reads the manifest.

## Install

Needs Python 3.10 or newer.

```bash
pip install -e .
```

Extras (`dev`, `vision`, `llm`), offline and HPC installation, what the
install costs on disk, and the environment variables are in
[chapter 02](docs/02-installation.md). `torchvision` is in the `vision` extra
rather than the core install, because it pins `torch` to one exact release and
would replace a build you matched to your driver.

## Quickstart

About a minute on CPU, no download. Not on PyPI yet, so install from a clone.

```bash
pip install -e .
fedbrew generate synthetic
fedbrew run synthetic/fedavg
ls outputs/synthetic/fedavg/
fedbrew report --run-dir outputs/synthetic/fedavg
```

`synthetic` / `synthetic/fedavg` are short names for
`--config data/configs/synthetic.yaml` / `--config configs/synthetic/fedavg.yaml`,
which still work as written. [Chapter 03](docs/03-quickstart.md) walks the
same pipeline command by command, with its own smaller fixture and real
recorded output.

## Development

```bash
pip install -e ".[dev]"
python -m pytest
ruff check tests tools fedbrew examples
ruff format --check tests tools fedbrew examples
```

`python -m pytest`, not bare `pytest`: some test modules import helpers from
`tests/`, and only the module form puts the working directory on the path.

Those four are the gate, and CI runs all of them on every push plus the
`quickstart`-marked end-to-end guard the default suite excludes. mypy arrives
with the `dev` extra, is configured in `pyproject.toml`, and does not gate. Tests that
need an optional extra skip when it is absent.

[CONTRIBUTING.md](CONTRIBUTING.md) is the entry point for changing this
codebase; it points at the chapter covering the working contract, guard
discipline, the suite, and the changes where the obvious fix would have
introduced a real defect. [FINDINGS.csv](FINDINGS.csv) is an internal audit's
findings, one row each, read with [FINDINGS.md](FINDINGS.md).

## Licence and citation

MIT. See [LICENSE](LICENSE).

If you use this software, please cite it. `CITATION.cff` at the repository root
carries the machine-readable record; GitHub renders it into APA and BibTeX from
the "Cite this repository" button.

```bibtex
@software{fedbrew,
  title   = {fedbrew},
  author  = {Dadras, Ali},
  version = {0.0.1},
  license = {MIT},
  year    = {2026}
}
```
