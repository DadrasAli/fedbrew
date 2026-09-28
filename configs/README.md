# Experiment configs

An inventory of what is here. **What the keys inside these files mean is
[chapter 04](../docs/04-configuration.md)**, which lists every one with its
type, default and validator, and is checked against the code.

One directory per dataset; one file per method. The path is the experiment:
`configs/femnist/fedprox.yaml` is FedProx on FEMNIST. Nothing else in the
filename, because the directory already says which dataset it is.

`femnist/` and each directory under `examples/` also hold a `_base.yaml`: the
keys every arm of that family shares, written once. An arm file names it under
`extends:` and states only its own keys -- the rule, its rates, its tags -- so
an arm file is not the whole run. `fedbrew config show configs/femnist/fedprox.yaml`
prints one resolved, and `run.json` records it. A family base is not a run
config and is refused by `fedbrew run`; chapter 04 §2.2 has the rules.

```
femnist/     fedavg  fedavg_ft  fedprox  scaffold  delta_sgd  fedlalr
             fedadam  fedyogi  fedadagrad  centralized
mnist/       fedavg  scaffold  centralized
synthetic/   fedavg                                     (the README's quickstart)
openimage/   fedavg                                     (needs data this repo does not ship)
medmcqa/     fedavg_lora  fedavg_fullmodel
oasst1/      fedavg_instruct  fedavg_base  fedavg_lora_hpc
examples/    pl-1d/                     (7 arms)
             drift-quad/  drift-quad-rate/  drift-quad-floor/      (8 arms each)
             fed-lasso/  fed-lasso-l2/  fed-lasso-smooth/  (9 arms each, 1 arm)
             simplex-lsq/  simplex-lsq-feasible/               (8 arms, 3 arms)
             nonconvex-simplex/                                       (8 arms)
             fed-logistic-l1-a9a-lambda0.001/                         (1 arm)
             fed-logistic-l1-a9a-lambda0.03/                          (1 arm)
             fed-logistic-l1-a9a-lambda0.05/                          (1 arm)
             fed-logistic-l1-gisette-lambda0.001/                     (1 arm)
             fed-logistic-l1-gisette-lambda5e-4/                      (1 arm)
             fed-logistic-l1-gisette-lambda5e-5/                      (1 arm)
             fed-logistic-l1-ijcnn1-32-lambda0.01/                    (1 arm)
             fed-logistic-l1-synthetic-1000-lambda0.001/              (1 arm)
             fed-logistic-l1-synthetic-1000-lambda0.03/               (1 arm)
             fed-logistic-l1-synthetic-kappa1-lambda0.01/             (1 arm)
             fed-logistic-l1-synthetic-kappa10-lambda0.01/            (1 arm)
             fed-logistic-l1-synthetic-kappa100-lambda0.01/           (1 arm)
             fed-logistic-l1-synthetic-lambda0.03/                    (1 arm)
```

`examples/` is one directory per *dial setting* rather than per dataset,
because those examples turn a dial by generating different data:
`drift-quad-rate/` and `drift-quad-floor/` hold the same eight arms as
`drift-quad/` against a second and a third generated dataset, and the `-l2`,
`-smooth` and `-feasible` directories are controls of the same shape. The
`fed-<loss>-<penalty>-<dataset>-lambda<λ>` directories are
`examples/fed-logistic-l1`'s settings, one problem on one dataset each, with
one untuned FedAvg arm. Their task, model and generator are defined
outside the package, in each example's `problem.py`, which every config names
— the key that does that is [chapter 04](../docs/04-configuration.md)'s, and
what it is for is [chapter 12](../docs/12-extending.md)'s.

`fedbrew run --config configs/<dataset>/<method>.yaml` runs any of them once its
dataset exists. Every arm above has a generator config under `data/configs/`
except `openimage/`: there is no `openimage` generator, and its manifest has to
be produced outside this repository from FedScale's OpenImage partition. The
config says so, at the line naming the manifest. The SLURM
sweep scripts in `SLURMs/` read one of these as their base and write the
per-arm variants, resolved, to `data/generated/slurm_configs/`, so a
hyperparameter change that should apply to every arm belongs in the base file
here -- or, for a family, in its `_base.yaml`.

## The other three directories

| Directory | Contents | Runnable? |
|---|---|---|
| `llm_assets/` | Model download/preparation manifests, referenced by `model.preparation_config` in the LLM configs. `_hpc` variants write to `$FL_CACHE_ROOT`. | No — consumed by `fedbrew prepare-llm` |
| `dev/` | Small fixtures for tests, smoke runs, and the getting-started docs. Not experiments; nothing here is meant to be reported. | Yes, in seconds |
| `reference_evaluation.yaml` | Every `evaluation:` and `reporting:` option in one runnable 12-round MNIST job, documented inline. Read this before editing a config's evaluation block. | Yes, on CPU |

## Naming

`_lora` / `_fullmodel` on the LLM configs is what is federated: a LoRA adapter
(~8 MB checkpoints) or all 494M parameters (~2 GB). `_instruct` / `_base` is
which Qwen2.5-0.5B checkpoint the run starts from. `_hpc` means the config
resolves paths through `$FL_DATA_ROOT` / `$FL_CACHE_ROOT` rather than the
in-repo `data/` tree.
