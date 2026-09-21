# Data generation configs

An inventory of what is here. **How generation works — the four partition
strategies, the manifest format and the split guarantees — is
[chapter 05](../../docs/05-data-and-partitioning.md).**

`fedbrew generate --config data/configs/<name>.yaml` writes
`data/generated/<name>/manifest.json`. **The filename matches the dataset
directory it produces** — that correspondence is the provenance record, so a
config here is kept even when nothing references it, and renaming one means
regenerating the dataset.

The MNIST set is one config per partition strategy the code implements —
`iid`, `label_skew`, `dirichlet`, `quantity_skew` — so the four show four
different things. It was previously three `label_skew` configs differing only
in `labels_per_client`; the two retired values are recorded in
`mnist_one_label.yaml` beside the key they set, because neither `manifest.json`
nor `partition_stats.json` stores a strategy's parameters and the config file
is the only place they live.

| Config | Produces | Partitioning |
|---|---|---|
| `mnist_iid.yaml` | `mnist_iid` | 1000 clients, uniform draw — the IID baseline |
| `mnist_one_label.yaml` | `mnist_one_label` | 1000 clients, `label_skew` at 1 digit each — pathological non-IID |
| `mnist_dirichlet.yaml` | `mnist_dirichlet` | 1000 clients, `dirichlet` at α=0.1 — graded label skew |
| `mnist_quantity_skew.yaml` | `mnist_quantity_skew` | 1000 clients, `quantity_skew` — all ten digits, lognormal amounts |
| `femnist_natural.yaml` | `femnist_natural` | Natural `writer_id` federation, no synthetic partitioning |
| `synthetic.yaml` | `synthetic` | 50 clients, `dirichlet` at α=0.3 — the README's quickstart |
| `synthetic_classification.yaml` | `synthetic_classification` | IID synthetic |
| `synthetic_label_skew.yaml` | `synthetic_label_skew` | Label-skewed synthetic |
| `medmcqa_qwen05b_20clients.yaml` | `medmcqa_qwen05b_20clients` | 20 clients, one per medical specialty |
| `oasst1_qwen05b_20clients.yaml` | `oasst1_qwen05b_20clients` | 20 clients, Qwen2.5-0.5B-Instruct tokenizer |
| `oasst1_qwen05b_base_20clients.yaml` | `oasst1_qwen05b_base_20clients` | Same split, base-model tokenizer |
| `oasst1_qwen05b_4clients_hpc.yaml` | `$FL_DATA_ROOT/oasst1_qwen05b_4clients` | 4 clients, HPC paths |
| `examples/pl-1d.yaml` | `examples/pl-1d` | 8 analytic clients, scalar PL objective — generator from `examples/pl-1d/problem.py` |
| `examples/drift-quad.yaml` | `examples/drift-quad` | 8 analytic clients, κ=100 ζ=1 — generator from `examples/drift-quad/problem.py` |
| `examples/drift-quad-rate.yaml` | `examples/drift-quad-rate` | the same, ζ=0 |
| `examples/drift-quad-floor.yaml` | `examples/drift-quad-floor` | the same, κ=1 |
| `examples/fed-lasso.yaml` | `examples/fed-lasso` | 8 analytic clients, λ=0.05 ζ=0.4 — generator from `examples/fed-lasso/problem.py` |
| `examples/fed-lasso-smooth.yaml` | `examples/fed-lasso-smooth` | the same, λ=0 — the control that attributes the floor |
| `examples/simplex-lsq.yaml` | `examples/simplex-lsq` | 8 analytic clients, optimum outside the simplex — generator from `examples/simplex-lsq/problem.py` |
| `examples/simplex-lsq-feasible.yaml` | `examples/simplex-lsq-feasible` | the same, optimum inside it — the control |
| `examples/nonconvex-simplex.yaml` | `examples/nonconvex-simplex` | 8 analytic clients, K₅ plus a decoy star — generator from `examples/nonconvex-simplex/problem.py` |

## Subdirectories

`assets/` holds raw-dataset download manifests (`dataset_identifier`,
`revision`, `cache_dir`). They have no `dataset.name` or `output_dir` and do
not partition anything — they are consumed by `fedbrew prepare-oasst1` and by the
`dataset_preparation_config:` key of the SFT configs above.

`dev/` holds tiny corpora used only by the test suite.

`examples/` holds the generator configs for the problems in
[`examples/`](../../examples/), whose generators are defined outside the
package and reach `fedbrew generate` through `dataset.extensions`. Each one's
data has a known answer, which its manifest carries under `reference`.
