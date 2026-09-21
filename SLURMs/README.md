# SLURMs/

Two example submit scripts. There is no built-in sweep command: a sweep is a
grid of generated configs submitted as independent SLURM jobs, and these show
the two ways to do that.

| Script | One job per... | Use when |
|---|---|---|
| `example_sweep.sh` | configuration | The default. Each run gets a whole GPU. |
| `example_sweep_packed.sh` | pack of `PACK_SIZE` configurations | Runs leave the GPU mostly idle — which is the normal case for cross-device federated simulation, where each round is a small amount of GPU maths wrapped in a lot of Python. |

Both write one generated config per grid point under
`data/generated/slurm_configs/<script>/<timestamp>_<pid>/`, so a second
submission cannot overwrite configs that queued jobs from the first have not
read yet.

Neither is meant to be run as-is. Edit the "Sweep values" and "Base config"
blocks; keep everything site-specific in the environment.

```bash
SLURM_ACCOUNT=my-account CONDA_ENV_NAME=fedbrew bash SLURMs/example_sweep.sh
```

## Environment variables

| Variable | Default | What it is |
|---|---|---|
| `SLURM_ACCOUNT` | **required** | Your accounting/project code — the argument to `#SBATCH -A`. No default on purpose: a wrong one spends someone else's allocation. |
| `SLURM_TIME` | `48:00:00` / `72:00:00` | Wall-time request. Size it from a previous run's `run.json` `timing` block, using `median_sec_per_round` rather than `mean_sec_per_round`. |
| `SLURM_GPUS` | `1` | GPUs per job. One is right for this workload: clients run sequentially, so a second GPU idles. |
| `SLURM_PARTITION` | unset | Partition/queue (`#SBATCH -p`). Empty accepts the cluster default. |
| `SLURM_CONSTRAINT` | unset | Node feature constraint (`#SBATCH -C`), e.g. a high-memory node type. Empty if your cluster has no such feature. |
| `CLUSTER_PYTHON_MODULE` | unset | Environment module to `module load` inside the job, e.g. a Conda distribution. Empty skips the `module purge`/`module load` pair entirely. |
| `CONDA_ENV_NAME` | unset | Conda environment holding the installed package. Empty skips activation and uses whatever Python the job inherits. |
| `LOG_DIR` | `logs_and_errs` | Where job stdout/stderr go. Git-ignored. |
| `PACK_SIZE` | `4` | *Packed only.* Runs sharing one GPU. |
| `CPUS_PER_JOB` | `16` | *Packed only.* CPU cores the scheduler gives the job, used to split the BLAS/OpenMP thread pools between packed runs. Check what your cluster allocates per GPU. |
| `MAX_PACKABLE_PR` | `1.0` | *Packed only.* Participation rates at or above this are refused rather than packed. Set to `2` to disable. |

The `FL_*` path variables are not read by these scripts, and are read by less
of the package than their names suggest: `fedbrew check-hpc` reports all four,
`fedbrew inspect-data` resolves a manifest path through `FL_DATA_ROOT`, and any
config path containing one is expanded by `os.path.expandvars`. Nothing else
reads them. Export them in your shell or job environment and reference them
from config paths — [chapter 02 §3](../docs/02-installation.md) has the full
table.

## Sizing a pack

Raising `PACK_SIZE` is not free, and GPU memory is rarely the binding
constraint. Measure one run's GPU memory, host RSS and GPU utilisation, then
leave headroom on all three. Host RAM usually runs out first: a round holds
every participating client's shard resident, so memory scales with
`participation_rate` × client count — and packing multiplies that by
`PACK_SIZE`. `MAX_PACKABLE_PR` exists for exactly that reason.
[Chapter 11 §3](../docs/11-performance-and-cost.md) explains why the loop holds
them.

The packed script sorts runs by an estimated per-round cost
(`local_iterations × participation_rate`) before chunking, so each pack holds runs
of similar duration. Mixing a fast configuration with a much slower one wastes
the tail of the job: the fast runs exit and the pack holds a GPU for a single
remaining run.

## Resume

The packed script decides `--resume-latest` inside the job, not at submit time,
by checking for `latest.pt` or `round_*.pt` under the run's checkpoint
directory. This is deliberate: a requeued attempt must pick up checkpoints its
earlier attempt wrote, which do not exist when the script runs. The flag errors
out when no checkpoint is present, so it cannot be passed unconditionally, and
`best.pt` alone must not trigger a resume — resuming reads `latest.pt` and
`round_*.pt` only.
