#!/bin/bash
# ============================================================
# Example sweep: one SLURM job per configuration.
#
# There is no built-in sweep command. This script is the intended pattern:
# expand a grid of hyperparameters into one generated config per point, then
# submit each as an independent job so they run concurrently.
#
# Adapt the "Sweep values" block; everything site-specific is an environment
# variable (see SLURMs/README.md). Nothing here is specific to FedAvg or to
# FEMNIST beyond the two paths in the "Base config" block.
#
# Usage:
#   SLURM_ACCOUNT=my-account bash SLURMs/example_sweep.sh
# ============================================================

set -euo pipefail

# ------------------------------------------------------------
# Site configuration -- every cluster-specific value lives here.
# ------------------------------------------------------------

# Your SLURM accounting/project code, the argument to `#SBATCH -A`.
# Required: there is no sensible default, and a wrong one wastes someone
# else's allocation.
SLURM_ACCOUNT="${SLURM_ACCOUNT:?set SLURM_ACCOUNT to your SLURM accounting code}"

# Wall-time request. Size it from a previous run's run.json "timing" block:
# median_sec_per_round * rounds, plus margin. Prefer the median over the mean --
# round one pays one-off costs (CUDA context, shard cache fill, lazily built
# clients) that the mean spreads over the whole job.
SLURM_TIME="${SLURM_TIME:-48:00:00}"

# GPUs per job. One is right for this workload: clients run sequentially, so a
# second GPU sits idle. See SLURMs/example_sweep_packed.sh to share one GPU
# between several runs instead.
SLURM_GPUS="${SLURM_GPUS:-1}"

# Optional partition/queue name (`#SBATCH -p`). Leave empty to accept the
# cluster default.
SLURM_PARTITION="${SLURM_PARTITION:-}"

# Optional node feature constraint (`#SBATCH -C`), e.g. a high-memory node
# type. A federated round holds every participating client's shard at once, so
# a high participation rate over many clients can need far more host RAM than
# the default share of a node. Leave empty if your cluster has no such feature.
SLURM_CONSTRAINT="${SLURM_CONSTRAINT:-}"

# Optional environment module to load inside the job, e.g. a Conda/Python
# distribution module. Leave empty if your cluster does not use modules or if
# your environment is already on PATH.
CLUSTER_PYTHON_MODULE="${CLUSTER_PYTHON_MODULE:-}"

# Conda environment holding the installed package (`pip install -e .`).
# Leave empty to skip activation and use whatever Python the job inherits.
CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"

# Where job stdout/stderr are written. Git-ignored.
LOG_DIR="${LOG_DIR:-logs_and_errs}"

# ------------------------------------------------------------
# Sweep values -- edit these
# ------------------------------------------------------------
batch_sizes=(16 32)
learning_rates=(0.01 0.03 0.05)
participation_rates=(0.4 0.6 1.0)
local_iterations=(1 2 4)
data_paths=(
  "data/generated/synthetic_label_skew/manifest.json"
)

# ------------------------------------------------------------
# Base config / output root
#
# The synthetic label-skew problem is the default here so the example runs
# with no download and no optional extra -- generate it once with
#   fedbrew generate --config data/configs/synthetic_label_skew.yaml
# It is five clients of tiny tensors; it exercises the mechanics, not the
# science. The realistic case is a cross-device dataset, e.g.
#   base_config="configs/femnist/fedavg.yaml"
#   base_output="outputs/femnist/fedavg"
#   data_paths=("data/generated/femnist_natural/manifest.json")
# which needs the vision extra and a generation step of its own.
# ------------------------------------------------------------
base_config="configs/dev/synthetic_label_skew.yaml"
base_output="outputs/synthetic_label_skew/fedavg"

# ------------------------------------------------------------
# Generated configs
#
# Each submission gets its own timestamped directory. Without this, a second
# submission would overwrite config files that queued jobs from the first
# submission have not read yet, and those jobs would silently train the wrong
# hyperparameters.
# ------------------------------------------------------------
run_id="$(date +%Y%m%d_%H%M%S)_$$"
generated_root="data/generated/slurm_configs/example_sweep/${run_id}"
mkdir -p "${generated_root}" "${LOG_DIR}"

echo "Generated config root: ${generated_root}"
echo

# ------------------------------------------------------------
# Assemble the optional #SBATCH lines, omitting the ones left unset.
# ------------------------------------------------------------
extra_sbatch=""
[ -n "${SLURM_PARTITION}" ] && extra_sbatch+="#SBATCH -p ${SLURM_PARTITION}"$'\n'
[ -n "${SLURM_CONSTRAINT}" ] && extra_sbatch+="#SBATCH -C ${SLURM_CONSTRAINT}"$'\n'

env_setup=""
[ -n "${CLUSTER_PYTHON_MODULE}" ] && env_setup+="module purge"$'\n'"module load ${CLUSTER_PYTHON_MODULE}"$'\n'
[ -n "${CONDA_ENV_NAME}" ] && env_setup+='source "$(conda info --base)/etc/profile.d/conda.sh"'$'\n'"conda activate ${CONDA_ENV_NAME}"$'\n'

# ============================================================
# Submit loop
# ============================================================
submitted=0

  for bs in "${batch_sizes[@]}"; do
    for lr in "${learning_rates[@]}"; do
      for pr in "${participation_rates[@]}"; do
        for data_path in "${data_paths[@]}"; do

          data_slug="${data_path#data/generated/}"
          data_slug="${data_slug%/manifest.json}"
          data_slug="${data_slug//\//_}"

          for local_iteration in "${local_iterations[@]}"; do

            # One config file per grid point, derived from the base config.
            exp_config="${generated_root}/lr${lr}_${data_slug}_li${local_iteration}_bs${bs}_pr${pr}.yaml"
            # Patch the keys that have no CLI override into the config, and pass
            # the ones that do as flags below. A sed pattern that stops matching
            # after a schema change fails silently, so keep this list short and
            # prefer a flag wherever one exists.
            cp "${base_config}" "${exp_config}"
            sed -i "s/^  batch_size:.*/  batch_size: ${bs}/"                 "${exp_config}"
            sed -i "s/^  learning_rate:.*/  learning_rate: ${lr}/"           "${exp_config}"
            sed -i "s/^  participation_rate:.*/  participation_rate: ${pr}/" "${exp_config}"
            sed -i "s|^  path:.*|  path: ${data_path}|"                      "${exp_config}"

            output_dir="${base_output}/lr_${lr}/data_${data_slug}/local_iterations_${local_iteration}/bs_${bs}/pr_${pr}"
            tag="lr${lr}_${data_slug}_li${local_iteration}_bs${bs}_pr${pr}"

            # SLURM job names cannot usefully carry dots; they end up in log
            # filenames. Replace them so lr0.01 becomes lr0p01.
            job_name="${tag//./p}"

            # The CLI flags are redundant with the generated config above and
            # are passed anyway: run.json's "config" section then records the same
            # value twice from two sources, which catches a sed pattern that
            # silently stopped matching after a config-schema change.
            cmd="fedbrew run --config ${exp_config} \
--batch-size ${bs} \
--lr ${lr} \
--local-iterations ${local_iteration} \
--participation-rate ${pr} \
--output-dir ${output_dir} \
--tag ${tag}"

            echo "Submitting: ${job_name}"

            sbatch <<EOF_JOB
#!/bin/bash
#SBATCH -A ${SLURM_ACCOUNT}
#SBATCH --job-name=${job_name}
#SBATCH --output=${LOG_DIR}/${job_name}_%j.out
#SBATCH --error=${LOG_DIR}/${job_name}_%j.err
#SBATCH --time=${SLURM_TIME}
#SBATCH -n 1
#SBATCH --gpus=${SLURM_GPUS}
${extra_sbatch}
${env_setup}
${cmd}
EOF_JOB

            submitted=$(( submitted + 1 ))
            # Space out submissions; some schedulers rate-limit a fast burst.
            sleep 1

          done
        done
      done
    done
  done

echo
echo "Submitted ${submitted} jobs. Logs: ${LOG_DIR}/"
