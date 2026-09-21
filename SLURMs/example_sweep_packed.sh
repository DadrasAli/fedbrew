#!/bin/bash
# ============================================================
# Example sweep: several runs share one GPU ("packed").
#
# Difference vs SLURMs/example_sweep.sh: instead of one job per configuration,
# this packs PACK_SIZE configurations into a single job that share one GPU,
# launched as concurrent background processes.
#
# Why pack at all: a cross-device round does a small amount of GPU maths per
# client wrapped in a large amount of Python, so a single run can leave an
# accelerator mostly idle, with the GPU waiting on the framework. Whether
# PACK_SIZE runs sharing one GPU finish sooner than the same runs one after
# another depends on the model, the data and the node, so time one pack against
# one run before relying on it.
#
# Results are identical either way. This only changes how runs are scheduled.
#
# Before raising PACK_SIZE, measure one run's actual footprint -- GPU memory,
# host RSS, and GPU utilisation -- and leave headroom on all three. The limit
# is usually host RAM or CPU count, not GPU memory: a round holds every
# participating client's shard resident, so memory scales with
# participation_rate * client count, and packing multiplies that by PACK_SIZE.
#
# Usage:
#   SLURM_ACCOUNT=my-account bash SLURMs/example_sweep_packed.sh
# ============================================================

set -euo pipefail

# ------------------------------------------------------------
# Site configuration -- every cluster-specific value lives here.
# ------------------------------------------------------------

# Your SLURM accounting/project code, the argument to `#SBATCH -A`.
SLURM_ACCOUNT="${SLURM_ACCOUNT:?set SLURM_ACCOUNT to your SLURM accounting code}"

# Wall-time request. A pack takes roughly as long as its slowest member, not
# the sum -- but budget generously: concurrent runs do slow each other down.
SLURM_TIME="${SLURM_TIME:-72:00:00}"

# GPUs per job. Packing shares ONE GPU between PACK_SIZE runs, so this stays 1.
SLURM_GPUS="${SLURM_GPUS:-1}"

# Optional partition/queue name (`#SBATCH -p`). Empty accepts the default.
SLURM_PARTITION="${SLURM_PARTITION:-}"

# Optional node feature constraint (`#SBATCH -C`), e.g. a high-memory node
# type. More likely to be needed here than in the unpacked script, since
# PACK_SIZE runs hold their client shards simultaneously.
SLURM_CONSTRAINT="${SLURM_CONSTRAINT:-}"

# Optional environment module to load inside the job. Empty to skip.
CLUSTER_PYTHON_MODULE="${CLUSTER_PYTHON_MODULE:-}"

# Conda environment holding the installed package. Empty to skip activation.
CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"

# Job stdout/stderr directory. Git-ignored.
LOG_DIR="${LOG_DIR:-logs_and_errs}"

# How many runs share one GPU. Set from a measured footprint, not by guessing.
PACK_SIZE="${PACK_SIZE:-4}"

# CPU cores the scheduler gives this job. Used only to split the BLAS/OpenMP
# thread pools between the packed runs; without that split, PACK_SIZE processes
# each spawn a full-width thread pool and thrash against each other. Check what
# your cluster allocates per GPU.
CPUS_PER_JOB="${CPUS_PER_JOB:-16}"

# ------------------------------------------------------------
# Sweep values -- edit these
# ------------------------------------------------------------
batch_sizes=(16 32)
learning_rates=(0.01 0.03 0.05 0.1)
participation_rates=(0.4 0.6)
local_iterations=(1 4 8)
data_paths=(
  "data/generated/synthetic_label_skew/manifest.json"
)

# Runs at or above this participation rate are refused rather than packed:
# host memory scales with the number of participating clients, so a
# high-participation run can exhaust a node on its own. Submit those with
# SLURMs/example_sweep.sh instead. Set to 2 to disable the guard.
MAX_PACKABLE_PR="${MAX_PACKABLE_PR:-1.0}"

# ------------------------------------------------------------
# Base config / output root
#
# Synthetic label skew by default: no download, no optional extra. Generate
# it once with
#   fedbrew generate --config data/configs/synthetic_label_skew.yaml
# Packing is not worth much on a problem this small -- it is here so the
# example is runnable. The realistic case is a cross-device dataset, e.g.
#   base_config="configs/femnist/fedavg.yaml"
#   base_output="outputs/femnist/fedavg"
#   data_paths=("data/generated/femnist_natural/manifest.json")
# where a single run really does leave the GPU mostly idle.
# ------------------------------------------------------------
base_config="configs/dev/synthetic_label_skew.yaml"
base_output="outputs/synthetic_label_skew/fedavg"

# ------------------------------------------------------------
# Generated configs -- one directory per submission, so a later submission
# cannot overwrite configs that queued jobs have not read yet.
# ------------------------------------------------------------
run_id="$(date +%Y%m%d_%H%M%S)_$$"
generated_root="data/generated/slurm_configs/example_sweep_packed/${run_id}"
mkdir -p "${generated_root}" "${LOG_DIR}"

echo "Generated config root: ${generated_root}"
echo "Pack size: ${PACK_SIZE} runs per GPU"
echo

extra_sbatch=""
[ -n "${SLURM_PARTITION}" ] && extra_sbatch+="#SBATCH -p ${SLURM_PARTITION}"$'\n'
[ -n "${SLURM_CONSTRAINT}" ] && extra_sbatch+="#SBATCH -C ${SLURM_CONSTRAINT}"$'\n'

env_setup=""
[ -n "${CLUSTER_PYTHON_MODULE}" ] && env_setup+="module purge"$'\n'"module load ${CLUSTER_PYTHON_MODULE}"$'\n'
[ -n "${CONDA_ENV_NAME}" ] && env_setup+='source "$(conda info --base)/etc/profile.d/conda.sh"'$'\n'"conda activate ${CONDA_ENV_NAME}"$'\n'

# ============================================================
# Phase 1 -- build every run. Nothing is submitted yet.
# ============================================================
built=()

  for bs in "${batch_sizes[@]}"; do
    for lr in "${learning_rates[@]}"; do
      for pr in "${participation_rates[@]}"; do
        for data_path in "${data_paths[@]}"; do

          data_slug="${data_path#data/generated/}"
          data_slug="${data_slug%/manifest.json}"
          data_slug="${data_slug//\//_}"

          for local_iteration in "${local_iterations[@]}"; do

            if awk -v p="${pr}" -v m="${MAX_PACKABLE_PR}" 'BEGIN{exit !(p>=m)}'; then
              echo "Skipping pr=${pr}: too memory-heavy to pack; submit it with example_sweep.sh"
              continue
            fi

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
            log_name="${tag//./p}"

            cmd="fedbrew run --config ${exp_config} --batch-size ${bs} --lr ${lr} --local-iterations ${local_iteration} --participation-rate ${pr} --output-dir ${output_dir} --tag ${tag}"

            # Per-round cost is roughly proportional to local_iterations *
            # participation_rate. Recorded so packs can group runs of similar
            # duration -- see the sort below.
            cost="$(awk -v e="${local_iteration}" -v p="${pr}" 'BEGIN{printf "%012.4f", e*p}')"

            built+=("${cost}"$'\t'"${cmd}"$'\t'"${log_name}"$'\t'"${output_dir}")

          done
        done
      done
    done
  done

if [ "${#built[@]}" -eq 0 ]; then
  echo "Nothing to submit." >&2
  exit 1
fi

# ------------------------------------------------------------
# Sort by estimated cost so each pack holds runs that finish at roughly the
# same time. Without this a pack mixes a fast configuration with one several
# times slower: the fast runs exit early and the job spends its whole tail as
# a single run on a GPU it is still holding for the pack.
# ------------------------------------------------------------
mapfile -t sorted_runs < <(printf '%s\n' "${built[@]}" | sort -n)

cmds=(); tags=(); outdirs=()
for line in "${sorted_runs[@]}"; do
  cmds+=("$(cut -f2 <<<"${line}")")
  tags+=("$(cut -f3 <<<"${line}")")
  outdirs+=("$(cut -f4 <<<"${line}")")
done

total=${#cmds[@]}
n_packs=$(( (total + PACK_SIZE - 1) / PACK_SIZE ))
echo "Total runs: ${total}  ->  jobs to submit: ${n_packs} (would be ${total} unpacked)"
echo

# ============================================================
# Phase 2 -- chunk into packs, one job per pack.
# ============================================================
pack_idx=0
for (( start=0; start<total; start+=PACK_SIZE )); do
  pack_idx=$(( pack_idx + 1 ))
  job_name="example_sweep_pack${pack_idx}"

  end=$(( start + PACK_SIZE ))
  [ "${end}" -gt "${total}" ] && end=${total}

  # Build the fragment that launches each run in the background. Anything that
  # must resolve inside the job -- $! , $SLURM_JOB_ID -- is escaped here.
  body=""
  for (( k=start; k<end; k++ )); do
    # Decide --resume-latest inside the job rather than at submit time: a
    # requeued or resubmitted run has to pick up checkpoints its earlier
    # attempt wrote, which do not exist yet when this script runs. The flag
    # errors out when no checkpoint is present, so it can never be passed
    # unconditionally. best.pt alone must not trigger a resume -- resuming
    # reads latest.pt and round_*.pt only.
    body+="ckpt_dir=\"${outdirs[k]}/checkpoints\""$'\n'
    body+="resume_flag=\"\""$'\n'
    body+="if [ -f \"\${ckpt_dir}/latest.pt\" ] || compgen -G \"\${ckpt_dir}/round_*.pt\" >/dev/null 2>&1; then"$'\n'
    body+="  resume_flag=\"--resume-latest\""$'\n'
    body+="  echo \"[pack] resuming ${tags[k]}\""$'\n'
    body+="else"$'\n'
    body+="  echo \"[pack] starting ${tags[k]} fresh\""$'\n'
    body+="fi"$'\n'
    body+="${cmds[k]} \${resume_flag} > ${LOG_DIR}/${tags[k]}_\${SLURM_JOB_ID}.out 2> ${LOG_DIR}/${tags[k]}_\${SLURM_JOB_ID}.err &"$'\n'
    body+="pids+=(\$!)"$'\n'
    body+="names+=(\"${tags[k]}\")"$'\n'
    body+=$'\n'
  done

  echo "Submitting pack ${pack_idx}/${n_packs}: ${job_name}"
  for (( k=start; k<end; k++ )); do echo "  - ${tags[k]}"; done

  sbatch <<EOF_JOB
#!/bin/bash
#SBATCH -A ${SLURM_ACCOUNT}
#SBATCH --job-name=${job_name}
#SBATCH --output=${LOG_DIR}/${job_name}_%j.pack.out
#SBATCH --error=${LOG_DIR}/${job_name}_%j.pack.err
#SBATCH --time=${SLURM_TIME}
#SBATCH -n 1
#SBATCH --gpus=${SLURM_GPUS}
# Requeue on preemption or node failure. Combined with the per-run resume
# detection below, a requeued job continues from its last checkpoint instead
# of restarting at round 0.
#SBATCH --requeue
${extra_sbatch}
${env_setup}
# Split the job's CPU allocation across the packed runs so their BLAS/OpenMP
# thread pools do not oversubscribe the cores and thrash.
export OMP_NUM_THREADS=\$(( ${CPUS_PER_JOB} / ${PACK_SIZE} ))
[ "\${OMP_NUM_THREADS}" -lt 1 ] && export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=\${OMP_NUM_THREADS}

echo "[pack] job \${SLURM_JOB_ID} on \$(hostname), ${PACK_SIZE} runs sharing ${SLURM_GPUS} GPU"
echo "[pack] OMP_NUM_THREADS=\${OMP_NUM_THREADS}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

pids=()
names=()

${body}
# Wait for every run and report per-run exit status. Without this the job's
# exit code would be the last background process's, and a failed run in the
# middle of a pack would go unnoticed.
fail=0
for i in "\${!pids[@]}"; do
  if wait "\${pids[\$i]}"; then
    echo "[pack] OK      \${names[\$i]}"
  else
    rc=\$?
    echo "[pack] FAILED  \${names[\$i]} (exit \${rc})"
    fail=1
  fi
done

echo "[pack] all runs finished, fail=\${fail}"
exit \${fail}
EOF_JOB

  echo "Submitted."
  echo "----------------------------------------"
  sleep 1
done
echo
echo "Submitted ${n_packs} packed jobs covering ${total} runs. Logs: ${LOG_DIR}/"
