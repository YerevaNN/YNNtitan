#!/bin/bash
# Merge sharded DCP checkpoint to a single .pt file (dcp_to_torch).
# Paths match train_configs/llama3_170m.toml + experiment hash from training run.
#
# Submit from repo root:
#   sbatch merge_dcp_checkpoint.slurm
#
#SBATCH --job-name=dcp_merge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=all
#SBATCH --time=2:00:00
#SBATCH --output=/mnt/weka/gsimonyan/results/slurm_logs/dcp_merge/%j.out
#SBATCH --error=/mnt/weka/gsimonyan/results/slurm_logs/dcp_merge/%j.err

set -eo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-/home/gsimonyan/YNNtitan}"
cd "${REPO_ROOT}"

mkdir -p /mnt/weka/gsimonyan/results/slurm_logs/dcp_merge

if [[ -f .env_vars ]]; then
    # shellcheck source=/dev/null
    source .env_vars
fi

CONDA_ROOT="${CONDA_ROOT:-/mnt/weka/shared-cache/miniforge3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-ynntitan}"
# shellcheck source=/dev/null
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

PYTHON="${CONDA_PREFIX}/bin/python"
unset PYTHONHOME
unset PYTHONPATH

# From llama3_170m.toml save_folder + experiment hash discovered under that folder
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/mnt/weka/gsimonyan/checkpoints/Llama-3.2-170M/dcc4447f70bf445cb5e24cfc/step-20000}"
OUT_PT="${OUT_PT:-/mnt/weka/gsimonyan/checkpoints/Llama-3.2-170M/dcc4447f70bf445cb5e24cfc/step-20000_merged.pt}"

echo "dcp_merge job=${SLURM_JOB_ID:-local} host=$(hostname)"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "OUT_PT=${OUT_PT}"

if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
    echo "ERROR: missing checkpoint directory ${CHECKPOINT_DIR}" >&2
    exit 1
fi
if [[ "$(find "${CHECKPOINT_DIR}" -maxdepth 1 -name '*.distcp' 2>/dev/null | wc -l)" -lt 1 ]]; then
    echo "ERROR: no *.distcp shards in ${CHECKPOINT_DIR}" >&2
    exit 1
fi

echo "Starting dcp_to_torch at $(date -Is)"
"${PYTHON}" -u -m torch.distributed.checkpoint.format_utils dcp_to_torch "${CHECKPOINT_DIR}" "${OUT_PT}"
echo "Finished dcp_to_torch at $(date -Is)"

ls -lh "${OUT_PT}"
echo "OK: wrote ${OUT_PT}"
