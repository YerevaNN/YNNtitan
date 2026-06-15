#!/bin/bash
# Export merged torchtitan .pt to HuggingFace format (1 GPU).
# Set HF_EXPORT_* in .env_vars_dev, then submit from repo root:
#   source .env_vars_dev
#   sbatch --output="${SLURM_LOG_DIR}/job_%j/%j.out" --error="${SLURM_LOG_DIR}/job_%j/%j.err" export_merged_pt_to_hf.slurm
#
#SBATCH --job-name=pt2hf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=h100
#SBATCH --time=2:00:00
#SBATCH --output=logs/pt2hf_%j.out
#SBATCH --error=logs/pt2hf_%j.err

set -eo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs

if [[ -f .env_vars_dev ]]; then
    # shellcheck source=/dev/null
    source .env_vars_dev
elif [[ -f .env_vars ]]; then
    # shellcheck source=/dev/null
    source .env_vars
fi

: "${HF_EXPORT_CHECKPOINT:?Set HF_EXPORT_CHECKPOINT in .env_vars_dev}"
: "${HF_EXPORT_OUTPUT_DIR:?Set HF_EXPORT_OUTPUT_DIR in .env_vars_dev}"
: "${HF_EXPORT_TRAIN_CONFIG:?Set HF_EXPORT_TRAIN_CONFIG in .env_vars_dev}"

CONDA_ROOT="${CONDA_ROOT:?Set CONDA_ROOT in .env_vars_dev}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-ynntitan}"
# shellcheck source=/dev/null
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

PYTHON="${CONDA_PREFIX}/bin/python"
unset PYTHONHOME
unset PYTHONPATH

echo "pt2hf job=${SLURM_JOB_ID:-local} host=$(hostname)"
echo "CHECKPOINT=${HF_EXPORT_CHECKPOINT}"
echo "TRAIN_CONFIG=${HF_EXPORT_TRAIN_CONFIG}"
echo "OUTPUT_DIR=${HF_EXPORT_OUTPUT_DIR}"

if [[ ! -f "${HF_EXPORT_CHECKPOINT}" ]]; then
    echo "ERROR: missing merged checkpoint ${HF_EXPORT_CHECKPOINT}" >&2
    exit 1
fi

echo "Starting at $(date -Is)"
exec "${PYTHON}" -u export_merged_pt_to_hf.py \
    --job.config_file "${HF_EXPORT_TRAIN_CONFIG}" \
    --checkpoint "${HF_EXPORT_CHECKPOINT}" \
    --output-dir "${HF_EXPORT_OUTPUT_DIR}"
