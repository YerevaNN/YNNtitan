#!/bin/bash
# Export merged torchtitan .pt to HuggingFace format (1 GPU).
# Submit from repo root: sbatch export_merged_pt_to_hf.slurm
#
# Override defaults:
#   CHECKPOINT=/path/to.pt OUTPUT_DIR=/path/to/hf HF_CONFIG_DIR=/path/to/hf_template sbatch export_merged_pt_to_hf.slurm
#
#SBATCH --job-name=pt2hf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=all
#SBATCH --time=2:00:00
#SBATCH --output=/mnt/weka/gsimonyan/results/slurm_logs/pt2hf/%j.out
#SBATCH --error=/mnt/weka/gsimonyan/results/slurm_logs/pt2hf/%j.err

set -eo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-/home/gsimonyan/YNNtitan}"
cd "${REPO_ROOT}"

mkdir -p /mnt/weka/gsimonyan/results/slurm_logs/pt2hf

if [[ -f .env_vars_dev ]]; then
    # shellcheck source=/dev/null
    source .env_vars_dev
elif [[ -f .env_vars ]]; then
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

CHECKPOINT="${CHECKPOINT:-/mnt/weka/gsimonyan/checkpoints/Llama-3.2-170M/ad16ed143de54345b194bf33/step-20000_merged.pt}"
TRAIN_CONFIG="${TRAIN_CONFIG:-./train_configs/llama3_170m.toml}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/weka/gsimonyan/checkpoints/Llama-3.2-170M/ad16ed143de54345b194bf33/hf_export_step20000}"
HF_CONFIG_DIR="${HF_CONFIG_DIR:-${OUTPUT_DIR}}"

echo "pt2hf job=${SLURM_JOB_ID:-local} host=$(hostname)"
echo "CHECKPOINT=${CHECKPOINT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "HF_CONFIG_DIR=${HF_CONFIG_DIR}"

echo "Starting at $(date -Is)"
exec "${PYTHON}" -u export_merged_pt_to_hf.py \
    --job.config_file "${TRAIN_CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --output-dir "${OUTPUT_DIR}" \
    --hf-config-dir "${HF_CONFIG_DIR}"
