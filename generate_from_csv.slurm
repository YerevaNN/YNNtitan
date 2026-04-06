#!/bin/bash
# Run generate_from_csv.py on one GPU (torchtitan + merged checkpoint).
# Submit from repo root: sbatch generate_from_csv.slurm
#
#SBATCH --job-name=gen_csv
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=all
#SBATCH --time=2:00:00
#SBATCH --output=/mnt/weka/gsimonyan/results/slurm_logs/generate_csv/%j.out
#SBATCH --error=/mnt/weka/gsimonyan/results/slurm_logs/generate_csv/%j.err

set -eo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-/home/gsimonyan/YNNtitan}"
cd "${REPO_ROOT}"

mkdir -p /mnt/weka/gsimonyan/results/slurm_logs/generate_csv

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

CSV_PATH="${CSV_PATH:-/mnt/weka/gsimonyan/results/genetic/dock/2026-03-05/exp-0/dock.parp1/seed-1/0_mols.csv}"
CHECKPOINT="${CHECKPOINT:-/mnt/weka/gsimonyan/checkpoints/Llama-3.2-170M/dcc4447f70bf445cb5e24cfc/step-20000_merged.pt}"
TRAIN_CONFIG="${TRAIN_CONFIG:-./train_configs/llama3_170m.toml}"
OUT_JSONL="${OUT_JSONL:-/mnt/weka/gsimonyan/results/slurm_logs/generate_csv/${SLURM_JOB_ID:-local}_generations.jsonl}"

echo "gen_csv job=${SLURM_JOB_ID:-local} host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "CSV_PATH=${CSV_PATH}"
echo "CHECKPOINT=${CHECKPOINT}"
echo "OUT_JSONL=${OUT_JSONL}"

echo "Starting at $(date -Is)"
exec "${PYTHON}" -u generate_from_csv.py \
    --job.config_file "${TRAIN_CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --csv "${CSV_PATH}" \
    --prompt-column prompt \
    --csv-delimiter ";" \
    --num-samples 8 \
    --max-new-tokens 128 \
    --output-jsonl "${OUT_JSONL}"
