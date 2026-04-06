#!/bin/bash
# HF model: same CSV sampling as generate_from_csv.slurm (for comparison).
# sbatch generate_from_csv_hf.slurm
#
#SBATCH --job-name=gen_hf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=all
#SBATCH --time=2:00:00
#SBATCH --output=/mnt/weka/gsimonyan/results/slurm_logs/generate_csv_hf/%j.out
#SBATCH --error=/mnt/weka/gsimonyan/results/slurm_logs/generate_csv_hf/%j.err

set -eo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-/home/gsimonyan/YNNtitan}"
cd "${REPO_ROOT}"

mkdir -p /mnt/weka/gsimonyan/results/slurm_logs/generate_csv_hf

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

HF_MODEL_DIR="${HF_MODEL_DIR:-/mnt/weka/gsimonyan/checkpoints/Llama-3.2-170M/dcc4447f70bf445cb5e24cfc/hf_export_step20000}"
CSV_PATH="${CSV_PATH:-/mnt/weka/gsimonyan/results/genetic/dock/2026-03-05/exp-0/dock.parp1/seed-1/0_mols.csv}"
OUT_JSONL="${OUT_JSONL:-/mnt/weka/gsimonyan/results/slurm_logs/generate_csv_hf/${SLURM_JOB_ID:-local}_hf_generations.jsonl}"

echo "gen_hf job=${SLURM_JOB_ID:-local} host=$(hostname)"
echo "HF_MODEL_DIR=${HF_MODEL_DIR}"
echo "CSV_PATH=${CSV_PATH}"
echo "OUT_JSONL=${OUT_JSONL}"

echo "Starting at $(date -Is)"
exec "${PYTHON}" -u generate_from_csv_hf.py \
    --hf-model-dir "${HF_MODEL_DIR}" \
    --csv "${CSV_PATH}" \
    --prompt-column prompt \
    --csv-delimiter ";" \
    --num-samples 8 \
    --seed 42 \
    --max-new-tokens 128 \
    --max-seq-len 2048 \
    --temperature 0.8 \
    --top-p 0.95 \
    --output-jsonl "${OUT_JSONL}"
