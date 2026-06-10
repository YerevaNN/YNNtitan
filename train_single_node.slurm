#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Single-node training (full node, default 8 GPUs).
# Submit from the repo root:
#   sbatch train_single_node.slurm
#
# Override config or GPU count at submit time:
#   TRAIN_CONFIG=./train_configs/llama3_380m.toml N_GPUS=8 sbatch train_single_node.slurm
# Resume from step-18000:
#   TRAIN_CONFIG=./train_configs/llama3_380m_resume.toml N_GPUS=8 sbatch train_single_node.slurm
#
# On the login node, install deps into ynntitan with that env's Python (bare `pip` may hit base 3.13):
#   conda activate ynntitan && python -m pip install -e .

#SBATCH --job-name=titan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=184
#SBATCH --mem=200G
#SBATCH --partition=h100
#SBATCH --time=12:00:00
# Log paths: set SLURM_LOG_DIR in .env_vars_dev and submit with e.g.
#   source .env_vars_dev && sbatch --output="${SLURM_LOG_DIR}/job_%j/%j.out" --error="${SLURM_LOG_DIR}/job_%j/%j.err" train_single_node.slurm
#SBATCH --output=logs/titan_%j.out
#SBATCH --error=logs/titan_%j.err

# If your site uses gres instead of gpus-per-node, comment the line above and use:
# #SBATCH --gres=gpu:8

# No `set -u`: /etc/profile.d/*.sh (e.g. byobu) often uses unset vars as empty; nounset breaks those.
set -eo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
echo "titan_slurm job=${SLURM_JOB_ID:-local} host=$(hostname) repo=${REPO_ROOT}"

if [[ -f .env_vars_dev ]]; then
    # shellcheck source=/dev/null
    source .env_vars_dev
elif [[ -f .env_vars ]]; then
    # shellcheck source=/dev/null
    source .env_vars
fi

: "${DUMP_DIR:?Set DUMP_DIR in .env_vars_dev (copy from .env_vars)}"
mkdir -p "${DUMP_DIR}" logs

# Activate ynntitan before any global profile/bashrc so PATH/PYTHONPATH stay aligned with conda.
# (Sourcing ~/.bashrc first often reorders PATH or skips conda in non-interactive jobs.)
CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniforge3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-ynntitan}"
# shellcheck source=/dev/null
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "ERROR: conda activate failed (CONDA_PREFIX empty)." >&2
    exit 1
fi

PYTHON="${CONDA_PREFIX}/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
    echo "ERROR: missing interpreter: ${PYTHON}" >&2
    exit 1
fi

# Inherited from the submission shell, these often break conda (wrong site-packages order).
unset PYTHONHOME
unset PYTHONPATH

# Optional: CUDA / MPI modules after conda (uncomment if your site requires it).
# if [[ -f /etc/profile ]]; then
#     set +e
#     source /etc/profile
#     set -e
# fi

PURELIB=$("${PYTHON}" -c "import sysconfig; print(sysconfig.get_path('purelib'))")
if [[ ! -d "${PURELIB}/torch" ]]; then
    echo "ERROR: no ${PURELIB}/torch on this host — PyTorch is not installed in env ${CONDA_ENV_NAME} here (NFS / wrong node / different env than on login)." >&2
    ls -la "${PURELIB}" 2>/dev/null | head -25 >&2 || true
    exit 1
fi

# Pip may inherit PIP_CACHE_DIR=/opt/shared-cache/pip (not writable); use a user cache.
if [[ -z "${PIP_CACHE_DIR:-}" ]] || [[ ! -w "${PIP_CACHE_DIR}" ]]; then
    export PIP_CACHE_DIR="${HOME}/.cache/pip"
fi
mkdir -p "${PIP_CACHE_DIR}"

# Some clusters have root-only /tmp for wheel builds; prefer a user tmp when needed.
if [[ ! -w /tmp ]]; then
    export TMPDIR="${HOME}/.cache/tmp"
    mkdir -p "${TMPDIR}"
fi

# Editable install picks up repo changes (use env python so we never call base conda's pip).
"${PYTHON}" -m pip install -q -e "${REPO_ROOT}/"

if ! "${PYTHON}" -c "import torch" 2>/dev/null; then
    echo "ERROR: import torch failed for ${PYTHON} despite ${PURELIB}/torch existing." >&2
    "${PYTHON}" -c "import sys, site; print('executable:', sys.executable); print('purelib:', __import__('sysconfig').get_path('purelib')); print('site:', site.getsitepackages())" || true
    exit 1
fi

N_GPUS="${N_GPUS:-8}"
TRAIN_CONFIG="${TRAIN_CONFIG:-./train_configs/llama3_380m.toml}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TRAIN_ARGS=(--job.config_file "${TRAIN_CONFIG}")
if [[ -n "${EXTRA_TRAIN_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    TRAIN_ARGS+=(${EXTRA_TRAIN_ARGS})
fi

exec "${PYTHON}" -u -m torch.distributed.run \
    --nproc_per_node="${N_GPUS}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    --local-ranks-filter=0 \
    --role=rank \
    --tee=3 \
    train.py \
    "${TRAIN_ARGS[@]}"
