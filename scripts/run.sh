#!/bin/bash
set -eo pipefail

domain=geophysics
problem=ice-shelf-flow-laws
agent=codex
model=litellm_proxy/azure_ai/gpt-5.6-luna
effort=xhigh
judges=codex:litellm_proxy/azure_ai/gpt-5.6-luna
judge_effort=xhigh
gpu=0
rejudge=
scratch_dir=/tmp/$USER/eurekabench
jobs_dir=jobs

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate eureka-env
set -u

cd "$(dirname "$0")/.."
export UV_CACHE_DIR=/tmp/$USER/cache/uv
export XDG_CACHE_HOME=/tmp/$USER/cache
export PIP_CACHE_DIR=/tmp/$USER/cache/pip
export HF_HOME=/tmp/$USER/cache/huggingface
export TORCH_HOME=/tmp/$USER/cache/torch
export TRITON_CACHE_DIR=/tmp/$USER/cache/triton
export CUDA_CACHE_PATH=/tmp/$USER/cache/nv
export MPLCONFIGDIR=/tmp/$USER/cache/matplotlib
export APPTAINER_CACHEDIR=/tmp/$USER/apptainer/cache
export APPTAINER_TMPDIR=/tmp/$USER/apptainer/tmp
export TMPDIR=/tmp/$USER/tmp
export _JAVA_OPTIONS=-Djava.io.tmpdir=/tmp/$USER/tmp
export PYTHONPATH=$PWD
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$TMPDIR"

uv run --no-project --python "$CONDA_PREFIX/bin/python" python -m eurekabench.launch run \
    --domain "$domain" \
    --problem "$problem" \
    --agent "$agent" \
    --model "$model" \
    --effort "$effort" \
    --judges "$judges" \
    --judge_effort "$judge_effort" \
    --gpu "$gpu" \
    --rejudge "$rejudge" \
    --scratch_dir "$scratch_dir" \
    --jobs_dir "$jobs_dir"
