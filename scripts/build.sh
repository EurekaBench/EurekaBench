#!/bin/bash
set -eo pipefail

domain=neuroscience
scratch_dir=/tmp/$USER/eurekabench

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate eureka-env
set -u

cd "$(dirname "$0")/.."
export UV_CACHE_DIR=/tmp/$USER/cache/uv
export XDG_CACHE_HOME=/tmp/$USER/cache
export APPTAINER_CACHEDIR=/tmp/$USER/apptainer/cache
export TMPDIR=/tmp/$USER/tmp
export PYTHONPATH=$PWD
mkdir -p "$UV_CACHE_DIR" "$APPTAINER_CACHEDIR" "$TMPDIR" "/tmp/$USER/apptainer/build"
APPTAINER_TMPDIR=$(mktemp -d "/tmp/$USER/apptainer/build/eurekabench-XXXXXX")
export APPTAINER_TMPDIR
trap 'chmod -R u+rwX "$APPTAINER_TMPDIR" 2> /dev/null; rm -rf "$APPTAINER_TMPDIR"' EXIT

uv run --no-project --python "$CONDA_PREFIX/bin/python" python -m eurekabench.launch build \
    --domain "$domain" \
    --scratch_dir "$scratch_dir"
