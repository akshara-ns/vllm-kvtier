#!/bin/bash
# Run this ONCE on the PSC LOGIN NODE (not as a batch job).
# It creates the venv and installs all packages (requires internet).
# After this, submit psc_build.sh to compile the C extension on a GPU node.
#
# Usage: bash scripts/psc_setup.sh

set -euo pipefail

WORK_DIR="/ocean/projects/cis260009p/nadayanu/work/vllm"
# PSC has no Python 3.11 dev headers; use anaconda Python 3.12 which includes them
PYTHON="/opt/packages/anaconda3-2024.10-1/bin/python3.12"

echo "=== [$(date)] Setting up venv on $(hostname) ==="

source /etc/profile.d/modules.sh
module load cuda/12.4.0
module load gcc/10.2.0
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))

mkdir -p "$WORK_DIR/hf_cache" "$WORK_DIR/triton_cache" "$WORK_DIR/xdg_cache" "$WORK_DIR/logs"

cd "$WORK_DIR"

# Create clean venv from conda's Python 3.12 (has dev headers at include/python3.12)
echo "=== Creating clean venv from $PYTHON ==="
rm -rf .venv
source /ocean/projects/cis260009p/nadayanu/work/uv/env
uv venv --python "$PYTHON" .venv
source .venv/bin/activate

echo "=== Python: $(python --version) | $(python -c 'import sys; print(sys.executable)') ==="
echo "=== Headers: $(python -c 'import sysconfig; print(sysconfig.get_path(\"include\"))') ==="

# Bootstrap pip
echo "=== Bootstrapping pip ==="
python -m pip install --upgrade pip

# Install build tools required before any pyproject.toml-based install
echo "=== Installing build tools ==="
python -m pip install "setuptools>=77" packaging wheel setuptools_scm cmake ninja

echo "=== Installing torch==2.5.1+cu124 ==="
python -m pip install "torch==2.6.0" "numpy<2" \
    --index-url https://download.pytorch.org/whl/cu124

echo "=== torch: $(python -c 'import torch; print(torch.__version__)') ==="

# Register vLLM as Python package — VLLM_TARGET_DEVICE=empty skips CMake on login node
echo "=== Installing vLLM Python package (no C extension — needs GPU node) ==="
VLLM_TARGET_DEVICE=empty python -m pip install -e . --no-build-isolation --no-deps

# Install all Python runtime dependencies
echo "=== Installing vLLM runtime deps ==="
python -m pip install -r requirements/common.txt

echo ""
echo "=== Setup complete! ==="
echo "Now submit the build job to compile the C extension on a GPU node:"
echo "  sbatch scripts/psc_build.sh"
