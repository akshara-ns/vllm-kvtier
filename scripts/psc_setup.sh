#!/bin/bash
# Run this ONCE on the PSC LOGIN NODE (not as a batch job).
# It creates the venv and installs all packages (requires internet).
# After this, submit psc_build.sh to compile the C extension on a GPU node.
#
# Usage: bash scripts/psc_setup.sh

set -euo pipefail

WORK_DIR="/ocean/projects/cis260009p/nadayanu/work/vllm"

echo "=== [$(date)] Setting up venv on $(hostname) ==="

# Ensure conda is not interfering
conda deactivate 2>/dev/null || true
conda deactivate 2>/dev/null || true

source /etc/profile.d/modules.sh
module load cuda/12.4.0
module load gcc/10.2.0

mkdir -p "$WORK_DIR/hf_cache" "$WORK_DIR/triton_cache" "$WORK_DIR/xdg_cache" "$WORK_DIR/logs"

cd "$WORK_DIR"

# Nuke old venv and create a clean one from system Python 3.11 (not conda)
echo "=== Creating clean venv from /usr/bin/python3.11 ==="
rm -rf .venv
source /ocean/projects/cis260009p/nadayanu/work/uv/env
uv venv --python /usr/bin/python3.11 .venv
source .venv/bin/activate

echo "=== Python: $(python --version) | $(python -c 'import sys; print(sys.executable)') ==="

# Bootstrap pip inside the venv
echo "=== Bootstrapping pip ==="
python -m ensurepip --upgrade
python -m pip install --upgrade pip

# Install build tools required before any pyproject.toml-based install
echo "=== Installing build tools ==="
python -m pip install packaging setuptools wheel

# Use python -m pip throughout to avoid PATH issues
echo "=== Installing torch==2.5.1+cu124 ==="
python -m pip install "torch==2.5.1" "numpy<2" setuptools wheel \
    --index-url https://download.pytorch.org/whl/cu124

echo "=== torch: $(python -c 'import torch; print(torch.__version__)') ==="

# Install all vLLM Python dependencies (no C extension build yet — needs GPU node)
echo "=== Installing vLLM Python deps ==="
python -m pip install -e . --no-build-isolation --no-deps

echo ""
echo "=== Setup complete! ==="
echo "Now submit the build job to compile the C extension on a GPU node:"
echo "  sbatch scripts/psc_build.sh"
