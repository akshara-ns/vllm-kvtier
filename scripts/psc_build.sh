#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 01:30:00
#SBATCH -A cis260009p
#SBATCH --gpus=v100-32:1
#SBATCH --job-name=vllm_build
#SBATCH --output=/ocean/projects/cis260009p/nadayanu/work/vllm/logs/build_%j.log

WORK_DIR="/ocean/projects/cis260009p/nadayanu/work/vllm"

mkdir -p "$WORK_DIR/logs"
mkdir -p "$WORK_DIR/hf_cache"
mkdir -p "$WORK_DIR/triton_cache"
mkdir -p "$WORK_DIR/xdg_cache"

export HF_HOME="$WORK_DIR/hf_cache"
export TRITON_CACHE_DIR="$WORK_DIR/triton_cache"
export XDG_CACHE_HOME="$WORK_DIR/xdg_cache"

echo "=== [$(date)] Build started on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Load modules — must happen before creating venv so python links to right CUDA
source /etc/profile.d/modules.sh
module load cuda/12.4.0
module load gcc/10.2.0

# Unload conda so it doesn't interfere
module unload anaconda3 2>/dev/null || true
export PATH=$(echo $PATH | tr ':' '\n' | grep -v anaconda | tr '\n' ':')

echo "=== Python being used: $(which python3) ==="

# Nuke the old venv — it was created from conda's python
cd "$WORK_DIR"
rm -rf .venv

# Create a clean venv from the system python (not conda)
source /ocean/projects/cis260009p/nadayanu/work/uv/env
uv venv --python 3.12 .venv
source .venv/bin/activate

echo "=== Venv python: $(which python) — $(python --version) ==="
echo "=== Python origin: $(python -c 'import sys; print(sys.executable)') ==="

# Install torch first, pinned to cu124
uv pip install "torch==2.5.1" "numpy<2" setuptools wheel \
    --index-url https://download.pytorch.org/whl/cu124

echo "=== torch: $(python -c 'import torch; print(torch.__version__)') | CUDA: $(python -c 'import torch; print(torch.cuda.is_available())') ==="

# Verify torch is from the venv before building
echo "=== Torch location: $(python -c 'import torch; print(torch.__file__)') ==="
echo "=== Torch version: $(python -c 'import torch; print(torch.__version__)') ==="

# Build vLLM from source against the GPU node's live CUDA
# --no-build-isolation ensures the build uses THIS venv's torch, not a cached/system one
echo "=== Building vLLM (10-15 min) ==="
python -m pip install -e . --no-build-isolation

# Verify
echo "=== Verifying ==="
python -c "from vllm import LLM; print('vllm OK')"
python -c "import vllm; print('vllm version:', vllm.__version__)"

echo "=== [$(date)] Done! ==="
