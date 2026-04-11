#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 01:00:00
#SBATCH -A cis260009p
#SBATCH --gpus=v100-32:1
#SBATCH --job-name=vllm_build
#SBATCH --output=/ocean/projects/cis260009p/nadayanu/work/vllm/logs/build_%j.log

set -euo pipefail

WORK_DIR="/ocean/projects/cis260009p/nadayanu/work/vllm"
VENV="$WORK_DIR/.venv"

mkdir -p "$WORK_DIR/logs"
mkdir -p "$WORK_DIR/hf_cache"
mkdir -p "$WORK_DIR/triton_cache"
mkdir -p "$WORK_DIR/xdg_cache"

export HF_HOME="$WORK_DIR/hf_cache"
export TRITON_CACHE_DIR="$WORK_DIR/triton_cache"
export XDG_CACHE_HOME="$WORK_DIR/xdg_cache"

echo "=== [$(date)] Starting vLLM build on $(hostname) ==="
echo "Node GPU:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

source /etc/profile.d/modules.sh
module load cuda/12.4.0
module load gcc/10.2.0
echo "=== Modules loaded: CUDA $(nvcc --version | grep release | awk '{print $6}'), GCC $(gcc --version | head -1) ==="

source "$VENV/bin/activate"
echo "=== Python: $(which python) | $(python --version) ==="

echo "=== Removing stale _C.abi3.so if present ==="
rm -f "$WORK_DIR/vllm/_C.abi3.so"

echo "=== Installing torch==2.5.1+cu124 ==="
pip install "torch==2.5.1" "numpy<2" setuptools wheel \
    --index-url https://download.pytorch.org/whl/cu124
echo "torch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

echo "=== Building vLLM from source (this takes ~10-15 min) ==="
cd "$WORK_DIR"
pip install -e .

echo "=== Verifying install ==="
python -c "from vllm import LLM; print('vllm OK')"
python -c "import vllm; print('vllm version:', vllm.__version__)"

echo "=== [$(date)] Build complete! ==="
