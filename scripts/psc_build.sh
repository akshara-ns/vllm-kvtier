#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 01:00:00
#SBATCH -A cis260009p
#SBATCH --gpus=v100-32:1
#SBATCH --job-name=vllm_build
#SBATCH --output=/ocean/projects/cis260009p/nadayanu/work/vllm/logs/build_%j.log

# NOTE: Run scripts/psc_setup.sh on the LOGIN NODE first to install packages.
# This script only rebuilds the C extension (_C.abi3.so) on a GPU node.

WORK_DIR="/ocean/projects/cis260009p/nadayanu/work/vllm"

export HF_HOME="$WORK_DIR/hf_cache"
export TRITON_CACHE_DIR="$WORK_DIR/triton_cache"
export XDG_CACHE_HOME="$WORK_DIR/xdg_cache"

echo "=== [$(date)] Build started on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

source /etc/profile.d/modules.sh
module load cuda/12.4.0
module load gcc/10.2.0

cd "$WORK_DIR"
source .venv/bin/activate

echo "=== Python: $(python --version) | $(python -c 'import sys; print(sys.executable)') ==="
echo "=== Torch: $(python -c 'import torch; print(torch.__version__)') | CUDA: $(python -c 'import torch; print(torch.cuda.is_available())') ==="

# Delete stale .so so it gets recompiled fresh on this GPU node
rm -f "$WORK_DIR/vllm/_C.abi3.so"

echo "=== Rebuilding _C.abi3.so against GPU node CUDA (no downloads needed) ==="
python -m pip install --upgrade pip
python -m pip install -e . --no-build-isolation --no-deps

echo "=== Verifying ==="
python -c "from vllm import LLM; print('vllm OK')"
python -c "import vllm; print('vllm version:', vllm.__version__)"

echo "=== [$(date)] Done! ==="
