#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 02:00:00
#SBATCH -A cis260009p
#SBATCH --gpus=v100-32:1
#SBATCH --job-name=vllm_test
#SBATCH --output=vllm_test.log

# Add this right below the module loads in scripts/psc_submit.sh
export HF_TOKEN="hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
export HF_HOME="/jet/home/nadayanu/workspace_cis260009p/vllm/hf_cache"
export TRITON_CACHE_DIR="/jet/home/nadayanu/workspace_cis260009p/vllm/triton_cache"
export XDG_CACHE_HOME="/jet/home/nadayanu/workspace_cis260009p/vllm/xdg_cache"

# Load required modules (verified on PSC Bridges-2)
source /etc/profile.d/modules.sh
module load cuda/12.4.0    # Matches torch==2.5.1+cu124
module load gcc/10.2.0     # Stable C++17 support for PyTorch 2.5+

cd ~/workspace_cis260009p/vllm
source /ocean/projects/cis260009p/nadayanu/work/vllm/.venv/bin/activate

# Install exact Torch requirements matching the loaded CUDA module
pip install "torch==2.5.1" "numpy<2" setuptools wheel --index-url https://download.pytorch.org/whl/cu124

# Explicitly build vLLM C++ extensions + install dev dependencies
pip install -e ".[dev]"

# Run tests
bash scripts/gcp_setup_and_test.sh
