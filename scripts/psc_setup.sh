#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 02:00:00
#SBATCH -A cis260009p
#SBATCH --gpus=v100-32:1
#SBATCH --job-name=vllm_setup
#SBATCH --output=vllm_setup.log

source /etc/profile.d/modules.sh
module load cuda/12.4.0
module load gcc/10.2.0

cd /ocean/projects/cis260009p/nadayanu/work/vllm

# Wipe and recreate venv cleanly
rm -rf .venv_kvtier
python -m venv .venv_kvtier
source .venv_kvtier/bin/activate

# Install in correct order
pip install --upgrade pip setuptools setuptools_scm wheel

pip install torch==2.5.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

pip install -e ".[dev]"

echo "=== Install complete ==="
pip show vllm
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available())"
python -c "import vllm; print('vllm:', vllm.__version__)"
