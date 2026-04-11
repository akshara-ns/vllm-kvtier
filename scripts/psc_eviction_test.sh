#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 01:00:00
#SBATCH -A cis260009p
#SBATCH --gpus=v100-32:1
#SBATCH --job-name=eviction_test
#SBATCH --output=eviction_test.log

source /etc/profile.d/modules.sh
module load cuda/12.6.1
module load gcc/10.2.0

#export HF_TOKEN not needed for Qwen
export HF_HOME="/ocean/projects/cis260009p/nadayanu/work/vllm/hf_cache"
export TRITON_CACHE_DIR="/ocean/projects/cis260009p/nadayanu/work/vllm/triton_cache"
export XDG_CACHE_HOME="/ocean/projects/cis260009p/nadayanu/work/vllm/xdg_cache"

cd /ocean/projects/cis260009p/nadayanu/work/vllm
source .venv_kvtier/bin/activate

pip install -e ".[dev]" -q

python3 -m kv_cache_tiering.benchmarks.benchmark \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --policies lru attention hybrid \
    --dataset sharegpt \
    --dataset-path /ocean/projects/cis260009p/nadayanu/work/vllm/datasets/sharegpt.json \
    --num-prompts 50 \
    --max-model-len 4096 \
    --max-tokens 256 \
    --gpu-mem-util 0.12 \
    --cpu-bytes 4000000000 \
    --output /ocean/projects/cis260009p/nadayanu/work/vllm/benchmark_results/eviction_test_$(date +%Y%m%d_%H%M%S).json

echo "=== Checking eviction counts ==="
cat /ocean/projects/cis260009p/nadayanu/work/vllm/benchmark_results/eviction_test_*.json | python3 -c "
import json, sys
data = json.load(sys.stdin)
for policy, metrics in data.items():
    print(f'{policy}: total_evictions={metrics.get(\"total_evictions\", \"NOT FOUND\")}')
"
