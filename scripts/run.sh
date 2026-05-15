#!/bin/bash
# KSAM Experiment Runner
# Run on A100: bash scripts/run.sh
# Expected time: ~45-50 minutes

set -e

echo "============================================"
echo "KSAM: Kinematic Singularity Awareness Module"
echo "============================================"
echo ""

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
fi

# Install dependencies
echo "Installing dependencies..."
pip install -q torch numpy tqdm
pip install -q metaworld @ git+https://github.com/Farama-Foundation/Metaworld.git 2>/dev/null || \
pip install -q metaworld 2>/dev/null || \
echo "WARNING: MetaWorld install may need manual setup"

pip install -q -e . 2>/dev/null

echo ""
echo "Starting experiment..."
echo ""

# Run the main experiment
python scripts/run_ksam_experiment.py \
    --task reach-v2 \
    --episodes 500 \
    --eval_episodes 100 \
    --gpu 0 \
    --seed 42 \
    --output results

echo ""
echo "Results saved to results/ksam_results.json"
echo "Models saved to results/*.pt"
