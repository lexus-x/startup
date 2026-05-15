# KSAM: Kinematic Singularity Awareness Module

A modular, plug-and-play safety wrapper for Vision-Language-Action (VLA) models that prevents kinematic singularity failures in real-time robotic control.

## Quick Start (A100)

```bash
# Clone and setup
git clone https://github.com/lexus-x/startup.git
cd startup
pip install -e .

# Install MetaWorld
pip install metaworld

# Run full experiment (~45 min on A100)
bash scripts/run.sh

# Or run directly:
python scripts/run_ksam_experiment.py --task reach-v2 --episodes 500 --eval_episodes 100 --gpu 0

# Generate plots (after experiment):
pip install matplotlib
python scripts/plot_results.py --results results/ksam_results.json --output results/
```

## What It Does

1. **Baseline**: Trains a policy on MetaWorld via behavioral cloning + SAC
2. **Identifies singularity failures**: Records when the arm enters high-κ configurations
3. **KSAM wrapper**: Wraps the policy with Jacobian-conditioned safety gating (~4.2K params)
4. **Evaluation**: Measures success rate, singularity failures, and κ-binned performance

## Architecture

```
Observation + Joint Angles
    │
    ▼
┌──────────────┐     ┌─────────────────────┐
│ Jacobian     │────►│ Condition Number κ   │
│ Computation  │     │ (singularity metric) │
└──────────────┘     └──────────┬──────────┘
                                │
                    ┌───────────▼───────────┐
                    │  Gating MLP (5K params)│
                    │  g = σ(MLP(q))         │
                    │  × σ(α·(κ_thresh - κ)) │
                    └───────────┬───────────┘
                                │ gate ∈ [0,1]
                    ┌───────────▼───────────┐
                    │  Blend:               │
                    │  a = g·a_VLA          │
                    │    + (1-g)·a_safe     │
                    └───────────────────────┘
```

## Files

```
src/ksam/
  jacobian.py    - Real Panda DH kinematics, SVD-based κ, damped pseudo-inverse
  module.py      - KSAMWrapper (gate MLP + condition-based blending)
  trainer.py     - Standalone training utilities
scripts/
  run_ksam_experiment.py  - Full pipeline (BC + SAC + KSAM + eval)
  run.sh                  - One-command runner
  plot_results.py         - Publication-quality figures
tests/
  test_jacobian.py        - Jacobian sanity checks
```

## Results Output

After running, you get:
- `results/ksam_results.json` — all metrics in JSON
- `results/success_comparison.png` — bar chart: BC vs SAC vs SAC+KSAM
- `results/singularity_analysis.png` — singularity failure reduction
- `results/kappa_success_bins.png` — success rate by κ exposure level
- `results/*.pt` — model checkpoints

## Key Metrics

| Metric | What it shows |
|--------|--------------|
| Success rate | Overall task completion |
| Singularity failures | Episodes that failed DUE TO singularities |
| κ-binned success | Success rate stratified by singularity exposure |
| Gate statistics | How KSAM modulates behavior near singularities |
