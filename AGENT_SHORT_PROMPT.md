# SHORT PROMPT — Paste into any coding agent

```
TASK: Implement KSAM (Kinematic Singularity Awareness Module) for Octo-base on MetaWorld MT-10 and produce comparison results.

REPO: https://github.com/lexus-x/startup.git

STEPS:
1. git clone https://github.com/lexus-x/startup.git && cd startup
2. pip install torch metaworld mujoco -e .
3. pip install git+https://github.com/octo-models/octo.git  # if available
4. Read scripts/AGENT_PROMPT.md for full methodology
5. Run: python scripts/run_mt10_comparison.py --gpu 0 --eval_episodes 20
6. If Octo fails to load, the script auto-falls back to learned MLP policy
7. Results go to results/mt10_comparison.json + 3 PNG plots

WHAT KSAM DOES:
- Computes Jacobian condition number κ from joint angles (Sawyer 7-DOF)
- When κ is high (near singularity), blends VLA action with damped pseudo-inverse fallback
- ~5K trainable params, frozen base policy
- Detects when arm is in dangerous configurations and corrects before execution

EXPECTED OUTPUT:
- results/mt10_comparison.json (baseline vs KSAM success rates per task)
- results/mt10_per_task.png (bar chart)
- results/mt10_overall.png (overall comparison)
- results/mt10_kappa_dist.png (condition number distribution)

CONSTRAINTS:
- A100 GPU, use all VRAM
- ~50 min total
- Base policy stays FROZEN
- Commit after each milestone
- If something fails: reproduce, minimize, hypothesize, fix. Don't skip.

PRINCIPLES (Karpathy):
- Don't assume. State what you don't know.
- Simplest code that works. No overengineering.
- Surgical edits only. Every line traces to the task.
- Goal-driven: define success, loop until verified.
```
