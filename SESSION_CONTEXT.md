# KSAM Project — Full Session Context
# Paste this into a new session to continue.

---

## WHAT HAPPENED

User shared 9 GitHub repos for analysis. Deep research was conducted with 5 parallel sub-agents scraping arxiv, MetaWorld benchmarks, VLA surveys, and repo credibility.

### Repos Analyzed

**Set 1 (Lexus-X VLA robotics — ALL README-only shells, no code):**
1. `lexus-x/SC-vla` — Self-correction VLA (name-squats on real paper arXiv 2602.21633)
2. `lexus-x/Q-res` — Quantized residual steering (numbers self-admitted "theoretical projections")
3. `lexus-x/SIMIT` — Privileged sim-state token selection (novel combo, no code)
4. `lexus-x/L-physim` — Latent physics simulation (novel application, no code)
5. `lexus-x/ARC---VLA` / APEX — Token pruning (red ocean, 13+ papers exist)
6. `lexus-x/startup` / KSAM — Kinematic singularity awareness (most novel idea, no code)

**Set 2 (AI agent skills):**
7. `mattpocock/skills` — 15+ engineering skills (TDD, diagnose, grill-me). Most mature.
8. `multica-ai/andrej-karpathy-skills` — 4 principles in a CLAUDE.md. Simplest.
9. `jordan-gibbs/hyperresearch` — 16-step deep research pipeline. Most ambitious.

### Credibility Verdict on Lexus-X
- Account created Oct 2025, zero followers, zero stars
- ALL 7 VLA repos created within ~1 hour of each other on May 15, 2026
- **Zero actual code** in any repo — README-only shells
- No published papers on arxiv, no institutional affiliation
- "Lexus-X Research" has no verifiable existence
- **Verdict: AI-generated research marketing**

### Key Benchmark Findings (MetaWorld MT-10/MT-50)

| Benchmark | SOTA | Source | Status |
|-----------|------|--------|--------|
| MT-10 | **99%** | TOPPO (arXiv 2605.11473, May 2026) | **SOLVED** |
| MT-50 mean | **90.88%** | TOPPO | Near ceiling |
| MT-50 worst-10 tail | **56.5%** | TOPPO | **Active frontier** |
| LIBERO | 95-98% | Multiple | "Basically solved" (ICLR 2026 survey) |
| CALVIN ABC | 4.53 | FLOWER | "Almost saturated" |

- MetaWorld V1 vs V2 rewards were **silently replaced** (Meta-World+ NeurIPS 2025 paper)
- Cross-version comparisons are **INVALID**
- MetaWorld is primarily multi-task RL benchmark, NOT standard VLA benchmark

### VLA Field Saturation

From ICLR 2026 VLA survey (Moritz Reuss):
- Token pruning: RED OCEAN (13+ papers in 2025-2026)
- Residual correction: Established (A2C2, AnchorRefine, DejaVu, etc.)
- Self-correction for VLA: Active area (Self-Correcting VLA, CycleVLA)
- Privileged learning + token selection: **Novel combination** (closest: Pri4R)
- Latent physics simulation: **Novel application** (world models exist but not this framing)
- Kinematic singularity for VLAs: **GENUINE GAP** — almost no prior work

### Novelty Ranking (if code existed)
1. KSAM — Most novel (genuine gap in VLA literature)
2. SIMIT — Novel combination of existing ideas
3. L-PhySim — Novel application
4. SC-VLA — Reasonable, name-squatting issue
5. Q-Res — Honest about being theoretical
6. APEX — Least novel (red ocean)

---

## WHAT WAS BUILT

We decided to implement **KSAM** because:
- It fills a genuine gap (kinematic singularity awareness for VLAs)
- Classical robotics (Jacobian conditioning) + learned gating = clear novelty
- Q1/Q2 viable for IEEE RA-L or T-RO
- Achievable proof-of-concept in ~1 hour on A100

### Code pushed to: `https://github.com/lexus-x/startup.git`

**Branch:** `main`

**Files created:**
```
src/ksam/jacobian.py          — Real Sawyer DH kinematics, SVD-based κ, damped pseudo-inverse
src/ksam/jacobian_sawyer.py   — (in run_mt10_comparison.py, inline)
src/ksam/module.py            — KSAMWrapper with gate MLP (~5K params)
src/ksam/trainer.py           — Standalone training utilities
src/ksam/__init__.py           — Package exports

scripts/run_mt10_comparison.py — MAIN SCRIPT: end-to-end MT-10 comparison
scripts/run_ksam_experiment.py — Earlier version with SAC training
scripts/run.sh                 — Shell runner
scripts/plot_results.py        — Publication-quality plotting
scripts/plot_mt10.py           — MT-10 specific plots

tests/test_jacobian.py         — Jacobian sanity checks

AGENT_PROMPT.md                — Full methodology for coding agents
AGENT_SHORT_PROMPT.md          — Paste-ready one-liner for Claude Code/Codex
SESSION_CONTEXT.md             — THIS FILE
```

### How to run on A100:
```bash
git clone https://github.com/lexus-x/startup.git
cd startup
pip install torch metaworld mujoco -e .
python scripts/run_mt10_comparison.py --gpu 0 --eval_episodes 20
```

**Expected time:** ~50 minutes
**Output:** `results/mt10_comparison.json` + 3 PNG plots

### KSAM Architecture:
```
joint_angles [B, 7]
  → compute_sawyer_jacobian(q) → J [B, 6, 7]
  → condition_number(J) → κ [B]  (high = near singularity)
  → gate_mlp(q) → g_mlp [B, 1]  (learned)
  → gate = sigmoid(α·(κ_thresh - κ)) × g_mlp
  → a_final = gate × a_base + (1-gate) × a_safe
     where a_safe = J†·(JJ† + λ²I)⁻¹·a_base (damped pseudo-inverse)
```

Trainable: ~5K params (gate MLP + α + κ_thresh). Base policy: 100% FROZEN.

---

## NEXT STEPS

1. **Run the experiment** on A100 → get `mt10_comparison.json`
2. **If results are good** → frame for IEEE RA-L paper
3. **If results are weak** → honest reporting, pivot to MT-50 tail tasks or real robot
4. **Paper framing**: "First integration of analytical Jacobian conditioning into VLA forward pass as a trainable modular wrapper"
5. **Key metric to report**: Success rate stratified by κ-binned singularity exposure (not just mean)

### Q1/Q2 Paper Strategy:
- Target: IEEE RA-L (fast review) or IEEE T-RO (top venue)
- Novelty: Genuine gap — no prior VLA work addresses kinematic singularities
- Validation: MetaWorld MT-10 + MT-50 tail tasks + real robot (Franka/Sawyer)
- What NOT to claim: improvement on saturated benchmarks (MT-10 mean is 99%)
- What TO claim: singularity failure reduction, κ-binned success improvement

### If Octo doesn't load:
The script auto-falls back to MLP policy trained via behavioral cloning. Results are still valid as proof-of-concept.

### Karpathy Principles (follow throughout):
- Don't assume. State what you don't know.
- Simplest code that works. No overengineering.
- Surgical edits only.
- Goal-driven: define success, loop until verified.

### mattpocock/skills Principles (follow throughout):
- `/grill-me` before building — align on what we're actually building
- `/tdd` — red-green-refactor for any new code
- `/diagnose` — reproduce → minimize → hypothesize → fix
- Commit after each working milestone
