# KSAM VLA Implementation Agent Prompt
# =====================================
# Feed this entire file to any coding agent (Claude Code, Codex, etc.)
# It will implement KSAM with Octo-base on MetaWorld MT-10 and produce results.
# Expected time on A100: ~50-60 minutes end-to-end.

---

## YOUR MISSION

You are implementing **KSAM (Kinematic Singularity Awareness Module)** — a lightweight safety wrapper for VLA models — and producing benchmark results on **MetaWorld MT-10** using **Octo-base** as the frozen VLA backbone.

**Success criteria**: A JSON file with comparison results (baseline Octo vs Octo+KSAM) and 3 publication-quality plots.

---

## GUIDING PRINCIPLES (from Karpathy)

1. **Think before coding.** State assumptions. Don't hide confusion. If something is unclear about Octo's interface or MetaWorld's API, look it up first — don't guess.
2. **Simplicity first.** No abstractions for single-use code. No "flexibility" that wasn't requested. If 200 lines could be 50, rewrite it.
3. **Surgical changes.** Touch only what you must. Don't refactor things that aren't broken. Every changed line traces to the task.
4. **Goal-driven execution.** Define success criteria. Loop until verified. Don't tell the user what you're doing — show the result.

---

## CONSTRAINTS

- **Hardware**: 1× A100 GPU. Use ALL the VRAM. Batch as large as possible.
- **Time budget**: 60 minutes total. Every minute counts.
- **Base model**: Octo-base (frozen). Do NOT fine-tune Octo's weights.
- **Benchmark**: MetaWorld MT-10 (10 tasks, V2 rewards).
- **KSAM trainable params**: <10K total.
- **No speculative code.** Every function must be tested before moving on.
- **Commit after each working milestone.** If something breaks, you can revert.

---

## PHASE 0: ENVIRONMENT SETUP (5 min max)

```bash
# 0.1 Clone the KSAM repo
git clone https://github.com/lexus-x/startup.git
cd startup

# 0.2 Create clean environment
python -m venv venv && source venv/bin/activate

# 0.3 Install dependencies — ORDER MATTERS
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install mujoco metaworld
pip install -e .

# 0.4 Install Octo-base
pip install octo  # or: pip install git+https://github.com/octo-models/octo.git

# 0.5 Verify GPU
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.mem_get_info())"

# 0.6 Verify MetaWorld
python -c "
import metaworld
ml1 = metaworld.ML1('reach-v2')
env = ml1.train_classes['reach-v2']()
obs = env.reset()
print('MetaWorld OK, obs shape:', obs.shape)
print('Action space:', env.action_space)
"

# 0.7 Verify Octo loads
python -c "
from octo.model.octo_model import OctoModel
model = OctoModel.from_pretrained('octo-base')
print('Octo loaded, params:', sum(p.numel() for p in model.parameters()))
"
```

**CHECKPOINT**: If any step fails, debug before proceeding. Do NOT continue with broken setup.

**Commit**: `git add -A && git commit -m "chore: environment setup verified"`

---

## PHASE 1: UNDERSTAND THE SYSTEMS (read before writing code)

### 1.1 Octo-base Interface

Octo expects observations as a dict:
```python
{
    "image_primary": np.ndarray,    # [H, W, 3] uint8
    "pad_mask": np.ndarray,         # [H, W] bool
}
```
And returns actions: `np.ndarray` of shape `[action_dim]`.

Octo has a `model.create_example()` method. Check the actual API:
```python
python -c "
from octo.model.octo_model import OctoModel
model = OctoModel.from_pretrained('octo-base')
example = model.create_example()
print(type(example))
print(example.keys() if hasattr(example, 'keys') else dir(example))
"
```

### 1.2 MetaWorld MT-10 Interface

MT-10 = 10 tasks from MetaWorld. Each task has:
- `env.reset()` → observation (39-dim or image)
- `env.step(action)` → (obs, reward, done, info)
- `info["success"]` → 1.0 if task solved
- `env.sim.data.qpos[:7]` → joint angles (KEY for KSAM)
- `env.sim.data.qvel[:7]` → joint velocities

**CRITICAL**: MetaWorld uses Sawyer robot (7-DOF), NOT Panda. Update DH parameters accordingly.

Sawyer DH parameters (approximate):
```
Joint 1: a=0, alpha=-π/2, d=0.307
Joint 2: a=0, alpha=π/2, d=0
Joint 3: a=0.0825, alpha=π/2, d=0.390
Joint 4: a=-0.0825, alpha=-π/2, d=0
Joint 5: a=0, alpha=-π/2, d=0.384
Joint 6: a=0.088, alpha=π/2, d=0
Joint 7: a=0, alpha=0, d=0.107
```

### 1.3 KSAM Architecture

```
Input: joint_angles [B, 7]
  │
  ├─► compute_jacobian(q) → J [B, 6, 7]
  │     using DH forward kinematics
  │
  ├─► compute_condition_number(J) → κ [B]
  │     κ = σ_max / σ_min via SVD
  │
  ├─► gate_mlp(q) → g_mlp [B, 1]
  │     2-layer MLP, 64 hidden, sigmoid output
  │
  ├─► g = sigmoid(α · (κ_thresh - κ)) × g_mlp
  │     α, κ_thresh are learnable scalars
  │
  └─► a_final = g × a_octo + (1-g) × a_safe
        a_safe = J† · (J · J† + λ²I)⁻¹ · a_octo[:3]
        (damped pseudo-inverse fallback)
```

---

## PHASE 2: IMPLEMENT KSAM CORE (15 min max)

### Step 2.1: Fix Jacobian for Sawyer

The existing `src/ksam/jacobian.py` has Panda DH parameters. **You MUST update for Sawyer.**

Write `src/ksam/jacobian_sawyer.py`:
- Sawyer-specific DH parameters (see above)
- `compute_sawyer_jacobian(q)` → [B, 6, 7]
- `compute_condition_number(J)` → [B]
- `damped_pseudo_inverse(J, λ)` → [B, 7, 6]
- Test: verify that home position (q=0) gives reasonable κ (<50)

### Step 2.2: Verify Jacobian Correctness

```python
# Test: numerical Jacobian vs analytical
def numerical_jacobian(q, eps=1e-4):
    """Compute Jacobian via finite differences as ground truth."""
    # Forward kinematics: q → end-effector position
    # J_numerical[:, i] = (FK(q + eps*e_i) - FK(q - eps*e_i)) / (2*eps)
    ...

# Compare: |J_analytical - J_numerical| < 1e-3
```

**CHECKPOINT**: If analytical and numerical Jacobians don't match, fix before proceeding.

### Step 2.3: Implement KSAMWrapper

Update `src/ksam/module.py`:
- Takes Octo model as frozen backbone
- KSAM gating on top
- Forward pass: compute κ, gate, blend action
- Must handle Octo's specific input/output format

### Step 2.4: Unit Test

```python
python -c "
import torch
from ksam.jacobian_sawyer import compute_sawyer_jacobian, compute_condition_number
from ksam.module import KSAMWrapper

# Test Jacobian
q = torch.randn(4, 7)
J = compute_sawyer_jacobian(q)
kappa = compute_condition_number(J)
print(f'κ range: [{kappa.min():.1f}, {kappa.max():.1f}]')

# Test wrapper
import torch.nn as nn
dummy = nn.Linear(39, 4)
ksam = KSAMWrapper(dummy, action_dim=4)
action, debug = ksam(q, return_debug=True)
print(f'Gate: {debug[\"gate\"].detach().numpy()}')
print(f'KSAM params: {ksam.trainable_params()}')
"
```

**Commit**: `git add -A && git commit -m "feat: KSAM core implementation for Sawyer"`

---

## PHASE 3: BUILD MT-10 EVALUATION PIPELINE (10 min max)

Write `scripts/eval_mt10.py`:

### 3.1 Environment Setup

```python
import metaworld
import numpy as np

# MT-10 = first 10 tasks
TASKS = [
    'reach-v2', 'push-v2', 'pick-place-v2', 'door-open-v2',
    'drawer-open-v2', 'drawer-close-v2', 'button-press-topdown-v2',
    'peg-insert-side-v2', 'window-open-v2', 'window-close-v2'
]

def make_mt10_envs(seed=42):
    """Create all 10 MT-10 environments."""
    ml1 = metaworld.ML1(TASKS[0], seed=seed)  # Start with first task
    envs = {}
    for task_name in TASKS:
        env = ml1.train_classes[task_name]()
        task = [t for t in ml1.train_tasks if t.env_name == task_name][0]
        env.set_task(task)
        envs[task_name] = env
    return envs
```

### 3.2 Octo Policy Wrapper

```python
from octo.model.octo_model import OctoModel

class OctoPolicy:
    def __init__(self, model_name="octo-base"):
        self.model = OctoModel.from_pretrained(model_name)
    
    def get_action(self, image, instruction):
        """Get action from Octo given image and language instruction."""
        observation = {
            "image_primary": image[np.newaxis],  # [1, H, W, 3]
            "pad_mask": np.ones(1, dtype=bool),
        }
        action = self.model.sample_actions(
            observation,
            [instruction],
            rng=jax.random.PRNGKey(0),
        )
        return np.array(action[0])
```

### 3.3 KSAM Policy Wrapper

```python
class KSAMPolicy:
    def __init__(self, octo_policy, ksam_wrapper, device="cuda"):
        self.octo = octo_policy
        self.ksam = ksam_wrapper.to(device)
        self.device = device
    
    def get_action(self, image, instruction, joint_angles):
        # Get Octo action
        octo_action = self.octo.get_action(image, instruction)
        
        # Apply KSAM
        q = torch.FloatTensor(joint_angles).unsqueeze(0).to(self.device)
        with torch.no_grad():
            safe_action, debug = self.ksam(q, return_debug=True)
        
        return safe_action.cpu().numpy()[0], debug
```

### 3.4 Evaluation Loop

```python
def evaluate_mt10(policy, envs, num_episodes=20, max_steps=200):
    """
    Evaluate policy on all 10 MT-10 tasks.
    
    Returns:
        results: dict with per-task and overall success rates,
                 per-task κ statistics, singularity failure counts
    """
    results = {
        "per_task": {},
        "overall_successes": 0,
        "overall_episodes": 0,
        "singularity_failures": 0,
        "kappa_values": [],
    }
    
    for task_name, env in envs.items():
        task_successes = 0
        task_kappas = []
        
        for ep in range(num_episodes):
            obs = env.reset()
            ep_success = False
            
            for step in range(max_steps):
                # Get joint angles
                qpos = env.sim.data.qpos[:7].copy()
                
                # Get image (render if needed)
                image = env.render(offscreen=True, resolution=(256, 256))
                
                # Get action
                action, debug = policy.get_action(
                    image, f"{task_name}", qpos
                )
                
                # Step
                obs, reward, done, info = env.step(action)
                
                # Record
                if debug is not None:
                    kappa = debug["kappa"].item()
                    task_kappas.append(kappa)
                    results["kappa_values"].append(kappa)
                
                if info["success"] > 0.5:
                    ep_success = True
                
                if done:
                    break
            
            if ep_success:
                task_successes += 1
                results["overall_successes"] += 1
            results["overall_episodes"] += 1
        
        results["per_task"][task_name] = {
            "success_rate": task_successes / num_episodes,
            "mean_kappa": np.mean(task_kappas) if task_kappas else 0,
            "max_kappa": np.max(task_kappas) if task_kappas else 0,
        }
    
    results["overall_success_rate"] = results["overall_successes"] / results["overall_episodes"]
    return results
```

**Commit**: `git add -A && git commit -m "feat: MT-10 evaluation pipeline"`

---

## PHASE 4: RUN EXPERIMENT (20 min max)

Write `scripts/run_experiment.py`:

```python
#!/usr/bin/env python3
"""
KSAM vs Baseline comparison on MetaWorld MT-10.
Usage: python scripts/run_experiment.py --gpu 0 --eval_episodes 20
"""

import argparse, json, time, os
import numpy as np
import torch

# ... imports from above ...

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--eval_episodes", type=int, default=20)
    parser.add_argument("--output", type=str, default="results")
    args = parser.parse_args()
    
    device = f"cuda:{args.gpu}"
    os.makedirs(args.output, exist_ok=True)
    
    # 1. Load Octo (frozen)
    print("Loading Octo-base...")
    octo_policy = OctoPolicy("octo-base")
    
    # 2. Create MT-10 envs
    print("Creating MT-10 environments...")
    envs = make_mt10_envs()
    
    # 3. Evaluate baseline Octo
    print("\n=== Evaluating Baseline Octo ===")
    t0 = time.time()
    baseline_results = evaluate_mt10(octo_policy, envs, num_episodes=args.eval_episodes)
    baseline_time = time.time() - t0
    print(f"Baseline success rate: {baseline_results['overall_success_rate']:.1%}")
    print(f"Time: {baseline_time:.1f}s")
    
    # 4. Create and train KSAM
    print("\n=== Training KSAM ===")
    # Generate training data from baseline rollouts
    training_data = collect_training_data(octo_policy, envs, num_episodes=50)
    
    # Train KSAM wrapper
    ksam = KSAMWrapper(octo_policy.model, action_dim=4, device=device)
    train_ksam(ksam, training_data, num_epochs=30, device=device)
    
    # 5. Evaluate Octo + KSAM
    print("\n=== Evaluating Octo + KSAM ===")
    ksam_policy = KSAMPolicy(octo_policy, ksam, device=device)
    t0 = time.time()
    ksam_results = evaluate_mt10(ksam_policy, envs, num_episodes=args.eval_episodes)
    ksam_time = time.time() - t0
    print(f"KSAM success rate: {ksam_results['overall_success_rate']:.1%}")
    print(f"Time: {ksam_time:.1f}s")
    
    # 6. Save results
    comparison = {
        "baseline": baseline_results,
        "ksam": ksam_results,
        "improvement": ksam_results["overall_success_rate"] - baseline_results["overall_success_rate"],
        "ksam_trainable_params": ksam.trainable_params(),
        "baseline_time": baseline_time,
        "ksam_time": ksam_time,
    }
    
    with open(os.path.join(args.output, "mt10_comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    
    # 7. Print comparison table
    print("\n" + "="*60)
    print("RESULTS: MetaWorld MT-10")
    print("="*60)
    print(f"{'Task':<30} {'Octo':>10} {'Octo+KSAM':>12} {'Delta':>10}")
    print("-"*65)
    for task in TASKS:
        b = baseline_results["per_task"].get(task, {}).get("success_rate", 0)
        k = ksam_results["per_task"].get(task, {}).get("success_rate", 0)
        print(f"{task:<30} {b:>9.1%} {k:>11.1%} {k-b:>+9.1%}")
    print("-"*65)
    print(f"{'OVERALL':<30} {baseline_results['overall_success_rate']:>9.1%} "
          f"{ksam_results['overall_success_rate']:>11.1%} "
          f"{comparison['improvement']:>+9.1%}")
    
    print(f"\nKSAM trainable params: {ksam.trainable_params():,}")
    print(f"Results saved to {args.output}/mt10_comparison.json")

if __name__ == "__main__":
    main()
```

**Run it**:
```bash
python scripts/run_experiment.py --gpu 0 --eval_episodes 20
```

**Commit**: `git add -A && git commit -m "feat: MT-10 comparison experiment"`

---

## PHASE 5: GENERATE PLOTS (2 min)

Write `scripts/plot_mt10.py`:

```python
#!/usr/bin/env python3
"""Generate publication-quality comparison plots."""

import json, os, sys
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("pip install matplotlib")
    sys.exit(1)

def plot_all(results_path, output_dir):
    with open(results_path) as f:
        data = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    
    tasks = list(data["baseline"]["per_task"].keys())
    baseline_sr = [data["baseline"]["per_task"][t]["success_rate"] for t in tasks]
    ksam_sr = [data["ksam"]["per_task"][t]["success_rate"] for t in tasks]
    
    # Plot 1: Per-task comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(tasks))
    w = 0.35
    ax.bar(x - w/2, baseline_sr, w, label='Octo-base', color='#3498db', edgecolor='black')
    ax.bar(x + w/2, ksam_sr, w, label='Octo-base + KSAM', color='#e74c3c', edgecolor='black')
    ax.set_ylabel('Success Rate')
    ax.set_title('MetaWorld MT-10: Per-Task Success Rate')
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace('-v2','') for t in tasks], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mt10_per_task.png'), dpi=150)
    
    # Plot 2: Overall comparison
    fig, ax = plt.subplots(figsize=(6, 5))
    methods = ['Octo-base', 'Octo-base\n+ KSAM']
    rates = [data["baseline"]["overall_success_rate"], data["ksam"]["overall_success_rate"]]
    colors = ['#3498db', '#e74c3c']
    bars = ax.bar(methods, rates, color=colors, edgecolor='black', linewidth=1.5)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{rate:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=14)
    ax.set_ylabel('Success Rate')
    ax.set_title('MetaWorld MT-10: Overall Success Rate')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mt10_overall.png'), dpi=150)
    
    # Plot 3: κ distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    b_kappas = data["baseline"].get("kappa_values", [])
    k_kappas = data["ksam"].get("kappa_values", [])
    if b_kappas:
        ax.hist(b_kappas, bins=50, alpha=0.6, label='Baseline κ', color='#3498db')
    if k_kappas:
        ax.hist(k_kappas, bins=50, alpha=0.6, label='KSAM κ', color='#e74c3c')
    ax.set_xlabel('Condition Number κ')
    ax.set_ylabel('Count')
    ax.set_title('Jacobian Condition Number Distribution')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mt10_kappa_dist.png'), dpi=150)
    
    print(f"Plots saved to {output_dir}/")

if __name__ == "__main__":
    plot_all("results/mt10_comparison.json", "results/")
```

**Run**: `python scripts/plot_mt10.py`

**Commit**: `git add -A && git commit -m "feat: MT-10 comparison plots"`

---

## PHASE 6: FINAL OUTPUT

After all phases complete, you MUST have:

1. ✅ `results/mt10_comparison.json` — full comparison data
2. ✅ `results/mt10_per_task.png` — per-task bar chart
3. ✅ `results/mt10_overall.png` — overall comparison
4. ✅ `results/mt10_kappa_dist.png` — κ distribution
5. ✅ All code committed and pushed

**Final commit**: `git add -A && git commit -m "results: MT-10 KSAM vs baseline comparison" && git push`

---

## ERROR HANDLING RULES (from mattpocock/skills diagnose)

If something fails:

1. **Reproduce** — What exact command failed? What was the error?
2. **Minimize** — Can you reproduce with a smaller input? Fewer episodes?
3. **Hypothesize** — What are the 3 most likely causes?
4. **Instrument** — Add print statements to narrow it down
5. **Fix** — Apply the minimal fix
6. **Regression** — Verify the fix works and nothing else broke

**Never silently skip a step.** If MetaWorld API is different from expected, look up the actual API. If Octo's interface doesn't match, read the source code.

---

## TIMELINE

| Phase | Time | Gate |
|-------|------|------|
| 0. Setup | 5 min | All imports work, GPU detected |
| 1. Understand | 5 min | Can explain Octo's input/output format |
| 2. KSAM core | 15 min | Unit tests pass, Jacobian verified |
| 3. Eval pipeline | 10 min | Can run 1 episode on 1 task |
| 4. Experiment | 20 min | JSON results file exists |
| 5. Plots | 2 min | 3 PNG files exist |
| **Total** | **~57 min** | **All 5 output files exist** |

---

## WHAT NOT TO DO

- ❌ Don't install unnecessary packages
- ❌ Don't write code you can't test immediately
- ❌ Don't proceed past a checkpoint if it failed
- ❌ Don't fine-tune Octo's weights (KSAM is wrapper-only)
- ❌ Don't skip the Jacobian numerical verification
- ❌ Don't use placeholder/identity Jacobians
- ❌ Don't commit broken code
- ❌ Don't spend >5 min debugging any single issue — try a simpler approach first

---

## FINAL NOTE

This is a **proof-of-concept**, not a paper submission. The goal is:

1. Working code that wraps Octo with KSAM
2. Comparison numbers showing KSAM's effect on MT-10
3. Plots that can go into a paper

If the improvement is small or negative — **that's fine**. Report it honestly. A negative result with proper methodology is more valuable than a fake positive result.

Now go. Time starts now.
