#!/usr/bin/env python3
"""
KSAM Experiment Pipeline
========================
Trains a baseline policy on MetaWorld, identifies singularity failures,
then applies KSAM wrapper and measures improvement.

Usage:
    python scripts/run_ksam_experiment.py --task reach-v2 --episodes 500 --gpu 0
    
Expected runtime on A100: ~45-50 minutes
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ksam.module import KSAMWrapper
from src.ksam.jacobian import (
    compute_jacobian, compute_condition_number, compute_manipulability,
    damped_pseudo_inverse, forward_kinematics, PANDA_DH
)


# ============================================================
# 1. SIMPLE POLICY NETWORK
# ============================================================

class SimplePolicy(nn.Module):
    """MLP policy: joint_angles (7) -> action (4: dx,dy,dz,gripper)."""
    
    def __init__(self, obs_dim=20, act_dim=4, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim), nn.Tanh(),
        )
    
    def forward(self, obs):
        if isinstance(obs, dict):
            obs = obs["observation"]
        return self.net(obs)


# ============================================================
# 2. REPLAY BUFFER
# ============================================================

class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = []
        self.capacity = capacity
    
    def push(self, state, action, reward, next_state, done, joint_angles, next_joint_angles):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done, joint_angles, next_joint_angles))
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones, joints, next_joints = zip(*[self.buffer[i] for i in indices])
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1),
            torch.FloatTensor(np.array(joints)),
            torch.FloatTensor(np.array(next_joints)),
        )
    
    def __len__(self):
        return len(self.buffer)


# ============================================================
# 3. SAC AGENT (Simplified)
# ============================================================

class SACAgent:
    """Simplified SAC for fast training on MetaWorld."""
    
    def __init__(self, obs_dim, act_dim, hidden=256, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2, device="cuda"):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        
        self.actor = SimplePolicy(obs_dim, act_dim, hidden).to(device)
        self.critic1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        ).to(device)
        self.critic2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        ).to(device)
        self.target_critic1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        ).to(device)
        self.target_critic2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        ).to(device)
        
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optim = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optim = optim.Adam(self.critic2.parameters(), lr=lr)
    
    def select_action(self, obs, noise=0.1):
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action = self.actor(obs_t).cpu().numpy()[0]
        if noise > 0:
            action += np.random.normal(0, noise, size=action.shape)
            action = np.clip(action, -1.0, 1.0)
        return action
    
    def update(self, replay_buffer, batch_size=256):
        if len(replay_buffer) < batch_size:
            return {}
        
        states, actions, rewards, next_states, dones, joints, next_joints = replay_buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Critic update
        with torch.no_grad():
            next_actions = self.actor(next_states)
            next_sa = torch.cat([next_states, next_actions], dim=-1)
            target_q = rewards + self.gamma * (1 - dones) * torch.min(
                self.target_critic1(next_sa), self.target_critic2(next_sa)
            )
        
        sa = torch.cat([states, actions], dim=-1)
        q1_loss = nn.MSELoss()(self.critic1(sa), target_q)
        q2_loss = nn.MSELoss()(self.critic2(sa), target_q)
        
        self.critic1_optim.zero_grad()
        q1_loss.backward()
        self.critic1_optim.step()
        
        self.critic2_optim.zero_grad()
        q2_loss.backward()
        self.critic2_optim.step()
        
        # Actor update
        new_actions = self.actor(states)
        new_sa = torch.cat([states, new_actions], dim=-1)
        actor_loss = -torch.min(self.critic1(new_sa), self.critic2(new_sa)).mean()
        
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        
        # Soft update targets
        for p, tp in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for p, tp in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        
        return {"q1": q1_loss.item(), "q2": q2_loss.item(), "actor": actor_loss.item()}


# ============================================================
# 4. METAWORLD ENVIRONMENT WRAPPER
# ============================================================

class MetaWorldEnv:
    """
    MetaWorld environment wrapper that provides joint angles for KSAM.
    
    Observation: [ee_pos(3), ee_vel(3), obj_pos(3), obj_vel(3), 
                  joint_pos(7), joint_vel(7), gripper(1)] = 24 dims
    Action: [dx, dy, dz, gripper] = 4 dims
    """
    
    def __init__(self, task_name="reach-v2", seed=42, max_steps=200):
        try:
            import metaworld
            from metaworld.policies.sawyer_reach_v2_policy import SawyerReachV2Policy
        except ImportError:
            print("ERROR: MetaWorld not installed. Install with: pip install metaworld")
            print("  Or: pip install git+https://github.com/Farama-Foundation/Metaworld.git")
            sys.exit(1)
        
        self.task_name = task_name
        self.max_steps = max_steps
        
        # Initialize MetaWorld
        ml1 = metaworld.ML1(task_name, seed=seed)
        self.env = ml1.train_classes[task_name]()
        self.task = ml1.train_tasks[0]
        self.env.set_task(self.task)
        
        # Expert policy for generating demonstrations
        self.expert = SawyerReachV2Policy()
        
        self.step_count = 0
    
    def reset(self):
        obs = self.env.reset()
        self.step_count = 0
        self.env.set_task(self.task)
        return self._process_obs(obs)
    
    def _process_obs(self, obs):
        """Extract full observation including joint angles."""
        # MetaWorld Sawyer observation: 
        # [ee_pos(3), gripper_finger(1), obj_pos(3), obj_vel(3), 
        #  ee_pos(3), gripper_finger(1), obj_pos(3), obj_vel(3)] = 24
        # But joint angles are in env.sim.data.qpos[:7]
        
        qpos = self.env.sim.data.qpos[:7].copy()  # Joint angles
        qvel = self.env.sim.data.qvel[:7].copy()  # Joint velocities
        ee_pos = obs[:3]  # End-effector position
        obj_pos = obs[4:7] if len(obs) > 6 else np.zeros(3)
        
        # Construct observation with joint angles
        full_obs = np.concatenate([
            ee_pos,          # 3
            qpos,            # 7 (joint angles - KEY for KSAM)
            qvel,            # 7 (joint velocities)
            obj_pos,         # 3
        ]).astype(np.float32)
        
        return full_obs, qpos.copy()
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.step_count += 1
        
        full_obs, qpos = self._process_obs(obs)
        
        # Check success
        success = info.get("success", 0.0)
        
        # Compute singularity metrics
        with torch.no_grad():
            q_t = torch.FloatTensor(qpos).unsqueeze(0)
            J = compute_jacobian(q_t)
            kappa = compute_condition_number(J).item()
            manip = compute_manipulability(J).item()
        
        done = done or self.step_count >= self.max_steps
        
        return full_obs, qpos, reward, done, {
            "success": success,
            "kappa": kappa,
            "manipulability": manip,
            "is_near_singularity": kappa > 50.0,
        }
    
    def expert_action(self, obs):
        """Get expert action for demonstration generation."""
        # Expert uses raw MetaWorld observation format
        raw_obs = self.env._get_obs()
        return self.expert.get_action(raw_obs)


# ============================================================
# 5. TRAINING PIPELINE
# ============================================================

def collect_demonstrations(env, num_episodes=100, max_steps=200):
    """Collect expert demonstrations and record singularity statistics."""
    print(f"\n{'='*60}")
    print(f"COLLECTING {num_episodes} EXPERT DEMONSTRATIONS")
    print(f"{'='*60}")
    
    data = []
    stats = {
        "episodes": 0, "total_steps": 0, "successes": 0,
        "singularity_steps": 0, "max_kappa": 0, "kappa_values": [],
        "episode_successes": [], "episode_kappas": [],
    }
    
    for ep in range(num_episodes):
        obs_data = env.reset()
        obs, qpos = obs_data
        ep_reward = 0
        ep_kappas = []
        ep_success = False
        
        for step in range(max_steps):
            # Get expert action
            action = env.expert_action(obs)
            
            # Step environment
            next_obs, next_qpos, reward, done, info = env.step(action)
            
            # Store transition
            data.append({
                "obs": obs, "action": action, "reward": reward,
                "next_obs": next_obs, "done": done,
                "qpos": qpos, "next_qpos": next_qpos,
                "kappa": info["kappa"], "manipulability": info["manipulability"],
            })
            
            ep_reward += reward
            ep_kappas.append(info["kappa"])
            stats["kappa_values"].append(info["kappa"])
            stats["total_steps"] += 1
            
            if info["is_near_singularity"]:
                stats["singularity_steps"] += 1
            
            if info["success"] > 0.5:
                ep_success = True
            
            obs, qpos = next_obs, next_qpos
            if done:
                break
        
        stats["episodes"] += 1
        if ep_success:
            stats["successes"] += 1
        stats["episode_successes"].append(float(ep_success))
        stats["episode_kappas"].append(np.mean(ep_kappas))
        stats["max_kappa"] = max(stats["max_kappa"], max(ep_kappas))
        
        if (ep + 1) % 20 == 0:
            print(f"  Episode {ep+1}/{num_episodes}: "
                  f"success={ep_success}, avg_kappa={np.mean(ep_kappas):.1f}, "
                  f"max_kappa={max(ep_kappas):.1f}")
    
    print(f"\n  Expert success rate: {stats['successes']}/{stats['episodes']} "
          f"= {stats['successes']/stats['episodes']:.1%}")
    print(f"  Singularity steps: {stats['singularity_steps']}/{stats['total_steps']} "
          f"= {stats['singularity_steps']/stats['total_steps']:.1%}")
    print(f"  Max kappa: {stats['max_kappa']:.1f}")
    print(f"  Mean kappa: {np.mean(stats['kappa_values']):.1f}")
    
    return data, stats


def train_baseline_policy(data, obs_dim, act_dim, num_epochs=50, batch_size=256, 
                          lr=3e-4, device="cuda"):
    """Train a simple policy on demonstrations using behavioral cloning."""
    print(f"\n{'='*60}")
    print(f"TRAINING BASELINE POLICY (Behavioral Cloning)")
    print(f"{'='*60}")
    
    policy = SimplePolicy(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    
    # Prepare data
    obs_tensor = torch.FloatTensor(np.array([d["obs"] for d in data])).to(device)
    act_tensor = torch.FloatTensor(np.array([d["action"] for d in data])).to(device)
    
    dataset = torch.utils.data.TensorDataset(obs_tensor, act_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    start_time = time.time()
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        n_batches = 0
        for obs_batch, act_batch in dataloader:
            pred = policy(obs_batch)
            loss = nn.MSELoss()(pred, act_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.6f}")
    
    elapsed = time.time() - start_time
    print(f"  Training completed in {elapsed:.1f}s")
    print(f"  Final loss: {losses[-1]:.6f}")
    
    return policy


def train_sac_policy(env, obs_dim, act_dim, num_episodes=300, max_steps=200,
                     batch_size=256, device="cuda"):
    """Train a policy with SAC from scratch."""
    print(f"\n{'='*60}")
    print(f"TRAINING SAC POLICY ({num_episodes} episodes)")
    print(f"{'='*60}")
    
    agent = SACAgent(obs_dim, act_dim, device=device)
    buffer = ReplayBuffer()
    
    stats = {"rewards": [], "successes": [], "kappas": []}
    start_time = time.time()
    
    for ep in range(num_episodes):
        obs_data = env.reset()
        obs, qpos = obs_data
        ep_reward = 0
        ep_success = False
        ep_kappas = []
        
        for step in range(max_steps):
            action = agent.select_action(obs, noise=0.3 * max(0.1, 1.0 - ep / num_episodes))
            next_obs, next_qpos, reward, done, info = env.step(action)
            
            buffer.push(obs, action, reward, next_obs, float(done), qpos, next_qpos)
            
            if len(buffer) > batch_size:
                agent.update(buffer, batch_size)
            
            ep_reward += reward
            ep_kappas.append(info["kappa"])
            if info["success"] > 0.5:
                ep_success = True
            
            obs, qpos = next_obs, next_qpos
            if done:
                break
        
        stats["rewards"].append(ep_reward)
        stats["successes"].append(float(ep_success))
        stats["kappas"].append(np.mean(ep_kappas))
        
        if (ep + 1) % 50 == 0:
            recent_succ = np.mean(stats["successes"][-50:])
            recent_kappa = np.mean(stats["kappas"][-50:])
            print(f"  Episode {ep+1}/{num_episodes}: "
                  f"success={recent_succ:.1%}, avg_kappa={recent_kappa:.1f}, "
                  f"reward={np.mean(stats['rewards'][-50:]):.2f}")
    
    elapsed = time.time() - start_time
    print(f"  Training completed in {elapsed:.1f}s")
    print(f"  Final 50-ep success rate: {np.mean(stats['successes'][-50:]):.1%}")
    
    return agent.actor, stats


# ============================================================
# 6. KSAM TRAINING
# ============================================================

def train_ksam(policy, data, num_epochs=30, lr=1e-3, device="cuda"):
    """Train KSAM wrapper on collected data."""
    print(f"\n{'='*60}")
    print(f"TRAINING KSAM WRAPPER")
    print(f"{'='*60}")
    
    ksam = KSAMWrapper(policy, action_dim=4, device=device).to(device)
    
    print(f"  Trainable params: {ksam.trainable_params():,}")
    print(f"  Total params: {ksam.total_params():,}")
    
    # Only optimize KSAM parameters
    optimizer = optim.Adam(
        [p for p in ksam.parameters() if p.requires_grad],
        lr=lr
    )
    
    # Prepare data
    qpos_data = torch.FloatTensor(np.array([d["qpos"] for d in data])).to(device)
    target_actions = torch.FloatTensor(np.array([d["action"] for d in data])).to(device)
    kappa_data = torch.FloatTensor(np.array([d["kappa"] for d in data])).to(device)
    
    dataset = torch.utils.data.TensorDataset(qpos_data, target_actions, kappa_data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    
    start_time = time.time()
    history = {"loss": [], "gate": [], "kappa": []}
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_gate = 0
        epoch_kappa = 0
        n_batches = 0
        
        for qpos_b, target_b, kappa_b in dataloader:
            # Forward
            safe_action, debug = ksam(qpos_b, return_debug=True)
            
            # Loss: match expert action + penalize high gate near singularity
            action_loss = nn.MSELoss()(safe_action, target_b)
            
            # Singularity loss: gate should be LOW (safe) when kappa is HIGH
            is_near = (kappa_b > 50.0).float()
            sing_loss = (debug["gate"] * is_near).mean()
            
            # Smoothness
            smooth_loss = torch.abs(debug["gate"][1:] - debug["gate"][:-1]).mean() if len(debug["gate"]) > 1 else 0
            
            loss = action_loss + 5.0 * sing_loss + 0.1 * smooth_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in ksam.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_gate += debug["gate"].mean().item()
            epoch_kappa += debug["kappa"].mean().item()
            n_batches += 1
        
        history["loss"].append(epoch_loss / n_batches)
        history["gate"].append(epoch_gate / n_batches)
        history["kappa"].append(epoch_kappa / n_batches)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}: "
                  f"loss={history['loss'][-1]:.4f}, "
                  f"gate={history['gate'][-1]:.3f}, "
                  f"kappa={history['kappa'][-1]:.1f}")
    
    elapsed = time.time() - start_time
    print(f"  KSAM training completed in {elapsed:.1f}s")
    
    return ksam, history


# ============================================================
# 7. EVALUATION
# ============================================================

def evaluate_policy(env, policy, num_episodes=100, max_steps=200, use_ksam=False, 
                    ksam=None, device="cuda", label="Policy"):
    """Evaluate a policy and collect detailed statistics."""
    print(f"\n{'='*60}")
    print(f"EVALUATING: {label}")
    print(f"{'='*60}")
    
    stats = {
        "successes": 0, "episodes": num_episodes,
        "rewards": [], "episode_lengths": [],
        "singularity_episodes": 0,  # episodes that hit singularity
        "singularity_failure_episodes": 0,  # failed AND hit singularity
        "kappa_values": [], "max_kappas": [],
        "gate_values": [], "success_per_kappa_bin": defaultdict(list),
    }
    
    for ep in range(num_episodes):
        obs_data = env.reset()
        obs, qpos = obs_data
        ep_reward = 0
        ep_success = False
        ep_kappas = []
        ep_gates = []
        hit_singularity = False
        
        for step in range(max_steps):
            with torch.no_grad():
                if use_ksam and ksam is not None:
                    qpos_t = torch.FloatTensor(qpos).unsqueeze(0).to(device)
                    action_tensor, debug = ksam(qpos_t, return_debug=True)
                    action = action_tensor.cpu().numpy()[0]
                    ep_gates.append(debug["gate"].cpu().numpy()[0])
                else:
                    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    action = policy(obs_t).cpu().numpy()[0]
                    action = np.clip(action, -1.0, 1.0)
            
            next_obs, next_qpos, reward, done, info = env.step(action)
            
            ep_reward += reward
            ep_kappas.append(info["kappa"])
            
            if info["is_near_singularity"]:
                hit_singularity = True
            
            if info["success"] > 0.5:
                ep_success = True
            
            obs, qpos = next_obs, next_qpos
            if done:
                break
        
        stats["rewards"].append(ep_reward)
        stats["episode_lengths"].append(step + 1)
        stats["kappa_values"].extend(ep_kappas)
        stats["max_kappas"].append(max(ep_kappas))
        
        if ep_success:
            stats["successes"] += 1
        
        if hit_singularity:
            stats["singularity_episodes"] += 1
            if not ep_success:
                stats["singularity_failure_episodes"] += 1
        
        if ep_gates:
            stats["gate_values"].extend(ep_gates)
        
        # Bin by max kappa
        max_k = max(ep_kappas)
        if max_k < 10:
            bin_label = "low(<10)"
        elif max_k < 50:
            bin_label = "med(10-50)"
        elif max_k < 100:
            bin_label = "high(50-100)"
        else:
            bin_label = "extreme(>100)"
        stats["success_per_kappa_bin"][bin_label].append(float(ep_success))
    
    # Print results
    sr = stats["successes"] / stats["episodes"]
    print(f"  Success rate: {stats['successes']}/{stats['episodes']} = {sr:.1%}")
    print(f"  Mean reward: {np.mean(stats['rewards']):.2f}")
    print(f"  Mean episode length: {np.mean(stats['episode_lengths']):.1f}")
    print(f"  Episodes hitting singularity: {stats['singularity_episodes']} ({stats['singularity_episodes']/num_episodes:.1%})")
    print(f"  Singularity-caused failures: {stats['singularity_failure_episodes']}")
    print(f"  Mean kappa: {np.mean(stats['kappa_values']):.1f}")
    print(f"  Max kappa (mean): {np.mean(stats['max_kappas']):.1f}")
    
    if stats["gate_values"]:
        print(f"  Mean KSAM gate: {np.mean(stats['gate_values']):.3f}")
    
    # Success rate by kappa bin
    print(f"\n  Success rate by singularity exposure:")
    for bin_label in ["low(<10)", "med(10-50)", "high(50-100)", "extreme(>100)"]:
        if bin_label in stats["success_per_kappa_bin"]:
            bin_results = stats["success_per_kappa_bin"][bin_label]
            bin_sr = np.mean(bin_results) if bin_results else 0
            print(f"    {bin_label}: {bin_sr:.1%} ({len(bin_results)} episodes)")
    
    return stats


# ============================================================
# 8. MAIN EXPERIMENT
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="KSAM Experiment Pipeline")
    parser.add_argument("--task", type=str, default="reach-v2", help="MetaWorld task")
    parser.add_argument("--episodes", type=int, default=500, help="Training episodes")
    parser.add_argument("--eval_episodes", type=int, default=100, help="Evaluation episodes")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    args = parser.parse_args()
    
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Task: {args.task}")
    print(f"Seed: {args.seed}")
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    os.makedirs(args.output, exist_ok=True)
    
    # ---- Phase 1: Environment setup ----
    print("\n" + "="*60)
    print("PHASE 1: ENVIRONMENT SETUP")
    print("="*60)
    
    env = MetaWorldEnv(task_name=args.task, seed=args.seed)
    obs_dim = 20  # ee_pos(3) + qpos(7) + qvel(7) + obj_pos(3)
    act_dim = 4   # dx, dy, dz, gripper
    
    # ---- Phase 2: Collect demonstrations ----
    demo_data, demo_stats = collect_demonstrations(env, num_episodes=100)
    
    # ---- Phase 3: Train baseline policy (BC) ----
    baseline_policy = train_baseline_policy(
        demo_data, obs_dim, act_dim, num_epochs=50, device=device
    )
    
    # ---- Phase 4: Train SAC policy ----
    sac_policy, sac_train_stats = train_sac_policy(
        env, obs_dim, act_dim, num_episodes=args.episodes, device=device
    )
    
    # ---- Phase 5: Evaluate baseline ----
    baseline_stats = evaluate_policy(
        env, baseline_policy, num_episodes=args.eval_episodes,
        device=device, label="Baseline (BC Policy)"
    )
    
    sac_stats = evaluate_policy(
        env, sac_policy, num_episodes=args.eval_episodes,
        device=device, label="SAC Policy"
    )
    
    # ---- Phase 6: Train KSAM wrapper on SAC policy ----
    ksam, ksam_train_history = train_ksam(sac_policy, demo_data, device=device)
    
    # ---- Phase 7: Evaluate KSAM ----
    ksam_stats = evaluate_policy(
        env, sac_policy, num_episodes=args.eval_episodes,
        use_ksam=True, ksam=ksam, device=device, label="SAC + KSAM"
    )
    
    # ---- Phase 8: Compile results ----
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    results = {
        "task": args.task,
        "seed": args.seed,
        "device": device,
        "baseline": {
            "success_rate": baseline_stats["successes"] / baseline_stats["episodes"],
            "mean_reward": float(np.mean(baseline_stats["rewards"])),
            "singularity_episodes": baseline_stats["singularity_episodes"],
            "singularity_failures": baseline_stats["singularity_failure_episodes"],
            "mean_kappa": float(np.mean(baseline_stats["kappa_values"])),
            "max_kappa_mean": float(np.mean(baseline_stats["max_kappas"])),
        },
        "sac": {
            "success_rate": sac_stats["successes"] / sac_stats["episodes"],
            "mean_reward": float(np.mean(sac_stats["rewards"])),
            "singularity_episodes": sac_stats["singularity_episodes"],
            "singularity_failures": sac_stats["singularity_failure_episodes"],
            "mean_kappa": float(np.mean(sac_stats["kappa_values"])),
            "max_kappa_mean": float(np.mean(sac_stats["max_kappas"])),
        },
        "sac_ksam": {
            "success_rate": ksam_stats["successes"] / ksam_stats["episodes"],
            "mean_reward": float(np.mean(ksam_stats["rewards"])),
            "singularity_episodes": ksam_stats["singularity_episodes"],
            "singularity_failures": ksam_stats["singularity_failure_episodes"],
            "mean_kappa": float(np.mean(ksam_stats["kappa_values"])),
            "max_kappa_mean": float(np.mean(ksam_stats["max_kappas"])),
            "mean_gate": float(np.mean(ksam_stats["gate_values"])) if ksam_stats["gate_values"] else 0,
        },
        "ksam_params": ksam.trainable_params(),
        "success_by_kappa_bin": {
            "baseline": {k: float(np.mean(v)) for k, v in baseline_stats["success_per_kappa_bin"].items()},
            "sac": {k: float(np.mean(v)) for k, v in sac_stats["success_per_kappa_bin"].items()},
            "sac_ksam": {k: float(np.mean(v)) for k, v in ksam_stats["success_per_kappa_bin"].items()},
        },
    }
    
    # Print comparison table
    print(f"\n{'Method':<20} {'Success':>10} {'Sing. Fail':>12} {'Mean κ':>10} {'Max κ':>10}")
    print("-" * 65)
    print(f"{'Baseline (BC)':<20} {results['baseline']['success_rate']:>9.1%} "
          f"{results['baseline']['singularity_failures']:>11d} "
          f"{results['baseline']['mean_kappa']:>9.1f} "
          f"{results['baseline']['max_kappa_mean']:>9.1f}")
    print(f"{'SAC':<20} {results['sac']['success_rate']:>9.1%} "
          f"{results['sac']['singularity_failures']:>11d} "
          f"{results['sac']['mean_kappa']:>9.1f} "
          f"{results['sac']['max_kappa_mean']:>9.1f}")
    print(f"{'SAC + KSAM':<20} {results['sac_ksam']['success_rate']:>9.1%} "
          f"{results['sac_ksam']['singularity_failures']:>11d} "
          f"{results['sac_ksam']['mean_kappa']:>9.1f} "
          f"{results['sac_ksam']['max_kappa_mean']:>9.1f}")
    
    improvement = results['sac_ksam']['success_rate'] - results['sac']['success_rate']
    print(f"\n  KSAM improvement over SAC: {improvement:+.1%}")
    print(f"  KSAM trainable params: {results['ksam_params']:,}")
    
    # Singularity failure reduction
    if results['sac']['singularity_failures'] > 0:
        fail_reduction = 1 - results['sac_ksam']['singularity_failures'] / results['sac']['singularity_failures']
        print(f"  Singularity failure reduction: {fail_reduction:.1%}")
    
    # Save results
    output_path = os.path.join(args.output, "ksam_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {output_path}")
    
    # Save models
    torch.save(baseline_policy.state_dict(), os.path.join(args.output, "baseline_policy.pt"))
    torch.save(sac_policy.state_dict(), os.path.join(args.output, "sac_policy.pt"))
    torch.save(ksam.state_dict(), os.path.join(args.output, "ksam_wrapper.pt"))
    print(f"  Models saved to {args.output}/")
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
