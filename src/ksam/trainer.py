"""
KSAM Trainer - standalone training utilities.
Main experiment pipeline is in scripts/run_ksam_experiment.py.
"""

import torch
import torch.nn as nn
import time
from typing import Dict, List


class KSAMTrainer:
    """Lightweight trainer for KSAM gating parameters."""
    
    def __init__(self, model, lr=1e-3, device="cuda"):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=lr
        )
    
    def train_step(self, qpos, target_actions, kappa_values):
        """Single training step."""
        self.optimizer.zero_grad()
        
        safe_action, debug = self.model(qpos, return_debug=True)
        
        # Action matching
        action_loss = nn.MSELoss()(safe_action, target_actions)
        
        # Singularity avoidance: low gate when kappa is high
        is_near = (kappa_values > 50.0).float()
        sing_loss = (debug["gate"] * is_near).mean()
        
        # Smoothness
        smooth_loss = torch.abs(debug["gate"][1:] - debug["gate"][:-1]).mean() if len(debug["gate"]) > 1 else 0
        
        loss = action_loss + 5.0 * sing_loss + 0.1 * smooth_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], 1.0
        )
        self.optimizer.step()
        
        return {
            "loss": loss.item(),
            "action_loss": action_loss.item(),
            "gate_mean": debug["gate"].mean().item(),
            "kappa_mean": debug["kappa"].mean().item(),
        }
