"""
KSAM: Kinematic Singularity Awareness Module.
Wraps any policy with Jacobian-conditioned safety gating.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from .jacobian import (
    compute_jacobian, compute_condition_number, compute_manipulability,
    damped_pseudo_inverse, PANDA_DH
)


class KSAMWrapper(nn.Module):
    """
    Wraps a frozen policy with kinematic singularity awareness.
    
    When the arm approaches a singularity (high Jacobian condition number),
    KSAM blends the policy's action toward a damped pseudo-inverse fallback
    that prevents joint velocity explosions.
    
    Trainable params: ~4.2K (gating MLP + alpha + kappa_threshold)
    """
    
    def __init__(
        self,
        policy: nn.Module,
        action_dim: int = 4,
        damping: float = 0.1,
        hidden_dim: int = 64,
        device: str = "cuda",
    ):
        super().__init__()
        self.policy = policy
        self.action_dim = action_dim
        self.damping = damping
        self.device = device
        
        # Freeze base policy
        for p in self.policy.parameters():
            p.requires_grad = False
        self.policy.eval()
        
        # Gating MLP: joint_angles -> gate scalar
        self.gate_mlp = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        
        # Learnable singularity threshold and steepness
        self.alpha = nn.Parameter(torch.tensor(5.0))
        self.kappa_threshold = nn.Parameter(torch.tensor(50.0))
    
    def forward(
        self,
        joint_angles: torch.Tensor,
        observation: Optional[Dict] = None,
        return_debug: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Args:
            joint_angles: [batch, 7] joint configuration
            observation: dict for base policy (images, etc.)
        
        Returns:
            action: [batch, action_dim] safe action
            debug_info: optional dict with kappa, gate, etc.
        """
        batch_size = joint_angles.shape[0]
        
        # 1. Compute Jacobian and condition number
        J = compute_jacobian(joint_angles)  # [B, 6, 7]
        kappa = compute_condition_number(J)  # [B]
        manip = compute_manipulability(J)    # [B]
        
        # 2. Gating signal
        gate_condition = torch.sigmoid(
            self.alpha * (self.kappa_threshold - kappa)
        )  # [B]
        gate_mlp = self.gate_mlp(joint_angles).squeeze(-1)  # [B]
        gate = (gate_condition * gate_mlp).clamp(0.0, 1.0)  # [B]
        
        # 3. Base policy action (frozen)
        with torch.no_grad():
            if observation is not None:
                vla_action = self.policy(observation)
            else:
                vla_action = self.policy(joint_angles)
        
        # Ensure action_dim match
        if vla_action.shape[-1] < self.action_dim:
            pad = torch.zeros(batch_size, self.action_dim - vla_action.shape[-1],
                            device=vla_action.device, dtype=vla_action.dtype)
            vla_action = torch.cat([vla_action, pad], dim=-1)
        vla_action = vla_action[:, :self.action_dim]
        
        # 4. Safe fallback via damped pseudo-inverse
        # Map task-space action through J to get safe joint velocities, then back
        J_pinv = damped_pseudo_inverse(J, self.damping)  # [B, 7, 6]
        
        # Assume first 3 dims of action are Cartesian velocity
        ee_cmd = vla_action[:, :3].unsqueeze(-1)  # [B, 3, 1]
        # Pad to 6D for full spatial velocity
        ee_cmd_6d = torch.zeros(batch_size, 6, 1, device=vla_action.device, dtype=vla_action.dtype)
        ee_cmd_6d[:, :3] = ee_cmd
        
        joint_vel = torch.bmm(J_pinv, ee_cmd_6d).squeeze(-1)  # [B, 7]
        ee_vel_safe = torch.bmm(J, joint_vel.unsqueeze(-1)).squeeze(-1)[:, :3]  # [B, 3]
        
        # 5. Blend: a_final = g * a_vla + (1-g) * a_safe
        safe_action = vla_action.clone()
        g = gate.unsqueeze(-1)
        safe_action[:, :3] = g * vla_action[:, :3] + (1 - g) * ee_vel_safe
        
        if return_debug:
            return safe_action, {
                "kappa": kappa,
                "manipulability": manip,
                "gate": gate,
                "gate_condition": gate_condition,
                "gate_mlp": gate_mlp,
                "vla_action": vla_action,
                "safe_action": safe_action,
                "is_near_singularity": kappa > self.kappa_threshold.item(),
            }
        return safe_action, None
    
    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def total_params(self):
        return sum(p.numel() for p in self.parameters())
