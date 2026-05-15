"""
KSAM Wrapper Module
Integrates kinematic singularity awareness with frozen VLA models.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from .jacobian import compute_jacobian, compute_condition_number, damped_pseudo_inverse


class KSAMWrapper(nn.Module):
    """
    Kinematic Singularity Awareness Module (KSAM) wrapper for VLA models.
    
    This module wraps a frozen VLA model and adds a learnable gating mechanism
    that interpolates between VLA actions and safe fallback controls when the
    robot approaches kinematic singularities.
    
    The base VLA weights remain 100% frozen - only the gating MLP parameters
    are trained (~4.2K parameters total).
    """
    
    def __init__(
        self,
        vla_model: nn.Module,
        robot_type: str = "sawyer",
        damping_factor: float = 0.1,
        hidden_dim: int = 64,
    ):
        """
        Initialize KSAM wrapper.
        
        Args:
            vla_model: Pretrained VLA model (weights will be frozen)
            robot_type: Robot morphology ("sawyer", "franka", "kinova")
            damping_factor: Damping coefficient for pseudo-inverse fallback
            hidden_dim: Hidden dimension for gating MLP
        """
        super().__init__()
        
        # Store frozen VLA model
        self.vla_model = vla_model
        self.robot_type = robot_type
        self.damping_factor = damping_factor
        
        # Freeze VLA parameters
        for param in self.vla_model.parameters():
            param.requires_grad = False
        self.vla_model.eval()
        
        # Determine number of joints based on robot type
        self.n_joints = 7 if robot_type in ["sawyer", "franka"] else 6
        
        # Gating MLP: joint angles -> gating scalar
        # Architecture: [n_joints] -> [hidden_dim] -> [hidden_dim] -> [1]
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.n_joints, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        
        # Learnable parameters for gating function
        self.alpha = nn.Parameter(torch.tensor(5.0))  # Steepness
        self.kappa_threshold = nn.Parameter(torch.tensor(100.0))  # Singularity threshold
        
        # Action dimension (will be inferred from VLA output)
        self.action_dim = None
    
    def forward(
        self,
        observation: Dict[str, Any],
        return_debug_info: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through KSAM-wrapped VLA.
        
        Args:
            observation: Dictionary containing:
                - "image": Visual input [B, H, W, C] or [B, C, H, W]
                - "language": Language embedding [B, seq_len, dim]
                - "proprioception": Proprioceptive state
                    - "joint_angles": Joint configuration [B, n_joints]
                    - "end_effector_pos": End-effector position [B, 3] (optional)
            return_debug_info: If True, return additional debug information
        
        Returns:
            action: Safe action tensor [B, action_dim]
            debug_info (optional): Dictionary with intermediate values
        """
        # Extract proprioception
        proprio = observation["proprioception"]
        joint_angles = proprio["joint_angles"]  # [B, n_joints]
        
        batch_size = joint_angles.shape[0]
        
        # Step 1: Compute Jacobian and condition number
        J = compute_jacobian(joint_angles, self.robot_type)  # [B, 6, n_joints]
        kappa = compute_condition_number(J)  # [B]
        
        # Step 2: Compute gating signal
        # g = σ(α * (κ_threshold - κ(q)))
        gate_input = self.alpha * (self.kappa_threshold - kappa)  # [B]
        gate_mlp_output = self.gate_mlp(joint_angles).squeeze(-1)  # [B]
        
        # Combine learned gate with condition-based modulation
        g = torch.sigmoid(gate_input) * gate_mlp_output  # [B]
        g = g.clamp(0.0, 1.0)  # Ensure valid range
        
        # Step 3: Get VLA action (frozen backbone)
        with torch.no_grad():
            vla_action = self.vla_model(observation)  # [B, action_dim]
        
        if self.action_dim is None:
            self.action_dim = vla_action.shape[-1]
        
        # Step 4: Compute safe fallback action using damped pseudo-inverse
        # For demonstration: use damped IK to track desired end-effector velocity
        # In practice, this could be a simple damping controller or impedance control
        
        # Extract desired end-effector velocity from VLA action
        # Assuming action format: [vx, vy, vz, wx, wy, wz, gripper]
        ee_vel_cmd = vla_action[:, :6]  # [B, 6]
        
        # Compute joint velocities via damped pseudo-inverse
        J_dagger = damped_pseudo_inverse(J, self.damping_factor)  # [B, n_joints, 6]
        joint_vel_safe = torch.bmm(J_dagger, ee_vel_cmd.unsqueeze(-1)).squeeze(-1)  # [B, n_joints]
        
        # Convert joint velocities back to task space for consistent action format
        # This ensures a_safe has the same format as a_VLA
        ee_vel_safe = torch.bmm(J, joint_vel_safe.unsqueeze(-1)).squeeze(-1)  # [B, 6]
        
        # Construct safe action (preserve gripper command from VLA)
        safe_action = torch.cat([ee_vel_safe, vla_action[:, 6:]], dim=-1)  # [B, action_dim]
        
        # Step 5: Blend VLA action and safe action
        # a_final = g * a_VLA + (1 - g) * a_safe
        g_expanded = g.unsqueeze(-1).expand_as(vla_action)  # [B, action_dim]
        final_action = g_expanded * vla_action + (1.0 - g_expanded) * safe_action
        
        if return_debug_info:
            debug_info = {
                "condition_number": kappa,
                "gate_value": g,
                "vla_action": vla_action,
                "safe_action": safe_action,
                "is_near_singularity": kappa > self.kappa_threshold.item(),
            }
            return final_action, debug_info
        
        return final_action
    
    def get_num_trainable_params(self) -> int:
        """Return the number of trainable parameters (excluding frozen VLA)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self) -> int:
        """Return total parameter count including frozen VLA."""
        return sum(p.numel() for p in self.parameters())
