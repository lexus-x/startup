"""
Jacobian computation and singularity analysis for robotic manipulators.
Supports Sawyer, Franka, and Kinova robot morphologies.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


def compute_jacobian(joint_angles: torch.Tensor, robot_type: str = "sawyer") -> torch.Tensor:
    """
    Compute the analytical Jacobian matrix for a given joint configuration.
    
    Args:
        joint_angles: Joint angles [batch_size, n_joints]
        robot_type: Robot morphology ("sawyer", "franka", "kinova")
    
    Returns:
        Jacobian matrix [batch_size, 6, n_joints]
    """
    batch_size, n_joints = joint_angles.shape
    
    # Simplified DH parameter-based Jacobian computation
    # In production, use robot-specific kinematics libraries (e.g., pinocchio, kdl)
    
    if robot_type == "sawyer":
        # Sawyer: 7 DoF
        assert n_joints == 7, f"Sawyer expects 7 joints, got {n_joints}"
        return _compute_sawyer_jacobian(joint_angles)
    elif robot_type == "franka":
        # Franka Emika Panda: 7 DoF
        assert n_joints == 7, f"Franka expects 7 joints, got {n_joints}"
        return _compute_franka_jacobian(joint_angles)
    elif robot_type == "kinova":
        # Kinova Jaco: 6 or 7 DoF
        return _compute_kinova_jacobian(joint_angles, n_joints)
    else:
        raise ValueError(f"Unsupported robot type: {robot_type}")


def _compute_sawyer_jacobian(q: torch.Tensor) -> torch.Tensor:
    """Compute Jacobian for Rethink Sawyer (7 DoF)."""
    batch_size = q.shape[0]
    J = torch.zeros(batch_size, 6, 7, device=q.device, dtype=q.dtype)
    
    # Simplified kinematic model (replace with exact DH parameters)
    # Link lengths (meters)
    L = [0.081, 0.0, 0.364, 0.0, 0.364, 0.0, 0.107]
    
    c = torch.cos(q)
    s = torch.sin(q)
    
    # Column-by-column Jacobian computation
    # J[:, :, i] = z_i x (p_ee - p_i) for revolute joints
    
    # This is a simplified approximation - use exact forward kinematics in production
    for i in range(7):
        # Angular velocity component (z-axis of joint i in base frame)
        J[:, :3, i] = 0.0  # Simplified
        
        # Linear velocity component
        for j in range(i, 7):
            # Accumulate contributions from distal links
            pass
    
    # Placeholder: return identity-like structure for demonstration
    # Replace with proper kinematic chain computation
    J[:, 0, 0] = 1.0
    J[:, 1, 1] = 1.0
    J[:, 2, 2] = 1.0
    J[:, 3, 3] = 1.0
    J[:, 4, 4] = 1.0
    J[:, 5, 5] = 1.0
    J[:, 5, 6] = 0.5  # Coupling term
    
    return J


def _compute_franka_jacobian(q: torch.Tensor) -> torch.Tensor:
    """Compute Jacobian for Franka Emika Panda (7 DoF)."""
    batch_size = q.shape[0]
    J = torch.zeros(batch_size, 6, 7, device=q.device, dtype=q.dtype)
    
    # Similar structure to Sawyer, different link parameters
    # Placeholder implementation
    J[:, 0, 0] = 1.0
    J[:, 1, 1] = 1.0
    J[:, 2, 2] = 1.0
    J[:, 3, 3] = 1.0
    J[:, 4, 4] = 1.0
    J[:, 5, 5] = 1.0
    J[:, 5, 6] = 0.3
    
    return J


def _compute_kinova_jacobian(q: torch.Tensor, n_joints: int) -> torch.Tensor:
    """Compute Jacobian for Kinova Jaco (6 or 7 DoF)."""
    batch_size = q.shape[0]
    J = torch.zeros(batch_size, 6, n_joints, device=q.device, dtype=q.dtype)
    
    # Placeholder implementation
    for i in range(min(6, n_joints)):
        J[:, i, i] = 1.0
    
    return J


def compute_condition_number(J: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute the condition number of the Jacobian matrix.
    
    κ(J) = σ_max(J) / σ_min(J)
    
    High condition numbers indicate proximity to kinematic singularities.
    
    Args:
        J: Jacobian matrix [batch_size, 6, n_joints]
        eps: Small constant for numerical stability
    
    Returns:
        Condition number [batch_size]
    """
    # Compute singular values using SVD
    try:
        U, S, Vh = torch.linalg.svd(J, full_matrices=False)
        sigma_max = S[:, 0]  # Largest singular value
        sigma_min = S[:, -1]  # Smallest singular value
        
        # Avoid division by zero
        sigma_min = torch.clamp(sigma_min, min=eps)
        
        kappa = sigma_max / sigma_min
        return kappa
    
    except torch.linalg.LinAlgError:
        # Fallback: return large value indicating singularity
        return torch.full((J.shape[0],), 1e6, device=J.device, dtype=J.dtype)


def damped_pseudo_inverse(
    J: torch.Tensor,
    damping_factor: float = 0.1,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Compute the damped least-squares pseudo-inverse of the Jacobian.
    
    J^# = J^T (J J^T + λ²I)^{-1}
    
    This provides a stable inverse near singularities at the cost of accuracy.
    
    Args:
        J: Jacobian matrix [batch_size, 6, n_joints]
        damping_factor: Damping coefficient λ
        eps: Small constant for numerical stability
    
    Returns:
        Damped pseudo-inverse [batch_size, n_joints, 6]
    """
    batch_size, m, n = J.shape
    
    # Compute J J^T
    JJT = torch.bmm(J, J.transpose(-2, -1))  # [batch, 6, 6]
    
    # Add damping: J J^T + λ²I
    I = torch.eye(m, device=J.device, dtype=J.dtype).unsqueeze(0)  # [1, 6, 6]
    damped_JJT = JJT + (damping_factor ** 2) * I
    
    # Compute inverse
    try:
        JJT_inv = torch.linalg.inv(damped_JJT)
    except torch.linalg.LinAlgError:
        # Use pseudo-inverse if inversion fails
        JJT_inv = torch.linalg.pinv(damped_JJT)
    
    # J^# = J^T (J J^T + λ²I)^{-1}
    J_dagger = torch.bmm(J.transpose(-2, -1), JJT_inv)
    
    return J_dagger


class KinematicSafetyLayer(nn.Module):
    """
    PyTorch module for kinematic safety monitoring.
    
    Computes condition numbers and provides safe fallback actions
    when the robot approaches singular configurations.
    """
    
    def __init__(self, robot_type: str = "sawyer", damping_factor: float = 0.1):
        super().__init__()
        self.robot_type = robot_type
        self.damping_factor = damping_factor
    
    def forward(self, joint_angles: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing kinematic safety metrics.
        
        Args:
            joint_angles: Joint configuration [batch_size, n_joints]
        
        Returns:
            Dictionary containing:
                - jacobian: Jacobian matrix
                - condition_number: κ(q)
                - damping_mask: Binary mask indicating singularity proximity
        """
        J = compute_jacobian(joint_angles, self.robot_type)
        kappa = compute_condition_number(J)
        
        # Binary mask: 1 if near singularity (κ > threshold)
        threshold = 100.0  # Typical threshold for industrial robots
        damping_mask = (kappa > threshold).float()
        
        return {
            "jacobian": J,
            "condition_number": kappa,
            "damping_mask": damping_mask,
        }
