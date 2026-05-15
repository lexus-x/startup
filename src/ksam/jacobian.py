"""
Real Jacobian computation for Franka Panda (7-DOF) using DH parameters.
Used by KSAM to detect kinematic singularities in real-time.
"""

import torch
import numpy as np
from typing import Optional

# Franka Emika Panda DH parameters (standard convention)
# [a, alpha, d, theta_offset]
PANDA_DH = torch.tensor([
    [0.0,     -torch.pi/2, 0.333,  0.0],
    [0.0,      torch.pi/2, 0.0,    0.0],
    [0.0825,   torch.pi/2, 0.316,  0.0],
    [-0.0825, -torch.pi/2, 0.0,    0.0],
    [0.0,      torch.pi/2, 0.384,  0.0],
    [0.088,    torch.pi/2, 0.0,    0.0],
    [0.0,      0.0,        0.107,  0.0],
], dtype=torch.float32)


def forward_kinematics(q, dh_params=PANDA_DH):
    """
    Compute forward kinematics for Panda arm.
    
    Args:
        q: Joint angles [batch, 7]
        dh_params: DH parameter table [7, 4]
    
    Returns:
        T_ee: End-effector pose [batch, 4, 4]
        T_list: List of intermediate transforms [batch, 7, 4, 4]
    """
    batch_size = q.shape[0]
    device = q.device
    dtype = q.dtype
    dh = dh_params.to(device=device, dtype=dtype)
    
    T_list = []
    T = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1).clone()
    
    for i in range(7):
        a_i = dh[i, 0]
        alpha_i = dh[i, 1]
        d_i = dh[i, 2]
        theta_i = q[:, i] + dh[i, 3]
        
        ct, st = torch.cos(theta_i), torch.sin(theta_i)
        ca, sa = torch.cos(alpha_i), torch.sin(alpha_i)
        
        T_i = torch.zeros(batch_size, 4, 4, device=device, dtype=dtype)
        T_i[:, 0, 0] = ct
        T_i[:, 0, 1] = -st * ca
        T_i[:, 0, 2] = st * sa
        T_i[:, 0, 3] = a_i * ct
        T_i[:, 1, 0] = st
        T_i[:, 1, 1] = ct * ca
        T_i[:, 1, 2] = -ct * sa
        T_i[:, 1, 3] = a_i * st
        T_i[:, 2, 1] = sa
        T_i[:, 2, 2] = ca
        T_i[:, 2, 3] = d_i
        T_i[:, 3, 3] = 1.0
        
        T = torch.bmm(T, T_i)
        T_list.append(T.clone())
    
    return T, T_list


def compute_jacobian(q, dh_params=PANDA_DH):
    """
    Compute 6x7 geometric Jacobian for Panda arm.
    
    J_i = [z_{i-1} x (p_ee - p_{i-1}); z_{i-1}]
    
    Args:
        q: Joint angles [batch, 7]
    
    Returns:
        J: Geometric Jacobian [batch, 6, 7]
    """
    batch_size = q.shape[0]
    device = q.device
    dtype = q.dtype
    
    T_ee, T_list = forward_kinematics(q, dh_params)
    p_ee = T_ee[:, :3, 3]
    
    J = torch.zeros(batch_size, 6, 7, device=device, dtype=dtype)
    
    for i in range(7):
        T_i = T_list[i]
        z_i = T_i[:, :3, 2]
        p_i = T_i[:, :3, 3]
        
        dp = p_ee - p_i
        J[:, :3, i] = torch.cross(z_i, dp)
        J[:, 3:6, i] = z_i
    
    return J


def compute_condition_number(J, eps=1e-6):
    """
    Compute Jacobian condition number: kappa = sigma_max / sigma_min.
    High kappa -> near singularity -> unsafe.
    """
    S = torch.linalg.svd(J, full_matrices=False)[1]
    sigma_max = S[:, 0]
    sigma_min = torch.clamp(S[:, -1], min=eps)
    return sigma_max / sigma_min


def compute_manipulability(J, eps=1e-6):
    """Yoshikawa manipulability: w = sqrt(det(J @ J^T)). Low w -> near singularity."""
    JJT = torch.bmm(J, J.transpose(-2, -1))
    det = torch.clamp(torch.linalg.det(JJT), min=0.0)
    return torch.sqrt(det + eps)


def damped_pseudo_inverse(J, damping=0.1):
    """Damped least-squares pseudo-inverse: J# = J^T (JJ^T + lambda^2 I)^{-1}"""
    JJT = torch.bmm(J, J.transpose(-2, -1))
    I = torch.eye(6, device=J.device, dtype=J.dtype).unsqueeze(0)
    damped_inv = torch.linalg.inv(JJT + damping**2 * I)
    return torch.bmm(J.transpose(-2, -1), damped_inv)
