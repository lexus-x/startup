#!/usr/bin/env python3
"""Quick sanity check: verify Jacobian computation and singularity detection."""

import sys
sys.path.insert(0, "src")
import torch
from ksam.jacobian import compute_jacobian, compute_condition_number, compute_manipulability, forward_kinematics

def test_jacobian():
    print("Testing Jacobian computation...")
    
    # Home position (all zeros) - should NOT be singular
    q_home = torch.zeros(1, 7)
    J = compute_jacobian(q_home)
    kappa = compute_condition_number(J).item()
    manip = compute_manipulability(J).item()
    print(f"  Home position:     κ={kappa:.1f}, manip={manip:.4f}")
    assert kappa < 100, f"Home position should not be singular, got κ={kappa}"
    
    # Fully extended (all zeros except elbow straight) - SHOULD be near singular
    q_extended = torch.tensor([[0.0, -0.3, 0.0, -2.0, 0.0, 1.5, 0.0]])
    J = compute_jacobian(q_extended)
    kappa = compute_condition_number(J).item()
    manip = compute_manipulability(J).item()
    print(f"  Extended config:   κ={kappa:.1f}, manip={manip:.4f}")
    
    # Near-singular configuration (elbow near zero)
    q_sing = torch.tensor([[0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0]])
    J = compute_jacobian(q_sing)
    kappa = compute_condition_number(J).item()
    manip = compute_manipulability(J).item()
    print(f"  Near-singular:     κ={kappa:.1f}, manip={manip:.4f}")
    
    # Batch computation
    q_batch = torch.randn(8, 7)
    J_batch = compute_jacobian(q_batch)
    kappa_batch = compute_condition_number(J_batch)
    manip_batch = compute_manipulability(J_batch)
    print(f"  Batch (8): κ range=[{kappa_batch.min():.1f}, {kappa_batch.max():.1f}]")
    
    # FK check: end-effector should be within Panda reach (~0.855m)
    T_ee, _ = forward_kinematics(q_home)
    pos = T_ee[0, :3, 3]
    reach = torch.norm(pos).item()
    print(f"  Home EE position: {pos.numpy()}, reach={reach:.3f}m")
    assert reach < 1.0, f"Reach should be < 1m for Panda, got {reach}"
    
    print("\n  ✓ All Jacobian tests passed!")
    return True

def test_ksam_wrapper():
    print("\nTesting KSAM wrapper...")
    from ksam.module import KSAMWrapper
    import torch.nn as nn
    
    # Dummy policy
    dummy_policy = nn.Linear(20, 4)
    
    ksam = KSAMWrapper(dummy_policy, action_dim=4)
    print(f"  Trainable params: {ksam.trainable_params():,}")
    print(f"  Total params: {ksam.total_params():,}")
    
    q = torch.randn(4, 7)
    action, debug = ksam(q, return_debug=True)
    
    print(f"  Action shape: {action.shape}")
    print(f"  Kappa: {debug['kappa'].detach().numpy()}")
    print(f"  Gate: {debug['gate'].detach().numpy()}")
    
    # Gate should be between 0 and 1
    assert (debug['gate'] >= 0).all() and (debug['gate'] <= 1).all(), "Gate out of range"
    
    print("\n  ✓ All KSAM wrapper tests passed!")
    return True


if __name__ == "__main__":
    test_jacobian()
    test_ksam_wrapper()
    print("\n=== ALL TESTS PASSED ===")
