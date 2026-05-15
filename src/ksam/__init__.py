"""KSAM: Kinematic Singularity Awareness Module for VLA models."""

from .module import KSAMWrapper
from .jacobian import (
    compute_jacobian, compute_condition_number, compute_manipulability,
    damped_pseudo_inverse, forward_kinematics, PANDA_DH
)

__version__ = "0.2.0"
__all__ = [
    "KSAMWrapper",
    "compute_jacobian",
    "compute_condition_number",
    "compute_manipulability",
    "damped_pseudo_inverse",
    "forward_kinematics",
    "PANDA_DH",
]
