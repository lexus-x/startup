"""
KSAM: Kinematic Singularity Awareness Module
A modular safety wrapper for Vision-Language-Action models.
"""

from .module import KSAMWrapper
from .jacobian import compute_jacobian, compute_condition_number, damped_pseudo_inverse
from .trainer import KSAMTrainer

__version__ = "0.1.0"
__all__ = [
    "KSAMWrapper",
    "compute_jacobian",
    "compute_condition_number",
    "damped_pseudo_inverse",
    "KSAMTrainer",
]
