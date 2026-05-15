"""
KSAM Trainer
Training loop for KSAM parameters with frozen VLA backbone.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
from tqdm import tqdm
import time


class KSAMTrainer:
    """
    Trainer for KSAM wrapper parameters.
    
    The base VLA model remains frozen - only KSAM gating parameters are updated.
    Training is extremely fast (~12-18 minutes) due to minimal parameter count.
    """
    
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        device: str = "cuda",
        singularity_weight: float = 10.0,
        action_weight: float = 1.0,
    ):
        """
        Initialize KSAM trainer.
        
        Args:
            model: KSAMWrapper instance
            lr: Learning rate
            device: Training device ("cuda" or "cpu")
            singularity_weight: Weight for singularity avoidance loss
            action_weight: Weight for action matching loss
        """
        self.model = model.to(device)
        self.device = device
        self.singularity_weight = singularity_weight
        self.action_weight = action_weight
        
        # Only optimize KSAM parameters (VLA is frozen)
        self.optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
        )
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
    
    def compute_loss(
        self,
        prediction: torch.Tensor,
        target_action: torch.Tensor,
        debug_info: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Compute composite loss for KSAM training.
        
        Loss = action_weight * L_action + singularity_weight * L_singularity
        
        Args:
            prediction: Predicted action from KSAM
            target_action: Ground truth action
            debug_info: Debug information from forward pass
        
        Returns:
            Total loss scalar
        """
        # Action matching loss: ensure KSAM doesn't degrade nominal performance
        action_loss = self.mse_loss(prediction, target_action)
        
        # Singularity avoidance loss: penalize high gate values near singularities
        kappa = debug_info["condition_number"]
        gate = debug_info["gate_value"]
        
        # Desired: gate should be low when kappa is high (near singularity)
        # Penalize: high gate when kappa > threshold
        is_near_singularity = (kappa > self.model.kappa_threshold).float()
        singularity_loss = self.mse_loss(gate, torch.zeros_like(gate)) * is_near_singularity.mean()
        
        # Alternative: encourage smooth gating behavior
        # Penalize rapid changes in gate value
        smoothness_loss = torch.mean(torch.abs(gate[:-1] - gate[1:])) if len(gate) > 1 else 0.0
        
        # Total loss
        total_loss = (
            self.action_weight * action_loss +
            self.singularity_weight * singularity_loss +
            0.1 * smoothness_loss  # Small weight for smoothness
        )
        
        return total_loss
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: Dictionary containing:
                - "observation": Input observation dict
                - "action": Ground truth action
        
        Returns:
            Dictionary with loss components
        """
        # Move data to device
        observation = self._move_to_device(batch["observation"])
        target_action = batch["action"].to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        prediction, debug_info = self.model(observation, return_debug_info=True)
        
        # Compute loss
        loss = self.compute_loss(prediction, target_action, debug_info)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            max_norm=1.0,
        )
        
        # Optimizer step
        self.optimizer.step()
        
        return {
            "total_loss": loss.item(),
            "action_loss": self.mse_loss(prediction, target_action).item(),
            "singularity_loss": debug_info["condition_number"].mean().item(),
            "mean_gate": debug_info["gate_value"].mean().item(),
            "mean_kappa": debug_info["condition_number"].mean().item(),
        }
    
    def train(
        self,
        dataloader: DataLoader,
        num_epochs: int = 50,
        log_interval: int = 10,
    ) -> Dict[str, list]:
        """
        Full training loop.
        
        Args:
            dataloader: PyTorch DataLoader with training data
            num_epochs: Number of training epochs
            log_interval: Log every N batches
        
        Returns:
            Training history dictionary
        """
        history = {
            "total_loss": [],
            "action_loss": [],
            "mean_kappa": [],
            "mean_gate": [],
        }
        
        start_time = time.time()
        total_batches = len(dataloader) * num_epochs
        batch_count = 0
        
        print(f"Starting KSAM training on {self.device}")
        print(f"Trainable parameters: {self.model.get_num_trainable_params():,}")
        print(f"Total epochs: {num_epochs}")
        print("-" * 60)
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in pbar:
                metrics = self.train_step(batch)
                epoch_losses.append(metrics)
                
                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{metrics['total_loss']:.4f}",
                    "κ": f"{metrics['mean_kappa']:.1f}",
                    "gate": f"{metrics['mean_gate']:.3f}",
                })
                
                batch_count += 1
                
                # Log intermediate results
                if batch_count % log_interval == 0:
                    avg_loss = sum(m["total_loss"] for m in epoch_losses[-log_interval:]) / log_interval
                    print(f"  Batch {batch_count}/{total_batches}: avg_loss={avg_loss:.4f}")
            
            # Epoch summary
            epoch_avg = {k: sum(m[k] for m in epoch_losses) / len(epoch_losses) for k in history.keys()}
            for k, v in epoch_avg.items():
                history[k].append(v)
            
            print(f"Epoch {epoch+1} complete: avg_loss={epoch_avg['total_loss']:.4f}")
        
        total_time = time.time() - start_time
        print("-" * 60)
        print(f"Training completed in {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"Final loss: {history['total_loss'][-1]:.4f}")
        
        return history
    
    def _move_to_device(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively move observation dictionary to device."""
        moved_obs = {}
        for key, value in obs.items():
            if isinstance(value, torch.Tensor):
                moved_obs[key] = value.to(self.device)
            elif isinstance(value, dict):
                moved_obs[key] = self._move_to_device(value)
            else:
                moved_obs[key] = value
        return moved_obs
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "alpha": self.model.alpha.item(),
            "kappa_threshold": self.model.kappa_threshold.item(),
        }, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        with torch.no_grad():
            self.model.alpha.fill_(checkpoint["alpha"])
            self.model.kappa_threshold.fill_(checkpoint["kappa_threshold"])
        print(f"Checkpoint loaded from {path}")
