import torch
import numpy as np
from collections import deque
from typing import Optional, Tuple

class DeadFeatureTracker:
    def __init__(
        self,
        num_features: int,
        activation_threshold: float = 1e-3,
        window_size: int = 10_000_000,
        update_interval: int = 10_000
    ):
        """
        Track dead features over a large number of samples.
        
        Args:
            num_features: Number of features to track
            activation_threshold: Threshold below which a feature activation is considered inactive
            window_size: Number of samples to track (default 10M as per paper)
            update_interval: How often to update the tracking window
        """
        self.num_features = num_features
        self.activation_threshold = activation_threshold
        self.window_size = window_size
        self.update_interval = update_interval
        
        # Initialize tracking arrays
        self.samples_seen = 0
        self.current_batch_size = 0
        
        # Track the last activation times for each feature
        self.last_activation = torch.zeros(num_features, dtype=torch.long)
        
        # Keep track of activation history for reporting
        self.activation_history = deque(maxlen=100)  # Keep last 100 activation ratios
        
    def update(self, feature_activations: torch.Tensor) -> Tuple[float, Optional[dict]]:
        """
        Update tracking with new batch of activations.
        
        Args:
            feature_activations: Tensor of shape [batch_size, num_features] containing feature activations
            
        Returns:
            current_dead_ratio: Fraction of features currently considered dead
            stats: Optional dict with additional statistics if update_interval is reached
        """
        if not isinstance(feature_activations, torch.Tensor):
            feature_activations = torch.tensor(feature_activations)
            
        batch_size = feature_activations.shape[0]
        self.current_batch_size += batch_size
        self.samples_seen += batch_size
        
        # Find which features were active in this batch
        active_features = (feature_activations.abs() >= self.activation_threshold).any(dim=0)
        
        # Update last activation times for active features
        self.last_activation[active_features] = self.samples_seen
        
        # Calculate current dead features (those not activated within window_size samples)
        samples_since_activation = self.samples_seen - self.last_activation
        dead_features = samples_since_activation >= self.window_size
        current_dead_ratio = dead_features.float().mean().item()
        
        # Store activation ratio history
        self.activation_history.append(1.0 - current_dead_ratio)
        
        # Generate detailed stats if we've hit the update interval
        stats = None
        if self.current_batch_size >= self.update_interval:
            self.current_batch_size = 0
            stats = self._generate_stats(dead_features)
            
        return current_dead_ratio, stats
    
    def _generate_stats(self, dead_features: torch.Tensor) -> dict:
        """Generate detailed statistics about feature usage."""
        dead_ratio = dead_features.float().mean().item()
        
        # Calculate average activation ratio over recent history
        avg_activation = sum(self.activation_history) / len(self.activation_history)
        
        # Calculate distribution of times since last activation
        samples_since_activation = self.samples_seen - self.last_activation
        samples_since_activation = samples_since_activation.float() / self.window_size
        
        return {
            'dead_ratio': dead_ratio,
            'samples_seen': self.samples_seen,
            'avg_activation_ratio': avg_activation,
            'max_samples_since_activation': samples_since_activation.max().item(),
            'median_samples_since_activation': samples_since_activation.median().item(),
        }
    
    def get_dead_features(self) -> torch.Tensor:
        """Return boolean mask of currently dead features."""
        samples_since_activation = self.samples_seen - self.last_activation
        return samples_since_activation >= self.window_size