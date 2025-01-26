import torch
from typing import Optional, Tuple

class DeadFeatureTracker:
    def __init__(
        self,
        num_features: int,
        activation_threshold: float = 1e-8,
        window_size: int = 10_000_000,
        update_interval: int = 10_000
    ):
        self.num_features = num_features
        self.activation_threshold = activation_threshold
        self.window_size = window_size
        self.update_interval = update_interval
        
        self.samples_seen = 0
        self.current_batch_size = 0
        
        # Initialize last activation times to -window_size to avoid false dead features
        self.last_activation = torch.ones(num_features, dtype=torch.long) * (-window_size)
        
    def update(self, feature_activations: torch.Tensor) -> Tuple[float, Optional[dict]]:
        if not isinstance(feature_activations, torch.Tensor):
            feature_activations = torch.tensor(feature_activations)
            
        batch_size = feature_activations.shape[0]
        self.current_batch_size += batch_size
        self.samples_seen += batch_size
        
        # Find which features were active in this batch
        active_features = (feature_activations.abs() >= self.activation_threshold).any(dim=0)
        
        # Update last activation times for active features
        self.last_activation[active_features] = self.samples_seen
        
        # Only count features as dead if we've seen enough samples
        if self.samples_seen >= self.window_size:
            samples_since_activation = self.samples_seen - self.last_activation
            dead_features = samples_since_activation >= self.window_size
            current_dead_ratio = dead_features.float().mean().item()
        else:
            current_dead_ratio = 0.0
        
        # Generate detailed stats if we've hit the update interval
        stats = None
        if self.current_batch_size >= self.update_interval:
            self.current_batch_size = 0
            stats = self._generate_stats()
            
        return current_dead_ratio, stats
    
    def get_dead_features(self) -> torch.Tensor:
        """Return boolean mask of currently dead features."""
        if self.samples_seen < self.window_size:
            return torch.zeros(self.num_features, dtype=torch.bool)
            
        samples_since_activation = self.samples_seen - self.last_activation
        return samples_since_activation >= self.window_size
    
    def _generate_stats(self) -> dict:
        """Generate detailed statistics about feature usage."""
        if self.samples_seen < self.window_size:
            dead_ratio = 0.0
        else:
            dead_features = self.get_dead_features()
            dead_ratio = dead_features.float().mean().item()
        
        samples_since_activation = self.samples_seen - self.last_activation
        samples_since_activation = samples_since_activation.float() / self.window_size
        
        return {
            'dead_ratio': dead_ratio,
            'samples_seen': self.samples_seen,
            'max_samples_since_activation': samples_since_activation.max().item(),
            'median_samples_since_activation': samples_since_activation.median().item(),
        }