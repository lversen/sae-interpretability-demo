import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional
from deadfeatures import DeadFeatureTracker
import time
from datetime import timedelta
from torch.cuda.amp import GradScaler, autocast

class SparseTransformer(nn.Module):
    def __init__(self, X, D: int, F: int, M: int, st_model_path: str, 
                 lambda_l1: float = 5.0, num_heads: int = 8, device: str = 'cuda',
                 window_size: int = 10_000_000, update_interval: int = 10_000,
                 activation_threshold: float = 1e-3):
        super(SparseTransformer, self).__init__()
        
        # Ensure model directory exists
        os.makedirs(os.path.dirname(st_model_path), exist_ok=True)
        self.D = D  # Input dimension
        self.F = F  # Feature dimension
        self.M = M  # Attention dimension
        self.st_model_path = st_model_path
        self.lambda_l1 = lambda_l1
        self.device = device
        self.num_heads = num_heads
        self.activation_threshold = activation_threshold

        # Ensure M is divisible by num_heads
        if M % num_heads != 0:
            new_M = ((M // num_heads) + 1) * num_heads
            print(f"Warning: Adjusting attention dimension from {M} to {new_M} to be divisible by {num_heads} heads")
            self.M = new_M

        # Initialize layers
        self.input_proj = nn.Linear(D, self.M)
        self.output_proj = nn.Linear(self.M, D)
        self.attention = nn.MultiheadAttention(self.M, num_heads, batch_first=True)
        
        # Initialize weights properly
        self._init_weights()
        # Layer normalization for better stability
        self.norm_q = nn.LayerNorm(self.M)
        self.norm_kv = nn.LayerNorm(self.M)
        
        # Initialize feature tracking
        self.feature_tracker = DeadFeatureTracker(
            num_features=self.F,
            activation_threshold=self.activation_threshold,
            window_size=window_size,
            update_interval=update_interval
        )
        
        # Initialize feature subset and move to device
        self.initialize_feature_subset(X)
        self.to(device)
        
    def _init_weights(self):
        # Linear layers: Kaiming Normal
        nn.init.kaiming_normal_(self.input_proj.weight, mode='fan_in', nonlinearity='linear')
        nn.init.kaiming_normal_(self.output_proj.weight, mode='fan_in', nonlinearity='linear')
        
        # Smaller initialization for attention weights (PyTorch doesn't expose these directly)
        # Workaround: Re-initialize MultiheadAttention parameters
        for name, param in self.attention.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=0.02)  # Transformer-style init
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        
        # Initialize feature subset (X_data) with small values
        if hasattr(self, 'X_data'):
            nn.init.normal_(self.X_data, mean=0.0, std=0.01)

    def initialize_feature_subset(self, X):
        """Initialize with small random values, not input data."""
        self.X_data = nn.Parameter(
            torch.randn((self.F, self.D), device=self.device) * 0.01  # Small initial values
        )

    def forward(self, x):
        """Forward pass with caching for attention weights and value norms."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32)).to(self.device)
        elif isinstance(x, torch.Tensor) and x.device != self.device:
            x = x.to(self.device)
            
        # Project inputs to attention dimension
        query = self.input_proj(x)
        key = self.input_proj(self.X_data)
        value = self.input_proj(self.X_data)
        
        # Apply layer normalization
        query = self.norm_q(query)
        key = self.norm_kv(key)
        value = self.norm_kv(value)
        
        # Apply attention and get weights
        attn_output, attention_weights = self.attention(
            query=query,
            key=key,
            value=value,
            need_weights=True
        )
        
        # Cache attention weights and value norms for feature activations
        self.cached_attention_weights = attention_weights
        self.cached_V_norms = torch.norm(value, p=2, dim=-1)
        
        # Project back to input dimension
        x_hat = self.output_proj(attn_output)
        
        # Update feature tracking during training
        if self.training:
            feature_acts = attention_weights * self.cached_V_norms[None, :]
            dead_ratio, _ = self.feature_tracker.update(feature_acts)
        
        return x, x_hat, attention_weights

    def compute_losses(self, x: torch.Tensor, x_hat: torch.Tensor, attention_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute losses with improved regularization."""
        # L2 reconstruction loss with normalization
        L2_loss = torch.mean(torch.sum((x - x_hat)**2, dim=1)) / self.D
        
        # L1 regularization on attention weights with scaling
        L1_loss = self.lambda_l1 * torch.mean(torch.sum(attention_weights, dim=-1))
        
        total_loss = L2_loss + L1_loss
        return total_loss, L2_loss, L1_loss

    def feature_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Get feature activations using cached values."""
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x.astype(np.float32)).to(self.device)
            self.forward(x)  # Ensure cached values are updated
            return self.cached_attention_weights * self.cached_V_norms[None, :]

    def preprocess(self, X, normalize: bool = True):
        """Preprocess input data using optional L2 normalization."""
        if not isinstance(X, (np.ndarray, torch.Tensor)):
            raise ValueError("Input data must be a numpy array or torch tensor.")
        
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X.astype(np.float32)).to(self.device)
        elif isinstance(X, torch.Tensor) and X.device != self.device:
            X = X.to(self.device)

        if normalize:
            C = torch.mean(torch.norm(X, p=2, dim=1)) / self.D
            X_normalized = X / C
            return X_normalized
            return X
    def gini_coefficient(self, tensor: torch.Tensor) -> float:
        sorted_vals, _ = torch.sort(tensor.flatten().abs())
        n = sorted_vals.shape[0]
        idx = torch.arange(1, n+1, device=tensor.device)
        return (torch.sum(sorted_vals * idx) / (n * torch.sum(sorted_vals)) - (n + 1) / (2 * n)).item()
    def train_and_validate(self, X_train, X_val, learning_rate: float = 1e-3, batch_size: int = 4096, 
                        target_steps: int = 200000, grad_clip: float = 1.0, log_freq: Optional[int] = None):
        """Train with improved scheduling, mixed precision, and time estimation."""
        scaler = GradScaler()  # For mixed precision training
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # Preprocess data (ensure it's on the CPU if using pin_memory)
        X_train = self.preprocess(X_train, normalize=True).cpu()  # Move to CPU for pin_memory
        X_val = self.preprocess(X_val, normalize=True).cpu()      # Move to CPU for pin_memory
        
        train_dataset = TensorDataset(X_train)
        val_dataset = TensorDataset(X_val)
        
        # Use pin_memory=True only if data is on CPU
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)
        
        steps_per_epoch = len(train_loader)
        num_epochs = max(1, target_steps // steps_per_epoch)
        total_steps = num_epochs * steps_per_epoch
        warmup_steps = total_steps // 20
        
        if log_freq is None:
            log_freq = steps_per_epoch // 5
        
        print("\nTraining Configuration:")
        print(f"Total Steps: {total_steps}")
        print(f"Warmup Steps: {warmup_steps}")
        print(f"Batch Size: {batch_size}")
        print(f"Initial Learning Rate: {learning_rate}")
        print(f"Number of Attention Heads: {self.num_heads}")
        print(f"Gradient Clipping: {grad_clip}")
        
        best_val_loss = float('inf')
        step = 0
        l1 = self.lambda_l1
        
        start_time = time.time()
        times_per_step = []
        
        for epoch in range(num_epochs):
            self.train()
            for batch in train_loader:
                step_start_time = time.time()
                batch = batch[0].to(self.device)  # Move batch to GPU
                
                # Warmup phase
                if step < warmup_steps:
                    lr = learning_rate * (step / warmup_steps)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                    self.lambda_l1 = l1 * (step / warmup_steps)
                else:
                    self.lambda_l1 = l1
                
                optimizer.zero_grad()
                with autocast():  # Mixed precision
                    x, x_hat, attention_weights = self.forward(batch)
                    total_loss, L2_loss, L1_loss = self.compute_losses(x, x_hat, attention_weights)
                scaler.scale(total_loss).backward()  # Scale loss for mixed precision
                torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                
                # Update timing statistics
                step_time = time.time() - step_start_time
                times_per_step.append(step_time)
                if len(times_per_step) > 10000:  # Keep last 100 steps for moving average
                    times_per_step.pop(0)
                
                # Logging and validation
                if step % log_freq == 0:
                    self.eval()
                    val_loss = 0.0
                    val_batches = 0
                    
                    with torch.no_grad():
                        for val_batch in val_loader:
                            val_x = val_batch[0].to(self.device)  # Move validation batch to GPU
                            val_x, val_x_hat, val_attn = self.forward(val_x)
                            val_total_loss, _, _ = self.compute_losses(val_x, val_x_hat, val_attn)
                            val_loss += val_total_loss.item()
                            val_batches += 1
                    
                    avg_val_loss = val_loss / val_batches
                    dead_ratio = len(self.feature_tracker.get_dead_features()) / self.F if self.feature_tracker.samples_seen >= self.feature_tracker.window_size else 0.0
                    sparsity = (attention_weights <= self.activation_threshold).float().mean().item()
                    tracking_progress = min(self.feature_tracker.samples_seen / self.feature_tracker.window_size, 1.0)
                    
                    # Calculate time estimates
                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    avg_time_per_step = np.mean(times_per_step)
                    remaining_steps = total_steps - step
                    estimated_remaining_time = avg_time_per_step * remaining_steps
                    
                    # Format time strings
                    elapsed_str = str(timedelta(seconds=int(elapsed_time)))
                    remaining_str = str(timedelta(seconds=int(estimated_remaining_time)))
                    # During training logging:
                    gini = self.gini_coefficient(attention_weights)
                    print(
                        f"Step: {step:6d}/{total_steps} ({step/total_steps:3.1%}) | "
                        f"Epoch: {epoch:3d} | "
                        f"Train Loss: {total_loss.item():8.4f} | "
                        f"Val Loss: {avg_val_loss:8.4f} | "
                        f"L1 λ: {self.lambda_l1:4.2f} | "
                        f"Dead: {dead_ratio:5.1%} | "
                        f"Sparse: {sparsity:5.1%} | "
                        f"Sparsity Gini: {gini:.3f} | "
                        f"Fill: {tracking_progress:5.1%} | "
                        f"Time remaining: {remaining_str}"
                    )
                    
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        torch.save(self.state_dict(), self.st_model_path)
                    
                    self.train()
                
                step += 1
        
        total_time = time.time() - start_time
        total_time_str = str(timedelta(seconds=int(total_time)))
        
        print(f"\nTraining completed in {total_time_str}")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Final λ: {self.lambda_l1:.2f}")