import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional
from deadfeatures import DeadFeatureTracker
import math

class SparseTransformer(nn.Module):
    def __init__(self, X, n: int, m: int, a: int, st_model_path: str,
                 lambda_l1: float = 5.0, num_heads: int = 1, device: str = 'cuda',
                 window_size: int = 10_000_000, update_interval: int = 10_000,
                 activation_threshold: float = 1e-3):
        """
        Initialize Sparse Transformer model
        
        Args:
            X: Reference data used for attention
            n: Input dimension
            m: Feature dimension (number of features)
            a: Attention dimension
            st_model_path: Path to save model
            lambda_l1: L1 regularization strength
            num_heads: Number of attention heads
            device: Device to use (cuda or cpu)
            window_size: Window size for dead feature tracking
            update_interval: Interval for dead feature stats updates
            activation_threshold: Threshold for considering a feature activated
        """
        super().__init__()
        
        # Store parameters
        self.n, self.m, self.a = n, m, a
        self.st_model_path = st_model_path
        self.lambda_l1 = lambda_l1
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.activation_threshold = activation_threshold
        
        # Store reference data for attention
        self.X_data = self.type_check(X)
        
        # Training state
        self.steps = 0
        self.total_steps = 0
        self.memory_update_freq = 1
        
        self.W_q = nn.Linear(n, a)
        self.W_k = nn.Linear(n, a)
        self.W_v = nn.Linear(n, n)
        
        # Normalization layers
        self.norm_q = nn.LayerNorm(a)
        self.norm_k = nn.LayerNorm(a)
        self.norm_v = nn.LayerNorm(n)
        
        # Initialize feature tracker
        self.feature_tracker = DeadFeatureTracker(
            num_features=self.m,
            activation_threshold=activation_threshold,
            window_size=window_size,
            update_interval=update_interval
        )
        
        # Initialize memory indices - randomly sample from reference data
        self.register_buffer('memory_indices', 
                           torch.randint(0, self.X_data.shape[0], (m,), device=self.device))
                           
        # Initialize weights properly
        self.initialize_weights()
        self.to(self.device)
    
    def initialize_weights(self):
        """
        Initialize weights according to updated SAE training configuration:
        - Projection matrices initialized with random values
        - Output projection initialized to have columns with L2 norms between 0.05 and 1
        """
        with torch.no_grad():
            # Initialize input projection normally
            nn.init.xavier_normal_(self.W_q.weight)
            nn.init.xavier_normal_(self.W_k.weight)
            
            # Initialize output projection similar to SAE decoder
            output_weight = torch.randn(self.n, self.a)
            
            # Normalize columns to have unit norm
            norms = torch.norm(output_weight, p=2, dim=0)
            output_weight = output_weight / norms
            
            # Scale columns to have norms between 0.05 and 1
            target_norms = 0.05 + 0.95 * torch.rand(self.a)
            output_weight = output_weight * target_norms
            
            # Assign to model parameters
            self.W_v.weight.data = output_weight
    
    def type_check(self, x):
        """Ensure data is on the correct device and has the right type"""
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x.astype(np.float32)).to(self.device)
        return x.to(self.device) if x.device != self.device else x
    
    def scaled_dot_product_attention(self, query, key, value, attn_mask=None, dropout_p=0.0,
            is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias = attn_mask + attn_bias

        if enable_gqa:
            key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
            value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        return attn_weight @ value, attn_weight, value

    def forward(self, x):
        """
        Forward pass through the Sparse Transformer
        
        Args:
            x: Input tensor [batch_size, n]
            
        Returns:
            Tuple of (input, reconstruction, attention weights, value vectors)
        """
        # Update memory indices periodically during training
        if self.training and self.steps % self.memory_update_freq == 0:
            with torch.no_grad():
                # Only update after warmup and before decay
                if self.steps > self.total_steps // 20 and self.steps < self.total_steps * 0.8:
                    self.memory_indices = torch.randint(0, self.X_data.shape[0], 
                                                    (self.m,), device=self.device)
        
        self.steps += 1
        
        # Get cross attention context from memory
        X_cross = self.X_data[self.memory_indices]  # Shape: [m, n]
        
        # Preprocess data
        C = self.preprocess(X_cross)
        X_cross = X_cross / C
        
        # Type conversion for input x
        x = self.type_check(x)  # Shape: [batch_size, n]
        x = x / C  # Apply same scaling
        
        # Project to attention space
        q = self.norm_q(self.W_q(x))  # Shape: [N, a]
        k = self.norm_k(self.W_k(X_cross))  # Shape: [m, a]
        v = self.norm_v(self.W_v(X_cross))  # Shape: [m, n]
        
        # Reshape for attention (add seq_len dimension of 1)
        query = query.unsqueeze(1)  # [batch_size, 1, a]
        key = key.unsqueeze(0)  # [1, n, a]
        value = value.unsqueeze(0)  # [1, n, m]
        
        # Compute attention
        attn_output, attn_weights, values = self.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0
        )
        
        # Reshape output
        attn_output = attn_output.squeeze(1)  # [batch_size, n]
        attn_weights = attn_weights.squeeze(1)  # [batch_size, m]
        
        # Project back to input space
        reconstruction = attn_output  # [batch_size, n]
        
        # Update feature tracking during training
        if self.training:
            self.feature_tracker.update(attn_weights)
        
        return x, reconstruction, attn_weights, value.squeeze(0)

    def compute_loss(self, x: torch.Tensor, x_hat: torch.Tensor, 
                    f: torch.Tensor, v:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute loss following SAE loss pattern:
        L = (1/|X|) * Σ ||x - x̂||₂² + λ * Σᵢ |fᵢ(x)| ||vᵢ||₂
        
        Args:
            x: Input tensor
            x_hat: Reconstructed input
            f: Feature activations (attention weights)
            v: Value vectors
            
        Returns:
            Tuple of (total_loss, L2_loss, L1_penalty)
        """
        # L2 reconstruction loss
        L2_loss = torch.mean(torch.norm(x - x_hat, p=2, dim=1)**2)
        
        # Get L2 norms of value vectors
        v_norms = torch.norm(v, p=2, dim=1)
        
        # Sparsity penalty with L2 norm weighting
        L1_penalty = self.lambda_l1 * torch.mean(torch.sum(f * v_norms, dim=1))
        
        return L2_loss + L1_penalty, L2_loss, L1_penalty

    @torch.no_grad()
    def feature_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Get feature activations (attention weights) for input data"""
        x = self.type_check(x)
        _, _, f, _ = self.forward(x)
        return f

    def preprocess(self, X):
        """
        Scale dataset so that E_x[||x||₂] = √D as specified
        """
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X.astype(np.float32)).to(self.device)
        elif isinstance(X, torch.Tensor) and X.device != self.device:
            X = X.to(self.device)

        # Calculate scaling factor C so that E_x[||x||₂] = √D after dividing by C
        mean_norm = torch.mean(torch.norm(X, p=2, dim=1))
        C = mean_norm / np.sqrt(self.n)
        
        return C

    def train_and_validate(self, X_train, X_val, learning_rate=5e-5, batch_size=4096, target_steps=200_000):
        """
        Train the Sparse Transformer using the updated configuration:
        - Adam optimizer (beta1=0.9, beta2=0.999, no weight decay)
        - Learning rate ~5e-5 with linear decay in last 20% of training
        - λ warmup over first 5% of training
        - Gradient clipping to norm 1
        - 200k training steps by default
        """
        optimizer = optim.Adam(
            self.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0)

        # Preprocess data
        C = self.preprocess(X_train)
        X_train = X_train.clone() / C
        X_val = X_val.clone() / C

        # Setup data loaders
        train_dataset = TensorDataset(X_train)
        val_dataset = TensorDataset(X_val)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False)

        # Calculate required epochs to reach target steps
        steps_per_epoch = len(train_loader)
        num_epochs = max(1, target_steps // steps_per_epoch)
        actual_total_steps = num_epochs * steps_per_epoch
        self.total_steps = actual_total_steps
        
        # Set memory update frequency
        self.memory_update_freq = max(1, int(self.total_steps / 100))
        
        # Initialize training parameters
        warmup_steps = actual_total_steps // 20  # First 5% for lambda warmup
        decay_start_step = int(actual_total_steps * 0.8)  # Start decay at 80% of training
        step = 0
        best_val_loss = float('inf')
        final_lambda = self.lambda_l1

        print("\nTraining Configuration:")
        print(f"Total Steps: {actual_total_steps}")
        print(f"Epochs: {num_epochs}")
        print(f"Steps per Epoch: {steps_per_epoch}")
        print(f"Batch Size: {batch_size}")
        print(f"Warmup Steps: {warmup_steps}")
        print(f"Learning Rate Decay Start: {decay_start_step}")
        print(f"Memory Update Frequency: {self.memory_update_freq} steps")

        print("\nMetrics:")
        print("  Loss    - Training loss for current batch")
        print("  ValLoss - Average validation loss")
        print("  λ       - Current L1 regularization strength")
        print("  Dead%   - Percentage of features with no activation in 10M samples")
        print("  Sparse% - Percentage of non-zero activations")
        print(f"\n{'Step':>8} {'Epoch':>5} {'Loss':>8} {'ValLoss':>8} {'λ':>5} {'Dead%':>6} {'Sparse%':>7}")

        for epoch in range(num_epochs):
            self.train()
            epoch_train_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                x = batch[0]

                # Forward pass
                x, x_hat, f_x, v = self.forward(x)

                # Update feature tracking
                dead_ratio, stats = self.feature_tracker.update(f_x)

                # Lambda warmup - linear increase from 0 to final_lambda over first 5% of steps
                if step < warmup_steps:
                    self.lambda_l1 = (step / warmup_steps) * final_lambda
                else:
                    self.lambda_l1 = final_lambda

                # Learning rate decay - linear decay to zero over last 20% of steps
                if step >= decay_start_step:
                    progress = (step - decay_start_step) / (actual_total_steps - decay_start_step)
                    new_lr = learning_rate * (1 - progress)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr

                # Compute loss and update
                total_loss, L2_loss, L1_loss = self.compute_loss(x, x_hat, f_x, v)
                total_loss.backward()

                # Gradient clipping to norm 1
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_train_loss += total_loss.item()
                num_batches += 1
                step += 1

                # Periodic validation and logging
                if batch_idx % (len(train_loader) // 5) == 0:
                    self.eval()
                    val_loss = 0.0
                    val_batches = 0

                    with torch.no_grad():
                        for val_batch in val_loader:
                            x_val = val_batch[0]
                            x_val, x_hat_val, f_x_val, v_val = self.forward(x_val)
                            val_total_loss, _, _ = self.compute_loss(
                                x_val, x_hat_val, f_x_val, v_val)
                            val_loss += val_total_loss.item()
                            val_batches += 1

                    avg_val_loss = val_loss / val_batches

                    # Calculate sparsity
                    sparsity = (f_x.abs() >= self.activation_threshold).float().mean().item()
                    
                    # Note: We intentionally do NOT save the model with lowest validation loss
                    # because the lowest loss would occur early when L1 penalty is minimal
                    
                    print(f"{step:8d} {epoch:5d} {total_loss.item():8.4f} {avg_val_loss:8.4f} "
                          f"{self.lambda_l1:5.2f} {dead_ratio*100:6.2f}% {sparsity*100:7.2f}%")
                    
                    # Save periodic checkpoints if desired
                    if step % 50000 == 0:
                        checkpoint_path = f"{self.st_model_path}.step{step}"
                        torch.save(self.state_dict(), checkpoint_path)
                        print(f"Saved checkpoint at step {step} to {checkpoint_path}")

                    self.train()

        print(f"\nTraining completed:")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Final dead feature ratio: {dead_ratio:.1%}")
        print(f"Steps completed: {step}/{actual_total_steps}")
        print(f"Final λ: {self.lambda_l1:.2f}")
        
        # Load best model
        self.load_state_dict(torch.load(self.st_model_path))
        return self