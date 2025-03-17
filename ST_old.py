import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional, Callable
from deadfeatures import DeadFeatureTracker
import time
import math
from datetime import timedelta
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F


class SparseTransformer(nn.Module):
    def __init__(self, X, n: int, m: int, a: int, st_model_path: str,
                 lambda_l1: float = 5.0, num_heads: int = 1, device: str = 'cuda',
                 window_size: int = 10_000_000, update_interval: int = 1_000,
                 activation_threshold: float = 1e-3, use_direct_kv: bool = False,
                 activation: str = 'relu', attention_fn: str = 'softmax'):
        super().__init__()
        
        # Model parameters
        self.n, self.m, self.a = n, m, a
        self.st_model_path = st_model_path
        self.lambda_l1 = lambda_l1
        self.device = device
        self.X = self.type_check(X)
        self.activation_threshold = activation_threshold
        self.memory_update_freq = 1
        self.steps = 0
        self.total_steps = 0
        self.use_direct_kv = use_direct_kv
        
        # Set activation function
        self.activation_name = activation
        self.activation = self._get_activation_function(activation)
        
        # Set attention function
        self.attention_fn_name = attention_fn
        self.attention_fn = self._get_attention_function(attention_fn)
        
        print(f"Using activation function: {self.activation_name}")
        print(f"Using attention function: {self.attention_fn_name}")
        
        # Projections - query projection always needed
        self.W_q = nn.Linear(n, a)
        
        if self.use_direct_kv:
            # Direct K-V approach with parameter matrices
            self.W_k_direct = nn.Parameter(torch.Tensor(self.m, self.a))
            self.W_v_direct = nn.Parameter(torch.Tensor(self.m, self.n))
            # Initialize parameters
            nn.init.normal_(self.W_k_direct, mean=0.0, std=0.02)
            nn.init.kaiming_uniform_(self.W_v_direct, a=math.sqrt(5))
        else:
            # Original memory bank approach
            self.W_k = nn.Linear(n, a)
            self.W_v = nn.Linear(n, n)
        
        # Normalization layers
        self.norm_q = nn.LayerNorm(a)
        self.norm_k = nn.LayerNorm(a)
        self.norm_v = nn.LayerNorm(n)

        # Feature tracking
        self.feature_tracker = DeadFeatureTracker(
            num_features=self.m,
            activation_threshold=activation_threshold,
            window_size=window_size,
            update_interval=update_interval
        )
        
        # Initialize memory indices (only needed for original approach)
        if not self.use_direct_kv:
            self.register_buffer('memory_indices', 
                              torch.randint(0, self.X.shape[0], (m,), device=device))
        else:
            # Create a dummy memory_indices buffer for compatibility
            self.register_buffer('memory_indices', 
                              torch.zeros(m, dtype=torch.long, device=device))
        
        self.to(device)
    
    def _get_activation_function(self, activation_name: str) -> Callable:
        """
        Returns the activation function based on the name.
        """
        activation_functions = {
            'relu': torch.relu,
            'leaky_relu': lambda x: torch.nn.functional.leaky_relu(x, negative_slope=0.1),
            'gelu': torch.nn.functional.gelu,
            'sigmoid': torch.sigmoid,
            'tanh': torch.tanh,
            'none': lambda x: x,  # Identity function (no activation)
            'softplus': torch.nn.functional.softplus,
            'elu': torch.nn.functional.elu,
            'selu': torch.nn.functional.selu,
        }
        
        if activation_name.lower() not in activation_functions:
            print(f"Warning: Activation function '{activation_name}' not supported. "
                  f"Falling back to 'relu'. Supported activations: {list(activation_functions.keys())}")
            return activation_functions['relu']
        
        return activation_functions[activation_name.lower()]
    
    def _get_attention_function(self, attention_name: str) -> Callable:
        """
        Returns the attention function based on the name.
        
        This function determines how attention scores are processed.
        """
        def sparsemax(scores, dim=-1):
            """Sparsemax function - a sparse alternative to softmax"""
            # Implementation of sparsemax
            zeroes = torch.zeros_like(scores)
            scores = torch.where(scores < -1e10, zeroes, scores)  # Replace large negative values
            sorted_scores, _ = torch.sort(scores, descending=True, dim=dim)
            cumsum = torch.cumsum(sorted_scores, dim=dim)
            range_idx = torch.arange(1, scores.size(dim) + 1, device=scores.device)
            range_idx = range_idx.view([-1] + [1] * (scores.dim() - 1))
            range_idx = range_idx.transpose(0, dim)
            threshold = (cumsum - 1) / range_idx
            mask = sorted_scores > threshold
            mask_sum = mask.sum(dim=dim, keepdim=True).clamp(min=1)
            mask_threshold = torch.gather(threshold, dim, mask_sum - 1)
            return torch.clamp(scores - mask_threshold, min=0)
        
        def normalized_activation(scores, dim=-1):
            """Apply activation then normalize to ensure weights sum to 1"""
            scores = self.activation(scores)  # Apply the model's activation function
            # Avoid division by zero
            sum_scores = scores.sum(dim=dim, keepdim=True).clamp(min=1e-6)
            return scores / sum_scores
        
        def direct_activation(scores, dim=-1):
            """Apply activation directly without normalization"""
            return self.activation(scores)  # Apply the model's activation function
        
        def relu_softmax(scores, dim=-1):
            """Apply ReLU first, then softmax - handles negative values differently"""
            scores = F.relu(scores)
            return F.softmax(scores, dim=dim)
        
        def custom_softmax(scores, dim=-1, beta=1.0):
            """Softmax with temperature parameter"""
            return F.softmax(scores * beta, dim=dim)
        
        attention_functions = {
            'softmax': lambda x, dim=-1: F.softmax(x, dim=dim),
            'sparsemax': sparsemax,
            'normalized_activation': normalized_activation,
            'direct_activation': direct_activation,
            'relu_softmax': relu_softmax,
            'softmax_hard': lambda x, dim=-1: custom_softmax(x, dim=dim, beta=2.0),
            'softmax_soft': lambda x, dim=-1: custom_softmax(x, dim=dim, beta=0.5),
        }
        
        if attention_name.lower() not in attention_functions:
            print(f"Warning: Attention function '{attention_name}' not supported. "
                  f"Falling back to 'softmax'. Supported functions: {list(attention_functions.keys())}")
            return attention_functions['softmax']
        
        return attention_functions[attention_name.lower()]
    
    def type_check(self, x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x.astype(np.float32)).to(self.device)
        return x.to(self.device) if x.device != self.device else x
    
    def scaled_dot_product_attention(self, query, key, value, attn_mask=None, dropout_p=0.0,
            is_causal=False, scale=None, enable_gqa=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Customized scaled dot-product attention with replaceable attention function.
        """
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        
        # Compute attention scores
        attn_scores = query @ key.transpose(-2, -1) * scale_factor
        
        # Apply masks if needed
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_scores += attn_bias

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_scores += attn_mask

        if enable_gqa:
            key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
            value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

        # Apply custom attention function instead of hardcoded softmax
        attn_weight = self.attention_fn(attn_scores, dim=-1)
        
        # Apply dropout if training
        if dropout_p > 0.0 and self.training:
            attn_weight = F.dropout(attn_weight, p=dropout_p)
            
        # Compute output
        output = attn_weight @ value
        
        return output, attn_weight, value

    def forward(self, x):
        # Update memory indices periodically during training (only for memory bank approach)
        if not self.use_direct_kv and self.training and self.steps > self.total_steps // 20 and self.steps < self.total_steps * 0.8 and self.steps % self.memory_update_freq == 0:
            with torch.no_grad():
                self.memory_indices = torch.randint(0, self.X.shape[0], 
                                                   (self.m,), device=self.device)
                print("X_CROSS UPDATED | X_CROSS UPDATED | X_CROSS UPDATED | X_CROSS UPDATED | X_CROSS UPDATED | X_CROSS UPDATED | X_CROSS UPDATED | X_CROSS UPDATED")
        self.steps += 1
        
        # Type conversion for input x
        x = self.type_check(x)  # Shape: [N, n]
        
        # Project to query space with activation function
        q_pre = self.W_q(x)
        q_act = self.activation(q_pre)  # Apply activation function
        q = self.norm_q(q_act)  # Shape: [N, a]
        
        if self.use_direct_kv:
            # DIRECT K-V APPROACH
            # Apply activation to the raw parameter matrices
            k_act = self.activation(self.W_k_direct)  # Apply activation function
            k = self.norm_k(k_act)  # Shape: [m, a]
            
            v_act = self.activation(self.W_v_direct)  # Apply activation function
            v = self.norm_v(v_act)  # Shape: [m, n]
        else:
            # ORIGINAL MEMORY BANK APPROACH
            # Get cross attention context
            X_cross = self.X[self.memory_indices]  # Shape: [m, n]
            C = self.preprocess(X_cross)
            X_cross /= C
            
            # Project to attention space with activation functions
            k_pre = self.W_k(X_cross)
            k_act = self.activation(k_pre)  # Apply activation function
            k = self.norm_k(k_act)  # Shape: [m, a]
            
            v_pre = self.W_v(X_cross)
            v_act = self.activation(v_pre)  # Apply activation function
            v = self.norm_v(v_act)  # Shape: [m, n]
        
        # Use custom attention function instead of hardcoded softmax
        x_hat, f, V = self.scaled_dot_product_attention(q, k, v, dropout_p=0)
        
        if self.training:
            self.feature_tracker.update(f)
        
        return x, x_hat, f, V

    def compute_loss(self, x: torch.Tensor, x_hat: torch.Tensor, 
                    f: torch.Tensor, v:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute loss according to the paper's specification:
        L = (1/|X|) * Σ ||x - x̂||₂² + λ * Σᵢ |fᵢ(x)| ||Wdᵢ||₂
        
        Args:
            x: Input tensor of shape [batch_size, n] where n is input dimension
            x_hat: Reconstructed input of shape [batch_size, n]
            f_x: Feature activations of shape [batch_size, m] where m is number of features
        """
        # L2 reconstruction term: (1/|X|) * Σ ||x - x̂||₂²
        # First compute L2 norm of difference vectors, then square, then average
        L2_loss = torch.mean(torch.norm(x - x_hat, p=2, dim=1)**2)
        
        # Get L2 norms of decoder weight columns: shape [m]
        v_norms = torch.norm(v, p=2, dim=1)
        # Sparsity penalty: sum over features (m), average over batch
        L1_penalty = self.lambda_l1 * torch.mean(torch.sum(f * v_norms, dim=1))
        
        return L2_loss + L1_penalty, L2_loss, L1_penalty

    @torch.no_grad()
    def feature_activations(self, x: torch.Tensor) -> torch.Tensor:
        x = self.type_check(x)
        _, _, f, _ = self.forward(x)
        return f

    def preprocess(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X.astype(np.float32)).to(self.device)
        elif isinstance(X, torch.Tensor) and X.device != self.device:
            X = X.to(self.device)

        C = torch.mean(torch.norm(X, p=2, dim=1)) / np.sqrt(self.n)
        return(C)
        
    def gini_coefficient(self, tensor: torch.Tensor) -> float:
        sorted_vals, _ = torch.sort(tensor.flatten().abs())
        n = sorted_vals.shape[0]
        idx = torch.arange(1, n+1, device=tensor.device)
        return (torch.sum(sorted_vals * idx) / (n * torch.sum(sorted_vals)) - (n + 1) / (2 * n)).item()

    def train_and_validate(self, X_train, X_val, learning_rate=5e-5, batch_size=4096, target_steps=200_000):
        """
        Train the Sparse Autoencoder targeting a specific number of steps while tracking dead features.

        Args:
            X_train: Training data
            X_val: Validation data
            learning_rate: Initial learning rate
            batch_size: Batch size for training
            target_steps: Target number of training steps (default 200k as per paper)
        """
        optimizer = optim.Adam(
            self.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0)



        C = self.preprocess(X_train)
        X_train /= C
        X_val /= C

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
        self.memory_update_freq = int(self.total_steps/100)
        # Initialize training parameters
        warmup_steps = actual_total_steps // 20  # First 5% for lambda warmup
        # Start decay at 80% of training
        decay_start_step = int(actual_total_steps * 0.8)
        step = 0
        best_val_loss = float('inf')
        final_lambda = self.lambda_l1

        # Initialize feature tracker if not already done
        if not hasattr(self, 'feature_tracker'):
            self.feature_tracker = DeadFeatureTracker(
                num_features=self.m,
                window_size=10_000_000,  # As specified in paper
                update_interval=10_000
            )

        print("\nTraining Configuration:")
        print(f"Total Steps: {actual_total_steps}")
        print(f"Epochs: {num_epochs}")
        print(f"Steps per Epoch: {steps_per_epoch}")
        print(f"Batch Size: {batch_size}")
        print(f"Warmup Steps: {warmup_steps}")
        print(f"Learning Rate Decay Start: {decay_start_step}")
        print(f"Using {'direct K-V matrices' if self.use_direct_kv else 'memory bank approach'}")

        print("\nMetrics:")
        print("  Loss    - Training loss for current batch")
        print("  ValLoss - Average validation loss")
        print("  λ       - Current L1 regularization strength")
        print("  Dead%   - Percentage of features with no activation in 10M samples")
        print("  Sparse% - Percentage of non-zero activations")
        print("  Track%  - Percentage of 10M sample tracking window completed")
        print(f"\n{'Step':>8} {'Epoch':>5} {'Loss':>8} {'ValLoss':>8} {
              'λ':>5} {'Dead%':>6} {'Sparse%':>7} {'Track%':>7}")

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

                # Lambda warmup
                if step < warmup_steps:
                    self.lambda_l1 = (step / warmup_steps) * final_lambda
                else:
                    self.lambda_l1 = final_lambda

                # Learning rate decay in last 20%
                if step >= decay_start_step:
                    progress = (step - decay_start_step) / \
                        (actual_total_steps - decay_start_step)
                    # Linear decay to zero
                    new_lr = learning_rate * (1 - progress)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr

                # Compute loss and update
                total_loss, L2_loss, L1_loss = self.compute_loss(x, x_hat, f_x, v)
                total_loss.backward()

                # Gradient clipping as per paper
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_train_loss += total_loss.item()
                num_batches += 1
                step += 1

                # Periodic validation and logging
                if num_batches % (len(train_loader) // 1) == 0:
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

                    # Calculate sparsity and tracking progress
                    sparsity = (
                        f_x.abs() >= self.feature_tracker.activation_threshold).float().mean().item()
                    tracking_progress = min(
                        self.feature_tracker.samples_seen / self.feature_tracker.window_size, 1.0)

                    print(f"Step: {step:6d}/{actual_total_steps} ({step/actual_total_steps:3.1%}) | "
                          f"Epoch: {epoch:3d} | LR: {
                              optimizer.param_groups[0]['lr']:.2e} | "
                          f"Train: {total_loss.item():8.4f} | Val: {
                        val_loss:8.4f} | "
                        f"L1_loss: {L1_loss:4.2f} | L2_loss: {
                              L2_loss:4.2f} | "
                        f"L1 λ: {self.lambda_l1:4.2f} | Sparse: {sparsity:5.1%} | ")


                    self.train()

        print(f"\nTraining completed:")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Final dead feature ratio: {dead_ratio:.1%}")
        print(f"Steps completed: {step}/{actual_total_steps}")
        print(f"Final λ: {self.lambda_l1:.2f}")
        torch.save(self.state_dict(), self.st_model_path)