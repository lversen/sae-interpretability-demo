import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional
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
                 activation_threshold: float = 1e-4):
        super().__init__()
        
        # Core attributes
        self.n, self.m, self.a = n, m, a
        self.st_model_path = st_model_path
        self.lambda_l1 = lambda_l1
        self.device = device
        self.X = self.type_check(X)
        self.activation_threshold = activation_threshold
        self.memory_update_freq = 1
        self.steps = 0
        self.total_steps = 0
        # Projections
        self.W_q = nn.Linear(n, a)
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
        
        # Initialize memory indices
        self.register_buffer('memory_indices', 
                           torch.randint(0, self.X.shape[0], (m,), device=device))
        
        self.to(device)
    
    def type_check(self, x):
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
        # Update memory indices periodically during training
        if self.training and self.steps > self.total_steps // 20 and self.steps % (self.total_steps // 100) == 0:
            with torch.no_grad():
                self.memory_indices = torch.randint(0, self.X.shape[0], 
                                                    (self.m,), device=self.device)
                print("X_CROSS UPDATED | X_CROSS UPDATED | X_CROSS UPDATED | X_CROSS UPDATED | X_CROSS UPDATED | X_CROSS UPDATED | X_CROSS UPDATED | X_CROSS UPDATED")
        self.steps += 1
        
        # Get cross attention context
        X_cross = self.X[self.memory_indices]  # Shape: [m, n]
        C = self.preprocess(X_cross)
        X_cross /= C
        
        # Type conversion for input x
        x = self.type_check(x)  # Shape: [N, n]
        
        # Project to attention space
        q = self.norm_q(self.W_q(x))  # Shape: [N, a]
        k = self.norm_k(self.W_k(X_cross))  # Shape: [m, a]
        v = self.norm_v(self.W_v(X_cross))  # Shape: [m, n]
        
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

        # Preprocess data
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
        self.memory_update_freq = int(actual_total_steps/100)
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
                if num_batches % (len(train_loader) // 5) == 0:
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
# =============================================================================
#     def train_and_validate(self, X_train, X_val, learning_rate: float = 1e-3,
#                           batch_size: int = 4096, target_steps: int = 1000,
#                           grad_clip: float = 1.0, log_freq: Optional[int] = 1):
#         # Setup
#         # Initialize GradScaler for mixed precision training
#         scaler = torch.amp.GradScaler()
#         optimizer = optim.Adam(self.parameters(), lr=learning_rate, betas=(0.9, 0.999))
#         steps_per_epoch = len(X_train) // batch_size
#         num_epochs = max(1, target_steps // steps_per_epoch)
#         total_steps = num_epochs * steps_per_epoch
#         warmup_steps = total_steps // 20
#         decay_start_step = int(total_steps * 0.8)
#
#         # AMP setup
#         amp_dtype = torch.float16 if torch.cuda.is_available() else torch.bfloat16
#         amp_device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
#         autocast_context = torch.amp.autocast(device_type=amp_device_type, dtype=amp_dtype, cache_enabled=True)
#
#         # Data setup - ensure data is on CPU before creating DataLoader
#         X_train = self.preprocess(X_train, normalize=True)
#         X_val = self.preprocess(X_val, normalize=True)
#
#         # Move data to CPU for DataLoader
#         if X_train.device != torch.device('cpu'):
#             X_train = X_train.cpu()
#         if X_val.device != torch.device('cpu'):
#             X_val = X_val.cpu()
#
#         train_loader = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=True, pin_memory=True)
#         val_loader = DataLoader(TensorDataset(X_val), batch_size=batch_size, pin_memory=True)
#
#         log_freq = log_freq or steps_per_epoch // 5
#         best_val_loss = float('inf')
#         step = last_eval_step = 0
#         l1 = self.lambda_l1
#         times_per_step = []
#         start_time = time.time()
#
#         print(f"\nTraining: {total_steps} steps, {warmup_steps} warmup, batch {batch_size}, "
#               f"lr {learning_rate}, heads {self.num_heads}, clip {grad_clip}")
#
#         for epoch in range(num_epochs):
#             self.current_epoch = epoch  # Add this line
#             self.train()
#             for batch in train_loader:
#                 step_start = time.time()
#                 optimizer.zero_grad(set_to_none=True)
#
#                 # Learning rate scheduling
#                 self.lambda_l1 = l1 * min(step / warmup_steps, 1.0) if step < warmup_steps else l1
#
#                 # Forward pass
#                 batch = batch[0].to(self.device)
#                 with autocast_context:
#                     outputs = self.forward(batch)
#                     total_loss, L2_loss, L1_loss = self.compute_losses(*outputs)
#
#                 # Backward pass
#                 scaler.scale(total_loss).backward()
#                 scaler.unscale_(optimizer)
#                 torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
#                 scaler.step(optimizer)
#                 scaler.update()
#
#                 # Timing and cleanup
#                 times_per_step.append(time.time() - step_start)
#                 if len(times_per_step) > 3:
#                     times_per_step.pop(0)
#
#                 if step % 1000 == 0:
#                     torch.cuda.empty_cache()
#
#                 # Logging and validation
#                 if step % log_freq == 0:
#                     self.eval()
#                     with torch.no_grad():
#                         val_loss = sum(self.compute_losses(*self.forward(val_batch[0].to(self.device)))[0]
#                                      for val_batch in val_loader)
#                     val_loss /= len(val_loader)
#
#                     # Metrics
#                     sparsity = (outputs[2] <= self.activation_threshold).float().mean().item()
#                     gini = self.gini_coefficient(outputs[2])
#                     remaining_time = str(timedelta(seconds=int(np.mean(times_per_step) * (total_steps - step))))
#
#                     print(f"Step: {step:6d}/{total_steps} ({step/total_steps:3.1%}) | "
#                           f"Epoch: {epoch:3d} | LR: {optimizer.param_groups[0]['lr']:.2e} | "
#                           f"Train: {total_loss.item():8.4f} | Val: {val_loss:8.4f} | "
#                           f"L1_loss: {L1_loss:4.2f} | L2_loss: {L2_loss:4.2f} | "
#                           f"L1 λ: {self.lambda_l1:4.2f} | Sparse: {sparsity:5.1%} | "
#                           f"Gini: {gini:.3f} | ETA: {remaining_time}")
#
#                     if val_loss < best_val_loss:
#                         best_val_loss = val_loss
#                         torch.save(self.state_dict(), self.st_model_path)
#
#                     self.train()
#
#                 step += 1
#
#         print(f"\nTraining completed in {str(timedelta(seconds=int(time.time() - start_time)))}")
#         print(f"Best validation loss: {best_val_loss:.4f}")
#         print(f"Final learning rate: {optimizer.param_groups[0]['lr']:.2e}")
# =============================================================================
