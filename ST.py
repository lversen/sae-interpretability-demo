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
import gc
from datetime import timedelta
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, List, Any, Union, Optional


class SparseTransformer(nn.Module):
    def __init__(self, X, n: int, m: int, a: int, st_model_path: str,
                 lambda_l1: float = 5.0, num_heads: int = 1, device: str = 'cuda',
                 window_size: int = 10_000_000, update_interval: int = 1_000,
                 activation_threshold: float = 1e-3, use_mixed_precision: bool = False):
        """
        Initialize the Sparse Transformer model.
        
        Args:
            X: Input data tensor or array
            n: Input dimension
            m: Feature dimension (number of memory vectors)
            a: Attention dimension
            st_model_path: Path to save model checkpoints
            lambda_l1: L1 regularization strength
            num_heads: Number of attention heads (currently only 1 fully supported)
            device: Device to use ('cuda' or 'cpu')
            window_size: Window size for tracking dead features
            update_interval: Update interval for feature tracking
            activation_threshold: Threshold for considering a feature activated
            use_mixed_precision: Whether to use mixed precision training
        """
        super().__init__()
        
        # Model parameters
        self.n, self.m, self.a = n, m, a
        self.st_model_path = st_model_path
        self.lambda_l1 = lambda_l1
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.X = self.type_check(X)
        self.activation_threshold = activation_threshold
        self.memory_update_freq = 1
        self.steps = 0
        self.total_steps = 0
        self.use_mixed_precision = use_mixed_precision
        self.training_history = {"steps": [], "train_loss": [], "val_loss": [], 
                                "l1_loss": [], "l2_loss": [], "lambda": [], 
                                "dead_ratio": [], "sparsity": []}
        
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
                           torch.randint(0, self.X.shape[0], (m,), device=self.device))
        self.memory_indices_for_update = None  # For custom memory index update strategy
        
        # Initialize weights
        self.initialize_weights()
        
        self.to(self.device)
    
    def type_check(self, x):
        """Ensure input tensors are on the correct device and have the right type"""
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x.astype(np.float32)).to(self.device)
        return x.to(self.device) if x.device != self.device else x
    
    def scaled_dot_product_attention(self, query, key, value, attn_mask=None, dropout_p=0.0,
                is_causal=False, scale=None, enable_gqa=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Efficient scaled dot-product attention implementation
            
            This is an optimized version for the ST model's specific usage pattern:
            - Uses direct batch matrix multiplication for better performance
            - Handles typical ST model dimensions efficiently
            - Minimizes memory allocations for faster calculation
            
            Args:
                query: Query tensor of shape [N, a] where N is batch size and a is attention dimension
                key: Key tensor of shape [m, a] where m is feature dimension and a is attention dimension
                value: Value tensor of shape [m, n] where m is feature dimension and n is input dimension
                attn_mask: Optional mask tensor
                dropout_p: Dropout probability (applied to attention weights)
                is_causal: Whether to use causal (autoregressive) attention
                scale: Optional custom scale factor for attention scores
                enable_gqa: Whether to enable grouped query attention
                
            Returns:
                Tuple containing:
                - output: Attention output tensor of shape [N, n]
                - attn_weight: Attention weight matrix of shape [N, m]
                - value: Value tensor of shape [m, n]
            """
            # Pre-calculate scale factor once
            scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
            
            # Fast path for the common case in ST (no masks, no gqa)
            if attn_mask is None and not is_causal and not enable_gqa:
                # Compute attention weights with optimized matrix multiplication
                attn_weight = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
                
                # Apply softmax directly (most common case)
                attn_weight = F.softmax(attn_weight, dim=-1)
                
                # Apply dropout only during training
                if dropout_p > 0.0 and self.training:
                    attn_weight = F.dropout(attn_weight, p=dropout_p)
                
                # Compute output with optimized matrix multiplication
                output = torch.matmul(attn_weight, value)
                
                return output, attn_weight, value
            
            # Fallback path for less common cases (masks, causal attention, or GQA)
            L, S = query.size(-2), key.size(-2)
            
            # Compute attention weights
            attn_weight = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
            
            # Apply causal mask if needed (for autoregressive models)
            if is_causal:
                mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
                attn_weight.masked_fill_(~mask, float("-inf"))
            
            # Apply attention mask if provided (for controlling attention patterns)
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    attn_weight.masked_fill_(~attn_mask, float("-inf"))
                else:
                    attn_weight += attn_mask
                    
            # Handle generalized query attention (for efficient multi-head attention)
            if enable_gqa:
                key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
                value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)
            
            # Apply softmax and dropout
            attn_weight = F.softmax(attn_weight, dim=-1)
            if dropout_p > 0.0 and self.training:
                attn_weight = F.dropout(attn_weight, p=dropout_p)
            
            # Compute output
            output = torch.matmul(attn_weight, value)
            
            return output, attn_weight, value

    def forward(self, x):
        # Update memory indices periodically during training
        if self.training and self.steps > self.total_steps // 20 and self.steps < self.total_steps * 0.8 and self.steps % self.memory_update_freq == 0:
            with torch.no_grad():
                # Use custom indices if available, otherwise use random selection
                if self.memory_indices_for_update is not None:
                    self.memory_indices = self.memory_indices_for_update
                    self.memory_indices_for_update = None  # Clear for next update
                else:
                    self.memory_indices = torch.randint(0, self.X.shape[0], 
                                                      (self.m,), device=self.device)
        self.steps += 1
        
        # Get cross attention context
        X_cross = self.X[self.memory_indices]  # Shape: [m, n]
        C = self.preprocess(X_cross)
        X_cross = X_cross / C  # Use new tensor to avoid modifying self.X
        
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
        L = (1/|X|) * Σ ||x - x̂||₂² + λ * Σᵢ |fᵢ(x)| ||Vᵢ||₂
        
        Args:
            x: Input tensor
            x_hat: Reconstructed tensor
            f: Feature activations
            v: Value vectors
            
        Returns:
            Tuple containing:
            - total_loss: Combined loss (L2 + L1 penalty)
            - L2_loss: Reconstruction loss component
            - L1_penalty: Sparsity penalty component
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
        """
        Return the feature activations for a given input
        
        Args:
            x: Input tensor of shape [N, n]
            
        Returns:
            Feature activations of shape [N, m]
        """
        x = self.type_check(x)
        _, _, f, _ = self.forward(x)
        return f

    def preprocess(self, X):
        """
        Preprocess input data to have consistent scale
        
        Args:
            X: Input tensor
            
        Returns:
            Scaling factor C such that E[||x||₂] = √n after dividing by C
        """
        X = self.type_check(X)
        # Calculate scaling factor C so that E_x[||x||₂] = √n after dividing by C
        mean_norm = torch.mean(torch.norm(X, p=2, dim=1))
        C = mean_norm / np.sqrt(self.n)
        return C
        
    def initialize_weights(self):
        """
        Initialize weights with a reasonable scheme for ST model.
        
        For ST, we initialize:
        - Query and Key projections with normal distribution scaled by input dimension
        - Value projection with a scheme similar to the attention heads in transformers
        """
        with torch.no_grad():
            # Initialize query and key projections
            nn.init.normal_(self.W_q.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.W_k.weight, mean=0.0, std=0.02)
            
            # For value projection, use Kaiming initialization
            nn.init.kaiming_uniform_(self.W_v.weight, a=math.sqrt(5))
            
            # Initialize biases to zero if they exist
            if hasattr(self.W_q, 'bias') and self.W_q.bias is not None:
                nn.init.zeros_(self.W_q.bias)
            if hasattr(self.W_k, 'bias') and self.W_k.bias is not None:
                nn.init.zeros_(self.W_k.bias)
            if hasattr(self.W_v, 'bias') and self.W_v.bias is not None:
                nn.init.zeros_(self.W_v.bias)
    
    def compute_feature_statistics(self, data_loader, max_batches=10):
        """
        Compute statistics about feature activations on a dataset.
        
        Args:
            data_loader: DataLoader for the dataset
            max_batches: Maximum number of batches to process
            
        Returns:
            dict: Dictionary of feature statistics including means, maximums,
                 activation frequencies, and sparsity metrics
        """
        self.eval()
        activations_list = []
        
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= max_batches:
                    break
                    
                x = batch[0].to(self.device)
                _, _, f_x, _ = self.forward(x)
                activations_list.append(f_x.cpu())
        
        if not activations_list:
            return None
            
        # Concatenate all batches
        all_activations = torch.cat(activations_list, dim=0)
        
        # Compute basic statistics
        mean_activations = torch.mean(all_activations, dim=0)
        max_activations = torch.max(all_activations, dim=0)[0]
        activation_frequency = (all_activations > self.activation_threshold).float().mean(dim=0)
        
        # Find top active features
        avg_feature_activity = activation_frequency.numpy()
        top_indices = np.argsort(-avg_feature_activity)[:10]
        
        # Compute coactivation matrix for top features
        top_activations = all_activations[:, top_indices]
        binary_activations = (top_activations > self.activation_threshold).float()
        coactivation = torch.mm(binary_activations.t(), binary_activations) / binary_activations.shape[0]
        
        # Calculate sparsity
        sparsity = 1.0 - (all_activations > self.activation_threshold).float().mean().item()
        
        return {
            'mean_activations': mean_activations.numpy(),
            'max_activations': max_activations.numpy(),
            'activation_frequency': avg_feature_activity,
            'top_active_features': top_indices,
            'coactivation_matrix': coactivation.numpy(),
            'sparsity': sparsity,
            'total_samples': all_activations.shape[0]
        }
        
    def plot_training_history(self, save_path: Optional[str] = None) -> Figure:
        """
        Plot the training history metrics.
        
        Args:
            save_path: Optional path to save the plot to
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        if len(self.training_history["steps"]) == 0:
            raise ValueError("No training history available to plot")
            
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot loss curves
        ax = axs[0, 0]
        ax.plot(self.training_history["steps"], self.training_history["train_loss"], label="Train Loss")
        ax.plot(self.training_history["steps"], self.training_history["val_loss"], label="Val Loss")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Loss")
        ax.set_title("Loss Curves")
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Plot L1 and L2 components
        ax = axs[0, 1]
        ax.plot(self.training_history["steps"], self.training_history["l1_loss"], label="L1 Loss")
        ax.plot(self.training_history["steps"], self.training_history["l2_loss"], label="L2 Loss")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Loss Component")
        ax.set_title("Loss Components")
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Plot lambda
        ax = axs[1, 0]
        ax.plot(self.training_history["steps"], self.training_history["lambda"])
        ax.set_xlabel("Steps")
        ax.set_ylabel("Lambda (λ)")
        ax.set_title("L1 Regularization Strength")
        ax.grid(alpha=0.3)
        
        # Plot sparsity and dead ratio
        ax = axs[1, 1]
        ax.plot(self.training_history["steps"], [s*100 for s in self.training_history["sparsity"]], 
                label="Sparsity %")
        ax.plot(self.training_history["steps"], [d*100 for d in self.training_history["dead_ratio"]], 
                label="Dead Features %")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Percentage")
        ax.set_title("Sparsity Metrics")
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        fig.suptitle(f"Training History - Sparse Transformer", fontsize=16)
        plt.subplots_adjust(top=0.93)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Training history plot saved to {save_path}")
            
        return fig

    def train_and_validate(self, X_train, X_val, learning_rate=5e-5, batch_size=4096, 
                          target_steps=200_000, checkpoint_freq=200000, save_best=False, 
                          enable_checkpoints=False, save_full_state=False, resume_from=None,
                          grad_accum_steps=1, eval_freq=None):
        """
        Train the Sparse Transformer targeting a specific number of steps while tracking dead features.
        
        Args:
            X_train: Training data tensor [samples, features]
            X_val: Validation data tensor [samples, features]
            learning_rate: Initial learning rate (default: 5e-5)
            batch_size: Batch size for training (default: 4096)
            target_steps: Target number of training steps (default: 200,000)
            checkpoint_freq: How often to save checkpoints (default: 200000 steps)
            save_best: Whether to save the best model based on validation loss (default: False)
            enable_checkpoints: Whether to save periodic checkpoints at all (default: False)
            save_full_state: Whether to save optimizer state and extra data in checkpoints (default: False)
            resume_from: Optional checkpoint path to resume training from
            grad_accum_steps: Number of gradient accumulation steps (default: 1)
            eval_freq: How often to evaluate on validation set (if None, use 5 times per epoch)
        
        Returns:
            self: The trained model
        """
        # Initialize optimizer
        optimizer = optim.Adam(
            self.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0)
        
        # Initialize mixed precision scaler if requested
        scaler = GradScaler() if self.use_mixed_precision else None
        
        # Preprocess data - scale so E_x[||x||₂] = √n
        C = self.preprocess(X_train)
        X_train = X_train / C  # X_train is already a copy from the DataLoader
        X_val = X_val / C
        
        # Track training start time
        start_time = time.time()

        # Setup data loaders with optimizations
        train_dataset = TensorDataset(X_train)
        val_dataset = TensorDataset(X_val)
        
        # Check if tensors are on CPU before using pin_memory
        # Only use pin_memory for CPU tensors when using CUDA
        pin_memory = False
        if torch.cuda.is_available() and X_train.device.type == 'cpu':
            pin_memory = True
            
        # Configure number of workers based on system
        # Use 0 workers as default for safety, can be adjusted based on hardware
        num_workers = 0  
            
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            pin_memory=pin_memory, num_workers=num_workers)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            pin_memory=pin_memory, num_workers=num_workers)

        # Calculate required epochs to reach target steps
        steps_per_epoch = len(train_loader)
        num_epochs = max(1, target_steps // steps_per_epoch)
        actual_total_steps = num_epochs * steps_per_epoch
        self.total_steps = actual_total_steps
        self.memory_update_freq = max(1, int(self.total_steps/100))  # Ensure it's at least 1
        
        # Set evaluation frequency
        if eval_freq is None:
            eval_freq = max(1, steps_per_epoch // 5)  # 5 times per epoch
        
        # Initialize training parameters
        warmup_steps = actual_total_steps // 20  # First 5% for lambda warmup
        decay_start_step = int(actual_total_steps * 0.8)  # Start decay at 80% of training
        step = 0
        best_val_loss = float('inf')
        final_lambda = self.lambda_l1
        
        # Resume from checkpoint if provided
        if resume_from and os.path.exists(resume_from):
            try:
                print(f"Resuming training from checkpoint: {resume_from}")
                checkpoint = torch.load(resume_from, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    self.load_state_dict(checkpoint['model_state_dict'])
                    if 'optimizer_state_dict' in checkpoint:
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    if 'step' in checkpoint:
                        step = checkpoint['step']
                    if 'training_history' in checkpoint:
                        self.training_history = checkpoint['training_history']
                    if 'feature_tracker' in checkpoint:
                        self.feature_tracker = checkpoint['feature_tracker']
                    if 'lambda_l1' in checkpoint:
                        self.lambda_l1 = checkpoint['lambda_l1']
                    print(f"Successfully resumed from step {step}")
                else:
                    # Just a state dict, not a full checkpoint
                    self.load_state_dict(checkpoint)
                    print("Loaded model weights only, not training state")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Starting training from scratch instead")

        print("\n" + "="*50)
        print("SPARSE TRANSFORMER TRAINING CONFIGURATION")
        print("="*50)
        print(f"Input dimension (n): {self.n}")
        print(f"Feature dimension (m): {self.m}")
        print(f"Attention dimension (a): {self.a}")
        print(f"Total Steps: {actual_total_steps:,}")
        print(f"Batch Size: {batch_size:,}")
        print(f"Gradient Accumulation Steps: {grad_accum_steps}")
        print(f"Effective Batch Size: {batch_size * grad_accum_steps:,}")
        print(f"Learning Rate: {learning_rate:.1e}")
        print(f"L1 Regularization (λ): {final_lambda:.2f}")
        print(f"Mixed Precision: {'Enabled' if self.use_mixed_precision else 'Disabled'}")
        print(f"Device: {self.device}")
        print("="*50)

        print("\nTraining Progress:")
        progress_header = f"{'Step':>10} {'Epoch':>5} {'Loss':>8} {'ValLoss':>8} {'λ':>5} {'Dead%':>6} {'Sparse%':>7} {'Time':<10} {'ETA':<10}"
        print("="*80)
        print(progress_header)
        print("-"*80)

        for epoch in range(num_epochs):
            self.train()
            epoch_train_loss = 0.0
            num_batches = 0

            # Reset gradients at the beginning of each epoch
            optimizer.zero_grad()
            accum_batch = 0
            running_loss = 0.0

            for batch_idx, batch in enumerate(train_loader):
                # Apply lambda warmup during initial phase of training
                if step < warmup_steps:
                    self.lambda_l1 = (step / warmup_steps) * final_lambda
                else:
                    self.lambda_l1 = final_lambda

                # Apply learning rate decay in last 20% of training
                if step >= decay_start_step:
                    progress = (step - decay_start_step) / (actual_total_steps - decay_start_step)
                    new_lr = learning_rate * (1 - progress)  # Linear decay to zero
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                
                x = batch[0]
                
                # Forward pass with optional mixed precision
                if self.use_mixed_precision:
                    with autocast():
                        x, x_hat, f_x, v = self.forward(x)
                        total_loss, L2_loss, L1_loss = self.compute_loss(x, x_hat, f_x, v)
                        # Scale loss by the number of accumulation steps for proper scaling
                        total_loss = total_loss / grad_accum_steps
                    
                    # Backward pass with gradient scaling
                    scaler.scale(total_loss).backward()
                    running_loss += total_loss.item() * grad_accum_steps  # Track the actual loss
                    
                    # Update weights if we've accumulated enough gradients
                    accum_batch += 1
                    if accum_batch >= grad_accum_steps:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        accum_batch = 0
                else:
                    # Standard forward and backward pass
                    x, x_hat, f_x, v = self.forward(x)
                    total_loss, L2_loss, L1_loss = self.compute_loss(x, x_hat, f_x, v)
                    # Scale loss by the number of accumulation steps for proper scaling
                    total_loss = total_loss / grad_accum_steps
                    total_loss.backward()
                    running_loss += total_loss.item() * grad_accum_steps  # Track the actual loss
                    
                    # Update weights if we've accumulated enough gradients
                    accum_batch += 1
                    if accum_batch >= grad_accum_steps:
                        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        accum_batch = 0
                
                # Update feature tracking
                dead_ratio, stats = self.feature_tracker.update(f_x)

                epoch_train_loss += running_loss
                running_loss = 0.0  # Reset for next accumulation
                num_batches += 1
                step += 1

                # Periodic validation and logging
                if step % eval_freq == 0:
                    self.eval()
                    val_loss = 0.0
                    l1_val_loss = 0.0
                    l2_val_loss = 0.0
                    val_batches = 0

                    with torch.no_grad():
                        for val_batch in val_loader:
                            x_val = val_batch[0]
                            x_val, x_hat_val, f_x_val, v_val = self.forward(x_val)
                            val_total_loss, l2_val, l1_val = self.compute_loss(
                                x_val, x_hat_val, f_x_val, v_val)
                            val_loss += val_total_loss.item()
                            l1_val_loss += l1_val.item()
                            l2_val_loss += l2_val.item()
                            val_batches += 1

                    avg_val_loss = val_loss / val_batches
                    avg_l1_val_loss = l1_val_loss / val_batches
                    avg_l2_val_loss = l2_val_loss / val_batches

                    # Calculate sparsity
                    sparsity = (f_x.abs() >= self.feature_tracker.activation_threshold).float().mean().item()
                    
                    # Calculate elapsed time and estimated time remaining
                    elapsed_time = time.time() - start_time
                    if step > 0:
                        time_per_step = elapsed_time / step
                        remaining_steps = actual_total_steps - step
                        estimated_remaining = remaining_steps * time_per_step
                        elapsed_str = str(timedelta(seconds=int(elapsed_time))).split('.')[0]
                        remaining_str = str(timedelta(seconds=int(estimated_remaining))).split('.')[0]
                    else:
                        elapsed_str = "-"
                        remaining_str = "-"

                    # Log progress in clean tabular format
                    progress = f"{step:10,d} {epoch:5d} {total_loss.item()*grad_accum_steps:8.4f} {avg_val_loss:8.4f} {self.lambda_l1:5.2f} {dead_ratio*100:6.1f}% {sparsity*100:7.1f}% {elapsed_str:<10} {remaining_str:<10}"
                    print(progress)

                    # Update training history
                    self.training_history["steps"].append(step)
                    self.training_history["train_loss"].append(total_loss.item()*grad_accum_steps)
                    self.training_history["val_loss"].append(avg_val_loss)
                    self.training_history["l1_loss"].append(L1_loss.item()*grad_accum_steps)
                    self.training_history["l2_loss"].append(L2_loss.item()*grad_accum_steps)
                    self.training_history["lambda"].append(self.lambda_l1)
                    self.training_history["dead_ratio"].append(dead_ratio)
                    self.training_history["sparsity"].append(sparsity)
                    
                    # Create checkpoint for saving (with minimal info by default)
                    if save_full_state:
                        checkpoint = {
                            'model_state_dict': self.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'step': step,
                            'epoch': epoch,
                            'train_loss': total_loss.item()*grad_accum_steps,
                            'val_loss': avg_val_loss,
                            'lambda_l1': self.lambda_l1,
                            'dead_ratio': dead_ratio,
                            'sparsity': sparsity,
                            'training_history': self.training_history
                        }
                    else:
                        # Minimal checkpoint - just model weights
                        checkpoint = self.state_dict()
                    
                    # Save best model if requested and if we have a new best validation loss
                    if save_best and avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        best_model_path = f"{self.st_model_path}.best"
                        torch.save(checkpoint, best_model_path)
                        print(f"    ✓ New best model saved (val_loss: {avg_val_loss:.4f})")

                    # Save checkpoint periodically (with restrictions to minimize overhead)
                    if enable_checkpoints and step % checkpoint_freq == 0:
                        # Don't save if we're in first 10% of training or close to the end
                        remaining_steps = actual_total_steps - step
                        if step > actual_total_steps * 0.1 and remaining_steps > checkpoint_freq // 2:
                            checkpoint_path = f"{self.st_model_path}.step{step}"
                            torch.save(checkpoint, checkpoint_path)
                            print(f"    ✓ Checkpoint saved at step {step}")
                    
                    # Periodically clear CUDA cache to prevent memory fragmentation
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                    
                    self.train()

            # Make sure to update with any remaining accumulated gradients at the end of the epoch
            if accum_batch > 0:
                if self.use_mixed_precision:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                if self.use_mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

        # Final evaluation
        self.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_batches = 0
            for val_batch in val_loader:
                x_val = val_batch[0]
                x_val, x_hat_val, f_x_val, v_val = self.forward(x_val)
                val_total_loss, _, _ = self.compute_loss(x_val, x_hat_val, f_x_val, v_val)
                val_loss += val_total_loss.item()
                val_batches += 1
            final_val_loss = val_loss / val_batches
        
        # Training summary
        total_time = time.time() - start_time
        print("="*80)
        print("\nTRAINING SUMMARY:")
        print(f"Total time: {timedelta(seconds=int(total_time))}")
        print(f"Final validation loss: {final_val_loss:.4f}")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Final dead feature ratio: {dead_ratio:.1%}")
        print(f"Steps completed: {step:,}/{actual_total_steps:,}")
        print(f"Final λ: {self.lambda_l1:.2f}")
        
        # Save final model (always save full state for final model)
        final_checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if save_full_state else None,
            'step': step,
            'epoch': epoch,
            'train_loss': total_loss.item() * grad_accum_steps if 'total_loss' in locals() else None,
            'val_loss': final_val_loss,
            'lambda_l1': self.lambda_l1,
            'dead_ratio': dead_ratio,
            'sparsity': sparsity if 'sparsity' in locals() else None,
            'training_history': self.training_history,
            'total_training_time': total_time
        }
        
        # Clean up None values to save space
        final_checkpoint = {k: v for k, v in final_checkpoint.items() if v is not None}
        
        torch.save(final_checkpoint, self.st_model_path)
        print(f"Final model saved to {self.st_model_path}")
        
        # Plot training history if available
        if len(self.training_history["steps"]) > 5:  # Only if we have enough data points
            try:
                history_path = f"{os.path.splitext(self.st_model_path)[0]}_history.png"
                self.plot_training_history(save_path=history_path)
                print(f"Training history plot saved to {history_path}")
            except Exception as e:
                print(f"Could not generate training history plots: {e}")
        
        return self