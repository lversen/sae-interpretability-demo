import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import logging
import time
import gc
import os
from datetime import timedelta
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Tuple, Optional, Dict, List, Any, Union, Callable

# Import DeadFeatureTracker - original implementation should be available now
from deadfeatures import DeadFeatureTracker

class SparseTransformer(nn.Module):
    """
    Optimized SparseTransformer with faster attention computation and better performance.
    """
    
    def __init__(self, X, n: int, m: int, a: int, st_model_path: str,
                 lambda_l1: float = 5.0, num_heads: int = 1, device: str = 'cuda',
                 window_size: int = 10_000_000, update_interval: int = 1_000,
                 activation_threshold: float = 1e-3, use_mixed_precision: bool = True,
                 use_compile: bool = True, memory_strategy: str = 'diversity',
                 log_level: str = 'INFO', use_direct_kv: bool = True,
                 activation: str = 'relu', attention_fn: str = 'softmax'):
        """
        Initialize the Optimized Sparse Transformer model.
        
        Args:
            X: Input data tensor or array for the memory bank (only used for scaling)
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
            use_compile: Whether to use torch.compile for optimization (PyTorch 2.0+)
            memory_strategy: Strategy for memory bank updates (only relevant if use_direct_kv=False)
            log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
            use_direct_kv: Whether to use direct K-V matrices (recommended for speed)
            activation: Activation function to use in the model ('relu', 'gelu', etc.)
            attention_fn: Function to use for attention score processing
        """
        super().__init__()
        
        # Set up logging with minimal overhead
        self.logger = self._setup_logger(log_level)
        
        # Model parameters
        self.n, self.m, self.a = n, m, a
        self.st_model_path = st_model_path
        self.lambda_l1 = lambda_l1
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.activation_threshold = activation_threshold
        self.use_mixed_precision = use_mixed_precision
        self.use_compile = use_compile
        self.memory_strategy = memory_strategy
        self.use_direct_kv = use_direct_kv  # Default to direct K-V for speed
        self.steps = 0
        self.total_steps = 0
        
        # Cache scaling factor for X for faster preprocessing
        self.X = self.type_check(X)
        self.X_scaling_factor = self.preprocess(X)
        
        # Training history
        self.training_history = {"steps": [], "train_loss": [], "val_loss": [], 
                                "l1_loss": [], "l2_loss": [], "lambda": [], 
                                "dead_ratio": [], "sparsity": [], "avg_feature_norm": []}
        
        # Pre-compute scaling factor for attention (optimization)
        self.attention_scale_factor = 1.0 / math.sqrt(self.a)
        
        # Set activation function
        self.activation_name = activation
        self.activation = self._get_activation_function(activation)
        
        # Set attention function
        self.attention_fn_name = attention_fn
        self.attention_fn = self._get_attention_function(attention_fn)
        
        # Projections - always use query projection
        self.W_q = nn.Linear(n, a)
        
        # Only create the parameters we need based on approach
        if self.use_direct_kv:
            # Direct parameter matrices (faster approach)
            self.W_k_direct = nn.Parameter(torch.Tensor(self.m, self.a))
            self.W_v_direct = nn.Parameter(torch.Tensor(self.m, self.n))
        else:
            # Memory-based projection matrices
            self.W_k = nn.Linear(n, a)
            self.W_v = nn.Linear(n, n)
            # Initialize memory indices (only needed for original approach)
            self.register_buffer('memory_indices', self._initial_memory_selection())
        
        # We'll use only one set of norms to reduce compute
        self.norm_all = nn.LayerNorm(a)
        
        # Feature tracking with optimized update interval
        self.feature_tracker = DeadFeatureTracker(
            num_features=self.m,
            activation_threshold=activation_threshold,
            window_size=window_size,
            update_interval=update_interval
        )
        
        # Initialize weights
        self.initialize_weights()
        self.to(self.device)
        
        # Apply torch.compile if requested and available (major speedup)
        if self.use_compile and hasattr(torch, 'compile'):
            self.logger.info("Using torch.compile for model optimization")
            # Use max-autotune mode for best training performance
            self = torch.compile(self, mode='max-autotune')
        
        # Log configuration
        self.logger.info(f"Using {'direct K-V matrices' if self.use_direct_kv else 'memory bank'} approach")
        self.logger.info(f"Activation: {self.activation_name}, Attention: {self.attention_fn_name}")
    
    def _setup_logger(self, log_level: str) -> logging.Logger:
        """Setup logger for the model"""
        logger = logging.getLogger(f"OptimizedST_{id(self)}")
        
        # Set level based on parameter
        level = getattr(logging, log_level.upper(), logging.INFO)
        logger.setLevel(level)
        
        # Create handler if not already set up
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def _get_activation_function(self, activation_name: str):
        """Returns the activation function based on the name."""
        activation_functions = {
            'relu': F.relu,
            'leaky_relu': lambda x: F.leaky_relu(x, negative_slope=0.1),
            'gelu': F.gelu,
            'sigmoid': torch.sigmoid,
            'tanh': torch.tanh,
            'none': lambda x: x,  # Identity function
            'softplus': F.softplus,
            'elu': F.elu,
            'selu': F.selu,
        }
        
        if activation_name.lower() not in activation_functions:
            self.logger.warning(f"Activation '{activation_name}' not supported. Using 'relu'.")
            return activation_functions['relu']
        
        return activation_functions[activation_name.lower()]
    
    def _get_attention_function(self, attention_name: str):
        """Returns the attention function based on the name."""
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
        
        def relu_softmax(scores, dim=-1):
            """Apply ReLU first, then softmax - handles negative values differently"""
            scores = F.relu(scores)
            return F.softmax(scores, dim=dim)
        
        def custom_softmax(scores, dim=-1, beta=1.0):
            """Softmax with temperature parameter"""
            return F.softmax(scores * beta, dim=dim)
        
        def polynomial_attention(scores, dim=-1):
            """Use polynomial kernel (x^3) instead of exponential"""
            # Apply ReLU to ensure non-negativity
            scores = F.relu(scores)
            # Raise to power 3
            poly_scores = torch.pow(scores, 3)
            # Normalize
            sum_scores = torch.sum(poly_scores, dim=dim, keepdim=True).clamp(min=1e-6)
            return poly_scores / sum_scores
        
        def adaptive_sparse_attention(scores, dim=-1):
            """Adaptive sparse attention targeting consistent entropy"""
            # Sort scores in descending order
            sorted_scores, _ = torch.sort(scores, descending=True, dim=dim)
            
            # Adapt threshold based on sequence length
            seq_len = scores.size(dim)
            k = max(1, int(math.sqrt(seq_len)))
            
            # Use the k-th value as threshold
            if k < seq_len:
                threshold = sorted_scores.select(dim, k)
                # Add singleton dimension
                threshold = threshold.unsqueeze(dim)
            else:
                threshold = torch.min(sorted_scores, dim=dim, keepdim=True)[0]
            
            # Create mask for values above threshold
            mask = scores >= threshold
            
            # Apply mask and normalize
            masked_scores = scores.clone()
            masked_scores.masked_fill_(~mask, -float('inf'))
            return F.softmax(masked_scores, dim=dim)
            
        def relu_attention(scores, dim=-1):
            """Simple ReLU attention - applies ReLU and normalizes."""
            relu_scores = F.relu(scores)
            sum_scores = torch.sum(relu_scores, dim=dim, keepdim=True).clamp(min=1e-6)
            return relu_scores / sum_scores

        attention_functions = {
            'softmax': lambda x, dim=-1: F.softmax(x, dim=dim),
            'sparsemax': sparsemax,
            'normalized_activation': normalized_activation,
            'relu_softmax': relu_softmax,
            'softmax_hard': lambda x, dim=-1: custom_softmax(x, dim=dim, beta=2.0),
            'softmax_soft': lambda x, dim=-1: custom_softmax(x, dim=dim, beta=0.5),
            'polynomial_attention': polynomial_attention,
            'adaptive_sparse': adaptive_sparse_attention,
            'relu_attention': relu_attention,
        }
        
        if attention_name.lower() not in attention_functions:
            self.logger.warning(f"Attention '{attention_name}' not supported. Using 'softmax'.")
            return attention_functions['softmax']
        
        return attention_functions[attention_name.lower()]
    
    def _initial_memory_selection(self):
        """Initialize memory indices using a strategic selection approach."""
        # Only needed for original approach
        if self.use_direct_kv:
            return torch.zeros(self.m, dtype=torch.long, device=self.device)
            
        # For memory bank approach, just use random selection for speed
        return torch.randperm(self.X.shape[0])[:self.m].to(self.device)
    
    def type_check(self, x):
        """Ensure input tensors are on the correct device and have the right type"""
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x.astype(np.float32)).to(self.device)
        return x.to(self.device) if x.device != self.device else x

    def optimized_attention(self, query, key, value):
        """
        Optimized attention mechanism using PyTorch's native attention when possible
        
        Args:
            query: Query tensor [batch_size, query_dim]
            key: Key tensor [memory_size, key_dim]
            value: Value tensor [memory_size, value_dim]
            
        Returns:
            Tuple of (output, attention_weights)
        """
        # Use native attention when available and using standard softmax (fastest method)
        if hasattr(F, 'scaled_dot_product_attention') and self.attention_fn_name == 'softmax':
            # Reshape tensors for batch attention - minimize reshaping operations
            q_reshaped = query.unsqueeze(1)  # [batch_size, 1, query_dim]
            k_reshaped = key.unsqueeze(0)    # [1, memory_size, key_dim]
            v_reshaped = value.unsqueeze(0)  # [1, memory_size, value_dim]
            
            # Use native scaled dot-product attention (with flash attention if available)
            output = F.scaled_dot_product_attention(
                q_reshaped, k_reshaped, v_reshaped,
                scale=self.attention_scale_factor,
                dropout_p=0.0
            ).squeeze(1)  # [batch_size, value_dim]
            
            # Compute attention weights for feature tracking (done outside critical path when possible)
            if self.training:
                attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.attention_scale_factor
                attn_weights = F.softmax(attn_scores, dim=-1)
            else:
                # Don't compute weights during inference if not needed
                attn_weights = torch.zeros((query.size(0), key.size(0)), device=query.device)
                
        else:
            # Fallback for custom attention functions
            # Compute attention scores
            attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.attention_scale_factor
            
            # Apply custom attention function
            attn_weights = self.attention_fn(attn_scores, dim=-1)
            
            # Compute output
            output = torch.matmul(attn_weights, value)
            
        return output, attn_weights

    def forward(self, x):
        """Optimized forward pass for the SparseTransformer model."""
        self.steps += 1
        
        # Type conversion and preprocessing
        x = self.type_check(x)  # Shape: [batch_size, n]
        x = x / self.X_scaling_factor  # Use cached scaling factor
        
        if self.use_direct_kv:
            # DIRECT K-V APPROACH (faster)
            # Project input to query space
            q = self.norm_all(self.activation(self.W_q(x)))  # Shape: [batch_size, a]
            
            # Apply activation to the direct parameters
            k = self.activation(self.W_k_direct)  # Shape: [m, a]
            v = self.W_v_direct  # Shape: [m, n]
            
            # We can skip normalization of k and v for speed since they're learned directly
            
        else:
            # MEMORY BANK APPROACH
            # Get cross attention context
            X_cross = self.X[self.memory_indices]  # Shape: [m, n]
            X_cross = X_cross / self.X_scaling_factor
            
            # Project to attention space
            q = self.norm_all(self.activation(self.W_q(x)))  # Shape: [batch_size, a]
            k = self.activation(self.W_k(X_cross))  # Shape: [m, a]
            v = self.W_v(X_cross)  # Shape: [m, n]
        
        # Compute attention - optimized path
        x_hat, f = self.optimized_attention(q, k, v)
        
        # Update feature tracker during training
        if self.training:
            dead_ratio, _ = self.feature_tracker.update(f)
        
        return x, x_hat, f, v

    def compute_loss(self, x, x_hat, f, v):
        """
        Compute loss with L1 penalty: L2 reconstruction + L1 regularization
        """
        # L2 reconstruction term
        L2_loss = torch.mean(torch.norm(x - x_hat, p=2, dim=1)**2)
        
        # L1 sparsity penalty
        v_norms = torch.norm(v, p=2, dim=1)
        L1_penalty = self.lambda_l1 * torch.mean(torch.sum(f * v_norms, dim=1))
        
        return L2_loss + L1_penalty, L2_loss, L1_penalty

    def feature_activations(self, x):
        """Return the feature activations for a given input"""
        x = self.type_check(x)
        _, _, f, _ = self.forward(x)
        return f

    def preprocess(self, X):
        """Calculate scaling factor C so that E_x[||x||₂] = √n after dividing by C"""
        X = self.type_check(X)
        mean_norm = torch.mean(torch.norm(X, p=2, dim=1))
        C = mean_norm / np.sqrt(self.n)
        return C
        
    def initialize_weights(self):
        """Initialize weights efficiently"""
        with torch.no_grad():
            # Initialize query projection
            nn.init.normal_(self.W_q.weight, mean=0.0, std=0.02)
            if hasattr(self.W_q, 'bias') and self.W_q.bias is not None:
                nn.init.zeros_(self.W_q.bias)
            
            if self.use_direct_kv:
                # Initialize direct K-V matrices
                nn.init.normal_(self.W_k_direct, mean=0.0, std=0.02)
                nn.init.kaiming_uniform_(self.W_v_direct, a=math.sqrt(5))
            else:
                # Initialize memory-based projections
                nn.init.normal_(self.W_k.weight, mean=0.0, std=0.02)
                nn.init.kaiming_uniform_(self.W_v.weight, a=math.sqrt(5))
                
                if hasattr(self.W_k, 'bias') and self.W_k.bias is not None:
                    nn.init.zeros_(self.W_k.bias)
                if hasattr(self.W_v, 'bias') and self.W_v.bias is not None:
                    nn.init.zeros_(self.W_v.bias)
    
    def train_and_validate(self, X_train, X_val, learning_rate=5e-5, batch_size=4096, 
                          target_steps=200_000, checkpoint_freq=50000, save_best=True, 
                          enable_checkpoints=True, resume_from=None, grad_accum_steps=1, 
                          eval_freq=None, scheduler_type=None, early_stopping=False,
                          early_stopping_patience=5, warmup_steps_pct=0.05,
                          final_decay_pct=0.2):
        """
        Optimized training method with mixed precision and fast iteration.
        """
        # Initialize optimizer - Adam with stable defaults
        optimizer = torch.optim.Adam(
            self.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0)
        
        # Initialize mixed precision scaler if enabled (faster training)
        scaler = GradScaler() if self.use_mixed_precision and self.device != 'cpu' else None
        
        # Set up scheduler if requested
        scheduler = None
        if scheduler_type == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=1000, T_mult=2, eta_min=learning_rate / 100
            )
            
        # Calculate required epochs to reach target steps
        steps_per_epoch = max(1, len(X_train) // batch_size)
        num_epochs = max(1, target_steps // steps_per_epoch)
        actual_total_steps = num_epochs * steps_per_epoch
        self.total_steps = actual_total_steps
        
        # Set evaluation frequency
        if eval_freq is None:
            eval_freq = max(1, steps_per_epoch // 5)  # 5 times per epoch
        
        # Training parameters
        warmup_steps = int(actual_total_steps * warmup_steps_pct)
        decay_start_step = int(actual_total_steps * (1 - final_decay_pct))
        step = 0
        best_val_loss = float('inf')
        final_lambda = self.lambda_l1
        
        # Set up progress tracking
        progress_bar = tqdm(total=actual_total_steps, desc="Training", dynamic_ncols=True)
        
        # Reset gradients at the beginning
        optimizer.zero_grad()
        accum_batch = 0
        early_stop = False
        val_loss_value = 0.0
        
        # Create train and validation datasets
        train_dataset = torch.utils.data.TensorDataset(X_train)
        val_dataset = torch.utils.data.TensorDataset(X_val)
        
        # Configure data loaders
        pin_memory = torch.cuda.is_available() and X_train.device.type == 'cpu'
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            pin_memory=pin_memory, num_workers=2 if pin_memory else 0)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size*2, shuffle=False,
            pin_memory=pin_memory, num_workers=2 if pin_memory else 0)
        
        # Training loop
        for epoch in range(num_epochs):
            if early_stop:
                break
                
            self.train()
            
            for batch_idx, (batch,) in enumerate(train_loader):
                if step >= actual_total_steps:
                    break
                    
                # Apply lambda warmup
                if step < warmup_steps:
                    self.lambda_l1 = (step / warmup_steps) * final_lambda
                else:
                    self.lambda_l1 = final_lambda
                
                # Apply learning rate decay
                if scheduler is None and step >= decay_start_step:
                    progress = (step - decay_start_step) / (actual_total_steps - decay_start_step)
                    new_lr = learning_rate * (1 - progress)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                
                # Forward pass with mixed precision
                with autocast(device_type=self.device.split(':')[0] if ':' in self.device else self.device, 
                             enabled=self.use_mixed_precision):
                    x, x_hat, f_x, v = self.forward(batch)
                    total_loss, L2_loss, L1_loss = self.compute_loss(x, x_hat, f_x, v)
                    total_loss = total_loss / grad_accum_steps
                
                # Backward pass
                if self.use_mixed_precision and scaler is not None:
                    scaler.scale(total_loss).backward()
                else:
                    total_loss.backward()
                
                # Gradient accumulation
                accum_batch += 1
                if accum_batch >= grad_accum_steps:
                    if self.use_mixed_precision and self.device != 'cpu':
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                        optimizer.step()
                        
                    # Step scheduler if enabled
                    if scheduler is not None:
                        scheduler.step()
                        
                    optimizer.zero_grad()
                    accum_batch = 0
                
                # Update step counter
                step += 1
                
                # Get feature tracking stats
                dead_features = self.feature_tracker.get_dead_features()
                dead_ratio = dead_features.float().mean().item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'd%': f"{dead_ratio*100:.0f}",
                    'λ': f"{self.lambda_l1:.1f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.1e}",
                    'loss': f"{total_loss.item()*grad_accum_steps:.3f}",
                    'val': f"{val_loss_value:.3f}"
                })
                progress_bar.update(1)
                
                # Periodic validation
                if step % eval_freq == 0:
                    val_loss, val_l2_loss, val_l1_loss = self._evaluate(val_loader)
                    val_loss_value = val_loss
                    
                    # Update training history
                    self._update_training_history(
                        step, total_loss.item()*grad_accum_steps, val_loss,
                        L1_loss.item()*grad_accum_steps, L2_loss.item()*grad_accum_steps,
                        dead_ratio, f_x, v
                    )
                    
                    # Save best model if requested
                    if save_best and val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(self.state_dict(), f"{self.st_model_path}.best")
                    
                    # Check for early stopping
                    if early_stopping and step > actual_total_steps * 0.5:
                        early_stop = self._check_early_stopping(
                            val_loss, early_stopping_patience)
                        if early_stop:
                            break
                    
                    # Release memory
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            if early_stop:
                break
        
        progress_bar.close()
        
        # Final evaluation
        final_val_loss, _, _ = self._evaluate(val_loader)
        
        # Save final model
        torch.save(self.state_dict(), self.st_model_path)
        self.logger.info(f"Training complete. Final validation loss: {final_val_loss:.4f}")
        
        return self
    
    def _evaluate(self, val_loader):
        """Evaluate model on validation data"""
        self.eval()
        val_loss = 0.0
        l1_val_loss = 0.0
        l2_val_loss = 0.0
        val_batches = 0
        
        device_type = self.device.split(':')[0] if ':' in self.device else self.device
        
        with torch.no_grad():
            for batch, in val_loader:
                # Forward pass
                with autocast(device_type=device_type, enabled=self.use_mixed_precision):
                    x_val, x_hat_val, f_x_val, v_val = self.forward(batch)
                    batch_loss, l2_val, l1_val = self.compute_loss(x_val, x_hat_val, f_x_val, v_val)
                
                # Update metrics
                val_loss += batch_loss.item()
                l1_val_loss += l1_val.item()
                l2_val_loss += l2_val.item()
                val_batches += 1
        
        # Calculate averages
        avg_val_loss = val_loss / max(1, val_batches)
        avg_l1_val_loss = l1_val_loss / max(1, val_batches)
        avg_l2_val_loss = l2_val_loss / max(1, val_batches)
        
        self.train()
        return avg_val_loss, avg_l2_val_loss, avg_l1_val_loss
    
    def _update_training_history(self, step, train_loss, val_loss, l1_loss, l2_loss, 
                                dead_ratio, f_x, v):
        """Update training history metrics"""
        # Calculate additional metrics
        sparsity = (f_x.abs() >= self.activation_threshold).float().mean().item()
        v_norms = torch.norm(v, p=2, dim=1)
        avg_feature_norm = v_norms.mean().item()
        
        # Update dictionary
        self.training_history["steps"].append(step)
        self.training_history["train_loss"].append(train_loss)
        self.training_history["val_loss"].append(val_loss)
        self.training_history["l1_loss"].append(l1_loss)
        self.training_history["l2_loss"].append(l2_loss)
        self.training_history["lambda"].append(self.lambda_l1)
        self.training_history["dead_ratio"].append(dead_ratio)
        self.training_history["sparsity"].append(sparsity)
        self.training_history["avg_feature_norm"].append(avg_feature_norm)
    
    def _check_early_stopping(self, val_loss, patience, min_delta=0.001):
        """Check if early stopping criteria are met"""
        if len(self.training_history["val_loss"]) <= patience:
            return False
            
        recent_losses = self.training_history["val_loss"][-(patience+1):]
        best_loss = min(recent_losses[:-1])
        
        return val_loss > best_loss - min_delta
    
    def plot_training_history(self, save_path: Optional[str] = None, figsize=(16, 12)) -> Figure:
        """
        Plot the training history metrics.
        
        Args:
            save_path: Optional path to save the plot to
            figsize: Figure size
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        if len(self.training_history["steps"]) == 0:
            raise ValueError("No training history available to plot")
            
        fig, axs = plt.subplots(3, 2, figsize=figsize)
        
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
        
        # Plot feature norms if available
        ax = axs[2, 0]
        if "avg_feature_norm" in self.training_history and self.training_history["avg_feature_norm"]:
            ax.plot(self.training_history["steps"], self.training_history["avg_feature_norm"])
            ax.set_xlabel("Steps")
            ax.set_ylabel("Average Feature Norm")
            ax.set_title("Feature Magnitude")
            ax.grid(alpha=0.3)
        else:
            ax.set_visible(False)
            
        # Add a training settings summary
        ax = axs[2, 1]
        ax.axis('off')
        settings_text = (
            f"MODEL SETTINGS\n"
            f"Dimensions: n={self.n}, m={self.m}, a={self.a}\n"
            f"Final λ: {self.lambda_l1:.2f}\n"
            f"Memory approach: {'Direct K-V' if self.use_direct_kv else 'Memory Bank'}\n"
            f"Training steps: {self.steps}\n"
            f"Mixed precision: {self.use_mixed_precision}\n"
            f"Dead features: {self.feature_tracker.get_dead_features().sum().item()} / {self.m}"
        )
        ax.text(0.5, 0.5, settings_text, 
                ha='center', va='center', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        fig.suptitle(f"Training History - Optimized Sparse Transformer", fontsize=16)
        plt.subplots_adjust(top=0.93)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"Training history plot saved to {save_path}")
            
        return fig
    
    def plot_decoder_weights(self, input_shape=(28, 28), num_weights=16, 
                            rows=4, cols=4, figsize=(12, 12),
                            cmap='coolwarm', save_path=None, display_plot=True):
        """
        Plot decoder weights as images
        
        Args:
            input_shape: Shape to reshape weights to (e.g., (28, 28) for MNIST)
            num_weights: Number of weights to visualize
            rows, cols: Grid dimensions for plotting
            figsize: Figure size
            cmap: Colormap to use
            save_path: Optional path to save the figure
            display_plot: Whether to display the plot
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        self.eval()  # Switch to eval mode
        
        with torch.no_grad():
            # Get value vectors
            if self.use_direct_kv:
                # Direct K-V approach
                weight_matrix = self.W_v_direct.cpu().numpy()
            else:
                # Memory bank approach
                X_cross = self.X[self.memory_indices[:50]]  # Just use a subset
                v = self.W_v(X_cross)  # Shape: [subset_of_m, n]
                weight_matrix = v.cpu().numpy()
        
        # Calculate L2 norms to identify important features
        feature_norms = np.linalg.norm(weight_matrix, axis=1)
        sorted_indices = np.argsort(-feature_norms)  # Sort by descending norm
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        fig.suptitle(f"Optimized ST: Value Vectors (Step {self.steps})", fontsize=16)
        
        # Plot each weight vector
        for i in range(min(rows * cols, min(num_weights, len(sorted_indices)))):
            # Get the index of the i-th highest norm weight
            idx = sorted_indices[i]
            
            # Get the weight vector
            weight = weight_matrix[idx]
            
            # Calculate norm for title
            norm = feature_norms[idx]
            
            # Reshape to input shape if possible
            if np.prod(input_shape) == weight.shape[0]:
                weight_img = weight.reshape(input_shape)
            else:
                # If dimensions don't match, reshape to a square or rectangle
                side = int(np.sqrt(weight.shape[0]))
                weight_img = weight[:side*side].reshape(side, side)
                
            # Create subplot
            ax = fig.add_subplot(rows, cols, i+1)
            
            # Determine color range for better visualization
            vmax = max(abs(weight_img.max()), abs(weight_img.min()))
            
            # Plot the weight image
            im = ax.imshow(weight_img, cmap=cmap, vmin=-vmax, vmax=vmax)
            
            # Add title with feature index and norm
            ax.set_title(f"F{idx}\nNorm: {norm:.2f}", fontsize=10)
            
            # Remove axis ticks for cleaner look
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Add a colorbar
        plt.colorbar(im, ax=fig.axes, shrink=0.7)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
        
        # Save figure if requested
        if save_path is not None:
            # Replace {step} placeholder with actual step if present
            actual_save_path = save_path.format(step=self.steps) if '{step}' in save_path else save_path
            plt.savefig(actual_save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Decoder weights plot saved to {actual_save_path}")
        
        # Display if requested
        if display_plot:
            plt.show()
        
        return fig

# Usage example:
if __name__ == "__main__":
    # Define your parameters
    n = 768  # Input dimension (e.g., hidden state size)
    m = 256  # Feature dimension
    a = 64   # Attention dimension
    
    # Create dummy data for example
    train_data = torch.randn(10000, n)
    val_data = torch.randn(1000, n)
    
    # Choose the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize the model
    model = OptimizedSparseTransformer(
        X=train_data,  # Your training data
        n=n, 
        m=m,
        a=a,
        st_model_path="models/optimized_st.pt",
        use_mixed_precision=True,
        use_compile=True,  # Enable torch.compile for PyTorch 2.0+
        use_direct_kv=True,  # Use direct K-V approach for speed
        device=device
    )
    
    # Train the model
    model.train_and_validate(
        X_train=train_data,
        X_val=val_data,
        batch_size=256,  # Adjust based on your GPU memory
        learning_rate=5e-5,
        target_steps=1000,  # Short example run, use 50000+ for real training
        grad_accum_steps=1,
        eval_freq=100
    )
    
    # Plot the training history
    model.plot_training_history(save_path="training_history.png")
    
    # Plot the decoder weights (if input dimension is suitable for visualization)
    model.plot_decoder_weights(
        input_shape=(int(np.sqrt(n)), int(np.sqrt(n))),
        save_path="decoder_weights.png"
    )