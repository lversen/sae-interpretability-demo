import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional, Callable, Dict, Any, List
from deadfeatures import DeadFeatureTracker
import time
import math
import logging
from datetime import timedelta
from torch.cuda.amp import GradScaler
# Import autocast in a way that works with different PyTorch versions
try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast
import torch.nn.functional as F
from tqdm.auto import tqdm


class SparseTransformer(nn.Module):
    def __init__(self, X, n: int, m: int, a: int, st_model_path: str,
                 lambda_l1: float = 5.0, num_heads: int = 1, device: str = 'cuda',
                 window_size: int = 10_000_000, update_interval: int = 1_000,
                 activation_threshold: float = 1e-3, use_direct_kv: bool = False,
                 activation: str = 'relu', attention_fn: str = 'softmax',
                 use_mixed_precision: bool = False, log_level: str = 'INFO'):
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
        self.use_mixed_precision = use_mixed_precision
        
        # Set up logging
        self.logger = self._setup_logger(log_level)
        
        # Set activation function
        self.activation_name = activation
        self.activation = self._get_activation_function(activation)
        
        # Set attention function
        self.attention_fn_name = attention_fn
        self.attention_fn = self._get_attention_function(attention_fn)
        
        self.logger.info(f"Using activation function: {self.activation_name}")
        self.logger.info(f"Using attention function: {self.attention_fn_name}")
        
        # Training history tracking
        self.training_history = {
            "steps": [], 
            "train_loss": [], 
            "val_loss": [], 
            "l1_loss": [], 
            "l2_loss": [], 
            "lambda": [], 
            "dead_ratio": [], 
            "sparsity": [], 
            "avg_feature_norm": []
        }
        
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
    
    def _setup_logger(self, log_level: str) -> logging.Logger:
        """Setup logger for the model"""
        logger = logging.getLogger(f"SparseTransformer_Old_{id(self)}")
        
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
            self.logger.warning(f"Activation function '{activation_name}' not supported. "
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
            self.logger.warning(f"Attention function '{attention_name}' not supported. "
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
                self.logger.info("Memory indices updated for memory bank approach")
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

    def _cleanup_memory(self):
        """Release memory when possible"""
        if torch.cuda.is_available():
            # Force garbage collection
            import gc
            gc.collect()
            torch.cuda.empty_cache()

    def train_and_validate(self, X_train, X_val, learning_rate=5e-5, batch_size=4096, 
                          target_steps=200_000, grad_accum_steps=1, 
                          eval_freq=None, use_mixed_precision=None):
        """
        Train the Sparse Transformer targeting a specific number of steps using tqdm for progress tracking.

        Args:
            X_train: Training data
            X_val: Validation data
            learning_rate: Initial learning rate
            batch_size: Batch size for training
            target_steps: Target number of training steps (default 200k as per paper)
            grad_accum_steps: Number of gradient accumulation steps (default: 1)
            eval_freq: Evaluation frequency in steps (default: 5 evaluations per epoch)
            use_mixed_precision: Whether to use mixed precision training (overrides instance setting)
        """
        # Setup optimizer
        optimizer = optim.Adam(
            self.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0)

        # Track training start time
        start_time = time.time()
        
        # Set mixed precision if overridden in function call
        if use_mixed_precision is not None:
            self.use_mixed_precision = use_mixed_precision
            
        # Initialize mixed precision scaler if requested
        scaler = GradScaler() if self.use_mixed_precision else None

        # Preprocess data - scale so E_x[||x||₂] = √n
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
        self.memory_update_freq = max(1, int(self.total_steps/100))
        
        # Set evaluation frequency if not specified
        if eval_freq is None:
            eval_freq = max(1, steps_per_epoch // 5)  # 5 evaluations per epoch
        
        # Initialize training parameters
        warmup_steps = actual_total_steps // 20  # First 5% for lambda warmup
        decay_start_step = int(actual_total_steps * 0.8)  # Start decay at 80% of training
        step = 0
        best_val_loss = float('inf')
        final_lambda = self.lambda_l1
        val_loss_value = 0.0  # Initialize for progress bar

        # Print training configuration
        self.logger.info("\nTraining Configuration:")
        self.logger.info(f"Total Steps: {actual_total_steps}")
        self.logger.info(f"Epochs: {num_epochs}")
        self.logger.info(f"Steps per Epoch: {steps_per_epoch}")
        self.logger.info(f"Batch Size: {batch_size}")
        self.logger.info(f"Gradient Accumulation Steps: {grad_accum_steps}")
        self.logger.info(f"Effective Batch Size: {batch_size * grad_accum_steps}")
        self.logger.info(f"Warmup Steps: {warmup_steps}")
        self.logger.info(f"Learning Rate Decay Start: {decay_start_step}")
        self.logger.info(f"Evaluation Frequency: {eval_freq} steps")
        self.logger.info(f"Mixed Precision: {self.use_mixed_precision}")
        self.logger.info(f"Using {'direct K-V matrices' if self.use_direct_kv else 'memory bank approach'}")

        # Set up progress tracking with tqdm - matching ST style
        progress_bar = tqdm(
            total=actual_total_steps,
            desc="Training",
            dynamic_ncols=True,
            miniters=20,
            bar_format='{l_bar}{bar:15}{r_bar}'
        )
        
        # Reserve space with initial postfix
        progress_bar.set_postfix({
            'd%': f"{0:.0f}",
            'L': f"{self.lambda_l1:.1f}",
            'lr': f"{learning_rate:.1e}",
            'loss': f"{0.000:.3f}",
            'val': f"{0.000:.3f}"
        })

        # Reset gradients at the beginning of training
        optimizer.zero_grad()
        accum_batch = 0
        running_loss = 0.0

        for epoch in range(num_epochs):
            self.train()
            epoch_train_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                # Get input data
                x = batch[0]
                
                # Apply lambda warmup during initial phase of training
                if step < warmup_steps:
                    self.lambda_l1 = (step / warmup_steps) * final_lambda
                else:
                    self.lambda_l1 = final_lambda
                
                # Apply learning rate decay in last 20%
                if step >= decay_start_step:
                    progress = (step - decay_start_step) / (actual_total_steps - decay_start_step)
                    # Linear decay to zero
                    new_lr = learning_rate * (1 - progress)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr

                # Forward pass with optional mixed precision
                # Handle different PyTorch versions gracefully
                try:
                    # Newer PyTorch version (2.0+)
                    with autocast(self.device if self.device == 'cuda' else 'cpu', 
                                enabled=self.use_mixed_precision):
                        x, x_hat, f_x, v = self.forward(x)
                        total_loss, L2_loss, L1_loss = self.compute_loss(x, x_hat, f_x, v)
                        # Scale loss by the number of accumulation steps for proper scaling
                        scaled_loss = total_loss / grad_accum_steps
                except TypeError:
                    # Older PyTorch version
                    if self.device == 'cuda' and self.use_mixed_precision:
                        with autocast(enabled=self.use_mixed_precision):
                            x, x_hat, f_x, v = self.forward(x)
                            total_loss, L2_loss, L1_loss = self.compute_loss(x, x_hat, f_x, v)
                            # Scale loss by the number of accumulation steps for proper scaling
                            scaled_loss = total_loss / grad_accum_steps
                    else:
                        # CPU or mixed precision disabled
                        x, x_hat, f_x, v = self.forward(x)
                        total_loss, L2_loss, L1_loss = self.compute_loss(x, x_hat, f_x, v)
                        # Scale loss by the number of accumulation steps for proper scaling
                        scaled_loss = total_loss / grad_accum_steps
                
                # Backward pass with appropriate handling for mixed precision
                if self.use_mixed_precision:
                    scaler.scale(scaled_loss).backward()
                    running_loss += scaled_loss.item() * grad_accum_steps  # Track the actual loss
                else:
                    scaled_loss.backward()
                    running_loss += scaled_loss.item() * grad_accum_steps  # Track the actual loss
                
                # Accumulate batches
                accum_batch += 1
                
                # Update weights if we've accumulated enough gradients
                if accum_batch >= grad_accum_steps:
                    if self.use_mixed_precision:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                        optimizer.step()
                    
                    optimizer.zero_grad()
                    accum_batch = 0
                
                # Update feature tracking
                dead_ratio, stats = self.feature_tracker.update(f_x)
                
                # Update training metrics
                epoch_train_loss += total_loss.item()
                num_batches += 1
                step += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'd%': f"{dead_ratio*100:.0f}",
                    'λ': f"{self.lambda_l1:.1f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.1e}",
                    'loss': f"{total_loss.item():.3f}",
                    'val': f"{val_loss_value:.3f}",
                })
                progress_bar.update(1)

                # Periodic validation
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
                    val_loss_value = avg_val_loss  # Store for progress bar

                    # Calculate sparsity
                    sparsity = (f_x.abs() >= self.feature_tracker.activation_threshold).float().mean().item()
                    
                    # Calculate feature norms
                    v_norms = torch.norm(v, p=2, dim=1)
                    avg_feature_norm = v_norms.mean().item()
                    
                    # Calculate elapsed time and estimated time remaining
                    elapsed_time = time.time() - start_time
                    time_per_step = elapsed_time / step
                    remaining_steps = actual_total_steps - step
                    estimated_remaining = remaining_steps * time_per_step
                    
                    # Update training history
                    self.training_history["steps"].append(step)
                    self.training_history["train_loss"].append(total_loss.item())
                    self.training_history["val_loss"].append(avg_val_loss)
                    self.training_history["l1_loss"].append(L1_loss.item())
                    self.training_history["l2_loss"].append(L2_loss.item())
                    self.training_history["lambda"].append(self.lambda_l1)
                    self.training_history["dead_ratio"].append(dead_ratio)
                    self.training_history["sparsity"].append(sparsity)
                    self.training_history["avg_feature_norm"].append(avg_feature_norm)
                    
                    # Log detailed metrics to the progress bar - matching ST style
                    progress_bar.set_postfix({
                        'd%': f"{dead_ratio*100:.0f}",
                        'L': f"{self.lambda_l1:.1f}",
                        'lr': f"{optimizer.param_groups[0]['lr']:.1e}",
                        'loss': f"{total_loss.item():.3f}",
                        'val': f"{avg_val_loss:.3f}"
                    })
                    
                    # Update training history (but don't log it)
                    self.training_history["steps"].append(step)
                    self.training_history["train_loss"].append(total_loss.item())
                    self.training_history["val_loss"].append(avg_val_loss)
                    self.training_history["l1_loss"].append(L1_loss.item())
                    self.training_history["l2_loss"].append(L2_loss.item())
                    self.training_history["lambda"].append(self.lambda_l1)
                    self.training_history["dead_ratio"].append(dead_ratio)
                    self.training_history["sparsity"].append(sparsity)
                    self.training_history["avg_feature_norm"].append(avg_feature_norm)
                    
                    # Periodically clear memory
                    self._cleanup_memory()
                    
                    # Return to training mode
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
                accum_batch = 0

        # Close progress bar
        progress_bar.close()

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
        
        # Training summary - keep simple with minimal logging
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {timedelta(seconds=int(total_time))}")
        
        # Final model save - simple version with just the state dict
        torch.save(self.state_dict(), self.st_model_path)
        self.logger.info(f"Model saved to {self.st_model_path}")
        
        return self