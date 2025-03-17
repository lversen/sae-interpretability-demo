import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional, Dict, List, Any, Union, Callable
from deadfeatures import DeadFeatureTracker
import time
import math
import gc
from datetime import timedelta
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from IPython.display import clear_output, display
import matplotlib
matplotlib.rcParams['figure.max_open_warning'] = 50  # Suppress max figure warnings
matplotlib.use('TkAgg')  # You might need to change this depending on your system
from tqdm.auto import tqdm
import logging
from sklearn.cluster import KMeans

class SparseTransformer(nn.Module):
    """
    SparseTransformer with customizable activation and attention functions.
    """
    
    def __init__(self, X, n: int, m: int, a: int, st_model_path: str,
                 lambda_l1: float = 5.0, num_heads: int = 1, device: str = 'cuda',
                 window_size: int = 10_000_000, update_interval: int = 1_000,
                 activation_threshold: float = 1e-3, use_mixed_precision: bool = True,
                 use_compile: bool = False, memory_strategy: str = 'diversity',
                 log_level: str = 'INFO', use_direct_kv: bool = True,
                 activation: str = 'relu', attention_fn: str = 'softmax'):
        """
        Initialize the Sparse Transformer model.
        
        Args:
            X: Input data tensor or array for the memory bank
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
            memory_strategy: Strategy for memory bank updates ('random', 'diversity', 'kmeans', 'importance')
            log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
            use_direct_kv: Whether to use direct K-V matrices instead of memory bank approach
            activation: Activation function to use in the model
            attention_fn: Function to use for attention score processing
        """
        super().__init__()
        
        # Set up logging
        self.logger = self._setup_logger(log_level)
        
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
        self.use_compile = use_compile
        self.memory_strategy = memory_strategy
        self.use_direct_kv = use_direct_kv  # Flag to control which approach to use
        self.training_history = {"steps": [], "train_loss": [], "val_loss": [], 
                                "l1_loss": [], "l2_loss": [], "lambda": [], 
                                "dead_ratio": [], "sparsity": [], "avg_feature_norm": []}
        
        # Set activation function
        self.activation_name = activation
        self.activation = self._get_activation_function(activation)
        
        # Set attention function
        self.attention_fn_name = attention_fn
        self.attention_fn = self._get_attention_function(attention_fn)
        
        # Projections
        self.W_q = nn.Linear(n, a)  # Always needed
        
        if self.use_direct_kv:
            # Only create direct parameter matrices
            self.W_k_direct = nn.Parameter(torch.Tensor(self.m, self.a))
            self.W_v_direct = nn.Parameter(torch.Tensor(self.m, self.n))
        else:
            # Only create memory-based projection matrices
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
        self.register_buffer('memory_indices', self._initial_memory_selection())
        self.memory_indices_for_update = None
        
        # Initialize weights
        self.initialize_weights()
        self.val = 0.000
        self.to(self.device)
        
        # Apply torch.compile if requested and available (PyTorch 2.0+)
        if self.use_compile and hasattr(torch, 'compile'):
            self.logger.info("Using torch.compile for model optimization")
            self = torch.compile(self)
        else:
            if self.use_compile and not hasattr(torch, 'compile'):
                self.logger.warning("torch.compile requested but not available. Using regular model.")
        
        # Log configuration
        self.logger.info(f"Using {'direct K-V matrices' if self.use_direct_kv else 'memory bank'} approach")
        self.logger.info(f"Using activation function: {self.activation_name}")
        self.logger.info(f"Using attention function: {self.attention_fn_name}")
    
    def _setup_logger(self, log_level: str) -> logging.Logger:
        """Setup logger for the model"""
        import logging
        logger = logging.getLogger(f"SparseTransformer_{id(self)}")
        
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
        
        def length_scaled_softmax(scores, dim=-1):
            """Softmax with temperature scaling based on sequence length"""
            seq_length = scores.size(dim)
            # Scale temperature based on log(N) as suggested in the paper
            temperature = math.sqrt(math.log(seq_length) / math.sqrt(scores.size(-1)))
            return F.softmax(scores / temperature, dim=dim)
        
        def softmax_with_bias(scores, dim=-1):
            """Softmax with a bias term in the denominator (as in the paper)"""
            exp_scores = torch.exp(scores)
            sum_exp = torch.sum(exp_scores, dim=dim, keepdim=True)
            # Add bias term to denominator
            return exp_scores / (1.0 + sum_exp)
        
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
            """
            Simple ReLU attention - applies ReLU and normalizes.
            
            This creates a sparse attention pattern by zeroing out all negative values
            and then normalizing the result.
            
            Args:
                scores: Attention scores
                dim: Dimension to apply attention over
            """
            # Apply ReLU to zero out negative values
            relu_scores = F.relu(scores)
            
            # Normalize to make weights sum to 1
            # Add small epsilon to avoid division by zero
            sum_scores = torch.sum(relu_scores, dim=dim, keepdim=True).clamp(min=1e-6)
            return relu_scores / sum_scores
        def tanh_scale_shift_attention(scores, dim=-1):
            """Attention mechanism using tanh followed by scale and shift"""
            # Apply tanh with alpha=1.0 (can be adjusted as a hyperparameter)
            tanh_scores = torch.tanh(scores)
            
            # Scale and shift (maps from [-1,1] to [0,1])
            transformed_scores = 0.5 * tanh_scores + 0.5
            
            # Normalize to make weights sum to 1
            sum_scores = torch.sum(transformed_scores, dim=dim, keepdim=True).clamp(min=1e-6)
            return transformed_scores / sum_scores
        
        attention_functions = {
            'softmax': lambda x, dim=-1: F.softmax(x, dim=dim),
            'sparsemax': sparsemax,
            'normalized_activation': normalized_activation,
            'direct_activation': direct_activation,
            'relu_softmax': relu_softmax,
            'softmax_hard': lambda x, dim=-1: custom_softmax(x, dim=dim, beta=2.0),
            'softmax_soft': lambda x, dim=-1: custom_softmax(x, dim=dim, beta=0.5),
            'length_scaled_softmax': length_scaled_softmax,
            'softmax_with_bias': softmax_with_bias,
            'polynomial_attention': polynomial_attention,
            'adaptive_sparse': adaptive_sparse_attention,
            'relu_attention': relu_attention,
            'tanh_scale_shift': tanh_scale_shift_attention,
        }
        
        if attention_name.lower() not in attention_functions:
            self.logger.warning(f"Attention function '{attention_name}' not supported. "
                            f"Falling back to 'softmax'. Supported functions: {list(attention_functions.keys())}")
            return attention_functions['softmax']
        
        return attention_functions[attention_name.lower()]
    
    def _initial_memory_selection(self) -> torch.Tensor:
        """
        Initialize memory indices using a strategic selection approach.
        Only used when not using direct K-V matrices.
        """
        # Only needed for original approach
        if self.use_direct_kv:
            return torch.zeros(self.m, dtype=torch.long, device=self.device)
            
        self.logger.info(f"Initializing memory bank with {self.m} indices using strategy: {self.memory_strategy}")
        
        if self.X.shape[0] <= self.m:
            # If we have fewer samples than memory size, use all samples
            self.logger.info(f"Data size ({self.X.shape[0]}) <= memory size ({self.m}). Using all samples.")
            return torch.arange(self.X.shape[0], device=self.device)
        
        # For larger datasets, use a smarter selection strategy
        if self.memory_strategy == 'kmeans' and self.X.shape[0] > 1000:
            try:
                # Use a subset for faster clustering if data is large
                sample_size = min(10000, self.X.shape[0])
                sample_indices = torch.randperm(self.X.shape[0])[:sample_size]
                X_sample = self.X[sample_indices].cpu().numpy()
                
                # Use KMeans for initial selection
                kmeans = KMeans(n_clusters=min(self.m, sample_size), 
                                random_state=42, n_init=1)
                kmeans.fit(X_sample)
                
                # Find closest points to centroids
                distances = kmeans.transform(X_sample)
                closest_indices = [sample_indices[np.argmin(distances[:, i])].item() 
                                  for i in range(min(self.m, sample_size))]
                
                # If we need more indices, add random ones
                if len(closest_indices) < self.m:
                    remaining = self.m - len(closest_indices)
                    additional_indices = torch.randperm(self.X.shape[0])[:remaining]
                    all_indices = closest_indices + additional_indices.tolist()
                else:
                    all_indices = closest_indices
                
                return torch.tensor(all_indices, device=self.device)
            except Exception as e:
                self.logger.warning(f"KMeans initialization failed: {e}. Falling back to random selection.")
                # Fall back to random selection if KMeans fails
                return torch.randperm(self.X.shape[0])[:self.m].to(self.device)
        else:
            # Random selection as fallback
            return torch.randperm(self.X.shape[0])[:self.m].to(self.device)
    
    def type_check(self, x):
        """Ensure input tensors are on the correct device and have the right type"""
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x.astype(np.float32)).to(self.device)
        return x.to(self.device) if x.device != self.device else x

    def scaled_dot_product_attention(self, query, key, value, attn_mask=None, dropout_p=0.0,
                is_causal=False, scale=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Customized scaled dot-product attention with replaceable attention function.
        """
        # Pre-calculate scale factor once
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        
        # Fast path for the common case in ST (no masks, no causal)
        if attn_mask is None and not is_causal:
            # Compute raw attention scores with optimized matrix multiplication
            attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
            
            # Apply custom attention function instead of hardcoded softmax
            attn_weight = self.attention_fn(attn_scores, dim=-1)
            
            # Apply dropout only during training
            if dropout_p > 0.0 and self.training:
                attn_weight = F.dropout(attn_weight, p=dropout_p)
            
            # Compute output with optimized matrix multiplication
            output = torch.matmul(attn_weight, value)
            
            return output, attn_weight, value
        
        # Fallback path for less common cases (masks, causal attention)
        L, S = query.size(-2), key.size(-2)
        
        # Compute attention scores
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
        
        # Apply causal mask if needed (for autoregressive models)
        if is_causal:
            mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
            attn_scores.masked_fill_(~mask, float("-inf"))
        
        # Apply attention mask if provided (for controlling attention patterns)
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(~attn_mask, float("-inf"))
            else:
                attn_scores += attn_mask
        
        # Apply custom attention function and dropout
        attn_weight = self.attention_fn(attn_scores, dim=-1)
        if dropout_p > 0.0 and self.training:
            attn_weight = F.dropout(attn_weight, p=dropout_p)
        
        # Compute output
        output = torch.matmul(attn_weight, value)
        
        return output, attn_weight, value

    def forward(self, x):
        """
        Forward pass for the SparseTransformer model.
        """
        # Update memory indices periodically during training (for original approach)
        if not self.use_direct_kv and self.training and self.steps % self.memory_update_freq == 0:
            # Only update after initial phase and before final phase
            if self.steps > self.total_steps // 20 and self.steps < self.total_steps * 0.8:
                with torch.no_grad():
                    # Use custom indices if available, otherwise use strategy-based update
                    if self.memory_indices_for_update is not None:
                        self.memory_indices = self.memory_indices_for_update
                        self.memory_indices_for_update = None
                    else:
                        self.memory_indices = self.update_memory_indices()
        
        self.steps += 1
        
        # Type conversion for input x
        x = self.type_check(x)  # Shape: [N, n]
        x /= self.preprocess(x)
        
        if self.use_direct_kv:
            # DIRECT K-V APPROACH
            # Project input to query space with activation function
            q_pre = self.W_q(x)
            q_act = self.activation(q_pre)  # Apply activation function
            q = self.norm_q(q_act)  # Shape: [N, a]
            
            # For direct K-V, we can apply activation to the raw parameters
            k_act = self.activation(self.W_k_direct)  # Apply activation function
            k = self.norm_k(k_act)  # Shape: [m, a]
            
            v_act = self.activation(self.W_v_direct)  # Apply activation function
            v = self.norm_v(v_act)  # Shape: [m, n]
        else:
            # ORIGINAL APPROACH WITH MEMORY BANK
            # Get cross attention context
            X_cross = self.X[self.memory_indices]  # Shape: [m, n]
            C = self.preprocess(X_cross)
            X_cross = X_cross / C  # Use new tensor to avoid modifying self.X
            
            # Project to attention space with activation functions
            q_pre = self.W_q(x)
            q_act = self.activation(q_pre)  # Apply activation function
            q = self.norm_q(q_act)  # Shape: [N, a]
            
            k_pre = self.W_k(X_cross)
            k_act = self.activation(k_pre)  # Apply activation function
            k = self.norm_k(k_act)  # Shape: [m, a]
            
            v_pre = self.W_v(X_cross)
            v_act = self.activation(v_pre)  # Apply activation function
            v = self.norm_v(v_act)  # Shape: [m, n]
        
        # Use PyTorch's native attention if available and using standard softmax
        if hasattr(F, 'scaled_dot_product_attention') and self.attention_fn_name == 'softmax':
            # Reshape tensors for batch attention
            q_reshaped = q.unsqueeze(1)  # [N, 1, a]
            k_reshaped = k.unsqueeze(0)  # [1, m, a]
            v_reshaped = v.unsqueeze(0)  # [1, m, n]
            
            # Compute attention with native function
            x_hat = F.scaled_dot_product_attention(
                q_reshaped, k_reshaped, v_reshaped,
                scale=1.0 / math.sqrt(self.a),
                dropout_p=0.0
            ).squeeze(1)  # [N, n]
            
            # Compute feature activations (attention weights)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.a)
            f = F.softmax(attn_scores, dim=-1)
        else:
            # Fall back to custom implementation with custom attention function
            x_hat, f, _ = self.scaled_dot_product_attention(q, k, v, dropout_p=0)
            
        if self.training:
            self.feature_tracker.update(f)
        
        return x, x_hat, f, v

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
        # Input validation
        if x.shape != x_hat.shape:
            raise ValueError(f"Input and reconstruction shapes don't match: {x.shape} vs {x_hat.shape}")
        if f.shape[0] != x.shape[0] or f.shape[1] != self.m:
            raise ValueError(f"Feature activation shape incorrect: {f.shape}, expected [{x.shape[0]}, {self.m}]")
            
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
        with torch.no_grad():
            # Initialize query projection (always used)
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
    
    def update_memory_indices(self, f_x=None):
        """
        Update memory indices using the selected strategy.
        Only used for the original approach.
        """
        # Only relevant for original approach
        if self.use_direct_kv:
            return torch.zeros(self.m, dtype=torch.long, device=self.device)
            
        # If we're in the first 10% of training, use random selection regardless
        if self.steps < self.total_steps * 0.1:
            return torch.randint(0, self.X.shape[0], (self.m,), device=self.device)
            
        # Apply the selected strategy
        if self.memory_strategy == 'diversity' and f_x is not None:
            # Select indices with highest feature diversity
            f_binary = (f_x > self.activation_threshold).float()
            feature_diversity = f_binary.sum(dim=1)  # Count active features per sample
            _, top_indices = torch.topk(feature_diversity, min(self.m, len(feature_diversity)))
            return top_indices
            
        elif self.memory_strategy == 'kmeans':
            try:
                # Use a subset for faster clustering if data is large
                sample_size = min(10000, self.X.shape[0])
                sample_indices = torch.randperm(self.X.shape[0])[:sample_size]
                X_sample = self.X[sample_indices].cpu().numpy()
                
                # Use KMeans
                kmeans = KMeans(n_clusters=min(self.m, sample_size), random_state=42, n_init=1)
                kmeans.fit(X_sample)
                
                # Find closest points to centroids
                distances = kmeans.transform(X_sample)
                closest_indices = [sample_indices[np.argmin(distances[:, i])].item() 
                                  for i in range(min(self.m, sample_size))]
                
                # If we need more indices, add random ones
                if len(closest_indices) < self.m:
                    remaining = self.m - len(closest_indices)
                    additional_indices = torch.randperm(self.X.shape[0])[:remaining]
                    all_indices = closest_indices + additional_indices.tolist()
                else:
                    all_indices = closest_indices
                
                return torch.tensor(all_indices, device=self.device)
            except Exception as e:
                self.logger.warning(f"KMeans memory update failed: {e}. Falling back to random selection.")
                return torch.randint(0, self.X.shape[0], (self.m,), device=self.device)
                
        elif self.memory_strategy == 'importance' and f_x is not None:
            # Select samples based on importance (reconstruction error)
            # To approximate this, we can use samples that have high attention dispersion
            entropy = -torch.sum(f_x * torch.log(f_x + 1e-10), dim=1)
            _, top_indices = torch.topk(entropy, min(self.m, len(entropy)))
            return top_indices
            
        else:
            # Default to random selection
            return torch.randint(0, self.X.shape[0], (self.m,), device=self.device)
    
    def _cleanup_memory(self):
        """Release memory when possible"""
        if torch.cuda.is_available():
            # Force garbage collection
            gc.collect()
            torch.cuda.empty_cache()
    
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
        
        # Calculate feature norms
        X_cross = self.X[self.memory_indices]  # Shape: [m, n]
        with torch.no_grad():
            v = self.norm_v(self.W_v(X_cross))  # Shape: [m, n]
            v_norms = torch.norm(v, p=2, dim=1).cpu().numpy()
        
        # Enhanced stats - feature stability and coverage
        feature_stability = 1.0 - torch.std(all_activations, dim=0) / (mean_activations + 1e-8)
        feature_stability = feature_stability.numpy()
        
        return {
            'mean_activations': mean_activations.numpy(),
            'max_activations': max_activations.numpy(),
            'activation_frequency': avg_feature_activity,
            'top_active_features': top_indices,
            'coactivation_matrix': coactivation.numpy(),
            'sparsity': sparsity,
            'feature_norms': v_norms,
            'feature_stability': feature_stability,
            'total_samples': all_activations.shape[0]
        }
    
    def analyze_feature_importance(self, data_loader, top_n=10):
        """
        Analyze which features are most important for the model
        
        Args:
            data_loader: DataLoader with samples to analyze
            top_n: Number of top features to return
            
        Returns:
            Dictionary with feature importance metrics
        """
        self.eval()
        all_activations = []
        
        with torch.no_grad():
            for batch in data_loader:
                x = batch[0].to(self.device)
                _, _, f_x, _ = self.forward(x)
                all_activations.append(f_x.cpu())
        
        # Concatenate all batches
        activations = torch.cat(all_activations, dim=0)
        
        # Calculate average activation per feature
        avg_activation = activations.mean(dim=0)
        
        # Find top features
        top_values, top_indices = torch.topk(avg_activation, top_n)
        
        # Calculate L1 sparsity
        sparsity_per_feature = (activations > self.activation_threshold).float().mean(dim=0)
        
        # Calculate entropy (lower is more selective/specialized)
        binary_activations = (activations > self.activation_threshold).float()
        entropy = torch.zeros_like(avg_activation)
        for i in range(activations.shape[1]):
            # Count co-occurrences with other features
            co_occur = torch.mm(binary_activations[:, i:i+1].t(), binary_activations)
            co_occur = co_occur / (binary_activations[:, i].sum() + 1e-8)
            # Calculate entropy of co-occurrence distribution
            entropy[i] = -torch.sum(co_occur * torch.log(co_occur + 1e-8))
            
        return {
            'top_indices': top_indices.numpy(),
            'top_values': top_values.numpy(),
            'avg_activation': avg_activation.numpy(),
            'activation_frequency': sparsity_per_feature.numpy(),
            'feature_entropy': entropy.numpy()
        }
        
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
            f"Memory strategy: {self.memory_strategy}\n"
            f"Training steps: {self.steps}\n"
            f"Mixed precision: {self.use_mixed_precision}\n"
            f"Dead features: {self.feature_tracker.get_dead_features().sum().item()} / {self.m}"
        )
        ax.text(0.5, 0.5, settings_text, 
                ha='center', va='center', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        fig.suptitle(f"Training History - Sparse Transformer", fontsize=16)
        plt.subplots_adjust(top=0.93)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"Training history plot saved to {save_path}")
            
        return fig
    
    def early_stopping_check(self, val_loss, patience, min_delta=0.001):
        """
        Check if early stopping criteria are met
        
        Args:
            val_loss: Current validation loss
            patience: Number of consecutive checks without improvement before stopping
            min_delta: Minimum change in validation loss to be considered an improvement
            
        Returns:
            Boolean indicating whether to stop training
        """
        # We need at least patience+1 validation points to check
        if len(self.training_history["val_loss"]) <= patience:
            return False
            
        # Get recent validation losses
        recent_losses = self.training_history["val_loss"][-(patience+1):]
        best_loss = min(recent_losses[:-1])  # Best loss before current one
        
        # Check if current loss is not better than best by min_delta
        if val_loss > best_loss - min_delta:
            # No improvement for 'patience' steps
            return True
            
        return False




    def plot_decoder_weights_during_training(self, input_shape=(28, 28), 
                                    num_weights=16, rows=4, cols=4, figsize=(12, 12),
                                    cmap='coolwarm', save_path=None, display_plot=True,
                                    block=False):
        """
        Plot decoder weights during training with FORCED display
        
        Args:
            input_shape: Shape to reshape weights to (e.g., (28, 28) for MNIST)
            num_weights: Number of weights to visualize
            rows, cols: Number of rows and columns in the grid
            figsize: Figure size (width, height) in inches
            cmap: Colormap to use for visualization
            save_path: Optional path to save the figure
            display_plot: Whether to display the plot
            block: Whether to block execution until plot window is closed
        
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Extract decoder weights
        self.eval()  # Switch to eval mode
        
        with torch.no_grad():
            # Get a sample batch from memory bank
            X_cross = self.X[self.memory_indices[:50]]  # Just use a subset
            
            # Get value projection weights
            v = self.norm_v(self.W_v(X_cross))  # Shape: [subset_of_m, n]
            
            # Convert to numpy for plotting
            weight_matrix = v.cpu().numpy()
        
        # Calculate L2 norms to identify important features
        feature_norms = np.linalg.norm(weight_matrix, axis=1)
        sorted_indices = np.argsort(-feature_norms)  # Sort by descending norm
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        fig.suptitle(f"ST Model: Value Projection Weights (Step {self.steps})", fontsize=16)
        
        # Plot each weight vector
        for i in range(min(rows * cols, min(num_weights, len(sorted_indices)))):
            # Get the index of the i-th highest norm weight
            idx = sorted_indices[i]
            
            # Get the weight vector
            weight = weight_matrix[idx]
            
            # Calculate norm for title
            norm = feature_norms[idx]
            
            # Reshape to the input shape
            weight_img = weight.reshape(input_shape)
            
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
        
        # FORCE the display to show the plot
        if display_plot:
            # Draw the figure
            fig.canvas.draw()
            # Display but don't block training
            plt.show(block=block)
            # Give matplotlib time to display the figure
            plt.pause(0.1)
        
        return fig

    # To ensure training continues after showing plots, add this wrapper function
    def train_with_visualizations(self, **train_params):
        """
        Wrapper for train_and_validate that ensures matplotlib is properly configured for live plotting
        
        Args:
            **train_params: Parameters to pass to train_and_validate
            
        Returns:
            The trained model
        """
        # Set up matplotlib for interactive plotting
        plt.ion()  # Turn on interactive mode
        
        # Call the normal training function
        result = self.train_and_validate(**train_params)
        
        # Turn off interactive mode when done
        plt.ioff()
        
        return result

    def train_and_validate(self, X_train, X_val, learning_rate=5e-5, batch_size=4096, 
                            target_steps=200_000, checkpoint_freq=50000, save_best=False, 
                            enable_checkpoints=True, save_full_state=True, resume_from=None,
                            grad_accum_steps=1, eval_freq=None, scheduler_type=None,
                            early_stopping=False, early_stopping_patience=5, 
                            warmup_steps_pct=0.05, final_decay_pct=0.2,
                            plot_weights_freq=0, plot_input_shape=(28, 28),
                            plot_save_dir=None):
        """
        Enhanced training method with more options and better monitoring.
        
        Args:
            X_train: Training data tensor [samples, features]
            X_val: Validation data tensor [samples, features]
            learning_rate: Initial learning rate (default: 5e-5)
            batch_size: Batch size for training (default: 4096)
            target_steps: Target number of training steps (default: 200,000)
            checkpoint_freq: How often to save checkpoints (default: 50000 steps)
            save_best: Whether to save the best model based on validation loss (default: True)
            enable_checkpoints: Whether to save periodic checkpoints (default: True)
            save_full_state: Whether to save full state in checkpoints (default: True)
            resume_from: Optional checkpoint path to resume training from
            grad_accum_steps: Number of gradient accumulation steps (default: 1)
            eval_freq: How often to evaluate on validation set (if None, use 5 times per epoch)
            scheduler_type: Type of learning rate scheduler ('cosine', 'linear', 'constant')
            early_stopping: Whether to enable early stopping (default: True)
            early_stopping_patience: Patience for early stopping (default: 5 evaluations)
            warmup_steps_pct: Percentage of steps for lambda and LR warmup (default: 0.05)
            final_decay_pct: Percentage of steps for final decay phase (default: 0.2)
            plot_weights_freq: How often to plot decoder weights (0 to disable)
            plot_input_shape: Shape to reshape weights to (e.g., (28, 28) for MNIST)
            plot_save_dir: Directory to save weight plots
        
        Returns:
            self: The trained model
        """
        # Initialize optimizer
        optimizer = optim.Adam(
            self.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0)
        
        # Initialize mixed precision scaler if requested
        scaler = GradScaler() if self.use_mixed_precision else None
        
        # Initialize scheduler if requested
        scheduler = None
        if scheduler_type == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=1000,                 # Initial restart period
                T_mult=2,                 # Double period after each restart
                eta_min=learning_rate / 100  # Minimum learning rate
            )
        elif scheduler_type == 'onecycle':
            from torch.optim.lr_scheduler import OneCycleLR
            scheduler = OneCycleLR(
                optimizer,
                max_lr=learning_rate * 2,
                total_steps=target_steps,
                pct_start=warmup_steps_pct,
                div_factor=25,
                final_div_factor=100
            )
        
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
        pin_memory = torch.cuda.is_available() and X_train.device.type == 'cpu'
            
        # Configure number of workers based on system
        num_workers = 0  # Default safe value, can be increased based on system resources
            
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            pin_memory=pin_memory, num_workers=num_workers)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size * 2, shuffle=False,  # Larger batch for validation
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
        warmup_steps = int(actual_total_steps * warmup_steps_pct)  # First N% for lambda warmup
        decay_start_step = int(actual_total_steps * (1 - final_decay_pct))  # Start decay at (1-N)% of training
        step = 0
        best_val_loss = float('inf')
        final_lambda = self.lambda_l1
        
        # Set up directory for weight plots if enabled
        if plot_weights_freq > 0 and plot_save_dir:
            os.makedirs(plot_save_dir, exist_ok=True)
            self.logger.info(f"Weight plots will be saved to {plot_save_dir} every {plot_weights_freq} steps")
        
        # Resume from checkpoint if provided
        if resume_from and os.path.exists(resume_from):
            try:
                self.logger.info(f"Resuming training from checkpoint: {resume_from}")
                checkpoint = torch.load(resume_from, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    self.load_state_dict(checkpoint['model_state_dict'])
                    if 'optimizer_state_dict' in checkpoint and save_full_state:
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    if 'step' in checkpoint:
                        step = checkpoint['step']
                    if 'training_history' in checkpoint:
                        self.training_history = checkpoint['training_history']
                    if 'feature_tracker' in checkpoint:
                        self.feature_tracker = checkpoint['feature_tracker']
                    if 'lambda_l1' in checkpoint:
                        self.lambda_l1 = checkpoint['lambda_l1']
                    self.logger.info(f"Successfully resumed from step {step}")
                else:
                    # Just a state dict, not a full checkpoint
                    self.load_state_dict(checkpoint)
                    self.logger.info("Loaded model weights only, not training state")
            except Exception as e:
                self.logger.error(f"Error loading checkpoint: {e}")
                self.logger.info("Starting training from scratch instead")

        # Prepare a more concise configuration display
        config_summary = (
            f"\n{'='*60}\n"
            f"ST TRAINING CONFIG\n"
            f"{'='*60}\n"
            f"Dimensions: n={self.n}, m={self.m}, a={self.a}\n"
            f"Steps: {actual_total_steps:,} | Batch: {batch_size:,} x {grad_accum_steps} = {batch_size * grad_accum_steps:,}\n"
            f"LR: {learning_rate:.1e} | λ: {final_lambda:.2f} | Scheduler: {scheduler_type}\n"
            f"Memory strategy: {self.memory_strategy} | Update freq: {self.memory_update_freq}\n"
            f"Features: AMP={'✓' if self.use_mixed_precision else '✗'} | Early stopping={'✓' if early_stopping else '✗'} | Compile={'✓' if self.use_compile else '✗'}\n"
            f"Device: {self.device}"
        )
        
        self.logger.info(config_summary)

        # Set up progress tracking with tqdm
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
            'λ': f"{self.lambda_l1:.1f}",
            'lr': f"{learning_rate:.1e}",
            'loss': f"{0.000:.3f}",
            'val': f"{0.000:.3f}"
        })
        progress_bar.update(step)  # Update with existing steps if resuming

        # Reset gradients at the beginning of training
        optimizer.zero_grad()
        accum_batch = 0
        running_loss = 0.0
        early_stop = False
        
        for epoch in range(num_epochs):
            if early_stop:
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break
                
            self.train()

            for batch_idx, batch in enumerate(train_loader):
                # Skip steps if resuming
                if step >= actual_total_steps:
                    break
                    
                # Apply lambda warmup during initial phase of training
                if step < warmup_steps:
                    self.lambda_l1 = (step / warmup_steps) * final_lambda
                else:
                    self.lambda_l1 = final_lambda

                # Apply manual learning rate decay in last 20% if not using scheduler
                if scheduler is None and step >= decay_start_step:
                    progress = (step - decay_start_step) / (actual_total_steps - decay_start_step)
                    new_lr = learning_rate * (1 - progress)  # Linear decay to zero
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                
                x = batch[0]
                
                # Forward pass with optional mixed precision
                with autocast(device_type=self.device.split(':')[0], enabled=self.use_mixed_precision):
                    x, x_hat, f_x, v = self.forward(x)
                    total_loss, L2_loss, L1_loss = self.compute_loss(x, x_hat, f_x, v)
                    # Scale loss by the number of accumulation steps for proper scaling
                    total_loss = total_loss / grad_accum_steps
                
                # Backward pass with appropriate handling for mixed precision
                if self.use_mixed_precision:
                    scaler.scale(total_loss).backward()
                    running_loss += total_loss.item() * grad_accum_steps  # Track the actual loss
                else:
                    total_loss.backward()
                    running_loss += total_loss.item() * grad_accum_steps  # Track the actual loss
                
                # Update weights if we've accumulated enough gradients
                accum_batch += 1
                if accum_batch >= grad_accum_steps:
                    if self.use_mixed_precision:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                        optimizer.step()
                        
                    # Step the scheduler if enabled
                    if scheduler is not None:
                        scheduler.step()
                        
                    optimizer.zero_grad()
                    accum_batch = 0
                
                # Update feature tracking
                dead_ratio, stats = self.feature_tracker.update(f_x)

                # Update step counter and progress bar
                step += 1
                # Use consistent keys with last known validation value
                progress_bar.set_postfix({
                    'd%': f"{dead_ratio*100:.0f}",
                    'λ': f"{self.lambda_l1:.1f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.1e}",
                    'loss': f"{total_loss.item()*grad_accum_steps:.3f}",
                    'val': f"{self.val:.3f}"
                })
                progress_bar.update(1)
                
                # Plot decoder weights periodically if enabled
                if plot_weights_freq > 0 and step % plot_weights_freq == 0:
                    try:
                        # Create a custom save path if directory is provided
                        save_path = None
                        if plot_save_dir:
                            save_path = os.path.join(plot_save_dir, f"weights_step_{step:06d}.png")
                            
                        # Run the plotting function
                        self.plot_decoder_weights_during_training(
                            input_shape=plot_input_shape,
                            num_weights=16,  # A reasonable number for visualization
                            save_path=save_path,
                            display_plot=True  # Try to display the plot
                        )
                    except Exception as e:
                        self.logger.warning(f"Error plotting decoder weights: {e}")

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
                    self.val = avg_val_loss

                    # Calculate sparsity and feature norms
                    sparsity = (f_x.abs() >= self.feature_tracker.activation_threshold).float().mean().item()
                    v_norms = torch.norm(v, p=2, dim=1)
                    avg_feature_norm = v_norms.mean().item()
                    
                    # Calculate elapsed time and estimated time remaining
                    elapsed_time = time.time() - start_time
                    time_per_step = elapsed_time / step
                    remaining_steps = actual_total_steps - step
                    estimated_remaining = remaining_steps * time_per_step
                    
                    # Update progress bar with compact validation info
                    progress_bar.set_postfix({
                        'd%': f"{dead_ratio*100:.0f}",
                        'λ': f"{self.lambda_l1:.1f}",
                        'lr': f"{optimizer.param_groups[0]['lr']:.1e}",
                        'loss': f"{total_loss.item()*grad_accum_steps:.3f}",
                        'val': f"{avg_val_loss:.3f}",
                    })

                    # Update training history
                    self.training_history["steps"].append(step)
                    self.training_history["train_loss"].append(total_loss.item()*grad_accum_steps)
                    self.training_history["val_loss"].append(avg_val_loss)
                    self.training_history["l1_loss"].append(L1_loss.item()*grad_accum_steps)
                    self.training_history["l2_loss"].append(L2_loss.item()*grad_accum_steps)
                    self.training_history["lambda"].append(self.lambda_l1)
                    self.training_history["dead_ratio"].append(dead_ratio)
                    self.training_history["sparsity"].append(sparsity)
                    self.training_history["avg_feature_norm"].append(avg_feature_norm)
                    
                    # Create checkpoint for saving
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
                        self.logger.info(f"New best model saved (val_loss: {avg_val_loss:.4f})")

                    # Save checkpoint periodically (with restrictions to minimize overhead)
                    if enable_checkpoints and step % checkpoint_freq == 0:
                        # Don't save if we're in first 10% of training or close to the end
                        remaining_steps = actual_total_steps - step
                        if step > actual_total_steps * 0.1 and remaining_steps > checkpoint_freq // 2:
                            checkpoint_path = f"{self.st_model_path}.step{step}"
                            torch.save(checkpoint, checkpoint_path)
                            self.logger.info(f"Checkpoint saved at step {step}")
                    
                    # Check early stopping
                    if early_stopping and step > actual_total_steps * 0.5:  # Only check after half of training
                        early_stop = self.early_stopping_check(
                            avg_val_loss, early_stopping_patience)
                        if early_stop:
                            self.logger.info(f"Early stopping triggered at step {step}")
                            break
                    
                    # Periodically clear memory
                    self._cleanup_memory()
                    
                    self.train()
            
            if early_stop:
                break

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
        
        # Training summary (concise version)
        total_time = time.time() - start_time
        summary = (
            f"\n{'='*60}\n"
            f"TRAINING COMPLETE - {timedelta(seconds=int(total_time))}\n"
            f"Val loss: {final_val_loss:.4f} (best: {best_val_loss:.4f})\n"
            f"Dead features: {dead_ratio:.1%} | Steps: {step:,}/{actual_total_steps:,}\n"
            f"Final λ: {self.lambda_l1:.2f}"
        )
        self.logger.info(summary)
        
        # Save final model
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
        self.logger.info(f"Final model saved to {self.st_model_path}")
        
        # Plot training history if available
        if len(self.training_history["steps"]) > 5:  # Only if we have enough data points
            try:
                history_path = f"{os.path.splitext(self.st_model_path)[0]}_history.png"
                self.plot_training_history(save_path=history_path)
                self.logger.info(f"Training history plot saved to {history_path}")
            except Exception as e:
                self.logger.error(f"Could not generate training history plots: {e}")
        
        return self
    
    def estimate_optimal_hyperparameters(self, X_sample, test_steps=100, batch_size=256,
                                        learning_rates=[1e-5, 5e-5, 1e-4],
                                        lambda_values=[1.0, 5.0, 10.0],
                                        warmup_pct=0.1):
        """
        Estimate optimal hyperparameters by running short test runs
        
        Args:
            X_sample: Sample data for testing
            test_steps: Number of steps to run for each test
            batch_size: Batch size to use
            learning_rates: List of learning rates to try
            lambda_values: List of lambda values to try
            warmup_pct: Percentage of steps for lambda warmup
            
        Returns:
            Dictionary with recommended hyperparameters
        """
        self.logger.info("Estimating optimal hyperparameters...")
        
        # Split data into train/val
        sample_size = min(1000, len(X_sample))
        X_sample = X_sample[:sample_size]
        X_train, X_val = torch.split(X_sample, [int(0.8*sample_size), sample_size - int(0.8*sample_size)])
        
        # Preprocess data
        C = self.preprocess(X_train)
        X_train = X_train / C
        X_val = X_val / C
        
        # Setup data loaders
        train_dataset = TensorDataset(X_train)
        val_dataset = TensorDataset(X_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Store results
        results = []
        
        # Test different hyperparameter combinations
        for lr in learning_rates:
            for lambda_val in lambda_values:
                self.logger.info(f"Testing LR={lr}, λ={lambda_val}")
                
                # Reset model weights
                self.initialize_weights()
                self.lambda_l1 = lambda_val
                
                # Initialize optimizer
                optimizer = optim.Adam(self.parameters(), lr=lr)
                
                # Training loop
                self.train()
                train_losses = []
                
                for step in range(test_steps):
                    # Get batch (with wrapping)
                    batch_idx = step % len(train_loader)
                    if batch_idx == 0:
                        train_iter = iter(train_loader)
                    
                    x = next(train_iter)[0]
                    
                    # Apply lambda warmup
                    if step < test_steps * warmup_pct:
                        self.lambda_l1 = (step / (test_steps * warmup_pct)) * lambda_val
                    else:
                        self.lambda_l1 = lambda_val
                    
                    # Forward and backward pass
                    x, x_hat, f_x, v = self.forward(x)
                    total_loss, _, _ = self.compute_loss(x, x_hat, f_x, v)
                    
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                    
                    train_losses.append(total_loss.item())
                
                # Validation
                self.eval()
                val_loss = 0.0
                val_batches = 0
                
                with torch.no_grad():
                    for val_batch in val_loader:
                        x_val = val_batch[0]
                        x_val, x_hat_val, f_x_val, v_val = self.forward(x_val)
                        val_total_loss, _, _ = self.compute_loss(x_val, x_hat_val, f_x_val, v_val)
                        val_loss += val_total_loss.item()
                        val_batches += 1
                
                avg_val_loss = val_loss / val_batches
                
                # Calculate sparsity
                with torch.no_grad():
                    sparsity = (f_x_val.abs() >= self.activation_threshold).float().mean().item()
                
                # Store results
                results.append({
                    'lr': lr,
                    'lambda': lambda_val,
                    'train_loss': np.mean(train_losses[-10:]),  # Average of last 10 steps
                    'val_loss': avg_val_loss,
                    'sparsity': sparsity,
                    'loss_decrease': train_losses[0] - np.mean(train_losses[-10:])
                })
                
                self.logger.info(f"Results: train_loss={results[-1]['train_loss']:.4f}, "
                               f"val_loss={avg_val_loss:.4f}, sparsity={sparsity:.2%}")
        
        # Find best hyperparameters
        # Sort by validation loss
        results.sort(key=lambda x: x['val_loss'])
        best_config = results[0]
        
        # Print recommendation
        self.logger.info("\nHyperparameter Recommendation:")
        self.logger.info(f"Learning Rate: {best_config['lr']}")
        self.logger.info(f"Lambda (L1): {best_config['lambda']}")
        self.logger.info(f"Expected validation loss: {best_config['val_loss']:.4f}")
        self.logger.info(f"Expected sparsity: {best_config['sparsity']:.2%}")
        
        # Reset model to initial state
        self.initialize_weights()
        
        return {
            'learning_rate': best_config['lr'],
            'lambda': best_config['lambda'],
            'all_results': results
        }
    
    def set_direct_kv_mode(self, use_direct_kv: bool):
        # Check if we're actually changing the mode
        if self.use_direct_kv == use_direct_kv:
            return
            
        self.use_direct_kv = use_direct_kv
        
        # Remove old parameters and create new ones
        if self.use_direct_kv:
            # Remove memory approach parameters
            if hasattr(self, 'W_k'):
                delattr(self, 'W_k')
            if hasattr(self, 'W_v'):
                delattr(self, 'W_v')
                
            # Create direct K-V parameters
            self.W_k_direct = nn.Parameter(torch.Tensor(self.m, self.a))
            self.W_v_direct = nn.Parameter(torch.Tensor(self.m, self.n))
            
            # Initialize new parameters
            with torch.no_grad():
                nn.init.normal_(self.W_k_direct, mean=0.0, std=0.02)
                nn.init.kaiming_uniform_(self.W_v_direct, a=math.sqrt(5))
        else:
            # Remove direct K-V parameters
            if hasattr(self, 'W_k_direct'):
                delattr(self, 'W_k_direct')
            if hasattr(self, 'W_v_direct'):
                delattr(self, 'W_v_direct')
                
            # Create memory approach parameters
            self.W_k = nn.Linear(self.n, self.a)
            self.W_v = nn.Linear(self.n, self.n)
            
            # Initialize new parameters
            with torch.no_grad():
                nn.init.normal_(self.W_k.weight, mean=0.0, std=0.02)
                nn.init.kaiming_uniform_(self.W_v.weight, a=math.sqrt(5))
        
        self.logger.warning("Model parameters have been reset! You will need to retrain.")