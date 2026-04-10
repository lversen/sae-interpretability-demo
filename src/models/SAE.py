import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Union, Tuple, Optional, Dict, Any
from deadfeatures import DeadFeatureTracker
import sys
import time
import os
import gc
import re
import glob
import math
import logging
from datetime import timedelta
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from tqdm.auto import tqdm

# Mixed precision imports
from torch.amp import autocast
from torch.cuda.amp import GradScaler

class SparseAutoencoder(nn.Module):
    def __init__(self, n: int, m: int, sae_model_path: str, lambda_l1: float = 5, device: str = 'cuda', 
                 activation: str = 'relu', window_size: int = 10_000_000, update_interval: int = 10_000, 
                 activation_threshold: float = 1e-3, use_mixed_precision: bool = True, log_level: str = 'INFO',
                 use_compile: bool = False):
        super(SparseAutoencoder, self).__init__()
        self.n = n
        self.m = m
        self.sae_model_path = sae_model_path
        self.lambda_l1 = lambda_l1
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.use_mixed_precision = use_mixed_precision
        self.use_compile = use_compile
        
        # Logger setup
        self.logger = self._setup_logger(log_level)
        self.logger.info(f"Initializing EnhancedSparseAutoencoder with n={n}, m={m}")

        # Set activation function
        self.activation_name = activation
        self.activation = self._get_activation_function(activation)

        # Initialize components
        self.W_e = nn.Linear(n, m, bias=False)
        self.W_d = nn.Linear(m, n, bias=False)
        self.b_e = nn.Parameter(torch.zeros(m))
        self.b_d = nn.Parameter(torch.zeros(n))
        
        # Initialize feature tracker
        self.feature_tracker = DeadFeatureTracker(
            num_features=m,
            window_size=window_size,
            update_interval=update_interval,
            activation_threshold=activation_threshold
        )

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

        self.initialize_weights()
        self.to(self.device)
        
        # Apply torch.compile if requested and available (PyTorch 2.0+)
        if self.use_compile:
            if hasattr(torch, 'compile'):
                self.logger.info("Using torch.compile to optimize model execution")
                self = torch.compile(self)
            else:
                self.logger.warning("torch.compile requested but not available. Using regular model.")
                self.use_compile = False
                
        self.logger.info(f"Model initialized on {self.device}")

    def _setup_logger(self, log_level: str) -> logging.Logger:
        """Setup logger for the model"""
        logger = logging.getLogger(f"EnhancedSAE_{id(self)}")
        
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

    def initialize_weights(self):
        """
        Initialize weights according to updated SAE training configuration:
        - W_d columns have random directions with L2 norms between 0.05 and 1
        - W_e is initialized as W_d^T
        - Biases are initialized to zeros
        """
        with torch.no_grad():
            # Create random weight matrix for decoder
            W_d = torch.randn(self.n, self.m)
            
            # Normalize columns to have unit norm
            norms = torch.norm(W_d, p=2, dim=0)
            W_d = W_d / norms
            
            # Scale columns to have norms between 0.05 and 1
            target_norms = 0.05 + 0.95 * torch.rand(self.m)
            W_d = W_d * target_norms
            
            # Assign to model parameters
            self.W_d.weight.data = W_d
            self.W_e.weight.data = W_d.t()  # W_e = W_d^T
            
            # Biases initialized to zeros (already done in __init__)
        
        self.logger.info("Model weights initialized")

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32)).to(self.device)
        elif isinstance(x, torch.Tensor) and x.device != self.device:
            x = x.to(self.device)

        # Use the selected activation function
        f_x = self.activation(self.W_e(x) + self.b_e)
        x_hat = self.W_d(f_x) + self.b_d
        
        return x, x_hat, f_x

    def compute_loss(self, x, x_hat, f_x):
        """
        Compute loss according to the paper's specification:
        L = (1/|X|) * Σ ||x - x̂||₂² + λ * Σᵢ |fᵢ(x)| ||Wdᵢ||₂
        """
        # L2 reconstruction loss
        L2_loss = torch.mean(torch.norm(x - x_hat, p=2, dim=1)**2)
        
        # Get L2 norms of decoder weight columns
        W_d_norms = torch.norm(self.W_d.weight, p=2, dim=0)
        
        # Sparsity penalty: λ * Σᵢ |fᵢ(x)| ||Wdᵢ||₂
        L1_penalty = self.lambda_l1 * torch.mean(torch.sum(f_x * W_d_norms, dim=1))
        
        total_loss = L2_loss + L1_penalty
        
        return total_loss, L2_loss, L1_penalty

    def feature_activations(self, x):
        """Calculate feature activations with L2 norm weighting"""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32)).to(self.device)
        elif isinstance(x, torch.Tensor) and x.device != self.device:
            x = x.to(self.device)

        # Use the selected activation function
        f_x = self.activation(self.W_e(x) + self.b_e)
        return f_x * torch.norm(self.W_d.weight, p=2, dim=0)

    def resume_from_checkpoint(self, checkpoint_path):
        """Resume training from a checkpoint"""
        self.logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Check if it's a full checkpoint or just model state
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.load_state_dict(checkpoint['model_state_dict'])
                current_step = checkpoint.get('step', 0)
                
                # Load training history if available
                if 'training_history' in checkpoint:
                    self.training_history = checkpoint['training_history']
                    self.logger.info(f"Loaded training history with {len(self.training_history['steps'])} steps")
                
                if 'lambda_l1' in checkpoint:
                    self.lambda_l1 = checkpoint['lambda_l1']
                    
                self.logger.info(f"Resumed from step {current_step} with lambda_l1 = {self.lambda_l1}")
            else:
                # Just a state dict
                self.load_state_dict(checkpoint)
                self.logger.info("Loaded model weights only")
            
            return True
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")
            return False

    def preprocess(self, X):
        """
        Scale dataset so that E_x[||x||₂] = √n as specified
        """
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X.astype(np.float32)).to(self.device)
        elif isinstance(X, torch.Tensor) and X.device != self.device:
            X = X.to(self.device)

        # Calculate scaling factor C so that E_x[||x||₂] = √n after dividing by C
        mean_norm = torch.mean(torch.norm(X, p=2, dim=1))
        C = mean_norm / np.sqrt(self.n)
        
        return C

    def feature_vectors(self):
        """Return normalized feature vectors (decoder columns)"""
        return self.W_d.weight / torch.norm(self.W_d.weight, p=2, dim=0)

    def _cleanup_memory(self):
        """Release memory when possible"""
        if torch.cuda.is_available():
            # Force garbage collection
            gc.collect()
            torch.cuda.empty_cache()

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
            f"Dimensions: n={self.n}, m={self.m}\n"
            f"Final λ: {self.lambda_l1:.2f}\n"
            f"Activation: {self.activation_name}\n"
            f"Training steps: {self.training_history['steps'][-1] if self.training_history['steps'] else 0}\n"
            f"Mixed precision: {self.use_mixed_precision}\n"
            f"PyTorch Compile: {self.use_compile}\n"
            f"Dead features: {self.feature_tracker.get_dead_features().sum().item()} / {self.m}"
        )
        ax.text(0.5, 0.5, settings_text, 
                ha='center', va='center', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        fig.suptitle(f"Training History - Enhanced Sparse Autoencoder", fontsize=16)
        plt.subplots_adjust(top=0.93)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"Training history plot saved to {save_path}")
            
        return fig

    def plot_feature_vectors(self, num_features=16, input_shape=(28, 28), figsize=(12, 12), 
                            save_path=None, cmap='viridis'):
        """
        Plot feature vectors (decoder weights) as images
        
        Args:
            num_features: Number of features to display
            input_shape: Shape to reshape features to (e.g., 28x28 for MNIST)
            figsize: Figure size
            save_path: Path to save the plot
            cmap: Colormap to use
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        self.eval()
        
        # Get feature vectors (decoder weights)
        with torch.no_grad():
            W_d = self.W_d.weight.cpu().numpy()
            
        # Calculate feature norms
        feature_norms = np.linalg.norm(W_d, axis=0)
        
        # Get indices of features with highest norms
        top_indices = np.argsort(-feature_norms)[:num_features]
        
        # Create subplot grid
        rows = int(np.ceil(np.sqrt(num_features)))
        cols = int(np.ceil(num_features / rows))
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows * cols > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
            
        # Plot each feature
        for i, idx in enumerate(top_indices):
            if i >= len(axes):
                break
                
            feature = W_d[:, idx]
            
            # Reshape to input shape if possible
            if np.prod(input_shape) == feature.shape[0]:
                feature = feature.reshape(input_shape)
                
            # Normalize for better visualization
            vmax = np.max(np.abs(feature))
            
            # Plot
            im = axes[i].imshow(feature, cmap=cmap, vmin=-vmax, vmax=vmax)
            axes[i].set_title(f"Feature {idx}\nNorm: {feature_norms[idx]:.2f}")
            axes[i].axis('off')
            
        # Hide unused subplots
        for i in range(len(top_indices), len(axes)):
            axes[i].axis('off')
            
        plt.tight_layout()
        plt.suptitle("Top Feature Vectors (Decoder Weights)", fontsize=16)
        plt.subplots_adjust(top=0.9)
        
        # Add colorbar
        fig.colorbar(im, ax=axes.tolist(), shrink=0.7)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"Feature vectors plot saved to {save_path}")
            
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

    def train_and_validate(self, X_train, X_val, learning_rate=5e-5, batch_size=4096, 
                          target_steps=200_000, checkpoint_freq=50000, save_best=False, 
                          enable_checkpoints=False, resume_from=None, grad_accum_steps=1, 
                          eval_freq=None, scheduler_type=None, early_stopping=False, 
                          early_stopping_patience=5, warmup_steps_pct=0.05, 
                          final_decay_pct=0.2, plot_weights_freq=0, 
                          plot_input_shape=(28, 28), plot_save_dir=None):
        """
        Enhanced training method with mixed precision, better logging, and visualization.
        
        Args:
            X_train: Training data tensor
            X_val: Validation data tensor
            learning_rate: Initial learning rate
            batch_size: Batch size for training
            target_steps: Target number of training steps
            checkpoint_freq: How often to save checkpoints
            save_best: Whether to save the best model based on validation loss
            enable_checkpoints: Whether to save periodic checkpoints
            resume_from: Optional checkpoint path to resume training from
            grad_accum_steps: Number of gradient accumulation steps
            eval_freq: How often to evaluate on validation set
            scheduler_type: Type of learning rate scheduler ('cosine', 'linear', 'constant')
            early_stopping: Whether to enable early stopping
            early_stopping_patience: Patience for early stopping
            warmup_steps_pct: Percentage of steps for lambda and LR warmup
            final_decay_pct: Percentage of steps for final decay phase
            plot_weights_freq: How often to plot decoder weights (0 to disable)
            plot_input_shape: Shape to reshape weights to for visualization
            plot_save_dir: Directory to save weight plots
            
        Returns:
            self: The trained model
        """
        # Initialize optimizer
        optimizer = optim.Adam(
            self.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0)
        
        # Initialize mixed precision scaler if requested
        scaler = GradScaler() if self.use_mixed_precision else None
        
        # We'll primarily use the simple linear decay for the final portion of training
        # Only create a PyTorch scheduler if explicitly requested
        scheduler = None
        if scheduler_type == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=1000,                 # Initial restart period
                T_mult=2,                 # Double period after each restart
                eta_min=learning_rate / 100  # Minimum learning rate
            )
        # Note: We don't recommend using these other schedulers, as the simple linear 
        # decay in the last part of training (configured by final_decay_pct) is preferred
            
        # Try to resume from checkpoint
        if resume_from:
            # Try to resume from specified checkpoint
            self.resume_from_checkpoint(resume_from)
        else:
            # Look for latest checkpoint
            checkpoint_pattern = f"{self.sae_model_path}.step*"
            checkpoint_files = glob.glob(checkpoint_pattern)
            
            if checkpoint_files and not '--force_retrain' in sys.argv:
                # Sort checkpoints by step number
                def get_step(filepath):
                    match = re.search(r'step(\d+)', filepath)
                    if match:
                        return int(match.group(1))
                    return 0
                
                checkpoint_files.sort(key=get_step)
                latest_checkpoint = checkpoint_files[-1]
                self.resume_from_checkpoint(latest_checkpoint)

        # Track training start time
        start_time = time.time()

        # Preprocess data - scale so E_x[||x||₂] = √n
        C = self.preprocess(X_train)
        X_train = X_train.clone() / C
        X_val = X_val.clone() / C

        # Setup data loaders
        train_dataset = TensorDataset(X_train)
        val_dataset = TensorDataset(X_val)
        
        # Check if tensors are on CPU before using pin_memory
        pin_memory = torch.cuda.is_available() and X_train.device.type == 'cpu'
            
        # Configure data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size * 2, shuffle=False, pin_memory=pin_memory)

        # Calculate training parameters
        steps_per_epoch = len(train_loader)
        num_epochs = max(1, target_steps // steps_per_epoch)
        actual_total_steps = num_epochs * steps_per_epoch
        
        # Set evaluation frequency
        if eval_freq is None:
            eval_freq = max(1, steps_per_epoch // 5)  # 5 times per epoch
        
        # Initialize training parameters
        warmup_steps = int(actual_total_steps * warmup_steps_pct)  # First N% for lambda warmup
        decay_start_step = int(actual_total_steps * (1 - final_decay_pct))  # Start decay at (1-N)% of training
        best_val_loss = float('inf')
        final_lambda = self.lambda_l1
        
        # Set up directory for weight plots if enabled
        if plot_weights_freq > 0 and plot_save_dir:
            os.makedirs(plot_save_dir, exist_ok=True)
            self.logger.info(f"Weight plots will be saved to {plot_save_dir} every {plot_weights_freq} steps")

        # Determine starting step from training history
        step = self.training_history["steps"][-1] if self.training_history["steps"] else 0
            
        # Prepare a more concise configuration display
        config_summary = (
            f"\n{'='*60}\n"
            f"ENHANCED SAE TRAINING CONFIG\n"
            f"{'='*60}\n"
            f"Dimensions: n={self.n}, m={self.m}\n"
            f"Steps: {actual_total_steps:,} | Batch: {batch_size:,} x {grad_accum_steps} = {batch_size * grad_accum_steps:,}\n"
            f"LR: {learning_rate:.1e} | λ: {final_lambda:.2f} | LR decay in final {final_decay_pct*100:.0f}% of training\n"
            f"Features: AMP={'✓' if self.use_mixed_precision else '✗'} | Compile={'✓' if self.use_compile else '✗'} | λ warmup during first {warmup_steps_pct*100:.0f}% of training\n"
            f"Device: {self.device}"
        )
        
        self.logger.info(config_summary)

        # Set up progress tracking with tqdm
        progress_bar = tqdm(
            total=actual_total_steps,
            desc="Training",
            dynamic_ncols=True,
            miniters=20,
            initial=step  # Start from current step
        )
        
        # Reserve space with initial postfix
        progress_bar.set_postfix({
            'd%': f"{0:.0f}",
            'λ': f"{self.lambda_l1:.1f}",
            'lr': f"{learning_rate:.1e}",
            'loss': f"{0.000:.3f}",
            'val': f"{0.000:.3f}"
        })

        # Reset gradients at the beginning of training
        optimizer.zero_grad()
        accum_batch = 0
        running_loss = 0.0
        early_stop = False
        val_loss_value = 0.0  # Initialize for progress bar
        
        for epoch in range(num_epochs):
            if early_stop:
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break
                
            self.train()

            for batch_idx, batch in enumerate(train_loader):
                # Skip steps if we've already done enough
                if step >= actual_total_steps:
                    break
                    
                # Apply lambda warmup during initial phase of training
                if step < warmup_steps:
                    self.lambda_l1 = (step / warmup_steps) * final_lambda
                else:
                    self.lambda_l1 = final_lambda

                # Apply manual learning rate decay in last portion of training (controlled by final_decay_pct)
                # This is the primary form of learning rate scheduling we're using as per your preference
                if scheduler is None and step >= decay_start_step:
                    progress = (step - decay_start_step) / (actual_total_steps - decay_start_step)
                    new_lr = learning_rate * (1 - progress)  # Linear decay to zero
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                
                x = batch[0]
                
                # Forward pass with optional mixed precision
                with autocast(device_type=self.device if self.device == 'cpu' else 'cuda',
                             enabled=self.use_mixed_precision):
                    x, x_hat, f_x = self.forward(x)
                    total_loss, L2_loss, L1_loss = self.compute_loss(x, x_hat, f_x)
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
                    'val': f"{val_loss_value:.3f}"
                })
                progress_bar.update(1)
                
                # Plot feature vectors periodically if enabled
                if plot_weights_freq > 0 and step % plot_weights_freq == 0:
                    try:
                        # Create a custom save path if directory is provided
                        save_path = None
                        if plot_save_dir:
                            save_path = os.path.join(plot_save_dir, f"weights_step_{step:06d}.png")
                            
                        # Run the plotting function
                        self.plot_feature_vectors(
                            input_shape=plot_input_shape,
                            num_features=16,  # A reasonable number for visualization
                            save_path=save_path
                        )
                        plt.close()  # Close to prevent too many open figures
                    except Exception as e:
                        self.logger.warning(f"Error plotting feature vectors: {e}")

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
                            x_val, x_hat_val, f_x_val = self.forward(x_val)
                            val_total_loss, l2_val, l1_val = self.compute_loss(
                                x_val, x_hat_val, f_x_val)
                            val_loss += val_total_loss.item()
                            l1_val_loss += l1_val.item()
                            l2_val_loss += l2_val.item()
                            val_batches += 1

                    avg_val_loss = val_loss / val_batches
                    avg_l1_val_loss = l1_val_loss / val_batches
                    avg_l2_val_loss = l2_val_loss / val_batches
                    val_loss_value = avg_val_loss  # Store for progress bar

                    # Calculate sparsity and feature norms
                    sparsity = (f_x.abs() >= self.feature_tracker.activation_threshold).float().mean().item()
                    W_d_norms = torch.norm(self.W_d.weight, p=2, dim=0)
                    avg_feature_norm = W_d_norms.mean().item()
                    
                    # Calculate elapsed time and estimated time remaining
                    elapsed_time = time.time() - start_time
                    time_per_step = elapsed_time / step
                    remaining_steps = actual_total_steps - step
                    estimated_remaining = remaining_steps * time_per_step
                    
                    # Update progress bar with validation info
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
                    
                    # Save best model if requested and if we have a new best validation loss
                    if save_best and avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        best_model_path = f"{self.sae_model_path}.best"
                        torch.save(checkpoint, best_model_path)
                        self.logger.info(f"New best model saved (val_loss: {avg_val_loss:.4f})")

                    # Save checkpoint periodically
                    if enable_checkpoints and step % checkpoint_freq == 0:
                        checkpoint_path = f"{self.sae_model_path}.step{step}"
                        torch.save(checkpoint, checkpoint_path)
                        self.logger.info(f"Checkpoint saved at step {step}")
                    
                    # Check early stopping (disabled by default)
                    if early_stopping and step > actual_total_steps * 0.5:  # Only check after half of training
                        early_stop = self.early_stopping_check(
                            avg_val_loss, early_stopping_patience)
                        if early_stop:
                            self.logger.info(f"Early stopping triggered at step {step}")
                            break
                    # Note: Early stopping is disabled by default as per your preference
                    
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
                x_val, x_hat_val, f_x_val = self.forward(x_val)
                val_total_loss, _, _ = self.compute_loss(x_val, x_hat_val, f_x_val)
                val_loss += val_total_loss.item()
                val_batches += 1
            final_val_loss = val_loss / val_batches
        
        # Training summary
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
            'optimizer_state_dict': optimizer.state_dict(),
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
        
        torch.save(final_checkpoint, self.sae_model_path)
        self.logger.info(f"Final model saved to {self.sae_model_path}")
        
        # Plot training history if available
        if len(self.training_history["steps"]) > 5:  # Only if we have enough data points
            try:
                history_path = f"{os.path.splitext(self.sae_model_path)[0]}_history.png"
                self.plot_training_history(save_path=history_path)
                self.logger.info(f"Training history plot saved to {history_path}")
            except Exception as e:
                self.logger.error(f"Could not generate training history plots: {e}")
        
        return self