import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional, Dict, Any
from deadfeatures import DeadFeatureTracker
import math
import time
from torch.optim.lr_scheduler import CosineAnnealingLR


class SparseTransformer(nn.Module):
    def __init__(self, X, n: int, m: int, a: int, st_model_path: str,
                 lambda_l1: float = 5.0, num_heads: int = 1, device: str = 'cuda',
                 window_size: int = 10_000_000, update_interval: int = 10_000,
                 activation_threshold: float = 1e-3, seed: int = 42):
        """
        Enhanced Sparse Transformer model preserving original architecture
        
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
            seed: Random seed for weight initialization
        """
        super().__init__()
        
        # Store parameters
        self.n, self.m, self.a = n, m, a
        self.st_model_path = st_model_path
        self.lambda_l1 = lambda_l1
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.activation_threshold = activation_threshold
        
        # Set seed for reproducibility
        torch.manual_seed(seed)
        
        # Store reference data for attention
        self.X_data = self.type_check(X)
        
        # Training state
        self.steps = 0
        self.total_steps = 0
        self.memory_update_freq = 1
        self.memory_update_strategy = 'random'  # Can be 'random', 'stratified', 'diverse'
        self.best_feature_metrics = {}
        
        # Keeping the original projections with original parameter names
        self.W_q = nn.Linear(n, a, bias=False)
        self.W_k = nn.Linear(n, a, bias=False)
        self.W_v = nn.Linear(n, n, bias=False)  # Original had value projection to same dimension as input
        
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
        
        # Track attention entropy and sparsity metrics
        self.attention_entropy_history = []
        self.sparsity_history = []
        
        # Initialize weights properly
        self.initialize_weights()
        self.to(self.device)
    
    def initialize_weights(self):
        """
        Improved weight initialization with better scaling
        """
        with torch.no_grad():
            # Initialize query projection with orthogonal initialization
            nn.init.orthogonal_(self.W_q.weight)
            
            # Initialize key projection with orthogonal initialization
            nn.init.orthogonal_(self.W_k.weight)
            
            # Initialize value projection matrix with controlled norms
            V_weight = torch.randn(self.n, self.n)
            
            # Orthogonalize value projection
            if self.n <= 10000:  # Only for reasonably sized matrices
                # Orthogonalize for better conditioning
                u, s, v = torch.linalg.svd(V_weight, full_matrices=False)
                V_weight = u @ v
            
            # Normalize columns to have unit norm
            norms = torch.norm(V_weight, p=2, dim=0)
            V_weight = V_weight / norms.clamp(min=1e-8)
            
            # Scale columns to have norms between 0.05 and 1
            target_norms = 0.05 + 0.95 * torch.rand(self.n)
            V_weight = V_weight * target_norms
            
            self.W_v.weight.data = V_weight
    
    def type_check(self, x):
        """Ensure data is on the correct device and has the right type"""
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x.astype(np.float32)).to(self.device)
        return x.to(self.device) if x.device != self.device else x
    
    def scaled_dot_product_attention(self, query, key, value, attn_mask=None, dropout_p=0.0,
            is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
        """
        Enhanced scaled_dot_product_attention implementation with stability improvements
        """
        L, S = query.size(-2), key.size(-2)
        
        # Improved scaling with proper normalization
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        
        # Create attention bias matrix
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        
        # Handle causal masking
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        # Handle attention mask
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias = attn_mask + attn_bias

        # Handle grouped query attention
        if enable_gqa:
            key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
            value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

        # Compute attention scores with better numerical stability
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        
        # Add bias
        attn_weight += attn_bias
        
        # Apply softmax with improved numerical stability
        attn_weight = torch.softmax(attn_weight, dim=-1)
        
        # Apply dropout if training
        if dropout_p > 0 and self.training:
            attn_weight = torch.dropout(attn_weight, dropout_p, self.training)
        
        # Compute output
        output = attn_weight @ value
        
        return output, attn_weight, value
    
    def compute_attention_entropy(self, attention_weights):
        """Compute entropy of attention distribution as a sparsity metric"""
        # Avoid log(0)
        eps = 1e-10
        log_attn = torch.log(attention_weights + eps)
        entropy = -torch.sum(attention_weights * log_attn, dim=-1)
        return entropy.mean().item()
    
    def update_memory_indices(self, strategy='random'):
        """
        Update memory indices with different strategies
        
        Args:
            strategy: Strategy to use for updating memory indices
                - 'random': Random sampling from reference data
                - 'diverse': Sample to maximize feature diversity
                - 'active': Prioritize samples that activate different features
        """
        with torch.no_grad():
            if strategy == 'random':
                # Simple random sampling
                self.memory_indices = torch.randint(0, self.X_data.shape[0], 
                                                (self.m,), device=self.device)
            elif strategy == 'diverse':
                # Try to select diverse samples
                # This is just a placeholder - in a real implementation, you'd use 
                # k-means++ or another diversity-promoting algorithm
                indices = torch.randint(0, self.X_data.shape[0], 
                                    (self.m * 3,), device=self.device)
                selected = torch.randperm(indices.size(0), device=self.device)[:self.m]
                self.memory_indices = indices[selected]
            elif strategy == 'active':
                # Prioritize samples that activate different features
                # This is just a placeholder - in a real implementation, you'd track
                # which samples activate which features and sample accordingly
                self.memory_indices = torch.randint(0, self.X_data.shape[0], 
                                                (self.m,), device=self.device)
            else:
                raise ValueError(f"Unknown memory update strategy: {strategy}")

    def forward(self, x):
        """
        Enhanced forward pass implementation with improved memory management
        """
        # Update memory indices periodically during training
        if self.training and self.steps % self.memory_update_freq == 0:
            with torch.no_grad():
                # Only update after warmup and before decay
                if self.steps > self.total_steps // 20 and self.steps < self.total_steps * 0.8:
                    self.update_memory_indices(self.memory_update_strategy)
                    if self.memory_update_strategy != 'random':
                        print(f"Memory indices updated using {self.memory_update_strategy} strategy")
        
        self.steps += 1
        
        # Get cross attention context from memory
        X_cross = self.X_data[self.memory_indices]  # Shape: [m, n]
        
        # Preprocess data - scale by mean norm
        C = self.preprocess(X_cross)
        X_cross = X_cross / C
        
        # Type conversion for input x
        x = self.type_check(x)  # Shape: [batch_size, n]
        x = x / C  # Apply same scaling
        
        # Project using the original approach - preserving dimensions
        q = self.norm_q(self.W_q(x))  # Shape: [batch_size, a]
        k = self.norm_k(self.W_k(X_cross))  # Shape: [m, a]
        v = self.norm_v(self.W_v(X_cross))  # Shape: [m, n]
        
        # Compute attention
        x_hat, attn_weights, value = self.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0
        )
        
        # Track attention metrics during training
        if self.training:
            # Update feature activity tracking
            dead_ratio, stats = self.feature_tracker.update(attn_weights)
            
            # Track attention entropy (lower is more sparse)
            attention_entropy = self.compute_attention_entropy(attn_weights)
            self.attention_entropy_history.append((self.steps, attention_entropy))
            
            # Track sparsity
            sparsity = (attn_weights.abs() >= self.activation_threshold).float().mean().item()
            self.sparsity_history.append((self.steps, sparsity))
        
        return x, x_hat, attn_weights, value

    def compute_loss(self, x: torch.Tensor, x_hat: torch.Tensor, 
                    f: torch.Tensor, v:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute loss following SAE loss pattern with L2 reconstruction + weighted L1 penalty
        
        L = (1/|X|) * Σ ||x - x̂||₂² + λ * Σᵢ |fᵢ(x)| ||vᵢ||₂
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
    
    @torch.no_grad()
    def get_feature_stats(self, dataloader) -> Dict[str, Any]:
        """
        Compute comprehensive feature activation statistics across a dataset
        
        Args:
            dataloader: DataLoader containing samples to analyze
            
        Returns:
            Dictionary of feature statistics
        """
        self.eval()
        
        # Initialize statistics
        all_activations = []
        activation_counts = torch.zeros(self.m, device=self.device)
        
        # Process all batches
        for batch in dataloader:
            x = batch[0].to(self.device)
            
            # Get feature activations
            _, _, f, _ = self.forward(x)
            
            # Track activations
            active_features = (f.abs() >= self.activation_threshold)
            activation_counts += active_features.sum(dim=0)
            
            # Store batch activations
            all_activations.append(f.cpu())
        
        # Combine all activations
        all_activations = torch.cat(all_activations, dim=0)
        total_samples = all_activations.shape[0]
        
        # Calculate statistics
        mean_activations = all_activations.abs().mean(dim=0).numpy()
        max_activations = all_activations.abs().max(dim=0)[0].numpy()
        activation_rates = activation_counts.cpu().numpy() / total_samples
        
        # Calculate sparsity metrics
        avg_sparsity = 1.0 - (all_activations.abs() >= self.activation_threshold).float().mean().item()
        mean_features_per_sample = (all_activations.abs() >= self.activation_threshold).sum(dim=1).float().mean().item()
        
        # Identify most and least used features
        feature_usage = activation_counts.cpu().numpy()
        most_used = np.argsort(-feature_usage)[:10]
        least_used = np.argsort(feature_usage)[:10]
        dead_features = np.where(feature_usage == 0)[0]
        
        return {
            'avg_sparsity': avg_sparsity,
            'mean_features_per_sample': mean_features_per_sample,
            'activation_rates': activation_rates,
            'mean_activations': mean_activations,
            'max_activations': max_activations,
            'most_used_features': most_used,
            'least_used_features': least_used,
            'dead_features': dead_features,
            'dead_feature_count': len(dead_features),
            'total_features': self.m
        }

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

    def train_and_validate(self, X_train, X_val, learning_rate=5e-5, batch_size=4096, target_steps=200_000,
                          lr_schedule='linear', early_stopping_patience=None):
        """
        Enhanced training procedure with improved scheduling and monitoring
        
        Args:
            X_train: Training data
            X_val: Validation data
            learning_rate: Initial learning rate
            batch_size: Batch size for training
            target_steps: Target number of training steps
            lr_schedule: Learning rate schedule ('linear', 'cosine', 'warmup_cosine')
            early_stopping_patience: Early stopping patience in epochs (None to disable)
        """
        # Create optimizer
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
        self.memory_update_freq = max(1, int(self.total_steps/100))
        
        # Initialize training parameters
        warmup_steps = actual_total_steps // 20  # First 5% for lambda warmup
        decay_start_step = int(actual_total_steps * 0.8)  # Start decay at 80% of training
        step = 0
        best_val_loss = float('inf')
        final_lambda = self.lambda_l1
        
        # Initialize early stopping
        no_improvement_epochs = 0
        best_dead_ratio = 1.0
        
        # Initialize learning rate scheduler if using cosine schedule
        if lr_schedule == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=actual_total_steps)
        
        # Initialize training metrics tracking
        train_losses = []
        val_losses = []
        dead_ratios = []
        learning_rates = []
        
        print("\nTraining Configuration:")
        print(f"Total Steps: {actual_total_steps}")
        print(f"Epochs: {num_epochs}")
        print(f"Steps per Epoch: {steps_per_epoch}")
        print(f"Batch Size: {batch_size}")
        print(f"Warmup Steps: {warmup_steps}")
        print(f"Learning Rate Schedule: {lr_schedule}")
        print(f"Memory Update Frequency: {self.memory_update_freq} steps")
        print(f"Memory Update Strategy: {self.memory_update_strategy}")

        print("\nMetrics:")
        print("  Loss    - Training loss for current batch")
        print("  ValLoss - Average validation loss")
        print("  λ       - Current L1 regularization strength")
        print("  Dead%   - Percentage of features with no activation in 10M samples")
        print("  Sparse% - Percentage of non-zero activations")
        print("  LR      - Current learning rate")
        print(f"\n{'Step':>8} {'Epoch':>5} {'Loss':>8} {'ValLoss':>8} {'λ':>5} {'Dead%':>6} {'Sparse%':>7} {'LR':>10}")

        # Track training start time
        start_time = time.time()

        for epoch in range(num_epochs):
            self.train()
            epoch_train_loss = 0.0
            epoch_L2_loss = 0.0
            epoch_L1_loss = 0.0
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

                # Update learning rate based on schedule
                if lr_schedule == 'linear':
                    # Linear decay to zero over last 20% of steps
                    if step >= decay_start_step:
                        progress = (step - decay_start_step) / (actual_total_steps - decay_start_step)
                        new_lr = learning_rate * (1 - progress)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = new_lr
                elif lr_schedule == 'cosine':
                    # Let the scheduler handle it
                    pass
                elif lr_schedule == 'warmup_cosine':
                    # Linear warmup followed by cosine decay
                    if step < warmup_steps:
                        # Warmup phase
                        warmup_factor = step / warmup_steps
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = learning_rate * warmup_factor
                    else:
                        # Cosine decay phase
                        cosine_steps = actual_total_steps - warmup_steps
                        cosine_progress = (step - warmup_steps) / cosine_steps
                        cosine_factor = 0.5 * (1 + math.cos(math.pi * cosine_progress))
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = learning_rate * cosine_factor

                # Compute loss and update
                total_loss, L2_loss, L1_loss = self.compute_loss(x, x_hat, f_x, v)
                total_loss.backward()

                # Gradient clipping to norm 1
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Update learning rate scheduler if using cosine schedule
                if lr_schedule == 'cosine':
                    scheduler.step()

                # Track losses
                epoch_train_loss += total_loss.item()
                epoch_L2_loss += L2_loss.item()
                epoch_L1_loss += L1_loss.item()
                num_batches += 1
                step += 1
                
                # Track current learning rate
                current_lr = optimizer.param_groups[0]['lr']
                learning_rates.append((step, current_lr))

                # Periodic validation and logging
                if batch_idx % (len(train_loader) // 5) == 0:
                    self.eval()
                    val_loss = 0.0
                    val_L2_loss = 0.0
                    val_L1_loss = 0.0
                    val_batches = 0

                    with torch.no_grad():
                        for val_batch in val_loader:
                            x_val = val_batch[0]
                            x_val, x_hat_val, f_x_val, v_val = self.forward(x_val)
                            val_total_loss, val_L2, val_L1 = self.compute_loss(
                                x_val, x_hat_val, f_x_val, v_val)
                            val_loss += val_total_loss.item()
                            val_L2_loss += val_L2.item()
                            val_L1_loss += val_L1.item()
                            val_batches += 1

                    avg_val_loss = val_loss / val_batches
                    avg_val_L2 = val_L2_loss / val_batches
                    avg_val_L1 = val_L1_loss / val_batches
                    
                    # Track validation losses
                    val_losses.append((step, avg_val_loss))
                    train_losses.append((step, total_loss.item()))
                    dead_ratios.append((step, dead_ratio))

                    # Calculate sparsity
                    sparsity = (f_x.abs() >= self.activation_threshold).float().mean().item()
                    
                    # Track progress in best feature metrics
                    if dead_ratio < best_dead_ratio:
                        best_dead_ratio = dead_ratio
                        self.best_feature_metrics = {
                            'step': step,
                            'dead_ratio': dead_ratio,
                            'sparsity': sparsity,
                            'val_loss': avg_val_loss,
                            'train_loss': total_loss.item(),
                        }
                    
                    # Note: We intentionally do NOT save the model with lowest validation loss
                    # because the lowest loss would occur early when L1 penalty is minimal
                    
                    print(f"{step:8d} {epoch:5d} {total_loss.item():8.4f} {avg_val_loss:8.4f} "
                          f"{self.lambda_l1:5.2f} {dead_ratio*100:6.2f}% {sparsity*100:7.2f}% {current_lr:.3e}")
                    
                    # Save periodic checkpoints if desired
                    if step % 50000 == 0:
                        checkpoint_path = f"{self.st_model_path}.step{step}"
                        torch.save({
                            'step': step,
                            'epoch': epoch,
                            'model_state_dict': self.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'train_loss': total_loss.item(),
                            'val_loss': avg_val_loss,
                            'dead_ratio': dead_ratio,
                            'sparsity': sparsity,
                            'lambda_l1': self.lambda_l1,
                            'learning_rate': current_lr,
                        }, checkpoint_path)
                        print(f"Saved checkpoint at step {step} to {checkpoint_path}")

                    self.train()
            
            # End of epoch processing
            epoch_train_loss /= num_batches
            epoch_L2_loss /= num_batches
            epoch_L1_loss /= num_batches
            
            # Check for early stopping
            if early_stopping_patience is not None:
                # Early stopping based on dead feature ratio rather than validation loss
                if dead_ratio < best_dead_ratio:
                    best_dead_ratio = dead_ratio
                    no_improvement_epochs = 0
                    
                    # Save model at best dead ratio
                    best_model_path = f"{self.st_model_path}.best"
                    torch.save(self.state_dict(), best_model_path)
                else:
                    no_improvement_epochs += 1
                    if no_improvement_epochs >= early_stopping_patience:
                        print(f"\nEarly stopping triggered after {epoch+1} epochs")
                        # Load best model before stopping
                        if os.path.exists(best_model_path):
                            self.load_state_dict(torch.load(best_model_path))
                        break

        # Calculate training time
        training_time = time.time() - start_time
        training_hours = training_time / 3600
        
        print(f"\nTraining completed in {training_hours:.2f} hours:")
        print(f"Final validation loss: {avg_val_loss:.4f}")
        print(f"Final dead feature ratio: {dead_ratio:.1%}")
        print(f"Steps completed: {step}/{actual_total_steps}")
        print(f"Final λ: {self.lambda_l1:.2f}")
        
        # Save the final model (we want this one, not the "best" by loss)
        torch.save({
            'step': step,
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if epoch == num_epochs-1 else None,
            'train_loss': epoch_train_loss,
            'train_L2_loss': epoch_L2_loss,
            'train_L1_loss': epoch_L1_loss,
            'val_loss': avg_val_loss,
            'val_L2_loss': avg_val_L2,
            'val_L1_loss': avg_val_L1,
            'dead_ratio': dead_ratio,
            'lambda_l1': self.lambda_l1,
            'best_feature_metrics': self.best_feature_metrics,
            'training_history': {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'dead_ratios': dead_ratios,
                'learning_rates': learning_rates,
                'attention_entropy': self.attention_entropy_history,
                'sparsity': self.sparsity_history,
            },
            'training_time_hours': training_hours,
        }, self.st_model_path)
        
        print(f"Saved final model to {self.st_model_path}")
        
        # Compute and save detailed feature statistics
        feature_stats = self.get_feature_stats(val_loader)
        stats_path = f"{self.st_model_path}.stats.pt"
        torch.save(feature_stats, stats_path)
        print(f"Saved feature statistics to {stats_path}")
        
        print(f"\nFeature Activation Summary:")
        print(f"- Dead features: {feature_stats['dead_feature_count']}/{feature_stats['total_features']} ({feature_stats['dead_feature_count']/feature_stats['total_features']:.1%})")
        print(f"- Average sparsity: {feature_stats['avg_sparsity']:.2%}")
        print(f"- Mean features activated per sample: {feature_stats['mean_features_per_sample']:.1f}")
        
        return self
        
    def load_checkpoint(self, checkpoint_path):
        """
        Load model from checkpoint with all training state
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path)
        
        # Load model state
        self.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore training metrics if available
        if 'best_feature_metrics' in checkpoint:
            self.best_feature_metrics = checkpoint['best_feature_metrics']
        
        if 'training_history' in checkpoint:
            history = checkpoint['training_history']
            if 'attention_entropy' in history:
                self.attention_entropy_history = history['attention_entropy']
            if 'sparsity' in history:
                self.sparsity_history = history['sparsity']
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        if 'step' in checkpoint:
            print(f"Restored model from step {checkpoint['step']}")
        
        return checkpoint