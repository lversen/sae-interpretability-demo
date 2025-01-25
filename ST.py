import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple
from deadfeatures import DeadFeatureTracker

class SparseTransformer(nn.Module):
    def __init__(self, X, D: int, F: int, M: int, st_model_path: str, lambda_l1: float = 5.0, device: str = 'cuda'):
        super(SparseTransformer, self).__init__()
        self.D = D  # Input dimension
        self.F = F  # Feature dimension (analogous to m in SAE)
        self.M = M  # Attention dimension
        self.st_model_path = st_model_path
        self.lambda_l1 = lambda_l1
        self.device = device
        
        X_idx= np.random.choice(X.shape[0], size=self.F, replace=False)
        X = X[X_idx]
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X.astype(np.float32)).to(self.device)
        elif isinstance(X, torch.Tensor) and X.device != self.device:
            X = X.to(self.device)
        self.X = X
        
        # Initialize linear transformations
        self.W_q = nn.Linear(D, M, bias=True)
        self.W_k = nn.Linear(D, M, bias=True)
        self.W_v = nn.Linear(D, D, bias=True)
        
        # Initialize feature tracker for monitoring dead features
        self.feature_tracker = DeadFeatureTracker(
            num_features=F,
            window_size=10_000_000,  # As per PDF
            update_interval=10_000
        )
        
        self.initialize_weights()
        self.to(device)
    
    def initialize_weights(self):
        """Initialize weights with random directions and controlled L2 norms"""
        with torch.no_grad():
            for layer in [self.W_q, self.W_k, self.W_v]:
                # Initialize with random orthogonal matrix
                nn.init.orthogonal_(layer.weight)
                
                # Set L2 norms between 0.05 and 1 as per PDF
                target_norms = 0.05 + 0.95 * torch.rand(layer.weight.size(0))
                current_norms = torch.norm(layer.weight.data, p=2, dim=1)
                layer.weight.data = layer.weight.data * (target_norms / current_norms).unsqueeze(1)
                
                # Initialize biases to zero as per PDF
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def preprocess(self, X):
        """Preprocess input data according to PDF specifications"""
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X.astype(np.float32)).to(self.device)
        elif isinstance(X, torch.Tensor) and X.device != self.device:
            X = X.to(self.device)

        # Scale dataset by single constant C = E[||x||_2]/D as per PDF
        C = torch.mean(torch.norm(X, p=2, dim=1)) / self.D
        return X / C

    def compute_losses(self, x: torch.Tensor, x_hat: torch.Tensor, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute losses according to PDF specifications"""
        # L2 reconstruction loss normalized by input dimension
        L2_loss = torch.mean(torch.sum((x - x_hat)**2, dim=1)) / self.D
        
        l1_loss = torch.abs(x - x_hat)  # [batch_size, D]
        
        # Reduce along the feature dimension, weighted by attention
        attention_weights = f.mean(dim=1, keepdim=True)  # Reduce attention over F if needed
        weighted_loss = attention_weights * l1_loss  # Broadcasting weights to match x dimensions
        L1_loss = weighted_loss.mean()

        total_loss = L2_loss + L1_loss
        return total_loss, L2_loss, L1_loss

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with scaled dot-product attention"""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32)).to(self.device)
        elif isinstance(x, torch.Tensor) and x.device != self.device:
            x = x.to(self.device)
            
        # Compute query, key, and value vectors
        Q = self.W_q(x)                     # [batch_size, M]
        K = self.W_k(self.X)      # [F, M]
        V = self.W_v(self.X)      # [F, D]
        
        attention_scores = torch.matmul(Q, K.T)
        attention_scores = attention_scores
        f = torch.softmax(attention_scores, dim=1)  # [batch_size, F]
        
        # Compute reconstruction
        x_hat = torch.matmul(f, V)  # [batch_size, D]
        
        return x, x_hat, f

    def feature_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Get feature activations scaled by L2 norms of value vectors"""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32)).to(self.device)
        elif isinstance(x, torch.Tensor) and x.device != self.device:
            x = x.to(self.device)
        
        with torch.no_grad():
            Q = self.W_q(x)
            K = self.W_k(self.X)
            V = self.W_v(self.X)
            
            attention_scores = torch.matmul(Q, K.T)
            f = torch.softmax(attention_scores, dim=1)
            
            # Scale activations by L2 norm of value vectors as per PDF
            V_norms = torch.norm(V, p=2, dim=1)
            return f * V_norms[None, :]

    def train_and_validate(self, X_train, X_val, learning_rate=5e-5, batch_size=4096, target_steps=200000):
        """Train the model according to PDF specifications"""
        # Initialize Adam optimizer with beta1=0.9, beta2=0.999, no weight decay
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        
        # Preprocess data
        X_train = self.preprocess(X_train)
        X_val = self.preprocess(X_val)
        
        # Setup data loaders
        train_dataset = TensorDataset(X_train)
        val_dataset = TensorDataset(X_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Calculate training schedule
        steps_per_epoch = len(train_loader)
        num_epochs = max(1, target_steps // steps_per_epoch)
        actual_total_steps = num_epochs * steps_per_epoch
        final_lambda = 5.0  # Match SAE default
        
        # Training timing parameters as per PDF
        warmup_steps = actual_total_steps // 20  # First 5% for lambda warmup
        decay_start_step = int(actual_total_steps * 0.8)  # Start decay at 80% of training
        
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
        print(f"\n{'Step':>8} {'Epoch':>5} {'Loss':>8} {'ValLoss':>8} {'λ':>5} {'Dead%':>6} {'Sparse%':>7} {'Track%':>7}")
        
        best_val_loss = float('inf')
        step = 0
        
        for epoch in range(num_epochs):
            self.train()
            for batch, in train_loader:
                # Lambda warmup in first 5% of training
                if step < warmup_steps:
                    current_lambda = final_lambda * (step / warmup_steps)
                else:
                    current_lambda = final_lambda
                self.lambda_l1 = current_lambda
                
                # Learning rate decay in last 20%
                if step >= decay_start_step:
                    progress = (step - decay_start_step) / (actual_total_steps - decay_start_step)
                    new_lr = learning_rate * (1 - progress)  # Linear decay to zero
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                
                optimizer.zero_grad()
                x, x_hat, f = self.forward(batch)
                
                # Update feature tracking
                dead_ratio, stats = self.feature_tracker.update(f)
                
                # Compute and backpropagate loss
                total_loss, L2_loss, L1_loss = self.compute_losses(x, x_hat, f)
                total_loss.backward()
                
                # Clip gradient norm to 1.0 as per PDF
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                
                # Calculate sparsity metrics
                sparsity = (f.abs() >= self.feature_tracker.activation_threshold).float().mean().item()
                tracking_progress = min(self.feature_tracker.samples_seen / self.feature_tracker.window_size, 1.0)
                
                # Periodic validation and logging
                if step % (steps_per_epoch // 5) == 0:
                    self.eval()
                    val_loss = 0.0
                    val_batches = 0
                    
                    with torch.no_grad():
                        for val_batch, in val_loader:
                            x, x_hat, f = self.forward(val_batch)
                            batch_loss, _, _ = self.compute_losses(x, x_hat, f)
                            val_loss += batch_loss.item()
                            val_batches += 1
                        
                        avg_val_loss = val_loss / val_batches
                        
                        # Print metrics in consistent format
                        print(f"{step:8d} {epoch:5d} {total_loss.item():8.4f} {avg_val_loss:8.4f} "
                              f"{current_lambda:5.2f} {dead_ratio:6.1%} {sparsity:7.1%} {tracking_progress:7.1%}")
                        
                        # Save best model
                        if avg_val_loss < best_val_loss:
                            best_val_loss = avg_val_loss
                            torch.save(self.state_dict(), self.st_model_path)
                    
                    self.train()
                
                step += 1
        
        print(f"\nTraining completed:")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Final dead feature ratio: {dead_ratio:.1%}")
        print(f"Steps completed: {step}/{actual_total_steps}")
        print(f"Final λ: {self.lambda_l1:.2f}")