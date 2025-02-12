import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Union
from deadfeatures import DeadFeatureTracker

class SparseAutoencoder(nn.Module):
    def __init__(self, n: int, m: int, sae_model_path: str, lambda_l1: float = 1, device: str = 'cuda'):
        super(SparseAutoencoder, self).__init__()
        self.n = n
        self.m = m
        self.sae_model_path = sae_model_path
        self.lambda_l1 = lambda_l1
        self.device = device

        self.W_e = nn.Linear(n, m, bias=False)
        self.W_d = nn.Linear(m, n, bias=False)
        self.b_e = nn.Parameter(torch.zeros(m))
        self.b_d = nn.Parameter(torch.zeros(n))
        
        # Initialize feature tracker
        self.feature_tracker = DeadFeatureTracker(
            num_features=m,  # m is the number of features
            window_size=10_000_000,  # As specified in the paper
            update_interval=10_000  # Update stats every 10k samples
        )

        self.initialize_weights()
        self.to(device)

    def initialize_weights(self):
        with torch.no_grad():
            W_d = torch.randn(self.n, self.m)
            norms = torch.norm(W_d, p=2, dim=0)
            target_norms = 0.05 + 0.95 * torch.rand(self.m)
            W_d = W_d * (target_norms / norms)
            self.W_d.weight.data = W_d
            self.W_e.weight.data = W_d.t()
            self.b_e.data = torch.zeros(self.m)
            self.b_d.data = torch.zeros(self.n)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32)).to(self.device)
        elif isinstance(x, torch.Tensor) and x.device != self.device:
            x = x.to(self.device)

        f_x = torch.relu(self.W_e(x) + self.b_e)
        x_hat = self.W_d(f_x) + self.b_d
        
        return x, x_hat, f_x

    def compute_loss(self, x, x_hat, f_x):
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
            W_d_norms = torch.norm(self.W_d.weight, p=2, dim=0)
            
            # Sparsity penalty: sum over features (m), average over batch
            L1_penalty = self.lambda_l1 * torch.mean(torch.sum(f_x * W_d_norms, dim=1))
            
            total_loss = L2_loss + L1_penalty
            
            return total_loss, L2_loss, L1_penalty

    def preprocess(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X.astype(np.float32)).to(self.device)
        elif isinstance(X, torch.Tensor) and X.device != self.device:
            X = X.to(self.device)

        C = torch.mean(torch.norm(X, p=2, dim=1)) / np.sqrt(self.n)
        return(C)

    def feature_vectors(self):
        return self.W_d.weight / torch.norm(self.W_d.weight, p=2, dim=0)

    def feature_activations(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32)).to(self.device)
        elif isinstance(x, torch.Tensor) and x.device != self.device:
            x = x.to(self.device)

        f_x = torch.relu(self.W_e(x) + self.b_e)
        return f_x * torch.norm(self.W_d.weight, p=2, dim=0)

    def train_and_validate(self, X_train, X_val, learning_rate=1e-3, batch_size=4096, target_steps=10_000):
        """
        Train the Sparse Autoencoder targeting a specific number of steps while tracking dead features.
        
        Args:
            X_train: Training data
            X_val: Validation data
            learning_rate: Initial learning rate
            batch_size: Batch size for training
            target_steps: Target number of training steps (default 200k as per paper)
        """
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0)
        
        # Preprocess data
        C = self.preprocess(X_train)
        X_train /= C
        X_val /= C

        # Setup data loaders
        train_dataset = TensorDataset(X_train)
        val_dataset = TensorDataset(X_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Calculate required epochs to reach target steps
        steps_per_epoch = len(train_loader)
        num_epochs = max(1, target_steps // steps_per_epoch)
        actual_total_steps = num_epochs * steps_per_epoch
        
        # Initialize training parameters
        warmup_steps = actual_total_steps // 20  # First 5% for lambda warmup
        decay_start_step = int(actual_total_steps * 0.8)  # Start decay at 80% of training
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
        print(f"\n{'Step':>8} {'Epoch':>5} {'Loss':>8} {'ValLoss':>8} {'λ':>5} {'Dead%':>6} {'Sparse%':>7} {'Track%':>7}")

        for epoch in range(num_epochs):
            self.train()
            epoch_train_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                x = batch[0]

                # Forward pass
                x, x_hat, f_x = self.forward(x)
                
                # Update feature tracking
                dead_ratio, stats = self.feature_tracker.update(f_x)

                # Lambda warmup
                if step < warmup_steps:
                    self.lambda_l1 = (step / warmup_steps) * final_lambda
                else:
                    self.lambda_l1 = final_lambda
                
                # Learning rate decay in last 20%
                if step >= decay_start_step:
                    progress = (step - decay_start_step) / (actual_total_steps - decay_start_step)
                    new_lr = learning_rate * (1 - progress)  # Linear decay to zero
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr

                # Compute loss and update
                total_loss, L2_loss, L1_loss = self.compute_loss(x, x_hat, f_x)
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
                            x_val, x_hat_val, f_x_val = self.forward(x_val)
                            val_total_loss, _, _ = self.compute_loss(x_val, x_hat_val, f_x_val)
                            val_loss += val_total_loss.item()
                            val_batches += 1
                    
                    avg_val_loss = val_loss / val_batches
                    
                    # Calculate sparsity and tracking progress
                    sparsity = (f_x.abs() >= self.feature_tracker.activation_threshold).float().mean().item()
                    tracking_progress = min(self.feature_tracker.samples_seen / self.feature_tracker.window_size, 1.0)
                    

                    print(f"Step: {step:6d}/{actual_total_steps} ({step/actual_total_steps:3.1%}) | "
                          f"Epoch: {epoch:3d} | LR: {
                              optimizer.param_groups[0]['lr']:.2e} | "
                          f"Train: {total_loss.item():8.4f} | Val: {
                        val_loss:8.4f} | "
                        f"L1_loss: {L1_loss:4.2f} | L2_loss: {
                              L2_loss:4.2f} | "
                        f"L1 λ: {self.lambda_l1:4.2f} | Sparse: {sparsity:5.1%} | ")
                    

                    # Save best model
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        torch.save(self.state_dict(), self.sae_model_path)
                    
                    self.train()


        print(f"\nTraining completed:")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Final dead feature ratio: {dead_ratio:.1%}")
        print(f"Steps completed: {step}/{actual_total_steps}")
        print(f"Final λ: {self.lambda_l1:.2f}")