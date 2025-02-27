import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Union, Tuple
from deadfeatures import DeadFeatureTracker
import sys

class SparseAutoencoder(nn.Module):
    def __init__(self, n: int, m: int, sae_model_path: str, lambda_l1: float = 5, device: str = 'cuda'):
        super(SparseAutoencoder, self).__init__()
        self.n = n
        self.m = m
        self.sae_model_path = sae_model_path
        self.lambda_l1 = lambda_l1
        self.device = device if torch.cuda.is_available() else 'cpu'

        # Initialize components
        self.W_e = nn.Linear(n, m, bias=False)
        self.W_d = nn.Linear(m, n, bias=False)
        self.b_e = nn.Parameter(torch.zeros(m))
        self.b_d = nn.Parameter(torch.zeros(n))
        
        # Initialize feature tracker with 10M window as specified
        self.feature_tracker = DeadFeatureTracker(
            num_features=m,
            window_size=10_000_000,
            update_interval=10_000
        )

        self.initialize_weights()
        self.to(self.device)

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

    def resume_from_checkpoint(self, checkpoint_path):
        """Resume training from a checkpoint"""
        print(f"\nResuming from checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Check if it's a full checkpoint or just model state
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.load_state_dict(checkpoint['model_state_dict'])
                current_step = checkpoint.get('step', 0)
                if 'lambda_l1' in checkpoint:
                    self.lambda_l1 = checkpoint['lambda_l1']
                print(f"Resumed from step {current_step} with lambda_l1 = {self.lambda_l1}")
            else:
                # Just a state dict
                self.load_state_dict(checkpoint)
                print("Loaded model weights only")
            
            return True
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False

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
        """
        # L2 reconstruction loss
        L2_loss = torch.mean(torch.norm(x - x_hat, p=2, dim=1)**2)
        
        # Get L2 norms of decoder weight columns
        W_d_norms = torch.norm(self.W_d.weight, p=2, dim=0)
        
        # Sparsity penalty: λ * Σᵢ |fᵢ(x)| ||Wdᵢ||₂
        L1_penalty = self.lambda_l1 * torch.mean(torch.sum(f_x * W_d_norms, dim=1))
        
        total_loss = L2_loss + L1_penalty
        
        return total_loss, L2_loss, L1_penalty

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

    def feature_activations(self, x):
        """Calculate feature activations with L2 norm weighting"""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32)).to(self.device)
        elif isinstance(x, torch.Tensor) and x.device != self.device:
            x = x.to(self.device)

        f_x = torch.relu(self.W_e(x) + self.b_e)
        return f_x * torch.norm(self.W_d.weight, p=2, dim=0)

    def train_and_validate(self, X_train, X_val, learning_rate=5e-5, batch_size=4096, target_steps=200_000):
        """
        Train the SAE following the updated configuration:
        - Adam optimizer (beta1=0.9, beta2=0.999, no weight decay)
        - Learning rate ~5e-5 with linear decay in last 20% of training
        - λ warmup over first 5% of training
        - Gradient clipping to norm 1
        """
        # Try to find and load the latest checkpoint
        import glob
        import re
        import os
        
        # Check if any checkpoints exist
        checkpoint_pattern = f"{self.sae_model_path}.step*"
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        # Start from step 0 by default
        start_step = 0
        
        if checkpoint_files and not '--force_retrain' in sys.argv:
            # Sort checkpoints by step number
            def get_step(filepath):
                match = re.search(r'step(\d+)', filepath)
                if match:
                    return int(match.group(1))
                return 0
            
            checkpoint_files.sort(key=get_step)
            latest_checkpoint = checkpoint_files[-1]
            
            # Try to load the checkpoint
            try:
                checkpoint = torch.load(latest_checkpoint, map_location=self.device)
                
                # Check if it's a full checkpoint or just state_dict
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.load_state_dict(checkpoint['model_state_dict'])
                    start_step = checkpoint.get('step', 0)
                    if 'lambda_l1' in checkpoint:
                        self.lambda_l1 = checkpoint['lambda_l1']
                    print(f"\nResuming training from checkpoint {latest_checkpoint}")
                    print(f"Starting from step {start_step} with lambda_l1 = {self.lambda_l1}")
                else:
                    # Just a state dict
                    self.load_state_dict(checkpoint)
                    # Try to extract step from filename
                    start_step = get_step(latest_checkpoint)
                    print(f"\nResuming training from checkpoint {latest_checkpoint}")
                    print(f"Starting from approximately step {start_step}")
            except Exception as e:
                print(f"\nError loading checkpoint: {e}")
                print("Starting training from scratch")
                start_step = 0
        
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0)
        
        # Preprocess data - scale so that E_x[||x||₂] = √n
        C = self.preprocess(X_train)
        X_train = X_train.clone() / C
        X_val = X_val.clone() / C

        # Setup data loaders
        train_dataset = TensorDataset(X_train)
        val_dataset = TensorDataset(X_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Calculate training parameters
        steps_per_epoch = len(train_loader)
        num_epochs = max(1, target_steps // steps_per_epoch)
        actual_total_steps = num_epochs * steps_per_epoch
        
        # Initialize training parameters
        warmup_steps = actual_total_steps // 20  # First 5% for lambda warmup
        decay_start_step = int(actual_total_steps * 0.8)  # Start decay at 80% of training
        best_val_loss = float('inf')
        final_lambda = self.lambda_l1
        
        # Calculate which epoch to start from
        start_epoch = start_step // steps_per_epoch
        start_batch_in_epoch = start_step % steps_per_epoch
        
        # Initialize feature tracker state if resuming
        if start_step > 0:
            # If we're resuming, let's feed some batches through the model
            # to warm up the feature tracker
            warmup_batches = min(10, len(train_loader))
            print(f"Warming up feature tracker with {warmup_batches} batches...")
            self.eval()
            with torch.no_grad():
                for i, (batch,) in enumerate(train_loader):
                    if i >= warmup_batches:
                        break
                    _, _, f_x = self.forward(batch)
                    self.feature_tracker.update(f_x)

        print("\nTraining Configuration:")
        print(f"Total Steps: {actual_total_steps}")
        print(f"Epochs: {num_epochs}")
        print(f"Steps per Epoch: {steps_per_epoch}")
        print(f"Batch Size: {batch_size}")
        print(f"Warmup Steps: {warmup_steps}")
        print(f"Learning Rate Decay Start: {decay_start_step}")
        print(f"Starting from step: {start_step} (epoch {start_epoch}, batch {start_batch_in_epoch})")
        
        print("\nMetrics:")
        print("  Loss    - Training loss for current batch")
        print("  ValLoss - Average validation loss")
        print("  λ       - Current L1 regularization strength")
        print("  Dead%   - Percentage of features with no activation in 10M samples")
        print("  Sparse% - Percentage of non-zero activations")
        print(f"\n{'Step':>8} {'Epoch':>5} {'Loss':>8} {'ValLoss':>8} {'λ':>5} {'Dead%':>6} {'Sparse%':>7}")

        # Resume from the correct epoch
        current_step = start_step
        
        for epoch in range(start_epoch, num_epochs):
            self.train()
            epoch_train_loss = 0.0
            num_batches = 0

            # This is a cleaner way to handle batch skipping for the first resumed epoch
            batch_loader = enumerate(train_loader)
            if epoch == start_epoch and start_batch_in_epoch > 0:
                # Skip batches in first epoch if needed
                print(f"Skipping {start_batch_in_epoch} batches in epoch {epoch}...")
                for _ in range(start_batch_in_epoch):
                    try:
                        next(batch_loader)
                    except StopIteration:
                        # This shouldn't happen, but just in case
                        break

            for batch_idx, batch in batch_loader:
                # Make the step count absolute across epochs
                actual_batch_idx = batch_idx if epoch > start_epoch else batch_idx + start_batch_in_epoch
                current_step = epoch * steps_per_epoch + actual_batch_idx
                
                optimizer.zero_grad()
                x = batch[0]

                # Forward pass
                x, x_hat, f_x = self.forward(x)
                
                # Update feature tracking
                dead_ratio, stats = self.feature_tracker.update(f_x)

                # Lambda warmup - linear increase from 0 to final_lambda over first 5% of steps
                if current_step < warmup_steps:
                    self.lambda_l1 = (current_step / warmup_steps) * final_lambda
                else:
                    self.lambda_l1 = final_lambda
                
                # Learning rate decay - linear decay to zero over last 20% of steps
                if current_step >= decay_start_step:
                    progress = (current_step - decay_start_step) / (actual_total_steps - decay_start_step)
                    new_lr = learning_rate * (1 - progress)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr

                # Compute loss and update
                total_loss, L2_loss, L1_loss = self.compute_loss(x, x_hat, f_x)
                total_loss.backward()
                
                # Gradient clipping to norm 1
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_train_loss += total_loss.item()
                num_batches += 1
                
                # Force output flush for real-time monitoring
                sys.stdout.flush()

                # Periodic validation and logging
                if batch_idx % (len(train_loader) // 5) == 0:
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
                    
                    # Calculate sparsity
                    sparsity = (f_x.abs() >= self.feature_tracker.activation_threshold).float().mean().item()
                    
                    # Note: We intentionally do NOT save the model with lowest validation loss
                    # because the lowest loss would occur early when L1 penalty is minimal
                    
                    print(f"{current_step:8d} {epoch:5d} {total_loss.item():8.4f} {avg_val_loss:8.4f} "
                          f"{self.lambda_l1:5.2f} {dead_ratio*100:6.2f}% {sparsity*100:7.2f}%")
                    sys.stdout.flush()  # Force output flush
                    
                    # Save periodic checkpoints
                    if current_step % 50000 == 0:
                        checkpoint_path = f"{self.sae_model_path}.step{current_step}"
                        checkpoint = {
                            'model_state_dict': self.state_dict(),
                            'step': current_step,
                            'epoch': epoch,
                            'train_loss': total_loss.item(),
                            'val_loss': avg_val_loss,
                            'lambda_l1': self.lambda_l1,
                            'dead_ratio': dead_ratio,
                            'feature_tracker_samples': self.feature_tracker.samples_seen
                        }
                        torch.save(checkpoint, checkpoint_path)
                        print(f"Saved checkpoint at step {current_step} to {checkpoint_path}")
                        sys.stdout.flush()  # Force output flush
                    
                    self.train()

        # Get the latest validation results
        avg_val_loss = 0
        if 'avg_val_loss' in locals():
            avg_val_loss = avg_val_loss

        print(f"\nTraining completed:")
        print(f"Final validation loss: {avg_val_loss:.4f}")
        print(f"Final dead feature ratio: {dead_ratio:.1%}")
        print(f"Steps completed: {current_step}/{actual_total_steps}")
        print(f"Final λ: {self.lambda_l1:.2f}")
        
        # Save the final model (we want this one, not the "best" by loss)
        final_checkpoint = {
            'model_state_dict': self.state_dict(),
            'step': current_step,
            'epoch': epoch,
            'train_loss': total_loss.item() if 'total_loss' in locals() else None,
            'val_loss': avg_val_loss,
            'lambda_l1': self.lambda_l1,
            'dead_ratio': dead_ratio if 'dead_ratio' in locals() else None
        }
        torch.save(final_checkpoint, self.sae_model_path)
        print(f"Saved final model to {self.sae_model_path}")
        
        return self