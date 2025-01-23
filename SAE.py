import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Union

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
        Compute loss and return scalar value for backpropagation
        """
        # L2 reconstruction term
        L2_loss = torch.mean(torch.sum((x - x_hat)**2, dim=1))
        
        # Sparsity penalty
        W_d_norms = torch.norm(self.W_d.weight, p=2, dim=0)  # Column L2 norms
        L1_penalty = self.lambda_l1 * torch.mean(torch.sum(f_x * W_d_norms, dim=1))
        
        # Normalize by input dimension as per paper
        total_loss = (L2_loss + L1_penalty) / self.n
        return total_loss, L2_loss / self.n, L1_penalty / self.n

    def preprocess(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X.astype(np.float32)).to(self.device)
        elif isinstance(X, torch.Tensor) and X.device != self.device:
            X = X.to(self.device)

        C = torch.mean(torch.norm(X, p=2, dim=1)) / self.n
        X_normalized = X / C
        return X_normalized

    def feature_vectors(self):
        return self.W_d.weight / torch.norm(self.W_d.weight, p=2, dim=0)

    def feature_activations(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32)).to(self.device)
        elif isinstance(x, torch.Tensor) and x.device != self.device:
            x = x.to(self.device)

        f_x = torch.relu(self.W_e(x) + self.b_e)
        return f_x * torch.norm(self.W_d.weight, p=2, dim=0)

    def train_and_validate(self, X_train, X_val, learning_rate=1e-3, batch_size=4096, num_epochs=100):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        
        X_train = self.preprocess(X_train)
        X_val = self.preprocess(X_val)

        train_dataset = TensorDataset(X_train)
        val_dataset = TensorDataset(X_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        total_steps = len(train_loader) * num_epochs
        decay_start_step = int(total_steps * 0.8)  # Start decay at 80% of training
        
        total_steps = len(train_loader) * num_epochs
        warmup_steps = total_steps // 20
        final_lambda = 5.0

        best_val_loss = float('inf')
        step = 0

        print(f"Training for {total_steps} total steps with {warmup_steps} warmup steps")
        print(f"{'Epoch':>5} {'Step':>8} {'Train Loss':>12} {'L2 Loss':>12} {'L1 Loss':>12} {'Val Loss':>12} {'λ':>8} {'Sparsity':>10}")
        print("-" * 80)

        for epoch in range(num_epochs):
            self.train()
            epoch_train_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                x = batch[0]

                x, x_hat, f_x = self.forward(x)

                if step < warmup_steps:
                    self.lambda_l1 = (step / warmup_steps) * final_lambda
                else:
                    self.lambda_l1 = final_lambda
                
                current_step = epoch * len(train_loader) + batch_idx

                # Linear learning rate decay in last 20% of training
                if current_step >= decay_start_step:
                    progress = (current_step - decay_start_step) / (total_steps - decay_start_step)
                    new_lr = learning_rate * (1 - progress)  # Linear decay to zero
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                total_loss, L2_loss, L1_loss = self.compute_loss(x, x_hat, f_x)
                total_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_train_loss += total_loss.item()
                num_batches += 1
                step += 1

                sparsity = (f_x < 1e-3).float().mean().item()

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
                    
                    print(f"{epoch:5d} {step:8d} {total_loss.item():12.6f} {L2_loss.item():12.6f} "
                          f"{L1_loss.item():12.6f} {avg_val_loss:12.6f} {self.lambda_l1:8.3f} {sparsity:10.3f}")

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        torch.save(self.state_dict(), self.sae_model_path)
                    
                    self.train()

            avg_train_loss = epoch_train_loss / num_batches
            print(f"\nEpoch {epoch} Summary:")
            print(f"Average Train Loss: {avg_train_loss:.6f}")
            print(f"Best Validation Loss: {best_val_loss:.6f}")
            print(f"Current λ: {self.lambda_l1:.3f}")
            
            with torch.no_grad():
                all_features = self.feature_activations(X_val)
                dead_feature_ratio = (torch.max(all_features, dim=0)[0] < 1e-3).float().mean().item()
                print(f"Dead Feature Ratio: {dead_feature_ratio:.3f}")
            
            if dead_feature_ratio > 0.01:
                print("Warning: More than 1% dead features detected")