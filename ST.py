import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Union

class SparseTransformer(nn.Module):
    def __init__(self, X, D, F, M, st_model_path, lambda_l1=1, device='cuda'):
        super(SparseTransformer, self).__init__()
        self.D = D
        self.F = F
        self.M = M
        self.st_model_path = st_model_path
        self.lambda_l1 = lambda_l1
        self.device = device
        self.temperature = nn.Parameter(torch.ones(1) * np.sqrt(M))

        # Initialize memory bank
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X.astype(np.float32))
        X_idx = np.random.choice(X.shape[0], size=self.F, replace=False)
        self.register_buffer('memory_bank', X[X_idx])

        # Layernorm for inputs and memory bank
        self.input_norm = nn.LayerNorm(D)
        self.memory_norm = nn.LayerNorm(D)
        
        # Residual projection
        self.residual_proj = nn.Linear(D, D, bias=False)
        
        # Query, Key, Value projections
        self.W_q = nn.Linear(D, M, bias=True)
        self.W_k = nn.Linear(D, M, bias=True)
        self.W_v = nn.Linear(D, D, bias=True)
        
        # Initialize weights
        for layer in [self.W_q, self.W_k, self.W_v, self.residual_proj]:
            nn.init.orthogonal_(layer.weight)
            if hasattr(layer, 'bias') and layer.bias is not None:
                nn.init.zeros_(layer.bias)

        self.to(device)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32)).to(self.device)
        elif isinstance(x, torch.Tensor) and x.device != self.device:
            x = x.to(self.device)

        # Normalize input and memory
        x_normed = self.input_norm(x)
        memory_normed = self.memory_norm(self.memory_bank)
        
        # Project to query, key, value spaces
        Q = self.W_q(x_normed)  # [batch_size, M]
        K = self.W_k(memory_normed)  # [F, M]
        V = self.W_v(memory_normed)  # [F, D]

        # Compute attention scores with temperature scaling
        attention_scores = torch.matmul(Q, K.T) / self.temperature  # [batch_size, F]
        
        # Apply softmax with numerical stability
        attention_scores = attention_scores - attention_scores.max(dim=1, keepdim=True)[0]
        f = torch.softmax(attention_scores, dim=1)  # [batch_size, F]
        
        # Compute attended values
        attended = torch.matmul(f, V)  # [batch_size, D]
        
        # Add residual connection
        x_hat = attended + self.residual_proj(x)

        return x, x_hat, f, V

    def compute_losses(self, x, x_hat, f, V):
        # L2 reconstruction loss
        L2_loss = torch.mean(torch.sum((x - x_hat)**2, dim=1))
        
        # L1 sparsity penalty on attention weights
        V_norms = torch.norm(V, p=2, dim=1)  # [F]
        L1_loss = self.lambda_l1 * torch.mean(torch.sum(f * V_norms[None, :], dim=1))
        
        # Temperature regularization
        temp_reg = 0.01 * torch.abs(self.temperature - np.sqrt(self.M))
        
        total_loss = L2_loss + L1_loss + temp_reg
        return total_loss, L2_loss, L1_loss, temp_reg

    def preprocess(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X.astype(np.float32)).to(self.device)
        elif isinstance(X, torch.Tensor) and X.device != self.device:
            X = X.to(self.device)

        C = torch.mean(torch.norm(X, p=2, dim=1)) / self.D
        X_normalized = X / C

        return X_normalized
    def train_and_validate(self, X_train, X_val, learning_rate, batch_size, num_epochs=1, patience=3):
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.3, patience=patience//2, 
            verbose=True, min_lr=1e-6
        )

        X_train = self.preprocess(X_train)
        X_val = self.preprocess(X_val)

        train_dataset = TensorDataset(X_train)
        val_dataset = TensorDataset(X_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        best_val_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            self.train()
            total_train_loss = 0
            running_sparsity = []

            print(f"\nEpoch {epoch+1}")
            print(f"{'Batch':<10}{'Loss':<15}{'L2 Loss':<15}{'L1 Loss':<15}{'Temp':<10}{'Sparsity':<15}")
            print("-" * 80)

            for i, batch in enumerate(train_loader):
                optimizer.zero_grad()
                x = batch[0]
                x, x_hat, f, V = self.forward(x)
                
                # Compute losses
                total_loss, L2_loss, L1_loss, temp_reg = self.compute_losses(x, x_hat, f, V)
                total_loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                
                total_train_loss += total_loss.item()
                sparsity = (f < 1e-3).float().mean().item()
                running_sparsity.append(sparsity)

                if i % (len(train_loader) // 4) == 0:
                    print(f"{i:<10}{total_loss.item():<15.4f}{L2_loss.item():<15.4f}"
                          f"{L1_loss.item():<15.4f}{self.temperature.item():<10.4f}"
                          f"{sparsity:<15.4f}")

            avg_train_loss = total_train_loss / len(train_loader)
            avg_sparsity = np.mean(running_sparsity)

            # Validation
            self.eval()
            total_val_loss = 0
            val_sparsity = []
            
            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0]
                    x, x_hat, f, V = self.forward(x)
                    total_loss, _, _, _ = self.compute_losses(x, x_hat, f, V)
                    total_val_loss += total_loss.item()
                    val_sparsity.append((f < 1e-3).float().mean().item())

            avg_val_loss = total_val_loss / len(val_loader)
            avg_val_sparsity = np.mean(val_sparsity)

            # Print epoch summary
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            print(f"Train Sparsity: {avg_sparsity:.4f}, Val Sparsity: {avg_val_sparsity:.4f}")
            print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
            print(f"Temperature: {self.temperature.item():.4f}")

            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
                torch.save(self.state_dict(), self.st_model_path)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping after {patience} epochs without improvement.")
                    self.load_state_dict(torch.load(self.st_model_path, weights_only=True))
                    break

    def feature_activations(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32)).to(self.device)
        elif isinstance(x, torch.Tensor) and x.device != self.device:
            x = x.to(self.device)

        x = self.input_norm(x)
        memory = self.memory_norm(self.memory_bank)
        
        Q = self.W_q(x)
        K = self.W_k(memory)
        V = self.W_v(memory)

        attention_scores = torch.matmul(Q, K.T) / self.temperature
        f = torch.softmax(attention_scores, dim=1)

        return f * torch.norm(V, p=2, dim=1)[None, :]