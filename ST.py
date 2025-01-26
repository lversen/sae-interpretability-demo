from sparsemax import Sparsemax
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple

class SparseTransformer(nn.Module):
    def __init__(self, X, D: int, F: int, M: int, st_model_path: str, 
                 lambda_l1: float = 5.0, device: str = 'cuda'):
        super(SparseTransformer, self).__init__()
        self.D = D  # Input dimension
        self.F = F  # Feature dimension
        self.M = M  # Attention dimension
        self.st_model_path = st_model_path
        self.lambda_l1 = lambda_l1
        self.device = device

        # Initialize transformations
        self.W_q = nn.Linear(D, M, bias=True)
        self.W_k = nn.Linear(D, M, bias=True)
        self.W_v = nn.Linear(D, D, bias=True)
        
        # Initialize sparsemax - using simple implementation
        self.sparsemax = Sparsemax(dim=-1)
        
        # Initialize feature tracking statistics
        self.feature_usage_count = torch.zeros(self.F)
        self.samples_seen = 0
        self.usage_history = []
        
        self.initialize_weights()
        self.initialize_feature_subset(X)
        self.to(device)
        
    def initialize_feature_subset(self, X):
        """Initialize feature subset more carefully."""
        # Ensure X is on CPU for numpy operations
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        
        # Compute mean and std of each sample
        sample_stats = np.sqrt(np.sum(X * X, axis=1))
        
        # Sort by magnitude and take evenly spaced samples
        sorted_indices = np.argsort(sample_stats)
        step = len(sorted_indices) / self.F
        selected_indices = [int(i * step) for i in range(self.F)]
        
        self.X_data = X[sorted_indices[selected_indices]]

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32)).to(self.device)
        elif isinstance(x, torch.Tensor) and x.device != self.device:
            x = x.to(self.device)
            
        # Ensure X_data is on correct device
        if isinstance(self.X_data, np.ndarray):
            self.X_data = torch.from_numpy(self.X_data.astype(np.float32)).to(self.device)
        elif isinstance(self.X_data, torch.Tensor) and self.X_data.device != self.device:
            self.X_data = self.X_data.to(self.device)
            
        # Compute transformations
        Q = self.W_q(x)  # [batch_size, M]
        K = self.W_k(self.X_data)  # [F, M]
        V = self.W_v(self.X_data)  # [F, D]
        
        # Scale Q and K for more stable attention scores
        Q = Q / np.sqrt(self.M)
        K = K / np.sqrt(self.M)
        
        # Compute attention scores (no additional scaling needed)
        attention_scores = torch.matmul(Q, K.T)
        
        # Apply sparsemax
        attention_weights = self.sparsemax(attention_scores)
        
        # Update feature statistics during training
        if self.training:
            self.update_feature_stats(attention_weights)
        
        # Compute reconstruction
        x_hat = torch.matmul(attention_weights, V)
        
        return x, x_hat, attention_weights

    def feature_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Get feature activations with simplified sparsemax attention."""
        with torch.no_grad():
            Q = self.W_q(x)
            K = self.W_k(self.X_data)
            V = self.W_v(self.X_data)
            
            attention_scores = torch.matmul(Q, K.T) / np.sqrt(self.M)
            attention_weights = self.sparsemax(attention_scores)
            
            # Scale by value norms
            V_norms = torch.norm(V, p=2, dim=1)
            return attention_weights * V_norms[None, :]
    
    def reset_feature_stats(self):
        """Initialize or reset feature utilization statistics."""
        self.feature_usage_count = torch.zeros(self.F, device=self.device)
        self.samples_seen = 0
        self.usage_history = []
    
    def initialize_weights(self):
        """Initialize weights with controlled scaling."""
        with torch.no_grad():
            # Initialize all weights with small uniform values
            bound = 0.01  # Small initial bound
            
            # Query transformation
            nn.init.uniform_(self.W_q.weight, -bound, bound)
            if self.W_q.bias is not None:
                nn.init.zeros_(self.W_q.bias)
            
            # Key transformation (same scale as query)
            nn.init.uniform_(self.W_k.weight, -bound, bound)
            if self.W_k.bias is not None:
                nn.init.zeros_(self.W_k.bias)
            
            # Value transformation (can be slightly larger)
            nn.init.uniform_(self.W_v.weight, -bound*2, bound*2)
            if self.W_v.bias is not None:
                nn.init.zeros_(self.W_v.bias)

    def update_feature_stats(self, attention_weights: torch.Tensor):
        """Update feature utilization statistics."""
        # Ensure tensors are on the same device
        if self.feature_usage_count.device != attention_weights.device:
            self.feature_usage_count = self.feature_usage_count.to(attention_weights.device)
        
        # Count features above zero (sparsemax can produce true zeros)
        active_features = (attention_weights > 0).float()
        self.feature_usage_count += active_features.sum(dim=0)
        self.samples_seen += attention_weights.size(0)
        
        # Compute current usage statistics
        if self.samples_seen % 1000 == 0:  # Log every 1000 samples
            feature_usage = (self.feature_usage_count / self.samples_seen).cpu().numpy()
            self.usage_history.append({
                'step': self.samples_seen,
                'active_features': (feature_usage > 0).mean(),
                'usage_std': feature_usage.std(),
                'max_usage': feature_usage.max()
            })
        
        return {
            'active_features': (active_features.mean(dim=0) > 0).float().mean().item(),
            'attention_sparsity': (attention_weights == 0).float().mean().item()
        }



    def compute_losses(self, x: torch.Tensor, x_hat: torch.Tensor, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute losses with explicit sparsity monitoring."""
        # L2 reconstruction loss
        L2_loss = torch.mean(torch.sum((x - x_hat)**2, dim=1)) / self.D
        
        # L1 regularization on attention weights
        L1_loss = self.lambda_l1 * torch.mean(torch.sum(f, dim=1))
        
        total_loss = L2_loss + L1_loss
        return total_loss, L2_loss, L1_loss

    def train_and_validate(self, X_train, X_val, learning_rate=5e-5, batch_size=4096, target_steps=200000):
        """Train with enhanced sparsity monitoring."""
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        
        # Preprocess data
        X_train = self.preprocess(X_train)
        X_val = self.preprocess(X_val)
        
        train_dataset = TensorDataset(X_train)
        val_dataset = TensorDataset(X_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        steps_per_epoch = len(train_loader)
        num_epochs = max(1, target_steps // steps_per_epoch)
        total_steps = num_epochs * steps_per_epoch
        warmup_steps = total_steps // 20
        final_lambda = self.lambda_l1
        
        print("\nEnhanced Training Configuration:")
        print(f"Total Steps: {total_steps}")
        print(f"Epochs: {num_epochs}")
        print(f"Batch Size: {batch_size}")
        
        print("\nMetrics:")
        print("  Loss    - Training loss")
        print("  ValLoss - Validation loss")
        print("  λ       - L1 regularization strength")
        print("  Active% - Percentage of active features")
        print("  Sparse% - Percentage of zero attention weights")
        
        print(f"\n{'Step':>8} {'Epoch':>5} {'Loss':>8} {'ValLoss':>8} {'λ':>5} {'Active%':>7} {'Sparse%':>7}")
        
        best_val_loss = float('inf')
        step = 0
        
        self.reset_feature_stats()  # Reset feature tracking at start of training
        
        for epoch in range(num_epochs):
            self.train()
            for batch, in train_loader:
                # Lambda warmup
                if step < warmup_steps:
                    self.lambda_l1 = final_lambda * (step / warmup_steps)
                
                optimizer.zero_grad()
                x, x_hat, f = self.forward(batch)
                
                total_loss, L2_loss, L1_loss = self.compute_losses(x, x_hat, f)
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                
                # Periodic validation and logging
                if step % (steps_per_epoch // 5) == 0:
                    self.eval()
                    val_loss = 0.0
                    val_batches = 0
                    val_stats = {'active_features': 0.0, 'attention_sparsity': 0.0}
                    
                    with torch.no_grad():
                        for val_batch, in val_loader:
                            x, x_hat, f = self.forward(val_batch)
                            batch_loss, _, _ = self.compute_losses(x, x_hat, f)
                            val_loss += batch_loss.item()
                            val_batches += 1
                            
                            # Update validation statistics
                            active = (f > 0).float()
                            val_stats['active_features'] += (active.mean(dim=0) > 0).float().mean().item()
                            val_stats['attention_sparsity'] += (f == 0).float().mean().item()
                        
                        avg_val_loss = val_loss / val_batches
                        val_stats['active_features'] /= val_batches
                        val_stats['attention_sparsity'] /= val_batches
                        
                        print(f"{step:8d} {epoch:5d} {total_loss.item():8.4f} {avg_val_loss:8.4f} "
                              f"{self.lambda_l1:5.2f} {val_stats['active_features']:7.1%} "
                              f"{val_stats['attention_sparsity']:7.1%}")
                        
                        if avg_val_loss < best_val_loss:
                            best_val_loss = avg_val_loss
                            # Save only the model parameters, not buffers
                            torch.save(self.state_dict(), self.st_model_path)
                    
                    self.train()
                
                step += 1
        
        # Print final sparsity analysis
        with torch.no_grad():
            feature_usage = (self.feature_usage_count / self.samples_seen).cpu().numpy()
            print(f"\nTraining completed:")
            print(f"Best validation loss: {best_val_loss:.4f}")
            print(f"Final active features: {(feature_usage > 0).mean():.1%}")
            print(f"Feature usage std: {feature_usage.std():.4f}")
            print(f"Steps completed: {step}/{total_steps}")
            print(f"Final λ: {self.lambda_l1:.2f}")

    def preprocess(self, X):
        """Preprocess input data."""
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X.astype(np.float32)).to(self.device)
        elif isinstance(X, torch.Tensor) and X.device != self.device:
            X = X.to(self.device)
        
        C = torch.mean(torch.norm(X, p=2, dim=1)) / self.D
        return X / C

