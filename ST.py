import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class SparseTransformer(nn.Module):
    def __init__(self, X, D, F, M, st_model_path, lambda_l1=0.1, device='cuda'):
        super(SparseTransformer, self).__init__()
        self.D = D
        self.F = F
        self.M = M
        self.st_model_path = st_model_path
        self.lambda_l1 = lambda_l1
        self.device = device
        # Initialize temperature to a smaller value for sharper attention
        self.temperature = nn.Parameter(torch.ones(1) * np.sqrt(M/4))
        
        # Initialize memory bank
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X.astype(np.float32))
        X_idx = np.random.choice(X.shape[0], size=self.F, replace=False)
        self.register_buffer('memory_bank', X[X_idx])
        
        # Initialize weights with constrained L2 norms
        self.W_q = nn.Linear(D, M, bias=True)
        self.W_k = nn.Linear(D, M, bias=True)
        self.W_v = nn.Linear(D, D, bias=True)
        
        # Initialize weights with random directions and fixed L2 norms
        with torch.no_grad():
            for layer in [self.W_q, self.W_k, self.W_v]:
                nn.init.orthogonal_(layer.weight)
                # Set L2 norms between 0.05 and 1
                norms = 0.05 + 0.95 * torch.rand(layer.weight.size(0))
                layer.weight.data = layer.weight.data * (norms / torch.norm(layer.weight.data, p=2, dim=1)).unsqueeze(1)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)  # Initialize biases to zero as per PDF

        self.to(device)

    def preprocess(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X.astype(np.float32)).to(self.device)
        elif isinstance(X, torch.Tensor) and X.device != self.device:
            X = X.to(self.device)

        # Scale dataset by single constant as per PDF
        C = torch.mean(torch.norm(X, p=2, dim=1)) / self.D
        X_normalized = X / C
        return X_normalized

    def compute_losses(self, x, x_hat, f, V):
        # L2 reconstruction loss normalized by input dimension
        L2_loss = torch.mean(torch.sum((x - x_hat)**2, dim=1)) / self.D
        
        # Modified sparsity penalty with minimum activation guarantee
        V_norms = torch.norm(V, p=2, dim=1)  # [F]
        feature_usage = torch.mean(f, dim=0)  # Average usage of each feature
        min_usage = 0.01  # Minimum desired feature usage
        
        # L1 sparsity term with feature usage regularization
        L1_term = torch.mean(torch.sum(f * V_norms[None, :], dim=1))
        usage_penalty = torch.mean(torch.relu(min_usage - feature_usage)) * 10.0
        
        L1_loss = (self.lambda_l1 * L1_term - usage_penalty) / self.D
        
        total_loss = L2_loss + L1_loss
        return total_loss, L2_loss, L1_loss

    def train_and_validate(self, X_train, X_val, learning_rate=1e-4, batch_size=4096, target_steps=200000):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=1e-5)
        
        X_train = self.preprocess(X_train)
        X_val = self.preprocess(X_val)
        
        train_dataset = TensorDataset(X_train)
        val_dataset = TensorDataset(X_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Calculate required epochs to reach target steps
        steps_per_epoch = len(train_loader)
        num_epochs = max(1, target_steps // steps_per_epoch)
        actual_total_steps = num_epochs * steps_per_epoch
        
        # Timing calculations
        warmup_steps = actual_total_steps // 20  # First 5% for lambda warmup
        decay_start = int(0.8 * actual_total_steps)  # Last 20% for learning rate decay
        final_lambda = 5.0
        
        print(f"Training for {actual_total_steps} total steps ({num_epochs} epochs)")
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Warmup steps: {warmup_steps}")
        
        best_val_loss = float('inf')
        step = 0
        
        for epoch in range(num_epochs):
            self.train()
            epoch_train_loss = 0.0
            num_batches = 0
            
            for i, (batch,) in enumerate(train_loader):
                # Lambda warmup
                if step < warmup_steps:
                    self.lambda_l1 = final_lambda * (step / warmup_steps)
                else:
                    self.lambda_l1 = final_lambda
                
                # Learning rate decay in last 20%
                if step > decay_start:
                    progress = (step - decay_start) / (actual_total_steps - decay_start)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = learning_rate * (1 - progress)
                
                optimizer.zero_grad()
                x, x_hat, f, V = self.forward(batch)
                total_loss, L2_loss, L1_loss = self.compute_losses(x, x_hat, f, V)
                total_loss.backward()
                
                # Clip gradient norm to 1.0
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                
                epoch_train_loss += total_loss.item()
                num_batches += 1
                step += 1
                
                # Log training progress
                if i % (len(train_loader) // 5) == 0:
                    dead_features = (torch.max(f, dim=0)[0] < 1e-3).float().mean().item()
                    print(f"Epoch {epoch}, Step {step}, Loss: {total_loss.item():.4f}, "
                        f"L2 Loss: {L2_loss.item():.4f}, L1 Loss: {L1_loss.item():.4f}, "
                        f"Dead features: {dead_features:.3f}, Lambda: {self.lambda_l1:.3f}")
            
            # Validation
            self.eval()
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    x, x_hat, f, V = self.forward(batch[0])
                    loss, _, _ = self.compute_losses(x, x_hat, f, V)
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            avg_train_loss = epoch_train_loss / num_batches
            
            print(f"\nEpoch {epoch} Summary:")
            print(f"Average Train Loss: {avg_train_loss:.6f}")
            print(f"Average Val Loss: {avg_val_loss:.6f}")
            print(f"Current λ: {self.lambda_l1:.3f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.state_dict(), self.st_model_path)
                print(f"New best validation loss: {best_val_loss:.6f}")
                
            print(f"Best Validation Loss: {best_val_loss:.6f}")

    def forward(self, x):
        # Forward pass implementation remains largely the same
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32)).to(self.device)
        elif isinstance(x, torch.Tensor) and x.device != self.device:
            x = x.to(self.device)
            
        Q = self.W_q(x)
        K = self.W_k(self.memory_bank)
        V = self.W_v(self.memory_bank)
        
        attention_scores = torch.matmul(Q, K.T) / self.temperature
        attention_scores = attention_scores - attention_scores.max(dim=1, keepdim=True)[0]
        f = torch.softmax(attention_scores, dim=1)
        
        x_hat = torch.matmul(f, V)
        
        return x, x_hat, f, V

    def feature_activations(self, x):
        """Get feature activations with L2 norm scaling as per PDF"""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32)).to(self.device)
        elif isinstance(x, torch.Tensor) and x.device != self.device:
            x = x.to(self.device)
        
        Q = self.W_q(x)
        K = self.W_k(self.memory_bank)
        V = self.W_v(self.memory_bank)
        
        attention_scores = torch.matmul(Q, K.T) / self.temperature
        f = torch.softmax(attention_scores, dim=1)
        
        # Scale activations by L2 norm of value vectors
        return f * torch.norm(V, p=2, dim=1)[None, :]