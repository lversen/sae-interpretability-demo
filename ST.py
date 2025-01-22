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
        self.X = X
        self.D = D
        self.F = F
        self.M = M
        self.st_model_path = st_model_path
        self.lambda_l1 = lambda_l1
        self.device = device

        self.W_q = nn.Linear(D, M)
        self.W_k = nn.Linear(D, M)
        self.W_v = nn.Linear(D, D)

        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)

        self.to(device)

    def typecheck(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32)).to(self.device)
            return (x)
        elif isinstance(x, torch.Tensor) and x.device != self.device:
            x = x.to(self.device)
            return (x)

    def forward(self, x):
        X = self.X
        X_idx = np.random.choice(X.shape[0], size=self.F, replace=False)
        self.X = X[X_idx]
        X = self.X
        x = self.typecheck(x)
        X = self.typecheck(X)

        Q = self.W_q(x)  # Shape: [batch_size, M]
        K = self.W_k(X)  # Shape: [num_cross, M]
        V = self.W_v(X)  # Shape: [F, num_cross]

        # Compute attention scores and apply softmax along dim=1 (cross attention dimension)
        attention_scores = torch.matmul(
            # M is attention dim
            Q, K.T) / torch.sqrt(torch.tensor(self.M).float())
        f = torch.softmax(attention_scores, dim=1)  # Add dim=1 parameter

        x_hat = torch.matmul(f, V)

        return x, x_hat, f, V

    def loss_j(self, x, x_hat, f, V):
        L2_pen = torch.sum((x - x_hat)**2, dim=1)
        L1_pen = self.lambda_l1 * \
            torch.sum(f * torch.norm(V, p=2, dim=1), dim=1)

        return L1_pen + L2_pen

    def preprocess(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X.astype(np.float32)).to(self.device)
        elif isinstance(X, torch.Tensor) and X.device != self.device:
            X = X.to(self.device)

        C = torch.mean(torch.norm(X, p=2, dim=1)) / self.D
        X_normalized = X / C

        return X_normalized

    def train_and_validate(self, X_train, X_val, learning_rate, batch_size, num_epochs=1, patience=3, batch_indices: Union[List[int], str] = 'auto'):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=1, verbose=True)

        X_train = self.preprocess(X_train)
        X_val = self.preprocess(X_val)

        train_dataset = TensorDataset(X_train)
        val_dataset = TensorDataset(X_val)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False)

        best_val_loss = float('inf')
        epochs_without_improvement = 25

        # Determine which batches to analyze
        num_batches = len(train_loader)
        if batch_indices == 'auto':
            batch_indices = [num_batches // 5, num_batches //
                             2, num_batches * 4 // 5, num_batches - 1]
        elif isinstance(batch_indices, list):
            # Ensure indices are within range
            batch_indices = [i % num_batches for i in batch_indices]
        else:
            raise ValueError(
                "batch_indices must be 'auto' or a list of integers")

        for epoch in range(num_epochs):
            self.train()
            total_train_loss = 0

            print(f"\nEpoch {epoch+1}")
            print(f"{'Batch':<10}{'Loss':<15}{'Input Mean':<15}{'Input Std':<15}{
                  'Output Mean':<15}{'Output Std':<15}{'Feature Sparsity':<20}")
            print("-" * 100)

            for i, batch in enumerate(train_loader):
                optimizer.zero_grad()
                x = batch[0]
                x, x_hat, f, V = self.forward(x)
                loss = self.loss_j(x, x_hat, f, V)
                batch_loss = torch.mean(loss).item()
                loss = torch.mean(loss)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

                if i in batch_indices:
                    input_mean = x.mean().item()
                    input_std = x.std().item()
                    output_mean = x_hat.mean().item()
                    output_std = x_hat.std().item()
                    feature_sparsity = (f == 0).float().mean().item()

                    print(f"{i:<10}{batch_loss:<15.4f}{input_mean:<15.4f}{input_std:<15.4f}{
                          output_mean:<15.4f}{output_std:<15.4f}{feature_sparsity:<20.4f}")

            avg_train_loss = total_train_loss / len(train_loader)

            self.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0]
                    x, x_hat, f, V = self.forward(x)
                    loss = self.loss_j(x, x_hat, f, V)
                    loss = torch.mean(loss)
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)

            scheduler.step(avg_val_loss)

            print(f"\nEpoch {epoch+1} Summary:")
            print(f"Average Train Loss: {
                  avg_train_loss:.4f}, Average Validation Loss: {avg_val_loss:.4f}")
            print(f"Current learning rate: {
                  optimizer.param_groups[0]['lr']:.6f}")

            # Early stopping logic
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
                torch.save(self.state_dict(), self.st_model_path)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered. No improvement for {
                          patience} epochs.")
                    self.load_state_dict(torch.load(self.st_model_path))
                    break

    def feature_activations(self, x):
        x = self.typecheck(x)
        X = self.X
        X = self.typecheck(X)

        Q = self.W_q(x)
        K = self.W_k(X)
        V = self.W_v(X)

        attention_scores = torch.matmul(Q, K.T) / self.D
        f = torch.softmax(attention_scores, dim=1)  # Add dim=1 parameter

        return f * torch.norm(V, p=2, dim=1)
