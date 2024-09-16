import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset, random_split
from collections import deque


class SparseAutoencoder(nn.Module):
    def __init__(self, D, F, lambda_l1=1, device='cuda'):
        super(SparseAutoencoder, self).__init__()
        self.D = D  # Input dimension (residual stream dimension)
        self.F = F  # Number of features
        self.lambda_l1 = lambda_l1
        self.device = device

        # Encoder: W^enc ∈ ℝ^(F×D)
        self.encoder = nn.Linear(D, F)
        # Decoder: W^dec ∈ ℝ^(D×F)
        self.decoder = nn.Linear(F, D)

        # Initialize weights
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)

        # Learned biases
        self.b_enc = nn.Parameter(torch.zeros(F))
        self.b_dec = nn.Parameter(torch.zeros(D))

        # Move model to the specified device (GPU or CPU)
        self.to(device)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32)).to(self.device)
        elif isinstance(x, torch.Tensor) and x.device != self.device:
            x = x.to(self.device)

        f = torch.relu(self.encoder(x) + self.b_enc)
        x_hat = self.b_dec + self.decoder(f)
        return x, x_hat, f

    def loss_j(self, x, x_hat, f):
        # Sum along feature dimension
        L2_pen = torch.sum((x - x_hat)**2, dim=1)
        # Sum along feature dimension
        L1_pen = self.lambda_l1 * \
            torch.sum(f * torch.norm(self.decoder.weight, p=2, dim=0), dim=1)
        return L1_pen + L2_pen  # Return a tensor with one value per sample in the batch

    def preprocess(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X.astype(np.float32)).to(self.device)
        elif isinstance(X, torch.Tensor) and X.device != self.device:
            X = X.to(self.device)

        # Normalize the data
        C = torch.mean(torch.norm(X, p=2, dim=1)) / self.D
        X_normalized = X / C

        return X_normalized

    def train_and_validate(self, X_train, X_val, learning_rate, batch_size, num_epochs=1):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=1, verbose=True)

        # Preprocess training and validation data
        X_train = self.preprocess(X_train)
        X_val = self.preprocess(X_val)

        train_dataset = TensorDataset(X_train)
        val_dataset = TensorDataset(X_val)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False)

        for epoch in range(num_epochs):
            self.train()
            total_train_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                x = batch[0]
                x, x_hat, f = self.forward(x)
                loss = self.loss_j(x, x_hat, f)
                loss = torch.mean(loss)  # Ensure loss is a scalar
                loss.backward()
                print("Batch loss: " + loss.item())
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            # Validation step
            self.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0]
                    x, x_hat, f = self.forward(x)
                    loss = self.loss_j(x, x_hat, f)
                    loss = torch.mean(loss)  # Ensure loss is a scalar
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)

            # Step the scheduler
            scheduler.step(avg_val_loss)

            print(f"Epoch {epoch+1}, Average Train Loss: {
                  avg_train_loss:.4f}, Average Validation Loss: {avg_val_loss:.4f}")
            print(f"Current learning rate: {
                  optimizer.param_groups[0]['lr']:.6f}")

    def feature_vectors(self):
        return self.decoder.weight/torch.norm(self.decoder.weight, p=2, dim=0)

    def feature_activations(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32)).to(self.device)
        elif isinstance(x, torch.Tensor) and x.device != self.device:
            x = x.to(self.device)

        f = torch.relu(self.encoder(x) + self.b_enc)
        return f*torch.norm(self.decoder.weight, p=2, dim=0)
