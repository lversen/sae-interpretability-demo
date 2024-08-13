import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
        L2_pen = torch.sum((x - x_hat)**2)
        L1_pen = self.lambda_l1*(f@torch.norm(self.decoder.weight, p=2, dim=0))
        return L1_pen + L2_pen
    
    def preprocess(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X.astype(np.float32)).to(self.device)
        elif isinstance(X, torch.Tensor) and X.device != self.device:
            X = X.to(self.device)
        
        C = torch.mean(torch.norm(X, p=2, dim=1))/self.D
        X /= C
        return X


    def train_and_validate(self, X, learning_rate, batch_size, num_epochs=1):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        X = self.preprocess(X)
        dataset = TensorDataset(X)
        
        # Split the dataset into train and validation sets
        train_size = int(0.75 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        for epoch in range(num_epochs):
            self.train()
            total_train_loss = 0
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                batch_loss = 0
                for x in batch[0]:
                    x, x_hat, f = self.forward(x)
                    loss_j = self.loss_j(x, x_hat, f)
                    batch_loss += loss_j
                
                batch_loss /= len(batch[0])
                batch_loss.backward()
                optimizer.step()
                
                total_train_loss += batch_loss.item()
                print(f"Epoch {epoch+1}, Train Batch {batch_idx+1}/{len(train_loader)}, Loss: {batch_loss.item():.4f}")
            
            avg_train_loss = total_train_loss / len(train_loader)
            
            # Validation step
            self.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    batch_loss = 0
                    for x in batch[0]:
                        x, x_hat, f = self.forward(x)
                        loss_j = self.loss_j(x, x_hat, f)
                        batch_loss += loss_j
                    
                    batch_loss /= len(batch[0])
                    total_val_loss += batch_loss.item()
                    print(f"Epoch {epoch+1}, Val Batch {batch_idx+1}/{len(val_loader)}, Loss: {batch_loss.item():.4f}")
            
            avg_val_loss = total_val_loss / len(val_loader)
            
            print(f"Epoch {epoch+1}, Average Train Loss: {avg_train_loss:.4f}, Average Validation Loss: {avg_val_loss:.4f}")
            print("-" * 50)

    def feature_vectors(self):
        return self.decoder.weight/torch.norm(self.decoder.weight, p=2, dim=0)
    
    def feature_activations(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32)).to(self.device)
        elif isinstance(x, torch.Tensor) and x.device != self.device:
            x = x.to(self.device)
        
        f = torch.relu(self.encoder(x) + self.b_enc)
        return f*torch.norm(self.decoder.weight, p=2, dim=0)