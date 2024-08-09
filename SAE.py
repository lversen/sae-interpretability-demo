import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import deque

class SparseAutoencoder(nn.Module):
    def __init__(self, D, F, lambda_l1=1):
        super(SparseAutoencoder, self).__init__()
        self.D = D  # Input dimension (residual stream dimension)
        self.F = F  # Number of features
        self.lambda_l1 = lambda_l1

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

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32))
        f = torch.relu(self.encoder(x) + self.b_enc)
        x_hat = self.b_dec + self.decoder(f)
        return(x, x_hat, f)

    def loss_j(self, x, x_hat, f):
        L2_pen = torch.sum((x - x_hat)**2)
        L1_pen = self.lambda_l1*(f@torch.norm(self.decoder.weight, p=2, dim=0))
        return(L1_pen + L2_pen)
    
    def preprocess(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X.astype(np.float32))
        C = torch.mean(torch.norm(X, p=2, dim=1))/self.D
        X /= C
        return(X)

    def train(self, X, learning_rate, batch_size, num_epochs=1):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        X = self.preprocess(X)
        dataset = TensorDataset(X)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()
                batch_loss = 0
                for x in batch[0]:
                    x, x_hat, f = self.forward(x)
                    loss_j = self.loss_j(x, x_hat, f)
                    batch_loss += loss_j
                
                batch_loss /= len(batch[0])
                print(batch_loss.item())
                batch_loss.backward()
                optimizer.step()
                
                total_loss += batch_loss.item()
                
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}, Average Loss: {avg_loss}")
    
    def feature_vectors(self):
        return(self.decoder.weight/torch.norm(self.decoder.weight, p=2, dim=0))
    
    def feature_activations(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32))
        f = torch.relu(self.encoder(x) + self.b_enc)
        return(f*torch.norm(self.decoder.weight, p=2, dim=0))