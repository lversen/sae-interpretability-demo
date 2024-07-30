import torch
import torch.nn as nn
import torch.optim as optim

class SparseAutoencoder(nn.Module):
    def __init__(self, activation_matrix, hidden_dim, lambda_l1):
        super(SparseAutoencoder, self).__init__()
        self.input_dim = activation_matrix.shape[1]  # D is inferred from the input matrix
        self.hidden_dim = hidden_dim  # F features
        self.lambda_l1 = lambda_l1

        # Encoder
        self.encoder = nn.Linear(self.input_dim, self.hidden_dim)
        
        # Decoder
        self.decoder = nn.Linear(self.hidden_dim, self.input_dim)

        # Initialize weights
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)

    def normalize_input(self, x):
        # Scalar normalization
        print(x.shape)
        norm = torch.norm(x, p=2, dim=1, keepdim=True)
        return x / norm * (self.input_dim ** 0.5)

    def encode(self, x):
        return torch.relu(self.encoder(x))

    def decode(self, encoded):
        return self.decoder(encoded)

    def forward(self, x):
        # Normalize input
        x_normalized = self.normalize_input(x)
        
        # Encode
        encoded = self.encode(x_normalized)
        
        # Decode
        decoded = self.decode(encoded)
        
        return decoded, encoded


    def loss_function(self, x, x_hat, encoded):
        # Reconstruction loss (L2 penalty)
        mse_loss = nn.MSELoss()(x_hat, x)
        
        # L1 penalty on feature activations
        l1_loss = self.lambda_l1 * torch.sum(torch.abs(encoded) * torch.norm(self.decoder.weight, p=2, dim=0))
        
        return mse_loss + l1_loss

