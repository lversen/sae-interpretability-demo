import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    """Minimal sparse autoencoder for interpretability demos.

    Same core idea as the full version in src/models/SAE.py: an
    overcomplete hidden layer (n_features > n_in) trained with an L1
    penalty on the activations. The penalty forces only a few features
    to fire per input, which pushes each feature toward representing
    one distinct concept instead of many overlapping ones (superposition).
    """

    def __init__(self, n_in: int, n_features: int):
        super().__init__()
        self.W_enc = nn.Linear(n_in, n_features, bias=True)
        self.W_dec = nn.Linear(n_features, n_in, bias=True)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.W_enc(x))

    def forward(self, x: torch.Tensor):
        f = self.encode(x)
        x_hat = self.W_dec(f)
        return x_hat, f

    def loss(self, x: torch.Tensor, x_hat: torch.Tensor, f: torch.Tensor, l1_coef: float = 1e-3):
        recon = F.mse_loss(x_hat, x)
        sparsity = f.abs().mean()
        return recon + l1_coef * sparsity, recon, sparsity
