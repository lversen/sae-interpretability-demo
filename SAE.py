import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
class SparseAutoencoder(nn.Module):
    def __init__(self, X, F, batch_size):
        super(SparseAutoencoder, self).__init__()
        self.D = X.shape[1]
        self.encoder = nn.Linear(F, self.D)

        self.decoder = nn.Linear(self.D, F)     
    def normalize(self, x):
        c = np.average(np.sqrt(np.sum(x**2, axis=1)))/self.D
        return(x/c)
    
    def forward(self, x):
        x = self.normalize(x)
        x_t = torch.from_numpy(x.astype(np.float32))
        W_enc = self.encoder(x_t)
