import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Union
from SAE import SparseAutoencoder

class SparseTransformer(nn.Module):
    def __init__(self, D, F, sae_model_path, lambda_l1=1, device='cuda'):
        super(SparseTransformer, self).__init__()
        self.W_q = nn.Linear(D, F)
        self.W_k = nn.Linear(D, F)
        self.W_v = nn.Linear(D, F)
        
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        
        self.to(device)
    
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32)).to(self.device)
        elif isinstance(x, torch.Tensor) and x.device != self.device:
            x = x.to(self.device)
        self.Q = self.W_q(x)
        self.K = self.W_k(x)
        self.V = self.W_v(x)
        x_hat = nn.Softmax(torch.matmul(self.Q, self.K.transpose()))
