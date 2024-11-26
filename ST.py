import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Union
# Cross attention using x and x2
# X_2 use same datapoints size F
class SparseTransformer(nn.Module):
    def __init__(self, D, F, sae_model_path, lambda_l1=1, device='cuda', chunk_size=128):
        super(SparseTransformer, self).__init__()
        self.D = D
        self.F = F
        self.sae_model_path = sae_model_path
        self.lambda_l1 = lambda_l1
        self.device = device
        self.chunk_size = chunk_size  # Add chunk size parameter

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(D, D)
        self.W_k = nn.Linear(D, D)
        self.W_v = nn.Linear(D, D)
# =============================================================================
#         self.W_o = nn.Linear(F, D)
# =============================================================================
        
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
# =============================================================================
#         nn.init.xavier_uniform_(self.W_o.weight)
# =============================================================================
        
        self.to(device)
    
    def process_chunk(self, x_chunk, K, V):
        """Process a chunk of queries against all keys/values"""
        Q = self.W_q(x_chunk)  # Shape: (chunk_size, F)
        
        # Compute attention scores for this chunk
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))  # Shape: (chunk_size, n)
        attention_scores = attention_scores / np.sqrt(self.F)
        
        # Apply softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # Apply attention weights to values
        attended_values = torch.matmul(attention_weights, V)  # Shape: (chunk_size, F)
        
        # Project back to input dimension
        x_hat_chunk = self.W_o(attended_values)  # Shape: (chunk_size, D)
        torch.cuda.empty_cache()
        return x_hat_chunk, attention_weights

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32)).to(self.device)
        elif isinstance(x, torch.Tensor) and x.device != self.device:
            x = x.to(self.device)

        batch_size = x.size(0)
        
        # Compute K, V for all inputs once
        K = self.W_k(x)  # Shape: (batch_size, F)
        V = self.W_v(x)  # Shape: (batch_size, F)

        # Process in chunks
        x_hat_chunks = []
        attention_weights_chunks = []
        
        for i in range(0, batch_size, self.chunk_size):
            end_idx = min(i + self.chunk_size, batch_size)
            x_chunk = x[i:end_idx]
            
            # Process chunk
            x_hat_chunk, attention_weights_chunk = self.process_chunk(x_chunk, K, V)
            
            x_hat_chunks.append(x_hat_chunk)
            attention_weights_chunks.append(attention_weights_chunk)


        # Concatenate results
        x_hat = torch.cat(x_hat_chunks, dim=0)
        attention_weights = torch.cat(attention_weights_chunks, dim=0)

        return x, x_hat, attention_weights

    def loss_j(self, x, x_hat, f):
        # Use chunked computation for loss if needed
        batch_size = x.size(0)
        total_l2_pen = 0
        total_l1_pen = 0
        
        for i in range(0, batch_size, self.chunk_size):
            end_idx = min(i + self.chunk_size, batch_size)
            
            # Compute reconstruction loss for chunk
            l2_pen = torch.sum((x[i:end_idx] - x_hat[i:end_idx])**2, dim=1)
            total_l2_pen += torch.sum(l2_pen)
            
            # Compute sparsity loss for chunk
            l1_pen = self.lambda_l1 * torch.sum(torch.abs(f[i:end_idx]), dim=1)
            total_l1_pen += torch.sum(l1_pen)

        # Average the losses
        avg_loss = (total_l2_pen + total_l1_pen) / batch_size
        return avg_loss

    def train_and_validate(self, X_train, X_val, learning_rate, batch_size, num_epochs=1, patience=3, batch_indices: Union[List[int], str] = 'auto'):
        # Adjust batch size to be no larger than chunk size
        batch_size = min(batch_size, self.chunk_size)
        
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)

        X_train = self.preprocess(X_train)
        X_val = self.preprocess(X_val)

        train_dataset = TensorDataset(X_train)
        val_dataset = TensorDataset(X_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        best_val_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            self.train()
            total_train_loss = 0

            for batch in train_loader:
                optimizer.zero_grad()
                x = batch[0]
                x, x_hat, f = self.forward(x)
                loss = self.loss_j(x, x_hat, f)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

                # Clear cache after each batch
                torch.cuda.empty_cache()

            self.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0]
                    x, x_hat, f = self.forward(x)
                    loss = self.loss_j(x, x_hat, f)
                    total_val_loss += loss.item()

                    # Clear cache after each batch
                    torch.cuda.empty_cache()

            avg_train_loss = total_train_loss / len(train_loader)
            avg_val_loss = total_val_loss / len(val_loader)

            print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
            
            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
                torch.save(self.state_dict(), self.sae_model_path)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered. No improvement for {patience} epochs.")
                    self.load_state_dict(torch.load(self.sae_model_path))
                    break

    def preprocess(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X.astype(np.float32)).to(self.device)
        elif isinstance(X, torch.Tensor) and X.device != self.device:
            X = X.to(self.device)

        C = torch.mean(torch.norm(X, p=2, dim=1)) / self.D
        X_normalized = X / C

        return X_normalized
    
    def feature_activations(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32)).to(self.device)
        elif isinstance(x, torch.Tensor) and x.device != self.device:
            x = x.to(self.device)
    
        batch_size = x.size(0)
        feature_activations_list = []
        
        # Process in chunks to reduce memory usage
        for i in range(0, batch_size, self.chunk_size):
            end_idx = min(i + self.chunk_size, batch_size)
            x_chunk = x[i:end_idx]
            
            # Compute Q, K, V for this chunk
            Q = self.W_q(x_chunk)  # Shape: (chunk_size, F)
            K = self.W_k(x_chunk)  # Shape: (chunk_size, F)
            V = self.W_v(x_chunk)  # Shape: (chunk_size, F)
            
            # Compute attention scores and apply ReLU for sparsity
            attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.F)
            attention_weights = torch.softmax(attention_scores, dim=-1)
            
            # Combine attention and values (similar to SAE's feature*decoder_norm)
            features = torch.relu(torch.matmul(attention_weights, V))
            # Scale by output projection norm (analogous to SAE's decoder norm)
            features = features * torch.norm(self.W_o.weight, p=2, dim=0)
            
            feature_activations_list.append(features)
            
            # Clean up memory
            del Q, K, V, attention_scores, attention_weights
            torch.cuda.empty_cache()
    
        # Concatenate results
        try:
            return torch.cat(feature_activations_list, dim=0)
        finally:
            del feature_activations_list
            torch.cuda.empty_cache()