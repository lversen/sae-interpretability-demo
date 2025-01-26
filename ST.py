import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple
from deadfeatures import DeadFeatureTracker

class SparseTransformer(nn.Module):
    def __init__(self, X, D: int, F: int, M: int, st_model_path: str, 
                 lambda_l1: float = 5.0, device: str = 'cuda'):
        super(SparseTransformer, self).__init__()
        self.D = D  # Input dimension
        self.F = F  # Feature dimension
        self.M = M  # Attention dimension
        self.st_model_path = st_model_path
        self.lambda_l1 = lambda_l1
        self.device = device

        # Initialize transformations with separate bias terms
        self.W_q = nn.Linear(D, M)
        self.W_k = nn.Linear(D, M)
        self.W_v = nn.Linear(D, D)
        
        # Temperature parameter for attention scaling
        self.temperature = nn.Parameter(torch.ones(1) * np.sqrt(M))
        
        # Initialize feature tracking
        self.feature_tracker = DeadFeatureTracker(
            num_features=self.F,
            activation_threshold=1e-8,
            window_size=10_000_000,
            update_interval=10_000
        )
        
        self.initialize_weights()
        self.initialize_feature_subset(X)
        self.to(device)
        
    def initialize_feature_subset(self, X):
        """Initialize feature subset using simple random selection.
        
        Args:
            X: Input data array (numpy array or torch tensor)
        """
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
            
        n_samples = len(X)
        
        # Use numpy's random choice for efficient selection
        selected_indices = np.random.choice(
            n_samples, 
            size=min(self.F, n_samples), 
            replace=False  # Ensure no duplicates
        )
        
        self.X_data = X[selected_indices]

    def softmax(self, z):
        """Standard softmax with numerical stability."""
        z = z - torch.max(z, dim=-1, keepdim=True)[0]  # For numerical stability
        exp_z = torch.exp(z)
        return exp_z / torch.sum(exp_z, dim=-1, keepdim=True)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32)).to(self.device)
        elif isinstance(x, torch.Tensor) and x.device != self.device:
            x = x.to(self.device)
            
        # Ensure X_data is on correct device
        if isinstance(self.X_data, np.ndarray):
            self.X_data = torch.from_numpy(self.X_data.astype(np.float32)).to(self.device)
        elif isinstance(self.X_data, torch.Tensor) and self.X_data.device != self.device:
            self.X_data = self.X_data.to(self.device)
            
        # Compute transformations
        Q = self.W_q(x)
        K = self.W_k(self.X_data)
        V = self.W_v(self.X_data)
        
        # Compute attention scores with fixed scaling
        attention_scores = torch.matmul(Q, K.T) / torch.sqrt(torch.tensor(self.M).float())
        
        # Apply softmax
        attention_weights = self.softmax(attention_scores)
        
        # Update feature tracking during training
        if self.training:
            V_norms = torch.norm(V, p=2, dim=1)
            feature_acts = attention_weights * V_norms[None, :]
            dead_ratio, _ = self.feature_tracker.update(feature_acts)
        
        # Compute reconstruction
        x_hat = torch.matmul(attention_weights, V)
        
        return x, x_hat, attention_weights

    def compute_losses(self, x: torch.Tensor, x_hat: torch.Tensor, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute losses with improved regularization."""
        # L2 reconstruction loss with normalization
        L2_loss = torch.mean(torch.sum((x - x_hat)**2, dim=1)) / self.D
        
        # L1 regularization on attention weights with scaling
        L1_loss = self.lambda_l1 * torch.mean(torch.sum(f, dim=1)) / self.F
        
        total_loss = L2_loss + L1_loss
        return total_loss, L2_loss, L1_loss

    def initialize_weights(self):
        """Initialize weights with careful scaling."""
        with torch.no_grad():
            # Initialize Query transformation
            nn.init.normal_(self.W_q.weight, std=0.02)
            nn.init.zeros_(self.W_q.bias)
            
            # Initialize Key transformation
            nn.init.normal_(self.W_k.weight, std=0.02)
            nn.init.zeros_(self.W_k.bias)
            
            # Initialize Value transformation
            nn.init.normal_(self.W_v.weight, std=0.02)
            nn.init.zeros_(self.W_v.bias)
            
            # Initialize temperature
            self.temperature.data = torch.ones(1) * np.sqrt(self.M)

    def feature_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Get feature activations."""
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x.astype(np.float32)).to(self.device)
                
            Q = self.W_q(x)
            K = self.W_k(self.X_data)
            V = self.W_v(self.X_data)
            
            attention_scores = torch.matmul(Q, K.T) / self.temperature
            attention_weights = self.softmax(attention_scores)
            
            V_norms = torch.norm(V, p=2, dim=1)
            return attention_weights * V_norms[None, :]

    def train_and_validate(self, X_train, X_val, learning_rate=1e-3, batch_size=4096, target_steps=200000):
        """Train with improved scheduling."""
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=target_steps)
        
        # Preprocess data
        X_train = self.preprocess(X_train)
        X_val = self.preprocess(X_val)
        
        train_dataset = TensorDataset(X_train)
        val_dataset = TensorDataset(X_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        steps_per_epoch = len(train_loader)
        num_epochs = max(1, target_steps // steps_per_epoch)
        total_steps = num_epochs * steps_per_epoch
        
        warmup_steps = total_steps // 20
        
        print("\nTraining Configuration:")
        print(f"Total Steps: {total_steps}")
        print(f"Warmup Steps: {warmup_steps}")
        print(f"Batch Size: {batch_size}")
        print(f"Initial Learning Rate: {learning_rate}")
        
        best_val_loss = float('inf')
        step = 0
        l1 = self.lambda_l1
        for epoch in range(num_epochs):
            self.train()
            for batch in train_loader:
                batch = batch[0]
                
                # Warmup phase
                if step < warmup_steps:
                    # Linear warmup of learning rate
                    lr = learning_rate * (step / warmup_steps)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                    # Linear warmup of L1 regularization from 0 to final_lambda
                    self.lambda_l1  = l1 * (step / warmup_steps)  # warm up to lambda=5.0 as per paper
                else:
                    self.lambda_l1 = l1  # maintain final lambda value after warmup
                
                optimizer.zero_grad()
                x, x_hat, f = self.forward(batch)
                total_loss, L2_loss, L1_loss = self.compute_losses(x, x_hat, f)
                total_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                
                if step >= warmup_steps:
                    scheduler.step()
                
                # Logging and validation
                if step % (steps_per_epoch // 5) == 0:
                    self.eval()
                    val_loss = 0.0
                    val_batches = 0
                    
                    with torch.no_grad():
                        for val_batch in val_loader:
                            val_x = val_batch[0]
                            val_x, val_x_hat, val_f = self.forward(val_x)
                            val_total_loss, _, _ = self.compute_losses(val_x, val_x_hat, val_f)
                            val_loss += val_total_loss.item()
                            val_batches += 1
                    
                    avg_val_loss = val_loss / val_batches
                    # Only start counting dead features after we've seen enough samples
                    if self.feature_tracker.samples_seen >= self.feature_tracker.window_size:
                        dead_ratio = len(self.feature_tracker.get_dead_features()) / self.F
                    else:
                        dead_ratio = 0.0  # Not enough samples to consider features dead
                    sparsity = (f == 0).float().mean().item()
                    tracking_progress = min(self.feature_tracker.samples_seen / self.feature_tracker.window_size, 1.0)
                    
                    print(
                        f"Step: {step:6d} | "
                        f"Epoch: {epoch:3d} | "
                        f"Train Loss: {total_loss.item():8.4f} | "
                        f"Val Loss: {avg_val_loss:8.4f} | "
                        f"L1 λ: {self.lambda_l1:4.2f} | "
                        f"Dead Features: {dead_ratio:5.1%} | "
                        f"Sparsity: {sparsity:5.1%} | "
                        f"Window Fill: {tracking_progress:5.1%}"
                    )
                    
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        torch.save(self.state_dict(), self.st_model_path)
                    
                    self.train()
                
                step += 1
        
        print(f"\nTraining completed:")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Final λ: {self.lambda_l1:.2f}")

    def preprocess(self, X):
        """Preprocess input data using L2 normalization per the SAE paper."""
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X.astype(np.float32)).to(self.device)
        elif isinstance(X, torch.Tensor) and X.device != self.device:
            X = X.to(self.device)

        # Scale by mean L2 norm as in SAE paper
        C = torch.mean(torch.norm(X, p=2, dim=1)) / self.D
        X_normalized = X / C
        return X_normalized