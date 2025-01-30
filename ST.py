import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional
from deadfeatures import DeadFeatureTracker
import time
from datetime import timedelta
from torch.cuda.amp import GradScaler, autocast

class SparseTransformer(nn.Module):
    def __init__(self, X, n: int, m: int, a: int, st_model_path: str, 
                 lambda_l1: float = 5.0, num_heads: int = 1, device: str = 'cuda',
                 window_size: int = 10_000_000, update_interval: int = 10_000,
                 activation_threshold: float = 1e-3):
        super(SparseTransformer, self).__init__()
        
        # Ensure model directory exists
        os.makedirs(os.path.dirname(st_model_path), exist_ok=True)
        self.X = X
        self.n = n  # Input dimension
        self.m = m  # Feature dimension
        self.a = a  # Attention dimension
        self.st_model_path = st_model_path
        self.lambda_l1 = lambda_l1
        self.device = device
        self.num_heads = num_heads
        self.activation_threshold = activation_threshold
        self.idx = np.random.choice(np.shape(self.X)[0], size=self.m, replace=False)
        # Ensure a is divisible by num_heads
        if a % num_heads != 0:
            new_a = ((a // num_heads) + 1) * num_heads
            print(f"Warning: Adjusting attention dimension from {a} to {new_a} to be divisible by {num_heads} heads")
            self.a = new_a
        self.embed_dim = self.a*num_heads
        self.attention = nn.MultiheadAttention(self.embed_dim, num_heads, vdim=n, batch_first=True)
        
        self.W_q = nn.Linear(a, n)
        self.W_k = nn.Linear(a, n)
        self.W_v = nn.Linear(n, n)
        
        # Initialize feature tracking
        self.feature_tracker = DeadFeatureTracker(
            num_features=self.m,
            activation_threshold=self.activation_threshold,
            window_size=window_size,
            update_interval=update_interval
        )

        self.to(device)

    def type_check(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32)).to(self.device)
        elif isinstance(x, torch.Tensor) and x.device != self.device:
            x = x.to(self.device)
        return(x)
    def forward(self, x):
        if self.training:
            # Sample once per epoch (add epoch tracking to class)
            if not hasattr(self, "current_epoch"):
                self.current_epoch = 0
                self.X_idx = np.random.choice(np.shape(self.X)[0], size=self.m, replace=False)
            X_cross = self.X[self.X_idx]
        else:
            # Use fixed indices for inference
            X_cross = self.X[self.idx]
            X_cross = self.type_check(X_cross)
        X_cross = self.type_check(X_cross)
        x = self.type_check(x)
            
        # Project inputs to attention dimension
        Q = torch.matmul(x, self.W_q.weight)
        K = torch.matmul(X_cross, self.W_k.weight)
        V = torch.matmul(X_cross, self.W_v.weight)

        # Layer normalization for Q, K, V
        Q = nn.functional.layer_norm(Q, (Q.size(-1),))
        K = nn.functional.layer_norm(K, (K.size(-1),))
        V = nn.functional.layer_norm(V, (V.size(-1),))
        
        # Scale Q to prevent dot products from growing too large
        Q = Q / torch.sqrt(torch.tensor(self.a, dtype=torch.float32, device=self.device))
        # Apply attention and get weights
        attention_output, attention_weights = self.attention(
            query=Q,
            key=K,
            value=V,
            need_weights=True
        )
        
        f = attention_weights
        x_hat = torch.matmul(f, V)

        if self.training:
            dead_ratio, _ = self.feature_tracker.update(f)
        return x, x_hat, f, V

    def compute_losses(self, x: torch.Tensor, x_hat: torch.Tensor, f: torch.Tensor, V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute losses with improved regularization."""
        # L2 reconstruction loss with normalization
        L2_loss = torch.mean(torch.sum((x - x_hat)**2, dim=1))
        # L1 regularization on attention weights with scaling
        L1_loss = self.lambda_l1 * torch.sum(torch.norm(f, dim=0)*torch.norm(V, dim=1), dim=-1)

        total_loss = L2_loss + L1_loss
        return total_loss, L2_loss, L1_loss

    def feature_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Get feature activations using cached values."""
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x.astype(np.float32)).to(self.device)
            x, x_hat, f, V = self.forward(x)
            return f*torch.norm(V, dim=1)

    def preprocess(self, X, normalize: bool = True):
        """Preprocess input data using optional L2 normalization."""
        if not isinstance(X, (np.ndarray, torch.Tensor)):
            raise ValueError("Input data must be a numpy array or torch tensor.")
        
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X.astype(np.float32)).to(self.device)
        elif isinstance(X, torch.Tensor) and X.device != self.device:
            X = X.to(self.device)

        if normalize:
            C = torch.mean(torch.norm(X, p=2, dim=1)) / np.sqrt(self.n)
            X_normalized = X / C
            return X_normalized
    def gini_coefficient(self, tensor: torch.Tensor) -> float:
        sorted_vals, _ = torch.sort(tensor.flatten().abs())
        n = sorted_vals.shape[0]
        idx = torch.arange(1, n+1, device=tensor.device)
        return (torch.sum(sorted_vals * idx) / (n * torch.sum(sorted_vals)) - (n + 1) / (2 * n)).item()
    def train_and_validate(self, X_train, X_val, learning_rate: float = 1e-3, batch_size: int = 4096, 
                        target_steps: int = 200000, grad_clip: float = 1.0, log_freq: Optional[int] = None):
        """Train with improved scheduling, mixed precision, and time estimation."""
        scaler = GradScaler()  # For mixed precision training
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        
        # Preprocess data (ensure it's on the CPU if using pin_memory)
        X_train = self.preprocess(X_train, normalize=True).cpu()  # Move to CPU for pin_memory
        X_val = self.preprocess(X_val, normalize=True).cpu()      # Move to CPU for pin_memory
        
        train_dataset = TensorDataset(X_train)
        val_dataset = TensorDataset(X_val)
        
        # Use pin_memory=True only if data is on CPU
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)
        
        steps_per_epoch = len(train_loader)
        num_epochs = max(1, target_steps // steps_per_epoch)
        total_steps = num_epochs * steps_per_epoch
        warmup_steps = total_steps // 20
        
        if log_freq is None:
            log_freq = steps_per_epoch // 5
        
        print("\nTraining Configuration:")
        print(f"Total Steps: {total_steps}")
        print(f"Warmup Steps: {warmup_steps}")
        print(f"Batch Size: {batch_size}")
        print(f"Initial Learning Rate: {learning_rate}")
        print(f"Number of Attention Heads: {self.num_heads}")
        print(f"Gradient Clipping: {grad_clip}")
        
        best_val_loss = float('inf')
        step = 0
        l1 = self.lambda_l1
        
        start_time = time.time()
        times_per_step = []
        
        for epoch in range(num_epochs):
            self.train()
            for batch in train_loader:
                step_start_time = time.time()
                batch = batch[0].to(self.device)  # Move batch to GPU
                
                # Warmup phase
                if step < warmup_steps:
                    lr = learning_rate * (step / warmup_steps)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                    self.lambda_l1 = l1 * (step / warmup_steps)
                else:
                    self.lambda_l1 = l1
                
                optimizer.zero_grad()
                with autocast():  # Mixed precision
                    x, x_hat, f, V = self.forward(batch)
                    total_loss, L2_loss, L1_loss = self.compute_losses(x, x_hat, f, V)
                scaler.scale(total_loss).backward()  # Scale loss for mixed precision
                torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                
                # Update timing statistics
                step_time = time.time() - step_start_time
                times_per_step.append(step_time)
                if len(times_per_step) > 10000:  # Keep last 100 steps for moving average
                    times_per_step.pop(0)
                
                # Logging and validation
                if step % log_freq == 0:
                    self.eval()
                    val_loss = 0.0
                    val_batches = 0
                    
                    with torch.no_grad():
                        for val_batch in val_loader:
                            val_x = val_batch[0].to(self.device)  # Move validation batch to GPU
                            val_x, val_x_hat, val_f, val_V = self.forward(val_x)
                            val_total_loss, _, _ = self.compute_losses(val_x, val_x_hat, val_f, val_V)
                            val_loss += val_total_loss.item()
                            val_batches += 1
                    
                    avg_val_loss = val_loss / val_batches
                    dead_ratio = len(self.feature_tracker.get_dead_features()) / self.m if self.feature_tracker.samples_seen >= self.feature_tracker.window_size else 0.0
                    sparsity = (f <= self.activation_threshold).float().mean().item()
                    tracking_progress = min(self.feature_tracker.samples_seen / self.feature_tracker.window_size, 1.0)
                    
                    # Calculate time estimates
                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    avg_time_per_step = np.mean(times_per_step)
                    remaining_steps = total_steps - step
                    estimated_remaining_time = avg_time_per_step * remaining_steps
                    
                    # Format time strings
                    elapsed_str = str(timedelta(seconds=int(elapsed_time)))
                    remaining_str = str(timedelta(seconds=int(estimated_remaining_time)))
                    # During training logging:
                    gini = self.gini_coefficient(f)
                    print(
                        f"Step: {step:6d}/{total_steps} ({step/total_steps:3.1%}) | "
                        f"Epoch: {epoch:3d} | "
                        f"Train Loss: {total_loss.item():8.4f} | "
                        f"Val Loss: {avg_val_loss:8.4f} | "
                        f"L1 λ: {self.lambda_l1:4.2f} | "
                        f"Dead: {dead_ratio:5.1%} | "
                        f"Sparse: {sparsity:5.1%} | "
                        f"Sparsity Gini: {gini:.3f} | "
                        f"Fill: {tracking_progress:5.1%} | "
                        f"Time remaining: {remaining_str}"
                    )
                    
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        torch.save(self.state_dict(), self.st_model_path)
                    
                    self.train()
                
                step += 1
        
        total_time = time.time() - start_time
        total_time_str = str(timedelta(seconds=int(total_time)))
        
        print(f"\nTraining completed in {total_time_str}")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Final λ: {self.lambda_l1:.2f}")