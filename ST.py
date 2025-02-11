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
import torch.nn.functional as F

class SparseTransformer(nn.Module):
    def __init__(self, X, n: int, m: int, a: int, st_model_path: str, 
                 lambda_l1: float = 5.0, num_heads: int = 1, device: str = 'cuda',
                 window_size: int = 10_000_000, update_interval: int = 10_000,
                 activation_threshold: float = 1e-3):
        super().__init__()
        os.makedirs(os.path.dirname(st_model_path), exist_ok=True)
        
        # Core attributes
        self.X = X
        self.n, self.m = n, m
        self.st_model_path = st_model_path
        self.lambda_l1 = lambda_l1
        self.device = device
        self.activation_threshold = activation_threshold
        
        # Attention setup
        self.num_heads = num_heads
        self.a = ((a // num_heads) + 1) * num_heads if a % num_heads != 0 else a
        self.embed_dim = self.a * num_heads
        # Model components
        self.attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=num_heads, vdim=n, batch_first=True, dropout=0)
        self.W_q = nn.Linear(self.embed_dim, n)
        self.W_k = nn.Linear(self.embed_dim, n)
        self.W_v = nn.Linear(n, n)
        self.W_o = nn.Linear(self.embed_dim, n)
        # Feature tracking
        self.feature_tracker = DeadFeatureTracker(
            num_features=self.m,
            activation_threshold=activation_threshold,
            window_size=window_size,
            update_interval=update_interval
        )
        


        # Initialize indices for inference
        self.X_idx = np.random.choice(np.shape(self.X)[0], size=self.m, replace=False)
        self.to(device)

    def type_check(self, x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x.astype(np.float32)).to(self.device)
        return x.to(self.device) if x.device != self.device else x

    def forward(self, x):
        # Sample data
        if self.training:
            if not hasattr(self, "current_epoch"):
                self.current_epoch = 0
            
            # Get the current epoch from the training loop
            current_epoch = getattr(self, "current_epoch", 0)
            
            # Update indices if epoch changed
            if current_epoch != getattr(self, "last_epoch", -1):
                self.X_idx = np.random.choice(np.shape(self.X)[0], size=self.m, replace=False)
                self.last_epoch = current_epoch
            
            X_cross = self.X[self.X_idx]
        else:
            X_cross = self.X[self.X_idx]
        
        # Convert types
        X_cross = self.type_check(X_cross)
        x = self.type_check(x)
        
        # Compute attention components
        Q = torch.matmul(x, self.W_q.weight)
        K = torch.matmul(X_cross, self.W_k.weight)
        V = torch.matmul(X_cross, self.W_v.weight)
        
        # Normalize
        for tensor in (Q, K, V):
            tensor = F.layer_norm(tensor, (tensor.size(-1),))
        Q = Q / torch.sqrt(torch.tensor(self.a, dtype=torch.float32, device=self.device))
        

        attention_output, attention_weights = self.attention(Q, K, V, need_weights=True, average_attn_weights=True)
        # Compute outputs
        f = attention_weights
        x_hat = self.W_o(attention_output)
        
        if self.training:
            self.feature_tracker.update(f)
        
        return x, x_hat, f, V
    def compute_losses(self, x: torch.Tensor, x_hat: torch.Tensor, f: torch.Tensor, V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        L2_loss = F.mse_loss(x_hat, x)
        L1_loss = self.lambda_l1 * F.l1_loss(f, torch.zeros_like(f), reduction='sum')
        return L2_loss + L1_loss, L2_loss, L1_loss

    @torch.no_grad()
    def feature_activations(self, x: torch.Tensor) -> torch.Tensor:
        x = self.type_check(x)
        _, _, f, _ = self.forward(x)
        return f

    def preprocess(self, X, normalize: bool = True):
        if not isinstance(X, (np.ndarray, torch.Tensor)):
            raise ValueError("Input must be numpy array or torch tensor")
        
        X = self.type_check(X)
        if normalize:
            C = torch.mean(torch.norm(X, p=2, dim=1)) / np.sqrt(self.n)
            X = X / C
        return X

    def gini_coefficient(self, tensor: torch.Tensor) -> float:
        sorted_vals, _ = torch.sort(tensor.flatten().abs())
        n = sorted_vals.shape[0]
        idx = torch.arange(1, n+1, device=tensor.device)
        return (torch.sum(sorted_vals * idx) / (n * torch.sum(sorted_vals)) - (n + 1) / (2 * n)).item()

    def train_and_validate(self, X_train, X_val, learning_rate: float = 1e-3, 
                          batch_size: int = 4096, target_steps: int = 10_000, 
                          grad_clip: float = 1.0, log_freq: Optional[int] = 1):
        # Setup
        # Initialize GradScaler for mixed precision training
        scaler = torch.amp.GradScaler()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        steps_per_epoch = len(X_train) // batch_size
        num_epochs = max(1, target_steps // steps_per_epoch)
        total_steps = num_epochs * steps_per_epoch
        warmup_steps = total_steps // 20
        decay_start_step = int(total_steps * 0.8)
        
        # AMP setup
        amp_dtype = torch.float16 if torch.cuda.is_available() else torch.bfloat16
        amp_device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        autocast_context = torch.amp.autocast(device_type=amp_device_type, dtype=amp_dtype, cache_enabled=True)
        
        # Data setup - ensure data is on CPU before creating DataLoader
        X_train = self.preprocess(X_train, normalize=True)
        X_val = self.preprocess(X_val, normalize=True)
        
        # Move data to CPU for DataLoader
        if X_train.device != torch.device('cpu'):
            X_train = X_train.cpu()
        if X_val.device != torch.device('cpu'):
            X_val = X_val.cpu()
            
        train_loader = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(TensorDataset(X_val), batch_size=batch_size, pin_memory=True)
        
        log_freq = log_freq or steps_per_epoch // 5
        best_val_loss = float('inf')
        step = last_eval_step = 0
        l1 = self.lambda_l1
        times_per_step = []
        start_time = time.time()
        
        print(f"\nTraining: {total_steps} steps, {warmup_steps} warmup, batch {batch_size}, "
              f"lr {learning_rate}, heads {self.num_heads}, clip {grad_clip}")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch  # Add this line
            self.train()
            for batch in train_loader:
                step_start = time.time()
                optimizer.zero_grad(set_to_none=True)
                
                # Learning rate scheduling
                self.lambda_l1 = l1 * min(step / warmup_steps, 1.0) if step < warmup_steps else l1

                # Forward pass
                batch = batch[0].to(self.device)
                with autocast_context:
                    outputs = self.forward(batch)
                    total_loss, L2_loss, L1_loss = self.compute_losses(*outputs)
                
                # Backward pass
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                
                # Timing and cleanup
                times_per_step.append(time.time() - step_start)
                if len(times_per_step) > 3:
                    times_per_step.pop(0)
                
                if step % 1000 == 0:
                    torch.cuda.empty_cache()
                
                # Logging and validation
                if step % log_freq == 0:
                    self.eval()
                    with torch.no_grad():
                        val_loss = sum(self.compute_losses(*self.forward(val_batch[0].to(self.device)))[0] 
                                     for val_batch in val_loader)
                    val_loss /= len(val_loader)
                    
                    # Metrics
                    sparsity = (outputs[2] <= self.activation_threshold).float().mean().item()
                    gini = self.gini_coefficient(outputs[2])
                    remaining_time = str(timedelta(seconds=int(np.mean(times_per_step) * (total_steps - step))))
                    
                    print(f"Step: {step:6d}/{total_steps} ({step/total_steps:3.1%}) | "
                          f"Epoch: {epoch:3d} | LR: {optimizer.param_groups[0]['lr']:.2e} | "
                          f"Train: {total_loss.item():8.4f} | Val: {val_loss:8.4f} | "
                          f"L1_loss: {L1_loss:4.2f} | L2_loss: {L2_loss:4.2f} | "
                          f"L1 λ: {self.lambda_l1:4.2f} | Sparse: {sparsity:5.1%} | "
                          f"Gini: {gini:.3f} | ETA: {remaining_time}")
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(self.state_dict(), self.st_model_path)
                    
                    self.train()
                
                step += 1
        
        print(f"\nTraining completed in {str(timedelta(seconds=int(time.time() - start_time)))}")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Final learning rate: {optimizer.param_groups[0]['lr']:.2e}")