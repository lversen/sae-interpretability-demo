#!/usr/bin/env python3
"""
Model Trainer for analyze_gptneo.py

This script trains SAE and ST models in a format and directory structure
that is directly compatible with analyze_gptneo.py.

Example usage:
    # Train SAE models for GPT-Neo layers 0-5
    python train_for_analyze_gptneo_fixed.py --model EleutherAI/gpt-neo-125m --layers 0-5 --decomposition sae

    # Train both SAE and ST models with custom parameters
    python train_for_analyze_gptneo_fixed.py --model EleutherAI/gpt-neo-1.3B --layers 8-16 --decomposition st --text_file text.txt
"""

import os
import sys
import argparse
import torch
import numpy as np
import time
from tqdm import tqdm
from datetime import datetime, timedelta
from transformers import (
    GPTNeoForCausalLM, AutoTokenizer, GPT2LMHeadModel,
    AutoConfig, AutoModelForCausalLM
)
import matplotlib.pyplot as plt
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import concurrent.futures
import glob
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

# Try to import SAE and ST modules
try:
    # If available in your environment
    from SAE import SparseAutoencoder
    from ST import SparseTransformer
    DECOMP_AVAILABLE = True
    logger.info("Using original SAE and ST implementations")
except ImportError:
    logger.warning("Original SAE or ST modules not found. Using simplified implementations.")
    DECOMP_AVAILABLE = False

# Define simplified versions if originals are not available
class SimplifiedSAE:
    """Simplified SAE implementation when the original is not available"""
    
    def __init__(self, n, m, device='cuda', lambda_l1=1.0, sae_model_path=None):
        self.n = n  # Input dimension
        self.m = m  # Feature dimension
        self.device = device
        self.lambda_l1 = lambda_l1
        self.sae_model_path = sae_model_path
        
        # Initialize weights
        self.encoder = torch.nn.Linear(n, m, bias=True).to(device)
        self.decoder = torch.nn.Linear(m, n, bias=True).to(device)
        
        # Initialize with sensible values
        torch.nn.init.xavier_uniform_(self.encoder.weight)
        torch.nn.init.xavier_uniform_(self.decoder.weight)
        
    def forward(self, x):
        # Encoder - ReLU activation
        h = torch.relu(self.encoder(x))
        # Decoder
        x_hat = self.decoder(h)
        return x_hat, h
    
    def train_and_validate(self, train_tensor, val_tensor, learning_rate=1e-3, 
                          batch_size=64, target_steps=5000):
        """Simple training loop"""
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=learning_rate
        )
        
        # Create simple dataloader
        train_dataset = torch.utils.data.TensorDataset(train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        logger.info(f"Training simplified SAE for {target_steps} steps...")
        step = 0
        total_loss = 0
        
        # Progress bar for training
        progress_bar = tqdm(total=target_steps, desc="Training SAE")
        
        while step < target_steps:
            for batch in train_loader:
                x = batch[0]
                
                # Forward pass
                x_hat, h = self.forward(x)
                
                # Compute loss
                recon_loss = torch.mean((x_hat - x) ** 2)
                l1_loss = self.lambda_l1 * torch.mean(torch.abs(h))
                loss = recon_loss + l1_loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Log progress
                step += 1
                total_loss += loss.item()
                
                if step % 100 == 0:
                    avg_loss = total_loss / 100
                    progress_bar.set_postfix(loss=f"{avg_loss:.6f}")
                    total_loss = 0
                
                progress_bar.update(1)
                
                if step >= target_steps:
                    break
        
        progress_bar.close()
        logger.info("SAE training completed!")
        return self
    
    def feature_activations(self, x):
        """Get feature activations for input x"""
        with torch.no_grad():
            h = torch.relu(self.encoder(x))
        return h
        
    def save(self, model_path):
        """Save model to path"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Create state dict
        state_dict = {
            'model_state_dict': {
                'encoder.weight': self.encoder.weight,
                'encoder.bias': self.encoder.bias,
                'decoder.weight': self.decoder.weight,
                'decoder.bias': self.decoder.bias
            },
            'lambda_l1': self.lambda_l1
        }
        
        torch.save(state_dict, model_path)
        logger.info(f"Model saved to {model_path}")

class SimplifiedST:
    """Simplified ST implementation when the original is not available"""
    
    def __init__(self, X, n, m, a, device='cuda', lambda_l1=1.0, st_model_path=None):
        self.n = n  # Input dimension
        self.m = m  # Feature dimension
        self.a = a  # Attention dimension
        self.device = device
        self.lambda_l1 = lambda_l1
        self.st_model_path = st_model_path
        
        # Initialize weights
        self.W_q = torch.nn.Linear(n, a, bias=True).to(device)
        self.W_k = torch.nn.Linear(n, a, bias=True).to(device)
        self.W_v = torch.nn.Linear(n, n, bias=True).to(device)
        
        # Initialize memory
        self.memory_values = torch.nn.Parameter(torch.randn(m, n).to(device))
        self.memory_keys = torch.nn.Parameter(torch.randn(m, a).to(device))
        
        # Initialize with sensible values
        torch.nn.init.xavier_uniform_(self.W_q.weight)
        torch.nn.init.xavier_uniform_(self.W_k.weight)
        torch.nn.init.xavier_uniform_(self.W_v.weight)
        torch.nn.init.xavier_uniform_(self.memory_values)
        torch.nn.init.xavier_uniform_(self.memory_keys)
    
    def forward(self, x):
        # Calculate queries
        queries = self.W_q(x)
        
        # Calculate attention scores
        attention_scores = torch.matmul(queries, self.memory_keys.t())
        
        # Apply softmax
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # Get output
        output = torch.matmul(attention_weights, self.memory_values)
        return output, attention_weights
    
    def train_and_validate(self, train_tensor, val_tensor, learning_rate=1e-3, 
                          batch_size=64, target_steps=5000):
        """Simple training loop"""
        optimizer = torch.optim.Adam([
            {'params': self.W_q.parameters()},
            {'params': self.W_k.parameters()},
            {'params': self.W_v.parameters()},
            {'params': self.memory_values},
            {'params': self.memory_keys}
        ], lr=learning_rate)
        
        # Create simple dataloader
        train_dataset = torch.utils.data.TensorDataset(train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        logger.info(f"Training simplified ST for {target_steps} steps...")
        step = 0
        total_loss = 0
        
        # Progress bar for training
        progress_bar = tqdm(total=target_steps, desc="Training ST")
        
        while step < target_steps:
            for batch in train_loader:
                x = batch[0]
                
                # Forward pass
                output, attention_weights = self.forward(x)
                
                # Compute loss
                recon_loss = torch.mean((output - x) ** 2)
                l1_loss = self.lambda_l1 * torch.mean(torch.abs(attention_weights))
                loss = recon_loss + l1_loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Log progress
                step += 1
                total_loss += loss.item()
                
                if step % 100 == 0:
                    avg_loss = total_loss / 100
                    progress_bar.set_postfix(loss=f"{avg_loss:.6f}")
                    total_loss = 0
                
                progress_bar.update(1)
                
                if step >= target_steps:
                    break
        
        progress_bar.close()
        logger.info("ST training completed!")
        return self
    
    def feature_activations(self, x):
        """Get feature activations for input x"""
        with torch.no_grad():
            _, attention_weights = self.forward(x)
        return attention_weights
    
    def save(self, model_path):
        """Save model to path"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Create state dict
        state_dict = {
            'model_state_dict': {
                'W_q.weight': self.W_q.weight,
                'W_q.bias': self.W_q.bias,
                'W_k.weight': self.W_k.weight,
                'W_k.bias': self.W_k.bias,
                'W_v.weight': self.W_v.weight,
                'W_v.bias': self.W_v.bias,
                'memory_values': self.memory_values,
                'memory_keys': self.memory_keys
            },
            'lambda_l1': self.lambda_l1
        }
        
        torch.save(state_dict, model_path)
        logger.info(f"Model saved to {model_path}")

class ModelTrainer:
    """Class to handle loading a model and training SAE/ST models on its activations"""
    
    def __init__(self, model_name="EleutherAI/gpt-neo-125m", device=None, local_model_path=None):
        """
        Initialize the model trainer
        
        Args:
            model_name: HuggingFace model name or path
            device: Computing device ('cuda' or 'cpu')
            local_model_path: Path to a locally downloaded model
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        # Set extended timeouts via environment variables
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # 5 minutes
        os.environ["TRANSFORMERS_REQUEST_TIMEOUT"] = "300"  # 5 minutes
        
        # Determine model path and type
        self.using_local_model = local_model_path is not None
        model_path = local_model_path if self.using_local_model else model_name
        self.model_path = model_path
        self.model_name = model_name
        
        # Detect model type from name
        if 'gpt-neo' in model_name.lower():
            self.model_type = 'gpt-neo'
        elif 'gpt2' in model_name.lower():
            self.model_type = 'gpt2'
        elif 'opt' in model_name.lower():
            self.model_type = 'opt'
        else:
            self.model_type = 'auto'  # Try to auto-detect
        
        logger.info(f"Loading model '{model_name}' (type: {self.model_type}) on {device}...")
        
        try:
            # Load tokenizer
            if self.using_local_model:
                # Check for tokenizer in different possible locations
                if os.path.exists(os.path.join(local_model_path, "tokenizer")):
                    tokenizer_path = os.path.join(local_model_path, "tokenizer")
                else:
                    tokenizer_path = local_model_path
                
                logger.info(f"Loading tokenizer from {tokenizer_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path,
                    local_files_only=True
                )
            else:
                # Select appropriate tokenizer based on model type
                if self.model_type == 'gpt-neo' or self.model_type == 'gpt2':
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                else:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Ensure padding token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with proper configuration for getting all hidden states
            if self.using_local_model:
                # Check for model in different possible locations
                if os.path.exists(os.path.join(local_model_path, "model")):
                    model_weights_path = os.path.join(local_model_path, "model")
                else:
                    model_weights_path = local_model_path
                
                logger.info(f"Loading model from {model_weights_path}")
                # Load configuration first to set output_hidden_states
                config = AutoConfig.from_pretrained(
                    model_weights_path, 
                    output_hidden_states=True
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_weights_path,
                    config=config,
                    local_files_only=True,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16 if device == 'cuda' else torch.float32
                )
            else:
                # Load from HuggingFace with appropriate configuration
                config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    config=config,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16 if device == 'cuda' else torch.float32
                )
            
            self.model = self.model.to(device)
            self.model.eval()  # Set to evaluation mode
            
            # Get number of layers
            try:
                if self.model_type == 'gpt-neo':
                    self.num_layers = len(self.model.transformer.h)
                elif self.model_type == 'gpt2':
                    self.num_layers = len(self.model.transformer.h)
                elif self.model_type == 'opt':
                    self.num_layers = len(self.model.model.decoder.layers)
                else:
                    # Try to identify model structure
                    if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                        self.num_layers = len(self.model.transformer.h)
                    elif hasattr(self.model, 'model') and hasattr(self.model.model, 'decoder') and hasattr(self.model.model.decoder, 'layers'):
                        self.num_layers = len(self.model.model.decoder.layers)
                    else:
                        self.num_layers = 12  # Default fallback
                logger.info(f"Model has {self.num_layers} layers")
            except AttributeError:
                # For different model architectures
                logger.warning("Couldn't determine number of layers. Using default range of 12 layers.")
                self.num_layers = 12
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.error("\nTROUBLESHOOTING SUGGESTIONS:")
            logger.error("1. Download the model locally first using download_model.py:")
            logger.error("   python download_model.py --model EleutherAI/gpt-neo-125m --output models/gpt-neo-125m")
            logger.error("   Then use: --local_model_path models/gpt-neo-125m")
            logger.error("2. Try a smaller model like 'gpt2' or 'distilgpt2'")
            raise
    
    def process_texts(self, texts, layer_indices=None):
        """
        Process multiple texts and extract hidden states.
        Uses reason.ipynb approach: concatenate all texts with a space,
        and use cumulative sums to track token positions.
        
        Args:
            texts: List of texts to analyze
            layer_indices: List of layer indices to extract (defaults to all layers)
            
        Returns:
            Tuple of (hidden_states, token_to_text_map, input_texts_token_lengths, input_text)
        """
        # Concatenate all texts with a space, exactly like reason.ipynb
        input_text = " ".join(texts)
        
        # Tokenize and get token lengths for each paragraph
        input_texts_token_lengths = [
            len(self.tokenizer.encode(paragraph))
            for paragraph in texts
        ]
        logger.debug("Token lengths per paragraph:", input_texts_token_lengths)
        
        # Calculate cumulative sums for tracking token positions
        cumulative_lengths = np.cumsum(input_texts_token_lengths)
        logger.debug("Cumulative sums:", cumulative_lengths)
        
        # Encode the entire text at once
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        
        # Run the model with output_hidden_states=True to get all layer states
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Extract all hidden states
        all_hidden_states = outputs.hidden_states
        
        # Get total sequence length
        seq_len = all_hidden_states[0].size(1)
        token_indices = np.arange(seq_len)
        
        # Assign text group IDs to tokens using searchsorted (exact match to reason.ipynb)
        group_ids = np.searchsorted(cumulative_lengths, token_indices, side="right")
        
        # Filter hidden states if layer_indices is specified
        if layer_indices is not None:
            # Filter to only include requested layers
            hidden_states = {
                f"layer_{i}": all_hidden_states[i+1].cpu().squeeze().numpy()  # +1 to skip embeddings
                for i in layer_indices if i < len(all_hidden_states)-1  # -1 to account for embeddings
            }
        else:
            # Include all layers except embeddings
            hidden_states = {
                f"layer_{i}": all_hidden_states[i+1].cpu().squeeze().numpy()  # +1 to skip embeddings
                for i in range(len(all_hidden_states)-1)  # -1 to account for embeddings
            }
        
        return hidden_states, group_ids, input_texts_token_lengths, input_text
    #
    def train_model_for_layer(self, layer_idx: int, activations: np.ndarray, 
                            model_type: str = 'sae', output_dir: str = 'models', 
                            feature_dim: int = None, attention_dim: int = None,
                            lambda_l1: float = 1.0, batch_size: int = 256,
                            learning_rate: float = 1e-4, target_steps: int = 5000,
                            force_retrain: bool = False,
                            # New parameters
                            use_mixed_precision: bool = False,
                            grad_accum_steps: int = 1,
                            eval_freq: int = None,
                            attention_fn: str = 'softmax',
                            use_memory_bank: bool = False,
                            use_old_st: bool = False,
                            activation_threshold: float = 1e-3,
                            auto_steps: bool = False,
                            auto_steps_base: int = 200000,
                            auto_steps_min: int = 5000,
                            auto_steps_max: int = 1000000,
                            auto_attention_dim: bool = False):
        """
        Train a decomposition model (SAE or ST) for a specific layer's activations.
        
        Args:
            layer_idx: Index of the layer to train for
            activations: Activation matrix to train on
            model_type: Type of model to train ('sae' or 'st')
            output_dir: Base directory to save trained models
            feature_dim: Feature dimension, defaults to input_dim / 4 if None
            attention_dim: Attention dimension for ST models
            lambda_l1: L1 regularization strength
            batch_size: Training batch size
            learning_rate: Learning rate
            target_steps: Number of training steps
            force_retrain: Whether to retrain existing models
            use_mixed_precision: Enable mixed precision training for ST model
            grad_accum_steps: Number of gradient accumulation steps
            eval_freq: Evaluation frequency during training
            attention_fn: Function to use for attention score processing
            use_memory_bank: Use memory bank approach instead of direct K-V matrices
            use_old_st: Use the original ST implementation
            activation_threshold: Threshold for feature activation
            auto_steps: Automatically calculate optimal training steps
            auto_steps_base: Base steps for auto calculation
            auto_steps_min: Minimum steps for auto calculation
            auto_steps_max: Maximum steps for auto calculation
            
        Returns:
            Path to the saved model
        """
        # Get dimensions
        n_samples, hidden_dim = activations.shape
        
        # Default feature dimension to hidden_dim / 4 if not specified
        if feature_dim is None:
            feature_dim = max(100, hidden_dim // 4)
            logger.info(f"Using default feature dimension: {feature_dim}")
        
        # For ST models, calculate attention dimension if not specified
        if model_type == 'st' and attention_dim is None:
            # Check if we should auto-calculate the attention dimension
            if 'auto_attention_dim' in locals() and auto_attention_dim:
                # Calculate attention dimension to match SAE parameter count
                attention_dim = calculate_attention_dim_for_equal_params(
                    n=hidden_dim, 
                    m=feature_dim,
                    use_direct_kv=not use_memory_bank
                )
                logger.info(f"Auto-calculated attention dimension to match SAE params: {attention_dim}")
            else:
                # Use a simple heuristic
                attention_dim = max(20, hidden_dim // 8)
                logger.info(f"Using default attention dimension: {attention_dim}")
        
        # Create model directory structure compatible with analyze_gptneo.py
        # Format: {output_dir}/{decomp_type}/layer_{layer_idx}_{decomp_type}.pt
        model_subdir = 'sae' if model_type == 'sae' else 'st'
        os.makedirs(os.path.join(output_dir, model_subdir), exist_ok=True)
        
        # Create the exact filename pattern that analyze_gptneo.py expects
        model_filename = f"layer_{layer_idx}_{model_type}.pt"
        model_path = os.path.join(output_dir, model_subdir, model_filename)
        
        # Check if model already exists
        if os.path.exists(model_path) and not force_retrain:
            logger.info(f"Model already exists at {model_path}. Skipping training.")
            return model_path
        
        # Calculate optimal steps if auto_steps is enabled
        if auto_steps:
            original_target_steps = target_steps
            target_steps = calculate_optimal_training_steps(
                feature_dimension=feature_dim,
                input_dimension=hidden_dim,
                model_type=model_type,
                base_steps=auto_steps_base,
                min_steps=auto_steps_min,
                max_steps=auto_steps_max
            )
            logger.info(f"Auto-calculated optimal steps: {target_steps} (was: {original_target_steps})")
        
        # Convert to PyTorch tensor
        activation_tensor = torch.from_numpy(activations).float().to(self.device)
        
        # Split data for training and validation
        split_idx = int(n_samples * 0.8)
        train_tensor = activation_tensor[:split_idx]
        val_tensor = activation_tensor[split_idx:]
        
        # Create and train the model
        if model_type == 'sae':
            if DECOMP_AVAILABLE:
                logger.info(f"Creating SAE model with dims: {hidden_dim} -> {feature_dim}")
                # Use the original implementation if available
                model = SparseAutoencoder(
                    n=hidden_dim,
                    m=feature_dim,
                    lambda_l1=lambda_l1,
                    device=self.device,
                    sae_model_path=model_path,  # Pass model path to constructor
                    activation='relu'  # SAE typically uses ReLU activation
                )
                
                # Train the model
                model.train_and_validate(
                    train_tensor,
                    val_tensor,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    target_steps=target_steps
                )
                
                # The model should be automatically saved by train_and_validate
                logger.info(f"SAE model for layer {layer_idx} saved to {model_path}")
            else:
                # Use simplified implementation
                logger.info(f"Creating simplified SAE model with dims: {hidden_dim} -> {feature_dim}")
                model = SimplifiedSAE(
                    n=hidden_dim,
                    m=feature_dim,
                    device=self.device,
                    lambda_l1=lambda_l1,
                    sae_model_path=model_path  # Pass model path to constructor
                )
                
                # Train the model
                model.train_and_validate(
                    train_tensor,
                    val_tensor,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    target_steps=target_steps
                )
                
                # Save model
                model.save(model_path)
                logger.info(f"Simplified SAE model for layer {layer_idx} saved to {model_path}")
                
        elif model_type == 'st':
            if DECOMP_AVAILABLE:
                logger.info(f"Creating ST model with dims: {hidden_dim} -> {feature_dim}, attention dim: {attention_dim}")
                # Check if we need to decide between ST and ST_old
                if use_old_st:
                    try:
                        # Import ST_old only if needed
                        import ST_old
                        logger.info("Using original ST implementation (ST_old)")
                        
                        # Create ST_old model
                        model = ST_old.SparseTransformer(
                            X=activation_tensor,
                            n=hidden_dim,
                            m=feature_dim,
                            a=attention_dim,
                            lambda_l1=lambda_l1,
                            device=self.device,
                            st_model_path=model_path,
                            activation_threshold=activation_threshold,
                            use_direct_kv=not use_memory_bank,
                            attention_fn=attention_fn
                        )
                    except ImportError:
                        logger.warning("ST_old not found, falling back to default ST")
                        use_old_st = False
                
                # If not using old ST or import failed, use the regular ST
                if not use_old_st:
                    # Use the original implementation
                    logger.info("Using regular ST implementation")
                    model = SparseTransformer(
                        X=activation_tensor,
                        n=hidden_dim,
                        m=feature_dim,
                        a=attention_dim,
                        lambda_l1=lambda_l1,
                        device=self.device,
                        st_model_path=model_path,
                        use_mixed_precision=use_mixed_precision,
                        use_direct_kv=not use_memory_bank,
                        attention_fn=attention_fn,
                        activation_threshold=activation_threshold
                    )
                
                # Train the model with appropriate parameters
                if use_old_st:
                    # Old ST has simpler interface
                    model.train_and_validate(
                        train_tensor,
                        val_tensor,
                        learning_rate=learning_rate,
                        batch_size=batch_size,
                        target_steps=target_steps
                    )
                else:
                    # New ST has more parameters
                    model.train_and_validate(
                        train_tensor,
                        val_tensor,
                        learning_rate=learning_rate,
                        batch_size=batch_size,
                        target_steps=target_steps,
                        grad_accum_steps=grad_accum_steps,
                        eval_freq=eval_freq
                    )
                
                # The model should be automatically saved by train_and_validate
                logger.info(f"ST model for layer {layer_idx} saved to {model_path}")
            else:
                # Use simplified implementation
                logger.info(f"Creating simplified ST model with dims: {hidden_dim} -> {feature_dim}, attention dim: {attention_dim}")
                model = SimplifiedST(
                    X=activation_tensor,
                    n=hidden_dim,
                    m=feature_dim,
                    a=attention_dim,
                    device=self.device,
                    lambda_l1=lambda_l1,
                    st_model_path=model_path  # Pass model path to constructor 
                )
                
                # Train the model
                model.train_and_validate(
                    train_tensor,
                    val_tensor,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    target_steps=target_steps
                )
                
                # Save model
                model.save(model_path)
                logger.info(f"Simplified ST model for layer {layer_idx} saved to {model_path}")
        
        return model_path

def parse_layers(layers_str):
    """Parse layers string into a list of indices"""
    layers = []
    
    if not layers_str:
        return layers
    
    # Process individual components
    parts = layers_str.split(',')
    for part in parts:
        part = part.strip()
        
        # Handle ranges like "0-5"
        if '-' in part:
            start, end = map(int, part.split('-'))
            layers.extend(range(start, end + 1))
        # Handle single layers
        else:
            layers.append(int(part))
    
    # Remove duplicates and sort
    layers = sorted(list(set(layers)))
    return layers

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Train SAE and ST models for analyze_gptneo.py')
    
    # Model selection
    parser.add_argument('--model', type=str, default='EleutherAI/gpt-neo-125m',
                      help='Model to analyze (HuggingFace model name or path)')
    parser.add_argument('--local_model_path', type=str, default=None,
                      help='Path to a locally downloaded model')
    parser.add_argument('--device', type=str, default=None,
                      help='Device to use for computation (cuda or cpu)')
    
    # Layer selection
    parser.add_argument('--layers', type=str, default='0-3',
                      help='Layers to train models for (e.g., "0,1,2" or "0-5" or "0,3-6")')
    
    # Decomposition options
    parser.add_argument('--decomposition', type=str, default='both',
                      choices=['sae', 'st', 'both'],
                      help='Type of decomposition models to train')
    parser.add_argument('--feature_dim', type=int, default=None,
                      help='Feature dimension for decomposition (default: input_dim/4)')
    parser.add_argument('--attention_dim', type=int, default=None,
                      help='Attention dimension for ST models')
    parser.add_argument('--auto_attention_dim', action='store_true',
                      help='Automatically calculate attention dimension to match SAE parameter count')
    parser.add_argument('--l1_lambda', type=float, default=1.0,
                      help='L1 regularization strength')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=256,
                      help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='Learning rate')
    parser.add_argument('--target_steps', type=int, default=5000,
                      help='Target number of training steps')
    parser.add_argument('--force_retrain', action='store_true',
                      help='Force retraining even if models already exist')
    
    # NEW: ST-specific parameters
    st_group = parser.add_argument_group('ST Model Configuration')
    st_group.add_argument('--use_mixed_precision', action='store_true',
                      help='Enable mixed precision training for ST model')
    st_group.add_argument('--grad_accum_steps', type=int, default=1,
                      help='Number of gradient accumulation steps for ST model')
    st_group.add_argument('--eval_freq', type=int, default=None,
                      help='Evaluation frequency during training (steps)')
    st_group.add_argument('--attention_fn', type=str, default='softmax',
                      choices=['softmax', 'sparsemax', 'normalized_activation', 'direct_activation', 
                              'relu_softmax', 'softmax_hard', 'softmax_soft',
                              'length_scaled_softmax', 'softmax_with_bias', 'polynomial_attention', 
                              'adaptive_sparse', 'relu_attention', 'tanh_scale_shift'],
                      help='Function to use for processing attention scores (ST models only)')
    st_group.add_argument('--use_memory_bank', action='store_true',
                      help='Use memory bank approach instead of direct K-V matrices')
    st_group.add_argument('--use_old_st', action='store_true',
                      help='Use the original ST implementation (ST_old.py) if available')
    st_group.add_argument('--activation_threshold', type=float, default=1e-3,
                      help='Activation threshold for ST feature tracking')
    
    # NEW: Auto-steps parameters
    auto_steps_group = parser.add_argument_group('Auto Steps Configuration')
    auto_steps_group.add_argument('--auto_steps', action='store_true',
                             help='Automatically determine optimal number of training steps based on feature dimension')
    auto_steps_group.add_argument('--auto_steps_base', type=int, default=200000,
                             help='Base number of steps for auto-steps calculation (default: 200000)')
    auto_steps_group.add_argument('--auto_steps_min', type=int, default=5000,
                             help='Minimum number of steps for auto-steps calculation (default: 5000)')
    auto_steps_group.add_argument('--auto_steps_max', type=int, default=1000000,
                             help='Maximum number of steps for auto-steps calculation (default: 1000000)')
    
    # Input texts
    parser.add_argument('--text_file', type=str, default=None,
                      help='File containing texts to analyze (one per line)')
    parser.add_argument('--texts', type=str, nargs='+', default=None,
                      help='Direct text inputs to analyze')
    parser.add_argument('--n_texts', type=int, default=100,
                      help='Number of generated sample texts if none provided')
    
    # Output options
    parser.add_argument('--output_dir', type=str, default='models',
                      help='Base directory to save trained models')
    parser.add_argument('--parallel', action='store_true',
                      help='Train models in parallel using multiple processes')
    parser.add_argument('--max_workers', type=int, default=None,
                      help='Maximum number of parallel workers (default: CPU count)')
    
    args = parser.parse_args()
    
    return args
def generate_sample_texts(n=100):
    """Generate diverse sample texts for training"""
    # Different text types
    text_types = [
        # Technical content
        [
            "Neural networks process information through layers of interconnected nodes, each applying weights and activation functions to transform input data.",
            "The transformer architecture revolutionized natural language processing by introducing self-attention mechanisms that capture long-range dependencies in text."
        ],
        # Creative writing
        [
            "The old oak tree stood sentinel at the edge of the forest, its gnarled branches reaching skyward like ancient fingers.",
            "Moonlight spilled across the quiet lake, turning the rippling water into a canvas of liquid silver."
        ],
        # News article
        [
            "Scientists announced today the discovery of a new exoplanet that may contain liquid water, raising hopes for finding extraterrestrial life.",
            "Global markets rallied yesterday following the central bank's decision to lower interest rates in response to recent economic indicators."
        ],
        # Casual conversation
        [
            "Hey, did you catch that new movie last weekend? I thought the special effects were amazing but the plot was predictable.",
            "We should meet up for coffee sometime next week. I've been wanting to tell you about my recent trip to Japan."
        ],
        # Academic writing
        [
            "The experiment yielded statistically significant results (p<0.01), suggesting a strong correlation between the variables under investigation.",
            "This paper presents a comprehensive analysis of the socioeconomic factors contributing to urban development in post-industrial regions."
        ],
        # Historical text
        [
            "In 1776, representatives from the thirteen colonies signed the Declaration of Independence, formally announcing their separation from Britain.",
            "The Roman Empire reached its greatest territorial extent under the rule of Trajan, encompassing vast regions around the Mediterranean Sea."
        ],
        # Business writing
        [
            "The quarterly financial report indicates a 12% increase in revenue, driven primarily by strong performance in emerging markets.",
            "Our five-year strategic plan aims to diversify product offerings while maintaining our core commitment to sustainability and ethical sourcing."
        ],
        # Medical text
        [
            "Patients presenting with these symptoms should be evaluated for possible autoimmune disorders, particularly those affecting the thyroid.",
            "The study found that regular exercise significantly reduced the risk of cardiovascular disease, with benefits observed across all age groups."
        ],
        # Legal text
        [
            "The plaintiff alleges that the defendant breached the terms of the contract by failing to deliver the specified goods within the timeframe.",
            "The court ruled that the statute of limitations had expired, thereby dismissing the case without further consideration of its merits."
        ],
        # Technical documentation
        [
            "To install the package, run 'pip install library-name' and import the required modules into your Python script.",
            "The API provides several endpoints for data retrieval, each requiring authentication tokens that must be included in the request header."
        ],
        # Philosophical
        [
            "The question of whether consciousness is an emergent property or fundamental to reality remains one of philosophy's most enduring debates.",
            "Freedom can be understood as either the absence of external constraints or the presence of meaningful choices aligned with one's values."
        ],
        # Instructional
        [
            "When baking bread, ensure that the yeast is fully activated before incorporating it into the flour mixture to achieve proper rising.",
            "To solve this type of differential equation, first identify whether it's separable, then isolate the variables on opposite sides of the equal sign."
        ]
    ]
    
    # Generate samples by selecting from different text types
    samples = []
    while len(samples) < n:
        for text_type in text_types:
            for text in text_type:
                samples.append(text)
                if len(samples) >= n:
                    break
            if len(samples) >= n:
                break
    
    return samples[:n]  # Ensure we return exactly n samples
def calculate_optimal_training_steps(
    feature_dimension: int, 
    input_dimension: int, 
    model_type: str = 'sae',
    base_steps: int = 200_000, 
    min_steps: int = 5_000, 
    max_steps: int = 1_000_000
) -> int:
    """
    Calculate the optimal number of training steps based on feature dimension and input dimension,
    following scaling laws for dictionary learning.
    
    Args:
        feature_dimension: The feature dimension (m)
        input_dimension: The input dimension (n)
        model_type: Type of model ('sae' or 'st')
        base_steps: Base number of steps for reference configuration (8*n features)
        min_steps: Minimum number of steps to return
        max_steps: Maximum number of steps to return
        
    Returns:
        Recommended number of training steps
    """
    # Calculate the ratio of features to input dimension
    feature_ratio = feature_dimension / input_dimension
    
    # Reference configuration: 8*n features with 200,000 steps
    reference_ratio = 8.0
    
    # Apply scaling law with exponent between 0.5 and 1
    # Using 0.75 as a middle ground based on power law relationship
    scaling_exponent = 0.75
    
    # Calculate the scaling factor based on feature ratio
    if feature_ratio <= 0:
        scaling_factor = 1.0  # Protection against division by zero
    else:
        scaling_factor = (feature_ratio / reference_ratio) ** scaling_exponent
    
    # Apply scaling factor to the base steps
    optimal_steps = int(base_steps * scaling_factor)
    
    # Ensure we're within reasonable bounds
    optimal_steps = max(min_steps, min(optimal_steps, max_steps))
    
    # ST models might benefit from more steps due to more complex optimization
    if model_type.lower() == 'st':
        optimal_steps = int(optimal_steps * 1.2)  # 20% more steps for ST models
        optimal_steps = min(optimal_steps, max_steps)
    
    return optimal_steps
def calculate_attention_dim_for_equal_params(n, m, use_direct_kv=False):
    """
    Calculate attention dimension 'a' that would make ST and SAE have equal parameters,
    considering both memory bank and direct KV approaches.
    
    For equal parameters with direct KV approach:
    SAE: (2*n*m + m + n)
    ST direct KV: a*(n + 1 + m) + m*n
    
    For equal parameters with memory bank approach:
    SAE: (2*n*m + m + n)
    ST memory bank: a*(2*n + 2) + n*(n + 1)
    
    Args:
        n: Input dimension
        m: Feature dimension
        use_direct_kv: Whether using direct KV or memory bank approach
        
    Returns:
        a: Attention dimension that makes parameter counts approximately equal
    """
    # SAE parameter count: 2*m*n + m + n
    sae_params = 2*m*n + m + n
    
    if use_direct_kv:
        # Direct KV parameter count: a*(n + 1 + m) + m*n
        # Solve for a:
        # a*(n + 1 + m) = sae_params - m*n
        # a = (sae_params - m*n) / (n + 1 + m)
        a = (sae_params - m*n) / (n + 1 + m)
    else:
        # Memory bank parameter count: a*(2*n + 2) + n*(n + 1)
        # Solve for a:
        # a*(2*n + 2) = sae_params - n*(n + 1)
        # a = (sae_params - n*(n + 1)) / (2*n + 2)
        a = (sae_params - n*(n + 1)) / (2*n + 2)
    
    # Ensure a doesn't become too small or negative
    a = max(1, int(a))
    
    return a
def main():
    """Main function for model training"""
    args = parse_args()
    start_time = time.time()
    
    # Parse layer indices
    layer_indices = parse_layers(args.layers)
    logger.info(f"Training models for layers: {layer_indices}")
    
    # Get input texts
    texts = []
    if args.text_file:
        # Load texts from file
        try:
            logger.info(f"Loading texts from {args.text_file}")
            with open(args.text_file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(texts)} texts")
        except Exception as e:
            logger.error(f"Error loading text file: {e}")
            logger.info("Using generated sample texts instead")
            texts = generate_sample_texts(args.n_texts)
    elif args.texts:
        # Use provided texts
        texts = args.texts
        logger.info(f"Using {len(texts)} provided texts")
    else:
        # Generate diverse sample texts
        texts = generate_sample_texts(args.n_texts)
        logger.info(f"Generated {len(texts)} sample texts")
    
    # Create model trainer
    trainer = ModelTrainer(
        model_name=args.model,
        device=args.device,
        local_model_path=args.local_model_path
    )
    
    # Process texts and extract hidden states
    logger.info(f"Extracting hidden states for {len(texts)} texts...")
    hidden_states, token_map, token_lengths, full_text = trainer.process_texts(texts, layer_indices)
    
    logger.info(f"Extracted hidden states from {len(hidden_states)} layers")
    for layer_name, activations in hidden_states.items():
        logger.info(f"  {layer_name}: shape {activations.shape}")
    
    # Create output directories
    for model_type in ['sae', 'st']:
        os.makedirs(os.path.join(args.output_dir, model_type), exist_ok=True)
    
    # Define tasks for training
    tasks = []
    for layer_idx in layer_indices:
        layer_name = f"layer_{layer_idx}"
        if layer_name in hidden_states:
            activations = hidden_states[layer_name]
            
            # Add SAE task
            if args.decomposition in ['sae', 'both']:
                tasks.append({
                    'layer_idx': layer_idx,
                    'activations': activations,
                    'model_type': 'sae',
                    'output_dir': args.output_dir,
                    'feature_dim': args.feature_dim,
                    'attention_dim': None,  # Not used for SAE
                    'lambda_l1': args.l1_lambda,
                    'batch_size': args.batch_size,
                    'learning_rate': args.learning_rate,
                    'target_steps': args.target_steps,
                    'force_retrain': args.force_retrain,
                    # Including new parameters for completeness
                    'auto_steps': args.auto_steps,
                    'auto_steps_base': args.auto_steps_base,
                    'auto_steps_min': args.auto_steps_min,
                    'auto_steps_max': args.auto_steps_max
                })
            
            # Add ST task
            if args.decomposition in ['st', 'both']:
                tasks.append({
                    'layer_idx': layer_idx,
                    'activations': activations,
                    'model_type': 'st',
                    'output_dir': args.output_dir,
                    'feature_dim': args.feature_dim,
                    'attention_dim': args.attention_dim,
                    'lambda_l1': args.l1_lambda,
                    'batch_size': args.batch_size,
                    'learning_rate': args.learning_rate,
                    'target_steps': args.target_steps,
                    'force_retrain': args.force_retrain,
                    # New ST-specific parameters
                    'use_mixed_precision': args.use_mixed_precision,
                    'grad_accum_steps': args.grad_accum_steps,
                    'eval_freq': args.eval_freq,
                    'attention_fn': args.attention_fn,
                    'use_memory_bank': args.use_memory_bank,
                    'use_old_st': args.use_old_st,
                    'activation_threshold': args.activation_threshold,
                    'auto_steps': args.auto_steps,
                    'auto_steps_base': args.auto_steps_base,
                    'auto_steps_min': args.auto_steps_min,
                    'auto_steps_max': args.auto_steps_max,
                    'auto_attention_dim': args.auto_attention_dim
                })
    
    logger.info(f"Prepared {len(tasks)} training tasks")
    
    # Display training configuration summary
    logger.info("\n" + "="*50)
    logger.info("TRAINING CONFIGURATION")
    logger.info("="*50)
    logger.info(f"Model: {args.model}")
    logger.info(f"Layers: {layer_indices}")
    logger.info(f"Decomposition: {args.decomposition}")
    logger.info(f"Feature dimension: {args.feature_dim}")
    if args.decomposition in ['st', 'both']:
        if args.auto_attention_dim:
            logger.info(f"Attention dimension: Auto-calculated to match SAE params")
        else:
            logger.info(f"Attention dimension: {args.attention_dim}")
        logger.info(f"Attention function: {args.attention_fn}")
        logger.info(f"Use memory bank: {args.use_memory_bank}")
        logger.info(f"Use old ST: {args.use_old_st}")
        logger.info(f"Mixed precision: {args.use_mixed_precision}")
        logger.info(f"Gradient accumulation steps: {args.grad_accum_steps}")
    logger.info(f"L1 lambda: {args.l1_lambda}")
    if args.auto_steps:
        logger.info(f"Auto steps: Enabled (base: {args.auto_steps_base}, min: {args.auto_steps_min}, max: {args.auto_steps_max})")
    else:
        logger.info(f"Target steps: {args.target_steps}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info("="*50 + "\n")
    
    # Train models
    results = []
    
    # Sequential training (parallel is disabled for this version to simplify debugging)
    logger.info("Training models sequentially")
    for task in tqdm(tasks, desc="Training models"):
        try:
            model_path = trainer.train_model_for_layer(**task)
            results.append({
                'layer_idx': task['layer_idx'],
                'model_type': task['model_type'],
                'model_path': model_path,
                'status': 'success'
            })
        except Exception as e:
            results.append({
                'layer_idx': task['layer_idx'],
                'model_type': task['model_type'],
                'status': 'failed',
                'error': str(e)
            })
            logger.error(f"Error training {task['model_type']} model for layer {task['layer_idx']}: {e}")
    
    # Calculate summary stats
    total_time = time.time() - start_time
    success_count = sum(1 for r in results if r['status'] == 'success')
    failed_count = sum(1 for r in results if r['status'] == 'failed')
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("TRAINING COMPLETE")
    logger.info("="*50)
    logger.info(f"Total time: {timedelta(seconds=int(total_time))}")
    logger.info(f"Successful models: {success_count}/{len(tasks)}")
    logger.info(f"Failed models: {failed_count}/{len(tasks)}")
    
    # Print model paths
    logger.info("\nTrained models:")
    for model_type in ['sae', 'st']:
        if args.decomposition in [model_type, 'both']:
            model_paths = [r['model_path'] for r in results 
                         if r['status'] == 'success' and r['model_type'] == model_type]
            if model_paths:
                logger.info(f"\n{model_type.upper()} models:")
                for path in model_paths:
                    logger.info(f"  {path}")
    
    # Print usage instructions for analyze_gptneo.py
    logger.info("\nTo use these models with analyze_gptneo.py, run:")
    
    # Build command with appropriate parameters
    cmd = f"python analyze_gptneo.py --model {args.model}"
    
    # Add decomposition type
    if args.decomposition == 'sae':
        cmd += f" --decomposition sae --sae_model_path {args.output_dir}/sae"
    elif args.decomposition == 'st':
        cmd += f" --decomposition st --st_model_path {args.output_dir}/st"
    else:
        cmd += f" --decomposition both --sae_model_path {args.output_dir}/sae --st_model_path {args.output_dir}/st"
    
    # Add layers if specified
    if layer_indices:
        cmd += f" --layers {','.join(map(str, layer_indices))}"
    
    # Add visualization flag
    cmd += " --visualize"
    
    logger.info(cmd)
    logger.info("="*50)

if __name__ == "__main__":
    main()