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

# Import SAE and ST modules
from SAE import SparseAutoencoder
from ST import SparseTransformer
from deadfeatures import DeadFeatureTracker
logger.info("Using enhanced SAE and ST implementations")
def _train_worker_process(task_dict, model_name, device, local_model_path, using_local_model):
    """
    Worker function to train a model in a separate process.
    Must be defined at module level for Windows compatibility.
    
    Args:
        task_dict: Dictionary with task parameters
        model_name: Name of the model
        device: Device to use
        local_model_path: Path to local model if available
        using_local_model: Whether using a local model
    
    Returns:
        Dictionary with result information
    """
    import torch
    import numpy as np
    import copy
    import logging
    
    # Configure logging for this worker
    logger = logging.getLogger(f"worker_{os.getpid()}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    try:
        # Create a new trainer instance for this process
        logger.info(f"Creating trainer instance with model {model_name} on {device}")
        from train_for_analyze_gptneo import ModelTrainer
        local_trainer = ModelTrainer(
            model_name=model_name,
            device=device,
            local_model_path=local_model_path if using_local_model else None
        )
        
        # Extract task info
        layer_idx = task_dict['layer_idx']
        model_type = task_dict['model_type']
        
        # Convert numpy arrays to torch tensors
        activations = task_dict['activations']
        if isinstance(activations, np.ndarray):
            logger.info(f"Converting numpy array to tensor for {model_type} model at layer {layer_idx}")
            activations = torch.from_numpy(activations).to(device)
        
        # Update the task with the tensor
        task_copy = copy.deepcopy(task_dict)
        task_copy['activations'] = activations
        
        # Train the model
        logger.info(f"Starting training for {model_type} model at layer {layer_idx}")
        model_path = local_trainer.train_model_for_layer(**task_copy)
        logger.info(f"Training completed successfully for {model_type} model at layer {layer_idx}")
        
        return {
            'layer_idx': layer_idx,
            'model_type': model_type,
            'model_path': model_path,
            'status': 'success'
        }
        
    except Exception as e:
        import traceback
        logger.error(f"Error in worker process: {e}")
        logger.error(traceback.format_exc())
        
        return {
            'layer_idx': task_dict.get('layer_idx', -1),
            'model_type': task_dict.get('model_type', 'unknown'),
            'status': 'failed',
            'error': str(e),
            'traceback': traceback.format_exc()
        }
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
    # Add this function to the ModelTrainer class

    def train_models_parallel(self, tasks, max_workers=None):
        """
        Train multiple models in parallel using ProcessPoolExecutor.
        Windows-compatible implementation using top-level worker function.
        
        Args:
            tasks: List of task dictionaries
            max_workers: Maximum number of parallel workers (default: CPU count)
            
        Returns:
            List of result dictionaries
        """
        import concurrent.futures
        import torch.multiprocessing as mp
        import copy
        
        # Determine number of workers
        if max_workers is None:
            max_workers = mp.cpu_count()
        else:
            max_workers = min(max_workers, mp.cpu_count())
        
        # Adjust for CUDA - only one process per GPU
        if self.device == 'cuda':
            num_gpus = torch.cuda.device_count()
            if num_gpus > 0:
                max_workers = min(max_workers, num_gpus)
            logger.info(f"CUDA enabled, limiting to {max_workers} workers for {num_gpus} GPUs")
        
        # Prepare tasks - convert torch tensors to numpy for pickling
        prepared_tasks = []
        for task in tasks:
            task_copy = copy.deepcopy(task)
            # Ensure activations are numpy arrays
            if isinstance(task_copy['activations'], torch.Tensor):
                task_copy['activations'] = task_copy['activations'].cpu().numpy()
            prepared_tasks.append(task_copy)
        
        # Use ProcessPoolExecutor for parallelism
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {}
            for task in prepared_tasks:
                future = executor.submit(
                    _train_worker_process,
                    task,
                    self.model_name,
                    self.device,
                    self.model_path,
                    self.using_local_model
                )
                future_to_task[future] = task
            
            # Process results as they complete
            for future in tqdm(concurrent.futures.as_completed(future_to_task), 
                            total=len(prepared_tasks), desc="Training models"):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result['status'] == 'success':
                        logger.info(f"Successfully trained {result['model_type']} model for layer {result['layer_idx']}")
                    else:
                        logger.error(f"Failed to train {result['model_type']} model for layer {result['layer_idx']}: {result.get('error', 'Unknown error')}")
                        if 'traceback' in result:
                            logger.debug(f"Traceback: {result['traceback']}")
                
                except Exception as e:
                    logger.error(f"Exception processing result: {e}")
                    results.append({
                        'layer_idx': task['layer_idx'],
                        'model_type': task['model_type'],
                        'status': 'failed',
                        'error': str(e)
                    })
        
        # Sort results by layer index and model type for consistency
        results.sort(key=lambda x: (x['layer_idx'], x['model_type']))
        
        return results

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
                            use_old_st: bool = True,
                            activation_threshold: float = 1e-3,
                            auto_steps: bool = False,
                            auto_steps_base: int = 200000,
                            auto_steps_min: int = 5000,
                            auto_steps_max: int = 1000000,
                            auto_attention_dim: bool = True,
                            # Additional SAE parameters
                            save_best: bool = False,
                            enable_checkpoints: bool = False,
                            checkpoint_freq: int = 50000,
                            early_stopping: bool = False,
                            early_stopping_patience: int = 5,
                            warmup_steps_pct: float = 0.05,
                            final_decay_pct: float = 0.2,
                            plot_weights_freq: int = 0,
                            scheduler_type: str = None,
                            window_size: int = 10_000_000,
                            update_interval: int = 10_000,
                            log_level: str = 'INFO'):
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
            use_mixed_precision: Enable mixed precision training
            grad_accum_steps: Number of gradient accumulation steps
            eval_freq: Evaluation frequency during training
            attention_fn: Function to use for attention score processing (ST only)
            use_memory_bank: Use memory bank approach instead of direct K-V matrices (ST only)
            use_old_st: Use the original ST implementation (ST only)
            activation_threshold: Threshold for feature activation
            auto_steps: Automatically calculate optimal training steps
            auto_steps_base: Base steps for auto calculation
            auto_steps_min: Minimum steps for auto calculation
            auto_steps_max: Maximum steps for auto calculation
            auto_attention_dim: Auto-calculate attention dimension (ST only)
            save_best: Whether to save the best model based on validation loss
            enable_checkpoints: Whether to save periodic checkpoints
            checkpoint_freq: How often to save checkpoints (steps)
            early_stopping: Whether to enable early stopping
            early_stopping_patience: Patience for early stopping
            warmup_steps_pct: Percentage of steps for lambda and LR warmup
            final_decay_pct: Percentage of steps for final decay phase
            plot_weights_freq: How often to plot decoder weights (0 to disable)
            scheduler_type: Type of learning rate scheduler ('cosine', 'linear', 'constant')
            window_size: Window size for feature tracking
            update_interval: Update interval for feature tracking
            log_level: Logging level
            
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
            if auto_attention_dim:
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
        
        # Use the entire dataset for both training and validation to maximize data usage
        train_tensor = activation_tensor
        val_tensor = activation_tensor  # Same as training set
        
        # Create and train the model
        if model_type == 'sae':
            logger.info(f"Creating SAE model with dims: {hidden_dim} -> {feature_dim}")
            # Use the enhanced SAE implementation
            model = SparseAutoencoder(
                n=hidden_dim,
                m=feature_dim,
                sae_model_path=model_path,  # Pass model path to constructor
                lambda_l1=lambda_l1,
                device=self.device,
                activation='relu',  # SAE typically uses ReLU activation
                window_size=window_size,
                update_interval=update_interval,
                activation_threshold=activation_threshold,
                use_mixed_precision=use_mixed_precision,
                log_level=log_level
            )
            
            # Train the model with enhanced parameters
            model.train_and_validate(
                train_tensor,
                val_tensor,
                learning_rate=learning_rate,
                batch_size=batch_size,
                target_steps=target_steps,
                checkpoint_freq=checkpoint_freq,
                save_best=save_best,
                enable_checkpoints=enable_checkpoints,
                grad_accum_steps=grad_accum_steps,
                eval_freq=eval_freq,
                scheduler_type=scheduler_type,
                early_stopping=early_stopping,
                early_stopping_patience=early_stopping_patience,
                warmup_steps_pct=warmup_steps_pct,
                final_decay_pct=final_decay_pct,
                plot_weights_freq=plot_weights_freq,
                plot_save_dir=os.path.join(output_dir, 'sae_plots') if plot_weights_freq > 0 else None
            )
            
            # The model should be automatically saved by train_and_validate
            logger.info(f"Enhanced SAE model for layer {layer_idx} saved to {model_path}")
                
        elif model_type == 'st':
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
                # Use the standard implementation
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
                # Standard ST has more parameters
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
    
    # ST-specific parameters
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
    
    # Auto-steps parameters
    auto_steps_group = parser.add_argument_group('Auto Steps Configuration')
    auto_steps_group.add_argument('--auto_steps', action='store_true',
                             help='Automatically determine optimal number of training steps based on feature dimension')
    auto_steps_group.add_argument('--auto_steps_base', type=int, default=200000,
                             help='Base number of steps for auto-steps calculation (default: 200000)')
    auto_steps_group.add_argument('--auto_steps_min', type=int, default=5000,
                             help='Minimum number of steps for auto-steps calculation (default: 5000)')
    auto_steps_group.add_argument('--auto_steps_max', type=int, default=1000000,
                             help='Maximum number of steps for auto-steps calculation (default: 1000000)')
    
    # NEW: Enhanced SAE parameters
    sae_group = parser.add_argument_group('Enhanced SAE Configuration')
    sae_group.add_argument('--checkpoint_freq', type=int, default=50000,
                      help='How often to save checkpoints (steps)')
    sae_group.add_argument('--save_best', action='store_true', default=False,
                      help='Save the best model based on validation loss')
    sae_group.add_argument('--enable_checkpoints', action='store_true', default=False,
                      help='Enable periodic checkpoints during training')
    sae_group.add_argument('--early_stopping', action='store_true',
                      help='Enable early stopping during training')
    sae_group.add_argument('--early_stopping_patience', type=int, default=5,
                      help='Number of evaluations without improvement before stopping')
    sae_group.add_argument('--warmup_steps_pct', type=float, default=0.05,
                      help='Percentage of steps for lambda warmup')
    sae_group.add_argument('--final_decay_pct', type=float, default=0.2,
                      help='Percentage of steps for final LR decay')
    sae_group.add_argument('--plot_weights_freq', type=int, default=0,
                      help='How often to plot decoder weights (0 to disable)')
    sae_group.add_argument('--scheduler_type', type=str, default=None,
                      choices=[None, 'cosine', 'linear', 'constant'],
                      help='Type of learning rate scheduler')
    sae_group.add_argument('--window_size', type=int, default=10_000_000,
                      help='Window size for feature tracking')
    sae_group.add_argument('--update_interval', type=int, default=10_000,
                      help='Update interval for feature tracking')
    sae_group.add_argument('--log_level', type=str, default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Logging level for SAE training')
    
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
                    # Mixed precision parameters
                    'use_mixed_precision': args.use_mixed_precision,
                    # Auto-steps parameters
                    'auto_steps': args.auto_steps,
                    'auto_steps_base': args.auto_steps_base,
                    'auto_steps_min': args.auto_steps_min,
                    'auto_steps_max': args.auto_steps_max,
                    # Enhanced SAE parameters
                    'grad_accum_steps': args.grad_accum_steps,
                    'eval_freq': args.eval_freq,
                    'save_best': args.save_best,
                    'enable_checkpoints': args.enable_checkpoints,
                    'checkpoint_freq': args.checkpoint_freq,
                    'early_stopping': args.early_stopping,
                    'early_stopping_patience': args.early_stopping_patience,
                    'warmup_steps_pct': args.warmup_steps_pct,
                    'final_decay_pct': args.final_decay_pct,
                    'plot_weights_freq': args.plot_weights_freq,
                    'scheduler_type': args.scheduler_type,
                    'window_size': args.window_size,
                    'update_interval': args.update_interval,
                    'activation_threshold': args.activation_threshold,
                    'log_level': args.log_level
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
                    # ST-specific parameters
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
    logger.info(f"Mixed precision: {args.use_mixed_precision}")
    logger.info(f"Gradient accumulation steps: {args.grad_accum_steps}")
    
    if args.decomposition in ['sae', 'both']:
        logger.info("\nSAE Configuration:")
        logger.info(f"  Lambda L1: {args.l1_lambda}")
        logger.info(f"  Window size: {args.window_size}")
        logger.info(f"  Update interval: {args.update_interval}")
        logger.info(f"  Activation threshold: {args.activation_threshold}")
        if args.auto_steps:
            logger.info(f"  Auto steps: Enabled (base: {args.auto_steps_base}, min: {args.auto_steps_min}, max: {args.auto_steps_max})")
        else:
            logger.info(f"  Target steps: {args.target_steps}")
        logger.info(f"  Learning rate: {args.learning_rate}")
        logger.info(f"  Batch size: {args.batch_size}")
        logger.info(f"  Warmup steps %: {args.warmup_steps_pct}")
        logger.info(f"  Final decay %: {args.final_decay_pct}")
        logger.info(f"  Early stopping: {args.early_stopping}")
        if args.early_stopping:
            logger.info(f"  Early stopping patience: {args.early_stopping_patience}")
        logger.info(f"  Plot weights frequency: {args.plot_weights_freq}")
        logger.info(f"  Scheduler type: {args.scheduler_type}")
        
    if args.decomposition in ['st', 'both']:
        logger.info("\nST Configuration:")
        if args.auto_attention_dim:
            logger.info(f"  Attention dimension: Auto-calculated to match SAE params")
        else:
            logger.info(f"  Attention dimension: {args.attention_dim}")
        logger.info(f"  Attention function: {args.attention_fn}")
        logger.info(f"  Use memory bank: {args.use_memory_bank}")
        logger.info(f"  Use old ST: {args.use_old_st}")
    
    logger.info("="*50 + "\n")
    
    # Train models
    results = []
    
    # Replace the training section in main() with this code
    if args.parallel:
        # Parallel training
        logger.info(f"Training models in parallel with max_workers={args.max_workers or 'CPU count'}")
        results = trainer.train_models_parallel(tasks, max_workers=args.max_workers)
    else:
        # Sequential training
        logger.info("Training models sequentially")
        results = []
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