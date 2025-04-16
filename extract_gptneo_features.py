#!/usr/bin/env python3
"""
Extract Features from GPT Neo for Model Training

This script extracts hidden layer activations from GPT Neo models, preprocesses them,
and saves them in a format suitable for training SAE and ST models. It now includes
text-based train/validation splitting for better model evaluation.

Example usage:
    # Extract features with default preprocessing and text-based split
    python extract_gptneo_features.py --texts "Neural networks are powerful"

    # Extract with specific preprocessing method and validation ratio
    python extract_gptneo_features.py --preprocess robust_norm --val_ratio 0.2 --layers 0 6 12 --text_file input.txt

    # Use a locally downloaded model with different preprocessing
    python extract_gptneo_features.py --local_model_path models/gpt-neo-125m --preprocess standardize
"""
import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from transformers import (
    GPTNeoForCausalLM, GPTNeoConfig, AutoTokenizer,
    GPT2LMHeadModel, GPT2Config, GPT2Tokenizer,
    AutoConfig, AutoModelForCausalLM
)

def parse_args():
    """Parse command-line arguments with added train/val split options"""
    parser = argparse.ArgumentParser(description='Extract features from GPT Neo with preprocessing and text-based train/val split')
    
    # Model options
    parser.add_argument('--model', type=str, default='EleutherAI/gpt-neo-125m',
                       help='GPT Neo model name or path')
    parser.add_argument('--local_model_path', type=str, default=None,
                       help='Path to locally downloaded model (prioritized over --model)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda or cpu)')
    
    # Layer options
    parser.add_argument('--layers', type=int, nargs='+', default=[6],
                       help='Layers to extract features from (0 for embedding layer)')
    
    # Input options
    parser.add_argument('--texts', type=str, nargs='+', default=None,
                       help='Texts to use for feature extraction')
    parser.add_argument('--text_file', type=str, default=None,
                       help='File containing texts to use, one per line')
    parser.add_argument('--generate_samples', action='store_true',
                       help='Generate diverse sample texts')
    parser.add_argument('--duplicate_text', action='store_true',
                       help='Duplicate the same text multiple times for testing')
    
    # Output options
    parser.add_argument('--output_dir', type=str, default='gptneo_features',
                       help='Directory to save extracted features')
    parser.add_argument('--save_format', type=str, default='npz', choices=['npz', 'pt'],
                       help='Format to save extracted features (npz or pt)')
    
    # Preprocessing options
    parser.add_argument('--preprocess', type=str, default='robust_norm',
                      choices=['none', 'robust_norm', 'standardize', 'minmax', 'tanh_norm'],
                      help='Preprocessing method to apply to features before saving')
    parser.add_argument('--visualize_preprocessing', action='store_true',
                      help='Create visualizations of features before and after preprocessing')
    parser.add_argument('--raw_suffix', type=str, default='_raw',
                      help='Suffix to add to raw (unprocessed) feature files')
    parser.add_argument('--skip_raw', action='store_true',
                      help='Skip saving raw (unprocessed) features')
    
    # Train/Val split options (NEW)
    parser.add_argument('--val_ratio', type=float, default=0.2,
                      help='Ratio of texts to use for validation (default: 0.2)')
    parser.add_argument('--random_seed', type=int, default=42,
                      help='Random seed for text-based train/val split')
    parser.add_argument('--save_combined', action='store_true',
                      help='Save combined features file for backward compatibility')
    
    # Other options
    parser.add_argument('--timeout', type=int, default=300,
                       help='Request timeout in seconds')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed information')
    
    return parser.parse_args()

def get_texts(args):
    """Get texts from command-line arguments or file"""
    texts = []
    
    if args.texts:
        texts = args.texts
        print(f"Using {len(texts)} texts from command line")
    elif args.text_file:
        try:
            with open(args.text_file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(texts)} texts from {args.text_file}")
        except Exception as e:
            print(f"Error loading text file: {e}")
            texts = ["The transformer architecture revolutionized natural language processing."]
    elif args.generate_samples:
        texts = [
            # Technical content
            "Neural networks process information through layers of interconnected nodes, each applying weights and activation functions to transform input data.",
            # Creative writing
            "The old oak tree stood sentinel at the edge of the forest, its gnarled branches reaching skyward like ancient fingers.",
            # News article
            "Scientists announced today the discovery of a new exoplanet that may contain liquid water, raising hopes for finding extraterrestrial life.",
            # Casual conversation
            "Hey, did you catch that new movie last weekend? I thought the special effects were amazing but the plot was predictable.",
            # Academic writing
            "The experiment yielded statistically significant results (p<0.01), suggesting a strong correlation between the variables under investigation.",
            # Historical text
            "In 1776, representatives from the thirteen colonies signed the Declaration of Independence, formally announcing their separation from Britain.",
            # Business writing
            "The quarterly financial report indicates a 12% increase in revenue, driven primarily by strong performance in emerging markets.",
            # Medical text
            "Patients presenting with these symptoms should be evaluated for possible autoimmune disorders, particularly those affecting the thyroid.",
            # Legal text
            "The plaintiff alleges that the defendant breached the terms of the contract by failing to deliver the specified goods within the timeframe.",
            # Technical documentation
            "To install the package, run 'pip install library-name' and import the required modules into your Python script."
        ]
        print(f"Generated {len(texts)} sample texts")
    else:
        # Default text
        texts = ["The transformer architecture revolutionized natural language processing by introducing self-attention mechanisms that capture long-range dependencies in text data."]
        print("Using default sample text")
    
    # Duplicate text if requested
    if args.duplicate_text and texts:
        first_text = texts[0]
        texts = [first_text] * 8  # Create 8 copies
        print(f"Duplicated text to create {len(texts)} copies")
    
    # Limit number of texts to avoid memory issues
    max_texts = 20
    if len(texts) > max_texts:
        print(f"Warning: Limiting to {max_texts} texts to avoid memory issues")
        texts = texts[:max_texts]
    
    return texts

def analyze_features(features, title="Feature Analysis", print_stats=True):
    """Analyze features and print statistics"""
    # Get basic statistics
    mean_val = np.mean(features)
    median_val = np.median(features)
    std_val = np.std(features)
    min_val = np.min(features)
    max_val = np.max(features)
    
    # Check for NaNs or Infs
    has_nan = np.isnan(features).any()
    has_inf = np.isinf(features).any()
    
    # Calculate norms
    sample_norms = np.linalg.norm(features, axis=1)
    mean_norm = np.mean(sample_norms)
    median_norm = np.median(sample_norms)
    min_norm = np.min(sample_norms)
    max_norm = np.max(sample_norms)
    
    # Target norm for preprocessing
    target_norm = np.sqrt(features.shape[1])
    
    if print_stats:
        print(f"\n===== {title} =====")
        print(f"Shape: {features.shape}")
        print(f"Values - Mean: {mean_val:.6f}, Median: {median_val:.6f}, Std: {std_val:.6f}")
        print(f"Range - Min: {min_val:.6f}, Max: {max_val:.6f}, Span: {max_val-min_val:.6f}")
        print(f"Sample Norms - Mean: {mean_norm:.6f}, Median: {median_norm:.6f}")
        print(f"Norm Range - Min: {min_norm:.6f}, Max: {max_norm:.6f}, Ratio: {max_norm/min_norm:.2f}")
        print(f"Target norm (√n): {target_norm:.6f}")
        print(f"Contains NaN: {has_nan}, Contains Inf: {has_inf}")
    
    return {
        'shape': features.shape,
        'mean': mean_val,
        'median': median_val,
        'std': std_val,
        'min': min_val,
        'max': max_val,
        'mean_norm': mean_norm,
        'median_norm': median_norm,
        'min_norm': min_norm,
        'max_norm': max_norm,
        'target_norm': target_norm,
        'has_nan': has_nan,
        'has_inf': has_inf
    }

def visualize_preprocessing(raw_features, preprocessed_features, layer_num, preprocess_method, output_dir):
    """Create visualizations comparing raw and preprocessed features"""
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Get statistics for both
    raw_stats = analyze_features(raw_features, f"Layer {layer_num} Raw", False)
    proc_stats = analyze_features(preprocessed_features, f"Layer {layer_num} Preprocessed", False)
    
    # Create figure with 3x2 subplots
    fig, axs = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle(f"GPT Neo Layer {layer_num} Features: {preprocess_method} Preprocessing", fontsize=16)
    
    # 1. Histograms of values
    axs[0, 0].hist(raw_features.flatten(), bins=100, alpha=0.7, density=True)
    axs[0, 0].set_title(f'Raw Feature Values Distribution\nMean={raw_stats["mean"]:.4f}, Std={raw_stats["std"]:.4f}')
    axs[0, 0].set_xlabel('Value')
    axs[0, 0].set_ylabel('Density')
    
    axs[0, 1].hist(preprocessed_features.flatten(), bins=100, alpha=0.7, density=True)
    axs[0, 1].set_title(f'Preprocessed Feature Values Distribution\nMean={proc_stats["mean"]:.4f}, Std={proc_stats["std"]:.4f}')
    axs[0, 1].set_xlabel('Value')
    axs[0, 1].set_ylabel('Density')
    
    # 2. Histograms of norms
    raw_norms = np.linalg.norm(raw_features, axis=1)
    proc_norms = np.linalg.norm(preprocessed_features, axis=1)
    
    axs[1, 0].hist(raw_norms, bins=50, alpha=0.7)
    axs[1, 0].axvline(raw_stats["mean_norm"], color='r', linestyle='--', 
                  label=f'Mean: {raw_stats["mean_norm"]:.2f}')
    axs[1, 0].axvline(raw_stats["target_norm"], color='g', linestyle='--', 
                  label=f'Target: {raw_stats["target_norm"]:.2f}')
    axs[1, 0].set_title('Raw Sample L2 Norms')
    axs[1, 0].set_xlabel('L2 Norm')
    axs[1, 0].set_ylabel('Count')
    axs[1, 0].legend()
    
    axs[1, 1].hist(proc_norms, bins=50, alpha=0.7)
    axs[1, 1].axvline(proc_stats["mean_norm"], color='r', linestyle='--', 
                  label=f'Mean: {proc_stats["mean_norm"]:.2f}')
    axs[1, 1].axvline(proc_stats["target_norm"], color='g', linestyle='--', 
                  label=f'Target: {proc_stats["target_norm"]:.2f}')
    axs[1, 1].set_title('Preprocessed Sample L2 Norms')
    axs[1, 1].set_xlabel('L2 Norm')
    axs[1, 1].set_ylabel('Count')
    axs[1, 1].legend()
    
    # 3. Feature heatmaps
    max_samples = min(100, raw_features.shape[0])
    max_features = min(100, raw_features.shape[1])
    
    im0 = axs[2, 0].imshow(raw_features[:max_samples, :max_features], aspect='auto', cmap='viridis')
    axs[2, 0].set_title(f'Raw Features Heatmap (First {max_samples}x{max_features})')
    axs[2, 0].set_xlabel('Feature Index')
    axs[2, 0].set_ylabel('Sample Index')
    plt.colorbar(im0, ax=axs[2, 0])
    
    im1 = axs[2, 1].imshow(preprocessed_features[:max_samples, :max_features], aspect='auto', cmap='viridis')
    axs[2, 1].set_title(f'Preprocessed Features Heatmap (First {max_samples}x{max_features})')
    axs[2, 1].set_xlabel('Feature Index')
    axs[2, 1].set_ylabel('Sample Index')
    plt.colorbar(im1, ax=axs[2, 1])
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
    plt.savefig(os.path.join(vis_dir, f"layer{layer_num}_{preprocess_method}_comparison.png"), dpi=150)
    plt.close()
    
    print(f"Saved visualization to {os.path.join(vis_dir, f'layer{layer_num}_{preprocess_method}_comparison.png')}")

def preprocess_features(features, method='robust_norm'):
    """Apply preprocessing to features"""
    if method == 'none':
        return features
    
    # Convert to tensor for easier operations
    tensor_input = isinstance(features, torch.Tensor)
    if tensor_input:
        features_tensor = features.float()
    else:
        features_tensor = torch.from_numpy(features).float()
    
    input_dim = features_tensor.shape[1]
    
    if method == 'standardize':
        # Zero mean, unit variance standardization
        mean = torch.mean(features_tensor, dim=0, keepdim=True)
        std = torch.std(features_tensor, dim=0, keepdim=True) + 1e-8
        preprocessed = (features_tensor - mean) / std
        
    elif method == 'minmax':
        # Min-max scaling to [0, 1] range
        min_val = torch.min(features_tensor, dim=0, keepdim=True)[0]
        max_val = torch.max(features_tensor, dim=0, keepdim=True)[0]
        range_val = max_val - min_val + 1e-8
        preprocessed = (features_tensor - min_val) / range_val
        
    elif method == 'tanh_norm':
        # Apply tanh to squash extreme values, then normalize
        # First standardize
        mean = torch.mean(features_tensor, dim=0, keepdim=True)
        std = torch.std(features_tensor, dim=0, keepdim=True) + 1e-8
        standardized = (features_tensor - mean) / std
        
        # Apply tanh to squash extreme values while preserving relative magnitudes
        squashed = torch.tanh(standardized)
        
        # Then normalize to expected norm
        norms = torch.norm(squashed, p=2, dim=1, keepdim=True)
        target_norm = np.sqrt(input_dim)
        scale_factor = target_norm / (norms + 1e-8)
        preprocessed = squashed * scale_factor
        
    elif method == 'robust_norm':
        # Robust L2 normalization with outlier handling (recommended for GPT Neo)
        # 1. Calculate percentiles (robust approach to avoid memory issues)
        sorted_vals, _ = torch.sort(features_tensor.flatten())
        n_elements = sorted_vals.shape[0]
        p01_idx = int(n_elements * 0.01)
        p99_idx = int(n_elements * 0.99)
        p01 = sorted_vals[p01_idx]
        p99 = sorted_vals[p99_idx]
        
        # 2. Clip outliers
        clipped = torch.clamp(features_tensor, min=p01, max=p99)
        
        # 3. Standardize
        mean = torch.mean(clipped, dim=0, keepdim=True)
        std = torch.std(clipped, dim=0, keepdim=True) + 1e-8
        standardized = (clipped - mean) / std
        
        # 4. Normalize to expected norm
        norms = torch.norm(standardized, p=2, dim=1, keepdim=True)
        target_norm = np.sqrt(input_dim)
        scale_factor = target_norm / (norms + 1e-8)
        preprocessed = standardized * scale_factor
    else:
        raise ValueError(f"Unknown preprocessing method: {method}")
    
    # Return same type as input
    if tensor_input:
        return preprocessed
    else:
        return preprocessed.numpy()

def create_text_based_split(token_to_text_map, val_ratio=0.2, random_seed=42):
    """
    Create train/validation splits based on texts rather than tokens.
    This ensures all tokens from the same text stay together in either
    the training or validation set.
    
    Args:
        token_to_text_map: Array mapping each token to its source text index
        val_ratio: Ratio of texts to use for validation (default: 0.2)
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_indices, val_indices) arrays
    """
    import numpy as np
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Get unique text indices
    unique_texts = np.unique(token_to_text_map)
    n_texts = len(unique_texts)
    
    print(f"Found {n_texts} unique text groups in token map")
    
    # Shuffle the text indices
    np.random.shuffle(unique_texts)
    
    # Determine split point
    val_size = max(1, int(n_texts * val_ratio))  # Ensure at least 1 text in validation
    train_size = n_texts - val_size
    
    # Split texts into train and validation sets
    train_text_indices = unique_texts[:train_size]
    val_text_indices = unique_texts[train_size:]
    
    print(f"Split texts into {train_size} for training and {val_size} for validation")
    
    # Create masks for tokens
    train_mask = np.isin(token_to_text_map, train_text_indices)
    val_mask = np.isin(token_to_text_map, val_text_indices)
    
    # Get token indices
    train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]
    
    # Print token counts
    print(f"Train set: {len(train_indices)} tokens from {train_size} texts")
    print(f"Validation set: {len(val_indices)} tokens from {val_size} texts")
    
    return train_indices, val_indices

def extract_features(args, texts):
    """Extract features from the GPT Neo model"""
    print("\nInitializing model analyzer...")
    
    # Set environment variables for timeouts
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(args.timeout)
    os.environ["TRANSFORMERS_REQUEST_TIMEOUT"] = str(args.timeout)
    
    # Determine device to use
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Set extended timeouts via environment variables
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # 5 minutes
    os.environ["TRANSFORMERS_REQUEST_TIMEOUT"] = "300"  # 5 minutes
    
    # Determine model path and type
    using_local_model = args.local_model_path is not None
    model_path = args.local_model_path if using_local_model else args.model
    
    # Detect model type from name
    if 'gpt-neo' in args.model.lower():
        model_type = 'gpt-neo'
    elif 'gpt2' in args.model.lower():
        model_type = 'gpt2'
    elif 'opt' in args.model.lower():
        model_type = 'opt'
    else:
        model_type = 'auto'  # Try to auto-detect
    
    print(f"Loading model '{args.model}' (type: {model_type}) on {device}...")
    
    try:
        # Load tokenizer
        if using_local_model:
            # Check for tokenizer in different possible locations
            if os.path.exists(os.path.join(args.local_model_path, "tokenizer")):
                tokenizer_path = os.path.join(args.local_model_path, "tokenizer")
            else:
                tokenizer_path = args.local_model_path
            
            print(f"Loading tokenizer from {tokenizer_path}")
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                local_files_only=True
            )
        else:
            # Select appropriate tokenizer based on model type
            if model_type == 'gpt-neo' or model_type == 'gpt2':
                tokenizer = GPT2Tokenizer.from_pretrained(args.model)
            else:
                tokenizer = AutoTokenizer.from_pretrained(args.model)
        
        # Ensure padding token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with proper configuration for getting all hidden states
        if using_local_model:
            # Check for model in different possible locations
            if os.path.exists(os.path.join(args.local_model_path, "model")):
                model_weights_path = os.path.join(args.local_model_path, "model")
            else:
                model_weights_path = args.local_model_path
            
            print(f"Loading model from {model_weights_path}")
            # Load configuration first to set output_hidden_states
            if model_type == 'gpt-neo':
                config = GPTNeoConfig.from_pretrained(
                    model_weights_path, 
                    output_hidden_states=True
                )
                model = GPTNeoForCausalLM.from_pretrained(
                    model_weights_path,
                    config=config,
                    local_files_only=True,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16 if device == 'cuda' else torch.float32
                )
            elif model_type == 'gpt2':
                config = GPT2Config.from_pretrained(
                    model_weights_path, 
                    output_hidden_states=True
                )
                model = GPT2LMHeadModel.from_pretrained(
                    model_weights_path,
                    config=config,
                    local_files_only=True,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16 if device == 'cuda' else torch.float32
                )
            else:
                config = AutoConfig.from_pretrained(
                    model_weights_path, 
                    output_hidden_states=True
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_weights_path,
                    config=config,
                    local_files_only=True,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16 if device == 'cuda' else torch.float32
                )
        else:
            # Load from HuggingFace with appropriate configuration
            if model_type == 'gpt-neo':
                config = GPTNeoConfig.from_pretrained(args.model, output_hidden_states=True)
                model = GPTNeoForCausalLM.from_pretrained(
                    args.model,
                    config=config,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16 if device == 'cuda' else torch.float32
                )
            elif model_type == 'gpt2':
                config = GPT2Config.from_pretrained(args.model, output_hidden_states=True)
                model = GPT2LMHeadModel.from_pretrained(
                    args.model,
                    config=config,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16 if device == 'cuda' else torch.float32
                )
            else:
                config = AutoConfig.from_pretrained(args.model, output_hidden_states=True)
                model = AutoModelForCausalLM.from_pretrained(
                    args.model,
                    config=config,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16 if device == 'cuda' else torch.float32
                )
        
        model = model.to(device)
        model.eval()  # Set to evaluation mode
        
        # Get number of layers
        try:
            if model_type == 'gpt-neo':
                num_layers = len(model.transformer.h)
            elif model_type == 'gpt2':
                num_layers = len(model.transformer.h)
            elif model_type == 'opt':
                num_layers = len(model.model.decoder.layers)
            else:
                # Try to identify model structure
                if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                    num_layers = len(model.transformer.h)
                elif hasattr(model, 'model') and hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers'):
                    num_layers = len(model.model.decoder.layers)
                else:
                    num_layers = 12  # Default fallback
            print(f"Model has {num_layers} layers")
        except AttributeError:
            # For different model architectures
            print("Couldn't determine number of layers. Using default range of 12 layers.")
            num_layers = 12
        
        print(f"\nExtracting features from {len(texts)} texts using {len(args.layers)} layers: {args.layers}")
        
        # Process texts using the same approach from analyze_gptneo.py
        # Concatenate all texts with a space, exactly like reason.ipynb
        input_text = " ".join(texts)
        
        # Tokenize and get token lengths for each paragraph
        input_texts_token_lengths = [
            len(tokenizer.encode(paragraph))
            for paragraph in texts
        ]
        print("Token lengths per paragraph:", input_texts_token_lengths)
        
        # Calculate cumulative sums for tracking token positions
        cumulative_lengths = np.cumsum(input_texts_token_lengths)
        print("Cumulative sums:", cumulative_lengths)
        
        # Encode the entire text at once
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        
        # Run the model with output_hidden_states=True to get all layer states
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Extract all hidden states
        all_hidden_states = outputs.hidden_states
        
        # Get total sequence length
        seq_len = all_hidden_states[0].size(1)
        token_indices = np.arange(seq_len)
        
        # Assign text group IDs to tokens using searchsorted (exact match to reason.ipynb)
        group_ids = np.searchsorted(cumulative_lengths, token_indices, side="right")
        
        # Filter hidden states to only include requested layers
        hidden_states = {
            f"layer_{i}": all_hidden_states[i].cpu().squeeze().numpy()
            for i in args.layers if i < len(all_hidden_states)
        }
        
        return hidden_states, group_ids, input_texts_token_lengths, input_text
        
    except Exception as e:
        print(f"\nError during initialization: {e}")
        print("\nTIP: If downloading fails, try downloading the model first:")
        print(f"  python download_model.py --model {args.model} --output models/gpt-neo")
        print("  Then use: --local_model_path models/gpt-neo")
        raise

def save_features(args, hidden_states, token_to_text_map, token_lengths, texts):
    """Save extracted features to disk with preprocessing and text-based train/val split"""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save raw and preprocessed features for each layer
    saved_paths = []
    
    print("\n" + "="*50)
    print(f"PREPROCESSING AND TRAIN/VAL SPLITTING: {args.preprocess}")
    print("="*50)
    
    for layer_name, activations in hidden_states.items():
        # Extract layer number from name (e.g., "layer_6" -> 6)
        layer_num = int(layer_name.split('_')[1])
        
        # Analyze raw features
        print(f"\nLayer {layer_num} features:")
        raw_stats = analyze_features(activations, f"Layer {layer_num} Raw Features")
        
        # Apply preprocessing
        if args.preprocess != 'none':
            print(f"\nApplying {args.preprocess} preprocessing...")
            preprocessed_features = preprocess_features(activations, args.preprocess)
            proc_stats = analyze_features(preprocessed_features, f"Layer {layer_num} Preprocessed Features")
            
            # Create visualization if requested
            if args.visualize_preprocessing:
                visualize_preprocessing(
                    activations, 
                    preprocessed_features, 
                    layer_num, 
                    args.preprocess, 
                    args.output_dir
                )
        else:
            preprocessed_features = activations
            print("No preprocessing applied.")
        
        # Create text-based train/val split
        print("\nCreating text-based train/validation split...")
        train_indices, val_indices = create_text_based_split(
            token_to_text_map,
            val_ratio=args.val_ratio,
            random_seed=args.random_seed
        )
        
        # Split features into train and validation sets
        train_features = preprocessed_features[train_indices]
        val_features = preprocessed_features[val_indices]
        
        # Split token maps
        train_token_map = token_to_text_map[train_indices]
        val_token_map = token_to_text_map[val_indices]
        
        # Save raw features (optional)
        if not args.skip_raw:
            if args.save_format == 'npz':
                raw_path = os.path.join(args.output_dir, f"layer{layer_num}_features{args.raw_suffix}.npz")
                np.savez_compressed(
                    raw_path,
                    features=activations,
                    token_to_text_map=token_to_text_map,
                    token_lengths=token_lengths,
                    texts=np.array(texts, dtype=object)
                )
                print(f"Saved raw features to {raw_path}")
            else:  # pytorch format
                raw_path = os.path.join(args.output_dir, f"layer{layer_num}_features{args.raw_suffix}.pt")
                torch.save({
                    'features': torch.from_numpy(activations) if isinstance(activations, np.ndarray) else activations,
                    'token_to_text_map': torch.from_numpy(token_to_text_map) if isinstance(token_to_text_map, np.ndarray) else token_to_text_map,
                    'token_lengths': token_lengths,
                    'texts': texts
                }, raw_path)
                print(f"Saved raw features to {raw_path}")
            
            saved_paths.append(raw_path)
        
        # Save train features
        if args.save_format == 'npz':
            train_path = os.path.join(args.output_dir, f"layer{layer_num}_train_features.npz")
            np.savez_compressed(
                train_path,
                features=train_features,
                token_to_text_map=train_token_map,
                token_lengths=token_lengths,
                texts=np.array(texts, dtype=object),
                preprocessing_method=args.preprocess,
                preprocessing_timestamp=datetime.now().isoformat(),
                split_method="text_based",
                train_indices=train_indices,
                val_ratio=args.val_ratio
            )
        else:  # pytorch format
            train_path = os.path.join(args.output_dir, f"layer{layer_num}_train_features.pt")
            # Convert to tensor if numpy
            if isinstance(train_features, np.ndarray):
                train_tensor = torch.from_numpy(train_features)
            else:
                train_tensor = train_features
                
            if isinstance(train_token_map, np.ndarray):
                train_token_tensor = torch.from_numpy(train_token_map)
            else:
                train_token_tensor = train_token_map
                
            torch.save({
                'features': train_tensor,
                'token_to_text_map': train_token_tensor,
                'token_lengths': token_lengths,
                'texts': texts,
                'preprocessing_method': args.preprocess,
                'preprocessing_timestamp': datetime.now().isoformat(),
                'split_method': "text_based",
                'train_indices': train_indices,
                'val_ratio': args.val_ratio
            }, train_path)
        
        saved_paths.append(train_path)
        print(f"Saved training features to {train_path}")
        
        # Save validation features
        if args.save_format == 'npz':
            val_path = os.path.join(args.output_dir, f"layer{layer_num}_val_features.npz")
            np.savez_compressed(
                val_path,
                features=val_features,
                token_to_text_map=val_token_map,
                token_lengths=token_lengths,
                texts=np.array(texts, dtype=object),
                preprocessing_method=args.preprocess,
                preprocessing_timestamp=datetime.now().isoformat(),
                split_method="text_based",
                val_indices=val_indices,
                val_ratio=args.val_ratio
            )
        else:  # pytorch format
            val_path = os.path.join(args.output_dir, f"layer{layer_num}_val_features.pt")
            # Convert to tensor if numpy
            if isinstance(val_features, np.ndarray):
                val_tensor = torch.from_numpy(val_features)
            else:
                val_tensor = val_features
                
            if isinstance(val_token_map, np.ndarray):
                val_token_tensor = torch.from_numpy(val_token_map)
            else:
                val_token_tensor = val_token_map
                
            torch.save({
                'features': val_tensor,
                'token_to_text_map': val_token_tensor,
                'token_lengths': token_lengths,
                'texts': texts,
                'preprocessing_method': args.preprocess,
                'preprocessing_timestamp': datetime.now().isoformat(),
                'split_method': "text_based",
                'val_indices': val_indices,
                'val_ratio': args.val_ratio
            }, val_path)
        
        saved_paths.append(val_path)
        print(f"Saved validation features to {val_path}")
        
        # Also save combined features for backward compatibility
        if args.save_combined:
            if args.save_format == 'npz':
                output_path = os.path.join(args.output_dir, f"layer{layer_num}_features.npz")
                np.savez_compressed(
                    output_path,
                    features=preprocessed_features,
                    token_to_text_map=token_to_text_map,
                    token_lengths=token_lengths,
                    texts=np.array(texts, dtype=object),
                    preprocessing_method=args.preprocess,
                    preprocessing_timestamp=datetime.now().isoformat(),
                    train_indices=train_indices,
                    val_indices=val_indices
                )
            else:  # pytorch format
                output_path = os.path.join(args.output_dir, f"layer{layer_num}_features.pt")
                # Convert to tensor if numpy
                if isinstance(preprocessed_features, np.ndarray):
                    preprocessed_tensor = torch.from_numpy(preprocessed_features)
                else:
                    preprocessed_tensor = preprocessed_features
                    
                if isinstance(token_to_text_map, np.ndarray):
                    token_map_tensor = torch.from_numpy(token_to_text_map)
                else:
                    token_map_tensor = token_to_text_map
                    
                torch.save({
                    'features': preprocessed_tensor,
                    'token_to_text_map': token_map_tensor,
                    'token_lengths': token_lengths,
                    'texts': texts,
                    'preprocessing_method': args.preprocess,
                    'preprocessing_timestamp': datetime.now().isoformat(),
                    'train_indices': train_indices,
                    'val_indices': val_indices
                }, output_path)
            
            saved_paths.append(output_path)
            print(f"Saved combined features to {output_path} (for backward compatibility)")
    
    # Create a metadata file with information about the extraction and preprocessing
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    
    import json
    metadata = {
        'model': args.model if not args.local_model_path else args.local_model_path,
        'layers': args.layers,
        'num_texts': len(texts),
        'total_tokens': sum(token_lengths),
        'feature_dimensions': {layer_name: activations.shape[1] for layer_name, activations in hidden_states.items()},
        'preprocessing_method': args.preprocess,
        'split_method': "text_based",
        'val_ratio': args.val_ratio,
        'random_seed': args.random_seed,
        'text_samples': texts[:3] + ["..."] if len(texts) > 3 else texts,
        'extraction_date': datetime.now().isoformat()
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSaved metadata to {metadata_path}")
    print(f"Total tokens: {sum(token_lengths)}")
    
    return saved_paths

def main():
    """Main function for feature extraction with text-based train/val split"""
    args = parse_args()
    
    print("\n" + "="*50)
    print("GPT NEO FEATURE EXTRACTOR WITH TEXT-BASED TRAIN/VAL SPLIT")
    print("="*50)
    
    # Get texts for feature extraction
    texts = get_texts(args)
    
    # Extract features
    hidden_states, token_to_text_map, token_lengths, full_text = extract_features(args, texts)
    
    # Preprocess, split, and save features
    saved_paths = save_features(args, hidden_states, token_to_text_map, token_lengths, texts)
    
    # Print instructions for training
    print("\n" + "="*50)
    print("NEXT STEPS FOR TRAINING")
    print("="*50)
    print("To train models using these preprocessed features with text-based splits:")
    print(f"\npython train_multiple.py --datasets gptneo --gptneo_features_dir {args.output_dir} \\")
    print(f"  --gptneo_layers {' '.join(map(str, args.layers))} --model_types sae st --feature_dimensions 200")
    
    # Explicitly note these have been preprocessed and split
    print(f"\nNOTE: Features have been preprocessed with '{args.preprocess}' method and split into")
    print(f"train/validation sets using a text-based approach with {args.val_ratio*100:.0f}% validation ratio.")
    print("The main.py script should be updated to load the separate train/val files.")

if __name__ == "__main__":
    main()