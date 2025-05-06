import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def find_model_paths(base_dir="models", model_pattern="*model*.pth"):
    """
    Automatically find SAE and ST model paths in the given directory
    
    Args:
        base_dir: Base directory to search in
        model_pattern: Glob pattern to match model files
        
    Returns:
        Dictionary mapping model types to paths
    """
    # Make sure base directory exists
    if not os.path.exists(base_dir):
        print(f"Warning: Directory {base_dir} does not exist")
        return {}
    
    # Find all model files matching pattern
    model_files = glob.glob(os.path.join(base_dir, model_pattern))
    
    # Categorize by model type
    model_paths = {
        'sae': [],
        'st': []
    }
    
    for file_path in model_files:
        file_name = os.path.basename(file_path).lower()
        if 'sae' in file_name:
            model_paths['sae'].append(file_path)
        elif 'st' in file_name:
            model_paths['st'].append(file_path)
    
    # Sort each list to get most recent files first (assuming timestamp in filename)
    for model_type in model_paths:
        model_paths[model_type].sort(reverse=True)
    
    # Print found models
    for model_type, paths in model_paths.items():
        if paths:
            print(f"Found {len(paths)} {model_type.upper()} model(s):")
            for i, path in enumerate(paths):
                print(f"  {i+1}. {path}")
    
    # Create dictionary with best path for each model type
    best_paths = {}
    for model_type, paths in model_paths.items():
        if paths:
            best_paths[model_type] = paths[0]
    
    return best_paths

def load_sae_decoder_weights(model_path, device='cpu'):
    """
    Load decoder weights from an SAE model
    
    Args:
        model_path: Path to the SAE model file
        device: Device to use for computation
        
    Returns:
        Tuple with decoder weights, corresponding norms, and model info
    """
    try:
        print(f"Loading SAE model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            model_info = {
                'step': checkpoint.get('step', 0),
                'dead_ratio': checkpoint.get('dead_ratio', 0),
                'lambda_l1': checkpoint.get('lambda_l1', 0),
                'val_loss': checkpoint.get('val_loss', None)
            }
        else:
            state_dict = checkpoint
            model_info = {'step': 0, 'dead_ratio': 0, 'lambda_l1': 0, 'val_loss': None}
        
        # Extract decoder weights
        if 'W_d.weight' in state_dict:
            # Get decoder weight matrix - shape (n, m) for SAE
            decoder_weights = state_dict['W_d.weight'].cpu().numpy()
            
            # Calculate L2 norms of each column (feature)
            weight_norms = np.linalg.norm(decoder_weights, axis=0)
            
            print(f"Loaded decoder weights with shape {decoder_weights.shape}")
            return decoder_weights, weight_norms, model_info
        else:
            raise ValueError("Could not find decoder weights (W_d.weight) in the model")
            
    except Exception as e:
        print(f"Error loading SAE model: {e}")
        return None, None, {}

def find_dataset_paths(base_dir="data", dataset_pattern="*.csv"):
    """
    Automatically find dataset files in the given directory
    
    Args:
        base_dir: Base directory to search in
        dataset_pattern: Glob pattern to match dataset files
        
    Returns:
        Path to best dataset file
    """
    # Make sure base directory exists
    if not os.path.exists(base_dir):
        print(f"Warning: Directory {base_dir} does not exist")
        return None
    
    # Find all dataset files matching pattern
    dataset_files = glob.glob(os.path.join(base_dir, dataset_pattern))
    
    # Filter for likely MNIST datasets
    mnist_files = [f for f in dataset_files if 'mnist' in os.path.basename(f).lower()]
    
    # Prioritize training data
    training_files = [f for f in mnist_files if 'train' in os.path.basename(f).lower()]
    
    if training_files:
        print(f"Found {len(training_files)} MNIST training datasets:")
        for i, path in enumerate(training_files):
            print(f"  {i+1}. {path}")
        return training_files[0]  # Return the first training file
    elif mnist_files:
        print(f"Found {len(mnist_files)} MNIST datasets (no training specific):")
        for i, path in enumerate(mnist_files):
            print(f"  {i+1}. {path}")
        return mnist_files[0]  # Return the first MNIST file
    elif dataset_files:
        print(f"Found {len(dataset_files)} dataset files (non-MNIST):")
        for i, path in enumerate(dataset_files[:5]):  # Show only top 5 if many
            print(f"  {i+1}. {path}")
        if len(dataset_files) > 5:
            print(f"  ... and {len(dataset_files) - 5} more")
        return dataset_files[0]  # Return the first dataset file
    else:
        print("No dataset files found.")
        return None

def load_dataset(dataset_path, feature_columns=None, n_samples=1000, min_samples=None):
    """
    Load dataset for computing value vectors
    
    Args:
        dataset_path: Path to the dataset CSV file
        feature_columns: Column names for features (if None, use all numeric columns)
        n_samples: Number of samples to load
        min_samples: Minimum number of samples required (overrides n_samples if larger)
        
    Returns:
        Tensor with dataset features
    """
    try:
        import pandas as pd
        
        print(f"Loading dataset from {dataset_path}")
        df = pd.read_csv(dataset_path)
        
        # Determine feature columns
        if feature_columns is None:
            # Use all numeric columns
            feature_columns = df.select_dtypes(include=['number']).columns.tolist()
            # If first column is likely a label, exclude it
            if len(feature_columns) > 1 and ('label' in feature_columns[0].lower() or df[feature_columns[0]].nunique() < 20):
                feature_columns = feature_columns[1:]
                
        # Extract features
        features = df[feature_columns].values.astype(np.float32)
        
        # Determine how many samples to load
        samples_to_load = max(n_samples, min_samples or 0)
        
        # Make sure we don't try to load more than exists in the dataset
        samples_to_load = min(samples_to_load, len(features))
        
        print(f"Loading {samples_to_load} samples from dataset with {len(features)} total samples")
        
        # Select subset of samples (use first N samples to preserve indices)
        if samples_to_load < len(features):
            features = features[:samples_to_load]
            
        # Convert to tensor
        features_tensor = torch.from_numpy(features)
        
        print(f"Loaded {len(features)} samples with {len(feature_columns)} features")
        return features_tensor
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Falling back to random data")
        # Fallback to random data with sufficient size
        size_to_generate = max(n_samples, min_samples or 0)
        return torch.randn(size_to_generate, 784)  # Default to MNIST-like dimensions

def load_st_model_and_compute_value_vectors(model_path, data_tensor, device='cpu'):
    """
    Load an ST model and compute its value vectors
    
    Args:
        model_path: Path to the ST model file
        data_tensor: Input data tensor for computing value vectors
        device: Device to use for computation
        
    Returns:
        Tuple with value vectors, corresponding norms, and model info
    """
    try:
        print(f"Loading ST model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract relevant model parameters from checkpoint
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            model_info = {
                'step': checkpoint.get('step', 0),
                'dead_ratio': checkpoint.get('dead_ratio', 0),
                'lambda_l1': checkpoint.get('lambda_l1', 0),
                'val_loss': checkpoint.get('val_loss', None)
            }
        else:
            state_dict = checkpoint
            model_info = {'step': 0, 'dead_ratio': 0, 'lambda_l1': 0, 'val_loss': None}
        
        # Check if the model uses direct K-V approach
        use_direct_kv = 'W_k_direct' in state_dict and 'W_v_direct' in state_dict
        
        if use_direct_kv:
            print("Detected direct K-V approach in the model")
            # For direct approach, extract the value vectors directly
            if 'W_v_direct' in state_dict:
                # Get the value matrix directly
                v = state_dict['W_v_direct'].cpu()
                
                # Check if the model has a value normalization layer
                if 'norm_v.weight' in state_dict and 'norm_v.bias' in state_dict:
                    # Apply normalization if available
                    weight = state_dict['norm_v.weight'].cpu()
                    bias = state_dict['norm_v.bias'].cpu()
                    v = v * weight + bias
                
                # Calculate norms
                v_norms = torch.norm(v, p=2, dim=1)
                
                print(f"Extracted {len(v)} value vectors directly with shape {v.shape}")
                return v.numpy(), v_norms.numpy(), model_info
            else:
                raise ValueError("Could not find W_v_direct in the direct K-V model")
        else:
            # Original memory bank approach - rest of the existing function
            print("Using memory bank approach to compute value vectors")
            
            # Extract model dimensions and memory indices
            if 'W_q.weight' in state_dict and 'W_k.weight' in state_dict and 'W_v.weight' in state_dict:
                n = state_dict['W_v.weight'].shape[1]  # Input dimension
                a = state_dict['W_q.weight'].shape[0]  # Attention dimension
                
                # Get memory indices
                memory_indices = None
                if 'memory_indices' in state_dict:
                    memory_indices = state_dict['memory_indices'].cpu()
                    m = len(memory_indices)
                else:
                    # Estimate based on common ratios
                    m = 8 * n  # Common default is 8*n
            else:
                raise ValueError("Could not determine model dimensions from state_dict")
            
            print(f"Model dimensions: n={n}, m={m}, a={a}")
            
            # Verify memory indices are within bounds of data tensor
            if memory_indices is not None:
                max_index = memory_indices.max().item()
                data_size = data_tensor.shape[0]
                
                if max_index >= data_size:
                    print(f"Warning: Memory indices in model go up to {max_index} but dataset only has {data_size} samples")
                    print(f"Generating synthetic data for out-of-bounds indices")
                    
                    # Create extended tensor with synthetic data for missing indices
                    extended_tensor = torch.zeros((max_index + 1, data_tensor.shape[1]), 
                                               dtype=data_tensor.dtype)
                    
                    # Copy actual data
                    extended_tensor[:data_size] = data_tensor
                    
                    # Generate synthetic data for the rest
                    if data_size > 0:
                        # Use mean and std of real data for better synthetic data
                        mean = data_tensor.mean(dim=0)
                        std = data_tensor.std(dim=0)
                        extended_tensor[data_size:] = torch.randn(
                            (max_index + 1 - data_size, data_tensor.shape[1]), 
                            dtype=data_tensor.dtype) * std + mean
                    else:
                        # Just use random normal if no real data
                        extended_tensor[data_size:] = torch.randn(
                            (max_index + 1 - data_size, data_tensor.shape[1]), 
                            dtype=data_tensor.dtype)
                    
                    data_tensor = extended_tensor
                    print(f"Extended data tensor to size {data_tensor.shape}")
            
            # Create minimal ST model for computing value vectors
            from ST import SparseTransformer
            
            # Create a very minimal model just to compute value vectors
            st_model = SparseTransformer(
                X=data_tensor,
                n=n,
                m=m,
                a=a,
                st_model_path="temp.pth",  # Temporary path
                device=device,
                use_direct_kv=False  # Force memory bank approach
            )
            
            # Load state dict
            st_model.load_state_dict(state_dict)
            
            # Get memory indices (now guaranteed to be within bounds)
            memory_indices = st_model.memory_indices
            
            # Compute value vectors
            st_model.eval()
            with torch.no_grad():
                # Get memory samples
                X_cross = data_tensor[memory_indices]
                
                # Preprocess
                C = st_model.preprocess(X_cross)
                X_cross = X_cross / C
                
                # Compute value vectors
                v = st_model.norm_v(st_model.W_v(X_cross))
                
                # Calculate norms
                v_norms = torch.norm(v, p=2, dim=1)
            
            print(f"Computed {len(v)} value vectors with shape {v.shape}")
            return v.cpu().numpy(), v_norms.cpu().numpy(), model_info
            
    except Exception as e:
        print(f"Error computing value vectors: {e}")
        import traceback
        traceback.print_exc()
        return None, None, {}

# New functions for norm histogram visualization

def plot_norm_histogram(vector_norms, model_info, bins=50, figsize=(14, 8), 
                        model_type="SAE", log_scale=False, title=None, save_path=None):
    """
    Plot histogram of feature vector L2 norms
    
    Args:
        vector_norms: Array of L2 norms for each feature vector
        model_info: Dictionary with model information
        bins: Number of bins for histogram
        figsize: Figure size (width, height) in inches
        model_type: Type of model ('SAE' or 'ST') for title
        log_scale: Whether to use log scale for y-axis
        title: Optional custom title
        save_path: Optional path to save the figure
    """
    if vector_norms is None or len(vector_norms) == 0:
        print("No norms to plot.")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histogram
    counts, edges, patches = ax.hist(vector_norms, bins=bins, alpha=0.7, color='royalblue')
    
    # Add statistics as text
    stats_text = (
        f"Features: {len(vector_norms)}\n"
        f"Mean: {np.mean(vector_norms):.4f}\n"
        f"Median: {np.median(vector_norms):.4f}\n"
        f"Min: {np.min(vector_norms):.4f}\n"
        f"Max: {np.max(vector_norms):.4f}\n"
        f"Std Dev: {np.std(vector_norms):.4f}"
    )
    
    # Add dead ratio if available
    if model_info and 'dead_ratio' in model_info and model_info['dead_ratio'] > 0:
        stats_text += f"\nDead Ratio: {model_info['dead_ratio']*100:.2f}%"
    
    # Add stats text to plot
    ax.text(0.95, 0.95, stats_text,
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set log scale if requested
    if log_scale:
        ax.set_yscale('log')
        ax.set_ylabel('Log Count')
    else:
        ax.set_ylabel('Count')
    
    ax.set_xlabel('L2 Norm')
    ax.set_title('Distribution of Feature Vector L2 Norms')
    
    # Set main title
    if title:
        main_title = title
    else:
        step_str = f" (Step {model_info['step']})" if model_info.get('step', 0) > 0 else ""
        val_loss_str = f" - Val Loss: {model_info.get('val_loss', 0):.4f}" if model_info.get('val_loss') is not None else ""
        #main_title = f"{model_type} Model: Feature Vector Norm Distribution{step_str}{val_loss_str}"
    
    #fig.suptitle(main_title, fontsize=14)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
    
    # Save figure if requested
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    # Explicitly show the plot
    plt.show()
    
    return fig

def plot_cumulative_norm_distribution(vector_norms, figsize=(12, 6), 
                                     percentiles_to_mark=[50, 80, 90, 95, 99],
                                     title="Cumulative Norm Distribution", 
                                     save_path=None):
    """
    Plot cumulative distribution of feature norms showing what percentage
    of total norm is contained in the top N features
    
    Args:
        vector_norms: Array of L2 norms for each feature vector
        figsize: Figure size (width, height) in inches
        percentiles_to_mark: Percentiles to mark on the plot
        title: Plot title
        save_path: Optional path to save the figure
    """
    # Sort norms in descending order
    sorted_norms = np.sort(vector_norms)[::-1]
    
    # Calculate cumulative sum and normalize
    cum_sum = np.cumsum(sorted_norms)
    cum_sum_normalized = cum_sum / cum_sum[-1] * 100
    
    # Create x-axis (feature rank)
    x = np.arange(1, len(sorted_norms) + 1)
    x_pct = x / len(sorted_norms) * 100
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot cumulative sum
    ax.plot(x, cum_sum_normalized, 'b-', linewidth=2)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set labels and title
    ax.set_xlabel('Number of Features (sorted by norm, descending)')
    ax.set_ylabel('Cumulative % of Total Norm')
    ax.set_title(title)
    
    # Add diagonal reference line representing uniform distribution
    ax.plot([1, len(sorted_norms)], [0, 100], 'r--', alpha=0.5, 
            label='Uniform distribution')
    
    # Mark interesting percentiles
    for p in percentiles_to_mark:
        idx = np.searchsorted(cum_sum_normalized, p)
        if idx < len(x):
            feature_count = x[idx]
            feature_pct = x_pct[idx]
            
            # Draw horizontal and vertical lines
            ax.axhline(y=p, color='g', linestyle=':', alpha=0.5)
            ax.axvline(x=feature_count, color='g', linestyle=':', alpha=0.5)
            
            # Add annotation
            ax.text(feature_count * 1.05, p * 0.98, 
                   f"{p}% of norm in top {feature_count} features ({feature_pct:.1f}%)",
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Set axis limits
    ax.set_xlim(0, len(sorted_norms))
    ax.set_ylim(0, 101)
    
    # Add legend
    ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if requested
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    # Explicitly show the plot
    plt.show()
    
    return fig

def analyze_feature_norms(model_path=None, model_type=None, dataset_path=None,
                        bins=50, figsize=(14, 8), log_scale=False, 
                        save_dir=None, device='cpu', 
                        n_samples=10000, show_cumulative=True):
    """
    Analyze and visualize feature vector norms from a trained SAE or ST model
    
    Args:
        model_path: Path to the model file (if None, will try to auto-detect)
        model_type: Type of model ('sae' or 'st', if None, will try to auto-detect)
        dataset_path: Path to the dataset (needed for ST models, auto-detected if None)
        bins: Number of bins for histogram
        figsize: Figure size (width, height) in inches
        log_scale: Whether to use log scale for y-axis
        save_dir: Optional directory to save figures
        device: Device to use for computation
        n_samples: Number of data samples to load for ST models
        show_cumulative: Whether to also show cumulative distribution plot
    """
    # Auto-detect model path and type if not provided
    if model_path is None:
        # Find models
        model_paths = find_model_paths()
        
        # Use specified model type if provided
        if model_type is not None:
            if model_type.lower() in model_paths:
                model_path = model_paths[model_type.lower()]
            else:
                print(f"No {model_type.upper()} model found. Please check models directory.")
                return
        # If no model type specified, use first available model
        else:
            if model_paths:
                # Use SAE by default if available
                if 'sae' in model_paths:
                    model_type = 'sae'
                    model_path = model_paths['sae']
                else:
                    model_type = next(iter(model_paths.keys()))
                    model_path = model_paths[model_type]
            else:
                print("No models found. Please check models directory.")
                return
    
    # If model path provided but not model type, try to infer from filename
    if model_path is not None and model_type is None:
        file_name = os.path.basename(model_path).lower()
        if 'sae' in file_name:
            model_type = 'sae'
        elif 'st' in file_name:
            model_type = 'st'
        else:
            print(f"Could not determine model type from filename. Please specify model_type.")
            return
    
    print(f"Using {model_type.upper()} model from {model_path}")
    
    # Prepare save paths if directory is provided
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        histogram_save_path = os.path.join(save_dir, f"{model_name}_norm_histogram.png")
        cumulative_save_path = os.path.join(save_dir, f"{model_name}_cumulative_norm.png")
    else:
        histogram_save_path = None
        cumulative_save_path = None
    
    # For ST models, we need a dataset to compute value vectors
    if model_type.lower() == 'st':
        # Auto-detect dataset if not provided
        if dataset_path is None:
            dataset_path = find_dataset_paths()
            if dataset_path is None:
                print("No dataset found. Please provide dataset_path.")
                return
        
        # Check if we need to get memory indices first to determine minimum samples required
        try:
            checkpoint = torch.load(model_path, map_location=device)
            state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
            min_samples = None
            
            if 'memory_indices' in state_dict:
                memory_indices = state_dict['memory_indices'].cpu()
                min_samples = memory_indices.max().item() + 1
                print(f"Model needs at least {min_samples} samples based on memory indices")
        except Exception as e:
            print(f"Warning: Could not pre-check memory indices: {e}")
            min_samples = None
        
        # Load dataset
        data_tensor = load_dataset(dataset_path, n_samples=n_samples, min_samples=min_samples)
        
        # Load model and compute value vectors
        vectors, vector_norms, model_info = load_st_model_and_compute_value_vectors(
            model_path, data_tensor, device=device)
        
        model_type_name = "ST"
    else:  # SAE model
        # Load SAE decoder weights
        vectors, vector_norms, model_info = load_sae_decoder_weights(model_path, device=device)
        
        model_type_name = "SAE"
    
    # Plot vectors
    if vector_norms is not None:
        print(f"\nCreating norm histogram plot...")
        # Plot histogram
        fig1 = plot_norm_histogram(
            vector_norms,
            model_info,
            bins=bins,
            figsize=figsize,
            model_type=model_type_name,
            log_scale=log_scale,
            save_path=histogram_save_path
        )
        
        # Plot cumulative distribution if requested
        if show_cumulative:
            print(f"\nCreating cumulative distribution plot...")
            step_str = f" (Step {model_info['step']})" if model_info.get('step', 0) > 0 else ""
            fig2 = plot_cumulative_norm_distribution(
                vector_norms,
                title=f"{model_type_name} Model: Cumulative Norm Distribution{step_str}",
                save_path=cumulative_save_path
            )
    else:
        print("Failed to load feature vectors.")

def main():
    """Main function handling command line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze and visualize feature vector norms")
    parser.add_argument("--model_path", type=str, default=None, 
                       help="Path to the trained model file (if not specified, will try to auto-detect)")
    parser.add_argument("--model_type", type=str, choices=["sae", "st"], default=None,
                       help="Type of model (sae or st, if not specified, will try to auto-detect)")
    parser.add_argument("--dataset_path", type=str, default=None,
                       help="Path to the dataset CSV file (needed for ST models, auto-detected if None)")
    parser.add_argument("--bins", type=int, default=50,
                       help="Number of bins for histogram")
    parser.add_argument("--log_scale", action="store_true",
                       help="Use log scale for y-axis")
    parser.add_argument("--no_cumulative", action="store_true",
                       help="Don't show cumulative distribution plot")
    parser.add_argument("--save_dir", type=str, default=None,
                       help="Directory to save figures (optional)")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use for computation (default: cpu)")
    parser.add_argument("--n_samples", type=int, default=10000,
                       help="Number of data samples to load for ST models (default: 10000)")
    
    args = parser.parse_args()
    
    analyze_feature_norms(
        model_path=args.model_path,
        model_type=args.model_type,
        dataset_path=args.dataset_path,
        bins=args.bins,
        log_scale=args.log_scale,
        save_dir=args.save_dir,
        device=args.device,
        n_samples=args.n_samples,
        show_cumulative=not args.no_cumulative
    )

if __name__ == "__main__":
    main()