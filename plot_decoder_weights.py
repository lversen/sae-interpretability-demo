import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Fix OpenMP error by setting environment variable
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

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

def find_dataset_paths(base_dir="data", dataset_pattern="*.csv"):
    """
    Automatically find dataset files in the given directory
    
    Args:
        base_dir: Base directory to search in
        dataset_pattern: Glob pattern to match dataset files
        
    Returns:
        List of dataset paths
    """
    # Make sure base directory exists
    if not os.path.exists(base_dir):
        print(f"Warning: Directory {base_dir} does not exist")
        return []
    
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

def load_sae_decoder_weights(model_path, device='cpu'):
    """
    Load decoder weights from an SAE model
    
    Args:
        model_path: Path to the SAE model file
        device: Device to use for computation
        
    Returns:
        Tuple with decoder weights and corresponding norms
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
                'lambda_l1': checkpoint.get('lambda_l1', 0)
            }
        else:
            state_dict = checkpoint
            model_info = {'step': 0, 'dead_ratio': 0, 'lambda_l1': 0}
        
        # Extract decoder weights
        if 'W_d.weight' in state_dict:
            # Get decoder weight matrix - shape (n, m) for SAE
            # Each column is a feature vector of dimension n
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

def load_st_model_and_compute_value_vectors(model_path, data_tensor, device='cpu'):
    """
    Load an ST model and compute its value vectors
    
    Args:
        model_path: Path to the ST model file
        data_tensor: Input data tensor for computing value vectors
        device: Device to use for computation
        
    Returns:
        Tuple with value vectors and corresponding norms
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
                'lambda_l1': checkpoint.get('lambda_l1', 0)
            }
        else:
            state_dict = checkpoint
            model_info = {'step': 0, 'dead_ratio': 0, 'lambda_l1': 0}
        
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
            device=device
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

def plot_feature_vectors(vectors, vector_norms, model_info, 
                      input_shape=(28, 28), num_vectors=100, 
                      rows=10, cols=10, figsize=(20, 20),
                      cmap='coolwarm', title_prefix="", save_path=None):
    """
    Plot feature vectors (SAE decoder weights or ST value vectors)
    
    Args:
        vectors: Feature vectors to visualize
        vector_norms: Norms of the feature vectors
        model_info: Dictionary with model information
        input_shape: Shape to reshape vectors to (e.g., (28, 28) for MNIST)
        num_vectors: Number of vectors to visualize
        rows, cols: Number of rows and columns in the grid
        figsize: Figure size (width, height) in inches
        cmap: Colormap to use for visualization
        title_prefix: Prefix for the plot title
        save_path: Optional path to save the figure
    """
    if vectors is None or vectors.size == 0:
        print("No vectors provided.")
        return

    # Check vector shape and transpose if necessary
    if vectors.shape[0] == np.prod(input_shape) and vectors.shape[1] != np.prod(input_shape):
        # Shape is (n, m) where n is input dimension and m is feature count
        # This is correct for SAE decoder weights
        vector_dim = vectors.shape[0]
        feature_count = vectors.shape[1]
        print(f"Using columns of shape ({vector_dim},) as feature vectors")
        vectors_are_columns = True
    elif vectors.shape[1] == np.prod(input_shape) and vectors.shape[0] != np.prod(input_shape):
        # Shape is (m, n) where m is feature count and n is input dimension
        # This is correct for ST value vectors
        vector_dim = vectors.shape[1]
        feature_count = vectors.shape[0]
        print(f"Using rows of shape ({vector_dim},) as feature vectors")
        vectors_are_columns = False
    else:
        # Cannot determine orientation, try to guess based on the product
        if vectors.shape[0] == np.prod(input_shape):
            vector_dim = vectors.shape[0]
            feature_count = vectors.shape[1]
            print(f"Assuming columns of shape ({vector_dim},) as feature vectors")
            vectors_are_columns = True
        elif vectors.shape[1] == np.prod(input_shape):
            vector_dim = vectors.shape[1]
            feature_count = vectors.shape[0]
            print(f"Assuming rows of shape ({vector_dim},) as feature vectors")
            vectors_are_columns = False
        else:
            print(f"Warning: Vector shape {vectors.shape} doesn't match input shape {input_shape} with product {np.prod(input_shape)}")
            print("Will try to reshape individual vectors, but this may fail.")
            vector_dim = max(vectors.shape)
            feature_count = min(vectors.shape)
            # Try to guess based on which dimension is closer to the target
            vectors_are_columns = abs(vectors.shape[0] - np.prod(input_shape)) < abs(vectors.shape[1] - np.prod(input_shape))
            
    # Sort vectors by norm
    sorted_indices = np.argsort(-vector_norms)
    
    # Determine how many vectors to plot
    plot_vectors = min(num_vectors, feature_count)
    if plot_vectors != num_vectors:
        print(f"Warning: Requested {num_vectors} vectors but only {feature_count} available")
    
    # Create figure for visualization
    fig = plt.figure(figsize=figsize)
    
    # Create title with model info
    title = title_prefix
    if model_info:
        step_str = f" (Step {model_info['step']})" if model_info.get('step', 0) > 0 else ""
        title += f"{step_str}\nTop {plot_vectors} by L2 norm"
        if model_info.get('dead_ratio', 0) > 0:
            title += f" - Dead Features: {model_info['dead_ratio']*100:.1f}%"
    
    fig.suptitle(title, fontsize=20)
    
    # Create grid for vectors
    gs = GridSpec(rows, cols, figure=fig, wspace=0.1, hspace=0.1)
    
    # Plot each vector
    for i in range(min(rows * cols, plot_vectors)):
        # Get the index of the i-th highest norm vector
        idx = sorted_indices[i]
        
        # Get the vector based on orientation
        if vectors_are_columns:
            vector = vectors[:, idx]
        else:
            vector = vectors[idx, :]
        
        # Calculate norm for title
        norm = vector_norms[idx]
        
        # Create subplot
        ax = fig.add_subplot(gs[i // cols, i % cols])
        
        try:
            # Reshape to the input shape
            vector_img = vector.reshape(input_shape)
            
            # Determine color range for better visualization
            vmax = max(abs(vector_img.max()), abs(vector_img.min()))
            if vmax == 0:
                vmax = 1  # Avoid division by zero
            
            # Plot the vector image
            im = ax.imshow(vector_img, cmap=cmap, vmin=-vmax, vmax=vmax)
            
            # Add title with feature index and norm
            ax.set_title(f"F{idx}\nNorm: {norm:.2f}", fontsize=8)
            
        except Exception as e:
            print(f"Error reshaping vector to {input_shape}: {e}")
            ax.text(0.5, 0.5, f"Reshape Error\nVector shape: {vector.shape}", 
                   ha='center', va='center', transform=ax.transAxes)
        
        # Remove axis ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Add a colorbar to the last successful plot
    if 'im' in locals():
        plt.colorbar(im, ax=fig.axes, shrink=0.7)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    
    # Save figure if requested
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    return fig

def plot_decoder_weights(model_path=None, model_type=None, dataset_path=None, input_shape=(28, 28), 
                         num_weights=100, rows=10, cols=10, figsize=(20, 20),
                         cmap='coolwarm', save_path=None, device='cpu', n_samples=10000):
    """
    Plot decoder weights from a trained SAE or ST model
    
    Args:
        model_path: Path to the model file (if None, will try to auto-detect)
        model_type: Type of model ('sae' or 'st', if None, will try to auto-detect)
        dataset_path: Path to the dataset (needed for ST models, auto-detected if None)
        input_shape: Shape to reshape weights to (e.g., (28, 28) for MNIST)
        num_weights: Number of weights to visualize
        rows, cols: Number of rows and columns in the grid
        figsize: Figure size (width, height) in inches
        cmap: Colormap to use for visualization
        save_path: Optional path to save the figure
        device: Device to use for computation
        n_samples: Number of data samples to load for ST models (default: 10000)
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
    
    # For ST models, we need a dataset to compute value vectors
    if model_type.lower() == 'st':
        # Auto-detect dataset if not provided
        if dataset_path is None:
            dataset_path = find_dataset_paths()
            if dataset_path is None:
                print("No dataset found. Please provide --dataset_path.")
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
        
        title_prefix = "ST Model: Value Vectors"
    else:  # SAE model
        # Load SAE decoder weights
        vectors, vector_norms, model_info = load_sae_decoder_weights(model_path, device=device)
        
        title_prefix = "SAE Model: Decoder Weights"
    
    # Plot vectors
    if vectors is not None:
        plot_feature_vectors(
            vectors,
            vector_norms,
            model_info,
            input_shape=input_shape,
            num_vectors=num_weights,
            rows=rows,
            cols=cols,
            figsize=figsize,
            cmap=cmap,
            title_prefix=title_prefix,
            save_path=save_path
        )
    else:
        print("Failed to load feature vectors.")

def main():
    """Main function handling command line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot decoder weights or value vectors from trained models")
    parser.add_argument("--model_path", type=str, default=None, 
                       help="Path to the trained model file (if not specified, will try to auto-detect)")
    parser.add_argument("--model_type", type=str, choices=["sae", "st"], default=None,
                       help="Type of model (sae or st, if not specified, will try to auto-detect)")
    parser.add_argument("--dataset_path", type=str, default=None,
                       help="Path to the dataset CSV file (needed for ST models, auto-detected if None)")
    parser.add_argument("--num_weights", type=int, default=100,
                       help="Number of weights to visualize")
    parser.add_argument("--rows", type=int, default=10,
                       help="Number of rows in the grid")
    parser.add_argument("--cols", type=int, default=10, 
                       help="Number of columns in the grid")
    parser.add_argument("--save_path", type=str, default=None,
                       help="Path to save the figure (optional)")
    parser.add_argument("--cmap", type=str, default="coolwarm",
                       help="Colormap to use (default: coolwarm)")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use for computation (default: cpu)")
    parser.add_argument("--n_samples", type=int, default=10000,
                       help="Number of data samples to load for ST models (default: 10000)")
    
    args = parser.parse_args()
    
    plot_decoder_weights(
        model_path=args.model_path,
        model_type=args.model_type,
        dataset_path=args.dataset_path,
        num_weights=args.num_weights,
        rows=args.rows,
        cols=args.cols,
        cmap=args.cmap,
        save_path=args.save_path,
        device=args.device,
        n_samples=args.n_samples
    )

if __name__ == "__main__":
    main()