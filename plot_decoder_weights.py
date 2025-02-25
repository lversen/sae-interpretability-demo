import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Fix OpenMP error by setting environment variable
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def find_model_paths(base_dir="models", model_pattern="*mnist*.pth"):
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

def plot_decoder_weights(model_path=None, model_type=None, input_shape=(28, 28), 
                         num_weights=100, rows=10, cols=10, figsize=(20, 20),
                         cmap='coolwarm', save_path=None):
    """
    Plot decoder weights from a trained SAE or ST model
    
    Args:
        model_path: Path to the model file (if None, will try to auto-detect)
        model_type: Type of model ('sae' or 'st', if None, will try to auto-detect)
        input_shape: Shape to reshape weights to (e.g., (28, 28) for MNIST)
        num_weights: Number of weights to visualize
        rows, cols: Number of rows and columns in the grid
        figsize: Figure size (width, height) in inches
        cmap: Colormap to use for visualization
        save_path: Optional path to save the figure
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
    
    # Load model state dict
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        
        # Check if it's a checkpoint or just a state dict
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Extract decoder weights based on model type
    if model_type.lower() == 'sae':
        if 'W_d.weight' in state_dict:
            weight_matrix = state_dict['W_d.weight'].numpy()
            title_prefix = "SAE Model: Decoder Weights"
        else:
            raise ValueError("Could not find decoder weights (W_d.weight) in the model")
    else:  # ST model
        if 'W_v.weight' in state_dict:
            weight_matrix = state_dict['W_v.weight'].numpy()
            title_prefix = "ST Model: Value Projection Weights"
        else:
            raise ValueError("Could not find value weights (W_v.weight) in the model")
    
    print(f"Weight matrix shape: {weight_matrix.shape}")
    
    # Calculate L2 norms to sort by importance
    feature_norms = np.linalg.norm(weight_matrix, axis=0)
    sorted_indices = np.argsort(-feature_norms)  # Sort by descending norm
    
    # Determine how many weights to plot
    plot_weights = min(num_weights, weight_matrix.shape[1])
    if plot_weights != num_weights:
        print(f"Warning: Requested {num_weights} weights but model only has {weight_matrix.shape[1]}")
    
    # Create figure for visualization
    fig = plt.figure(figsize=figsize)
    fig.suptitle(f"{title_prefix}\n(Top {plot_weights} by L2 norm)", fontsize=20)
    
    # Create grid for weights
    gs = GridSpec(rows, cols, figure=fig, wspace=0.1, hspace=0.1)
    
    # Plot each weight vector
    for i in range(min(rows * cols, plot_weights)):
        # Get the index of the i-th highest norm weight
        idx = sorted_indices[i]
        
        # Get the weight vector
        weight = weight_matrix[:, idx]
        
        # Calculate norm for title
        norm = feature_norms[idx]
        
        # Reshape to the input shape
        weight_img = weight.reshape(input_shape)
        
        # Create subplot
        ax = fig.add_subplot(gs[i // cols, i % cols])
        
        # Determine color range for better visualization
        vmax = max(abs(weight_img.max()), abs(weight_img.min()))
        
        # Plot the weight image
        im = ax.imshow(weight_img, cmap=cmap, vmin=-vmax, vmax=vmax)
        
        # Add title with feature index and norm
        ax.set_title(f"F{idx}\nNorm: {norm:.2f}", fontsize=8)
        
        # Remove axis ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Add a colorbar
    plt.colorbar(im, ax=fig.axes, shrink=0.7)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    
    # Save figure if requested
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    return feature_norms, sorted_indices

def main():
    """Main function handling command line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot decoder weights from trained models")
    parser.add_argument("--model_path", type=str, default=None, 
                       help="Path to the trained model file (if not specified, will try to auto-detect)")
    parser.add_argument("--model_type", type=str, choices=["sae", "st"], default=None,
                       help="Type of model (sae or st, if not specified, will try to auto-detect)")
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
    
    args = parser.parse_args()
    
    plot_decoder_weights(
        model_path=args.model_path,
        model_type=args.model_type,
        num_weights=args.num_weights,
        rows=args.rows,
        cols=args.cols,
        cmap=args.cmap,
        save_path=args.save_path
    )

if __name__ == "__main__":
    main()