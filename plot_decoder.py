import matplotlib.pyplot as plt
import numpy as np
import torch
from ST import SparseTransformer
from SAE import SparseAutoencoder

def plot_decoder_matrix_columns(model,  X, num_cols=10, figsize=(15, 3), fixed_indices=None):
    """
    Plot columns from the V matrix of a Sparse Transformer model or the decoder matrix of a Sparse Autoencoder model trained on MNIST.
    
    Args:
        model: The trained Sparse Transformer or Sparse Autoencoder model
        num_cols: Number of columns to plot
        figsize: Figure size for the plot
        fixed_indices: Fixed indices to use for selecting data
    """
    
        
    with torch.no_grad():
        # Get fixed batch
        if fixed_indices is None:
            fixed_indices = np.arange(4096)  # Default fixed indices if not provided
        x = X[fixed_indices].to(model.device)
        
        # Forward pass to get V matrix or decoder matrix
        if isinstance(model, SparseTransformer):
            _, _, _, V = model(x)
        elif isinstance(model, SparseAutoencoder):
            model(x)
            V = model.W_d.weight.T
        else:
            raise ValueError("Unsupported model type")
        
        # Move V to CPU and convert to numpy
        V = V.cpu().numpy()
        
    # Ensure num_cols does not exceed the number of columns in V
    num_cols = min(num_cols, V.shape[0])

    # Determine grid size
    n_rows = int(np.ceil(np.sqrt(num_cols)))
    n_cols = int(np.ceil(num_cols / n_rows))

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle('Selected Columns from Decoder Matrix Reshaped as MNIST Images')
    
    # Flatten axes array for easy iteration
    axes = axes.flatten()
    
    # Select evenly spaced columns to display
    step = V.shape[0] // num_cols
    selected_indices = np.arange(0, V.shape[0], step)[:num_cols]
    
    # Plot each selected column
    for i, idx in enumerate(selected_indices):
        # Reshape column to 28x28 image
        img = V[idx].reshape(28, 28)
        
        # Normalize to [0, 1] for visualization
        img = (img - img.min()) / (img.max() - img.min())
        
        # Plot
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
    
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
    plt.show()

def plot_decoder_matrix_stats(model, X, fixed_indices=None):
    """
    Plot statistics about the V matrix or decoder matrix.
    
    Args:
        model: The trained Sparse Transformer or Sparse Autoencoder model
        fixed_indices: Fixed indices to use for selecting data
    """
    with torch.no_grad():
        # Get fixed batch
        if fixed_indices is None:
            fixed_indices = np.arange(64)  # Default fixed indices if not provided
        x = X[fixed_indices].to(model.device)
        
        # Forward pass to get V matrix or decoder matrix
        if isinstance(model, SparseTransformer):
            _, _, _, V = model(x)
        elif isinstance(model, SparseAutoencoder):
            model(x)
            V = model.W_d.weight.T
        else:
            raise ValueError("Unsupported model type")
        
        # Move V to CPU and convert to numpy
        V = V.cpu().numpy()
    
    # Calculate statistics
    norms = np.linalg.norm(V, axis=1)
    sparsity = np.mean(np.abs(V) <= 1e-3)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot histogram of column norms
    ax1.hist(norms, bins=50)
    ax1.set_title('Distribution of V Matrix Column Norms')
    ax1.set_xlabel('L2 Norm')
    ax1.set_ylabel('Count')
    
    # Plot heatmap of V matrix values with discrete colormap
    im = ax2.imshow(np.abs(V), aspect='auto', cmap='coolwarm', interpolation='nearest')
    ax2.set_title(f'V Matrix Values (Sparsity: {sparsity:.2%})')
    ax2.set_xlabel('Input Dimension')
    ax2.set_ylabel('Feature Index')
    plt.colorbar(im, ax=ax2)
    
    plt.tight_layout()
    plt.show()