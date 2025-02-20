import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_v_matrix_columns(model, num_cols=10, figsize=(15, 3)):
    """
    Plot columns from the V matrix of a Sparse Transformer model trained on MNIST.
    
    Args:
        model: The trained Sparse Transformer model
        num_cols: Number of columns to plot
        figsize: Figure size for the plot
    """
    # Get a batch of input data to generate V matrix
    if not hasattr(model, 'X') or model.X is None:
        raise ValueError("Model doesn't have stored data (X attribute)")
        
    with torch.no_grad():
        # Get random batch
        random_indices = torch.randint(0, model.X.shape[0], (64,))
        x = model.X[random_indices].to(model.device)
        
        # Forward pass to get V matrix
        _, _, _, V = model(x)
        
        # Move V to CPU and convert to numpy
        V = V.cpu().numpy()
        
    # Create figure
    fig, axes = plt.subplots(1, num_cols, figsize=figsize)
    fig.suptitle('Selected Columns from V Matrix Reshaped as MNIST Images')
    
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
        axes[i].set_title(f'Col {idx}')
    
    plt.tight_layout()
    plt.show()

def plot_v_matrix_stats(model):
    """
    Plot statistics about the V matrix.
    
    Args:
        model: The trained Sparse Transformer model
    """
    with torch.no_grad():
        # Get random batch
        random_indices = torch.randint(0, model.X.shape[0], (64,))
        x = model.X[random_indices].to(model.device)
        
        # Forward pass to get V matrix
        _, _, _, V = model(x)
        
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
    
    # Plot heatmap of V matrix values
    im = ax2.imshow(V, aspect='auto', cmap='viridis')
    ax2.set_title(f'V Matrix Values (Sparsity: {sparsity:.2%})')
    ax2.set_xlabel('Input Dimension')
    ax2.set_ylabel('Feature Index')
    plt.colorbar(im, ax=ax2)
    
    plt.tight_layout()
    plt.show()