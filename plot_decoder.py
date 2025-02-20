import matplotlib.pyplot as plt
import numpy as np
import torch

def get_feature_matrix(model):
    """
    Extract the feature matrix from either an SAE or ST model.
    
    Args:
        model: Either a SparseAutoencoder or SparseTransformer model
        
    Returns:
        numpy.ndarray: The feature matrix (decoder weights for SAE, V matrix for ST)
    """
    with torch.no_grad():
        if hasattr(model, 'W_d'):  # SAE model
            feature_matrix = model.W_d.weight.cpu().numpy().T
        else:  # ST model
            # Get random batch to generate V matrix
            if not hasattr(model, 'X') or model.X is None:
                raise ValueError("ST model doesn't have stored data (X attribute)")
            
            random_indices = torch.randint(0, model.X.shape[0], (64,))
            x = model.X[random_indices].to(model.device)
            
            # Forward pass to get V matrix
            _, _, _, V = model(x)
            feature_matrix = V.cpu().numpy()
            
    return feature_matrix

def plot_feature_matrix_columns(model, num_cols=10, figsize=(15, 3), input_shape=(28, 28)):
    """
    Plot columns from the feature matrix of either an SAE or ST model.
    
    Args:
        model: Either a SparseAutoencoder or SparseTransformer model
        num_cols: Number of columns to plot
        figsize: Figure size for the plot
        input_shape: Shape to reshape features into (e.g., (28, 28) for MNIST)
    """
    # Get feature matrix
    feature_matrix = get_feature_matrix(model)
    
    # Create figure
    fig, axes = plt.subplots(1, num_cols, figsize=figsize)
    model_type = "SAE" if hasattr(model, 'W_d') else "ST"
    fig.suptitle(f'Selected Features from {model_type} Model Reshaped as {input_shape} Images')
    
    # Select evenly spaced columns to display
    step = feature_matrix.shape[0] // num_cols
    selected_indices = np.arange(0, feature_matrix.shape[0], step)[:num_cols]
    
    # Plot each selected column
    for i, idx in enumerate(selected_indices):
        # Reshape column to specified input shape
        img = feature_matrix[idx].reshape(input_shape)
        
        # Normalize to [0, 1] for visualization
        img = (img - img.min()) / (img.max() - img.min())
        
        # Plot
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Feature {idx}')
    
    plt.tight_layout()
    plt.show()

def plot_feature_matrix_stats(model, threshold=1e-3):
    """
    Plot statistics about the feature matrix.
    
    Args:
        model: Either a SparseAutoencoder or SparseTransformer model
        threshold: Threshold for considering values as zero when computing sparsity
    """
    # Get feature matrix
    feature_matrix = get_feature_matrix(model)
    model_type = "SAE" if hasattr(model, 'W_d') else "ST"
    
    # Calculate statistics
    norms = np.linalg.norm(feature_matrix, axis=1)
    sparsity = np.mean(np.abs(feature_matrix) <= threshold)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot histogram of feature norms
    ax1.hist(norms, bins=50)
    ax1.set_title(f'Distribution of {model_type} Feature Matrix Norms')
    ax1.set_xlabel('L2 Norm')
    ax1.set_ylabel('Count')
    
    # Plot heatmap of feature matrix values
    im = ax2.imshow(feature_matrix, aspect='auto', cmap='viridis')
    ax2.set_title(f'{model_type} Feature Matrix Values (Sparsity: {sparsity:.2%})')
    ax2.set_xlabel('Input Dimension')
    ax2.set_ylabel('Feature Index')
    plt.colorbar(im, ax=ax2)
    
    plt.tight_layout()
    plt.show()

def compare_feature_matrices(sae_model, st_model, num_cols=5, figsize=(15, 6), input_shape=(28, 28)):
    """
    Compare feature matrices from SAE and ST models side by side.
    
    Args:
        sae_model: SparseAutoencoder model
        st_model: SparseTransformer model
        num_cols: Number of columns to plot per model
        figsize: Figure size for the plot
        input_shape: Shape to reshape features into (e.g., (28, 28) for MNIST)
    """
    # Get feature matrices
    sae_features = get_feature_matrix(sae_model)
    st_features = get_feature_matrix(st_model)
    
    # Create figure
    fig, axes = plt.subplots(2, num_cols, figsize=figsize)
    fig.suptitle('Comparison of SAE and ST Features')
    
    # Select evenly spaced columns to display
    sae_step = sae_features.shape[0] // num_cols
    st_step = st_features.shape[0] // num_cols
    sae_indices = np.arange(0, sae_features.shape[0], sae_step)[:num_cols]
    st_indices = np.arange(0, st_features.shape[0], st_step)[:num_cols]
    
    # Plot SAE features
    for i, idx in enumerate(sae_indices):
        img = sae_features[idx].reshape(input_shape)
        img = (img - img.min()) / (img.max() - img.min())
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title(f'SAE Feature {idx}')
    
    # Plot ST features
    for i, idx in enumerate(st_indices):
        img = st_features[idx].reshape(input_shape)
        img = (img - img.min()) / (img.max() - img.min())
        axes[1, i].imshow(img, cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title(f'ST Feature {idx}')
    
    plt.tight_layout()
    plt.show()

def plot_feature_matrix_activation_patterns(model, input_data, threshold=1e-3, figsize=(12, 4)):
    """
    Analyze and plot feature activation patterns for either model type.
    
    Args:
        model: Either a SparseAutoencoder or SparseTransformer model
        input_data: Input data tensor to analyze activations
        threshold: Threshold for considering activations as non-zero
        figsize: Figure size for the plot
    """
    # Move input data to model's device if needed
    if isinstance(input_data, np.ndarray):
        input_data = torch.from_numpy(input_data).float()
    input_data = input_data.to(model.device)
    
    # Get activations
    with torch.no_grad():
        if hasattr(model, 'W_d'):  # SAE model
            _, _, activations = model(input_data)
        else:  # ST model
            _, _, activations, _ = model(input_data)
    
    activations = activations.cpu().numpy()
    model_type = "SAE" if hasattr(model, 'W_d') else "ST"
    
    # Calculate statistics
    activation_freqs = np.mean(np.abs(activations) > threshold, axis=0)
    avg_activation = np.mean(np.abs(activations), axis=0)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot activation frequencies
    ax1.hist(activation_freqs, bins=50)
    ax1.set_title(f'{model_type} Feature Activation Frequencies')
    ax1.set_xlabel('Fraction of Samples')
    ax1.set_ylabel('Number of Features')
    
    # Plot average activation strengths
    ax2.hist(avg_activation, bins=50)
    ax2.set_title(f'{model_type} Average Feature Activation Strengths')
    ax2.set_xlabel('Average Activation')
    ax2.set_ylabel('Number of Features')
    
    plt.tight_layout()
    plt.show()