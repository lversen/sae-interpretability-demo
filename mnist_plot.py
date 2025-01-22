import torch
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from SAE import SparseAutoencoder
from ST import SparseTransformer

def plot_mnist_results(model_type='sae', activation_threshold=None):
    """
    Plot MNIST results with optional activation threshold.
    
    Args:
        model_type (str): Type of model ('sae' or 'st')
        activation_threshold (float, optional): If provided, threshold activations as fraction of max value
    """
    # Load more MNIST samples for ST to work with
    df = pd.read_csv('data/mnist_train.csv')
    X_full = df.iloc[:2000, 1:].values.astype(np.float32)  # Get 2000 samples for ST initialization
    samples = df.iloc[:20]  # Keep 20 samples for visualization
    
    # Setup model parameters
    D = 784  # Input dimension for MNIST
    F = 2 * D  # Feature dimension 
    
    if model_type.lower() == 'sae':
        model_path = 'models/sae_model_mnist_train.csv_mnist.pth'
        model = SparseAutoencoder(D, F, model_path)
    elif model_type.lower() == 'st':
        model_path = 'models/st_model_mnist_train.csv_mnist.pth'
        M = 100  # Attention dimension
        model = SparseTransformer(X_full, D, F, M, model_path)
        
        # Load model state with parameter conversion
        state_dict = torch.load(model_path, weights_only=True)
        model.load_state_dict(state_dict)  # Uses the overridden load_state_dict method
    else:
        raise ValueError("model_type must be 'sae' or 'st'")

    model.eval()

    # Get reconstructions for visualization samples
    X = samples.iloc[:, 1:].values.astype(np.float32)  # Skip label column
    X_tensor = torch.from_numpy(X).to(model.device)
    
    with torch.no_grad():
        features = None
        if model_type.lower() == 'sae':
            _, reconstructions, features = model(X_tensor)
            features = features.cpu().numpy()
            reconstructions = reconstructions.cpu().numpy()
        else:
            X_full_tensor = torch.from_numpy(X_full).to(model.device)
            _, reconstructions, features, V = model(X_tensor)
            features = features.cpu().numpy()
            reconstructions = reconstructions.cpu().numpy()

        if features is None:
            raise ValueError("Failed to get features from model")
        
        # Apply threshold if specified
        if activation_threshold is not None:
            max_activation = np.max(features)
            threshold_value = max_activation * activation_threshold
            features_thresholded = features.copy()
            features_thresholded[features_thresholded < threshold_value] = 0
            features_thresholded[features_thresholded >= threshold_value] = max_activation
            
            # Get new reconstructions with thresholded features
            features_tensor = torch.from_numpy(features_thresholded).to(model.device)
            if model_type.lower() == 'sae':
                reconstructions = model.decoder(features_tensor).cpu().numpy() + model.b_dec.cpu().numpy()
            else:
                X_full_tensor = torch.from_numpy(X_full).to(model.device)
                V = model.W_v(X_full_tensor)  # Get value vectors
                reconstructions = torch.matmul(features_tensor, V).cpu().numpy()
        else:
            features_thresholded = features

    # Plot original vs reconstructions
    fig, axes = plt.subplots(4, 10, figsize=(15, 6))
    plt.suptitle(f'MNIST Original (rows 1,3) vs {model_type.upper()} Reconstruction (rows 2,4)', y=1.02)
    
    # Fixed range for MNIST digits
    vmin_digits = 0
    vmax_digits = 255
    
    for i in range(10):
        # Plot original digit
        axes[0, i].imshow(X[i].reshape(28, 28), cmap='gray', vmin=vmin_digits, vmax=vmax_digits)
        axes[0, i].axis('off')
        axes[1, i].set_title(f'Label: {samples.iloc[i, 0]}')
        
        # Plot reconstruction
        axes[1, i].imshow(reconstructions[i].reshape(28, 28), cmap='gray', vmin=vmin_digits, vmax=vmax_digits)
        axes[1, i].axis('off')
        
        if i < 10:  # Plot next set of digits
            axes[2, i].imshow(X[i+10].reshape(28, 28), cmap='gray', vmin=vmin_digits, vmax=vmax_digits)
            axes[2, i].axis('off')
            axes[3, i].set_title(f'Label: {samples.iloc[i+10, 0]}')
            
            axes[3, i].imshow(reconstructions[i+10].reshape(28, 28), cmap='gray', vmin=vmin_digits, vmax=vmax_digits)
            axes[3, i].axis('off')

    plt.tight_layout()
    plt.show()

    # Plot feature activations
    plt.figure(figsize=(15, 5))
    vmin = np.min(features_thresholded[features_thresholded > 0]) if np.any(features_thresholded > 0) else 0
    vmax = np.max(features_thresholded)
    plt.imshow(features_thresholded.T, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar(label=f'Feature Activation [min={vmin:.2f}, max={vmax:.2f}]')
    
    title = f'{model_type.upper()} Feature Activations'
    if activation_threshold is not None:
        title += f' (threshold={activation_threshold:.2f})'
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('Feature Index')
    
    # Add text showing threshold if used
    if activation_threshold is not None:
        plt.text(0.02, 0.98, f'Threshold: {threshold_value:.4f}', 
                transform=plt.gca().transAxes, 
                bbox=dict(facecolor='white', alpha=0.8))
    plt.show()

    # Calculate and print reconstruction error
    mse = np.mean((X - reconstructions) ** 2)
    print(f"{model_type.upper()} Reconstruction MSE: {mse:.4f}")
    print(f"Max activation value: {np.max(features):.4f}")
    if activation_threshold is not None:
        print(f"Threshold value: {threshold_value:.4f}")
        print(f"Percentage of activations above threshold: {100 * np.mean(features > threshold_value):.2f}%")

# Example usage
print("Plotting SAE results without threshold:")
plot_mnist_results('st')  # No threshold
# =============================================================================
# 
# print("\nPlotting SAE results with threshold:")
# plot_mnist_results('st', activation_threshold=0.1)  # With threshold
# =============================================================================
