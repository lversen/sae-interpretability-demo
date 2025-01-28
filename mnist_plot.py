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
    # Load MNIST samples
    df = pd.read_csv('data/mnist_train.csv')
    X_full = df.iloc[:2000, 1:].values.astype(np.float32)  # Get 2000 samples for ST initialization
    samples = df.iloc[:20]  # Keep 20 samples for visualization
    
    # Setup model parameters
    D = 784  # Input dimension for MNIST
    F = 8 * D  # Feature dimension 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if model_type.lower() == 'sae':
        model_path = 'models/sae_model_mnist_train.csv_mnist.pth'
        model = SparseAutoencoder(D, F, model_path)
    elif model_type.lower() == 'st':
        model_path = 'models/st_model_mnist_train.csv_mnist.pth'
        M = int(D/2)  # Attention dimension
        model = SparseTransformer(
            X=X_full,
            D=D,
            F=F,
            M=M,
            st_model_path=model_path,
            num_heads=8  # Number of attention heads
        )
    else:
        raise ValueError("model_type must be 'sae' or 'st'")

    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Get reconstructions for visualization samples
    X = samples.iloc[:, 1:].values.astype(np.float32)  # Skip label column
    X_tensor = torch.from_numpy(X).to(model.device)
    
    with torch.no_grad():
        if model_type.lower() == 'sae':
            _, reconstructions, features = model(X_tensor)
            features = features.cpu().numpy()
            reconstructions = reconstructions.cpu().numpy()
        else:  # ST model
            X_full_tensor = torch.from_numpy(X_full).to(model.device)
            _, reconstructions, attention_weights = model(X_tensor)
            features = attention_weights.cpu().numpy()
            reconstructions = reconstructions.cpu().numpy()

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
                # For ST, recompute reconstructions using thresholded attention weights
                query = model.input_proj(X_tensor)
                key = model.input_proj(model.X_data)
                value = model.input_proj(model.X_data)
                
                query = model.norm_q(query)
                key = model.norm_kv(key)
                value = model.norm_kv(value)
                
                attn_output = torch.matmul(features_tensor, value)
                reconstructions = model.output_proj(attn_output).cpu().numpy()
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
if __name__ == "__main__":
    print("Plotting SAE results without threshold:")
    plot_mnist_results('st')  # No threshold
    
