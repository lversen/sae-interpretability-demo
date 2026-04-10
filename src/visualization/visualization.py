import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import torch

def load_sae_or_st_model(model_path, model_class, *args, **kwargs):
    """
    Load a previously trained SAE or ST model
    
    Args:
        model_path: Path to the saved model
        model_class: SparseAutoencoder or SparseTransformer class
        *args, **kwargs: Arguments to pass to model constructor
    """
    model = model_class(*args, **kwargs)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def analyze_feature_sparsity(activations, thresholds=None):
    """
    Analyze the sparsity of feature activations at different thresholds
    
    Args:
        activations: Feature activations array [samples, features]
        thresholds: List of thresholds to check (default: [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    """
    if thresholds is None:
        thresholds = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    
    results = {}
    for threshold in thresholds:
        # Calculate fraction of activations above threshold
        active_fraction = np.mean(np.abs(activations) > threshold)
        
        # Calculate fraction of features that activate at least once
        features_active = np.mean(np.max(np.abs(activations), axis=0) > threshold)
        
        # Calculate statistics of activation magnitudes
        magnitudes = np.abs(activations[np.abs(activations) > threshold]) if threshold > 0 else np.abs(activations)
        magnitude_stats = {
            'min': np.min(magnitudes) if len(magnitudes) > 0 else 0,
            'max': np.max(magnitudes) if len(magnitudes) > 0 else 0,
            'mean': np.mean(magnitudes) if len(magnitudes) > 0 else 0,
            'median': np.median(magnitudes) if len(magnitudes) > 0 else 0,
        }
        
        results[threshold] = {
            'active_fraction': active_fraction,
            'features_active': features_active,
            'magnitude_stats': magnitude_stats
        }
    
    # Format as DataFrame for easier viewing
    df_data = []
    for threshold, data in results.items():
        df_data.append({
            'threshold': threshold,
            'active_fraction': data['active_fraction'],
            'features_active': data['features_active'],
            'min_magnitude': data['magnitude_stats']['min'],
            'max_magnitude': data['magnitude_stats']['max'],
            'mean_magnitude': data['magnitude_stats']['mean'],
            'median_magnitude': data['magnitude_stats']['median'],
        })
    
    return pd.DataFrame(df_data)

def visualize_weight_distribution(model, is_sae=True):
    """
    Visualize the distribution of weight norms in the decoder/value matrix
    
    Args:
        model: The SAE or ST model
        is_sae: Boolean indicating if this is an SAE (True) or ST (False) model
    """
    with torch.no_grad():
        if is_sae:
            # Get decoder weight matrix
            W_d = model.W_d.weight.data.cpu().numpy()
            title = "SAE Decoder Weight Norms"
        else:
            # For ST, we need to extract the value vectors during a forward pass
            # This is a simplified approximation - actual ST would need sample data
            key_weights = model.input_proj.weight.data.cpu().numpy()
            title = "ST Key Weight Norms"
            W_d = key_weights
    
    # Calculate L2 norms of columns
    norms = np.linalg.norm(W_d, axis=0)
    
    # Plot distribution
    plt.figure(figsize=(10, 6))
    plt.hist(norms, bins=50)
    plt.title(title)
    plt.xlabel("L2 Norm")
    plt.ylabel("Count")
    plt.grid(alpha=0.3)
    plt.show()
    
    print(f"Weight norm statistics:")
    print(f"Min: {norms.min():.4f}")
    print(f"Max: {norms.max():.4f}")
    print(f"Mean: {norms.mean():.4f}")
    print(f"Median: {norms.median():.4f}")
    
    return norms

def visualize_feature_activations_per_class(activations, labels, max_features=20, figsize=(15, 8)):
    """
    Visualize average feature activations per class
    
    Args:
        activations: Feature activations array [samples, features]
        labels: Class labels array [samples]
        max_features: Maximum number of features to display
        figsize: Figure size for the plot
    """
    # Get unique labels
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    
    # Calculate average activations per class
    class_activations = []
    for label in unique_labels:
        class_mask = (labels == label)
        avg_activations = np.mean(activations[class_mask], axis=0)
        class_activations.append(avg_activations)
    
    # Convert to numpy array
    class_activations = np.array(class_activations)
    
    # Select top features by variance across classes
    feature_variance = np.var(class_activations, axis=0)
    top_features = np.argsort(-feature_variance)[:max_features]
    
    # Create plot
    plt.figure(figsize=figsize)
    plt.imshow(class_activations[:, top_features], aspect='auto', cmap='viridis')
    plt.colorbar(label='Average Activation')
    plt.xlabel('Feature Index (sorted by variance)')
    plt.ylabel('Class')
    plt.title('Average Feature Activation by Class')
    plt.yticks(np.arange(num_classes), unique_labels)
    plt.tight_layout()
    plt.show()
    
    return class_activations, top_features

def plot_activation_patterns(activations, num_samples=10, num_features=50, figsize=(15, 8)):
    """
    Plot activation patterns for a few samples
    
    Args:
        activations: Feature activations array [samples, features]
        num_samples: Number of samples to display
        num_features: Number of features to display
        figsize: Figure size for the plot
    """
    # Select samples and features
    if activations.shape[0] > num_samples:
        sample_indices = np.random.choice(activations.shape[0], num_samples, replace=False)
    else:
        sample_indices = np.arange(activations.shape[0])
    
    if activations.shape[1] > num_features:
        # Select features with highest activation variance
        feature_variance = np.var(activations, axis=0)
        feature_indices = np.argsort(-feature_variance)[:num_features]
    else:
        feature_indices = np.arange(activations.shape[1])
    
    # Create plot
    plt.figure(figsize=figsize)
    plt.imshow(activations[sample_indices][:, feature_indices], aspect='auto', cmap='viridis')
    plt.colorbar(label='Activation')
    plt.xlabel('Feature Index (sorted by variance)')
    plt.ylabel('Sample Index')
    plt.title('Feature Activation Patterns')
    plt.tight_layout()
    plt.show()

def compare_models_sparsity(models_dict, sample_data, threshold=1e-3):
    """
    Compare sparsity of different models
    
    Args:
        models_dict: Dictionary mapping model names to (model, is_sae) tuples
        sample_data: Sample data to compute activations [samples, features]
        threshold: Activation threshold
    """
    results = {}
    
    for model_name, (model, is_sae) in models_dict.items():
        # Compute activations
        with torch.no_grad():
            data_tensor = torch.from_numpy(sample_data).float().to(model.device)
            if is_sae:
                activations = model.feature_activations(data_tensor).cpu().numpy()
            else:
                activations = model.feature_activations(data_tensor).cpu().numpy()
        
        # Calculate statistics
        sparsity = np.mean(np.abs(activations) <= threshold)
        max_activation = np.max(np.abs(activations))
        active_features = np.mean(np.max(np.abs(activations), axis=0) > threshold)
        
        results[model_name] = {
            'sparsity': sparsity,
            'max_activation': max_activation,
            'active_features': active_features,
            'activations': activations
        }
    
    # Create comparison table
    df_data = []
    for model_name, data in results.items():
        df_data.append({
            'model': model_name,
            'sparsity (%)': data['sparsity'] * 100,
            'active_features (%)': data['active_features'] * 100,
            'max_activation': data['max_activation']
        })
    
    print("Model Sparsity Comparison:")
    return pd.DataFrame(df_data), results