import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

def analyze_and_visualize_features(features, title="Feature Analysis", save_dir="feature_analysis"):
    """
    Thoroughly analyze and visualize feature statistics to diagnose preprocessing issues.
    Works with both numpy arrays and PyTorch tensors.
    
    Args:
        features: numpy array or PyTorch tensor of shape [n_samples, n_features]
        title: title for the plots
        save_dir: directory to save visualizations
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to numpy if tensor
    if isinstance(features, torch.Tensor):
        features_np = features.detach().cpu().numpy()
    else:
        features_np = features
    
    # Basic statistics
    mean_val = np.mean(features_np)
    std_val = np.std(features_np)
    min_val = np.min(features_np)
    max_val = np.max(features_np)
    
    # Check for NaNs or Infs
    has_nan = np.isnan(features_np).any()
    has_inf = np.isinf(features_np).any()
    
    # Calculate sample norms
    sample_norms = np.linalg.norm(features_np, axis=1)
    mean_norm = np.mean(sample_norms)
    
    # Print statistics
    print(f"===== {title} =====")
    print(f"Shape: {features_np.shape}")
    print(f"Mean: {mean_val:.6f}")
    print(f"Std Dev: {std_val:.6f}")
    print(f"Min: {min_val:.6f}")
    print(f"Max: {max_val:.6f}")
    print(f"Contains NaN: {has_nan}")
    print(f"Contains Inf: {has_inf}")
    print(f"Average L2 norm: {mean_norm:.6f}")
    print(f"Expected norm for standard SAE preprocessing: {np.sqrt(features_np.shape[1]):.6f}")
    
    # Create visualization with multiple subplots
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Histogram of all values
    axs[0, 0].hist(features_np.flatten(), bins=100, alpha=0.7)
    axs[0, 0].set_title('Distribution of Feature Values')
    axs[0, 0].set_xlabel('Value')
    axs[0, 0].set_ylabel('Frequency')
    
    # 2. Histogram of sample norms
    axs[0, 1].hist(sample_norms, bins=50, alpha=0.7)
    axs[0, 1].axvline(mean_norm, color='r', linestyle='--', label=f'Mean: {mean_norm:.2f}')
    axs[0, 1].axvline(np.sqrt(features_np.shape[1]), color='g', linestyle='--', 
                   label=f'Target: {np.sqrt(features_np.shape[1]):.2f}')
    axs[0, 1].set_title('Distribution of Sample L2 Norms')
    axs[0, 1].set_xlabel('L2 Norm')
    axs[0, 1].set_ylabel('Frequency')
    axs[0, 1].legend()
    
    # 3. QQ plot to check for normality
    feature_means = np.mean(features_np, axis=0)
    stats.probplot(feature_means, dist="norm", plot=axs[0, 2])
    axs[0, 2].set_title('QQ Plot of Feature Means')
    
    # 4. Feature activation heatmap (first 100 samples, first 100 features)
    display_samples = min(100, features_np.shape[0])
    display_features = min(100, features_np.shape[1])
    im = axs[1, 0].imshow(features_np[:display_samples, :display_features], 
                       aspect='auto', cmap='viridis')
    axs[1, 0].set_title(f'Feature Activation Heatmap (First {display_samples}x{display_features})')
    axs[1, 0].set_xlabel('Feature Index')
    axs[1, 0].set_ylabel('Sample Index')
    plt.colorbar(im, ax=axs[1, 0])
    
    # 5. Distribution of feature means
    axs[1, 1].hist(feature_means, bins=50, alpha=0.7)
    axs[1, 1].set_title('Distribution of Feature Means')
    axs[1, 1].set_xlabel('Mean Value')
    axs[1, 1].set_ylabel('Frequency')
    
    # 6. Distribution of feature standard deviations
    feature_stds = np.std(features_np, axis=0)
    axs[1, 2].hist(feature_stds, bins=50, alpha=0.7)
    axs[1, 2].set_title('Distribution of Feature Standard Deviations')
    axs[1, 2].set_xlabel('Standard Deviation')
    axs[1, 2].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{title.replace(' ', '_').lower()}.png"), dpi=300)
    plt.close()
    
    # Create additional visualizations for outlier analysis
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Box plot of sample norms
    axs[0].boxplot(sample_norms)
    axs[0].set_title('Box Plot of Sample L2 Norms')
    axs[0].set_ylabel('L2 Norm')
    
    # 2. Box plot of feature values (sampled)
    # Flatten and sample for efficiency
    flat_sample = features_np.flatten()[np.random.choice(features_np.size, 10000)]
    axs[1].boxplot(flat_sample)
    axs[1].set_title('Box Plot of Feature Values (Sampled)')
    axs[1].set_ylabel('Value')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{title.replace(' ', '_').lower()}_outliers.png"), dpi=300)
    plt.close()
    
    return {
        'mean': mean_val,
        'std': std_val,
        'min': min_val,
        'max': max_val,
        'has_nan': has_nan,
        'has_inf': has_inf,
        'mean_norm': mean_norm,
        'target_norm': np.sqrt(features_np.shape[1])
    }

def preprocess_gptneo_features(features, method='robust_norm', debug=True):
    """
    Specialized preprocessing for GPT Neo features with multiple methods.
    
    Args:
        features: numpy array or PyTorch tensor of features
        method: preprocessing method to use
            - 'robust_norm': L2 normalization with outlier handling (recommended)
            - 'standardize': zero mean, unit variance standardization
            - 'minmax': scale to [0, 1] range
            - 'l2_norm': standard L2 normalization to sqrt(n)
            - 'tanh_norm': apply tanh to squash extreme values, then normalize
        debug: whether to print debug information
    
    Returns:
        Preprocessed features (same type as input)
    """
    # Convert to tensor if numpy
    is_numpy = isinstance(features, np.ndarray)
    if is_numpy:
        features_tensor = torch.from_numpy(features).float()
    else:
        features_tensor = features.float()
    
    # Get original device
    original_device = features_tensor.device
    
    # Move to CPU for processing
    features_tensor = features_tensor.cpu()
    
    # Apply selected preprocessing method
    input_dim = features_tensor.shape[1]
    
    if method == 'standardize':
        # Standard scaling (zero mean, unit variance)
        mean = torch.mean(features_tensor, dim=0, keepdim=True)
        std = torch.std(features_tensor, dim=0, keepdim=True) + 1e-8
        preprocessed = (features_tensor - mean) / std
        
    elif method == 'minmax':
        # Min-max scaling to [0, 1]
        min_val = torch.min(features_tensor, dim=0, keepdim=True)[0]
        max_val = torch.max(features_tensor, dim=0, keepdim=True)[0]
        range_val = max_val - min_val + 1e-8
        preprocessed = (features_tensor - min_val) / range_val
        
    elif method == 'l2_norm':
        # Standard L2 normalization to sqrt(n)
        norms = torch.norm(features_tensor, p=2, dim=1, keepdim=True)
        target_norm = np.sqrt(input_dim)
        scale_factor = target_norm / (norms + 1e-8)
        preprocessed = features_tensor * scale_factor
        
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
        # 1. Clip extreme outliers (based on percentiles)
        q1, q3 = torch.quantile(features_tensor, torch.tensor([0.25, 0.75]))
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        clipped = torch.clamp(features_tensor, lower_bound, upper_bound)
        
        # 2. Standardize
        mean = torch.mean(clipped, dim=0, keepdim=True)
        std = torch.std(clipped, dim=0, keepdim=True) + 1e-8
        standardized = (clipped - mean) / std
        
        # 3. Apply soft scaling with sigmoid to further reduce impact of outliers
        scaled = torch.sigmoid(standardized) * 2 - 1  # Map to [-1, 1]
        
        # 4. Normalize to expected norm
        norms = torch.norm(scaled, p=2, dim=1, keepdim=True)
        target_norm = np.sqrt(input_dim)
        scale_factor = target_norm / (norms + 1e-8)
        preprocessed = scaled * scale_factor
    else:
        raise ValueError(f"Unknown preprocessing method: {method}")
    
    # Print debug info if requested
    if debug:
        print(f"Preprocessing with method: {method}")
        print(f"Original shape: {features_tensor.shape}")
        print(f"Original mean: {torch.mean(features_tensor):.6f}")
        print(f"Original std: {torch.std(features_tensor):.6f}")
        print(f"Original min: {torch.min(features_tensor):.6f}")
        print(f"Original max: {torch.max(features_tensor):.6f}")
        print(f"Original mean norm: {torch.mean(torch.norm(features_tensor, dim=1)):.6f}")
        print(f"Target norm: {np.sqrt(input_dim):.6f}")
        print(f"Preprocessed mean: {torch.mean(preprocessed):.6f}")
        print(f"Preprocessed std: {torch.std(preprocessed):.6f}")
        print(f"Preprocessed min: {torch.min(preprocessed):.6f}")
        print(f"Preprocessed max: {torch.max(preprocessed):.6f}")
        print(f"Preprocessed mean norm: {torch.mean(torch.norm(preprocessed, dim=1)):.6f}")
    
    # Return to original device
    preprocessed = preprocessed.to(original_device)
    
    # Convert back to numpy if input was numpy
    if is_numpy:
        preprocessed = preprocessed.numpy()
    
    return preprocessed


# Modify SAE's preprocess function
def modified_sae_preprocess(features):
    """
    Enhanced preprocessing function for SAE when working with GPT Neo features.
    This is a drop-in replacement for SAE.preprocess method.
    """
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features.astype(np.float32))
        return_numpy = True
    else:
        return_numpy = False
    
    # Apply robust normalization
    preprocessed = preprocess_gptneo_features(features, method='robust_norm', debug=False)
    
    # For SAE, we just need to return the scaling factor C
    # The actual scaling is done in the calling function with X / C
    # Here, we return 1.0 since we've already applied the preprocessing
    C = 1.0
    
    if return_numpy and isinstance(preprocessed, torch.Tensor):
        preprocessed = preprocessed.numpy()
    
    return preprocessed, C


# Modify ST's preprocess function
def modified_st_preprocess(features):
    """
    Enhanced preprocessing function for ST when working with GPT Neo features.
    This can be used as a drop-in replacement for ST.preprocess method.
    """
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features.astype(np.float32))
        return_numpy = True
    else:
        return_numpy = False
    
    # Apply robust normalization
    preprocessed = preprocess_gptneo_features(features, method='robust_norm', debug=False)
    
    # For ST, we just need to return the scaling factor C
    # The actual scaling is done in the calling function with X / C
    # Here, we return 1.0 since we've already applied the preprocessing
    C = 1.0
    
    if return_numpy and isinstance(preprocessed, torch.Tensor):
        preprocessed = preprocessed.numpy()
    
    return preprocessed, C


# IMPLEMENTATION GUIDE:
# 1. First, analyze your GPT Neo features:
#    features = torch.load('gptneo_features/layer6_features.pt')['features']
#    analyze_and_visualize_features(features, "GPT Neo Layer 6 Features")

# 2. Replace the preprocessing in your training code:
#    # Before loading into model:
#    train_feature_extract, _ = modified_sae_preprocess(train_feature_extract)
#    val_feature_extract, _ = modified_sae_preprocess(val_feature_extract)

# 3. Alternative: Patch the model's preprocess method:
#    # For SAE:
#    original_preprocess = sae_model.preprocess
#    sae_model.preprocess = lambda X: 1.0  # Return 1.0 since we preprocess data in advance

#    # For ST:
#    original_preprocess = st_model.preprocess
#    st_model.preprocess = lambda X: 1.0  # Return 1.0 since we preprocess data in advance


# EXAMPLE USAGE in main.py:
"""
# After extracting features but before training:
if args.dataset == 'custom' and 'gptneo' in args.custom_features_file.lower():
    print("Applying specialized preprocessing for GPT Neo features...")
    # Analyze original features
    analyze_and_visualize_features(train_feature_extract, "GPT Neo Original")
    
    # Apply preprocessing
    train_feature_extract = preprocess_gptneo_features(train_feature_extract, method='robust_norm')
    val_feature_extract = preprocess_gptneo_features(val_feature_extract, method='robust_norm')
    
    # Analyze preprocessed features
    analyze_and_visualize_features(train_feature_extract, "GPT Neo Preprocessed")
    
    # Create tensors with preprocessed data
    train_tensor = torch.from_numpy(train_feature_extract).float().to(device)
    val_tensor = torch.from_numpy(val_feature_extract).float().to(device)
    
    # Patch the preprocess methods to avoid double preprocessing
    if args.model_type in ["sae", "both"]:
        # Define a monkey-patched preprocess method for SAE
        def patched_sae_preprocess(self, X):
            # Just return 1.0 since we've already preprocessed
            return 1.0
    
    if args.model_type in ["st", "both"]:
        # Define a monkey-patched preprocess method for ST
        def patched_st_preprocess(self, X):
            # Just return 1.0 since we've already preprocessed
            return 1.0
"""