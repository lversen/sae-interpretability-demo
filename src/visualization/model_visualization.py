import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import seaborn as sns
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Any
import umap
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import math
from scipy.cluster.hierarchy import linkage, dendrogram
from torch.utils.data import DataLoader, TensorDataset


def load_model_for_visualization(model_path, model_class, *args, **kwargs):
    """
    Load a trained model for visualization
    
    Args:
        model_path: Path to the saved model file
        model_class: Class of the model (SparseAutoencoder or SparseTransformer)
        *args, **kwargs: Additional arguments for model constructor
    
    Returns:
        Loaded model instance
    """
    model = model_class(*args, **kwargs)
    if model_path.endswith('.pt') or model_path.endswith('.pth'):
        # Try loading as state dict first
        try:
            model.load_state_dict(torch.load(model_path))
        except:
            # If that fails, try loading as a checkpoint
            checkpoint = torch.load(model_path)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                raise ValueError(f"Could not load model from {model_path}")
    model.eval()
    return model


def get_feature_vectors(model, is_sae=True):
    """
    Extract feature vectors from a model
    
    Args:
        model: Trained model (SAE or ST)
        is_sae: Whether the model is an SAE (True) or ST (False)
    
    Returns:
        Feature vectors as numpy array
    """
    with torch.no_grad():
        if is_sae:
            # For SAE, get decoder weight matrix
            features = model.W_d.weight.data.cpu().numpy()
        else:
            # For ST, get value projection matrix
            features = model.W_v.weight.data.cpu().numpy()
    
    return features


def visualize_decoder_weights(model, is_sae=True, input_shape=None, top_n=16, figsize=(12, 8)):
    """
    Visualize decoder weight vectors (for image data)
    
    Args:
        model: Trained model (SAE or ST)
        is_sae: Whether the model is an SAE (True) or ST (False)
        input_shape: Shape to reshape features to (e.g., (28, 28) for MNIST)
        top_n: Number of top features to visualize
        figsize: Figure size
    """
    # Get feature vectors
    feature_vectors = get_feature_vectors(model, is_sae)
    
    # Calculate L2 norms to identify important features
    feature_norms = np.linalg.norm(feature_vectors, axis=0)
    top_indices = np.argsort(-feature_norms)[:top_n]
    
    # Determine layout
    n_cols = min(4, top_n)
    n_rows = math.ceil(top_n / n_cols)
    
    # Create plot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    # Default shape if not provided
    if input_shape is None:
        dim = int(np.sqrt(feature_vectors.shape[0]))
        input_shape = (dim, dim)
    
    # Plot each feature
    for i, idx in enumerate(top_indices):
        if i < len(axes):
            feature = feature_vectors[:, idx]
            
            # Reshape for images
            if input_shape is not None:
                feature = feature.reshape(input_shape)
            
            # Plot with proper normalization
            vmax = max(abs(feature.max()), abs(feature.min()))
            im = axes[i].imshow(feature, cmap='coolwarm', vmin=-vmax, vmax=vmax)
            axes[i].set_title(f"Feature {idx}\nNorm: {feature_norms[idx]:.3f}")
            axes[i].axis('off')
    
    # Add colorbar
    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    
    plt.tight_layout()
    plt.suptitle("Top Model Features Visualization" + (" (SAE)" if is_sae else " (ST)"), y=1.02)
    plt.show()
    
    return top_indices


def visualize_mnist_reconstructions(model, data, is_sae=True, num_samples=10, figsize=(15, 4)):
    """
    Visualize original and reconstructed MNIST digits
    
    Args:
        model: Trained model (SAE or ST)
        data: Input data tensor [samples, 784]
        is_sae: Whether the model is an SAE (True) or ST (False)
        num_samples: Number of samples to visualize
        figsize: Figure size
    """
    model.eval()
    
    # Convert to tensor
    if isinstance(data, np.ndarray):
        data_tensor = torch.from_numpy(data).float().to(model.device)
    else:
        data_tensor = data.to(model.device)
    
    # Select samples
    indices = np.random.choice(data_tensor.shape[0], num_samples, replace=False)
    samples = data_tensor[indices]
    
    # Get reconstructions
    with torch.no_grad():
        if is_sae:
            _, reconstructions, _ = model(samples)
        else:
            _, reconstructions, _, _ = model(samples)
    
    # Convert to numpy
    originals = samples.cpu().numpy()
    reconstructions = reconstructions.cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(2, num_samples, figsize=figsize)
    
    # Plot originals and reconstructions
    for i in range(num_samples):
        # Original
        axes[0, i].imshow(originals[i].reshape(28, 28), cmap='gray')
        axes[0, i].set_title(f"Original {i}")
        axes[0, i].axis('off')
        
        # Reconstruction
        axes[1, i].imshow(reconstructions[i].reshape(28, 28), cmap='gray')
        axes[1, i].set_title(f"Reconstructed {i}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.suptitle(f"MNIST Reconstructions - {'SAE' if is_sae else 'ST'} Model", y=1.02)
    plt.show()
    
    # Calculate reconstruction error
    mse = np.mean((originals - reconstructions) ** 2)
    print(f"Average reconstruction MSE: {mse:.6f}")
    
    return mse


def plot_feature_activations(model, data, is_sae=True, threshold=1e-3, figsize=(14, 10)):
    """
    Visualize feature activations across a dataset
    
    Args:
        model: Trained model (SAE or ST)
        data: Input data tensor [samples, features]
        is_sae: Whether the model is an SAE (True) or ST (False)
        threshold: Activation threshold
        figsize: Figure size
    """
    model.eval()
    
    # Convert to tensor
    if isinstance(data, np.ndarray):
        data_tensor = torch.from_numpy(data).float().to(model.device)
    else:
        data_tensor = data.to(model.device)
    
    # Get feature activations
    with torch.no_grad():
        if is_sae:
            _, _, activations = model(data_tensor)
        else:
            _, _, activations, _ = model(data_tensor)
    
    # Convert to numpy
    activations = activations.cpu().numpy()
    
    # Create figure with multiple plots
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig)
    
    # 1. Activation matrix heatmap
    ax1 = fig.add_subplot(gs[0, :])
    im = ax1.imshow(activations[:100, :200].T, aspect='auto', cmap='viridis')
    ax1.set_title("Feature Activations (100 samples × 200 features)")
    ax1.set_xlabel("Sample")
    ax1.set_ylabel("Feature")
    plt.colorbar(im, ax=ax1)
    
    # 2. Activation distribution histogram
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(activations.flatten(), bins=50, log=True)
    ax2.set_title("Activation Magnitude Distribution (log scale)")
    ax2.set_xlabel("Activation Magnitude")
    ax2.set_ylabel("Count")
    ax2.axvline(threshold, color='red', linestyle='--', 
                label=f"Threshold={threshold}")
    ax2.legend()
    
    # 3. Feature activity heatmap
    ax3 = fig.add_subplot(gs[1, 1])
    feature_activity = (activations > threshold).mean(axis=0)
    sorted_idx = np.argsort(-feature_activity)
    feature_activity_sorted = feature_activity[sorted_idx]
    
    # Create a visualization with color gradient
    feature_idx = np.arange(len(feature_activity_sorted))
    ax3.bar(feature_idx[:50], feature_activity_sorted[:50], 
            color=plt.cm.viridis(np.linspace(0, 1, 50)))
    ax3.set_title("Top 50 Most Active Features")
    ax3.set_xlabel("Feature Rank")
    ax3.set_ylabel("Activation Rate")
    
    # Calculate sparsity statistics
    sparsity = 1 - (activations > threshold).mean()
    active_per_sample = (activations > threshold).sum(axis=1).mean()
    dead_features = np.sum(np.all(activations <= threshold, axis=0))
    
    # Add text box with statistics
    stats_text = (
        f"Sparsity: {sparsity:.1%}\n"
        f"Avg. active features/sample: {active_per_sample:.1f}\n"
        f"Dead features: {dead_features}/{activations.shape[1]} ({dead_features/activations.shape[1]:.1%})"
    )
    ax3.text(0.95, 0.95, stats_text, transform=ax3.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.7})
    
    plt.tight_layout()
    plt.suptitle(f"Feature Activation Analysis - {'SAE' if is_sae else 'ST'} Model", y=1.02)
    plt.show()
    
    return {
        'activations': activations,
        'sparsity': sparsity,
        'active_per_sample': active_per_sample,
        'dead_features': dead_features,
        'feature_activity': feature_activity
    }


def visualize_feature_embedding(model, is_sae=True, method='tsne', feature_labels=None, figsize=(10, 8)):
    """
    Visualize feature vectors in a 2D embedding space
    
    Args:
        model: Trained model (SAE or ST)
        is_sae: Whether the model is an SAE (True) or ST (False)
        method: Embedding method ('tsne', 'pca', 'umap')
        feature_labels: Optional labels for coloring features
        figsize: Figure size
    """
    # Get feature vectors
    feature_vectors = get_feature_vectors(model, is_sae)
    
    # Transpose to get features as rows
    features = feature_vectors.T
    
    # Compute embedding
    if method.lower() == 'tsne':
        embedding = TSNE(n_components=2, random_state=42).fit_transform(features)
        title = "t-SNE Embedding of Feature Vectors"
    elif method.lower() == 'pca':
        pca = PCA(n_components=2, random_state=42)
        embedding = pca.fit_transform(features)
        explained_var = pca.explained_variance_ratio_
        title = f"PCA Embedding of Feature Vectors\nExplained variance: {explained_var[0]:.1%}, {explained_var[1]:.1%}"
    elif method.lower() == 'umap':
        embedding = umap.UMAP(n_components=2, random_state=42).fit_transform(features)
        title = "UMAP Embedding of Feature Vectors"
    else:
        raise ValueError(f"Unknown embedding method: {method}")
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Determine coloring
    if feature_labels is not None:
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=feature_labels, 
                             cmap='tab10', alpha=0.7)
        plt.colorbar(scatter, label="Feature Label")
    else:
        # Color by feature norms
        norms = np.linalg.norm(features, axis=1)
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=norms, 
                             cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label="Feature Norm")
    
    plt.title(title)
    plt.xlabel(f"{method.upper()} Dimension 1")
    plt.ylabel(f"{method.upper()} Dimension 2")
    plt.tight_layout()
    plt.show()
    
    return embedding


def visualize_attention_patterns(st_model, data, num_samples=5, num_features=20, figsize=(12, 10)):
    """
    Visualize attention patterns in the ST model
    
    Args:
        st_model: Trained ST model
        data: Input data tensor [samples, features]
        num_samples: Number of samples to visualize
        num_features: Number of features to include in visualization
        figsize: Figure size
    """
    st_model.eval()
    
    # Convert to tensor
    if isinstance(data, np.ndarray):
        data_tensor = torch.from_numpy(data).float().to(st_model.device)
    else:
        data_tensor = data.to(st_model.device)
    
    # Select samples
    indices = np.random.choice(data_tensor.shape[0], num_samples, replace=False)
    samples = data_tensor[indices]
    
    # Get attention weights
    with torch.no_grad():
        _, _, attn_weights, _ = st_model(samples)
    
    # Convert to numpy
    attn_weights = attn_weights.cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(num_samples, 2, figsize=figsize, 
                            gridspec_kw={'width_ratios': [3, 1]})
    
    # Determine most active features across these samples
    feature_activity = attn_weights.mean(axis=0)
    top_features = np.argsort(-feature_activity)[:num_features]
    
    # Plot each sample's attention pattern
    for i in range(num_samples):
        # Heatmap of top features
        im = axes[i, 0].imshow(attn_weights[i, top_features].reshape(1, -1), 
                              aspect='auto', cmap='viridis')
        axes[i, 0].set_title(f"Sample {i+1} Attention Weights")
        axes[i, 0].set_yticks([])
        
        if i == num_samples-1:
            axes[i, 0].set_xlabel("Feature Index (sorted by activity)")
        
        # Bar chart of total attention
        sorted_attn = np.sort(attn_weights[i])[::-1]
        axes[i, 1].bar(np.arange(len(sorted_attn[:20])), sorted_attn[:20], 
                      color=plt.cm.viridis(np.linspace(0, 1, 20)))
        axes[i, 1].set_title(f"Top 20 Weights")
        
        if i == num_samples-1:
            axes[i, 1].set_xlabel("Rank")
        
        # Add stats as text
        entropy = -np.sum(attn_weights[i] * np.log(attn_weights[i] + 1e-10))
        active = np.sum(attn_weights[i] > 0.01)
        top5_pct = sorted_attn[:5].sum() / sorted_attn.sum()
        
        stats_text = (
            f"Entropy: {entropy:.2f}\n"
            f"Active: {active}\n"
            f"Top-5: {top5_pct:.1%}"
        )
        
        axes[i, 1].text(0.95, 0.95, stats_text, transform=axes[i, 1].transAxes,
                      verticalalignment='top', horizontalalignment='right',
                      bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.7})
    
    plt.tight_layout()
    plt.suptitle("ST Model Attention Pattern Analysis", y=1.02)
    plt.show()
    
    return attn_weights


def hierarchical_feature_clustering(model, is_sae=True, num_clusters=10, figsize=(12, 8)):
    """
    Create hierarchical clustering of model features
    
    Args:
        model: Trained model (SAE or ST)
        is_sae: Whether the model is an SAE (True) or ST (False)
        num_clusters: Number of clusters to highlight
        figsize: Figure size
    """
    # Get feature vectors
    feature_vectors = get_feature_vectors(model, is_sae)
    
    # Transpose to get features as rows
    features = feature_vectors.T
    
    # Compute linkage for hierarchical clustering
    Z = linkage(features, method='ward')
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create dendrogram
    dendrogram(
        Z,
        truncate_mode='lastp',
        p=num_clusters,
        leaf_rotation=90.,
        leaf_font_size=10.,
        show_contracted=True,
        above_threshold_color='gray'
    )
    
    plt.title(f"Hierarchical Clustering of Feature Vectors - {'SAE' if is_sae else 'ST'} Model")
    plt.xlabel("Feature Clusters")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.show()
    
    return Z


def compare_models_features(sae_model, st_model, data, threshold=1e-3, figsize=(15, 10)):
    """
    Compare feature activations between SAE and ST models
    
    Args:
        sae_model: Trained SAE model
        st_model: Trained ST model
        data: Input data tensor [samples, features]
        threshold: Activation threshold
        figsize: Figure size
    """
    sae_model.eval()
    st_model.eval()
    
    # Convert to tensor
    if isinstance(data, np.ndarray):
        data_tensor = torch.from_numpy(data).float()
    else:
        data_tensor = data
    
    # Get feature activations
    with torch.no_grad():
        # SAE activations
        sae_data = data_tensor.to(sae_model.device)
        _, _, sae_activations = sae_model(sae_data)
        sae_activations = sae_activations.cpu().numpy()
        
        # ST activations
        st_data = data_tensor.to(st_model.device)
        _, _, st_activations, _ = st_model(st_data)
        st_activations = st_activations.cpu().numpy()
    
    # Create figure with multiple plots
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig)
    
    # Define common color normalization
    vmax = max(
        np.percentile(sae_activations, 95),
        np.percentile(st_activations, 95)
    )
    
    # 1. SAE activation matrix heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(sae_activations[:50, :100].T, aspect='auto', cmap='viridis', vmax=vmax)
    ax1.set_title("SAE Feature Activations")
    ax1.set_xlabel("Sample")
    ax1.set_ylabel("Feature")
    plt.colorbar(im1, ax=ax1)
    
    # 2. ST activation matrix heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(st_activations[:50, :100].T, aspect='auto', cmap='viridis', vmax=vmax)
    ax2.set_title("ST Feature Activations")
    ax2.set_xlabel("Sample")
    ax2.set_ylabel("Feature")
    plt.colorbar(im2, ax=ax2)
    
    # 3. Activation distribution comparison
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(sae_activations.flatten(), bins=50, alpha=0.5, label="SAE", density=True)
    ax3.hist(st_activations.flatten(), bins=50, alpha=0.5, label="ST", density=True)
    ax3.set_title("Activation Magnitude Distribution")
    ax3.set_xlabel("Activation Magnitude")
    ax3.set_ylabel("Density")
    ax3.axvline(threshold, color='red', linestyle='--', 
                label=f"Threshold={threshold}")
    ax3.legend()
    ax3.set_ylim(0, 0.5)  # Limit y-axis for better comparison
    
    # 4. Sparsity comparison
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Calculate sparsity statistics
    sae_sparsity = 1 - (sae_activations > threshold).mean()
    st_sparsity = 1 - (st_activations > threshold).mean()
    
    sae_active_per_sample = (sae_activations > threshold).sum(axis=1).mean()
    st_active_per_sample = (st_activations > threshold).sum(axis=1).mean()
    
    sae_dead_features = np.sum(np.all(sae_activations <= threshold, axis=0))
    st_dead_features = np.sum(np.all(st_activations <= threshold, axis=0))
    
    # Data for bar chart
    models = ['SAE', 'ST']
    metrics = {
        'Sparsity (%)': [sae_sparsity*100, st_sparsity*100],
        'Avg Features/Sample': [sae_active_per_sample, st_active_per_sample],
        'Dead Features (%)': [sae_dead_features/sae_activations.shape[1]*100, 
                             st_dead_features/st_activations.shape[1]*100]
    }
    
    # Create grouped bar chart
    x = np.arange(len(models))
    width = 0.25
    multiplier = 0
    
    for metric, values in metrics.items():
        offset = width * multiplier
        ax4.bar(x + offset, values, width, label=metric)
        multiplier += 1
    
    # Add labels and legend
    ax4.set_title("Model Comparison")
    ax4.set_xticks(x + width)
    ax4.set_xticklabels(models)
    ax4.set_ylabel("Value")
    ax4.legend(loc='upper left')
    
    plt.tight_layout()
    plt.suptitle("SAE vs ST Feature Activation Comparison", y=1.02)
    plt.show()
    
    # Print detailed comparison
    print("Model Comparison:")
    print(f"{'Metric':<25} {'SAE':<15} {'ST':<15}")
    print("-" * 55)
    print(f"{'Sparsity':<25} {sae_sparsity:.2%} {st_sparsity:.2%}")
    print(f"{'Avg active features/sample':<25} {sae_active_per_sample:.2f} {st_active_per_sample:.2f}")
    print(f"{'Dead features':<25} {sae_dead_features}/{sae_activations.shape[1]} ({sae_dead_features/sae_activations.shape[1]:.2%}) {st_dead_features}/{st_activations.shape[1]} ({st_dead_features/st_activations.shape[1]:.2%})")
    
    return {
        'sae': {
            'activations': sae_activations,
            'sparsity': sae_sparsity,
            'active_per_sample': sae_active_per_sample,
            'dead_features': sae_dead_features
        },
        'st': {
            'activations': st_activations,
            'sparsity': st_sparsity,
            'active_per_sample': st_active_per_sample,
            'dead_features': st_dead_features
        }
    }


def analyze_model_trajectory(model_checkpoints, is_sae=True, metrics=None, figsize=(12, 8)):
    """
    Analyze model training trajectory from multiple checkpoints
    
    Args:
        model_checkpoints: List of checkpoint paths
        is_sae: Whether the model is an SAE (True) or ST (False)
        metrics: Dictionary of metrics in checkpoints to plot 
                (or None to extract from checkpoints)
        figsize: Figure size
    """
    if metrics is None:
        # Extract metrics from checkpoints
        metrics = {'steps': [], 'loss': [], 'val_loss': [], 'dead_ratio': [], 'lambda': []}
        
        for ckpt_path in sorted(model_checkpoints):
            try:
                ckpt = torch.load(ckpt_path)
                
                # Check if checkpoint is a full dictionary or just state_dict
                if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                    # Full checkpoint
                    step = ckpt.get('step', 0)
                    metrics['steps'].append(step)
                    metrics['loss'].append(ckpt.get('train_loss', 0))
                    metrics['val_loss'].append(ckpt.get('val_loss', 0))
                    metrics['dead_ratio'].append(ckpt.get('dead_ratio', 0))
                    metrics['lambda'].append(ckpt.get('lambda_l1', 0))
                else:
                    # Just extract step from filename
                    step = int(ckpt_path.split('step')[-1])
                    metrics['steps'].append(step)
                    metrics['loss'].append(0)  # Placeholder
                    metrics['val_loss'].append(0)  # Placeholder
                    metrics['dead_ratio'].append(0)  # Placeholder
                    metrics['lambda'].append(0)  # Placeholder
            except Exception as e:
                print(f"Error loading checkpoint {ckpt_path}: {e}")
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Loss curve
    ax1 = axes[0, 0]
    ax1.plot(metrics['steps'], metrics['loss'], 'b-', label='Train Loss')
    ax1.plot(metrics['steps'], metrics['val_loss'], 'r-', label='Val Loss')
    ax1.set_title("Loss Curve")
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Dead ratio curve
    ax2 = axes[0, 1]
    ax2.plot(metrics['steps'], [d*100 for d in metrics['dead_ratio']], 'g-')
    ax2.set_title("Dead Features Ratio")
    ax2.set_xlabel("Training Steps")
    ax2.set_ylabel("Dead Features (%)")
    ax2.grid(alpha=0.3)
    
    # 3. Lambda curve
    ax3 = axes[1, 0]
    ax3.plot(metrics['steps'], metrics['lambda'], 'm-')
    ax3.set_title("L1 Regularization Strength (λ)")
    ax3.set_xlabel("Training Steps")
    ax3.set_ylabel("λ Value")
    ax3.grid(alpha=0.3)
    
    # 4. Custom metric if available
    ax4 = axes[1, 1]
    if 'sparsity' in metrics:
        ax4.plot(metrics['steps'], [s*100 for s in metrics['sparsity']], 'c-')
        ax4.set_title("Feature Sparsity")
        ax4.set_xlabel("Training Steps")
        ax4.set_ylabel("Sparsity (%)")
    else:
        # Placeholder: Loss vs Dead ratio scatter
        ax4.scatter(metrics['loss'], [d*100 for d in metrics['dead_ratio']], 
                   alpha=0.7, c=metrics['steps'], cmap='viridis')
        ax4.set_title("Loss vs Dead Feature Ratio")
        ax4.set_xlabel("Loss")
        ax4.set_ylabel("Dead Features (%)")
        cbar = plt.colorbar(ax4.collections[0], ax=ax4)
        cbar.set_label("Training Steps")
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle(f"Model Training Trajectory - {'SAE' if is_sae else 'ST'}", y=1.02)
    plt.show()
    
    return metrics


def visualize_model_ensemble(models_dict, data, threshold=1e-3, figsize=(15, 8)):
    """
    Visualize feature activations across multiple models
    
    Args:
        models_dict: Dictionary of {model_name: (model, is_sae)}
        data: Input data tensor [samples, features]
        threshold: Activation threshold
        figsize: Figure size
    """
    # Get activations for each model
    activations_dict = {}
    
    for model_name, (model, is_sae) in models_dict.items():
        model.eval()
        
        # Convert to tensor
        if isinstance(data, np.ndarray):
            data_tensor = torch.from_numpy(data).float().to(model.device)
        else:
            data_tensor = data.to(model.device)
        
        # Get feature activations
        with torch.no_grad():
            if is_sae:
                _, _, activations = model(data_tensor)
            else:
                _, _, activations, _ = model(data_tensor)
        
        # Convert to numpy
        activations_dict[model_name] = activations.cpu().numpy()
    
    # Create figure with grid of subplots
    n_models = len(models_dict)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot each model's activations
    for i, (model_name, activations) in enumerate(activations_dict.items()):
        if i < len(axes):
            # Calculate sparsity statistics
            sparsity = 1 - (activations > threshold).mean()
            active_per_sample = (activations > threshold).sum(axis=1).mean()
            dead_features = np.sum(np.all(activations <= threshold, axis=0))
            
            # Plot activations heatmap
            im = axes[i].imshow(activations[:50, :100].T, aspect='auto', cmap='viridis')
            axes[i].set_title(f"{model_name}")
            axes[i].set_xlabel("Sample")
            axes[i].set_ylabel("Feature")
            
            # Add stats as text
            stats_text = (
                f"Sparsity: {sparsity:.1%}\n"
                f"Avg active/sample: {active_per_sample:.1f}\n"
                f"Dead: {dead_features}/{activations.shape[1]} ({dead_features/activations.shape[1]:.1%})"
            )
            
            axes[i].text(0.98, 0.02, stats_text, transform=axes[i].transAxes,
                        verticalalignment='bottom', horizontalalignment='right',
                        bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.7},
                        fontsize=8)
    
    # Hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.suptitle("Feature Activation Comparison Across Models", y=1.02)
    plt.show()
    
    # Create summary table
    summary = []
    for model_name, activations in activations_dict.items():
        sparsity = 1 - (activations > threshold).mean()
        active_per_sample = (activations > threshold).sum(axis=1).mean()
        dead_features = np.sum(np.all(activations <= threshold, axis=0))
        dead_ratio = dead_features / activations.shape[1]
        
        summary.append({
            'Model': model_name,
            'Sparsity': f"{sparsity:.1%}",
            'Avg Active/Sample': f"{active_per_sample:.1f}",
            'Dead Features': f"{dead_features}/{activations.shape[1]} ({dead_ratio:.1%})"
        })
    
    summary_df = pd.DataFrame(summary)
    print("Model Comparison Summary:")
    print(summary_df.to_string(index=False))
    
    return activations_dict, summary_df


def visualize_mnist_feature_winners(model, test_data, test_labels, is_sae=True, num_top_features=5, figsize=(12, 10)):
    """
    For each digit class, find and visualize the top-activating features
    
    Args:
        model: Trained model (SAE or ST)
        test_data: Test data [samples, 784]
        test_labels: Test labels [samples]
        is_sae: Whether the model is an SAE (True) or ST (False)
        num_top_features: Number of top features to visualize per class
        figsize: Figure size
    """
    model.eval()
    
    # Convert to torch tensors
    if isinstance(test_data, np.ndarray):
        data_tensor = torch.from_numpy(test_data).float().to(model.device)
    else:
        data_tensor = data_tensor.to(model.device)
    
    # Get feature activations
    with torch.no_grad():
        if is_sae:
            _, _, activations = model(data_tensor)
        else:
            _, _, activations, _ = model(data_tensor)
    
    # Convert to numpy
    activations = activations.cpu().numpy()
    
    # Get feature vectors
    feature_vectors = get_feature_vectors(model, is_sae)
    
    # Find winning features for each digit
    digit_classes = np.unique(test_labels)
    top_features = {}
    
    for digit in digit_classes:
        # Get indices for this digit
        indices = np.where(test_labels == digit)[0]
        
        # Get mean activation for each feature
        mean_activations = np.mean(activations[indices], axis=0)
        
        # Get top features
        top_indices = np.argsort(-mean_activations)[:num_top_features]
        top_features[digit] = top_indices
    
    # Create visualization
    num_digits = len(digit_classes)
    num_cols = num_top_features + 1  # +1 for digit examples
    
    fig, axes = plt.subplots(num_digits, num_cols, figsize=figsize)
    
    # For each digit class
    for i, digit in enumerate(digit_classes):
        # Show an example digit
        indices = np.where(test_labels == digit)[0]
        example_idx = indices[0]
        
        axes[i, 0].imshow(test_data[example_idx].reshape(28, 28), cmap='gray')
        axes[i, 0].set_title(f"Digit {digit}")
        axes[i, 0].axis('off')
        
        # Show top features for this digit
        for j, feature_idx in enumerate(top_features[digit]):
            # Get feature vector
            feature = feature_vectors[:, feature_idx]
            
            # Reshape and display
            feature_img = feature.reshape(28, 28)
            
            # Normalize for better visualization
            vmax = max(abs(feature_img.max()), abs(feature_img.min()))
            
            axes[i, j+1].imshow(feature_img, cmap='coolwarm', vmin=-vmax, vmax=vmax)
            
            # Get mean activation for this feature on this digit
            indices = np.where(test_labels == digit)[0]
            mean_act = np.mean(activations[indices, feature_idx])
            
            axes[i, j+1].set_title(f"F{feature_idx}\nAct: {mean_act:.2f}")
            axes[i, j+1].axis('off')
    
    plt.tight_layout()
    plt.suptitle(f"Top Features by Digit Class - {'SAE' if is_sae else 'ST'} Model", y=1.02)
    plt.show()
    
    # Create feature-digit activation heatmap
    plt.figure(figsize=(10, 8))
    
    # Prepare data
    feature_digit_matrix = np.zeros((num_top_features*num_digits, num_digits))
    feature_names = []
    
    for i, digit in enumerate(digit_classes):
        indices = np.where(test_labels == digit)[0]
        
        for j, feature_idx in enumerate(top_features[digit]):
            row_idx = i * num_top_features + j
            feature_names.append(f"D{digit}F{j+1}")
            
            # Get activation of this feature across all digit classes
            for k, target_digit in enumerate(digit_classes):
                target_indices = np.where(test_labels == target_digit)[0]
                mean_act = np.mean(activations[target_indices, feature_idx])
                feature_digit_matrix[row_idx, k] = mean_act
    
    # Create heatmap
    sns.heatmap(feature_digit_matrix, cmap='viridis', 
               xticklabels=digit_classes, 
               yticklabels=feature_names)
    
    plt.title("Feature Activation Across Digit Classes")
    plt.xlabel("Digit")
    plt.ylabel("Features (DxFy = Top y feature for digit x)")
    plt.tight_layout()
    plt.show()
    
    return top_features