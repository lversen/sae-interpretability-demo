import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Fix OpenMP error by setting environment variable
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Define distinct color schemes for each model type
SAE_CMAP = 'coolwarm'        # Red-Blue for SAE
ST_CMAP = 'viridis'          # Green-Purple for ST
SAE_COLOR = '#E41A1C'        # Red for SAE
ST_COLOR = '#4DAF4A'         # Green for ST

def setup_model_specific_style(model_type):
    """Set up plot style based on model type"""
    if model_type.lower() == 'sae':
        plt.style.use('default')
        plt.rcParams['figure.facecolor'] = '#FFF5F5'  # Light red background for SAE
        return SAE_CMAP, SAE_COLOR
    else:
        plt.style.use('default')
        plt.rcParams['figure.facecolor'] = '#F5FFF5'  # Light green background for ST
        return ST_CMAP, ST_COLOR

def add_model_identifier(ax, model_type):
    """Add model identifier to axis"""
    if model_type.lower() == 'sae':
        ax.text(0.02, 0.98, "SAE MODEL", transform=ax.transAxes, 
                fontsize=10, fontweight='bold', color=SAE_COLOR,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    else:
        ax.text(0.02, 0.98, "ST MODEL", transform=ax.transAxes, 
                fontsize=10, fontweight='bold', color=ST_COLOR,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

def detect_model_dimensions(model_path):
    """
    Detect the dimensions of a saved model safely using CPU
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Dictionary with detected dimensions and the state dict
    """
    try:
        # Load the state dict on CPU
        state_dict = torch.load(model_path, map_location='cpu')
        
        # Check if it's a checkpoint or just a state dict
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        
        # Detect dimensions based on parameter shapes
        dimensions = {}
        
        # For SAE models
        if 'W_e.weight' in state_dict and 'W_d.weight' in state_dict:
            w_e = state_dict['W_e.weight']
            w_d = state_dict['W_d.weight']
            
            dimensions['m'] = w_e.shape[0]  # Feature dimension
            dimensions['n'] = w_e.shape[1]  # Input dimension
            
            print(f"Detected SAE dimensions: n={dimensions['n']}, m={dimensions['m']}")
            dimensions['model_type'] = 'sae'
            
        # For ST models
        elif 'W_q.weight' in state_dict and 'W_k.weight' in state_dict and 'W_v.weight' in state_dict:
            w_q = state_dict['W_q.weight']
            w_k = state_dict['W_k.weight']
            w_v = state_dict['W_v.weight']
            
            dimensions['n'] = w_q.shape[1]  # Input dimension
            dimensions['a'] = w_q.shape[0]  # Attention dimension
            
            # For m (feature dimension), check memory_indices
            if 'memory_indices' in state_dict:
                dimensions['m'] = state_dict['memory_indices'].shape[0]
            else:
                # If memory_indices not found, use a reasonable value
                dimensions['m'] = 100
            
            print(f"Detected ST dimensions: n={dimensions['n']}, m={dimensions['m']}, a={dimensions['a']}")
            dimensions['model_type'] = 'st'
        
        return dimensions, state_dict
        
    except Exception as e:
        print(f"Error detecting dimensions: {e}")
        return None, None

def visualize_decoder_weights(state_dict, model_type='sae', input_shape=(28, 28), top_n=16, figsize=(15, 8)):
    """
    Visualize decoder weight vectors directly from state dict (for image data)
    """
    # Set up model-specific style
    cmap, color = setup_model_specific_style(model_type)
    
    # Get feature vectors based on model type
    if model_type == 'sae':
        feature_vectors = state_dict['W_d.weight'].numpy()
        title_prefix = "SAE Model: Decoder Weight Vectors"
    else:  # ST model
        feature_vectors = state_dict['W_v.weight'].numpy()
        title_prefix = "ST Model: Value Vectors"
    
    # Calculate L2 norms to identify important features
    feature_norms = np.linalg.norm(feature_vectors, axis=0)
    top_indices = np.argsort(-feature_norms)[:top_n]
    
    # Determine layout
    n_cols = min(4, top_n)
    n_rows = int(np.ceil(top_n / n_cols))
    
    # Create plot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    # Add a prominent title bar at the top with model type
    fig.text(0.5, 0.98, title_prefix, ha='center', va='top', 
             fontsize=16, fontweight='bold', color=color,
             bbox=dict(facecolor='white', alpha=0.8, boxstyle="round,pad=0.5"))
             
    axes = axes.flatten()
    
    # Plot each feature
    for i, idx in enumerate(top_indices):
        if i < len(axes):
            feature = feature_vectors[:, idx]
            
            # Reshape for images
            if input_shape is not None:
                feature = feature.reshape(input_shape)
            
            # Plot with proper normalization
            vmax = max(abs(feature.max()), abs(feature.min()))
            im = axes[i].imshow(feature, cmap=cmap, vmin=-vmax, vmax=vmax)
            axes[i].set_title(f"Feature {idx}\nNorm: {feature_norms[idx]:.3f}")
            axes[i].axis('off')
            
            # Add model identifier to first subplot only
            if i == 0:
                add_model_identifier(axes[i], model_type)
    
    # Add colorbar
    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for title
    plt.show()
    
    return top_indices

def visualize_mnist_features_simple(state_dict, model_type='sae', input_shape=(28, 28), figsize=(15, 10)):
    """
    Simple visualization of features learned by the model, safe for CPU-only usage
    """
    # Set up model-specific style
    cmap, color = setup_model_specific_style(model_type)
    
    # Get feature vectors based on model type
    if model_type == 'sae':
        if 'W_d.weight' in state_dict:
            feature_vectors = state_dict['W_d.weight'].numpy()
            title_prefix = "SAE Model"
        else:
            raise ValueError("Could not find decoder weights in state_dict")
    else:  # ST model
        if 'W_v.weight' in state_dict:
            feature_vectors = state_dict['W_v.weight'].numpy()
            title_prefix = "ST Model"
        else:
            raise ValueError("Could not find value weights in state_dict")
    
    # Calculate statistics
    feature_norms = np.linalg.norm(feature_vectors, axis=0)
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    
    # Add a prominent title
    fig.text(0.5, 0.98, f"{title_prefix}: Feature Statistics and Correlations", 
             ha='center', va='top', fontsize=16, fontweight='bold', color=color,
             bbox=dict(facecolor='white', alpha=0.8, boxstyle="round,pad=0.5"))
    
    # 1. Plot histogram of feature norms
    ax1 = plt.subplot(2, 2, 1)
    plt.hist(feature_norms, bins=30, color=color, alpha=0.7)
    plt.title(f"Feature Norms Distribution")
    plt.xlabel("L2 Norm")
    plt.ylabel("Count")
    plt.grid(alpha=0.3)
    add_model_identifier(ax1, model_type)
    
    # 2. Plot top features
    n_features = min(16, feature_vectors.shape[1])
    top_indices = np.argsort(-feature_norms)[:n_features]
    
    # Create mini-grid for feature visualization
    fig_grid = plt.figure(figsize=(12, 10))
    
    # Add a prominent title
    fig_grid.text(0.5, 0.98, f"{title_prefix}: Top {n_features} Features by Norm", 
                 ha='center', va='top', fontsize=16, fontweight='bold', color=color,
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle="round,pad=0.5"))
    
    n_cols = 4
    n_rows = int(np.ceil(n_features / n_cols))
    
    for i, idx in enumerate(top_indices[:n_features]):
        ax = plt.subplot(n_rows, n_cols, i+1)
        feature = feature_vectors[:, idx]
        
        # Reshape for images
        if input_shape is not None:
            feature = feature.reshape(input_shape)
        
        # Plot with proper normalization
        vmax = max(abs(feature.max()), abs(feature.min()))
        plt.imshow(feature, cmap=cmap, vmin=-vmax, vmax=vmax)
        plt.title(f"Feature {idx}")
        plt.axis('off')
        
        # Add model identifier to first subplot only
        if i == 0:
            add_model_identifier(ax, model_type)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for title
    plt.show()
    
    # 3. Plot feature correlation matrix
    fig_corr = plt.figure(figsize=(10, 8))
    
    # Add a prominent title
    fig_corr.text(0.5, 0.98, f"{title_prefix}: Feature Correlation Matrix", 
                 ha='center', va='top', fontsize=16, fontweight='bold', color=color,
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle="round,pad=0.5"))
    
    ax3 = plt.gca()
    
    n_for_corr = min(50, feature_vectors.shape[1])
    top_feat_indices = np.argsort(-feature_norms)[:n_for_corr]
    top_features = feature_vectors[:, top_feat_indices]
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(top_features.T)
    
    # Use diverging colormap for correlations
    diverg_cmap = 'coolwarm' if model_type == 'sae' else 'PiYG'
    
    # Plot correlation matrix
    sns.heatmap(corr_matrix, cmap=diverg_cmap, center=0, 
                xticklabels=False, yticklabels=False, ax=ax3)
    plt.title(f"Top {n_for_corr} Features", fontsize=14)
    add_model_identifier(ax3, model_type)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for title
    plt.show()
    
    # Print statistics
    print(f"\n{title_prefix} Feature Statistics:")
    print(f"Total features: {feature_vectors.shape[1]}")
    print(f"Average norm: {np.mean(feature_norms):.4f}")
    print(f"Min norm: {np.min(feature_norms):.4f}, Max norm: {np.max(feature_norms):.4f}")
    
    return feature_vectors, feature_norms

def analyze_mnist_digits(state_dict, test_data, test_labels, model_type='sae', input_shape=(28, 28)):
    """
    Analyze which features activate for different digits without running the model
    """
    # Set up model-specific style
    cmap, color = setup_model_specific_style(model_type)
    
    # Get features from state dict
    if model_type == 'sae':
        weight_vectors = state_dict['W_d.weight'].numpy()
        title_prefix = "SAE Model"
        # For SAE, the decoder matrix gives us a good idea of what patterns are learned
        feature_vectors = weight_vectors
        
    else:  # ST model
        # For ST, use the value vectors as the learned features
        feature_vectors = state_dict['W_v.weight'].numpy()
        title_prefix = "ST Model"
    
    # Calculate feature norms
    feature_norms = np.linalg.norm(feature_vectors, axis=0)
    
    # Visualize features for the most important columns
    n_features = min(25, feature_vectors.shape[1])
    top_indices = np.argsort(-feature_norms)[:n_features]
    
    # Create grid layout
    n_rows = 5
    n_cols = 5
    fig = plt.figure(figsize=(12, 12))
    
    # Add a prominent title
    fig.text(0.5, 0.98, f"{title_prefix}: Top {n_features} Features by Norm", 
            ha='center', va='top', fontsize=16, fontweight='bold', color=color,
            bbox=dict(facecolor='white', alpha=0.8, boxstyle="round,pad=0.5"))
    
    for i, idx in enumerate(top_indices):
        if i < n_rows * n_cols:
            ax = plt.subplot(n_rows, n_cols, i+1)
            feature = feature_vectors[:, idx]
            
            # Reshape for visualization
            feature_img = feature.reshape(input_shape)
            
            # Plot with proper normalization
            vmax = max(abs(feature_img.max()), abs(feature_img.min()))
            plt.imshow(feature_img, cmap=cmap, vmin=-vmax, vmax=vmax)
            plt.title(f"Feature {idx}")
            plt.axis('off')
            
            # Add model identifier to first subplot only
            if i == 0:
                add_model_identifier(ax, model_type)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for title
    plt.show()
    
    # Analyze feature similarity to different digits
    # This is a simplified approach that doesn't require running the model
    digit_similarities = {}
    
    # Create example digits figure
    fig_digits = plt.figure(figsize=(15, 3))
    
    # Add a prominent title
    fig_digits.text(0.5, 0.98, f"{title_prefix}: Average Digit Representations", 
                  ha='center', va='top', fontsize=16, fontweight='bold', color=color,
                  bbox=dict(facecolor='white', alpha=0.8, boxstyle="round,pad=0.5"))
    
    # Show example digits
    for digit in range(10):
        ax = plt.subplot(1, 10, digit+1)
        # Get examples of this digit
        digit_indices = np.where(test_labels == digit)[0]
        digit_samples = test_data[digit_indices[:50]]  # Use up to 50 examples
        
        # Compute average digit prototype
        avg_digit = np.mean(digit_samples, axis=0)
        
        plt.imshow(avg_digit.reshape(input_shape), cmap='gray')
        plt.title(f"Digit {digit}")
        plt.axis('off')
        
        # Add model identifier to first subplot only
        if digit == 0:
            add_model_identifier(ax, model_type)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for title
    plt.show()
    
    # Create digit feature analysis
    for digit in range(10):
        # Get examples of this digit
        digit_indices = np.where(test_labels == digit)[0]
        digit_samples = test_data[digit_indices[:50]]  # Use up to 50 examples
        
        # Compute average digit prototype
        avg_digit = np.mean(digit_samples, axis=0)
        
        # Compute similarity between the average digit and each feature
        similarities = []
        
        for i in range(feature_vectors.shape[1]):
            feature = feature_vectors[:, i]
            
            # Use cosine similarity (dot product of normalized vectors)
            feature_norm = np.linalg.norm(feature)
            digit_norm = np.linalg.norm(avg_digit)
            
            if feature_norm > 0 and digit_norm > 0:
                similarity = np.abs(np.dot(feature, avg_digit) / (feature_norm * digit_norm))
            else:
                similarity = 0
                
            similarities.append(similarity)
        
        # Find top matching features
        top_matches = np.argsort(-np.array(similarities))[:5]
        digit_similarities[digit] = {
            'top_matches': top_matches,
            'similarities': [similarities[i] for i in top_matches]
        }
        
        # Visualize top matches
        fig_matches = plt.figure(figsize=(15, 3))
        
        # Add a prominent title
        fig_matches.text(0.5, 0.98, f"{title_prefix}: Top Features for Digit {digit}", 
                       ha='center', va='top', fontsize=16, fontweight='bold', color=color,
                       bbox=dict(facecolor='white', alpha=0.8, boxstyle="round,pad=0.5"))
        
        for i, feature_idx in enumerate(top_matches):
            ax = plt.subplot(1, 5, i+1)
            feature = feature_vectors[:, feature_idx]
            
            # Reshape for visualization
            feature_img = feature.reshape(input_shape)
            
            # Plot with proper normalization
            vmax = max(abs(feature_img.max()), abs(feature_img.min()))
            plt.imshow(feature_img, cmap=cmap, vmin=-vmax, vmax=vmax)
            plt.title(f"Feature {feature_idx}\nSim: {similarities[feature_idx]:.3f}")
            plt.axis('off')
            
            # Add model identifier to first subplot only
            if i == 0:
                add_model_identifier(ax, model_type)
            
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for title
        plt.show()
    
    # Create summary heatmap
    similarity_matrix = np.zeros((10, feature_vectors.shape[1]))
    
    for digit in range(10):
        # Get examples of this digit
        digit_indices = np.where(test_labels == digit)[0]
        digit_samples = test_data[digit_indices[:50]]  # Use up to 50 examples
        
        # Compute average digit prototype
        avg_digit = np.mean(digit_samples, axis=0)
        
        # Compute similarity between the average digit and each feature
        for i in range(feature_vectors.shape[1]):
            feature = feature_vectors[:, i]
            
            # Use cosine similarity
            feature_norm = np.linalg.norm(feature)
            digit_norm = np.linalg.norm(avg_digit)
            
            if feature_norm > 0 and digit_norm > 0:
                similarity = np.abs(np.dot(feature, avg_digit) / (feature_norm * digit_norm))
            else:
                similarity = 0
                
            similarity_matrix[digit, i] = similarity
    
    # Create heatmap of top features
    fig_heatmap = plt.figure(figsize=(12, 8))
    
    # Add a prominent title
    fig_heatmap.text(0.5, 0.98, f"{title_prefix}: Feature-Digit Similarity Matrix", 
                   ha='center', va='top', fontsize=16, fontweight='bold', color=color,
                   bbox=dict(facecolor='white', alpha=0.8, boxstyle="round,pad=0.5"))
    
    ax_heat = plt.gca()
    
    # Select top features across all digits
    feature_importance = np.max(similarity_matrix, axis=0)
    top_feature_indices = np.argsort(-feature_importance)[:50]
    
    # Create heatmap for top features
    sns.heatmap(similarity_matrix[:, top_feature_indices], 
               cmap=cmap, 
               xticklabels=False,
               yticklabels=range(10),
               ax=ax_heat)
    plt.title(f"Feature Similarity to Digits", fontsize=14)
    plt.xlabel("Top 50 Features")
    plt.ylabel("Digit")
    add_model_identifier(ax_heat, model_type)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for title
    plt.show()
    
    return digit_similarities

def model_comparison_visualization(model_data, test_data, test_labels):
    """
    Create a comprehensive comparison between SAE and ST models
    """
    if len(model_data) < 2:
        print("Need at least 2 models for comparison")
        return
    
    # Reset plot style
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = 'white'
    
    # Create model comparison figure
    fig = plt.figure(figsize=(15, 10))
    
    # Add a prominent title
    fig.text(0.5, 0.98, "Model Comparison: SAE vs ST", 
            ha='center', va='top', fontsize=18, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.9, boxstyle="round,pad=0.5",
                     edgecolor='black', linewidth=2))
    
    # Get feature norms for each model
    feature_norms = {}
    feature_counts = {}
    feature_matrices = {}
    
    for model_name, data in model_data.items():
        model_type = data['dimensions']['model_type']
        
        if model_type == 'sae':
            features = data['state_dict']['W_d.weight'].numpy()
        else:  # ST
            features = data['state_dict']['W_v.weight'].numpy()
        
        feature_matrices[model_name] = features
        norms = np.linalg.norm(features, axis=0)
        feature_norms[model_name] = norms
        feature_counts[model_name] = len(norms)
    
    # 1. Norm distribution comparison
    ax1 = plt.subplot(2, 2, 1)
    for model_name, norms in feature_norms.items():
        if model_name == 'SAE':
            sns.kdeplot(norms, label=f"{model_name} (m={feature_counts[model_name]})", 
                      color=SAE_COLOR)
        else:
            sns.kdeplot(norms, label=f"{model_name} (m={feature_counts[model_name]})", 
                      color=ST_COLOR)
    
    plt.title("Feature Norm Distribution", fontsize=14)
    plt.xlabel("L2 Norm")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 2. Feature statistics comparison
    ax2 = plt.subplot(2, 2, 2)
    
    # Calculate statistics
    model_names = []
    mean_norms = []
    min_norms = []
    max_norms = []
    model_colors = []
    
    for model_name, norms in feature_norms.items():
        model_names.append(model_name)
        mean_norms.append(np.mean(norms))
        min_norms.append(np.min(norms))
        max_norms.append(np.max(norms))
        model_colors.append(SAE_COLOR if model_name == 'SAE' else ST_COLOR)
    
    # Bar positions
    x = np.arange(len(model_names))
    width = 0.25
    
    # Plot bars
    plt.bar(x - width, mean_norms, width, label='Mean')
    plt.bar(x, min_norms, width, label='Min')
    plt.bar(x + width, max_norms, width, label='Max')
    
    # Add model coloring
    for i, rect in enumerate(ax2.patches[:len(model_names)]):
        rect.set_facecolor(model_colors[i % len(model_colors)])
    
    plt.title("Feature Norm Statistics", fontsize=14)
    plt.xlabel("Model")
    plt.ylabel("Norm Value")
    plt.xticks(x, model_names)
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 3. Feature visualizations side by side
    ax3 = plt.subplot(2, 1, 2)
    
    # Find the top features for each model
    top_features = {}
    for model_name, features in feature_matrices.items():
        norms = feature_norms[model_name]
        top_indices = np.argsort(-norms)[:5]  # Get top 5 features
        top_features[model_name] = [features[:, idx].reshape(28, 28) for idx in top_indices]
    
    # Create side-by-side comparison
    gs = plt.GridSpec(2, 5, wspace=0.2, hspace=0.5)
    
    # SAE top features
    for i, feature in enumerate(top_features['SAE']):
        ax_sae = fig.add_subplot(gs[0, i])
        vmax = max(abs(feature.max()), abs(feature.min()))
        ax_sae.imshow(feature, cmap=SAE_CMAP, vmin=-vmax, vmax=vmax)
        ax_sae.set_title(f"SAE #{i+1}")
        ax_sae.axis('off')
    
    # ST top features
    for i, feature in enumerate(top_features['ST']):
        ax_st = fig.add_subplot(gs[1, i])
        vmax = max(abs(feature.max()), abs(feature.min()))
        ax_st.imshow(feature, cmap=ST_CMAP, vmin=-vmax, vmax=vmax)
        ax_st.set_title(f"ST #{i+1}")
        ax_st.axis('off')
    
    # Hide the original ax3
    ax3.axis('off')
    
    # Add a title to the comparison
    fig.text(0.5, 0.48, "Top 5 Features by Norm", ha='center', fontsize=14)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for title
    plt.show()
    
    # Create digit similarity comparison
    # For each digit, show the top feature from each model
    fig_digit_comp = plt.figure(figsize=(15, 8))
    
    # Add a prominent title
    fig_digit_comp.text(0.5, 0.98, "Model Comparison: Top Feature per Digit", 
                      ha='center', va='top', fontsize=18, fontweight='bold',
                      bbox=dict(facecolor='white', alpha=0.9, boxstyle="round,pad=0.5",
                               edgecolor='black', linewidth=2))
    
    # Compute digit similarity for each model
    model_digit_similarities = {}
    
    for model_name, data in model_data.items():
        model_type = data['dimensions']['model_type']
        
        if model_type == 'sae':
            features = data['state_dict']['W_d.weight'].numpy()
        else:  # ST
            features = data['state_dict']['W_v.weight'].numpy()
        
        # Calculate similarity for each digit
        digit_similarities = {}
        
        for digit in range(10):
            # Get examples of this digit
            digit_indices = np.where(test_labels == digit)[0]
            digit_samples = test_data[digit_indices[:50]]  # Use up to 50 examples
            
            # Compute average digit prototype
            avg_digit = np.mean(digit_samples, axis=0)
            
            # Compute similarity between the average digit and each feature
            similarities = []
            
            for i in range(features.shape[1]):
                feature = features[:, i]
                
                # Use cosine similarity
                feature_norm = np.linalg.norm(feature)
                digit_norm = np.linalg.norm(avg_digit)
                
                if feature_norm > 0 and digit_norm > 0:
                    similarity = np.abs(np.dot(feature, avg_digit) / (feature_norm * digit_norm))
                else:
                    similarity = 0
                    
                similarities.append(similarity)
            
            # Find top matching feature
            top_match = np.argmax(similarities)
            digit_similarities[digit] = {
                'top_match': top_match,
                'similarity': similarities[top_match],
                'feature': features[:, top_match].reshape(28, 28)
            }
        
        model_digit_similarities[model_name] = digit_similarities
    
    # Create side-by-side comparisons for each digit
    gs = plt.GridSpec(3, 10, wspace=0.2, hspace=0.5)
    
    # Show digits
    for digit in range(10):
        # Get examples of this digit
        digit_indices = np.where(test_labels == digit)[0]
        digit_samples = test_data[digit_indices[:50]]  # Use up to 50 examples
        
        # Compute average digit prototype
        avg_digit = np.mean(digit_samples, axis=0)
        
        # Plot digit
        ax_digit = fig.add_subplot(gs[0, digit])
        ax_digit.imshow(avg_digit.reshape(28, 28), cmap='gray')
        ax_digit.set_title(f"Digit {digit}")
        ax_digit.axis('off')
        
        # Plot SAE top feature
        ax_sae = fig.add_subplot(gs[1, digit])
        feature_sae = model_digit_similarities['SAE'][digit]['feature']
        vmax_sae = max(abs(feature_sae.max()), abs(feature_sae.min()))
        ax_sae.imshow(feature_sae, cmap=SAE_CMAP, vmin=-vmax_sae, vmax=vmax_sae)
        sim_sae = model_digit_similarities['SAE'][digit]['similarity']
        feature_idx_sae = model_digit_similarities['SAE'][digit]['top_match']
        ax_sae.set_title(f"SAE #{feature_idx_sae}\nSim:{sim_sae:.2f}")
        ax_sae.axis('off')
        
        # Plot ST top feature
        ax_st = fig.add_subplot(gs[2, digit])
        feature_st = model_digit_similarities['ST'][digit]['feature']
        vmax_st = max(abs(feature_st.max()), abs(feature_st.min()))
        ax_st.imshow(feature_st, cmap=ST_CMAP, vmin=-vmax_st, vmax=vmax_st)
        sim_st = model_digit_similarities['ST'][digit]['similarity']
        feature_idx_st = model_digit_similarities['ST'][digit]['top_match']
        ax_st.set_title(f"ST #{feature_idx_st}\nSim:{sim_st:.2f}")
        ax_st.axis('off')
    
    # Add row labels
    fig_digit_comp.text(0.02, 0.83, "DIGIT", ha='left', va='center', fontsize=14, fontweight='bold')
    fig_digit_comp.text(0.02, 0.5, "SAE", ha='left', va='center', fontsize=14, 
                      fontweight='bold', color=SAE_COLOR)
    fig_digit_comp.text(0.02, 0.17, "ST", ha='left', va='center', fontsize=14, 
                      fontweight='bold', color=ST_COLOR)
    
    plt.tight_layout(rect=[0.03, 0, 1, 0.96])  # Make room for title and row labels
    plt.show()

def safe_mnist_visualization():
    """
    CPU-safe visualization of MNIST models with fixed OpenMP issues
    """
    print("Loading MNIST data...")
    try:
        # Load MNIST data
        train_df = pd.read_csv('data/mnist_train.csv')
        test_df = pd.read_csv('data/mnist_test.csv')
        
        # Extract features and labels
        X_train = train_df.iloc[:, 1:].values.astype(np.float32) / 255.0  # Normalize
        y_train = train_df.iloc[:, 0].values
        
        X_test = test_df.iloc[:, 1:].values.astype(np.float32) / 255.0    # Normalize
        y_test = test_df.iloc[:, 0].values
        
        print("Data loaded successfully!")
    except Exception as e:
        print(f"Error loading MNIST data: {e}")
        return
    
    # Define model paths
    sae_model_path = 'models/sae_model_mnist_train.csv_mnist.pth'
    st_model_path = 'models/st_model_mnist_train.csv_mnist.pth'
    
    # Check which models exist
    models_available = []
    
    if os.path.exists(sae_model_path):
        print(f"Found SAE model at {sae_model_path}")
        models_available.append(('SAE', sae_model_path))
    
    if os.path.exists(st_model_path):
        print(f"Found ST model at {st_model_path}")
        models_available.append(('ST', st_model_path))
    
    if not models_available:
        print("No trained models found. Please check model paths.")
        return
    
    # Extract dimensions and state dicts
    model_data = {}
    
    for model_name, model_path in models_available:
        print(f"\nAnalyzing {model_name} model...")
        dimensions, state_dict = detect_model_dimensions(model_path)
        
        if dimensions is not None:
            model_data[model_name] = {
                'dimensions': dimensions,
                'state_dict': state_dict,
                'path': model_path
            }
            print(f"{model_name} model loaded successfully!")
        else:
            print(f"Could not analyze {model_name} model.")
    
    if not model_data:
        print("Could not load any models for visualization. Exiting.")
        return
    
    # Menu for visualizations
    visualizations = {
        1: "Feature Weight Visualization",
        2: "Feature Statistics and Correlations",
        3: "Digit-Specific Feature Analysis",
        4: "Model Comparison (if both models available)",
        5: "Exit"
    }
    
    while True:
        print("\nMNIST Model Visualization Options:")
        for key, value in visualizations.items():
            print(f"{key}: {value}")
        
        choice = input("\nSelect a visualization (1-5): ")
        
        try:
            choice = int(choice)
            
            if choice == 1:
                # Feature weight visualization
                for model_name, data in model_data.items():
                    print(f"\nVisualizing {model_name} feature weights...")
                    model_type = data['dimensions']['model_type']
                    visualize_decoder_weights(
                        data['state_dict'], 
                        model_type=model_type,
                        input_shape=(28, 28)
                    )
            
            elif choice == 2:
                # Feature statistics and correlations
                for model_name, data in model_data.items():
                    print(f"\nAnalyzing {model_name} feature statistics...")
                    model_type = data['dimensions']['model_type']
                    visualize_mnist_features_simple(
                        data['state_dict'],
                        model_type=model_type,
                        input_shape=(28, 28)
                    )
            
            elif choice == 3:
                # Digit-specific feature analysis
                for model_name, data in model_data.items():
                    print(f"\nAnalyzing {model_name} digit-specific features...")
                    model_type = data['dimensions']['model_type']
                    analyze_mnist_digits(
                        data['state_dict'],
                        X_test,
                        y_test,
                        model_type=model_type
                    )
            
            elif choice == 4:
                # Model comparison
                if len(model_data) >= 2:
                    print("\nComparing models...")
                    model_comparison_visualization(model_data, X_test, y_test)
                else:
                    print("Need at least 2 models for comparison.")
            
            elif choice == 5:
                print("Exiting visualization tool.")
                break
            
            else:
                print("Invalid choice. Please select a number between 1 and 5.")
                
        except ValueError:
            print("Invalid input. Please enter a number.")
        
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    safe_mnist_visualization()