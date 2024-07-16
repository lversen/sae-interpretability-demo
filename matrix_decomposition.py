import numpy as np
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import os

def decompose_matrix(feature_extract: np.ndarray, n_components: int = 3, dataset_name: str = "", model_name: str = ""):
    """
    Decomposes the feature extraction matrix using Truncated SVD and plots the explained variance ratio.
    
    Args:
    feature_extract (np.ndarray): The feature extraction matrix to decompose
    n_components (int): Number of components to keep (default is 3)
    dataset_name (str): Name of the dataset (for saving the plot)
    model_name (str): Name of the model used for feature extraction (for saving the plot)
    
    Returns:
    tuple: (U, s, Vt) where U and Vt are the left and right singular vectors, and s is the singular values
    """
    # Perform Truncated SVD
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(feature_extract)
    
    # Get the singular values and vectors
    U = svd.transform(feature_extract)
    s = svd.singular_values_
    Vt = svd.components_
    
    # Calculate explained variance ratio
    explained_variance_ratio = svd.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    # Plot explained variance ratio
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_components + 1), cumulative_variance_ratio, 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title(f'Explained Variance Ratio ({os.path.basename(dataset_name)}, {model_name})')
    
    # Create directory for saving plots
    plot_dir = os.path.join("plots", os.path.basename(dataset_name).replace('.csv', ''), model_name.replace('/', '_'))
    os.makedirs(plot_dir, exist_ok=True)
    
    # Save the plot
    plot_filename = f"explained_variance_{os.path.basename(dataset_name).replace('.csv', '')}_{model_name.replace('/', '_')}.png"
    plot_path = os.path.join(plot_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Explained variance plot saved to {plot_path}")
    print(f"Explained variance ratios: {explained_variance_ratio}")
    print(f"Total explained variance: {np.sum(explained_variance_ratio):.4f}")
    
    return U, s, Vt