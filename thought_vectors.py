#!/usr/bin/env python3
"""
Thought Vectors Visualization for Image Datasets

This script provides interactive visualization of the latent space (thought vectors)
of trained SAE and ST models on image datasets like MNIST.

Inspired by: https://gabgoh.github.io/ThoughtVectors/

Example usage:
    # Visualize SAE model latent space for MNIST
    python thought_vectors.py --model_path models/sae/relu/100/bs4096_lr5p0e-5_steps200000.pth --model_type sae --dataset mnist
    
    # Visualize ST model with custom settings
    python thought_vectors.py --model_path models/st/softmax/100/bs4096_lr5p0e-5_steps200000.pth --model_type st --dataset mnist --n_components 3
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Optional imports - we'll handle these with try/except
try:
    from SAE import SparseAutoencoder
    SAE_AVAILABLE = True
except ImportError:
    SAE_AVAILABLE = False
    print("Warning: Original SAE module not found. Will use simplified version.")

try:
    from ST_old import SparseTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    print("Warning: Original ST module not found. Will use simplified version.")


class SimplifiedSAE:
    """Simplified SAE implementation when the original is not available"""
    
    def __init__(self, n, m, device='cuda', lambda_l1=1.0, sae_model_path=None):
        self.n = n  # Input dimension
        self.m = m  # Feature dimension
        self.device = device
        self.lambda_l1 = lambda_l1
        self.sae_model_path = sae_model_path
        
        # Initialize weights
        self.W_e = torch.nn.Linear(n, m, bias=True).to(device)
        self.W_d = torch.nn.Linear(m, n, bias=True).to(device)
        
    def forward(self, x):
        # Encoder - ReLU activation
        h = torch.relu(self.W_e(x))
        # Decoder
        x_hat = self.W_d(h)
        return x_hat, h
    
    def load_state_dict(self, state_dict):
        """Load state dictionary with key name compatibility"""
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('W_e.'):
                new_state_dict[key] = value
            elif key.startswith('W_d.'):
                new_state_dict[key] = value
            else:
                new_state_dict[key] = value
        
        super_dict = {k: v for k, v in new_state_dict.items() 
                     if k in [key for key, _ in self.named_parameters()]}
        
        torch.nn.Module.load_state_dict(self, super_dict)
        
    def feature_activations(self, x):
        """Get feature activations for input x"""
        with torch.no_grad():
            h = torch.relu(self.W_e(x))
        return h
        
    def reconstruct(self, f):
        """Reconstruct input from feature activations"""
        with torch.no_grad():
            x_hat = self.W_d(f)
        return x_hat


def load_mnist_dataset(path="data/mnist_train.csv", n_samples=None, input_shape=(28, 28)):
    """
    Load MNIST dataset from CSV file
    
    Args:
        path: Path to the CSV file
        n_samples: Number of samples to load (None for all)
        input_shape: Shape of each image
        
    Returns:
        Tuple of (images_tensor, labels_tensor, label_names)
    """
    try:
        # Try to load with pandas
        df = pd.read_csv(path)
        
        # Identify the label column (usually first column or named 'label')
        label_col = 'label' if 'label' in df.columns else df.columns[0]
        
        # Get labels
        labels = df[label_col].values
        
        # Get features - all columns except the label column
        feature_cols = [col for col in df.columns if col != label_col]
        features = df[feature_cols].values
        
        # Limit samples if specified
        if n_samples is not None and n_samples < len(features):
            indices = np.random.choice(len(features), n_samples, replace=False)
            features = features[indices]
            labels = labels[indices]
        
        # Normalize features to [0, 1]
        features = features / 255.0
        
        # Convert to tensors
        features_tensor = torch.tensor(features, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        # For MNIST, label names are 0-9
        label_names = [str(i) for i in range(10)]
        
        print(f"Loaded {len(features)} samples with shape {features.shape}")
        return features_tensor, labels_tensor, label_names
    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Generating random data as fallback")
        
        # Generate random data as fallback
        n_samples = n_samples or 1000
        n_pixels = input_shape[0] * input_shape[1]
        features = np.random.rand(n_samples, n_pixels)
        labels = np.random.randint(0, 10, n_samples)
        
        features_tensor = torch.tensor(features, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        label_names = [str(i) for i in range(10)]
        
        return features_tensor, labels_tensor, label_names


def load_image_dataset(path, n_samples=None, input_shape=None):
    """
    Load any image dataset from CSV file
    
    Args:
        path: Path to the CSV file
        n_samples: Number of samples to load (None for all)
        input_shape: Shape of each image (if None, will be inferred)
        
    Returns:
        Tuple of (images_tensor, labels_tensor, label_names)
    """
    try:
        # Try to load with pandas
        df = pd.read_csv(path)
        
        # Identify the label column (usually first column or named 'label')
        label_candidates = ['label', 'class', 'category', 'target']
        label_col = None
        for candidate in label_candidates:
            if candidate in df.columns:
                label_col = candidate
                break
        
        if label_col is None:
            # Assume first column is the label
            label_col = df.columns[0]
        
        # Get labels
        labels = df[label_col].values
        
        # Get unique label names
        unique_labels = np.unique(labels)
        label_names = [str(label) for label in unique_labels]
        
        # Map labels to integers
        label_map = {label: i for i, label in enumerate(unique_labels)}
        labels_int = np.array([label_map[label] for label in labels])
        
        # Get features - all columns except the label column
        feature_cols = [col for col in df.columns if col != label_col]
        features = df[feature_cols].values
        
        # Limit samples if specified
        if n_samples is not None and n_samples < len(features):
            indices = np.random.choice(len(features), n_samples, replace=False)
            features = features[indices]
            labels_int = labels_int[indices]
        
        # Normalize features to [0, 1]
        if features.max() > 1.0:
            features = features / 255.0
        
        # Infer input shape if not provided
        if input_shape is None:
            # Try to infer from number of feature columns
            n_features = features.shape[1]
            # Check if it's a perfect square
            sqrt_features = int(np.sqrt(n_features))
            if sqrt_features ** 2 == n_features:
                input_shape = (sqrt_features, sqrt_features)
            else:
                # Use a default shape
                input_shape = (28, 28)  # Default to MNIST-like shape
        
        # Convert to tensors
        features_tensor = torch.tensor(features, dtype=torch.float32)
        labels_tensor = torch.tensor(labels_int, dtype=torch.long)
        
        print(f"Loaded {len(features)} samples with shape {features.shape}")
        print(f"Detected label column: {label_col}")
        print(f"Found {len(unique_labels)} unique labels: {unique_labels[:10]}...")
        
        return features_tensor, labels_tensor, label_names
    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Generating random data as fallback")
        
        # Generate random data as fallback
        n_samples = n_samples or 1000
        n_pixels = input_shape[0] * input_shape[1] if input_shape else 784
        features = np.random.rand(n_samples, n_pixels)
        labels = np.random.randint(0, 10, n_samples)
        
        features_tensor = torch.tensor(features, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        label_names = [str(i) for i in range(10)]
        
        return features_tensor, labels_tensor, label_names


def load_model(model_path, model_type='sae', device='cuda'):
    """
    Load a trained model from path
    
    Args:
        model_path: Path to the model file
        model_type: Type of model ('sae' or 'st')
        device: Device to load the model on ('cuda' or 'cpu')
        
    Returns:
        Loaded model
    """
    try:
        print(f"Loading {model_type.upper()} model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract state dict if needed
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Determine dimensions from the model
        if model_type.lower() == 'sae':
            # For SAE, get dimensions from W_e.weight or W_d.weight
            if 'W_e.weight' in state_dict:
                m, n = state_dict['W_e.weight'].shape
                print(f"Detected dimensions: n={n}, m={m}")
            elif 'W_d.weight' in state_dict:
                n, m = state_dict['W_d.weight'].shape
                print(f"Detected dimensions: n={n}, m={m}")
            else:
                raise ValueError("Could not determine dimensions from state_dict")
            
            # Create SAE model
            if SAE_AVAILABLE:
                model = SparseAutoencoder(n=n, m=m, sae_model_path=model_path, device=device)
            else:
                model = SimplifiedSAE(n=n, m=m, device=device)
                
            model.load_state_dict(state_dict)
            
        elif model_type.lower() == 'st':
            # For ST, get dimensions from weights
            if 'W_k_direct' in state_dict and 'W_v_direct' in state_dict:
                m, a = state_dict['W_k_direct'].shape
                m, n = state_dict['W_v_direct'].shape
                print(f"Detected dimensions (direct K-V): n={n}, m={m}, a={a}")
            elif 'W_q.weight' in state_dict:
                a, n = state_dict['W_q.weight'].shape
                print(f"Detected dimensions (partial): n={n}, a={a}")
                
                # Try to determine m from other parameters
                if 'memory_indices' in state_dict:
                    m = len(state_dict['memory_indices'])
                    print(f"Detected memory size (m): {m}")
                else:
                    # Default to a reasonable m
                    m = n * 4
                    print(f"Using default memory size (m): {m}")
            else:
                raise ValueError("Could not determine dimensions from state_dict")
            
            # Create ST model
            if ST_AVAILABLE:
                # Need dummy data for ST initialization
                dummy_data = torch.zeros((10, n), device=device)
                model = SparseTransformer(
                    X=dummy_data,
                    n=n, m=m, a=a,
                    st_model_path=model_path,  # Add this line
                    device=device
                )
                model.load_state_dict(state_dict)
            else:
                raise ValueError("ST module not available and no simplified version implemented")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return model
    
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None


def compute_latent_representations(model, data, model_type='sae', batch_size=64, device='cuda'):
    """
    Compute latent representations (thought vectors) for the data
    
    Args:
        model: Trained model
        data: Input data tensor
        model_type: Type of model ('sae' or 'st')
        batch_size: Batch size for processing
        device: Device to use for computation
        
    Returns:
        Latent representations
    """
    model.eval()
    latent_reps = []
    
    # Create DataLoader
    dataset = TensorDataset(data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for batch in data_loader:
            x = batch[0].to(device)
            
            if model_type.lower() == 'sae':
                # For SAE, get feature activations
                latent = model.feature_activations(x)
            elif model_type.lower() == 'st':
                # For ST, forward pass and get feature activations
                _, _, f, _ = model(x)
                latent = f
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            latent_reps.append(latent.cpu().numpy())
    
    # Concatenate all batches
    latent_reps = np.concatenate(latent_reps, axis=0)
    
    print(f"Computed latent representations with shape: {latent_reps.shape}")
    return latent_reps


def reduce_dimensions(latent_reps, method='tsne', n_components=2, random_state=42):
    """
    Reduce dimensions of latent representations for visualization
    
    Args:
        latent_reps: Latent representations
        method: Dimensionality reduction method ('tsne', 'pca', or 'umap')
        n_components: Number of components to reduce to
        random_state: Random state for reproducibility
        
    Returns:
        Reduced representations
    """
    print(f"Reducing dimensions with {method} to {n_components} components...")
    
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=random_state)
        reduced_reps = reducer.fit_transform(latent_reps)
    elif method.lower() == 'pca':
        reducer = PCA(n_components=n_components, random_state=random_state)
        reduced_reps = reducer.fit_transform(latent_reps)
    elif method.lower() == 'umap':
        reducer = umap.UMAP(n_components=n_components, random_state=random_state)
        reduced_reps = reducer.fit_transform(latent_reps)
    else:
        raise ValueError(f"Unsupported dimensionality reduction method: {method}")
    
    print(f"Reduced dimensions to shape: {reduced_reps.shape}")
    return reduced_reps


def reconstruct_from_feature_activations(model, features, model_type='sae', device='cuda'):
    """
    Reconstruct images from feature activations
    
    Args:
        model: Trained model
        features: Feature activations (latent representations)
        model_type: Type of model ('sae' or 'st')
        device: Device to use for computation
        
    Returns:
        Reconstructed images
    """
    model.eval()
    
    with torch.no_grad():
        features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
        
        if model_type.lower() == 'sae':
            if hasattr(model, 'reconstruct'):
                # If model has a direct reconstruct method
                reconstructions = model.reconstruct(features_tensor)
            else:
                # Use decoder to reconstruct
                reconstructions = model.W_d(features_tensor)
        elif model_type.lower() == 'st':
            # For ST, we need to attend to V using features
            if hasattr(model, 'W_v_direct'):
                # Direct K-V approach
                v = model.W_v_direct  # Shape: [m, n]
                # Use matmul to get weighted sum of value vectors
                reconstructions = torch.matmul(features_tensor, v)
            else:
                # Memory bank approach - this is approximate
                raise NotImplementedError("ST reconstruction from features not implemented for memory bank approach")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    return reconstructions.cpu().numpy()


def visualize_latent_space_static(reduced_reps, labels, label_names=None, title="Latent Space", figsize=(10, 8)):
    """
    Create a static visualization of the latent space
    
    Args:
        reduced_reps: Reduced latent representations
        labels: Labels for coloring the points
        label_names: Names of the labels for the legend
        title: Title of the plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    n_components = reduced_reps.shape[1]
    
    if n_components == 2:
        # 2D scatter plot
        plt.figure(figsize=figsize)
        
        unique_labels = np.unique(labels)
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(
                reduced_reps[mask, 0],
                reduced_reps[mask, 1],
                label=label_names[i] if label_names else f"Class {label}",
                alpha=0.7
            )
        
        plt.legend()
        plt.title(title)
        plt.tight_layout()
        
        return plt.gcf()
    
    elif n_components == 3:
        # 3D scatter plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        unique_labels = np.unique(labels)
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                reduced_reps[mask, 0],
                reduced_reps[mask, 1],
                reduced_reps[mask, 2],
                label=label_names[i] if label_names else f"Class {label}",
                alpha=0.7
            )
        
        ax.legend()
        ax.set_title(title)
        plt.tight_layout()
        
        return fig
    
    else:
        # Pairwise scatter plots for more than 3 dimensions
        n_dims = min(n_components, 4)  # Limit to at most 4 dimensions for pairwise plots
        
        fig, axes = plt.subplots(n_dims, n_dims, figsize=figsize)
        
        for i in range(n_dims):
            for j in range(n_dims):
                if i == j:
                    # Histogram on diagonal
                    ax = axes[i, j]
                    unique_labels = np.unique(labels)
                    for k, label in enumerate(unique_labels):
                        mask = labels == label
                        ax.hist(
                            reduced_reps[mask, i],
                            alpha=0.5,
                            label=label_names[k] if label_names else f"Class {label}"
                        )
                    ax.set_title(f"Dim {i+1}")
                else:
                    # Scatter plot on off-diagonal
                    ax = axes[i, j]
                    unique_labels = np.unique(labels)
                    for k, label in enumerate(unique_labels):
                        mask = labels == label
                        ax.scatter(
                            reduced_reps[mask, j],
                            reduced_reps[mask, i],
                            alpha=0.7,
                            s=5
                        )
                    ax.set_xlabel(f"Dim {j+1}")
                    ax.set_ylabel(f"Dim {i+1}")
        
        # Add legend to bottom right plot
        axes[-1, -1].legend()
        plt.suptitle(title)
        plt.tight_layout()
        
        return fig


def visualize_latent_space_interactive(reduced_reps, labels, original_data, model, latent_reps, input_shape=(28, 28),
                                      model_type='sae', label_names=None, title="Interactive Latent Space",
                                      n_samples_reconstruct=10, device='cuda'):
    """
    Create an interactive visualization of the latent space with Plotly
    
    Args:
        reduced_reps: Reduced latent representations
        labels: Labels for coloring the points
        original_data: Original input data
        model: Trained model
        latent_reps: Full latent representations
        input_shape: Shape of input images
        model_type: Type of model ('sae' or 'st')
        label_names: Names of the labels for the legend
        title: Title of the plot
        n_samples_reconstruct: Number of samples to reconstruct for each class
        device: Device to use for computation
        
    Returns:
        Plotly figure
    """
    n_components = reduced_reps.shape[1]
    
    # Prepare data for Plotly
    df = pd.DataFrame()
    
    if n_components >= 1:
        df['x'] = reduced_reps[:, 0]
    if n_components >= 2:
        df['y'] = reduced_reps[:, 1]
    if n_components >= 3:
        df['z'] = reduced_reps[:, 2]
    
    df['label'] = labels
    if label_names:
        df['label_name'] = [label_names[label] for label in labels]
    else:
        df['label_name'] = [f"Class {label}" for label in labels]
    
    # Create scatter plot
    if n_components == 1:
        # 1D scatter plot
        fig = px.scatter(
            df, x='x', color='label_name',
            title=title,
            labels={'x': 'Dimension 1', 'label_name': 'Class'},
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
    elif n_components == 2:
        # 2D scatter plot
        fig = px.scatter(
            df, x='x', y='y', color='label_name',
            title=title,
            labels={'x': 'Dimension 1', 'y': 'Dimension 2', 'label_name': 'Class'},
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
    elif n_components == 3:
        # 3D scatter plot
        fig = px.scatter_3d(
            df, x='x', y='y', z='z', color='label_name',
            title=title,
            labels={'x': 'Dimension 1', 'y': 'Dimension 2', 'z': 'Dimension 3', 'label_name': 'Class'},
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
    else:
        # Multiple 2D scatter plots for pairs of dimensions
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Dimensions 1 vs 2", "Dimensions 3 vs 4"],
            specs=[[{'type': 'xy'}, {'type': 'xy'}]]
        )
        
        unique_labels = np.unique(labels)
        for i, label in enumerate(unique_labels):
            mask = labels == label
            
            # Plot dimensions 1 vs 2
            fig.add_trace(
                go.Scatter(
                    x=reduced_reps[mask, 0],
                    y=reduced_reps[mask, 1],
                    mode='markers',
                    name=label_names[i] if label_names else f"Class {label}",
                    marker=dict(size=6, opacity=0.7)
                ),
                row=1, col=1
            )
            
            # Plot dimensions 3 vs 4 if available
            if n_components >= 4:
                fig.add_trace(
                    go.Scatter(
                        x=reduced_reps[mask, 2],
                        y=reduced_reps[mask, 3],
                        mode='markers',
                        name=label_names[i] if label_names else f"Class {label}",
                        marker=dict(size=6, opacity=0.7),
                        showlegend=False
                    ),
                    row=1, col=2
                )
        
        fig.update_layout(
            title=title,
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            xaxis2_title="Dimension 3",
            yaxis2_title="Dimension 4"
        )
    
    # Add sample reconstructions
    if hasattr(model, 'W_d') or model_type.lower() == 'sae':
        # Only add reconstructions for SAE models for now
        try:
            # Get samples from each class for reconstruction visualization
            samples_to_reconstruct = []
            latent_samples_to_reconstruct = []
            
            unique_labels = np.unique(labels)
            for label in unique_labels:
                mask = labels == label
                # Get indices of this class
                indices = np.where(mask)[0]
                # Randomly select n_samples_reconstruct indices
                if len(indices) >= n_samples_reconstruct:
                    selected_indices = np.random.choice(indices, n_samples_reconstruct, replace=False)
                else:
                    selected_indices = indices
                
                samples_to_reconstruct.extend(selected_indices)
                
                # Get latent representations for selected samples
                latent_samples_to_reconstruct.append(latent_reps[selected_indices])
            
            # Concatenate all latent samples
            latent_samples = np.concatenate(latent_samples_to_reconstruct, axis=0)
            
            # Reconstruct samples
            reconstructions = reconstruct_from_feature_activations(
                model, latent_samples, model_type=model_type, device=device)
            
            # Get original samples
            original_samples = original_data[samples_to_reconstruct].cpu().numpy()
            
            # Create a grid of images
            n_samples = len(samples_to_reconstruct)
            n_cols = min(10, n_samples)
            n_rows = (n_samples + n_cols - 1) // n_cols
            
            # Create a figure for original and reconstructed images
            img_fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=["Original Images", "Reconstructed Images"],
                vertical_spacing=0.1
            )
            
            # Original images
            original_grid = np.zeros((n_rows * input_shape[0], n_cols * input_shape[1]))
            for i in range(n_samples):
                row = i // n_cols
                col = i % n_cols
                # Reshape the sample to the input shape
                img = original_samples[i].reshape(input_shape)
                # Place it in the grid
                original_grid[row*input_shape[0]:(row+1)*input_shape[0], 
                             col*input_shape[1]:(col+1)*input_shape[1]] = img
            
            # Add original images to subplot
            img_fig.add_trace(
                go.Heatmap(
                    z=original_grid,
                    colorscale='Greys',
                    showscale=False
                ),
                row=1, col=1
            )
            
            # Reconstructed images
            recon_grid = np.zeros((n_rows * input_shape[0], n_cols * input_shape[1]))
            for i in range(n_samples):
                row = i // n_cols
                col = i % n_cols
                # Reshape the sample to the input shape
                img = reconstructions[i].reshape(input_shape)
                # Place it in the grid
                recon_grid[row*input_shape[0]:(row+1)*input_shape[0], 
                          col*input_shape[1]:(col+1)*input_shape[1]] = img
            
            # Add reconstructed images to subplot
            img_fig.add_trace(
                go.Heatmap(
                    z=recon_grid,
                    colorscale='Greys',
                    showscale=False
                ),
                row=2, col=1
            )
            
            # Update layout
            img_fig.update_layout(
                title="Sample Reconstructions",
                height=600,
                width=800
            )
            
            # Update axes
            img_fig.update_xaxes(showticklabels=False)
            img_fig.update_yaxes(showticklabels=False)
            
            # Return both figures
            return fig, img_fig
        
        except Exception as e:
            print(f"Error generating reconstructions: {e}")
            import traceback
            traceback.print_exc()
            
            # Return just the scatter plot
            return fig, None
    
    # If not SAE or error occurred, return just the scatter plot
    return fig, None


def interpolate_between_samples(model, data, latent_reps, sample_indices, n_steps=10, 
                               model_type='sae', input_shape=(28, 28), device='cuda'):
    """
    Interpolate between samples in latent space and reconstruct
    
    Args:
        model: Trained model
        data: Original data
        latent_reps: Latent representations
        sample_indices: Indices of samples to interpolate between
        n_steps: Number of interpolation steps
        model_type: Type of model ('sae' or 'st')
        input_shape: Shape of input images
        device: Device to use for computation
        
    Returns:
        Tuple of (animation_frames, original_samples)
    """
    if len(sample_indices) < 2:
        raise ValueError("Need at least 2 sample indices for interpolation")
    
    # Get original samples
    original_samples = data[sample_indices].cpu().numpy()
    
    # Get latent representations of samples
    sample_latent_reps = latent_reps[sample_indices]
    
    # Create interpolation steps between each consecutive pair
    interpolation_frames = []
    
    for i in range(len(sample_indices) - 1):
        start_latent = sample_latent_reps[i]
        end_latent = sample_latent_reps[i+1]
        
        # Generate interpolation steps
        for step in range(n_steps):
            # Calculate interpolation weight
            alpha = step / (n_steps - 1)
            # Interpolate
            interpolated_latent = (1 - alpha) * start_latent + alpha * end_latent
            # Add to frames
            interpolation_frames.append(interpolated_latent)
    
    # Reconstruct from interpolated latent representations
    reconstructions = reconstruct_from_feature_activations(
        model, np.stack(interpolation_frames), model_type=model_type, device=device)
    
    # Reshape reconstructions to image shape
    animation_frames = [recon.reshape(input_shape) for recon in reconstructions]
    
    return animation_frames, original_samples


def create_interpolation_animation(model, data, latent_reps, labels, n_samples_per_class=2, n_steps=10,
                                 model_type='sae', input_shape=(28, 28), device='cuda'):
    """
    Create an animation of interpolation between samples in latent space
    
    Args:
        model: Trained model
        data: Original data
        latent_reps: Latent representations
        labels: Labels for selecting samples
        n_samples_per_class: Number of samples to select from each class
        n_steps: Number of interpolation steps
        model_type: Type of model ('sae' or 'st')
        input_shape: Shape of input images
        device: Device to use for computation
        
    Returns:
        Plotly figure with animation
    """
    try:
        # Select samples from each class
        unique_labels = np.unique(labels)
        selected_indices = []
        
        for label in unique_labels:
            # Get indices of this class
            indices = np.where(labels == label)[0]
            # Randomly select n_samples_per_class indices
            if len(indices) >= n_samples_per_class:
                class_indices = np.random.choice(indices, n_samples_per_class, replace=False)
                selected_indices.extend(class_indices)
            else:
                selected_indices.extend(indices)
        
        # Interpolate between selected samples
        animation_frames, original_samples = interpolate_between_samples(
            model, data, latent_reps, selected_indices, n_steps, model_type, input_shape, device)
        
        # Create animation figure
        fig = go.Figure()
        
        # Add first frame
        fig.add_trace(
            go.Heatmap(
                z=animation_frames[0],
                colorscale='Greys',
                showscale=False
            )
        )
        
        # Create frames for animation
        frames = []
        for i, img in enumerate(animation_frames):
            frames.append(
                go.Frame(
                    data=[go.Heatmap(z=img, colorscale='Greys', showscale=False)],
                    name=f"frame{i}"
                )
            )
        
        fig.frames = frames
        
        # Add slider and buttons for animation control
        fig.update_layout(
            title="Latent Space Interpolation",
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 200, "redraw": True}, "fromcurrent": True}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }],
            sliders=[{
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 20},
                    "prefix": "Frame: ",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 300, "easing": "cubic-in-out"},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [
                            [f"frame{i}"],
                            {"frame": {"duration": 300, "redraw": True}, "mode": "immediate"}
                        ],
                        "label": str(i),
                        "method": "animate"
                    }
                    for i in range(len(animation_frames))
                ]
            }]
        )
        
        # Update axes
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        
        return fig
    
    except Exception as e:
        print(f"Error creating interpolation animation: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_grid_of_samples(samples, titles=None, input_shape=(28, 28), n_cols=5, main_title=None):
    """
    Create a grid of sample images for display
    
    Args:
        samples: List of sample images
        titles: List of titles for each sample
        input_shape: Shape of each image
        n_cols: Number of columns in the grid
        main_title: Main title for the entire figure
        
    Returns:
        Matplotlib figure
    """
    n_samples = len(samples)
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    
    for i, sample in enumerate(samples):
        row = i // n_cols
        col = i % n_cols
        
        # Get the axis to plot on
        if n_rows == 1 and n_cols == 1:
            ax = axes
        elif n_rows == 1:
            ax = axes[col]
        elif n_cols == 1:
            ax = axes[row]
        else:
            ax = axes[row, col]
        
        # Reshape the sample to the input shape
        img = sample.reshape(input_shape)
        
        # Plot the image
        ax.imshow(img, cmap='gray')
        
        # Set title if provided
        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=10)
        
        # Remove axes
        ax.axis('off')
    
    # Hide any unused subplots
    for i in range(n_samples, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        
        if n_rows == 1 and n_cols == 1:
            ax = axes
        elif n_rows == 1:
            ax = axes[col]
        elif n_cols == 1:
            ax = axes[row]
        else:
            ax = axes[row, col]
        
        ax.axis('off')
    
    # Add main title with proper spacing
    if main_title:
        fig.suptitle(main_title, fontsize=14, y=0.98)
        plt.subplots_adjust(top=0.85)  # Make room for the title
    
    plt.tight_layout()
    return fig


def analyze_latent_feature_importance(latent_reps, labels, n_features=5):
    """
    Analyze which latent features are most important for distinguishing classes
    
    Args:
        latent_reps: Latent representations
        labels: Labels for each sample
        n_features: Number of top features to return
        
    Returns:
        Tuple of (feature_importance, top_feature_indices)
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        
        # Train a random forest to identify important features
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(latent_reps, labels)
        
        # Get feature importances
        feature_importance = clf.feature_importances_
        
        # Get top features
        top_feature_indices = np.argsort(feature_importance)[::-1][:n_features]
        
        # Return importance scores and indices
        return feature_importance, top_feature_indices
    
    except Exception as e:
        print(f"Error analyzing feature importance: {e}")
        # Fallback: just return features with highest variance
        feature_variance = np.var(latent_reps, axis=0)
        top_feature_indices = np.argsort(feature_variance)[::-1][:n_features]
        return feature_variance, top_feature_indices


def generate_samples_by_modifying_features(model, latent_reps, top_feature_indices, 
                                         model_type='sae', input_shape=(28, 28), 
                                         min_val=-3, max_val=3, n_steps=5, device='cuda'):
    """
    Generate samples by modifying top features in latent space
    
    Args:
        model: Trained model
        latent_reps: Latent representations
        top_feature_indices: Indices of top features to modify
        model_type: Type of model ('sae' or 'st')
        input_shape: Shape of input images
        min_val: Minimum value for feature modification
        max_val: Maximum value for feature modification
        n_steps: Number of steps between min and max
        device: Device to use for computation
        
    Returns:
        Dictionary mapping feature index to list of modified samples
    """
    # Get mean latent representation to modify
    mean_latent = np.mean(latent_reps, axis=0)
    
    # Create dictionary to store results
    modified_samples = {}
    
    # For each top feature
    for feature_idx in top_feature_indices:
        # Create a range of values to set for this feature
        feature_values = np.linspace(min_val, max_val, n_steps)
        
        # Generate samples with modified feature
        feature_samples = []
        
        for value in feature_values:
            # Copy the mean latent representation
            modified_latent = mean_latent.copy()
            
            # Set the feature to the current value
            modified_latent[feature_idx] = value
            
            # Reshape for batch dimension
            modified_latent = np.expand_dims(modified_latent, axis=0)
            
            # Reconstruct from modified latent
            reconstruction = reconstruct_from_feature_activations(
                model, modified_latent, model_type=model_type, device=device)
            
            # Add to samples
            feature_samples.append(reconstruction[0])
        
        # Add to results
        modified_samples[feature_idx] = feature_samples
    
    return modified_samples


def compare_samples_to_centroids(latent_reps, labels, input_shape=(28, 28)):
    """
    Compare samples by their distance to class centroids in latent space
    
    Args:
        latent_reps: Latent representations of samples
        labels: Class labels
        input_shape: Shape of input images
        
    Returns:
        Dictionary with class centroids and centroid distances
    """
    unique_labels = np.unique(labels)
    class_centroids = {}
    
    # Calculate centroids for each class
    for label in unique_labels:
        mask = labels == label
        class_latent = latent_reps[mask]
        centroid = np.mean(class_latent, axis=0)
        class_centroids[label] = centroid
    
    # Calculate distances from each sample to its class centroid and other centroids
    centroid_distances = {}
    
    for i, sample_latent in enumerate(latent_reps):
        sample_label = labels[i]
        sample_distances = {}
        
        # Calculate distance to each centroid
        for label, centroid in class_centroids.items():
            distance = np.linalg.norm(sample_latent - centroid)
            sample_distances[label] = distance
        
        # Add to results
        centroid_distances[i] = sample_distances
    
    return {'centroids': class_centroids, 'distances': centroid_distances}


def generate_samples_along_principal_component(model, latent_reps, labels, component_index=0,
                                             model_type='sae', input_shape=(28, 28), 
                                             n_steps=5, scale=3.0, device='cuda'):
    """
    Generate samples along a principal component direction in latent space
    
    Args:
        model: Trained model
        latent_reps: Latent representations
        labels: Class labels
        component_index: Index of principal component to use
        model_type: Type of model ('sae' or 'st')
        input_shape: Shape of input images
        n_steps: Number of steps
        scale: Scale factor for the component
        device: Device to use for computation
        
    Returns:
        List of generated samples
    """
    # Compute PCA on latent representations
    pca = PCA(n_components=min(10, latent_reps.shape[1]))
    pca.fit(latent_reps)
    
    # Get the selected principal component
    if component_index >= pca.components_.shape[0]:
        component_index = 0
    
    principal_component = pca.components_[component_index]
    
    # Get mean latent representation
    mean_latent = np.mean(latent_reps, axis=0)
    
    # Generate samples along the principal component
    samples = []
    
    for step in np.linspace(-scale, scale, n_steps):
        # Modify mean latent along principal component
        modified_latent = mean_latent + step * principal_component
        
        # Reshape for batch dimension
        modified_latent = np.expand_dims(modified_latent, axis=0)
        
        # Reconstruct from modified latent
        reconstruction = reconstruct_from_feature_activations(
            model, modified_latent, model_type=model_type, device=device)
        
        # Add to samples
        samples.append(reconstruction[0])
    
    return samples


def create_sample_grid_from_features(model, latent_reps, top_feature_indices, model_type='sae',
                                    input_shape=(28, 28), device='cuda'):
    """
    Create a grid of samples by varying two top features
    
    Args:
        model: Trained model
        latent_reps: Latent representations
        top_feature_indices: Indices of top features to vary
        model_type: Type of model ('sae' or 'st')
        input_shape: Shape of input images
        device: Device to use for computation
        
    Returns:
        Grid of samples (2D array)
    """
    if len(top_feature_indices) < 2:
        raise ValueError("Need at least 2 top features")
    
    # Get the first two top features
    feature1 = top_feature_indices[0]
    feature2 = top_feature_indices[1]
    
    # Get mean latent representation
    mean_latent = np.mean(latent_reps, axis=0)
    
    # Get range of values for the two features
    feature1_values = np.linspace(-3, 3, 5)
    feature2_values = np.linspace(-3, 3, 5)
    
    # Create grid
    grid = np.zeros((len(feature1_values) * input_shape[0], len(feature2_values) * input_shape[1]))
    
    for i, val1 in enumerate(feature1_values):
        for j, val2 in enumerate(feature2_values):
            # Copy the mean latent representation
            modified_latent = mean_latent.copy()
            
            # Set the two features
            modified_latent[feature1] = val1
            modified_latent[feature2] = val2
            
            # Reshape for batch dimension
            modified_latent = np.expand_dims(modified_latent, axis=0)
            
            # Reconstruct from modified latent
            reconstruction = reconstruct_from_feature_activations(
                model, modified_latent, model_type=model_type, device=device)
            
            # Reshape to image
            img = reconstruction[0].reshape(input_shape)
            
            # Place in grid
            grid[i*input_shape[0]:(i+1)*input_shape[0], j*input_shape[1]:(j+1)*input_shape[1]] = img
    
    return grid


def create_thought_vectors_dashboard(model, data, labels, model_type='sae', 
                                   input_shape=(28, 28), device='cuda',
                                   reduction_method='tsne', n_components=2,
                                   label_names=None, output_dir='thought_vectors_output', n_features=5):
    """
    Create a complete dashboard for thought vectors visualization
    
    Args:
        model: Trained model
        data: Input data tensor
        labels: Labels tensor
        model_type: Type of model ('sae' or 'st')
        input_shape: Shape of input images
        device: Device to use for computation
        reduction_method: Dimensionality reduction method
        n_components: Number of components for reduction
        label_names: Names of the labels
        output_dir: Directory to save output
        
    Returns:
        None (saves files to output_dir)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Compute latent representations
    print("Computing latent representations...")
    latent_reps = compute_latent_representations(model, data, model_type, device=device)
    
    # 2. Reduce dimensions for visualization
    print(f"Reducing dimensions with {reduction_method}...")
    reduced_reps = reduce_dimensions(latent_reps, method=reduction_method, n_components=n_components)
    
    # 3. Visualize latent space with static plot
    print("Creating static visualization...")
    static_fig = visualize_latent_space_static(
        reduced_reps, labels.numpy(), label_names, 
        title=f"Latent Space Visualization ({model_type.upper()})")
    static_fig.savefig(os.path.join(output_dir, 'latent_space_static.png'), dpi=300)
    
    # 4. Create interactive visualization
    print("Creating interactive visualization...")
    interactive_fig, recon_fig = visualize_latent_space_interactive(
        reduced_reps, labels.numpy(), data, model, latent_reps, 
        input_shape, model_type, label_names)
    
    interactive_fig.write_html(os.path.join(output_dir, 'latent_space_interactive.html'))
    if recon_fig is not None:
        recon_fig.write_html(os.path.join(output_dir, 'sample_reconstructions.html'))
    
    # 5. Analyze feature importance
    print("Analyzing feature importance...")
    feature_importance, top_feature_indices = analyze_latent_feature_importance(latent_reps, labels.numpy(), n_features=n_features)
    
    # Save feature importance plot
    plt.figure(figsize=(10, 6))
    sorted_indices = np.argsort(feature_importance)[::-1]
    plt.bar(range(len(sorted_indices[:20])), feature_importance[sorted_indices[:20]])
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.title('Top 20 Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300)
    
    # 6. Generate samples by modifying features
    print("Generating samples by modifying features...")
    modified_samples = generate_samples_by_modifying_features(
        model, latent_reps, top_feature_indices[:n_features], model_type, input_shape, device=device)
    
    # Create a grid of samples for each top feature
    # When creating feature variation images
    for feature_idx, samples in modified_samples.items():
        importance_score = feature_importance[feature_idx]
        titles = [f"Value: {val:.2f}" for val in np.linspace(-3, 3, len(samples))]
        fig = create_grid_of_samples(samples, titles, input_shape, n_cols=len(samples), 
                                    main_title=f"Feature {feature_idx} Variation (Importance: {importance_score:.4f})")
        # Save with importance score in filename for easier sorting
        fig.savefig(os.path.join(output_dir, f'feature_{feature_idx}_imp_{importance_score:.4f}_variation.png'), dpi=300)
    
    # 7. Create feature grid visualization
    print("Creating feature grid visualization...")
    try:
        feature_grid = create_sample_grid_from_features(
            model, latent_reps, top_feature_indices[:2], model_type, input_shape, device)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(feature_grid, cmap='gray')
        plt.title(f"Grid of Samples - Features {top_feature_indices[0]} vs {top_feature_indices[1]}")
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, 'feature_grid.png'), dpi=300)
    except Exception as e:
        print(f"Error creating feature grid: {e}")
    
    # 8. Generate interpolation animation
    print("Creating interpolation animation...")
    interpolation_fig = create_interpolation_animation(
        model, data, latent_reps, labels.numpy(), n_samples_per_class=1, model_type=model_type, 
        input_shape=input_shape, device=device)
    
    if interpolation_fig is not None:
        interpolation_fig.write_html(os.path.join(output_dir, 'interpolation_animation.html'))
    
    # 9. Generate samples along principal component
    print("Generating samples along principal component...")
    pca_samples = generate_samples_along_principal_component(
        model, latent_reps, labels.numpy(), model_type=model_type, input_shape=input_shape, device=device)
    
    fig = create_grid_of_samples(
        pca_samples, 
        [f"PC1: {val:.2f}" for val in np.linspace(-3, 3, len(pca_samples))], 
        input_shape, n_cols=len(pca_samples))
    fig.suptitle("Samples along Principal Component 1")
    fig.savefig(os.path.join(output_dir, 'pc1_samples.png'), dpi=300)
    
    print(f"Thought Vectors dashboard created in {output_dir}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Thought Vectors Visualization for Image Datasets')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained model file')
    parser.add_argument('--model_type', type=str, default='sae', choices=['sae', 'st'],
                      help='Type of model')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='mnist', 
                      help='Dataset name or path to custom dataset')
    parser.add_argument('--dataset_path', type=str, default=None,
                      help='Path to dataset file (default depends on dataset name)')
    parser.add_argument('--n_samples', type=int, default=1000,
                      help='Number of samples to use (default: 1000)')
    parser.add_argument('--input_shape', type=str, default='28,28',
                      help='Input shape (default: 28,28 for MNIST)')
    parser.add_argument('--n_features', type=str, default=5,
                    help='Number of top features to analyze (default: 5)')
    # Visualization parameters
    parser.add_argument('--reduction_method', type=str, default='tsne',
                      choices=['tsne', 'pca', 'umap'],
                      help='Dimensionality reduction method')
    parser.add_argument('--n_components', type=int, default=2,
                      help='Number of components for dimensionality reduction')
    parser.add_argument('--output_dir', type=str, default='thought_vectors_output',
                      help='Directory to save output')
    
    # Device parameters
    parser.add_argument('--device', type=str, default=None,
                      help='Device to use for computation (default: cuda if available, else cpu)')
    
    # Parse arguments
    args = parser.parse_args()
    # Process input_shape
    args.input_shape = tuple(map(int, args.input_shape.split(',')))
    
    # Set device if not specified
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Set dataset path if not specified
    if args.dataset_path is None:
        if args.dataset.lower() == 'mnist':
            args.dataset_path = 'data/mnist_train.csv'
        elif args.dataset.lower() == 'fashion_mnist':
            args.dataset_path = 'data/fashion_mnist_train.csv'
        else:
            args.dataset_path = args.dataset
    
    return args


def main():
    """Main function"""
    args = parse_args()
    
    print(f"Using device: {args.device}")
    print(f"Model path: {args.model_path}")
    print(f"Model type: {args.model_type}")
    print(f"Dataset: {args.dataset}")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    args.n_features = int(args.n_features)
    # Load model
    model = load_model(args.model_path, args.model_type, args.device)
    if model is None:
        print("Failed to load model. Exiting.")
        return
    # Load dataset
    if args.dataset.lower() == 'mnist':
        data, labels, label_names = load_mnist_dataset(args.dataset_path, args.n_samples, args.input_shape)
    else:
        data, labels, label_names = load_image_dataset(args.dataset_path, args.n_samples, args.input_shape)
    
    # Create the thought vectors dashboard
    create_thought_vectors_dashboard(
        model, data, labels, args.model_type, args.input_shape, args.device,
        args.reduction_method, args.n_components, label_names, args.output_dir, args.n_features
    )


if __name__ == "__main__":
    main()