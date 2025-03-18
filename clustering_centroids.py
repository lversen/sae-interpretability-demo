import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
from typing import Dict, List, Tuple
import argparse

# Import the models
from SAE import SparseAutoencoder
from ST import SparseTransformer

# For loading data
import torch.utils.data
from torchvision import datasets, transforms

def load_mnist_data(batch_size=128, num_samples=1000):
    """Load MNIST dataset for testing"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load training dataset
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    # Limit the number of samples if needed
    if num_samples and num_samples < len(train_dataset):
        indices = torch.randperm(len(train_dataset))[:num_samples]
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    
    return train_loader

def load_models(sae_path, st_path, input_dim, feature_dim, attention_dim):
    """Load pre-trained SAE and ST models"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Try to infer model dimensions from the checkpoint files if not provided
    inferred_feature_dim = feature_dim
    inferred_attention_dim = attention_dim
    
    # Check SAE checkpoint for dimensions
    try:
        checkpoint = torch.load(sae_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Try to infer feature dimension from weight matrices
        if 'W_e.weight' in state_dict:
            inferred_feature_dim = state_dict['W_e.weight'].shape[0]
            print(f"Inferred feature dimension from SAE model: {inferred_feature_dim}")
    except Exception as e:
        print(f"Could not infer dimensions from SAE checkpoint: {e}")
    
    # Check ST checkpoint for dimensions
    try:
        checkpoint = torch.load(st_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Try to infer attention dimension from weight matrices
        if 'W_q.weight' in state_dict:
            inferred_attention_dim = state_dict['W_q.weight'].shape[0]
            print(f"Inferred attention dimension from ST model: {inferred_attention_dim}")
            
        # Also try to infer feature dimension if not yet determined
        if inferred_feature_dim is None:
            if 'memory_indices' in state_dict:
                inferred_feature_dim = len(state_dict['memory_indices'])
                print(f"Inferred feature dimension from ST model: {inferred_feature_dim}")
            elif 'W_k_direct' in state_dict:
                inferred_feature_dim = state_dict['W_k_direct'].shape[0]
                print(f"Inferred feature dimension from ST model: {inferred_feature_dim}")
    except Exception as e:
        print(f"Could not infer dimensions from ST checkpoint: {e}")
    
    # Use inferred dimensions if available
    feature_dim = inferred_feature_dim if inferred_feature_dim is not None else feature_dim
    attention_dim = inferred_attention_dim if inferred_attention_dim is not None else attention_dim
    
    # Load SAE model
    sae_model = SparseAutoencoder(
        n=input_dim,
        m=feature_dim,
        sae_model_path=sae_path,
        device=device
    )
    
    # Load ST model
    st_model = SparseTransformer(
        X=torch.randn(1000, input_dim),  # Placeholder for initialization
        n=input_dim,
        m=feature_dim,
        a=attention_dim,
        st_model_path=st_path,
        device=device
    )
    
    # Load saved weights if they exist
    if os.path.exists(sae_path):
        checkpoint = torch.load(sae_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            sae_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            sae_model.load_state_dict(checkpoint)
        print(f"Loaded SAE model from {sae_path}")
    else:
        print(f"Warning: SAE model file {sae_path} not found. Using untrained model.")
    
    if os.path.exists(st_path):
        checkpoint = torch.load(st_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            st_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            st_model.load_state_dict(checkpoint)
        print(f"Loaded ST model from {st_path}")
    else:
        print(f"Warning: ST model file {st_path} not found. Using untrained model.")
    
    return sae_model, st_model

def generate_embeddings(model, data_loader, model_type='sae'):
    """Generate embeddings using the specified model"""
    device = next(model.parameters()).device
    embeddings = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data = data.to(device)
            
            # Flatten the input if it's an image
            if len(data.shape) > 2:
                data = data.view(data.size(0), -1)
            
            # Generate embeddings based on model type
            if model_type.lower() == 'sae':
                _, _, feature_activations = model(data)
                batch_embeddings = feature_activations.cpu().numpy()
            else:  # ST model
                _, _, feature_activations, _ = model(data)
                batch_embeddings = feature_activations.cpu().numpy()
            
            embeddings.append(batch_embeddings)
            labels.append(target.cpu().numpy())
            
            # Limit the number of batches for faster processing
            if batch_idx >= 10:
                break
    
    # Concatenate all batches
    all_embeddings = np.vstack(embeddings)
    all_labels = np.concatenate(labels)
    
    return all_embeddings, all_labels

def apply_dimensionality_reduction(embeddings, method='tsne', random_state=42):
    """Apply dimensionality reduction to embeddings"""
    if method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=random_state)
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, random_state=random_state)
    elif method.lower() == 'umap':
        reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=random_state)
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")
    
    reduced_embeddings = reducer.fit_transform(embeddings)
    return reduced_embeddings

def plot_embeddings(embeddings_2d, labels, title, figure_size=(12, 10)):
    """Plot embeddings in 2D space with color-coded labels"""
    # Create a DataFrame for easier plotting
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'label': labels
    })
    
    # Create the plot
    plt.figure(figsize=figure_size)
    
    # Get unique labels
    unique_labels = sorted(np.unique(df['label']))
    
    # Create a color map
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    # Create a scatter plot for each label
    for i, label in enumerate(unique_labels):
        label_data = df[df['label'] == label]
        plt.scatter(label_data['x'], label_data['y'], 
                    color=colors[i], label=f"Class {label}", 
                    alpha=0.7, s=50)
    
    plt.title(title, fontsize=16)
    plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    return plt.gcf()

def calculate_centroids(embeddings_2d, labels):
    """Calculate centroids for each class in the embedding space"""
    # Create a DataFrame for the embedding data
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'label': labels
    })
    
    # Calculate centroids for each class
    centroids = df.groupby('label').mean().reset_index()
    return centroids

def plot_centroids(embeddings_2d, labels, centroids, title):
    """Plot embeddings with class centroids highlighted"""
    # Create a DataFrame for the embedding data
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'label': labels
    })
    
    # Create the plot
    plt.figure(figsize=(14, 12))
    
    # Get unique labels
    unique_labels = sorted(np.unique(df['label']))
    
    # Create a color map
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    # Plot individual points
    for i, label in enumerate(unique_labels):
        label_data = df[df['label'] == label]
        plt.scatter(label_data['x'], label_data['y'], 
                    color=colors[i], label=f"Class {label}", 
                    alpha=0.3, s=30)
    
    # Plot centroids with labels
    for i, row in centroids.iterrows():
        plt.scatter(row['x'], row['y'], color=colors[int(row['label'])], 
                   s=300, edgecolors='black', linewidths=2, alpha=1.0)
        plt.annotate(f"Class {int(row['label'])}", (row['x'], row['y']), 
                     fontsize=14, fontweight='bold', ha='center', va='center')
    
    plt.title(title, fontsize=16)
    plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    return plt.gcf()

def calculate_centroid_distances(centroids):
    """Calculate pairwise distances between class centroids"""
    # Extract coordinates and labels
    labels = centroids['label'].values
    points = centroids[['x', 'y']].values
    
    # Calculate pairwise distances
    n = len(points)
    distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            distances[i, j] = np.sqrt(np.sum((points[i] - points[j])**2))
    
    # Create a DataFrame for the distance matrix
    distance_df = pd.DataFrame(distances, index=labels, columns=labels)
    
    return distance_df

def plot_distance_heatmap(distance_matrix, title):
    """Plot distance matrix as a heatmap"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(distance_matrix, annot=True, cmap='viridis', fmt='.2f')
    plt.title(title, fontsize=16)
    plt.tight_layout()
    
    return plt.gcf()

def calculate_clustering_metrics(distance_matrix):
    """Calculate metrics to evaluate clustering quality"""
    # Average distance between all pairs of centroids
    avg_distance = np.mean(distance_matrix.values)
    
    # Calculate silhouette-like score (higher is better)
    # This is a simplified version inspired by silhouette score
    n_clusters = len(distance_matrix)
    min_distances = []
    
    for i in range(n_clusters):
        # Get distances from this centroid to all others
        distances = [distance_matrix.iloc[i, j] for j in range(n_clusters) if i != j]
        # Find the minimum distance to another cluster
        if distances:
            min_distances.append(min(distances))
    
    avg_min_distance = np.mean(min_distances) if min_distances else 0
    
    # Calculate a simplified Davies-Bouldin-like score (lower is better)
    # This is the ratio of intra-cluster to inter-cluster distances
    db_score = 1.0 / (avg_min_distance + 1e-10) if avg_min_distance > 0 else float('inf')
    
    return {
        'avg_distance': avg_distance,
        'avg_min_distance': avg_min_distance,
        'separation_score': avg_min_distance,  # Higher is better
        'davies_bouldin_like': db_score  # Lower is better
    }

def compare_models(sae_model, st_model, data_loader, reduction_method='tsne'):
    """Compare SAE and ST models by analyzing their clustering properties"""
    # Generate embeddings
    print("Generating SAE embeddings...")
    sae_embeddings, labels = generate_embeddings(sae_model, data_loader, model_type='sae')
    
    print("Generating ST embeddings...")
    st_embeddings, _ = generate_embeddings(st_model, data_loader, model_type='st')
    
    # Apply dimensionality reduction
    print(f"Applying {reduction_method} dimensionality reduction...")
    sae_embeddings_2d = apply_dimensionality_reduction(sae_embeddings, method=reduction_method)
    st_embeddings_2d = apply_dimensionality_reduction(st_embeddings, method=reduction_method)
    
    # Plot embeddings
    sae_plot = plot_embeddings(sae_embeddings_2d, labels, f"SAE Embeddings ({reduction_method.upper()})")
    st_plot = plot_embeddings(st_embeddings_2d, labels, f"ST Embeddings ({reduction_method.upper()})")
    
    # Calculate centroids
    sae_centroids = calculate_centroids(sae_embeddings_2d, labels)
    st_centroids = calculate_centroids(st_embeddings_2d, labels)
    
    # Plot centroids
    sae_centroid_plot = plot_centroids(sae_embeddings_2d, labels, sae_centroids, 
                                      f"SAE Embeddings with Centroids ({reduction_method.upper()})")
    st_centroid_plot = plot_centroids(st_embeddings_2d, labels, st_centroids, 
                                     f"ST Embeddings with Centroids ({reduction_method.upper()})")
    
    # Calculate centroid distances
    sae_distances = calculate_centroid_distances(sae_centroids)
    st_distances = calculate_centroid_distances(st_centroids)
    
    # Plot distance heatmaps
    sae_heatmap = plot_distance_heatmap(sae_distances, "SAE Centroid Distances")
    st_heatmap = plot_distance_heatmap(st_distances, "ST Centroid Distances")
    
    # Calculate clustering metrics
    sae_metrics = calculate_clustering_metrics(sae_distances)
    st_metrics = calculate_clustering_metrics(st_distances)
    
    # Prepare comparison results
    results = {
        'plots': {
            'sae_embeddings': sae_plot,
            'st_embeddings': st_plot,
            'sae_centroids': sae_centroid_plot,
            'st_centroids': st_centroid_plot,
            'sae_heatmap': sae_heatmap,
            'st_heatmap': st_heatmap
        },
        'distances': {
            'sae_distances': sae_distances,
            'st_distances': st_distances
        },
        'metrics': {
            'sae': sae_metrics,
            'st': st_metrics
        }
    }
    
    # Print comparison metrics
    print("\n===== Clustering Metrics Comparison =====")
    print(f"SAE - Avg Centroid Distance: {sae_metrics['avg_distance']:.4f}")
    print(f"ST - Avg Centroid Distance: {st_metrics['avg_distance']:.4f}")
    print(f"SAE - Cluster Separation Score: {sae_metrics['separation_score']:.4f}")
    print(f"ST - Cluster Separation Score: {st_metrics['separation_score']:.4f}")
    print(f"SAE - Davies-Bouldin-like Score: {sae_metrics['davies_bouldin_like']:.4f}")
    print(f"ST - Davies-Bouldin-like Score: {st_metrics['davies_bouldin_like']:.4f}")
    
    # Determine which model is better
    better_separation = 'SAE' if sae_metrics['separation_score'] > st_metrics['separation_score'] else 'ST'
    better_davies_bouldin = 'SAE' if sae_metrics['davies_bouldin_like'] < st_metrics['davies_bouldin_like'] else 'ST'
    
    print(f"\nBetter Cluster Separation: {better_separation}")
    print(f"Better Davies-Bouldin Score: {better_davies_bouldin}")
    
    return results

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Compare SAE and ST models by analyzing cluster centroids')
    
    parser.add_argument('--sae_path', type=str, required=True,
                        help='Path to the SAE model file (.pth)')
    parser.add_argument('--st_path', type=str, required=True,
                        help='Path to the ST model file (.pth)')
    parser.add_argument('--input_dim', type=int, default=784,
                        help='Input dimension (default: 784 for MNIST)')
    parser.add_argument('--feature_dim', type=int, default=None,
                        help='Feature dimension (will be inferred from models if not provided)')
    parser.add_argument('--attention_dim', type=int, default=None,
                        help='Attention dimension for ST model (will be inferred if possible)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for data loading (default: 128)')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples to use (default: 1000)')
    parser.add_argument('--reduction', type=str, default='tsne', choices=['tsne', 'pca', 'umap'],
                        help='Dimensionality reduction method (default: tsne)')
    parser.add_argument('--save_plots', action='store_true',
                        help='Save plots to files instead of displaying them')
    parser.add_argument('--output_dir', type=str, default='comparison_results',
                        help='Directory to save output plots if --save_plots is used')
    
    return parser.parse_args()

# Main execution
if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Verify that model files exist
    if not os.path.exists(args.sae_path):
        print(f"Error: SAE model file not found: {args.sae_path}")
        exit(1)
    if not os.path.exists(args.st_path):
        print(f"Error: ST model file not found: {args.st_path}")
        exit(1)
    
    # Try to infer dimensions from models if not provided
    feature_dim = args.feature_dim
    attention_dim = args.attention_dim
    
    # If feature_dim not provided, try to infer from model file name
    if feature_dim is None:
        # Try to extract from the path (assuming standard directory structure)
        for path in [args.sae_path, args.st_path]:
            try:
                # Check for feature dimension in directory structure
                # Assuming pattern like "models/mnist/sae/relu/500/..."
                parts = path.split('/')
                for part in parts:
                    if part.isdigit():
                        feature_dim = int(part)
                        print(f"Inferred feature dimension from path: {feature_dim}")
                        break
            except:
                pass
        
        # If still None, use a default
        if feature_dim is None:
            feature_dim = 500
            print(f"Could not infer feature dimension, using default: {feature_dim}")
    
    # If attention_dim not provided, try to infer or use default
    if attention_dim is None:
        try:
            # Try to extract from dimensions in loaded model (will do this more accurately later)
            # For now, just use a heuristic
            attention_dim = feature_dim // 4
            print(f"Using estimated attention dimension: {attention_dim}")
        except:
            attention_dim = 128
            print(f"Could not estimate attention dimension, using default: {attention_dim}")
    
    # Create output directory if saving plots
    if args.save_plots:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Will save plots to: {args.output_dir}")
    
    print(f"\nComparing models:")
    print(f"SAE model: {args.sae_path}")
    print(f"ST model: {args.st_path}")
    print(f"Input dimension: {args.input_dim}")
    print(f"Feature dimension: {feature_dim}")
    print(f"Attention dimension: {attention_dim}")
    print(f"Dimensionality reduction: {args.reduction.upper()}")
    
    # Load data
    print("\nLoading MNIST data...")
    data_loader = load_mnist_data(batch_size=args.batch_size, num_samples=args.num_samples)
    
    # Load models
    print("Loading models...")
    sae_model, st_model = load_models(args.sae_path, args.st_path, args.input_dim, feature_dim, attention_dim)
    
    # Compare models
    results = compare_models(sae_model, st_model, data_loader, reduction_method=args.reduction)
    
    # Save or show plots
    if args.save_plots:
        for name, fig in results['plots'].items():
            output_path = os.path.join(args.output_dir, f"{name}.png")
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to: {output_path}")
        
        # Save distance matrices to CSV
        for name, df in results['distances'].items():
            output_path = os.path.join(args.output_dir, f"{name}.csv")
            df.to_csv(output_path)
            print(f"Saved distance matrix to: {output_path}")
        
        # Save metrics comparison
        metrics_df = pd.DataFrame({
            'Metric': ['Average Centroid Distance', 'Cluster Separation Score', 'Davies-Bouldin-like Score'],
            'SAE': [
                results['metrics']['sae']['avg_distance'],
                results['metrics']['sae']['separation_score'],
                results['metrics']['sae']['davies_bouldin_like']
            ],
            'ST': [
                results['metrics']['st']['avg_distance'],
                results['metrics']['st']['separation_score'],
                results['metrics']['st']['davies_bouldin_like']
            ]
        })
        metrics_path = os.path.join(args.output_dir, "clustering_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Saved metrics comparison to: {metrics_path}")
        
        print("\nAll results have been saved to the output directory.")
    else:
        # Show all plots interactively
        plt.show()