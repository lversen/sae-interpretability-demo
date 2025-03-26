# Set environment variable at the top of the file to fix KMeans memory leak
import os
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.cluster import KMeans
import glob
from tqdm import tqdm
import argparse
from collections import defaultdict

def find_all_models(base_dir="models"):
    """
    Find all model files in the hierarchical folder structure.
    
    Args:
        base_dir: Base directory containing the models
        
    Returns:
        Dictionary mapping model paths to metadata
    """
    model_files = {}
    
    # Walk through the directory structure
    for dataset in os.listdir(base_dir):
        dataset_path = os.path.join(base_dir, dataset)
        if not os.path.isdir(dataset_path):
            continue
            
        for model_type in os.listdir(dataset_path):
            type_path = os.path.join(dataset_path, model_type)
            if not os.path.isdir(type_path) or model_type not in ['sae', 'st']:
                continue
                
            for function_type in os.listdir(type_path):
                function_path = os.path.join(type_path, function_type)
                if not os.path.isdir(function_path):
                    continue
                    
                for feature_dim in os.listdir(function_path):
                    feature_path = os.path.join(function_path, feature_dim)
                    if not os.path.isdir(feature_path):
                        continue
                        
                    # Find all .pth files in this directory
                    for model_file in glob.glob(os.path.join(feature_path, "*.pth")):
                        # Extract metadata from path
                        metadata = {
                            'dataset': dataset,
                            'model_type': model_type,
                            'function_type': function_type,
                            'feature_dimension': feature_dim,
                            'filename': os.path.basename(model_file),
                            'full_path': model_file
                        }
                        
                        # Parse parameters from filename
                        filename = os.path.basename(model_file)
                        for param in filename.replace('.pth', '').split('_'):
                            if param.startswith('bs'):
                                metadata['batch_size'] = param[2:]
                            elif param.startswith('lr'):
                                metadata['learning_rate'] = param[2:].replace('p', '.')
                            elif param.startswith('steps'):
                                metadata['steps'] = param[5:]
                            elif param.startswith('l1'):
                                metadata['l1_lambda'] = param[2:].replace('p', '.')
                            elif param.startswith('accum'):
                                metadata['grad_accum'] = param[5:]
                            elif param == 'memory':
                                metadata['memory'] = True
                            elif param == 'oldst':
                                metadata['old_st'] = True
                        
                        model_files[model_file] = metadata
    
    print(f"Found {len(model_files)} model files.")
    return model_files

def safely_open_model(model_path, device='cpu'):
    """Safely open a model file with proper error handling"""
    try:
        return torch.load(model_path, map_location=device, weights_only=True)
    except RuntimeError as e:
        # Fallback to weights_only=False if the above fails
        print(f"Error loading with weights_only=True, trying without: {e}")
        try:
            return torch.load(model_path, map_location=device)
        except Exception as e2:
            print(f"Failed to load model {model_path}: {e2}")
            return None
    except FileNotFoundError:
        print(f"Model file not found: {model_path}")
        return None
    except Exception as e:
        print(f"Unexpected error loading model {model_path}: {e}")
        return None

def extract_sae_feature_vectors(model_path, device='cpu'):
    """
    Extract feature vectors from an SAE model.
    
    Args:
        model_path: Path to the SAE model file
        device: Device to use for computation
        
    Returns:
        numpy array of feature vectors
    """
    try:
        print(f"Loading SAE model from {model_path}")
        # Use the safe loading function
        checkpoint = safely_open_model(model_path, device)
        if checkpoint is None:
            return None
        
        # Extract state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Extract decoder weights
        if 'W_d.weight' in state_dict:
            # Decoder weight matrix has shape (n, m) for SAE
            # Each column is a feature vector
            decoder_weights = state_dict['W_d.weight'].cpu().numpy()
            
            # Transpose to get feature vectors as rows
            feature_vectors = decoder_weights.T
            
            print(f"Extracted {feature_vectors.shape[0]} SAE feature vectors with dimension {feature_vectors.shape[1]}")
            return feature_vectors
        else:
            print("Could not find W_d.weight in the model state dict")
            return None
            
    except Exception as e:
        print(f"Error loading SAE model: {e}")
        return None

def extract_st_feature_vectors(model_path, device='cpu'):
    """
    Extract feature vectors from an ST model.
    
    Args:
        model_path: Path to the ST model file
        device: Device to use for computation
        
    Returns:
        numpy array of feature vectors
    """
    try:
        print(f"Loading ST model from {model_path}")
        # Use the safe loading function
        checkpoint = safely_open_model(model_path, device)
        if checkpoint is None:
            return None
        
        # Extract state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Check if this is a direct K-V model
        use_direct_kv = 'W_k_direct' in state_dict and 'W_v_direct' in state_dict
        
        if use_direct_kv:
            # Extract value vectors directly from the model
            if 'W_v_direct' in state_dict:
                value_vectors = state_dict['W_v_direct'].cpu().numpy()
                
                print(f"Extracted {value_vectors.shape[0]} ST direct value vectors with dimension {value_vectors.shape[1]}")
                return value_vectors
            else:
                print("Could not find W_v_direct in the model state dict")
                return None
        else:
            # For memory bank approach, we'd need data to compute value vectors
            # Since we don't have direct access to the data, we'll use the memory indices
            # and assume the W_v matrix represents the transformation
            print("Memory bank approach detected. Using W_v weights as approximation.")
            if 'W_v.weight' in state_dict:
                # W_v has shape (output_dim, input_dim) where output_dim is usually equal to input_dim
                # We'll use the rows as feature vectors
                value_approx = state_dict['W_v.weight'].cpu().numpy()
                
                print(f"Extracted {value_approx.shape[0]} ST memory-based value vectors with dimension {value_approx.shape[1]}")
                return value_approx
            else:
                print("Could not find W_v.weight in the model state dict")
                return None
                
    except Exception as e:
        print(f"Error loading ST model: {e}")
        return None

def compute_feature_distances(feature_vectors, metric='cosine'):
    """
    Compute distances between feature vectors.
    
    Args:
        feature_vectors: Array of feature vectors
        metric: Distance metric to use ('cosine' or 'euclidean')
        
    Returns:
        Dictionary of distance statistics
    """
    if feature_vectors is None or len(feature_vectors) == 0:
        return None
        
    # Calculate pairwise distances
    if metric == 'cosine':
        distances = cosine_distances(feature_vectors)
    else:  # euclidean
        distances = euclidean_distances(feature_vectors)
    
    # Get upper triangle of distance matrix (excluding diagonal)
    distances_upper = distances[np.triu_indices_from(distances, k=1)]
    
    # Calculate statistics
    stats = {
        'avg_distance': np.mean(distances_upper),
        'median_distance': np.median(distances_upper),
        'min_distance': np.min(distances_upper),
        'max_distance': np.max(distances_upper),
        'std_distance': np.std(distances_upper),
        'num_features': len(feature_vectors),
        'dimension': feature_vectors.shape[1]
    }
    
    return stats

def calculate_centroid(feature_vectors):
    """
    Calculate the centroid of a set of feature vectors.
    
    Args:
        feature_vectors: Array of feature vectors
        
    Returns:
        Centroid vector
    """
    if feature_vectors is None or len(feature_vectors) == 0:
        return None
        
    return np.mean(feature_vectors, axis=0)

def compute_average_centroid_distance(feature_vectors, num_clusters=10, metric='cosine'):
    """
    Compute the average distance between feature centroids.
    
    This function:
    1. Runs K-means clustering directly in the original feature space (no dimensionality reduction)
    2. Calculates average pairwise distances between centroids
    
    Note: Centroid distances are calculated in the original high-dimensional space
    to preserve all feature relationships and avoid information loss.
    
    Args:
        feature_vectors: Array of feature vectors
        num_clusters: Number of clusters to use
        metric: Distance metric to use ('cosine' or 'euclidean')
        
    Returns:
        Dictionary with centroid distance statistics
    """
    if feature_vectors is None or len(feature_vectors) == 0:
        return None
        
    # Check if we have enough feature vectors for clustering
    if len(feature_vectors) < num_clusters:
        print(f"Not enough feature vectors ({len(feature_vectors)}) for {num_clusters} clusters. Reducing clusters.")
        num_clusters = max(2, len(feature_vectors) // 2)
    
    try:
        # Run K-means clustering with safety measures
        print(f"Computing centroids in original feature space (no dimensionality reduction)")
        kmeans = KMeans(
            n_clusters=num_clusters, 
            random_state=42, 
            n_init=10,
            max_iter=300,  # Limit iterations
            tol=1e-4       # Convergence tolerance
        )
        kmeans.fit(feature_vectors)
        
        # Get centroids
        centroids = kmeans.cluster_centers_
        
        # Calculate centroid distances
        if metric == 'cosine':
            centroid_distances = cosine_distances(centroids)
        else:  # euclidean
            centroid_distances = euclidean_distances(centroids)
        
        # Get upper triangle of distance matrix (excluding diagonal)
        distances_upper = centroid_distances[np.triu_indices_from(centroid_distances, k=1)]
        
        # Calculate statistics
        stats = {
            'avg_centroid_distance': np.mean(distances_upper),
            'median_centroid_distance': np.median(distances_upper),
            'min_centroid_distance': np.min(distances_upper),
            'max_centroid_distance': np.max(distances_upper),
            'std_centroid_distance': np.std(distances_upper),
            'num_centroids': num_clusters,
            'dimension': feature_vectors.shape[1]
        }
        
        return stats
    except Exception as e:
        print(f"Error computing centroid distances: {e}")
        return None

def analyze_models_distances(models_dict, metrics=None, num_clusters=10):
    """
    Analyze distances for all models with improved error handling.
    
    Args:
        models_dict: Dictionary of model paths and metadata
        metrics: List of distance metrics to use (default: ['cosine', 'euclidean'])
        num_clusters: Number of clusters for centroid computation
        
    Returns:
        DataFrame with distance analysis results
    """
    if metrics is None:
        metrics = ['cosine', 'euclidean']
        
    results = []
    skipped = []
    
    for model_path, metadata in tqdm(models_dict.items(), desc="Analyzing models"):
        try:
            model_type = metadata['model_type']
            
            # Extract feature vectors based on model type
            if model_type == 'sae':
                feature_vectors = extract_sae_feature_vectors(model_path)
            else:  # 'st'
                feature_vectors = extract_st_feature_vectors(model_path)
                
            if feature_vectors is None:
                print(f"Skipping {model_path}: Could not extract feature vectors")
                skipped.append(model_path)
                continue
                
            # Calculate distances using each metric
            for metric in metrics:
                try:
                    # Process each metric separately to isolate errors
                    distance_stats = compute_feature_distances(feature_vectors, metric=metric)
                    centroid_stats = compute_average_centroid_distance(
                        feature_vectors, num_clusters=num_clusters, metric=metric)
                    
                    if distance_stats is None or centroid_stats is None:
                        continue
                        
                    # Combine all stats
                    result = {
                        'model_path': model_path,
                        'metric': metric,
                        **metadata,
                        **distance_stats,
                        **centroid_stats
                    }
                    
                    results.append(result)
                except Exception as metric_error:
                    print(f"Error processing metric {metric} for {model_path}: {metric_error}")
                    
        except Exception as model_error:
            print(f"Error processing model {model_path}: {model_error}")
            skipped.append(model_path)
    
    # Report on skipped models
    if skipped:
        print(f"Skipped {len(skipped)} models due to errors")
        
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    return results_df

def create_comparison_visualizations(results_df, output_dir='model_comparison'):
    """
    Create visualizations comparing models based on distance metrics.
    
    Args:
        results_df: DataFrame with distance analysis results
        output_dir: Directory to save visualizations
        
    Returns:
        List of generated visualization paths
    """
    os.makedirs(output_dir, exist_ok=True)
    visualizations = []
    
    # Add explanation about centroid calculation without dimensionality reduction
    info_text = """
    Note: All centroid distances were calculated in the original feature space.
    No dimensionality reduction was applied before centroid calculation to preserve
    all feature relationships and avoid information loss.
    """
    
    # Create information text file
    info_path = os.path.join(output_dir, 'analysis_information.txt')
    with open(info_path, 'w') as f:
        f.write(info_text)
    visualizations.append(info_path)
    
    # Ensure we have results to visualize
    if results_df.empty:
        print("No results to visualize.")
        return visualizations
    
    # Create visualization for each metric
    for metric in results_df['metric'].unique():
        # Create an explicit copy of the filtered DataFrame to avoid SettingWithCopyWarning
        metric_df = results_df[results_df['metric'] == metric].copy()
        
        if metric_df.empty:
            continue
            
        # 1. Bar chart comparing average centroid distances by model type and function
        plt.figure(figsize=(14, 8))
        
        # Ensure feature_dimension is numeric - now using the copy
        metric_df['feature_dimension'] = pd.to_numeric(metric_df['feature_dimension'])
        
        # Group by model type and function type
        grouped = metric_df.groupby(['model_type', 'function_type'])['avg_centroid_distance'].mean().reset_index()
        
        # Pivot for easier plotting
        pivot_df = grouped.pivot(index='function_type', columns='model_type', values='avg_centroid_distance')
        
        # Plot
        ax = pivot_df.plot(kind='bar', rot=45)
        plt.title(f'Average Centroid Distance by Model Type and Function ({metric} metric)')
        plt.ylabel('Average Centroid Distance')
        plt.xlabel('Function Type')
        plt.tight_layout()
        
        # Save figure
        bar_path = os.path.join(output_dir, f'avg_centroid_distance_{metric}.png')
        plt.savefig(bar_path)
        visualizations.append(bar_path)
        plt.close()
        
        # 2. Heatmap comparing average centroid distances by model type and function
        plt.figure(figsize=(12, 10))
        
        # Create pivot table for heatmap
        pivot_table = metric_df.pivot_table(
            values='avg_centroid_distance', 
            index=['model_type', 'function_type'],
            columns=['feature_dimension'],
            aggfunc='mean'
        )
        
        # Create heatmap
        sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt='.3f')
        plt.title(f'Average Centroid Distance by Configuration ({metric} metric)')
        plt.tight_layout()
        
        # Save figure
        heatmap_path = os.path.join(output_dir, f'centroid_distance_heatmap_{metric}.png')
        plt.savefig(heatmap_path)
        visualizations.append(heatmap_path)
        plt.close()
        
        # 3. Scatter plot of centroid distance vs. feature dimension
        plt.figure(figsize=(10, 6))
        
        # Create scatter plot with different colors for model types
        for model_type, group in metric_df.groupby('model_type'):
            plt.scatter(
                group['feature_dimension'], 
                group['avg_centroid_distance'],
                label=model_type,
                alpha=0.7
            )
            
        plt.title(f'Average Centroid Distance vs. Feature Dimension ({metric} metric)')
        plt.xlabel('Feature Dimension')
        plt.ylabel('Average Centroid Distance')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        scatter_path = os.path.join(output_dir, f'centroid_distance_vs_dimension_{metric}.png')
        plt.savefig(scatter_path)
        visualizations.append(scatter_path)
        plt.close()
        
        # 4. Comparison of centroid distance and feature distance
        plt.figure(figsize=(10, 6))
        
        plt.scatter(
            metric_df['avg_distance'],
            metric_df['avg_centroid_distance'],
            c=metric_df['feature_dimension'].astype(float),
            cmap='viridis',
            alpha=0.7
        )
        
        plt.title(f'Centroid Distance vs. Feature Distance ({metric} metric)')
        plt.xlabel('Average Feature Distance')
        plt.ylabel('Average Centroid Distance')
        plt.colorbar(label='Feature Dimension')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        compare_path = os.path.join(output_dir, f'centroid_vs_feature_distance_{metric}.png')
        plt.savefig(compare_path)
        visualizations.append(compare_path)
        plt.close()
    
    # 5. Create a comprehensive comparison table
    table_df = results_df.pivot_table(
        values=['avg_centroid_distance', 'std_centroid_distance', 'avg_distance'],
        index=['model_type', 'function_type', 'feature_dimension'],
        columns=['metric'],
        aggfunc='mean'
    ).round(4)
    
    # Save table as CSV
    table_path = os.path.join(output_dir, 'model_comparison_table.csv')
    table_df.to_csv(table_path)
    
    # Also save as Excel for more formatting options
    excel_path = os.path.join(output_dir, 'model_comparison_table.xlsx')
    table_df.to_excel(excel_path)
    
    visualizations.append(table_path)
    visualizations.append(excel_path)
    
    # Save full results
    full_results_path = os.path.join(output_dir, 'full_results.csv')
    results_df.to_csv(full_results_path, index=False)
    visualizations.append(full_results_path)
    
    return visualizations

def main():
    parser = argparse.ArgumentParser(description='Analyze centroid distances of trained models')
    parser.add_argument('--base_dir', type=str, default='models',
                      help='Base directory containing models')
    parser.add_argument('--output_dir', type=str, default='model_comparison',
                      help='Directory to save comparison results')
    parser.add_argument('--metrics', type=str, nargs='+', default=['cosine', 'euclidean'],
                      help='Distance metrics to use')
    parser.add_argument('--num_clusters', type=int, default=10,
                      help='Number of clusters for centroid computation')
    parser.add_argument('--device', type=str, default='cpu',
                      help='Device to use for model loading (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Find all models
    print(f"Searching for models in {args.base_dir}...")
    models_dict = find_all_models(args.base_dir)
    
    if not models_dict:
        print("No models found. Check the base directory path.")
        return
    
    # Analyze model distances
    print(f"Analyzing models using metrics: {args.metrics}")
    results = analyze_models_distances(models_dict, metrics=args.metrics, num_clusters=args.num_clusters)
    
    # Create visualizations
    print(f"Creating comparison visualizations in {args.output_dir}...")
    visualizations = create_comparison_visualizations(results, output_dir=args.output_dir)
    
    print(f"Analysis complete. Generated {len(visualizations)} visualizations and reports.")
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()