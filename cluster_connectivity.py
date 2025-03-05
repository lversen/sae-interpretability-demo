import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from pathlib import Path

def load_gexf_graph(file_path):
    """Load a GEXF graph file using NetworkX"""
    return nx.read_gexf(file_path)

def calculate_connectivity_matrix(graph, cluster_attribute='group'):
    """
    Calculate connectivity matrix between clusters.
    
    Connectivity(C_i, C_j) = \frac{\sum_{i \in C_i, j \in C_j} w_{ij}}{\sum w_{ij}}
    
    where w_{ij} is the weight of the edge from node i to node j.
    
    Args:
        graph: NetworkX graph
        cluster_attribute: Node attribute name for cluster/group
        
    Returns:
        connectivity_matrix: 2D numpy array with connectivity between clusters
        clusters: List of unique cluster IDs
        cluster_sizes: Dictionary mapping cluster IDs to their sizes
    """
    # Extract node to cluster mapping
    node_to_cluster = nx.get_node_attributes(graph, cluster_attribute)
    
    # If no clusters found, try alternative attribute names
    if not node_to_cluster:
        potential_attributes = ['group', 'cluster', 'community', 'module', 'label', 'category']
        for attr in potential_attributes:
            node_to_cluster = nx.get_node_attributes(graph, attr)
            if node_to_cluster:
                print(f"Using '{attr}' as cluster attribute")
                break
    
    # If still no clusters found, assign all nodes to one cluster
    if not node_to_cluster:
        print("No cluster attribute found, treating all nodes as one cluster")
        node_to_cluster = {node: "cluster_0" for node in graph.nodes()}
    
    # Get unique clusters and count nodes per cluster
    clusters = sorted(set(node_to_cluster.values()))
    cluster_sizes = {c: 0 for c in clusters}
    for node, cluster in node_to_cluster.items():
        cluster_sizes[cluster] += 1
    
    num_clusters = len(clusters)
    
    # Create mapping from cluster ID to index
    cluster_to_idx = {c: i for i, c in enumerate(clusters)}
    
    # Initialize numerator matrix (sum of weights between clusters)
    numerator = np.zeros((num_clusters, num_clusters))
    
    # Calculate total weight in the graph (denominator)
    total_weight = sum(data.get('weight', 1) for _, _, data in graph.edges(data=True))
    
    # Calculate connectivity between clusters
    for src, dst, data in graph.edges(data=True):
        if src in node_to_cluster and dst in node_to_cluster:
            src_cluster = node_to_cluster[src]
            dst_cluster = node_to_cluster[dst]
            weight = data.get('weight', 1)
            
            src_idx = cluster_to_idx[src_cluster]
            dst_idx = cluster_to_idx[dst_cluster]
            
            # Add weight to numerator
            numerator[src_idx, dst_idx] += weight
    
    # Calculate connectivity by dividing by total weight
    connectivity_matrix = numerator / total_weight
    
    return connectivity_matrix, clusters, cluster_sizes

def plot_connectivity_matrix(connectivity_matrix, clusters, cluster_sizes, title='Cluster Connectivity'):
    """
    Plot the connectivity matrix as a heatmap.
    
    Args:
        connectivity_matrix: 2D numpy array with connectivity values
        clusters: List of cluster labels
        cluster_sizes: Dictionary mapping cluster IDs to their sizes
        title: Title for the plot
    """
    # Create labels with cluster size information
    labels = [f"{c} (n={cluster_sizes[c]})" for c in clusters]
    
    fig, ax = plt.figure(figsize=(12, 10)), plt.gca()
    im = ax.imshow(connectivity_matrix, cmap='viridis')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Connectivity')
    
    # Set title and labels
    ax.set_title(title, fontsize=14)
    ax.set_xticks(np.arange(len(clusters)))
    ax.set_yticks(np.arange(len(clusters)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Destination Cluster', fontsize=12)
    ax.set_ylabel('Source Cluster', fontsize=12)
    
    # Add text annotations with connectivity values
    for i in range(len(clusters)):
        for j in range(len(clusters)):
            value = connectivity_matrix[i, j]
            text_color = 'white' if value > 0.5 else 'black'
            if value > 0.01:  # Only show values above a threshold for readability
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", 
                        color=text_color, fontsize=8)
    
    plt.tight_layout()
    return fig

def generate_summary_stats(connectivity_matrix, clusters, cluster_sizes):
    """
    Generate summary statistics for the connectivity matrix.
    
    Args:
        connectivity_matrix: 2D numpy array with connectivity values
        clusters: List of cluster labels
        cluster_sizes: Dictionary mapping cluster IDs to their sizes
    
    Returns:
        stats_df: DataFrame with summary statistics
    """
    # Calculate statistics
    stats = {
        'Cluster': clusters,
        'Size': [cluster_sizes[c] for c in clusters],
        'Self-connectivity': [connectivity_matrix[i, i] for i in range(len(clusters))],
        'Out-connectivity': [np.sum(connectivity_matrix[i, :]) - connectivity_matrix[i, i] for i in range(len(clusters))],
        'In-connectivity': [np.sum(connectivity_matrix[:, i]) - connectivity_matrix[i, i] for i in range(len(clusters))],
        'Total-connectivity': [np.sum(connectivity_matrix[i, :]) + np.sum(connectivity_matrix[:, i]) - 2*connectivity_matrix[i, i] for i in range(len(clusters))]
    }
    
    # Create DataFrame
    stats_df = pd.DataFrame(stats)
    
    # Add global statistics
    global_stats = pd.DataFrame({
        'Metric': ['Total clusters', 'Average cluster size', 'Max self-connectivity', 'Min self-connectivity', 
                  'Max connectivity value', 'Average connectivity', 'Total connectivity sum'],
        'Value': [
            len(clusters),
            np.mean(list(cluster_sizes.values())),
            np.max(np.diag(connectivity_matrix)),
            np.min(np.diag(connectivity_matrix)),
            np.max(connectivity_matrix),
            np.mean(connectivity_matrix),
            np.sum(connectivity_matrix)
        ]
    })
    
    return stats_df, global_stats

def process_graph(file_path, output_dir=None):
    """
    Process a single GEXF graph file.
    
    Args:
        file_path: Path to the GEXF graph file
        output_dir: Directory to save results (default: same as input file)
        
    Returns:
        connectivity_matrix: The computed connectivity matrix
        clusters: List of cluster labels
        cluster_sizes: Dictionary mapping cluster IDs to their sizes
    """
    # Create output directory if it doesn't exist
    if output_dir is None:
        output_dir = os.path.dirname(file_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base filename without extension
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    print(f"Processing {base_name}...")
    
    # Load graph
    graph = load_gexf_graph(file_path)
    
    # Print basic graph info
    print(f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    
    # Calculate connectivity matrix
    connectivity_matrix, clusters, cluster_sizes = calculate_connectivity_matrix(graph)
    
    # Plot matrix
    fig = plot_connectivity_matrix(
        connectivity_matrix, 
        clusters, 
        cluster_sizes, 
        title=f'Connectivity Matrix - {base_name}'
    )
    
    # Save plot
    plot_path = os.path.join(output_dir, f"{base_name}_connectivity_matrix.png")
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved connectivity matrix plot to {plot_path}")
    
    # Generate and save statistics
    stats_df, global_stats = generate_summary_stats(connectivity_matrix, clusters, cluster_sizes)
    
    # Save statistics to CSV
    stats_path = os.path.join(output_dir, f"{base_name}_connectivity_stats.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved cluster statistics to {stats_path}")
    
    # Save global statistics to CSV
    global_stats_path = os.path.join(output_dir, f"{base_name}_global_stats.csv")
    global_stats.to_csv(global_stats_path, index=False)
    print(f"Saved global statistics to {global_stats_path}")
    
    # Save raw connectivity matrix to CSV
    matrix_path = os.path.join(output_dir, f"{base_name}_connectivity_matrix.csv")
    matrix_df = pd.DataFrame(
        connectivity_matrix,
        index=[f"{c} (n={cluster_sizes[c]})" for c in clusters],
        columns=[f"{c} (n={cluster_sizes[c]})" for c in clusters]
    )
    matrix_df.to_csv(matrix_path)
    print(f"Saved raw connectivity matrix to {matrix_path}")
    
    return connectivity_matrix, clusters, cluster_sizes

def process_all_graphs(graph_dir, output_dir=None):
    """
    Process all GEXF graphs in a directory.
    
    Args:
        graph_dir: Directory containing GEXF graph files
        output_dir: Directory to save results (default: graph_dir/results)
    """
    # Set default output directory
    if output_dir is None:
        output_dir = os.path.join(graph_dir, "results")
    
    # Get all GEXF files
    gexf_files = [f for f in os.listdir(graph_dir) if f.endswith('.gexf')]
    
    if not gexf_files:
        print(f"No GEXF files found in {graph_dir}")
        return
    
    print(f"Found {len(gexf_files)} GEXF files to process")
    
    # Process each file
    for gexf_file in gexf_files:
        file_path = os.path.join(graph_dir, gexf_file)
        process_graph(file_path, output_dir)
        print("-" * 50)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate connectivity matrices between clusters for graph files')
    parser.add_argument('--input', '-i', type=str, help='Input GEXF file or directory containing GEXF files')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output directory (default: same as input)')
    
    args = parser.parse_args()
    
    if args.input:
        if os.path.isdir(args.input):
            process_all_graphs(args.input, args.output)
        elif os.path.isfile(args.input) and args.input.endswith('.gexf'):
            process_graph(args.input, args.output)
        else:
            print(f"Input {args.input} is not a valid GEXF file or directory")
    else:
        # Look for a graphs directory in the current working directory
        cwd = os.getcwd()
        potential_dirs = ['graphs', 'graph', 'gephi', 'gexf']
        found = False
        
        for dir_name in potential_dirs:
            dir_path = os.path.join(cwd, dir_name)
            if os.path.isdir(dir_path):
                found = True
                print(f"Found graph directory: {dir_path}")
                process_all_graphs(dir_path, args.output)
                break
        
        if not found:
            print("No input specified and no default graph directory found.")
            print("Usage: python cluster_connectivity.py --input path/to/file_or_directory [--output path/to/output_dir]")