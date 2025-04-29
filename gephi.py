import networkx as nx
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph
import warnings
import os
import random
import torch

def ensure_numpy_array(data):
    """
    Ensure the input data is a numpy array, converting from torch tensor if necessary.
    """
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)

def sanitize_name(name: str) -> str:
    """
    Sanitize file names and model names by replacing dots and other problematic characters with underscores or hyphens.
    """
    # Replace dots with hyphens (for version numbers)
    name = name.replace('.', '-')
    # Replace other problematic characters with underscores
    name = name.replace('/', '_').replace('\\', '_')
    return name

def select_random_labels(df: pd.DataFrame, 
                        title_column: str, 
                        n_random_labels: Optional[int] = None,
                        category_column: Optional[str] = None) -> List[str]:
    """
    Select random labels that have multiple occurrences.
    Returns list of selected labels.
    """
    if n_random_labels is None:
        return None
        
    # Determine which column to use for grouping
    grouping_column = category_column if category_column and category_column in df.columns else title_column
    
    # Count occurrences of each label
    label_counts = df[grouping_column].value_counts()
    
    # Only select from labels that have multiple occurrences
    valid_labels = label_counts[label_counts > 1].index.tolist()
    
    if not valid_labels:
        warnings.warn("No labels with multiple occurrences found")
        return None
        
    if n_random_labels > len(valid_labels):
        warnings.warn(f"Requested {n_random_labels} labels but only {len(valid_labels)} valid labels available")
        selected_labels = valid_labels
    else:
        selected_labels = np.random.choice(valid_labels, size=n_random_labels, replace=False)
    
    return selected_labels.tolist()

def filter_by_labels(df: pd.DataFrame, 
                    feature_extract: np.ndarray,
                    labels: List[str],
                    grouping_column: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Filter DataFrame and corresponding feature embeddings by given labels.
    """
    mask = df[grouping_column].isin(labels)
    filtered_df = df[mask].copy()  # Create an explicit copy
    filtered_feature_extract = feature_extract[mask]
    return filtered_df, filtered_feature_extract

def create_node_attributes(df: pd.DataFrame,
                         feature_extract: np.ndarray,
                         title_column: str, 
                         model: str, 
                         selected_labels: Optional[List[str]] = None,
                         category_column: Optional[str] = None) -> Tuple[Dict[int, str], Dict[str, Dict[str, Any]], np.ndarray]:
    """
    Create node mapping and attributes using pre-selected labels.
    Returns mapping, attributes, and filtered feature embeddings.
    """
    # Ensure feature_extract is a numpy array
    feature_extract = ensure_numpy_array(feature_extract)
    
    if title_column not in df.columns:
        raise ValueError(f"{title_column} is not a column in the DataFrame")
    
    # Sanitize model name
    model = sanitize_name(model)
    
    # Filter by pre-selected labels if provided
    if selected_labels:
        grouping_column = category_column if category_column and category_column in df.columns else title_column
        filtered_df, filtered_feature_extract = filter_by_labels(df, feature_extract, selected_labels, grouping_column)
    else:
        filtered_df = df.copy()  # Create an explicit copy
        filtered_feature_extract = feature_extract
    
    # Create unique node IDs while preserving label information
    if category_column and category_column in filtered_df.columns:
        filtered_df.loc[:, 'node_id'] = filtered_df.groupby(category_column).cumcount().astype(str)
        filtered_df.loc[:, 'display_title'] = filtered_df[category_column].astype(str) + '_node_' + filtered_df['node_id']
    else:
        filtered_df.loc[:, 'node_id'] = filtered_df.groupby(title_column).cumcount().astype(str)
        filtered_df.loc[:, 'display_title'] = filtered_df[title_column].astype(str) + '_node_' + filtered_df['node_id']
    
    # Create mapping and attributes
    combined_title = filtered_df['display_title'].astype(str) + ' ' + model
    mapping = dict(enumerate(combined_title))
    
    attributes = {}
    for index, row in filtered_df.iterrows():
        key = f"{row['display_title']} {model}"
        attributes[key] = row.to_dict()
        attributes[key]['Model Name'] = model
        attributes[key]['group'] = row[category_column] if category_column and category_column in filtered_df.columns else row[title_column]
    
    return mapping, attributes, filtered_feature_extract

def create_gephi_graph(feature_extract: np.ndarray, 
                      df: pd.DataFrame,
                      title_column: str,
                      model_name: str,
                      file_path: str,
                      selected_labels: Optional[List[str]] = None,
                      category_column: Optional[str] = None,
                      n_neighbors: int = 4,
                      min_edge_weight: float = 1e-10,
                      distance_mode: str = "l1"):  # Add minimum edge weight threshold
    """
    Create and export Gephi graph using pre-selected labels.
    
    Args:
        ...existing args...
        min_edge_weight: Minimum edge weight to include in the graph (default: 1e-10)
    """
    # Sanitize model name (but not file extension)
    model_name = sanitize_name(model_name)
    
    # Create node attributes using the same selected labels
    mapping, attributes, filtered_feature_extract = create_node_attributes(
        df, feature_extract, title_column, model_name, selected_labels, category_column
    )
    
    # Create the k-nearest neighbors graph
    nn = kneighbors_graph(filtered_feature_extract, n_neighbors=n_neighbors, mode="distance", metric=distance_mode, include_self="auto", n_jobs=-1)
    
    # Convert distances to weights
    nn.data = np.exp(-nn.data**2 / np.mean(nn.data)**2)
    
    # Create NetworkX graph
    G = nx.DiGraph(nn)
    H = nx.relabel_nodes(G, mapping)
    nx.set_node_attributes(H, attributes)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Split the file path into directory and filename
    dir_name, file_name = os.path.split(file_path)
    base_name, ext = os.path.splitext(file_name)
    sanitized_base_name = sanitize_name(base_name)
    sanitized_file_path = os.path.join(dir_name, sanitized_base_name + ext)
    
    # Export the graph
    nx.write_gexf(H, sanitized_file_path)
    print(f"Graph exported to {sanitized_file_path}")
    
    # Print graph statistics
    n_edges = H.number_of_edges()
    n_nodes = H.number_of_nodes()
    print(f"Graph statistics:")
    print(f"Number of nodes: {n_nodes}")
    print(f"Number of edges: {n_edges}")
    print(f"Average degree: {n_edges/n_nodes:.2f}")