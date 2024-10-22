import networkx as nx
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph
import warnings
import os
import random

def filter_by_random_labels(df: pd.DataFrame, 
                          title_column: str, 
                          n_random_labels: Optional[int] = None,
                          category_column: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[List[str]]]:
    """
    Filter DataFrame to include only n randomly selected labels.
    Returns filtered DataFrame and list of selected labels.
    """
    if n_random_labels is None:
        return df, None
        
    if category_column and category_column in df.columns:
        all_labels = df[category_column].unique()
    else:
        all_labels = df[title_column].unique()
        
    if n_random_labels > len(all_labels):
        warnings.warn(f"Requested {n_random_labels} labels but only {len(all_labels)} available")
        selected_labels = all_labels
    else:
        selected_labels = np.random.choice(all_labels, size=n_random_labels, replace=False)
        
    if category_column and category_column in df.columns:
        filtered_df = df[df[category_column].isin(selected_labels)]
    else:
        filtered_df = df[df[title_column].isin(selected_labels)]
        
    print(f"Selected {len(selected_labels)} random labels: {selected_labels}")
    return filtered_df, selected_labels

def node_attributes(df: pd.DataFrame, 
                   title_column: str, 
                   model: str, 
                   category_column: Optional[str] = None,
                   n_random_labels: Optional[int] = None) -> Tuple[Dict[int, str], Dict[str, Dict[str, Any]]]:
    """
    Create node mapping and attributes, optionally filtering by random labels.
    """
    print("Creating node mapping and attributes")
    if title_column not in df.columns:
        raise ValueError(f"{title_column} is not a column in the DataFrame")
    
    # Filter by random labels if specified
    filtered_df, selected_labels = filter_by_random_labels(
        df, title_column, n_random_labels, category_column)
    
    # Check for duplicate values in the title column
    if filtered_df[title_column].duplicated().any():
        warnings.warn(f"Duplicate values found in {title_column}. Adding unique suffixes to ensure uniqueness.")
        filtered_df[title_column] = filtered_df[title_column] + '_' + filtered_df.groupby(title_column).cumcount().astype(str)
    
    # Create a new column for the combined title and model
    combined_title = filtered_df[title_column].astype(str) + ' ' + model
    
    # Create the mapping
    mapping = dict(enumerate(combined_title))
    
    # Create attributes dictionary
    attributes = {}
    for index, row in filtered_df.iterrows():
        key = f"{row[title_column]} {model}"
        attributes[key] = row.to_dict()
        attributes[key]['Model Name'] = model
        
    print("Mapping and attributes completed")
    return mapping, attributes

def gephi(feature_extract: np.ndarray, 
          file_path: str, 
          model_name: str, 
          mapping: Dict[int, str], 
          attributes: Dict[str, Dict[str, Any]],
          n_neighbors: int = 4):
    """
    Create and export Gephi graph with specified number of neighbors.
    """
    nn = kneighbors_graph(feature_extract, n_neighbors=n_neighbors, mode="distance", metric="l1")
    nn.data = np.exp(-nn.data**2 / np.mean(nn.data)**2)
    
    G = nx.DiGraph(nn)
    H = nx.relabel_nodes(G, mapping)
    nx.set_node_attributes(H, attributes)
    
    nx.write_gexf(H, file_path)
    print(f"Graph exported to {file_path}")

def gephi_export(feature_extract: np.ndarray, 
                file_name: str, 
                model_name: str, 
                mapping: Dict[int, str], 
                attributes: Dict[str, Dict[str, Any]],
                n_neighbors: int = 4):
    """
    Export Gephi graph to a dedicated folder.
    """
    # Extract the base file name without extension and path
    base_file_name = os.path.splitext(os.path.basename(file_name))[0]
    
    # Create a folder with the same name as the input file
    folder_name = f"gephi_exports_{base_file_name}"
    os.makedirs(folder_name, exist_ok=True)
    
    # Clean up the model name
    model_name = model_name.replace("/", "_")
    
    # Create the output file path
    output_file = os.path.join(folder_name, f"{base_file_name}_{model_name}.gexf")
    
    # Call the gephi function with the new file path
    gephi(feature_extract, output_file, model_name, mapping, attributes, n_neighbors=n_neighbors)