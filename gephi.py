import networkx as nx
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph
import warnings
import os

def node_attributes(df: pd.DataFrame, title_column: str, model: str, category_column: Optional[str] = None) -> Tuple[Dict[int, str], Dict[str, Dict[str, Any]]]:
    print("Creating node mapping and attributes")
    if title_column not in df.columns:
        raise ValueError(f"{title_column} is not a column in the DataFrame")
    
    # Check for duplicate values in the title column
    if df[title_column].duplicated().any():
        warnings.warn(f"Duplicate values found in {title_column}. Adding unique suffixes to ensure uniqueness.")
        df[title_column] = df[title_column] + '_' + df.groupby(title_column).cumcount().astype(str)
    
    # Create a new column for the combined title and model
    combined_title = df[title_column].astype(str) + ' ' + model
    
    # Create the mapping
    mapping = dict(enumerate(combined_title))
    
    # Create attributes dictionary
    attributes = {}
    for index, row in df.iterrows():
        key = f"{row[title_column]} {model}"
        attributes[key] = row.to_dict()
        attributes[key]['Model Name'] = model
        
    print("Mapping and attributes completed")
    return mapping, attributes

def gephi(feature_extract: np.ndarray, file_path: str, model_name: str, mapping: Dict[int, str], attributes: Dict[str, Dict[str, Any]]):
    nn = kneighbors_graph(feature_extract, n_neighbors=4, mode="distance", metric="l1")
    nn.data = np.exp(-nn.data**2 / np.mean(nn.data)**2)
    
    G = nx.DiGraph(nn)
    H = nx.relabel_nodes(G, mapping)
    nx.set_node_attributes(H, attributes)
    
    nx.write_gexf(H, file_path)
    print(f"Graph exported to {file_path}")

def gephi_export(feature_extract: np.ndarray, file_name: str, model_name: str, mapping: Dict[int, str], attributes: Dict[str, Dict[str, Any]]):
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
    gephi(feature_extract, output_file, model_name, mapping, attributes)