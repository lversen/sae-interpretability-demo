import networkx as nx
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph
import warnings
def node_attributes(df: pd.DataFrame, title_column: str, model) -> Tuple[Dict[int, str], Dict[str, Dict[str, Any]]]:
    print("Creating node mapping and attributes")
    if title_column not in df.columns:
        raise ValueError(f"{title_column} is not a column in the DataFrame")
    
    # Check for duplicate values in the title column
    if df[title_column].duplicated().any():
        warnings.warn(f"Duplicate values found in {title_column}. Adding unique suffixes to ensure uniqueness.")
        df[title_column] = df[title_column] + '_' + df.groupby(title_column).cumcount().astype(str)
    
    df2 = df
    df2[title_column] = df2[title_column].to_numpy() + np.repeat(model, len(df))
    mapping = dict(enumerate(df2[title_column]))
    
    # Use reset_index to ensure unique index for to_dict
    df["Model Name"] = np.repeat(model, len(df))
    attributes = df.reset_index(drop=True).set_index(title_column).to_dict('index')
    print("Mapping and attributes completed")
    return mapping, attributes

def gephi(feature_extract: np.ndarray, file_name: str, model_name: str, mapping: Dict[int, str], attributes: Dict[str, Dict[str, Any]]):
    nn = kneighbors_graph(feature_extract, n_neighbors=4, mode="distance", metric="l1")
    nn.data = np.exp(-nn.data**2 / np.mean(nn.data)**2)
    
    G = nx.DiGraph(nn)
    H = nx.relabel_nodes(G, mapping)
    nx.set_node_attributes(H, attributes)
    
    output_file = f"{file_name}_{model_name.replace('/', '_')}.gexf"
    nx.write_gexf(H, output_file)
    print(f"Graph exported to {output_file}")

def gephi_export(feature_extract: np.ndarray, file_name: str, model_name: str, mapping: Dict[int, str], attributes: Dict[str, Dict[str, Any]]):
    file_name = file_name.replace(".csv", "").replace("data\\", "")
    model_name = model_name.replace("/", "_")
    gephi(feature_extract, file_name, model_name, mapping, attributes)
