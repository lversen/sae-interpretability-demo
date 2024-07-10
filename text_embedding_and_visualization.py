import os
import pickle
import warnings
from functools import lru_cache
from typing import List, Dict, Any, Tuple

# Third-party library imports
import numpy as np
import pandas as pd
import torch
import networkx as nx
from community import community_louvain
from sklearn.neighbors import kneighbors_graph
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.patches as mpatches
from matplotlib.patches import Wedge, Circle
import matplotlib.colors as mcolors

# Transformers and related imports
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

# LangChain imports
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings

from visualize_multi_category_graph import visualize_multi_category_graph
from feature_extraction_with_store import feature_extraction_with_store
from classify_with_networkx import classify_with_networkx
@lru_cache(maxsize=None)
def get_classifier(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return TextClassificationPipeline(model=model, tokenizer=tokenizer, device="cuda")

def language_classifier(df: pd.DataFrame, rows: np.ndarray, max_rows: int, columns: List[str], file_name: str) -> None:
    model_name = 'qanastek/51-languages-classifier'
    classifier = get_classifier(model_name)
    df_max = pd.read_csv(file_name)

    for c in columns:
        c_classified = f"{c}_classified"
        if c not in df.columns:
            raise ValueError(f"{c} is not a column in {file_name}")
        
        if c_classified not in df.columns:
            print(f"Classifying languages from '{c}'")
            data = classify_data(classifier, df[c])
            update_dataframes(df, df_max, c_classified, data, rows, max_rows)
        else:
            update_remaining_rows(df, df_max, c, c_classified, classifier, rows)

        df_max.to_csv(file_name, index=False)

def classify_data(classifier: TextClassificationPipeline, data: pd.Series) -> List[str]:
    return [d["label"] for d in classifier(list(data.to_numpy()))]

def update_dataframes(df: pd.DataFrame, df_max: pd.DataFrame, column: str, data: List[str], rows: np.ndarray, max_rows: int) -> None:
    df[column] = data
    data_max = np.full(max_rows, "empty", dtype='U100')
    data_max[rows] = data
    df_max[column] = data_max

def update_remaining_rows(df: pd.DataFrame, df_max: pd.DataFrame, c: str, c_classified: str, classifier: TextClassificationPipeline, rows: np.ndarray) -> None:
    mask = df[c_classified] == "empty"
    rows_remaining = df.index[mask]
    if len(rows_remaining) > 0:
        print(f"{len(rows_remaining)} rows have not been classified before")
        data_remaining = classify_data(classifier, df.loc[rows_remaining, c])
        df.loc[rows_remaining, c_classified] = data_remaining
        df_max.loc[rows[rows_remaining], c_classified] = data_remaining

def preprocess(df: pd.DataFrame, file_name: str, content_column: List[str], dataset_iteration: int, n: int) -> Tuple[pd.DataFrame, np.ndarray, int]:
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    
    if content_column:
        data_name = content_column[dataset_iteration]
        if data_name not in df.columns:
            raise ValueError(f"{data_name} is not a column in {file_name}")
        
        df = df[[data_name] + [col for col in df.columns if col != data_name]]
    else:
        data_name = df.select_dtypes(include=['object']).apply(lambda x: x.str.len().sum()).idxmax()
        df = df[[data_name] + [col for col in df.columns if col != data_name]]
    
    max_rows = len(df)
    rows = np.random.choice(df.index, size=min(n, max_rows), replace=False)
    df = df.loc[rows].reset_index(drop=True)
    df[df.columns[0]] = df[df.columns[0]].str.strip()
    
    return df, rows, max_rows

def preprocess_duplicates(df: pd.DataFrame, title_column: str, method: str = 'suffix') -> pd.DataFrame:
    """
    Preprocess the DataFrame to handle duplicate values in the title column.
    
    :param df: Input DataFrame
    :param title_column: Name of the column to check for duplicates
    :param method: Method to handle duplicates. Options: 'suffix', 'remove', 'concatenate'
    :return: Preprocessed DataFrame
    """
    if method == 'suffix':
        # Current behavior: add suffixes to duplicates
        if df[title_column].duplicated().any():
            df[title_column] = df[title_column] + '_' + df.groupby(title_column).cumcount().astype(str)
    elif method == 'remove':
        # Remove duplicate entries
        df = df.drop_duplicates(subset=[title_column])
    elif method == 'concatenate':
        # Concatenate with another column (e.g., 'ID') to make unique
        if 'ID' in df.columns:
            df[title_column] = df[title_column] + '_' + df['ID'].astype(str)
        else:
            raise ValueError("'ID' column not found for concatenation method")
    else:
        raise ValueError("Invalid method. Choose 'suffix', 'remove', or 'concatenate'")
    
    return df

def data_frame_init(file_name: str, content_column: List[str], dataset_iteration: int, n: int) -> Tuple[pd.DataFrame, np.ndarray, int]:
    df = pd.read_csv(file_name)
    return preprocess(df, file_name, content_column, dataset_iteration, n)

def node_attributes(df: pd.DataFrame, title_column: str) -> Tuple[Dict[int, str], Dict[str, Dict[str, Any]]]:
    if title_column not in df.columns:
        raise ValueError(f"{title_column} is not a column in the DataFrame")
    
    # Check for duplicate values in the title column
    if df[title_column].duplicated().any():
        warnings.warn(f"Duplicate values found in {title_column}. Adding unique suffixes to ensure uniqueness.")
        df[title_column] = df[title_column] + '_' + df.groupby(title_column).cumcount().astype(str)
    
    mapping = dict(enumerate(df[title_column]))
    
    # Use reset_index to ensure unique index for to_dict
    attributes = df.reset_index(drop=True).set_index(title_column).to_dict('index')
    
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


def run_all(
    datasets: List[str],
    models: List[str],
    n: int,
    content_column: List[str],
    id_column: List[str],
    category_column: List[str],
    n_neighbors: int = 5,
    max_categories: int = 10,
    batch_size: int = 32,
    create_graph: bool = False,
    use_networkx_classification: bool = False,
    force_new_embeddings: bool = False,
    embeddings_only: bool = False
) -> Dict[str, Dict[str, Any]]:
    model_dict = {}
    
    for i, dataset in enumerate(datasets):
        print(f"Processing dataset: {dataset}")
        
        # Load full dataset
        full_df = pd.read_csv(dataset)
        
        for model in models:
            print(f"Processing model: {model}")
            
            # Get consistent samples
            df, selected_indices = get_consistent_samples(full_df, n, dataset, model)
            
            # Feature extraction
            feature_extract, vectorstore = feature_extraction_with_store(
                full_df, model, batch_size, n, dataset, content_column[i], 
                force_new_embeddings=force_new_embeddings,
                embeddings_only=embeddings_only
            )
            
            if model not in model_dict:
                model_dict[model] = {}
            if dataset not in model_dict[model]:
                model_dict[model][dataset] = {}
            
            model_dict[model][dataset]['feature_extract'] = feature_extract
            model_dict[model][dataset]['vectorstore'] = vectorstore
            
            if use_networkx_classification:
                print(f"Performing NetworkX classification for {dataset} with model {model}")
                df, graph = classify_with_networkx(
                    feature_extract, df, id_column[i], 
                    category_column[i], vectorstore if not embeddings_only else None, 
                    n_neighbors=n_neighbors
                )
                model_dict[model][dataset]['networkx_graph'] = graph
                model_dict[model][dataset]['classified_df'] = df
                
                if create_graph:
                    # Visualize the multi-category graph
                    output_file = f"multi_category_graph_{dataset}_{model}_n{n}.png".replace('/', '_')
                    visualize_multi_category_graph(graph, df, id_column[i], output_file, max_categories=max_categories)
                    print(f"Multi-category graph saved as {output_file}")
            
    return model_dict

def get_consistent_samples(df: pd.DataFrame, n: int, dataset_name: str, model: str, max_n: int = None):
    base_dir = os.path.join("vectorstores_and_embeddings", dataset_name, model.replace('/', '_'))
    os.makedirs(base_dir, exist_ok=True)
    index_path = os.path.join(base_dir, f"index_max.pkl")
    
    if os.path.exists(index_path):
        with open(index_path, 'rb') as f:
            all_indices = pickle.load(f)
    else:
        max_n = max_n or len(df)
        all_indices = np.random.permutation(df.index)[:max_n]
        with open(index_path, 'wb') as f:
            pickle.dump(all_indices, f)
    
    selected_indices = all_indices[:n]
    return df.loc[selected_indices].reset_index(drop=True), selected_indices
