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

def batch_encode(model: SentenceTransformer, feature_list: List[str], batch_size: int, n: int) -> np.ndarray:
    embeddings = []
    total_batches = (len(feature_list) + batch_size - 1) // batch_size
    
    for i in range(0, len(feature_list), batch_size):
        batch = feature_list[i:i+batch_size]
        batch_embeddings = model.encode(batch, device='cuda', show_progress_bar=False)
        embeddings.extend(batch_embeddings)
        
        if i > 0:
            progress = (i + batch_size) / len(feature_list)
            print(f"\rEncoding progress: {progress:.1%}", end="", flush=True)
        
        torch.cuda.empty_cache()
    
    print("\nEncoding complete.")
    return np.array(embeddings)

def feature_extraction(df: pd.DataFrame, model_name: str, batch_size: int, n: int) -> np.ndarray:
    feature_list = df[df.columns[0]].tolist()
    model = SentenceTransformer(model_name, device='cuda', trust_remote_code=True)
    
    with torch.no_grad():
        feature_extract = batch_encode(model, feature_list, batch_size, n)
    
    print(f"Feature extraction shape: {feature_extract.shape}")
    return feature_extract

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


def feature_extraction_with_store(
    df: pd.DataFrame, 
    model: str, 
    batch_size: int, 
    n: int, 
    dataset_name: str, 
    content_column: str, 
    force_new_embeddings: bool = False,
    embeddings_only: bool = False
):
    # Create a directory for storing vectorstores and embeddings
    storage_dir = os.path.join("vectorstores_and_embeddings", dataset_name)
    
    vectorstore_path = os.path.join(storage_dir, f"vectorstore_{model.replace('/', '_')}_n{n}.pkl")
    embeddings_path = os.path.join(storage_dir, f"embeddings_{model.replace('/', '_')}_n{n}.npy")
    
    # Create the full path to the directory
    os.makedirs(os.path.dirname(vectorstore_path), exist_ok=True)
    
    if os.path.exists(embeddings_path) and not force_new_embeddings:
        print(f"Loading existing embeddings for {dataset_name} with model {model}")
        feature_extract = np.load(embeddings_path)
        
        # Check if the number of embeddings matches the current dataframe
        if len(feature_extract) != len(df):
            print("Number of stored embeddings doesn't match current data. Recomputing embeddings.")
            force_new_embeddings = True
        elif embeddings_only:
            return feature_extract, None
    
    if not embeddings_only and os.path.exists(vectorstore_path) and not force_new_embeddings:
        print(f"Loading existing vector store for {dataset_name} with model {model}")
        with open(vectorstore_path, "rb") as f:
            vectorstore = pickle.load(f)
    else:
        vectorstore = None
    
    if force_new_embeddings or not os.path.exists(embeddings_path):
        print(f"Creating new embeddings for {dataset_name} with model {model}")
        embeddings = HuggingFaceEmbeddings(model_name=model)
        texts = df[content_column].tolist()
        feature_extract = embeddings.embed_documents(texts)
        
        # Save embeddings
        np.save(embeddings_path, feature_extract)
        
        if not embeddings_only:
            print(f"Creating new vector store for {dataset_name} with model {model}")
            vectorstore = FAISS.from_texts(texts, embeddings)
            
            # Save vectorstore
            vectorstore.save_local(vectorstore_path)
    
    return feature_extract, vectorstore
def classify_with_networkx(feature_extract, df, id_column, category_column, vectorstore=None, n_neighbors=5):
    G = nx.Graph()
    
    # Add nodes with IDs and categories as attributes
    for i, (id_value, categories) in enumerate(zip(df[id_column], df[category_column])):
        G.add_node(i, id=id_value, categories=categories.split(',') if isinstance(categories, str) else categories)
    
    if vectorstore:
        # Use vectorstore for nearest neighbor search
        for i, content in enumerate(df[id_column]):  # Using id_column as content for matching
            results = vectorstore.similarity_search(content, k=n_neighbors+1)
            for doc in results[1:]:  # Skip the first result (self)
                # Find the index of the matching content in the dataframe
                matching_indices = df.index[df[id_column] == doc.page_content].tolist()
                if matching_indices:
                    j = matching_indices[0]
                    if i != j:  # Avoid self-loops
                        similarity = 1 - doc.metadata['score'] if 'score' in doc.metadata else 0.5
                        if similarity > 0.5:
                            G.add_edge(i, j, weight=similarity)
    else:
        # Use sklearn NearestNeighbors for embedding-based search
        nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, metric='cosine').fit(feature_extract)
        distances, indices = nbrs.kneighbors(feature_extract)
        
        for i, (dist, idx) in enumerate(zip(distances, indices)):
            for j, d in zip(idx[1:], dist[1:]):  # Skip the first one (self)
                similarity = 1 - d
                if similarity > 0.5:
                    G.add_edge(i, j, weight=similarity)
    
    # Perform community detection
    partition = community_louvain.best_partition(G)
    
    # Add the classification results to the dataframe
    df['networkx_class'] = df.index.map(partition)
    
    return df, G

def visualize_multi_category_graph(G, df, id_column, output_file, max_categories=10, min_edge_weight=0.6):
    # Perform community detection
    partition = community_louvain.best_partition(G)
    
    # Set node colors based on community
    unique_communities = sorted(set(partition.values()))
    community_color_map = plt.cm.tab20
    community_colors = {comm: community_color_map(i/len(unique_communities)) for i, comm in enumerate(unique_communities)}
    node_colors = [community_colors[partition[node]] for node in G.nodes()]
    
    # Create a spring layout with more spread
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Create figure and axes
    fig, (ax, cax) = plt.subplots(1, 2, figsize=(24, 20), gridspec_kw={'width_ratios': [20, 1]})
    
    # Draw edges (only those above the minimum weight)
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights)
    scaled_weights = [(w - min_edge_weight) / (max_weight - min_edge_weight) for w in edge_weights if w >= min_edge_weight]
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2, width=scaled_weights)
    
    # Define a custom color palette for categories
    custom_colors = [
        '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', 
        '#f032e6', '#bfef45', '#fabed4', '#469990', '#dcbeff', '#9A6324', '#fffac8', 
        '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9'
    ]

    # Prepare legend for categories
    all_categories = set()
    for node, data in G.nodes(data=True):
        if 'categories' in data:
            all_categories.update(data['categories'][:max_categories])
    category_counts = Counter(cat for node, data in G.nodes(data=True) 
                              for cat in data.get('categories', [])[:max_categories])
    top_categories = dict(sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10])
    
    # Assign colors to categories
    category_colors = {cat: custom_colors[i % len(custom_colors)] for i, cat in enumerate(top_categories.keys())}

    # Draw nodes
    for node, (x, y) in pos.items():
        # Draw main node circle (community color)
        circle = Circle((x, y), 0.02, facecolor=node_colors[node], edgecolor='white')
        ax.add_patch(circle)
        
        # Draw category pie chart (smaller, inside the main circle)
        if 'categories' in G.nodes[node]:
            categories = G.nodes[node]['categories'][:max_categories]
            category_counts = Counter(categories)
            total = sum(category_counts.values())
            
            start_angle = 0
            for category, count in category_counts.items():
                angle = 360 * count / total
                wedge = Wedge((x, y), 0.012, start_angle, start_angle+angle, 
                              facecolor=category_colors.get(category, '#808080'), edgecolor='none')
                ax.add_patch(wedge)
                start_angle += angle
    
    # Add labels for the most connected nodes
    top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:10]
    labels = {node: G.nodes[node]['id'] for node, degree in top_nodes}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8, font_weight="bold")
    
    # Add a color bar to show community assignments
    sm = plt.cm.ScalarMappable(cmap=community_color_map, norm=plt.Normalize(vmin=0, vmax=len(unique_communities)-1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Community', rotation=270, labelpad=15)
    
    # Create legend for top categories
    legend_elements = []
    for category, count in top_categories.items():
        patch = mpatches.Patch(facecolor=category_colors[category], edgecolor='black', label=f"{category} ({count})")
        legend_elements.append(patch)
    
    # Add the category legend to the plot
    ax.legend(handles=legend_elements, title="Top Categories", loc="upper left", bbox_to_anchor=(1, 1))

    # Add explanation for node colors
    ax.text(0.95, 0.05, "Node colors represent different communities\nInner pie charts show category distribution", 
            transform=ax.transAxes, ha='right', va='bottom', fontsize=10, 
            bbox=dict(facecolor='white', edgecolor='black', alpha=0.8))
    
    ax.set_title("Multi-Category Graph Visualization", fontsize=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Graph visualization saved as {output_file}")

def run_all(
    datasets: List[str],
    models: List[str],
    n: int,
    content_column: List[str],
    id_column: List[str],
    category_column: List[str],
    n_neighbors: int = 5,
    max_categories: int = 10,
    embeddings_only: bool = False,
    classify_language: List[str] = [],
    duplicate_method: str = 'suffix',
    batch_size: int = 16,
    create_graph: bool = False,
    force_new_embeddings: bool = False,
    use_networkx_classification: bool = False,

) -> Dict[str, Dict[str, Any]]:
    model_dict = {}
    
    for i, dataset in enumerate(datasets):
        print(f"Processing dataset: {dataset}")
        
        # Initialize dataframe
        df, rows, max_rows = data_frame_init(dataset, content_column, i, n)
            
        # Preprocess to handle duplicates
        if id_column:
            df = preprocess_duplicates(df, id_column[i], method=duplicate_method)
            
        # Perform language classification if requested
        if classify_language:
            with warnings.catch_warnings(record=True) as w:
                language_classifier(df, rows, max_rows, classify_language, dataset)
                if w:
                    print(f"Warning in language_classifier for {dataset}: {w[0].message}")
        
        # Get node attributes for graph creation
        if create_graph and id_column:
            with warnings.catch_warnings(record=True) as w:
                mapping, attributes = node_attributes(df, id_column[i])
                if w:
                    print(f"Warning in node_attributes for {dataset}: {w[0].message}")
        else:
            mapping, attributes = {}, {}
            
        # Process each model
        for model in models:
            print(f"Extracting features using model: {model}")
            feature_extract, vectorstore = feature_extraction_with_store(
                df, model, batch_size, n, dataset, content_column[i], 
                force_new_embeddings=force_new_embeddings,
                embeddings_only=embeddings_only
            )
            
            
            if create_graph and not embeddings_only:
                gephi_export(feature_extract, dataset, model, mapping, attributes)
                
            # Store feature extraction results and vector store
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
                
                # Visualize the multi-category graph
                output_file = f"multi_category_graph_{dataset}_{model}.png".replace('/', '_')
                visualize_multi_category_graph(graph, df, id_column[i], output_file, max_categories=max_categories)
                print(f"Multi-category graph saved as {output_file}")
                
            print(f"Completed processing for model: {model}")
            
        # Store the processed dataframe
        if 'processed_df' not in model_dict:
            model_dict['processed_df'] = {}
        model_dict['processed_df'][dataset] = df
        
        print(f"Completed processing for dataset: {dataset}")
    
    return model_dict
