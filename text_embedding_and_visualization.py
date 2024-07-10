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
from sklearn.neighbors import kneighbors_graph
from sentence_transformers import SentenceTransformer

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

def process_with_rag(df: pd.DataFrame, content_column: str, query_column: str, llm, embeddings, vectorstore):
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": PromptTemplate(
                template="Use the following context to answer the question: {context}\n\nQuestion: {question}\n\nAnswer:",
                input_variables=["context", "question"]
            )
        }
    )
    
    results = []
    for query in df[query_column]:
        result = rag_chain.invoke({"query": query})  # Changed from __call__ to invoke
        results.append(result['result'])
    
    df['rag_result'] = results
    return df


def create_or_load_vectorstore(
    df: pd.DataFrame, 
    content_column: str, 
    embeddings, 
    dataset_name: str, 
    model_name: str,
    n: int, 
    force_recreate: bool = False
):
    # Clean up model name for file naming
    model_name_clean = model_name.replace('/', '_').replace('\\', '_')
    
    vectorstore_path = f"vectorstore_{dataset_name}_{model_name_clean}_n{n}.pkl"
    
    if os.path.exists(vectorstore_path) and not force_recreate:
        print(f"Loading existing vector store for {dataset_name} with model {model_name} and n={n}")
        with open(vectorstore_path, "rb") as f:
            vectorstore = pickle.load(f)
    else:
        print(f"Creating new vector store for {dataset_name} with model {model_name} and n={n}")
        texts = df[content_column].tolist()
        vectorstore = FAISS.from_texts(texts, embeddings)
        with open(vectorstore_path, "wb") as f:
            pickle.dump(vectorstore, f)
    
    return vectorstore


def setup_rag_pipeline(model_name: str, embeddings_model: str):
    api_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not api_token:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in environment variables")
    
    llm = HuggingFaceEndpoint(
        repo_id=model_name,
        huggingfacehub_api_token=api_token,
        task="text-generation",
        model_kwargs={
            "max_new_tokens": 250,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.95,
        }
    )
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
    return llm, embeddings

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
        embeddings = HuggingFaceEmbeddings(model_name=model, show_progress=True)
        texts = df[content_column].tolist()
        feature_extract = embeddings.embed_documents(texts)
        
        # Save embeddings
        np.save(embeddings_path, feature_extract)
        
        if not embeddings_only:
            print(f"Creating new vector store for {dataset_name} with model {model}")
            text_embedding_pairs = list(zip(texts, feature_extract))
            vectorstore = FAISS.from_embeddings(text_embedding_pairs, embeddings)
            
            # Save vectorstore
            with open(vectorstore_path, "wb") as f:
                pickle.dump(vectorstore, f)
    
    return feature_extract, vectorstore


def run_all(
    datasets: List[str],
    models: List[str],
    n: int,
    content_column: List[str],
    title_column: List[str] = [],
    classify_language: List[str] = [],
    duplicate_method: str = 'suffix',
    batch_size: int = 16,
    create_graph: bool = False,
    use_rag: bool = False,
    rag_model: str = "google/flan-t5-base",
    rag_embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    query_column: str = "",
    force_new_embeddings: bool = False,
    embeddings_only: bool = False
) -> Dict[str, Dict[str, Any]]:
    model_dict = {}
    
    if use_rag:
        try:
            llm, rag_embeddings = setup_rag_pipeline(rag_model, rag_embeddings_model)
        except Exception as e:
            print(f"Error setting up RAG pipeline: {str(e)}")
            return model_dict
    
    for i, dataset in enumerate(datasets):
        print(f"Processing dataset: {dataset}")
        
        # Initialize dataframe
        df, rows, max_rows = data_frame_init(dataset, content_column, i, n)
            
        # Preprocess to handle duplicates
        if title_column:
            df = preprocess_duplicates(df, title_column[i], method=duplicate_method)
            
        # Perform language classification if requested
        if classify_language:
            with warnings.catch_warnings(record=True) as w:
                language_classifier(df, rows, max_rows, classify_language, dataset)
                if w:
                    print(f"Warning in language_classifier for {dataset}: {w[0].message}")
            
        # Process with RAG if requested
        if use_rag:
            print(f"Processing {dataset} with RAG")
            dataset_name = os.path.basename(dataset).split('.')[0]
            rag_vectorstore = create_or_load_vectorstore(
                df, 
                content_column[i], 
                rag_embeddings, 
                dataset_name,
                rag_embeddings_model,
                n, 
                force_new_embeddings
            )
            df = process_with_rag(df, content_column[i], query_column, llm, rag_embeddings, rag_vectorstore)
            
        # Get node attributes for graph creation
        if create_graph and title_column:
            with warnings.catch_warnings(record=True) as w:
                mapping, attributes = node_attributes(df, title_column[i])
                if w:
                    print(f"Warning in node_attributes for {dataset}: {w[0].message}")
        else:
            mapping, attributes = {}, {}
            
        # Process each model
        for model in models:
            print(f"Extracting features using model: {model}")
            feature_extract, model_vectorstore = feature_extraction_with_store(
                df, model, batch_size, n, dataset, content_column[i], 
                force_new_embeddings=force_new_embeddings,
                embeddings_only=embeddings_only
            )
                
            if create_graph and not embeddings_only:
                gephi_export(feature_extract, dataset, model, mapping, attributes)
                
            # Store feature extraction results and vector store
            if model not in model_dict:
                model_dict[model] = {}
            model_dict[model][dataset] = {
                'feature_extract': feature_extract,
                'vectorstore': model_vectorstore
            }
                
            print(f"Completed processing for model: {model}")
            
        # Store the processed dataframe
        if 'processed_df' not in model_dict:
            model_dict['processed_df'] = {}
        model_dict['processed_df'][dataset] = df
        
        print(f"Completed processing for dataset: {dataset}")
    
    return model_dict