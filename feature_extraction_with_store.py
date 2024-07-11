import os
import numpy as np
import pandas as pd
import pickle
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List

def feature_extraction_with_store(
    df: pd.DataFrame,
    full_df: pd.DataFrame,
    model: str,
    n: int,
    dataset_name: str,
    content_column: str,
    force_new_embeddings: bool = False
) -> np.ndarray:
    base_dir = os.path.join("embeddings", dataset_name, model.replace('/', '_'))
    os.makedirs(base_dir, exist_ok=True)
    
    embeddings_path = os.path.join(base_dir, "embeddings.npy")
    index_path = os.path.join(base_dir, "index_max.pkl")
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name=model)
    
    # Load or create index
    if os.path.exists(index_path) and not force_new_embeddings:
        with open(index_path, 'rb') as f:
            all_indices = pickle.load(f)
    else:
        all_indices = df.index.tolist()
        np.random.shuffle(all_indices)
        with open(index_path, 'wb') as f:
            pickle.dump(all_indices, f)

    selected_indices = all_indices[:n]
    
    # Load existing embeddings if available
    if os.path.exists(embeddings_path) and not force_new_embeddings:
        all_embeddings = np.load(embeddings_path)
        existing_n = len(all_embeddings)
    else:
        all_embeddings = np.array([])
        existing_n = 0
    
    # Determine which embeddings need to be computed
    new_indices = selected_indices[existing_n:]
    
    if new_indices or force_new_embeddings:
        print(f"Computing embeddings for {len(new_indices)} new samples")
        new_texts = full_df.loc[new_indices, content_column].tolist()
        new_embeddings = embeddings.embed_documents(new_texts)
        
        if len(all_embeddings) > 0 and not force_new_embeddings:
            all_embeddings = np.vstack([all_embeddings, new_embeddings])
        else:
            all_embeddings = np.array(new_embeddings)
        
        np.save(embeddings_path, all_embeddings)
        print(str(len(all_embeddings)) + " embeddings have been saved")
    
    # Ensure we have enough embeddings
    if len(all_embeddings) < n:
        raise ValueError(f"Not enough embeddings computed. Requested {n}, but only have {len(all_embeddings)}")
    
    # Select only the required embeddings
    feature_extract = all_embeddings[:n]
    
    return feature_extract