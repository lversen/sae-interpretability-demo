import os
import numpy as np
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Optional
import torch

def feature_extraction_with_store(
    df: pd.DataFrame,
    full_df: pd.DataFrame,
    model: str,
    n: int,
    dataset_name: str,
    content_column: str,
    force_new_embeddings: bool = False) -> np.ndarray:
    base_dir = os.path.join("embeddings", dataset_name, model.replace('/', '_'))
    os.makedirs(base_dir, exist_ok=True)
    
    embeddings_path = os.path.join(base_dir, "embeddings.npy")
    index_path = os.path.join(base_dir, "index.pkl")
    torch.cuda.empty_cache()
    
    # Initialize embeddings
    embeddings = SentenceTransformer(model, trust_remote_code=True, device="cuda")
    
    # Load or create index
    if os.path.exists(index_path) and not force_new_embeddings:
        with open(index_path, 'rb') as f:
            all_indices = pickle.load(f)
    else:
        all_indices = df.index.tolist()
        np.random.shuffle(all_indices)
    
    # Ensure all_indices has at least n elements
    if len(all_indices) < n:
        additional_indices = df.index[~df.index.isin(all_indices)].tolist()
        np.random.shuffle(additional_indices)
        all_indices.extend(additional_indices[:n - len(all_indices)])
    
    # Save the updated index
    with open(index_path, 'wb') as f:
        pickle.dump(all_indices, f)

    # Load existing embeddings if available
    if os.path.exists(embeddings_path) and not force_new_embeddings:
        all_embeddings = np.load(embeddings_path)
        existing_n = len(all_embeddings)
    else:
        all_embeddings = np.array([])
        existing_n = 0
    
    print(f"Existing embeddings: {existing_n}, Requested embeddings: {n}")
    
    # Determine which embeddings need to be computed
    if n > existing_n:
        new_indices = all_indices[existing_n:n]
        print(f"Computing embeddings for {len(new_indices)} new samples")
        new_texts = full_df.loc[new_indices, content_column].tolist()
        new_embeddings = embeddings.encode(new_texts)
        
        if len(all_embeddings) > 0:
            all_embeddings = np.vstack([all_embeddings, new_embeddings])
        else:
            all_embeddings = new_embeddings
        
        np.save(embeddings_path, all_embeddings)
        print(f"{len(all_embeddings)} embeddings have been saved")
    elif n < existing_n:
        print(f"Using a subset of {n} embeddings from the existing {existing_n}")
    else:
        print(f"Using all {existing_n} existing embeddings")
    
    # Select only the required embeddings
    feature_extract = all_embeddings[:n]
    

    return feature_extract