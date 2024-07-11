
import os
import pickle
import numpy as np
import pandas as pd
def get_consistent_samples(df: pd.DataFrame, n: int, dataset_name: str, model: str, max_n: int = None):
    base_dir = os.path.join("embeddings", dataset_name, model.replace('/', '_'))
    os.makedirs(base_dir, exist_ok=True)
    index_path = os.path.join(base_dir, f"index_max.pkl")
    
    max_n = min(max_n or len(df), len(df))
    n = min(n, max_n)
    
    if os.path.exists(index_path):
        with open(index_path, 'rb') as f:
            all_indices = pickle.load(f)
        # Filter out indices that are not in the current DataFrame
        all_indices = [idx for idx in all_indices if idx in df.index]
    else:
        all_indices = df.index.tolist()
        np.random.shuffle(all_indices)
        with open(index_path, 'wb') as f:
            pickle.dump(all_indices, f)
    
    selected_indices = all_indices[:n]
    return df.loc[selected_indices].reset_index(drop=True), selected_indices