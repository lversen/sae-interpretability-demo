
import os
import pickle
import numpy as np
import pandas as pd
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
