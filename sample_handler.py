
import os
import pickle
import numpy as np
import pandas as pd
def get_consistent_samples(df: pd.DataFrame, n: int, dataset_name: str, model: str, max_n: int = None):
    print(f"Requested n: {n}")
    base_dir = os.path.join("embeddings", dataset_name, model.replace('/', '_'))
    os.makedirs(base_dir, exist_ok=True)
    index_path = os.path.join(base_dir, f"index_max.pkl")
    
    max_n = min(max_n or len(df), len(df))
    print(f"Max n: {max_n}")
    n = min(n, max_n)
    print(f"Adjusted n: {n}")    

    if os.path.exists(index_path):
        with open(index_path, 'rb') as f:
            all_indices = pickle.load(f)
        # Filter out indices that are not in the current DataFrame
        all_indices = [idx for idx in all_indices if idx in df.index]
        
        # If we need more samples than we have, add more indices
        if n > len(all_indices):
            additional_indices = df.index[~df.index.isin(all_indices)].tolist()
            np.random.shuffle(additional_indices)
            all_indices.extend(additional_indices[:n - len(all_indices)])
    else:
        all_indices = df.index.tolist()
        np.random.shuffle(all_indices)

    # Always save the full list of indices
    with open(index_path, 'wb') as f:
        pickle.dump(all_indices, f)
    
    selected_indices = all_indices[:n]
    selected_df = df.loc[selected_indices].reset_index(drop=True)
    
    print(f"Selected {len(selected_indices)} samples")
    return selected_df, selected_indices