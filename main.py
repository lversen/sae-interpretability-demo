import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import random
import torch
import torch.nn as nn
import torch.optim as optim
from feature_extraction_with_store import feature_extraction_with_store
from gephi import node_attributes, gephi_export
from language_classification import language_classifier
from sample_handler import get_consistent_samples
from SAE import SparseAutoencoder

np.set_printoptions(suppress=True)
def split_categories(category_string: str, delimiter: str = ',') -> List[str]:
    """Split a category string into a list of categories."""
    return [cat.strip() for cat in category_string.split(delimiter) if cat.strip()]

def select_and_assign_exact_n_categories(df: pd.DataFrame, category: str, n: int, delimiter: str = ',') -> Tuple[pd.DataFrame, List[str]]:
    """
    Select exactly N unique categories and then filter and assign these categories to movies.
    """
    if category not in df.columns:
        raise ValueError(f"{category} is not a column in the dataset")
    
    # Split the categories and create a flat list of all categories
    all_categories = [cat.strip() for cats in df[category].apply(lambda x: split_categories(str(x), delimiter)) for cat in cats]
    
    # Count the occurrences of each unique category
    category_counts = Counter(all_categories)
    
    # Get exactly n unique categories
    selected_categories = [cat for cat, _ in category_counts.most_common(n)]
    
    print(f"Selected top {n} categories: {selected_categories}")
    
    # Function to assign a random category from selected categories if any match, else return None
    def assign_category(cat_string):
        cats = split_categories(str(cat_string), delimiter)
        matching_cats = [cat for cat in cats if cat in selected_categories]
        return random.choice(matching_cats) if matching_cats else None
    
    # Assign categories and filter out rows with no matching categories
    df['assigned_category'] = df[category].apply(assign_category)
    filtered_df = df.dropna(subset=['assigned_category'])
    
    # Print the distribution of assigned categories
    assigned_categories = filtered_df['assigned_category'].value_counts(normalize=True) * 100
    print("Distribution of assigned categories:")
    print(assigned_categories)
    
    return filtered_df, selected_categories

def run_all(
    datasets: List[str],
    models: List[str],
    n: int,
    feature_column: List[str],
    label_column: List[str],
    create_graph: bool = False,
    force_new_embeddings: bool = False,
    classify_language: List[str] = [],
    top_n_category: Optional[Dict[str, Dict[str, Any]]] = None
):
    
    for i, dataset in enumerate(datasets):
        print(f"Processing dataset: {dataset}")        
        # Load full dataset
        full_df = pd.read_csv(dataset)
        print(f"Full dataset shape: {full_df.shape}")

        # If top_n_category is specified, filter the dataset and assign categories
        category_column = None
        if top_n_category:
            category_info = top_n_category.get(dataset, {})
            if category_info:
                category_column = category_info['column']
                n_top = category_info['n']
                delimiter = category_info.get('delimiter', ',')
                full_df, selected_categories = select_and_assign_exact_n_categories(full_df, category_column, n_top, delimiter)
                print(f"Filtered dataset shape after selecting and assigning exactly {n_top} categories: {full_df.shape}")

        # Adjust n if it's larger than the filtered dataset
        original_n = n
        n = min(n, len(full_df))
        if n < original_n:
            print(f"Warning: Requested sample size ({original_n}) is larger than the available data ({n}). Using {n} samples.")

        for model in models:
            print(f"Processing model: {model}")
            
            # Get consistent samples
            df, indices = get_consistent_samples(full_df, n, dataset, model)
            print(f"Sample shape: {df.shape}")
            
            # Feature extraction
            feature_extract = feature_extraction_with_store(
                df, full_df, model, n, dataset, feature_column[i], 
                force_new_embeddings=force_new_embeddings
            )
            SAE = SparseAutoencoder(feature_extract, 200, 16)
            SAE.forward(feature_extract[0])
            print(SAE.W_dec.shape)
            
            if len(classify_language) != 0:
                indices = np.array(indices, dtype=np.int32)
                language_classifier(df, indices, classify_language, dataset)
            
            if create_graph:
                # Gephi export
                mapping, attributes = node_attributes(df, label_column[i], model, 'assigned_category')
                print(f"Exporting Gephi graph for {dataset} with model {model}")
                gephi_export(feature_extract, dataset, model, mapping, attributes)
    
    print("Processing complete for all datasets and models.")
if __name__ == "__main__":
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
    
    datasets = ["data/final_data.csv"]
    feature_column = ["Description"]
    label_column = ["Name"]
    models = [
        #"BAAI/bge-m3",
        #"intfloat/e5-large-v2",
        'whaleloops/phrase-bert',
        #"sentence-transformers/paraphrase-MiniLM-L6-v2",
        #"sentence-transformers/all-mpnet-base-v2"
    ]
    n = 10
    top_n_category = {"data/final_data.csv": {"column": "Genres", "n": 10, "delimiter": ","}}

    run_all(
        datasets=datasets,
        models=models,
        n=n,
        feature_column=feature_column,
        label_column=label_column,
    )