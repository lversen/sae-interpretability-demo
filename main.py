import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, deque
import random
import torch
from feature_extraction_with_store import feature_extraction_with_store
from gephi import node_attributes, gephi_export
from language_classification import language_classifier
from sample_handler import get_consistent_samples
from SAE import SparseAutoencoder
from test_dataset import harry_potter_df
# conda list --export > requirements.txt

np.set_printoptions(suppress=True)


def split_categories(category_string: str, delimiter: str = ',') -> List[str]:
    return [cat.strip() for cat in category_string.split(delimiter) if cat.strip()]


def select_and_assign_exact_n_categories(df: pd.DataFrame, category: str, n: int, delimiter: str = ',') -> Tuple[pd.DataFrame, List[str]]:
    if category not in df.columns:
        raise ValueError(f"{category} is not a column in the dataset")

    all_categories = [cat.strip() for cats in df[category].apply(
        lambda x: split_categories(str(x), delimiter)) for cat in cats]
    category_counts = Counter(all_categories)
    selected_categories = [cat for cat, _ in category_counts.most_common(n)]

    print(f"Selected top {n} categories: {selected_categories}")

    def assign_category(cat_string):
        cats = split_categories(str(cat_string), delimiter)
        matching_cats = [cat for cat in cats if cat in selected_categories]
        return random.choice(matching_cats) if matching_cats else None

    df['assigned_category'] = df[category].apply(assign_category)
    filtered_df = df.dropna(subset=['assigned_category'])

    assigned_categories = filtered_df['assigned_category'].value_counts(
        normalize=True) * 100
    print("Distribution of assigned categories:")
    print(assigned_categories)

    return filtered_df, selected_categories


def load_or_train_sae(sae, feature_extract, model_path, learning_rate, batch_size, num_epochs, reconstruction_error_threshold, force_retrain=False):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Convert feature_extract to a PyTorch tensor if it's a numpy array
    if isinstance(feature_extract, np.ndarray):
        feature_extract = torch.from_numpy(feature_extract).float()

    # Ensure feature_extract is on the same device as the model
    feature_extract = feature_extract.to(sae.device)

    if os.path.exists(model_path) and not force_retrain:
        try:
            sae.load_state_dict(torch.load(model_path, weights_only=True))
            print(f"Loaded pre-trained SAE model from {model_path}")

            with torch.no_grad():
                _, x_hat, _ = sae(feature_extract[:100])
            reconstruction_error = torch.mean(
                (feature_extract[:100] - x_hat) ** 2)

            if reconstruction_error > reconstruction_error_threshold:
                print(f"Loaded model seems untrained or poorly fitted (error: {
                      reconstruction_error:.4f}). Retraining...")
                sae.train_and_validate(
                    feature_extract, learning_rate=learning_rate, batch_size=batch_size, num_epochs=num_epochs)
                torch.save(sae.state_dict(), model_path)
                print(f"Retrained model saved to {model_path}")
            else:
                print(
                    f"Loaded model appears to be well-trained (error: {reconstruction_error:.4f})")
        except Exception as e:
            print(f"Error loading the model: {e}. Training a new one...")
            sae.train_and_validate(feature_extract, learning_rate=learning_rate,
                                   batch_size=batch_size, num_epochs=num_epochs)
            torch.save(sae.state_dict(), model_path)
            print(f"New model trained and saved to {model_path}")
    else:
        if force_retrain:
            print(f"Force retrain flag is set. Training a new model...")
        else:
            print(
                f"No pre-trained model found at {model_path}. Training a new one...")

        sae.train_and_validate(feature_extract, learning_rate=learning_rate,
                               batch_size=batch_size, num_epochs=num_epochs)
        torch.save(sae.state_dict(), model_path)
        print(f"New model trained and saved to {model_path}")

    return sae


def run_all(
    datasets: List[str],
    models: List[str],
    n: int,
    feature_column: List[str],
    label_column: List[str],
    create_graph: bool = False,
    force_new_embeddings: bool = False,
    classify_language: List[str] = [],
    top_n_category: Optional[Dict[str, Dict[str, Any]]] = None,
    sae_params: Dict[str, Any] = {},
    additional_dataset: Optional[Dict[str, pd.DataFrame]] = None
):
    all_feature_activations = {}
    
    for i, dataset in enumerate(datasets):
        print(f"Processing dataset: {dataset}")
        full_df = pd.read_csv(dataset)
        print(f"Full dataset shape: {full_df.shape}")

        category_column = None
        if top_n_category:
            category_info = top_n_category.get(dataset, {})
            if category_info:
                category_column = category_info['column']
                n_top = category_info['n']
                delimiter = category_info.get('delimiter', ',')
                full_df, selected_categories = select_and_assign_exact_n_categories(
                    full_df, category_column, n_top, delimiter)
                print(f"Filtered dataset shape after selecting and assigning exactly {n_top} categories: {full_df.shape}")

        original_n = n
        n = min(n, len(full_df))
        if n < original_n:
            print(f"Warning: Requested sample size ({original_n}) is larger than the available data ({n}). Using {n} samples.")

        for model in models:
            print(f"Processing model: {model}")

            df, indices = get_consistent_samples(full_df, n, dataset, model)
            print(f"Sample shape: {df.shape}")

            additional_df = additional_dataset.get(dataset) if additional_dataset else None
            
            feature_extract, additional_features = feature_extraction_with_store(
                df, full_df, model, n, dataset, feature_column[i],
                force_new_embeddings=force_new_embeddings,
                additional_data=additional_df
            )

            # Initialize and train/load SAE
            D = feature_extract.shape[1]
            F = 2 * D
            l1_lambda = 5
            sae = SparseAutoencoder(D, F, l1_lambda)
            sae_model_path = f'models/sae_model_{dataset}_{model}.pth'
            sae = load_or_train_sae(
                sae,
                feature_extract,
                model_path=sae_model_path,
                learning_rate=sae_params.get('learning_rate', 1e-3),
                batch_size=sae_params.get('batch_size', 40),
                num_epochs=sae_params.get('num_epochs', 20),
                reconstruction_error_threshold=sae_params.get('reconstruction_error_threshold', 0.1),
                force_retrain=sae_params.get('force_retrain', False)
            )

            # Get SAE features
            sae_feature_vectors = sae.feature_vectors()
            
            # Process additional features if available
            if additional_features is not None:
                additional_features_tensor = torch.from_numpy(additional_features).float().to(sae.device)
                additional_feature_activations = sae.feature_activations(additional_features_tensor)
                all_feature_activations[f"{dataset}_{model}"] = additional_feature_activations.cpu().data.numpy()

            if len(classify_language) != 0:
                indices = np.array(indices, dtype=np.int32)
                language_classifier(df, indices, classify_language, dataset)

            if create_graph:
                mapping, attributes = node_attributes(df, label_column[i], model, 'assigned_category')
                print(f"Exporting Gephi graph for {dataset} with model {model}")
                gephi_export(feature_extract, dataset, model, mapping, attributes)

    print("Processing complete for all datasets and models.")
    return all_feature_activations


if __name__ == "__main__":
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

    datasets = ["data/final_data.csv"]
    feature_column = ["Description"]
    label_column = ["Name"]
    models = ['whaleloops/phrase-bert']
    n = 44266

    # SAE hyperparameters
    sae_params = {
        'learning_rate': 5e-4,
        'batch_size': 2**10,
        'num_epochs': 150,
        'reconstruction_error_threshold': 0.1,
        'force_retrain': False # Set this to True when you want to retrain the model
    }

    # Load additional datasets
    from test_dataset import test_df
    additional_dataset = {
        "data/final_data.csv": test_df,
        # Add more datasets here if needed
        # "path/to/another/dataset.csv": another_df,
    }

    all_feature_activations = run_all(
        datasets=datasets,
        models=models,
        n=n,
        feature_column=feature_column,
        label_column=label_column,
        sae_params=sae_params,
        additional_dataset=additional_dataset
    )