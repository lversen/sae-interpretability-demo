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
# conda list --export > requirements.txt
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage


def plot_clustermap(feature_activations, figsize=(20, 20)):
    # Compute linkage for rows and columns
    row_linkage = linkage(feature_activations, method='ward')
    col_linkage = linkage(feature_activations.T, method='ward')

    # Create the clustermap
    g = sns.clustermap(
        feature_activations.T,
        figsize=figsize,
        row_linkage=col_linkage,
        col_linkage=row_linkage,
        cmap='viridis',
        xticklabels=False,
        yticklabels=False,
        cbar_pos=None  # Remove colorbar
    )

    # Remove axes labels
    g.ax_heatmap.set_xlabel('')
    g.ax_heatmap.set_ylabel('')

    # Remove the colorbar
    g.cax.remove()

    # Adjust layout to remove white spaces
    plt.tight_layout()

    return g


def plot_minimal_heatmap(feature_activations, figsize=(12*8, 8*8)):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the heatmap
    im = ax.imshow(feature_activations.T, aspect='auto',
                   cmap='viridis', interpolation='nearest')

    # Remove all axes
    ax.axis('off')

    # Remove all white space around the heatmap
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    return fig, ax


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


def load_or_train_sae(sae, train_feature_extract, val_feature_extract, model_path, learning_rate, batch_size, num_epochs, reconstruction_error_threshold, force_retrain=False):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Convert feature_extract to PyTorch tensors if they're numpy arrays
    if isinstance(train_feature_extract, np.ndarray):
        train_feature_extract = torch.from_numpy(train_feature_extract).float()
    if isinstance(val_feature_extract, np.ndarray):
        val_feature_extract = torch.from_numpy(val_feature_extract).float()

    # Ensure feature_extract tensors are on the same device as the model
    train_feature_extract = train_feature_extract.to(sae.device)
    val_feature_extract = val_feature_extract.to(sae.device)

    if os.path.exists(model_path) and not force_retrain:
        try:
            sae.load_state_dict(torch.load(model_path, weights_only=True))
            print(f"Loaded pre-trained SAE model from {model_path}")

            with torch.no_grad():
                _, x_hat, _ = sae(val_feature_extract[:100])
            reconstruction_error = torch.mean(
                (val_feature_extract[:100] - x_hat) ** 2)

            if reconstruction_error > reconstruction_error_threshold:
                print(f"Loaded model seems untrained or poorly fitted (error: {
                      reconstruction_error:.4f}). Retraining...")
                sae.train_and_validate(train_feature_extract, val_feature_extract,
                                       learning_rate=learning_rate, batch_size=batch_size, num_epochs=num_epochs)
                torch.save(sae.state_dict(), model_path)
                print(f"Retrained model saved to {model_path}")
            else:
                print(
                    f"Loaded model appears to be well-trained (error: {reconstruction_error:.4f})")
        except Exception as e:
            print(f"Error loading the model: {e}. Training a new one...")
            sae.train_and_validate(train_feature_extract, val_feature_extract,
                                   learning_rate=learning_rate, batch_size=batch_size, num_epochs=num_epochs)
            torch.save(sae.state_dict(), model_path)
            print(f"New model trained and saved to {model_path}")
    else:
        if force_retrain:
            print(f"Force retrain flag is set. Training a new model...")
        else:
            print(
                f"No pre-trained model found at {model_path}. Training a new one...")

        sae.train_and_validate(train_feature_extract, val_feature_extract,
                               learning_rate=learning_rate, batch_size=batch_size, num_epochs=num_epochs)
        torch.save(sae.state_dict(), model_path)
        print(f"New model trained and saved to {model_path}")

    return sae


def run_all(
    train_dataset: str,
    val_dataset: str,
    models: List[str],
    n_train: int,
    n_val: int,
    feature_column: str,
    label_column: str,
    create_graph: bool = False,
    force_new_embeddings: bool = False,
    classify_language: List[str] = [],
    top_n_category: Optional[Dict[str, Dict[str, Any]]] = None,
    sae_params: Dict[str, Any] = {}
) -> Dict[str, np.ndarray]:
    all_feature_activations = {}

    print(f"Processing train dataset: {train_dataset}")
    print(f"Processing validation dataset: {val_dataset}")

    train_df = pd.read_csv(train_dataset)
    val_df = pd.read_csv(val_dataset)
    print(f"Train dataset shape: {train_df.shape}")
    print(f"Validation dataset shape: {val_df.shape}")

    for model in models:
        print(f"Processing model: {model}")

        # Get consistent samples for train and validation sets
        train_sample_df, train_indices = get_consistent_samples(
            train_df, n_train, f"{train_dataset}_train", model)
        val_sample_df, val_indices = get_consistent_samples(
            val_df, n_val, f"{val_dataset}_val", model)
        print(f"Train sample shape: {train_sample_df.shape}")
        print(f"Validation sample shape: {val_sample_df.shape}")

        # Extract features for train and validation sets
        print(f"Extracting features for {len(train_sample_df)} train samples")
        train_feature_extract = feature_extraction_with_store(
            train_sample_df, train_df, model, len(train_sample_df), f"{
                train_dataset}_train", feature_column,
            force_new_embeddings=force_new_embeddings
        )
        print(f"Extracting features for {
              len(val_sample_df)} validation samples")
        val_feature_extract = feature_extraction_with_store(
            val_sample_df, val_df, model, len(val_sample_df), f"{
                val_dataset}_val", feature_column,
            force_new_embeddings=force_new_embeddings
        )

        # Initialize and train/load SAE
        D = train_feature_extract.shape[1]
        F = 2 * D
        l1_lambda = 5
        sae = SparseAutoencoder(D, F, l1_lambda)
        sae_model_path = f'models/sae_model_{os.path.basename(train_dataset)}_{
            model.replace("/", "_")}.pth'
        sae = load_or_train_sae(
            sae,
            train_feature_extract,
            val_feature_extract,
            model_path=sae_model_path,
            learning_rate=sae_params.get('learning_rate', 1e-3),
            batch_size=sae_params.get('batch_size', 40),
            num_epochs=sae_params.get('num_epochs', 20),
            reconstruction_error_threshold=sae_params.get(
                'reconstruction_error_threshold', 0.1),
            force_retrain=sae_params.get('force_retrain', False)
        )

        # Get feature activations (using validation set for consistency)
        with torch.no_grad():
            feature_activations = sae.feature_activations(
                torch.from_numpy(val_feature_extract).float().to(sae.device))
            all_feature_activations[f"{val_dataset}_{
                model}"] = feature_activations.cpu().numpy()

        if create_graph:
            mapping, attributes = node_attributes(
                val_sample_df, label_column, model, 'assigned_category')
            print(f"Exporting Gephi graph for {
                  val_dataset} with model {model}")
            gephi_export(val_feature_extract, val_dataset,
                         model, mapping, attributes)

    print("Processing complete for all datasets and models.")
    return val_sample_df, all_feature_activations


if __name__ == "__main__":
    train_dataset = "data/stack_exchange_train.csv"
    val_dataset = "data/stack_exchange_val.csv"
    feature_column = "sentences"
    label_column = "labels"
    models = ["Alibaba-NLP/gte-large-en-v1.5"]
    n_max = pd.read_csv("data/stack_exchange_train.csv").shape[0]
    n_train = n_max
    n_val = 10_000

    # SAE hyperparameters
    sae_params = {
        'learning_rate': 1e-3,
        'batch_size': 32,
        'num_epochs': 50,
        'reconstruction_error_threshold': 20,
        'force_retrain': False
    }

    df, feature_activations = run_all(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        models=models,
        n_train=n_train,
        n_val=n_val,
        feature_column=feature_column,
        label_column=label_column,
        sae_params=sae_params,
        create_graph=False,
        force_new_embeddings=False
    )
# =============================================================================
#     for key in feature_activations.keys():
#         fa = feature_activations[key]
#         #fig, ax = plot_minimal_heatmap(fa)
#         g = plot_clustermap(fa)
#         plt.close()  # Close the figure to free up memory
# =============================================================================
