import re
import os
import sys
import subprocess
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import Counter, deque
import random
import torch
from feature_extraction_with_store import feature_extraction_with_store
from gephi import *

from sample_handler import get_consistent_samples
from SAE import SparseAutoencoder
from ST import SparseTransformer
# conda list --export > requirements.txt
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def train_and_evaluate_decision_tree(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    clf = DecisionTreeClassifier(random_state=random_state)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return clf, accuracy, report



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


def load_or_train_model(model, train_feature_extract, val_feature_extract, model_path, learning_rate, batch_size, reconstruction_error_threshold, force_retrain=False):
    """
    Load or train a model with support for both SAE and ST model types.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Convert feature_extract to PyTorch tensors if they're numpy arrays
    if isinstance(train_feature_extract, np.ndarray):
        train_feature_extract = torch.from_numpy(train_feature_extract).float()
    if isinstance(val_feature_extract, np.ndarray):
        val_feature_extract = torch.from_numpy(val_feature_extract).float()

    # Ensure feature_extract tensors are on the same device as the model
    train_feature_extract = train_feature_extract.to(model.device)
    val_feature_extract = val_feature_extract.to(model.device)

    if os.path.exists(model_path) and not force_retrain:
        try:
            model.load_state_dict(torch.load(model_path))
            print(f"Loaded pre-trained model from {model_path}")

            # Check model quality differently based on model type
            with torch.no_grad():
                if isinstance(model, SparseAutoencoder):
                    _, x_hat, _ = model(val_feature_extract[:100])
                    reconstruction_error = torch.mean((val_feature_extract[:100] - x_hat) ** 2)
                else:  # SparseTransformer
                    # ST model returns x, x_hat, attention_weights
                    _, x_hat, _ = model(val_feature_extract[:100])
                    reconstruction_error = torch.mean((val_feature_extract[:100] - x_hat) ** 2)

            if reconstruction_error > reconstruction_error_threshold:
                print(f"Loaded model seems untrained or poorly fitted (error: {reconstruction_error:.4f}). Retraining...")
                model.train_and_validate(
                    train_feature_extract,
                    val_feature_extract,
                    learning_rate=learning_rate,
                    batch_size=batch_size
                )
                torch.save(model.state_dict(), model_path)
                print(f"Retrained model saved to {model_path}")
            else:
                print(f"Loaded model appears to be well-trained (error: {reconstruction_error:.4f})")
        except Exception as e:
            print(f"Error loading the model: {e}. Training a new one...")
            model.train_and_validate(
                train_feature_extract,
                val_feature_extract,
                learning_rate=learning_rate,
                batch_size=batch_size
            )
            torch.save(model.state_dict(), model_path)
            print(f"New model trained and saved to {model_path}")
    else:
        if force_retrain:
            print(f"Force retrain flag is set. Training a new model...")
        else:
            print(f"No pre-trained model found at {model_path}. Training a new one...")

        model.train_and_validate(
            train_feature_extract,
            val_feature_extract,
            learning_rate=learning_rate,
            batch_size=batch_size
        )
        torch.save(model.state_dict(), model_path)
        print(f"New model trained and saved to {model_path}")

    return model

# Add this at the top of the file with other imports


def direct_vector_feature_extraction(df: pd.DataFrame, feature_columns: list) -> np.ndarray:
    """
    Extract features directly from dataframe columns that already contain vector data.

    Args:
        df: DataFrame containing the feature columns
        feature_columns: List of column names containing the feature values

    Returns:
        numpy.ndarray: Feature matrix where each row is a sample and each column is a feature
    """
    # Convert specified columns to numpy array
    feature_matrix = df[feature_columns].values

    # Ensure the output is float32 for consistency with embeddings
    return feature_matrix.astype(np.float32)


def get_feature_extraction_fn(data_type: str):
    """
    Factory function to return appropriate feature extraction function based on data type.

    Args:
        data_type: String indicating the type of data ('text' or 'vector')

    Returns:
        function: Appropriate feature extraction function
    """
    from feature_extraction_with_store import feature_extraction_with_store

    if data_type == 'text':
        return feature_extraction_with_store
    elif data_type == 'vector':
        return direct_vector_feature_extraction
    else:
        raise ValueError(f"Unknown data type: {data_type}")


def sanitize_filename(filename):
    # Replace invalid filename characters with underscores
    return re.sub(r'[<>:"/\\|?*]', '_', filename)


def run_all(
    train_dataset: str,
    val_dataset: str,
    models: List[str],
    n_train: int,
    n_val: int,
    feature_column: Union[str, List[str]],
    label_column: str,
    data_type: str = 'text',
    perform_classification: bool = True,
    create_graph: bool = False,
    n_random_labels: Optional[int] = None,
    force_new_embeddings: bool = False,
    classify_language: List[str] = [],
    top_n_category: Optional[Dict[str, Dict[str, Any]]] = None,
    model_params: Dict[str, Any] = {},
    model_type: str = "sae",
    gephi_subset_size: Optional[int] = None  # New parameter
) -> Dict[str, Any]:
    all_feature_activations = {}
    classification_results = {}

    print(f"Processing train dataset: {train_dataset}")

    train_df = pd.read_csv(train_dataset)
    val_df = pd.read_csv(val_dataset)

    feature_extraction_fn = get_feature_extraction_fn(data_type)

    for model in models:
        print(f"Processing model: {model}")

        # Get full training samples for model training
        train_sample_df, train_indices = get_consistent_samples(
            train_df, n_train, f"{train_dataset}_train", model)
        val_sample_df, val_indices = get_consistent_samples(
            val_df, n_val, f"{val_dataset}_val", model)

        # Extract features and train model as before
        if data_type == 'text':
            train_feature_extract = feature_extraction_fn(
                train_sample_df, train_df, model, len(train_sample_df),
                f"{train_dataset}_train", feature_column,
                force_new_embeddings=force_new_embeddings
            )
            val_feature_extract = feature_extraction_fn(
                val_sample_df, val_df, model, len(val_sample_df),
                f"{val_dataset}_val", feature_column,
                force_new_embeddings=force_new_embeddings
            )
        else:
            train_feature_extract = feature_extraction_fn(
                train_sample_df, feature_column
            )
            val_feature_extract = feature_extraction_fn(
                val_sample_df, feature_column
            )

        # Model initialization and training (unchanged)
        n = train_feature_extract.shape[1]
        m = 8 * n
        l1_lambda = model_params.get('l1_lambda', 5)
        model_path = f'models/{model_type}_model_{os.path.basename(train_dataset)}_{model.replace("/", "_")}.pth'

        if model_type.lower() == "sae":
            sparse_model = SparseAutoencoder(n, m, model_path, l1_lambda)
        elif model_type.lower() == "st":
            a = 512  # Attention dimension
            X_cross = train_feature_extract
            # Initialize with multiple heads
            sparse_model = SparseTransformer(
                X=X_cross,
                D=n,
                F=m,
                M=a,  # Should be divisible by num_heads
                st_model_path='path/to/model.pth',
                lambda_l1=l1_lambda,
                num_heads=8  # New parameter for number of attention heads
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        Model = load_or_train_model(
            sparse_model,
            train_feature_extract,
            val_feature_extract,
            model_path=model_path,
            learning_rate=model_params.get('learning_rate', 1e-3),
            batch_size=model_params.get('batch_size', 40),
            reconstruction_error_threshold=model_params.get('reconstruction_error_threshold', 0.1),
            force_retrain=model_params.get('force_retrain', False)
        )

        # Get feature activations for the full dataset
        with torch.no_grad():
            feature_activations = Model.feature_activations(
                torch.from_numpy(train_feature_extract).float().to(Model.device))
            feature_activations = feature_activations.cpu().numpy()
            all_feature_activations[f"{train_dataset}_{model}"] = feature_activations

        if perform_classification and label_column in train_sample_df.columns:
            # Classification remains unchanged
            print("Performing decision tree classification on Model feature activations")
            Model_clf, Model_accuracy, Model_report = train_and_evaluate_decision_tree(
                all_feature_activations[f"{train_dataset}_{model}"], train_sample_df[label_column])
            classification_results[f"{model}_{model_type}"] = {
                "accuracy": Model_accuracy,
                "report": Model_report
            }

        if create_graph:
            # Create smaller subset for Gephi export if specified
            if gephi_subset_size is not None and gephi_subset_size < len(train_sample_df):
                gephi_df, gephi_indices = get_consistent_samples(
                    train_sample_df, gephi_subset_size, 
                    f"{train_dataset}_gephi", model)
                gephi_feature_extract = train_feature_extract[gephi_indices]
                gephi_feature_activations = feature_activations[gephi_indices]
            else:
                gephi_df = train_sample_df
                gephi_feature_extract = train_feature_extract
                gephi_feature_activations = feature_activations

            selected_labels = select_random_labels(
                gephi_df,
                label_column,
                n_random_labels=n_random_labels,
                category_column='assigned_category' if 'assigned_category' in gephi_df.columns else None
            )

            # Create Gephi graphs with the subset
            base_filename = os.path.splitext(os.path.basename(train_dataset))[0]
            sanitized_model = model.replace('.', '-')
            
            original_file_path = f"gephi_exports_{base_filename}/{base_filename}_{sanitized_model}_original.gexf"
            create_gephi_graph(
                gephi_feature_extract,
                gephi_df,
                label_column,
                f"{sanitized_model}_original",
                original_file_path,
                selected_labels=selected_labels,
                category_column='assigned_category' if 'assigned_category' in gephi_df.columns else None
            )

            model_file_path = f"gephi_exports_{base_filename}/{base_filename}_{sanitized_model}_{model_type}.gexf"
            create_gephi_graph(
                gephi_feature_activations,
                gephi_df,
                label_column,
                f"{sanitized_model}_{model_type}",
                model_file_path,
                selected_labels=selected_labels,
                category_column='assigned_category' if 'assigned_category' in gephi_df.columns else None
            )

    if perform_classification and classification_results:
        print("\nClassification Results:")
        for key, result in classification_results.items():
            print(f"\n{key}:")
            print(f"Accuracy: {result['accuracy']}")
            print("Classification Report:")
            print(result['report'])

        # Save classification results
        results_list = []
        for key, result in classification_results.items():
            model, type_ = key.rsplit('_', 1)
            results_list.append({
                'Model': model,
                'Type': type_,
                'Accuracy': result['accuracy']
            })

        results_df = pd.DataFrame(results_list)

        # Create a directory for results if it doesn't exist
        os.makedirs('results', exist_ok=True)

        # Save the results
        base_filename = os.path.splitext(os.path.basename(train_dataset))[0]
        results_file = os.path.join(
            'results', f'classification_results_{base_filename}.csv')
        results_df.to_csv(results_file, index=False)
        print(f"Classification results saved to {results_file}")

        # Save detailed classification reports
        for key, result in classification_results.items():
            model, type_ = key.rsplit('_', 1)
            sanitized_model = sanitize_filename(model)
            report_filename = f'classification_report_{
                sanitized_model}_{type_}_{base_filename}.txt'
            report_file = os.path.join('results', report_filename)

            with open(report_file, 'w') as f:
                f.write(f"Model: {model}\n")
                f.write(f"Type: {type_}\n")
                f.write(f"Accuracy: {result['accuracy']}\n\n")
                f.write("Classification Report:\n")
                f.write(result['report'])
            print(f"Detailed classification report saved to {report_file}")

    return train_sample_df, all_feature_activations, classification_results


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('medium')
    model_params = {
        'learning_rate': 1e-3, # 5e-5 for sae, 1e-3 for st
        'batch_size': 1024,
        'reconstruction_error_threshold': 999999999,
        'force_retrain': False,
        'l1_lambda': 1, # For ST attention dimension also controls sparsity
    }

    train_dataset = "data/mnist_train.csv"
    val_dataset = "data/mnist_test.csv"
    # List all feature columns (excluding label column)
    feature_columns = [str(i) for i in range(784)]  # MNIST is 28x28=784 pixels
    label_column = "label"
    models = ["mnist"]  # Dummy model name for consistency
    n_train = 30_000  # Adjust as needed
    n_val = 10000
    gephi_subset_size = 10000
    tsd, afa, cr = run_all(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        models=models,
        n_train=n_train,
        n_val=n_val,
        feature_column=feature_columns,  # Pass list of feature columns
        label_column=label_column,
        data_type='vector',  # Specify vector data type: 'vector' for vector, 'text' for text
        model_params=model_params,
        create_graph=True,
        n_random_labels=8,
        force_new_embeddings=False,
        perform_classification=False,
        model_type="st",
        gephi_subset_size=gephi_subset_size
    )




"""
    train_dataset = "data/stack_exchange_train.csv"
    val_dataset = "data/stack_exchange_val.csv"
    feature_columns = "sentences"
    label_column = "labels"
    models = ["Alibaba-NLP/gte-large-en-v1.5"]
    n_max = pd.read_csv("data/stack_exchange_train.csv").shape[0]
    n_train = 100_000
    n_val = 1000
    gephi_subset_size = 10000
"""