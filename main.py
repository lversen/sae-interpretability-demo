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
    Load or train a model with consistent preprocessing and loss computation for both SAE and ST models.
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

            # Check model quality with proper preprocessing and loss computation
            with torch.no_grad():
                if isinstance(model, SparseAutoencoder):
                    # Preprocess validation data
                    C = model.preprocess(val_feature_extract[:100])
                    val_preprocessed = val_feature_extract[:100] / C
                    # Get model outputs
                    x, x_hat, f_x = model(val_preprocessed)
                    # Compute loss using model's loss function
                    total_loss, reconstruction_error, _ = model.compute_loss(x, x_hat, f_x)
                else:  # SparseTransformer
                    # Preprocess validation data
                    C = model.preprocess(val_feature_extract[:100])
                    val_preprocessed = val_feature_extract[:100] / C
                    
                    # Get model outputs
                    x, x_hat, f, v = model(val_preprocessed)
                    # Compute loss using model's loss function
                    total_loss, reconstruction_error, _ = model.compute_loss(x, x_hat, f, v)

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

def create_multi_model_gephi_graphs(
    original_features: np.ndarray,
    sae_features: np.ndarray,
    st_features: np.ndarray,
    df: pd.DataFrame,
    label_column: str,
    base_filename: str,
    selected_labels: Optional[List[str]] = None,
    category_column: Optional[str] = None
):
    """
    Create Gephi graphs for original features, SAE, and ST models simultaneously.
    
    Args:
        original_features: Original feature embeddings
        sae_features: SAE model feature activations
        st_features: ST model feature activations
        df: DataFrame containing metadata
        label_column: Column name for labels
        base_filename: Base filename for exports
        selected_labels: Optional list of labels to include
        category_column: Optional category column name
    """
    # Create base export directory
    export_dir = f"gephi_exports_{base_filename}"
    os.makedirs(export_dir, exist_ok=True)
    
    # Create Gephi graphs for each representation
    models = {
        'original': original_features,
        'sae': sae_features,
        'st': st_features
    }
    
    for model_name, features in models.items():
        file_path = os.path.join(export_dir, f"{base_filename}_{model_name}.gexf")
        
        create_gephi_graph(
            features,
            df,
            label_column,
            model_name,
            file_path,
            selected_labels=selected_labels,
            category_column=category_column
        )
        print(f"Created {model_name} graph at {file_path}")

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
    model_type: str = "both",  # Changed to support both models
    gephi_subset_size: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run the complete analysis pipeline with support for both SAE and ST models.
    
    Args:
        train_dataset: Path to training dataset
        val_dataset: Path to validation dataset
        models: List of model names to use
        n_train: Number of training samples
        n_val: Number of validation samples
        feature_column: Column(s) containing features
        label_column: Column containing labels
        data_type: Type of data ('text' or 'vector')
        perform_classification: Whether to perform classification
        create_graph: Whether to create Gephi graphs
        n_random_labels: Number of random labels for Gephi visualization
        force_new_embeddings: Whether to force new embedding computation
        classify_language: Languages to classify
        top_n_category: Category selection parameters
        model_params: Model hyperparameters
        model_type: Type of model to use ('sae', 'st', or 'both')
        gephi_subset_size: Size of subset for Gephi visualization
    
    Returns:
        Tuple containing:
        - DataFrame with training samples
        - Dictionary of feature activations
        - Dictionary of classification results
    """
    all_feature_activations = {}
    classification_results = {}

    print(f"Processing train dataset: {train_dataset}")
    print(f"Model type: {model_type}")

    train_df = pd.read_csv(train_dataset)
    val_df = pd.read_csv(val_dataset)

    feature_extraction_fn = get_feature_extraction_fn(data_type)

    for model in models:
        print(f"\nProcessing model: {model}")

        # Get consistent samples for training
        train_sample_df, train_indices = get_consistent_samples(
            train_df, n_train, f"{train_dataset}_train", model)
        val_sample_df, val_indices = get_consistent_samples(
            val_df, n_val, f"{val_dataset}_val", model)

        # Extract features based on data type
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

        # Model initialization and training
        n = train_feature_extract.shape[1]  # Input dimension
        m = 100  # Feature dimension
        l1_lambda = model_params.get('l1_lambda', 5)
        
        # Store original features
        all_feature_activations[f"{train_dataset}_{model}_original"] = train_feature_extract
        
        # Train SAE model if requested
        if model_type in ["sae", "both"]:
            print("\nTraining SAE model...")
            sae_model_path = f'models/sae_model_{os.path.basename(train_dataset)}_{model.replace("/", "_")}.pth'
            sae_model = SparseAutoencoder(n, m, sae_model_path, l1_lambda)
            sae_model = load_or_train_model(
                sae_model,
                train_feature_extract,
                val_feature_extract,
                model_path=sae_model_path,
                learning_rate=model_params.get('learning_rate', 5e-5),
                batch_size=model_params.get('batch_size', 4096),
                reconstruction_error_threshold=model_params.get('reconstruction_error_threshold', 999999999),
                force_retrain=model_params.get('force_retrain', False)
            )
            with torch.no_grad():
                sae_activations = sae_model.feature_activations(
                    torch.from_numpy(train_feature_extract).float().to(sae_model.device))
                sae_activations = sae_activations.cpu().numpy()
                all_feature_activations[f"{train_dataset}_{model}_sae"] = sae_activations

        # Train ST model if requested
        if model_type in ["st", "both"]:
            print("\nTraining ST model...")
            st_model_path = f'models/st_model_{os.path.basename(train_dataset)}_{model.replace("/", "_")}.pth'
            a = 64  # Attention dimension
            st_model = SparseTransformer(
                X=train_feature_extract,
                n=n,
                m=m,
                a=a,
                st_model_path=st_model_path,
                lambda_l1=l1_lambda,
                num_heads=1
            )
            st_model = load_or_train_model(
                st_model,
                train_feature_extract,
                val_feature_extract,
                model_path=st_model_path,
                learning_rate=model_params.get('learning_rate', 1e-3),
                batch_size=model_params.get('batch_size', 4096),
                reconstruction_error_threshold=model_params.get('reconstruction_error_threshold', 999999999),
                force_retrain=model_params.get('force_retrain', False)
            )
            with torch.no_grad():
                st_activations = st_model.feature_activations(
                    torch.from_numpy(train_feature_extract).float().to(st_model.device))
                st_activations = st_activations.cpu().numpy()
                all_feature_activations[f"{train_dataset}_{model}_st"] = st_activations

        # Perform classification if requested
        if perform_classification and label_column in train_sample_df.columns:
            print("\nPerforming classification...")
            for model_suffix in ['original', 'sae', 'st']:
                if f"{train_dataset}_{model}_{model_suffix}" in all_feature_activations:
                    features = all_feature_activations[f"{train_dataset}_{model}_{model_suffix}"]
                    clf, accuracy, report = train_and_evaluate_decision_tree(
                        features, train_sample_df[label_column])
                    classification_results[f"{model}_{model_suffix}"] = {
                        "accuracy": accuracy,
                        "report": report
                    }

        # Create Gephi graphs if requested
        if create_graph:
            print("\nCreating Gephi graphs...")
            # Create subset for visualization if specified
            if gephi_subset_size is not None and gephi_subset_size < len(train_sample_df):
                gephi_df, gephi_indices = get_consistent_samples(
                    train_sample_df, gephi_subset_size, 
                    f"{train_dataset}_gephi", model)
                
                # Subset all feature activations
                gephi_features = {
                    'original': train_feature_extract[gephi_indices],
                    'sae': sae_activations[gephi_indices] if model_type in ["sae", "both"] else None,
                    'st': st_activations[gephi_indices] if model_type in ["st", "both"] else None
                }
            else:
                gephi_df = train_sample_df
                gephi_features = {
                    'original': train_feature_extract,
                    'sae': sae_activations if model_type in ["sae", "both"] else None,
                    'st': st_activations if model_type in ["st", "both"] else None
                }

            # Get selected labels for visualization
            selected_labels = select_random_labels(
                gephi_df,
                label_column,
                n_random_labels=n_random_labels,
                category_column='assigned_category' if 'assigned_category' in gephi_df.columns else None
            )

            # Create Gephi graphs for each representation
            base_filename = os.path.splitext(os.path.basename(train_dataset))[0]
            
            # Process each active model type
            for model_suffix, features in gephi_features.items():
                if features is not None:
                    file_path = f"gephi_exports_{base_filename}/{base_filename}_{model}_{model_suffix}.gexf"
                    create_gephi_graph(
                        features,
                        gephi_df,
                        label_column,
                        f"{model}_{model_suffix}",
                        file_path,
                        selected_labels=selected_labels,
                        category_column='assigned_category' if 'assigned_category' in gephi_df.columns else None
                    )
                    print(f"Created {model_suffix} graph at {file_path}")

    # Print summary statistics
    print("\nSummary of feature activations:")
    for key, features in all_feature_activations.items():
        non_zero = np.count_nonzero(features)
        total = features.size
        sparsity = (1 - non_zero/total) * 100
        print(f"\n{key}:")
        print(f"Shape: {features.shape}")
        print(f"Non-zero elements: {non_zero:,}")
        print(f"Sparsity: {sparsity:.2f}%")

    return train_sample_df, all_feature_activations, classification_results


if __name__ == "__main__":
    # Model parameters
    model_params = {
        'learning_rate': 5e-5,  # 5e-5 for SAE, 1e-3 for ST
        'batch_size': 4096,
        'reconstruction_error_threshold': 999999999,
        'force_retrain': False,
        'l1_lambda': 5,  # L1 regularization strength
    }

    # Dataset configuration
    train_dataset = "data/mnist_train.csv"
    val_dataset = "data/mnist_test.csv"
    feature_columns = [str(i) for i in range(784)]  # MNIST is 28x28=784 pixels
    label_column = "label"
    models = ["mnist"]  # Model identifier
    
    # Sample sizes
    n_train = 60_000  # Full MNIST training set
    n_val = 10_000    # Full MNIST test set
    gephi_subset_size = 10000  # Size of subset for visualization
    
    # Run the analysis with both SAE and ST models
    tsd, afa, cr = run_all(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        models=models,
        n_train=n_train,
        n_val=n_val,
        feature_column=feature_columns,
        label_column=label_column,
        data_type='vector',
        model_params=model_params,
        create_graph=True,
        n_random_labels=8,
        force_new_embeddings=False,
        perform_classification=False,
        model_type="both",  # This will train and visualize both SAE and ST
        gephi_subset_size=gephi_subset_size
    )

    # Print shapes of feature activations for comparison
    print("\nFeature Activation Shapes:")
    for key, features in afa.items():
        non_zero = np.count_nonzero(features)
        total = features.size
        sparsity = (1 - non_zero/total) * 100
        print(f"\n{key}:")
        print(f"Shape: {features.shape}")
        print(f"Non-zero elements: {non_zero:,}")
        print(f"Sparsity: {sparsity:.2f}%")


"""

    train_dataset = "data/stack_exchange_train.csv"
    val_dataset = "data/stack_exchange_val.csv"
    feature_columns = "sentences"
    label_column = "labels"
    models = ["Alibaba-NLP/gte-large-en-v1.5"]
    train_max = pd.read_csv("data/stack_exchange_train.csv").shape[0]
    val_max = pd.read_csv("data/stack_exchange_val.csv").shape[0]
    n_train = 10_000
    n_val = 1000
    gephi_subset_size = 10000

"""