import re
import os
import sys
import subprocess
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, deque
import random
import torch
from feature_extraction_with_store import feature_extraction_with_store
from gephi import *
from language_classification import language_classifier
from sample_handler import get_consistent_samples
from SAE import SparseAutoencoder
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


# Add this at the top of the file with other imports


def sanitize_filename(filename):
    # Replace invalid filename characters with underscores
    return re.sub(r'[<>:"/\\|?*]', '_', filename)


def run_all(
    train_dataset: str,
    val_dataset: str,
    models: List[str],
    n_train: int,
    n_val: int,
    feature_column: str,
    label_column: str,
    perform_classification: bool = True,
    create_graph: bool = False,
    n_random_labels: Optional[int] = None,
    force_new_embeddings: bool = False,
    classify_language: List[str] = [],
    top_n_category: Optional[Dict[str, Dict[str, Any]]] = None,
    sae_params: Dict[str, Any] = {}
) -> Dict[str, Any]:
    all_feature_activations = {}
    classification_results = {}

    print(f"Processing train dataset: {train_dataset}")


    train_df = pd.read_csv(train_dataset)
    val_df = pd.read_csv(val_dataset)


    for model in models:
        print(f"Processing model: {model}")

        train_sample_df, train_indices = get_consistent_samples(
            train_df, n_train, f"{train_dataset}_train", model)
        val_sample_df, val_indices = get_consistent_samples(
            val_df, n_val, f"{val_dataset}_val", model)



        train_feature_extract = feature_extraction_with_store(
            train_sample_df, train_df, model, len(train_sample_df), f"{train_dataset}_train", feature_column,
            force_new_embeddings=force_new_embeddings
        )

        val_feature_extract = feature_extraction_with_store(
            val_sample_df, val_df, model, len(val_sample_df), f"{val_dataset}_val", feature_column,
            force_new_embeddings=force_new_embeddings
        )

        if perform_classification and label_column in train_sample_df.columns:
            # Perform decision tree classification on the original embeddings
            print("Performing decision tree classification on original embeddings")
            original_clf, original_accuracy, original_report = train_and_evaluate_decision_tree(
                train_feature_extract, train_sample_df[label_column])
            classification_results[f"{model}_original"] = {
                "accuracy": original_accuracy,
                "report": original_report
            }

        # Initialize and train/load SAE
        D = train_feature_extract.shape[1]
        F = 2 * D
        l1_lambda = 5
        sae_model_path = f'models/sae_model_{os.path.basename(train_dataset)}_{model.replace("/", "_")}.pth'
        sae = SparseAutoencoder(D, F, sae_model_path, l1_lambda)
        sae = load_or_train_sae(
            sae,
            train_feature_extract,
            val_feature_extract,
            model_path=sae_model_path,
            learning_rate=sae_params.get('learning_rate', 1e-3),
            batch_size=sae_params.get('batch_size', 40),
            num_epochs=sae_params.get('num_epochs', 20),
            reconstruction_error_threshold=sae_params.get('reconstruction_error_threshold', 0.1),
            force_retrain=sae_params.get('force_retrain', False)
        )

        # Get feature activations (using validation set for consistency)
        with torch.no_grad():
            feature_activations = sae.feature_activations(
                torch.from_numpy(train_feature_extract).float().to(sae.device))
            # Move to CPU and convert to numpy immediately
            feature_activations = feature_activations.cpu().numpy()
            all_feature_activations[f"{train_dataset}_{model}"] = feature_activations
        if perform_classification and label_column in train_sample_df.columns:
            # Perform decision tree classification on the SAE feature activations
            print("Performing decision tree classification on SAE feature activations")
            sae_clf, sae_accuracy, sae_report = train_and_evaluate_decision_tree(
                all_feature_activations[f"{train_dataset}_{model}"], train_sample_df[label_column])
            classification_results[f"{model}_sae"] = {
                "accuracy": sae_accuracy,
                "report": sae_report
            }

        if create_graph:
            # First, select random labels that will be used for both graphs
            selected_labels = select_random_labels(
                train_sample_df,
                label_column,
                n_random_labels=n_random_labels,
                category_column='assigned_category' if 'assigned_category' in train_sample_df.columns else None
            )
            
            # Create graph for original embeddings
            base_filename = os.path.splitext(os.path.basename(train_dataset))[0]
            sanitized_model = model.replace('.', '-')  # Replace dots with hyphens in model name
            original_file_path = f"gephi_exports_{base_filename}/{base_filename}_{sanitized_model}_original.gexf"
            create_gephi_graph(
                train_feature_extract,
                train_sample_df,
                label_column,
                f"{sanitized_model}_original",
                original_file_path,
                selected_labels=selected_labels,
                category_column='assigned_category' if 'assigned_category' in train_sample_df.columns else None
            )
            
            # Create graph for SAE embeddings
            sae_file_path = f"gephi_exports_{base_filename}/{base_filename}_{sanitized_model}_sae.gexf"
            create_gephi_graph(
                feature_activations,
                train_sample_df,
                label_column,
                f"{sanitized_model}_sae",
                sae_file_path,
                selected_labels=selected_labels,
                category_column='assigned_category' if 'assigned_category' in train_sample_df.columns else None
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
        results_file = os.path.join('results', f'classification_results_{base_filename}.csv')
        results_df.to_csv(results_file, index=False)
        print(f"Classification results saved to {results_file}")

        # Save detailed classification reports
        for key, result in classification_results.items():
            model, type_ = key.rsplit('_', 1)
            sanitized_model = sanitize_filename(model)
            report_filename = f'classification_report_{sanitized_model}_{type_}_{base_filename}.txt'
            report_file = os.path.join('results', report_filename)

            with open(report_file, 'w') as f:
                f.write(f"Model: {model}\n")
                f.write(f"Type: {type_}\n")
                f.write(f"Accuracy: {result['accuracy']}\n\n")
                f.write("Classification Report:\n")
                f.write(result['report'])
            print(f"Detailed classification report saved to {report_file}")

    return train_sample_df, all_feature_activations, classification_results

def restart_kernel():
    """
    Restarts the current Python kernel/process.
    
    This function will:
    1. Get the current Python executable path
    2. Get the current script path
    3. Execute a new process with the same script
    4. Exit the current process
    """
    python = sys.executable
    script = sys.argv[0]
    
    print("Restarting kernel...")
    try:
        # Start new process
        subprocess.Popen([python, script])
        # Exit current process
        os._exit(0)
    except Exception as e:
        print(f"Error restarting kernel: {e}")

if __name__ == "__main__":
    train_dataset = "data/stack_exchange_train.csv"
    val_dataset = "data/stack_exchange_val.csv"
    feature_column = "sentences"
    label_column = "labels"
    models = ["Alibaba-NLP/gte-large-en-v1.5"]
    n_max = pd.read_csv("data/stack_exchange_train.csv").shape[0]
    n_train = n_max
    n_val = 1000

    # SAE hyperparameters
    sae_params = {
        'learning_rate': 1e-3,
        'batch_size': 32,
        'num_epochs': 100,
        'reconstruction_error_threshold': 20,
        'force_retrain': True
    }
    df, feature_activations, classification_results = run_all(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        models=models,
        n_train=n_train,
        n_val=n_val,
        feature_column=feature_column,
        label_column=label_column,
        sae_params=sae_params,
        create_graph=False,
        n_random_labels=8,
        force_new_embeddings=False,
        perform_classification=False
    )
    
    user_input = input("Restart kernel to release memory? y/n: ")
    if user_input.lower().strip() == 'y':
        restart_kernel()
    else:
        print("Kernel not restarting, memory will not be released.")        

