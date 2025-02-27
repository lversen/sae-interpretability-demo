import os
os.environ["LOKY_MAX_CPU_COUNT"] = "15"
import argparse
import pandas as pd
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple, Union
import random
import glob
from feature_extraction_with_store import feature_extraction_with_store
from sample_handler import get_consistent_samples
from SAE import SparseAutoencoder
from ST import SparseTransformer
from gephi import create_gephi_graph, select_random_labels
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import json

# Add custom dataset configurations
DATASET_CONFIGS = {
    "mnist": {
        "train_dataset": "data/mnist_train.csv",
        "val_dataset": "data/mnist_test.csv",
        "data_type": "vector",
        "feature_column": [str(i) for i in range(784)],
        "label_column": "label",
        "input_dimension": 784,
    },
    "stack_exchange": {
        "train_dataset": "data/stack_exchange_train.csv",
        "val_dataset": "data/stack_exchange_val.csv",
        "data_type": "text",
        "feature_column": "sentences",
        "label_column": "labels",
        "input_dimension": 1024,
    }
}

# Add LLM embedding model configurations
LLM_MODELS = {
    "gte-large": "Alibaba-NLP/gte-large-en-v1.5"
}

def save_config_to_file(args, filename="last_config.json"):
    """Save the current configuration to a JSON file"""
    # Convert args to dictionary
    config_dict = vars(args)
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(config_dict, f, indent=4)
    
    print(f"Configuration saved to {filename}")

def load_config_from_file(filename="last_config.json"):
    """Load configuration from a JSON file"""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            config_dict = json.load(f)
        return config_dict
    return None

def list_available_datasets():
    """List all available datasets in the data directory"""
    datasets = glob.glob("data/*.csv")
    return [os.path.basename(ds) for ds in datasets]

def parse_args():
    """Parse command line arguments with enhanced dataset and model options"""
    parser = argparse.ArgumentParser(description='Run SAE and ST training with flexible dataset and model configuration')
    
    # Dataset selection
    dataset_group = parser.add_argument_group('Dataset Configuration')
    dataset_group.add_argument('--dataset', type=str, default='mnist', 
                        choices=list(DATASET_CONFIGS.keys()), 
                        help='Predefined dataset configuration to use')
    dataset_group.add_argument('--list_datasets', action='store_true',
                        help='List all available datasets in the data directory')
    dataset_group.add_argument('--train_dataset', type=str, default=None, 
                        help='Path to training dataset (overrides dataset config)')
    dataset_group.add_argument('--val_dataset', type=str, default=None,
                        help='Path to validation dataset (overrides dataset config)')
    dataset_group.add_argument('--data_type', type=str, default=None, 
                        choices=['text', 'vector'],
                        help='Type of data (text or vector, overrides dataset config)')
    dataset_group.add_argument('--feature_column', type=str, nargs='+', default=None,
                        help='Column(s) containing features (overrides dataset config)')
    dataset_group.add_argument('--label_column', type=str, default=None,
                        help='Column containing labels (overrides dataset config)')
    dataset_group.add_argument('--input_dimension', type=int, default=None,
                        help='Input dimension (n) (overrides dataset config)')
    
    # Embedding model selection for text data
    embedding_group = parser.add_argument_group('Embedding Model Configuration')
    embedding_group.add_argument('--embedding_model', type=str, default='gte-large',
                        choices=list(LLM_MODELS.keys()),
                        help='Embedding model to use for text data')
    embedding_group.add_argument('--custom_embedding_model', type=str, default=None,
                        help='Custom HuggingFace model path for embeddings')
    embedding_group.add_argument('--force_reembedding', action='store_true',
                        help='Force recomputation of embeddings')
    
    # Model parameters
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--model_id', type=str, default='default',
                        help='Model identifier')
    model_group.add_argument('--model_type', type=str, default='both', 
                        choices=['sae', 'st', 'both'],
                        help='Type of model to train')
    model_group.add_argument('--n_train', type=int, default=None,
                        help='Number of training samples to use')
    model_group.add_argument('--n_val', type=int, default=None,
                        help='Number of validation samples to use')
    model_group.add_argument('--feature_dimension', type=int, default=None,
                        help='Feature dimension (m), defaults to 8*n if not specified')
    model_group.add_argument('--attention_dimension', type=int, default=None,
                        help='Attention dimension (a) for ST, defaults to n/2 if not specified')
    
    # Training parameters
    training_group = parser.add_argument_group('Training Configuration')
    training_group.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate')
    training_group.add_argument('--batch_size', type=int, default=4096,
                        help='Batch size')
    training_group.add_argument('--l1_lambda', type=float, default=5.0,
                        help='L1 regularization strength')
    training_group.add_argument('--target_steps', type=int, default=200000,
                        help='Target number of training steps')
    training_group.add_argument('--force_retrain', action='store_true',
                        help='Force retraining of models')
    
    # ST-specific parameters
    st_group = parser.add_argument_group('ST Model Configuration')
    st_group.add_argument('--use_mixed_precision', action='store_true',
                        help='Enable mixed precision training for ST model')
    st_group.add_argument('--activation_threshold', type=float, default=1e-3,
                        help='Activation threshold for ST feature tracking')
    st_group.add_argument('--grad_accum_steps', type=int, default=1,
                        help='Number of gradient accumulation steps for ST model')
    st_group.add_argument('--eval_freq', type=int, default=None,
                        help='Evaluation frequency during training (steps)')
    
    # Misc parameters
    misc_group = parser.add_argument_group('Miscellaneous Configuration')
    misc_group.add_argument('--visualize_decoder', action='store_true',
                        help='Visualize decoder matrix after training')
    misc_group.add_argument('--perform_classification', action='store_true',
                        help='Perform classification on the learned features')
    misc_group.add_argument('--create_graph', action='store_true',
                        help='Create Gephi graph visualization')
    misc_group.add_argument('--gephi_subset_size', type=int, default=1000,
                        help='Size of subset for Gephi visualization')
    misc_group.add_argument('--n_random_labels', type=int, default=10,
                        help='Number of random labels to select for graph visualization')
    misc_group.add_argument('--graph_neighbors', type=int, default=4,
                        help='Number of neighbors for graph creation')
    misc_group.add_argument('--save_config', action='store_true',
                        help='Save current configuration to a JSON file')
    misc_group.add_argument('--load_config', type=str, default=None,
                        help='Load configuration from a JSON file')
    
    args = parser.parse_args()
    
    # If list_datasets flag is set, list available datasets and exit
    if args.list_datasets:
        print("Available datasets in data directory:")
        datasets = list_available_datasets()
        for i, ds in enumerate(datasets):
            print(f"  {i+1}. {ds}")
        exit(0)
    
    # Load config from file if specified
    if args.load_config:
        loaded_config = load_config_from_file(args.load_config)
        if loaded_config:
            # Update args with loaded config, but keep command line overrides
            for k, v in loaded_config.items():
                if not hasattr(args, k) or getattr(args, k) is None or (
                    isinstance(getattr(args, k), bool) and 
                    k not in ['force_retrain', 'force_reembedding', 'use_mixed_precision',
                             'visualize_decoder', 'perform_classification', 'create_graph',
                             'save_config']
                ):
                    setattr(args, k, v)
            print(f"Loaded configuration from {args.load_config}")
    
    # Apply dataset defaults first, then override with explicit parameters
    dataset_config = DATASET_CONFIGS.get(args.dataset, DATASET_CONFIGS['mnist'])
    
    # Apply dataset configuration
    if args.train_dataset is None:
        args.train_dataset = dataset_config['train_dataset']
    if args.val_dataset is None:
        args.val_dataset = dataset_config['val_dataset']
    if args.data_type is None:
        args.data_type = dataset_config['data_type']
    if args.feature_column is None:
        args.feature_column = dataset_config['feature_column']
    if args.label_column is None:
        args.label_column = dataset_config['label_column']
    if args.input_dimension is None:
        args.input_dimension = dataset_config['input_dimension']
    
    # Apply default values for training/validation set sizes if not specified
    if args.n_train is None:
        # Use a reasonable default based on dataset
        if 'mnist' in args.train_dataset:
            args.n_train = 60000
        elif 'stack_exchange' in args.train_dataset:
            args.n_train = 60000
        else:
            args.n_train = 10000  # Default for other datasets
    
    if args.n_val is None:
        # Use a reasonable default based on dataset
        if 'mnist' in args.val_dataset:
            args.n_val = 10000
        elif 'stack_exhange' in args.val_dataset:
            args.n_val = 10000
        else:
            args.n_val = 2000  # Default for other datasets
    
    # Resolve embedding model
    if args.custom_embedding_model:
        args.embedding_model_path = args.custom_embedding_model
    else:
        args.embedding_model_path = LLM_MODELS.get(args.embedding_model, LLM_MODELS['gte-large'])
    
    # Save config if requested
    if args.save_config:
        save_config_to_file(args)
    
    return args

def get_feature_extraction_fn(data_type: str):
    """
    Factory function to return appropriate feature extraction function based on data type.
    """
    if data_type == 'text':
        return feature_extraction_with_store
    elif data_type == 'vector':
        return direct_vector_feature_extraction
    else:
        raise ValueError(f"Unknown data type: {data_type}")

def direct_vector_feature_extraction(df: pd.DataFrame, full_df: pd.DataFrame = None, 
                                   model: str = None, n: int = None,
                                   dataset_name: str = None, content_column: str = None,
                                   feature_columns: list = None, force_new_embeddings: bool = False,
                                   **kwargs) -> np.ndarray:
    """
    Extract features directly from dataframe columns that already contain vector data.
    
    Updated to match the signature of feature_extraction_with_store for compatibility.
    """
    feature_matrix = df[feature_columns].values if feature_columns else df[content_column].values
    return feature_matrix.astype(np.float32)

def train_and_evaluate_decision_tree(X, y, test_size=0.2, random_state=42):
    """Train and evaluate a decision tree classifier"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    
    clf = DecisionTreeClassifier(random_state=random_state)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return clf, accuracy, report

def visualize_feature_activations(activations, title="Feature Activations", figsize=(15, 5)):
    """Visualize feature activations as a heatmap"""
    plt.figure(figsize=figsize)
    plt.imshow(activations.T, aspect='auto', cmap='viridis')
    plt.colorbar(label='Activation Strength')
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('Feature Index')
    plt.tight_layout()
    plt.show()

def create_graphs(df, feature_activations, args):
    """
    Create Gephi graph visualizations for different feature extraction methods.
    
    Args:
        df: DataFrame containing the data
        feature_activations: Dictionary mapping model name to feature matrix
        args: Command line arguments
    """
    print("\n" + "="*50)
    print("Creating Gephi graph visualizations...")
    print("="*50)
    
    # Create output directory for graph files
    graphs_dir = 'graphs'
    os.makedirs(graphs_dir, exist_ok=True)
    
    # Take a subset for visualization if needed
    subset_size = min(args.gephi_subset_size, len(df))
    subset_df = df.iloc[:subset_size].copy()
    
    # Determine title column and category column
    title_column = args.label_column  # Use label column as node title by default
    category_column = args.label_column  # Use the same for category
    
    # Select random labels for consistent visualization
    selected_labels = select_random_labels(
        subset_df, 
        title_column=title_column, 
        n_random_labels=args.n_random_labels,
        category_column=category_column
    )
    
    print(f"Selected labels for visualization: {selected_labels}")
    
    # Process each feature extraction method
    for model_name, features in feature_activations.items():
        print(f"\nCreating graph for: {model_name}")
        
        # Take the same subset of features
        subset_features = features[:subset_size]
        
        # Create graph file path
        graph_file = os.path.join(graphs_dir, f"{args.model_id}_{model_name}.gexf")
        
        # Create the graph
        create_gephi_graph(
            feature_extract=subset_features,
            df=subset_df,
            title_column=title_column,
            model_name=model_name,
            file_path=graph_file,
            selected_labels=selected_labels,
            category_column=category_column,
            n_neighbors=args.graph_neighbors
        )
        
        print(f"Graph saved to: {graph_file}")
    
    print("\nGraph creation completed!")

def main():
    """Main function to run the SAE and ST training with enhanced dataset and model options"""
    args = parse_args()
    
    # Print configuration summary
    print("\n" + "="*50)
    print("CONFIGURATION SUMMARY")
    print("="*50)
    print(f"Dataset: {args.dataset}")
    print(f"  Train: {args.train_dataset} (n={args.n_train})")
    print(f"  Val: {args.val_dataset} (n={args.n_val})")
    print(f"  Data Type: {args.data_type}")
    if args.data_type == 'text':
        print(f"  Embedding Model: {args.embedding_model} ({args.embedding_model_path})")
    print(f"  Feature Column(s): {args.feature_column}")
    print(f"  Label Column: {args.label_column}")
    print(f"Model: {args.model_type.upper()}")
    print(f"  Input Dimension (n): {args.input_dimension}")
    print(f"  Feature Dimension (m): {args.feature_dimension or args.input_dimension*8}")
    if args.model_type in ['st', 'both']:
        print(f"  Attention Dimension (a): {args.attention_dimension or args.input_dimension//2}")
    print(f"Training:")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Target Steps: {args.target_steps}")
    print(f"  L1 Lambda: {args.l1_lambda}")
    print(f"  Force Retrain: {args.force_retrain}")
    print("="*50)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Setup feature dimensions
    n = args.input_dimension
    m = args.feature_dimension if args.feature_dimension else 8 * n
    a = args.attention_dimension if args.attention_dimension else 64
    
    # Model parameters
    model_params = {
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'target_steps': args.target_steps,
        'l1_lambda': args.l1_lambda,
        'force_retrain': args.force_retrain,
    }
    
    # Load datasets
    try:
        print(f"Loading datasets: {args.train_dataset}, {args.val_dataset}")
        train_df = pd.read_csv(args.train_dataset)
        val_df = pd.read_csv(args.val_dataset)
        
        print(f"Train dataset shape: {train_df.shape}")
        print(f"Validation dataset shape: {val_df.shape}")
    except Exception as e:
        print(f"Error loading datasets: {e}")
        print("Please check that the dataset files exist and are valid CSV files.")
        return
    
    # Get feature extraction function
    feature_extraction_fn = get_feature_extraction_fn(args.data_type)
    
    # Get consistent samples
    print(f"Getting consistent samples for training and validation")
    train_sample_df, train_indices = get_consistent_samples(
        train_df, args.n_train, f"{os.path.basename(args.train_dataset)}_train", args.model_id)
    val_sample_df, val_indices = get_consistent_samples(
        val_df, args.n_val, f"{os.path.basename(args.val_dataset)}_val", args.model_id)
    
    # Extract features
    print(f"Extracting features...")
    if args.data_type == 'text':
        print(f"Using embedding model: {args.embedding_model_path}")
        train_feature_extract = feature_extraction_fn(
            train_sample_df, train_df, args.embedding_model_path, len(train_sample_df),
            f"{os.path.basename(args.train_dataset)}_train", args.feature_column[0] if isinstance(args.feature_column, list) else args.feature_column,
            force_new_embeddings=args.force_reembedding
        )
        val_feature_extract = feature_extraction_fn(
            val_sample_df, val_df, args.embedding_model_path, len(val_sample_df),
            f"{os.path.basename(args.val_dataset)}_val", args.feature_column[0] if isinstance(args.feature_column, list) else args.feature_column,
            force_new_embeddings=args.force_reembedding
        )
    else:
        train_feature_extract = feature_extraction_fn(
            train_sample_df, train_df, 
            feature_columns=args.feature_column if isinstance(args.feature_column, list) else [args.feature_column]
        )
        val_feature_extract = feature_extraction_fn(
            val_sample_df, val_df, 
            feature_columns=args.feature_column if isinstance(args.feature_column, list) else [args.feature_column]
        )
    
    print(f"Feature matrix shapes - Train: {train_feature_extract.shape}, Val: {val_feature_extract.shape}")
    
    # Update input dimension if needed based on actual feature dimensionality
    if train_feature_extract.shape[1] != n:
        print(f"Input dimension updated from {n} to {train_feature_extract.shape[1]} based on extracted features")
        n = train_feature_extract.shape[1]
        # Update m and a accordingly if they were not explicitly set
        if args.feature_dimension is None:
            m = 8 * n
        if args.attention_dimension is None:
            a = 64
    
    # Create model directories
    os.makedirs('models', exist_ok=True)
    
    # Create tensors
    train_tensor = torch.from_numpy(train_feature_extract).float().to(device)
    val_tensor = torch.from_numpy(val_feature_extract).float().to(device)
    
    # Initialize results
    all_feature_activations = {}
    classification_results = {}
    
    # Store original features
    all_feature_activations[f"original"] = train_feature_extract
    
    # Train SAE model if requested
    if args.model_type in ["sae", "both"]:
        print("\n" + "="*50)
        print("Training SAE model...")
        print("="*50)
        
        dataset_name = os.path.splitext(os.path.basename(args.train_dataset))[0]
        model_suffix = f"{dataset_name}_{args.model_id}"
        if args.data_type == 'text':
            model_suffix += f"_{args.embedding_model}"
        
        sae_model_path = f'models/sae_model_{model_suffix}.pth'

        # Check if the SAE model file exists, and if so, examine its dimensions
        if os.path.exists(sae_model_path):
            try:
                # Load the model state dict to check dimensions
                checkpoint = torch.load(sae_model_path, map_location=device)
                
                # Extract state dict
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                # Determine feature dimension (m) from the model
                if 'W_e.weight' in state_dict:
                    saved_m = state_dict['W_e.weight'].shape[0]
                    print(f"Detected feature dimension from saved model: m={saved_m}")
                    # Use the detected dimension
                    m = saved_m
                else:
                    print(f"Could not detect feature dimension from model, using specified value: m={m}")
            except Exception as e:
                print(f"Error checking model dimensions: {e}")
                print(f"Using specified feature dimension: m={m}")

        # Create the SAE model with the correct dimensions
        sae_model = SparseAutoencoder(n, m, sae_model_path, args.l1_lambda, device)
        
        # Train or load model
        if args.force_retrain or not os.path.exists(sae_model_path):
            print(f"Training SAE from scratch...")
            sae_model.train_and_validate(
                train_tensor,
                val_tensor,
                learning_rate=model_params['learning_rate'],
                batch_size=model_params['batch_size'],
                target_steps=model_params['target_steps']
            )
        else:
            print(f"Loading pre-trained SAE model from {sae_model_path}")
            sae_model.load_state_dict(torch.load(sae_model_path))
            print(f"Model loaded successfully.")
        
        # Calculate feature activations
        with torch.no_grad():
            sae_activations = sae_model.feature_activations(train_tensor)
            sae_activations = sae_activations.cpu().numpy()
            all_feature_activations["sae"] = sae_activations
        
        print(f"SAE model saved to {sae_model_path}")
        
        if args.visualize_decoder:
            print("\nVisualizing SAE feature activations...")
            visualize_feature_activations(sae_activations, "SAE Feature Activations")
    
    # Train ST model if requested
    if args.model_type in ["st", "both"]:
        print("\n" + "="*50)
        print("Training ST model...")
        print("="*50)
        
        dataset_name = os.path.splitext(os.path.basename(args.train_dataset))[0]
        model_suffix = f"{dataset_name}_{args.model_id}"
        if args.data_type == 'text':
            model_suffix += f"_{args.embedding_model}"
            
        st_model_path = f'models/st_model_{model_suffix}.pth'

        # Check if the ST model file exists, and if so, examine its dimensions
        if os.path.exists(st_model_path):
            try:
                # Load the model state dict to check dimensions
                checkpoint = torch.load(st_model_path, map_location=device)
                
                # Extract state dict
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                # Determine dimensions from the model
                saved_m = None
                saved_a = None
                
                # Check for memory indices to determine m
                if 'memory_indices' in state_dict:
                    saved_m = len(state_dict['memory_indices'])
                    print(f"Detected feature dimension from saved model: m={saved_m}")
                
                # Check for attention dimension from W_q weight
                if 'W_q.weight' in state_dict:
                    saved_a = state_dict['W_q.weight'].shape[0]
                    print(f"Detected attention dimension from saved model: a={saved_a}")
                
                # Use detected dimensions if available
                if saved_m is not None:
                    m = saved_m
                if saved_a is not None:
                    a = saved_a
            except Exception as e:
                print(f"Error checking model dimensions: {e}")
                print(f"Using specified dimensions: m={m}, a={a}")

        # Create the ST model with the correct dimensions
        st_model = SparseTransformer(
            X=train_feature_extract,
            n=n,
            m=m,
            a=a,
            st_model_path=st_model_path,
            lambda_l1=args.l1_lambda,
            num_heads=1,
            device=device,
            activation_threshold=args.activation_threshold,
            use_mixed_precision=args.use_mixed_precision
        )
        
        # Train or load model
        if args.force_retrain or not os.path.exists(st_model_path):
            print(f"Training ST from scratch...")
            st_model.train_and_validate(
                train_tensor,
                val_tensor,
                learning_rate=model_params['learning_rate'],
                batch_size=model_params['batch_size'],
                target_steps=model_params['target_steps'],
                grad_accum_steps=args.grad_accum_steps,
                eval_freq=args.eval_freq
            )
        else:
            print(f"Loading pre-trained ST model from {st_model_path}")
            checkpoint = torch.load(st_model_path)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                st_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                st_model.load_state_dict(checkpoint)
            print(f"Model loaded successfully.")
        
        # Calculate feature activations
        with torch.no_grad():
            st_activations = st_model.feature_activations(train_tensor)
            st_activations = st_activations.cpu().numpy()
            all_feature_activations["st"] = st_activations
        
        print(f"ST model saved to {st_model_path}")
        
        if args.visualize_decoder:
            print("\nVisualizing ST feature activations...")
            visualize_feature_activations(st_activations, "ST Feature Activations")
    
    # Perform classification if requested
    if args.perform_classification and args.label_column in train_sample_df.columns:
        print("\n" + "="*50)
        print("Performing classification...")
        print("="*50)
        
        for model_suffix in all_feature_activations.keys():
            features = all_feature_activations[model_suffix]
            clf, accuracy, report = train_and_evaluate_decision_tree(
                features, train_sample_df[args.label_column])
            
            classification_results[f"{model_suffix}"] = {
                "accuracy": accuracy,
                "report": report
            }
            
            print(f"\nClassification results for {model_suffix}:")
            print(f"Accuracy: {accuracy:.4f}")
            print("Classification Report:")
            print(report)
    
    # Create graph visualizations if requested
    if args.create_graph:
        create_graphs(train_sample_df, all_feature_activations, args)
    
    print("\n" + "="*50)
    print("Training and analysis completed successfully!")
    print("="*50)
    
    # Print a helpful command for loading the saved configuration
    if args.save_config:
        print(f"\nTo reuse this configuration later, run with:")
        print(f"python main.py --load_config last_config.json")
        
        # Create a feature comparison if multiple models were trained
        if len(all_feature_activations) > 1 and 'original' in all_feature_activations:
            if 'sae' in all_feature_activations and 'st' in all_feature_activations:
                print("\nFeature Comparison Summary:")
                print(f"Original features dimension: {all_feature_activations['original'].shape}")
                print(f"SAE features dimension: {all_feature_activations['sae'].shape}")
                print(f"ST features dimension: {all_feature_activations['st'].shape}")
                
                # If classification was performed, show accuracy comparison
                if args.perform_classification:
                    print("\nClassification Accuracy:")
                    for model_name, results in classification_results.items():
                        print(f"  {model_name}: {results['accuracy']:.4f}")

if __name__ == "__main__":
    main()