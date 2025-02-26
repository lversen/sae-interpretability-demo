import os
import argparse
import pandas as pd
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple, Union
import random
from feature_extraction_with_store import feature_extraction_with_store
from sample_handler import get_consistent_samples
from SAE import SparseAutoencoder
from ST import SparseTransformer
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run SAE and ST training with improved configuration')
    
    # Dataset parameters
    parser.add_argument('--train_dataset', type=str, default='data/mnist_train.csv', 
                        help='Path to training dataset')
    parser.add_argument('--val_dataset', type=str, default='data/mnist_test.csv',
                        help='Path to validation dataset')
    parser.add_argument('--feature_column', type=str, nargs='+', 
                        default=[str(i) for i in range(784)],
                        help='Column(s) containing features (list for vectors)')
    parser.add_argument('--label_column', type=str, default='label',
                        help='Column containing labels')
    
    # Model parameters
    parser.add_argument('--model_id', type=str, default='mnist',
                        help='Model identifier')
    parser.add_argument('--model_type', type=str, default='both', choices=['sae', 'st', 'both'],
                        help='Type of model to train')
    parser.add_argument('--n_train', type=int, default=60000,
                        help='Number of training samples to use')
    parser.add_argument('--n_val', type=int, default=10000,
                        help='Number of validation samples to use')
    
    # Training parameters
    parser.add_argument('--input_dimension', type=int, default=784,
                        help='Input dimension (n)')
    parser.add_argument('--feature_dimension', type=int, default=None,
                        help='Feature dimension (m), defaults to 8*n if not specified')
    parser.add_argument('--attention_dimension', type=int, default=None,
                        help='Attention dimension (a) for ST, defaults to n/2 if not specified')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Batch size')
    parser.add_argument('--l1_lambda', type=float, default=5.0,
                        help='L1 regularization strength')
    parser.add_argument('--target_steps', type=int, default=200000,
                        help='Target number of training steps')
    parser.add_argument('--force_retrain', action='store_true',
                        help='Force retraining of models')
    
    # ST-specific parameters
    parser.add_argument('--use_mixed_precision', action='store_true',
                        help='Enable mixed precision training for ST model')
    parser.add_argument('--activation_threshold', type=float, default=1e-3,
                        help='Activation threshold for ST feature tracking')
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                        help='Number of gradient accumulation steps for ST model')
    parser.add_argument('--eval_freq', type=int, default=None,
                        help='Evaluation frequency during training (steps)')
    
    # Misc parameters
    parser.add_argument('--data_type', type=str, default='vector', choices=['text', 'vector'],
                        help='Type of data')
    parser.add_argument('--visualize_decoder', action='store_true',
                        help='Visualize decoder matrix after training')
    parser.add_argument('--perform_classification', action='store_true',
                        help='Perform classification on the learned features')
    parser.add_argument('--create_graph', action='store_true',
                        help='Create Gephi graph visualization')
    parser.add_argument('--gephi_subset_size', type=int, default=1000,
                        help='Size of subset for Gephi visualization')
    
    return parser.parse_args()

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

def direct_vector_feature_extraction(df: pd.DataFrame, feature_columns: list) -> np.ndarray:
    """
    Extract features directly from dataframe columns that already contain vector data.
    """
    feature_matrix = df[feature_columns].values
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

def main():
    """Main function to run the SAE and ST training"""
    args = parse_args()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Setup feature dimensions
    n = args.input_dimension
    m = args.feature_dimension if args.feature_dimension else 8 * n
    a = args.attention_dimension if args.attention_dimension else n // 2
    
    # Model parameters
    model_params = {
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'target_steps': args.target_steps,
        'l1_lambda': args.l1_lambda,
        'force_retrain': args.force_retrain,
    }
    
    # Load datasets
    print(f"Loading datasets: {args.train_dataset}, {args.val_dataset}")
    train_df = pd.read_csv(args.train_dataset)
    val_df = pd.read_csv(args.val_dataset)
    
    # Get feature extraction function
    feature_extraction_fn = get_feature_extraction_fn(args.data_type)
    
    # Get consistent samples
    print(f"Getting consistent samples for training and validation")
    train_sample_df, train_indices = get_consistent_samples(
        train_df, args.n_train, f"{args.train_dataset}_train", args.model_id)
    val_sample_df, val_indices = get_consistent_samples(
        val_df, args.n_val, f"{args.val_dataset}_val", args.model_id)
    
    # Extract features
    print(f"Extracting features...")
    if args.data_type == 'text':
        train_feature_extract = feature_extraction_fn(
            train_sample_df, train_df, args.model_id, len(train_sample_df),
            f"{args.train_dataset}_train", args.feature_column,
            force_new_embeddings=False
        )
        val_feature_extract = feature_extraction_fn(
            val_sample_df, val_df, args.model_id, len(val_sample_df),
            f"{args.val_dataset}_val", args.feature_column,
            force_new_embeddings=False
        )
    else:
        train_feature_extract = feature_extraction_fn(
            train_sample_df, args.feature_column
        )
        val_feature_extract = feature_extraction_fn(
            val_sample_df, args.feature_column
        )
    
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
        
        sae_model_path = f'models/sae_model_{os.path.basename(args.train_dataset)}_{args.model_id}.pth'
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
        
        st_model_path = f'models/st_model_{os.path.basename(args.train_dataset)}_{args.model_id}.pth'
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
            st_model.load_state_dict(torch.load(st_model_path))
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
    
    print("\n" + "="*50)
    print("Training completed successfully!")
    print("="*50)

if __name__ == "__main__":
    main()