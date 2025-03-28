#!/usr/bin/env python3
"""
Script to create graph visualizations for all trained models, preserving path information
in the output filenames.

Example usage:
    # Process all models in the models directory:
    python path_preserving_graphs.py --model_folder models

    # Process all models in a specific subfolder:
    python path_preserving_graphs.py --model_folder models/mnist/st
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
import glob
import re
import gc
from typing import Dict, List, Any, Optional, Tuple
import sys
import hashlib

# Import necessary modules from main.py and other files
sys.path.append('.')  # Ensure the current directory is in path
from gephi import create_gephi_graph, select_random_labels
from sample_handler import get_consistent_samples
from feature_extraction_with_store import feature_extraction_with_store
from SAE import SparseAutoencoder
from ST import SparseTransformer

# Configuration for different datasets
DATASET_CONFIGS = {
    "mnist": {
        "train_dataset": "data/mnist_train.csv",
        "val_dataset": "data/mnist_test.csv",
        "data_type": "vector",
        "feature_column": [str(i) for i in range(784)],
        "label_column": "label",
        "input_dimension": 784,
    },
    "fashion_mnist": {
        "train_dataset": "data/fashion_mnist_train.csv",
        "val_dataset": "data/fashion_mnist_test.csv",
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

# Function to find all model files and group them by dataset
def find_and_group_models(folder_path, specific_dataset=None):
    """Find all .pth model files and group them by dataset"""
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Find all .pth files in the folder and its subfolders
    model_files = glob.glob(os.path.join(folder_path, "**/*.pth"), recursive=True)
    
    if not model_files:
        raise ValueError(f"No model files (.pth) found in {folder_path}")
    
    # Group models by dataset
    dataset_models = {}
    
    for model_path in model_files:
        # Try to determine dataset from path
        dataset = None
        path_parts = model_path.split(os.sep)
        
        for dataset_name in DATASET_CONFIGS.keys():
            if dataset_name in path_parts:
                dataset = dataset_name
                break
        
        # If dataset is not detected from path, use "unknown"
        if dataset is None:
            dataset = "unknown"
        
        # If specific dataset is requested, skip others
        if specific_dataset and dataset != specific_dataset:
            continue
            
        # Add model to the appropriate dataset group
        if dataset not in dataset_models:
            dataset_models[dataset] = []
        
        dataset_models[dataset].append(model_path)
    
    # Print summary of what we found
    total_models = sum(len(models) for models in dataset_models.values())
    print(f"Found {total_models} model files across {len(dataset_models)} datasets")
    
    for dataset, models in dataset_models.items():
        print(f"  - {dataset}: {len(models)} models")
    
    return dataset_models

# Function to extract model information from path
def extract_model_info(model_path):
    """Extract model information from the file path"""
    # Try to determine model type from path
    path_parts = model_path.split(os.sep)
    
    # Default values
    model_type = None
    feature_dimension = None
    function_type = None  # For activation/attention function
    
    # Check for model type
    if 'sae' in path_parts:
        model_type = 'sae'
        # For SAE, activation function is typically the next folder after 'sae'
        sae_index = path_parts.index('sae')
        if sae_index + 1 < len(path_parts):
            # This might be the activation function
            function_type = path_parts[sae_index + 1]
    elif 'st' in path_parts:
        model_type = 'st'
        # For ST, attention function is typically the next folder after 'st'
        st_index = path_parts.index('st')
        if st_index + 1 < len(path_parts):
            # This might be the attention function
            function_type = path_parts[st_index + 1]
    
    # Extract feature dimension from the folder structure
    # Assuming structure like models/mnist/sae/relu/100/model.pth where 100 is feature_dimension
    for part in path_parts:
        if part.isdigit():
            feature_dimension = int(part)
            break
    
    # If not found in folder structure, try to extract from filename
    if feature_dimension is None:
        filename = os.path.basename(model_path)
        pattern = r'_(\d+)_'  # Look for pattern like _100_
        match = re.search(pattern, filename)
        if match:
            feature_dimension = int(match.group(1))
    
    return model_type, feature_dimension, function_type

# Function to extract feature dimension from model file
def extract_feature_dimension_from_model(model_path, device='cpu'):
    """Try to extract feature dimension from the model file itself"""
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Look for typical SAE/ST matrices to determine feature dimension
        if 'W_e.weight' in state_dict:  # SAE encoder weight
            return state_dict['W_e.weight'].shape[0]
        elif 'W_d.weight' in state_dict:  # SAE decoder weight
            return state_dict['W_d.weight'].shape[1]
        elif 'memory_indices' in state_dict:  # ST memory indices
            return len(state_dict['memory_indices'])
        elif 'W_k_direct' in state_dict:  # ST direct K-V key matrix
            return state_dict['W_k_direct'].shape[0]
        elif 'W_v_direct' in state_dict:  # ST direct K-V value matrix
            return state_dict['W_v_direct'].shape[0]
    except Exception as e:
        print(f"Error extracting dimensions from model file: {e}")
    
    return None

# Function to load and preprocess dataset
def load_dataset(dataset_name, n_samples=None):
    """Load dataset based on configuration"""
    dataset_config = DATASET_CONFIGS.get(dataset_name, None)
    if dataset_config is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Load training dataset
    train_path = dataset_config['train_dataset']
    print(f"Loading dataset from {train_path}")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Dataset file not found: {train_path}")
        
    train_df = pd.read_csv(train_path)
    
    # Limit samples if specified
    if n_samples is not None:
        n_samples = min(n_samples, len(train_df))
    else:
        n_samples = len(train_df)  # Use all samples if None specified
    
    # Get consistent samples
    sample_df, sample_indices = get_consistent_samples(
        train_df, n_samples, f"{dataset_name}_train", dataset_name)
    
    # Extract features
    if dataset_config['data_type'] == 'vector':
        # For vector data like MNIST
        feature_columns = dataset_config['feature_column']
        features = sample_df[feature_columns].values
    else:
        # For text data, would need embedding model
        # This is more complex and would need the embedding model specification
        raise ValueError(f"Text data not supported in this simplified script: {dataset_name}")
    
    print(f"Loaded {len(sample_df)} samples with {features.shape[1]} features")
    return sample_df, features, dataset_config

# Function to load a trained model
def load_model(model_path, model_type, input_dim, feature_dim, device='cuda'):
    """Load a trained model from file"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading {model_type.upper()} model from {model_path}")
    
    try:
        if model_type.lower() == 'sae':
            # Initialize SAE model
            model = SparseAutoencoder(
                n=input_dim,
                m=feature_dim,
                sae_model_path=model_path,
                device=device
            )
        elif model_type.lower() == 'st':
            # For ST model, we need more parameters
            # We'll use placeholder data for X since we're just loading a trained model
            placeholder_data = torch.zeros((100, input_dim), device=device)
            
            # Find attention dimension from the model file if possible
            try:
                checkpoint = torch.load(model_path, map_location=device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                    
                # Try to determine attention dimension from W_q weight
                if 'W_q.weight' in state_dict:
                    attention_dim = state_dict['W_q.weight'].shape[0]
                    print(f"Detected attention dimension: {attention_dim}")
                else:
                    # Use a reasonable default
                    attention_dim = input_dim // 4
                    print(f"Using default attention dimension: {attention_dim}")
            except Exception as e:
                print(f"Error extracting dimensions from model: {e}")
                attention_dim = input_dim // 4
            
            # Initialize ST model
            model = SparseTransformer(
                X=placeholder_data,
                n=input_dim,
                m=feature_dim,
                a=attention_dim,
                st_model_path=model_path,
                device=device
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load model weights
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Model loaded successfully")
        
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Function to extract features from a trained model
def extract_features(model, features, model_type, device='cuda'):
    """Extract feature activations from the model"""
    # Convert features to tensor
    feature_tensor = torch.tensor(features, dtype=torch.float32).to(device)
    
    # Extract features
    model.eval()
    with torch.no_grad():
        if model_type.lower() == 'sae':
            activations = model.feature_activations(feature_tensor)
        else:  # ST model
            activations = model.feature_activations(feature_tensor)
    
    # Convert to numpy
    feature_activations = activations.cpu().numpy()
    print(f"Extracted {feature_activations.shape[0]} feature activations with dimension {feature_activations.shape[1]}")
    
    return feature_activations

# Function to create path-preserving filename
def create_path_preserving_filename(model_path, model_type, dataset_name, feature_dimension):
    """Create a filename that preserves important path components"""
    # Get the relative path from the models directory
    rel_path = model_path
    if 'models/' in model_path:
        rel_path = model_path.split('models/')[1]
    
    # Extract important components
    path_parts = rel_path.split(os.sep)
    important_components = []
    
    # Always include dataset, model_type, and feature_dimension
    if dataset_name not in important_components:
        important_components.append(dataset_name)
    if model_type not in important_components:
        important_components.append(model_type)
    
    # Find activation/attention function
    model_type_index = -1
    try:
        model_type_index = path_parts.index(model_type)
    except ValueError:
        pass
    
    if model_type_index >= 0 and model_type_index + 1 < len(path_parts):
        function_type = path_parts[model_type_index + 1]
        if function_type not in important_components:
            important_components.append(function_type)
    
    # Add feature dimension if not already in components
    feature_str = str(feature_dimension)
    if feature_str not in important_components:
        important_components.append(feature_str)
    
    # Add the base filename without extension
    base_filename = os.path.splitext(os.path.basename(model_path))[0]
    
    # Check if the filename is too long, if so hash the original path
    if len('_'.join(important_components + [base_filename])) > 100:
        # Hash the full path to create a unique identifier
        path_hash = hashlib.md5(model_path.encode()).hexdigest()[:10]
        base_filename = f"{base_filename[:40]}_{path_hash}"
    
    # Combine all components
    filename = '_'.join(important_components + [base_filename]) + '.gexf'
    
    return filename

# Function to create graph visualization
def create_graph(df, features, model_path, model_type, dataset_name, feature_dimension, 
                label_column, n_random_labels=10, gephi_subset_size=1000, graph_neighbors=4):
    """Create graph visualization with path information preserved in filename"""
    # Create output directory
    graphs_dir = 'graphs'
    os.makedirs(graphs_dir, exist_ok=True)
    
    # Take a subset for visualization
    subset_size = min(gephi_subset_size, len(df))
    subset_df = df.iloc[:subset_size].copy()
    subset_features = features[:subset_size]
    
    # Determine title column and category column
    title_column = label_column
    category_column = label_column
    
    # Select random labels for consistent visualization
    selected_labels = select_random_labels(
        subset_df, 
        title_column=title_column, 
        n_random_labels=n_random_labels,
        category_column=category_column
    )
    
    print(f"Selected labels for visualization: {selected_labels}")
    
    # Create filename that preserves path information
    filename = create_path_preserving_filename(
        model_path, model_type, dataset_name, feature_dimension)
    
    # Create graph file path
    graph_file = os.path.join(graphs_dir, filename)
    
    # Extract model name for display in graph
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    
    # Create the graph
    create_gephi_graph(
        feature_extract=subset_features,
        df=subset_df,
        title_column=title_column,
        model_name=model_path.replace('/','_'),  # Use full path but make it filename-safe
        file_path=graph_file,
        selected_labels=selected_labels,
        category_column=category_column,
        n_neighbors=graph_neighbors
    )
    
    print(f"Graph saved to: {graph_file}")
    return graph_file

def process_model(model_path, model_type, dataset_name, feature_dimension, df, features, 
                 dataset_config, args):
    """Process a single model to create its graph"""
    print(f"\n{'='*80}")
    print(f"Processing model: {model_path}")
    print(f"Model type: {model_type}, Feature dimension: {feature_dimension}")
    print(f"{'='*80}")
    
    # If model type or feature dimension is still unknown, try to extract from model file
    if model_type is None or feature_dimension is None:
        if feature_dimension is None:
            dim = extract_feature_dimension_from_model(model_path, args.device)
            if dim is not None:
                feature_dimension = dim
                print(f"Extracted feature dimension from model: {feature_dimension}")
    
    # If we still don't have necessary information, skip this model
    if model_type is None:
        print(f"Could not determine model type for {model_path}. Skipping.")
        return False
    
    if feature_dimension is None:
        print(f"Could not determine feature dimension for {model_path}. Skipping.")
        return False
    
    # Load the model
    input_dimension = dataset_config['input_dimension']
    model = load_model(
        model_path, 
        model_type, 
        input_dimension, 
        feature_dimension,
        args.device
    )
    
    if model is None:
        print(f"Failed to load model. Skipping.")
        return False
    
    # Extract features from the model
    try:
        feature_activations = extract_features(model, features, model_type, args.device)
        
        # Free up memory by deleting model
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Create graph visualization
        graph_file = create_graph(
            df, 
            feature_activations, 
            model_path, 
            model_type, 
            dataset_name, 
            feature_dimension,
            dataset_config['label_column'],
            args.n_random_labels,
            args.gephi_subset_size,
            args.graph_neighbors
        )
        print(f"Successfully created graph: {graph_file}")
        return True
    except Exception as e:
        print(f"Error processing model: {e}")
        # Free up memory
        if 'model' in locals():
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return False

def main():
    parser = argparse.ArgumentParser(description='Create graph visualizations for trained models, preserving path information')
    
    # Model folder parameter
    parser.add_argument('--model_folder', type=str, required=True,
                      help='Path to a folder containing trained models')
    
    # Optional dataset filter
    parser.add_argument('--dataset', type=str, choices=list(DATASET_CONFIGS.keys()),
                      help='Process only models for this dataset (if specified)')
    
    # Graph parameters
    parser.add_argument('--n_samples', type=int, default=None,
                      help='Number of samples to use for feature extraction (all if not specified)')
    parser.add_argument('--n_random_labels', type=int, default=10,
                      help='Number of random labels to select for graph visualization')
    parser.add_argument('--gephi_subset_size', type=int, default=1000,
                      help='Size of subset for Gephi visualization')
    parser.add_argument('--graph_neighbors', type=int, default=4,
                      help='Number of neighbors for graph creation')
    
    # Device parameter
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use for model loading and feature extraction')
    
    args = parser.parse_args()
    
    # Find models and group by dataset
    try:
        dataset_models = find_and_group_models(args.model_folder, args.dataset)
        if not dataset_models:
            print("No models found to process")
            return
    except Exception as e:
        print(f"Error finding models: {e}")
        return
    
    # Track successful and failed models
    successful = []
    failed = []
    
    # Process each dataset
    for dataset_name, model_paths in dataset_models.items():
        # Skip unknown datasets
        if dataset_name == "unknown" or dataset_name not in DATASET_CONFIGS:
            print(f"Skipping {len(model_paths)} models with unknown dataset")
            failed.extend(model_paths)
            continue
        
        print(f"\n{'='*80}")
        print(f"PROCESSING DATASET: {dataset_name.upper()}")
        print(f"{'='*80}")
        
        # Load dataset (only once per dataset)
        try:
            df, features, dataset_config = load_dataset(dataset_name, args.n_samples)
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            failed.extend(model_paths)
            continue
        
        # Process all models for this dataset
        for model_path in model_paths:
            # Determine model type and feature dimension
            model_type, feature_dimension, _ = extract_model_info(model_path)
            
            # Process the model
            success = process_model(
                model_path, model_type, dataset_name, feature_dimension,
                df, features, dataset_config, args
            )
            
            if success:
                successful.append(model_path)
            else:
                failed.append(model_path)
            
            # Free up memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Successfully processed {len(successful)} models")
    print(f"Failed to process {len(failed)} models")
    
    if failed:
        print("\nFirst 10 failed models:")
        for model in failed[:10]:
            print(f"  - {model}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")
    
    print(f"\nGraph visualizations saved to the 'graphs' directory")

if __name__ == "__main__":
    main()