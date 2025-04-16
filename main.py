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
# Import both ST implementations with different names
import ST
import ST_old
import matplotlib.pyplot as plt
from gephi import create_gephi_graph, select_random_labels
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import json
# Add torchinfo import
from torchinfo import summary as model_summary

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

# Add LLM embedding model configurations
LLM_MODELS = {
    "gte-large": "Alibaba-NLP/gte-large-en-v1.5"
}

def calculate_attention_dim_for_equal_params(n, m, use_direct_kv=False):
    """
    Calculate attention dimension 'a' that would make ST and SAE have equal parameters,
    considering both memory bank and direct KV approaches.
    
    For equal parameters with direct KV approach:
    SAE: (2*n*m + m + n)
    ST direct KV: a*(n + 1 + m) + m*n + 2*n
    
    For equal parameters with memory bank approach:
    SAE: (2*n*m + m + n)
    ST memory bank: a*(2*n + 4) + n*(n + 3)
    
    Args:
        n: Input dimension
        m: Feature dimension
        use_direct_kv: Whether using direct KV or memory bank approach
        
    Returns:
        a: Attention dimension that makes parameter counts approximately equal
    """
    # SAE parameter count: 2*m*n + m + n
    sae_params = 2*m*n + m + n
    
    if use_direct_kv:
        # Direct KV parameter count: a*(n + 1 + m) + m*n + 2*n
        # Solve for a:
        # a*(n + 1 + m) = sae_params - m*n - 2*n
        # a = (sae_params - m*n - 2*n) / (n + 1 + m)
        a = (sae_params - m*n - 2*n) / (n + 1 + m)
    else:
        # Memory bank parameter count: a*(2*n + 4) + n*(n + 3)
        # Solve for a:
        # a*(2*n + 4) = sae_params - n*(n + 3)
        # a = (sae_params - n*(n + 3)) / (2*n + 4)
        a = (sae_params - n*(n + 3)) / (2*n + 4)
    
    # Ensure a doesn't become too small or negative
    a = max(1, int(a))
    
    return a

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

# Add new function to display model information
def display_model_info(model, model_name, input_size=None, verbose=1):
    """
    Display detailed information about a model using torchinfo.
    
    Args:
        model: The PyTorch model to analyze
        model_name: Name of the model for display
        input_size: Input size tuple for the model (optional)
        verbose: Verbosity level (0, 1, or 2)
    """
    print("\n" + "="*50)
    print(f"MODEL INFORMATION: {model_name}")
    print("="*50)
    
    # Basic parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    
    # If input_size provided, use torchinfo for detailed summary
    if input_size:
        print(f"\nDetailed Model Summary (input size: {input_size}):")
        model_summary(model, input_size=input_size, depth=2, verbose=verbose)
    else:
        print("\nDetailed Parameter Breakdown:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name:<30} {str(list(param.shape)):>20} {param.numel():>12,}")
    
    print("="*50)

def calculate_optimal_training_steps(
    feature_dimension: int, 
    input_dimension: int, 
    model_type: str = 'sae',
    base_steps: int = 200_000, 
    min_steps: int = 50_000, 
    max_steps: int = 1_000_000
) -> int:
    """
    Calculate the optimal number of training steps based on feature dimension and input dimension,
    following scaling laws for dictionary learning as described in the research paper.
    
    Args:
        feature_dimension: The feature dimension (m)
        input_dimension: The input dimension (n)
        model_type: Type of model ('sae' or 'st')
        base_steps: Base number of steps for reference configuration (8*n features)
        min_steps: Minimum number of steps to return
        max_steps: Maximum number of steps to return
        
    Returns:
        Recommended number of training steps
    """
    # Calculate the ratio of features to input dimension
    feature_ratio = feature_dimension / input_dimension
    
    # Reference configuration from the paper: 8*n features with 200,000 steps
    reference_ratio = 8.0
    
    # Apply scaling law based on the paper's findings
    # The paper suggests that optimal training steps scale with features with an exponent between 0.5 and 1
    # We use 0.75 as a middle ground based on the power law relationship observed in the paper
    scaling_exponent = 0.75
    
    # Calculate the scaling factor based on how our feature ratio differs from reference
    if feature_ratio <= 0:
        scaling_factor = 1.0  # Protection against division by zero
    else:
        scaling_factor = (feature_ratio / reference_ratio) ** scaling_exponent
    
    # Apply scaling factor to the base steps
    optimal_steps = int(base_steps * scaling_factor)
    
    # Ensure we're within reasonable bounds
    optimal_steps = max(min_steps, min(optimal_steps, max_steps))
    
    # ST models might benefit from more steps due to more complex optimization
    if model_type.lower() == 'st':
        optimal_steps = int(optimal_steps * 1.2)  # 20% more steps for ST models
        optimal_steps = min(optimal_steps, max_steps)
    
    return optimal_steps

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
<<<<<<< HEAD
    # Add argument for custom features file
    dataset_group.add_argument('--custom_features_file', type=str, default=None,
                        help='Path to pre-extracted features file (.npz or .pt) for custom dataset')
    # Add arguments for split GPT Neo files
    dataset_group.add_argument('--custom_train_file', type=str, default=None,
                        help='Path to pre-extracted training features file (.npz or .pt) for custom dataset')
    dataset_group.add_argument('--custom_val_file', type=str, default=None,
                        help='Path to pre-extracted validation features file (.npz or .pt) for custom dataset')
=======
>>>>>>> parent of 466e95f (checkpoint)
    
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
    model_group.add_argument('--model_id', type=str, default=None,
                        help='Model identifier (defaults to dataset name if not specified)')
    model_group.add_argument('--model_type', type=str, default='both', 
                        choices=['sae', 'st', 'both'],
                        help='Type of model to train')
    model_group.add_argument('--n_train', type=int, default=None,
                        help='Number of training samples to use (defaults to max available)')
    model_group.add_argument('--n_val', type=int, default=None,
                        help='Number of validation samples to use (defaults to max available)')
    model_group.add_argument('--feature_dimension', type=int, default=None,
                        help='Feature dimension (m), defaults to 8*n if not specified')
    model_group.add_argument('--attention_dimension', type=int, default=None,
                        help='Attention dimension (a) for ST, defaults to auto-calculated value to balance parameter count with SAE')
    model_group.add_argument('--activation', type=str, default='relu',
                        choices=['relu', 'leaky_relu', 'gelu', 'sigmoid', 'tanh', 'none'],
                        help='Activation function to use (currently only for SAE model)')
    model_group.add_argument('--attention_fn', type=str, default='softmax',
                        choices=['softmax', 'sparsemax', 'normalized_activation', 'direct_activation', 
                                'relu_softmax', 'softmax_hard', 'softmax_soft',
                                'length_scaled_softmax', 'softmax_with_bias', 'polynomial_attention', 'adaptive_sparse',
                                'relu_attention', 'tanh_scale_shift'],
                        help='Function to use for processing attention scores (ST models only)')
    
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
    # NEW: Add option to use the old ST implementation
    st_group.add_argument('--use_old_st', action='store_true',
                        help='Use the original ST implementation (ST_old.py) instead of the new one')
    st_group.add_argument('--use_mixed_precision', action='store_true',
                        help='Enable mixed precision training for ST model')
    st_group.add_argument('--activation_threshold', type=float, default=1e-3,
                        help='Activation threshold for ST feature tracking')
    st_group.add_argument('--grad_accum_steps', type=int, default=1,
                        help='Number of gradient accumulation steps for ST model')
    st_group.add_argument('--eval_freq', type=int, default=None,
                        help='Evaluation frequency during training (steps)')
    # NEW: Add option for original memory-based approach (direct K-V is default)
    st_group.add_argument('--use_memory_bank', action='store_true',
                        help='Use original memory bank approach instead of direct K-V matrices')
    
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
    # Add new parameter for model info display
    misc_group.add_argument('--show_model_info', action='store_true',
                        help='Display detailed model information using torchinfo')
    misc_group.add_argument('--model_info_verbosity', type=int, default=1, choices=[0, 1, 2],
                        help='Verbosity level for model information (0=minimal, 2=detailed)')
    # Add new parameters for automatic steps calculation
    misc_group.add_argument('--auto_steps', action='store_true',
                        help='Automatically determine optimal number of training steps based on feature dimension')
    misc_group.add_argument('--auto_steps_base', type=int, default=200000,
                        help='Base number of steps for auto-steps calculation (default: 200000)')
    misc_group.add_argument('--auto_steps_min', type=int, default=5000,
                        help='Minimum number of steps for auto-steps calculation (default: 50000)')
    misc_group.add_argument('--auto_steps_max', type=int, default=1000000,
                        help='Maximum number of steps for auto-steps calculation (default: 1000000)')
    misc_group.add_argument('--sae_save_path', type=str, default=None,
                        help='Custom save path for SAE model')
    misc_group.add_argument('--st_save_path', type=str, default=None,
                        help='Custom save path for ST model')
    misc_group.add_argument('--device', type=str, default=None,
                      help='Device to use for training (cuda or cpu)')
    
    args = parser.parse_args()
    
    # If list_datasets flag is set, list available datasets and exit
    if args.list_datasets:
        print("Available datasets in data directory:")
        datasets = list_available_datasets()
        for i, ds in enumerate(datasets):
            print(f"  {i+1}. {ds}")
        exit(0)
    if args.device:
        print(f"Explicitly using device: {args.device}")
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Defaulting to device: {device}")

    # Later in the code when creating models
    print(f"Creating model on device: {device}")
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
                             'save_config', 'use_memory_bank', 'show_model_info', 'auto_steps',
                             'use_old_st']  # Added use_old_st to the list
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
    
    # Set model_id based on dataset if not explicitly provided
    if args.model_id is None:
        args.model_id = args.dataset
    
    # Apply default values for training/validation set sizes based on actual dataset sizes
    if args.n_train is None or args.n_val is None:
        import pandas as pd
        import os
        
        # Load train dataset to get its size
        if args.n_train is None and os.path.exists(args.train_dataset):
            try:
                # Read only the number of rows without loading the entire dataset
                # First try a fast count method for CSV files
                with open(args.train_dataset, 'r') as f:
                    # Count lines but skip header
                    train_size = sum(1 for _ in f) - 1
                print(f"Found {train_size} samples in training dataset")
                args.n_train = train_size
            except Exception as e:
                print(f"Error counting training samples: {e}")
                # Fallback to defaults
                if 'mnist' in args.train_dataset:
                    args.n_train = 60000
                elif 'stack_exchange' in args.train_dataset:
                    args.n_train = 60000
                else:
                    args.n_train = 10000
        else:
            # Use reasonable defaults if can't determine size
            if args.n_train is None:
                if 'mnist' in args.train_dataset:
                    args.n_train = 60000
                elif 'stack_exchange' in args.train_dataset:
                    args.n_train = 60000
                else:
                    args.n_train = 10000
        
        # Load validation dataset to get its size
        if args.n_val is None and os.path.exists(args.val_dataset):
            try:
                # Read only the number of rows without loading the entire dataset
                with open(args.val_dataset, 'r') as f:
                    # Count lines but skip header
                    val_size = sum(1 for _ in f) - 1
                print(f"Found {val_size} samples in validation dataset")
                args.n_val = val_size
            except Exception as e:
                print(f"Error counting validation samples: {e}")
                # Fallback to defaults
                if 'mnist' in args.val_dataset:
                    args.n_val = 10000
                elif 'stack_exchange' in args.val_dataset:
                    args.n_val = 10000
                else:
                    args.n_val = 2000
        else:
            # Use reasonable defaults if can't determine size
            if args.n_val is None:
                if 'mnist' in args.val_dataset:
                    args.n_val = 10000
                elif 'stack_exchange' in args.val_dataset:
                    args.n_val = 10000
                else:
                    args.n_val = 2000
    
    # Resolve embedding model
    if args.custom_embedding_model:
        args.embedding_model_path = args.custom_embedding_model
    else:
        args.embedding_model_path = LLM_MODELS.get(args.embedding_model, LLM_MODELS['gte-large'])
    
    # Save config if requested
    if args.save_config:
        save_config_to_file(args)
    
    return args
def get_hierarchical_model_path(args, model_type):
    """
    Generate a hierarchical path for model saving based on model parameters.
    
    Structure:
    models/[dataset]/[model_type]/[activation_or_attention]/[feature_dimension]/[training_params].pth
    
    Args:
        args: Command line arguments
        model_type: 'sae' or 'st'
        
    Returns:
        Path string for saving the model
    """
    # Base directory is always models/dataset/
    base_dir = os.path.join('models', args.dataset)
    
    if model_type.lower() == 'sae':
        # SAE: models/[dataset]/sae/[activation]/[feature_dimension]/
        hierarchy_path = os.path.join(
            base_dir,
            'sae',
            args.activation,
            str(args.feature_dimension)
        )
    else:  # ST model
        # ST: models/[dataset]/st/[attention_fn]/[feature_dimension]/
        hierarchy_path = os.path.join(
            base_dir,
            'st',
            args.attention_fn,
            str(args.feature_dimension)
        )
    
    # Create the directory
    os.makedirs(hierarchy_path, exist_ok=True)
    
    # Create filename with training parameters
    filename_parts = []
    
    # Add batch size
    filename_parts.append(f"bs{args.batch_size}")
    
    # Add learning rate (replace decimal point)
    lr_str = str(args.learning_rate).replace('.', 'p')
    filename_parts.append(f"lr{lr_str}")
    
    # Add target steps
    if args.auto_steps:
        filename_parts.append(f"autosteps{args.auto_steps_base}")
    else:
        filename_parts.append(f"steps{args.target_steps}")
    
    # Add L1 lambda if not the default 5.0
    if args.l1_lambda != 5.0:
        l1_str = str(args.l1_lambda).replace('.', 'p')
        filename_parts.append(f"l1{l1_str}")
    
    # Add gradient accumulation if not 1
    if hasattr(args, 'grad_accum_steps') and args.grad_accum_steps != 1:
        filename_parts.append(f"accum{args.grad_accum_steps}")
    
    # Add memory bank flag for ST models
    if model_type.lower() == 'st' and args.use_memory_bank:
        filename_parts.append("memory")
    
    # Add old ST implementation flag
    if model_type.lower() == 'st' and args.use_old_st:
        filename_parts.append("oldst")
    
    # Combine all parts
    filename = f"{'_'.join(filename_parts)}.pth"
    
    # Full hierarchical path
    return os.path.join(hierarchy_path, filename)
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

<<<<<<< HEAD
def custom_feature_extraction(df: pd.DataFrame, full_df: pd.DataFrame = None, 
                            model: str = None, n: int = None,
                            dataset_name: str = None, content_column: str = None,
                            feature_columns: list = None, force_new_embeddings: bool = False,
                            custom_features_file: str = None, custom_train_file: str = None, 
                            custom_val_file: str = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features from pre-extracted GPT Neo files with support for text-based train/val splits.
    
    Args:
        df: DataFrame containing the data (not used for custom features)
        full_df: Full DataFrame (not used for custom features)
        model: Model name (not used for custom features)
        n: Input dimension (not used for custom features)
        dataset_name: Dataset name (not used for custom features)
        content_column: Content column name (not used for custom features)
        feature_columns: Feature column names (not used for custom features)
        force_new_embeddings: Force recomputation of embeddings (not used for custom features)
        custom_features_file: Path to pre-extracted features file (.npz or .pt)
        custom_train_file: Path to training features from text-based split (optional)
        custom_val_file: Path to validation features from text-based split (optional)
        
    Returns:
        Tuple of (train_features, val_features) as numpy arrays
    """
    # Check if separate train/val files are provided
    if custom_train_file and custom_val_file and os.path.exists(custom_train_file) and os.path.exists(custom_val_file):
        print(f"Loading train features from: {custom_train_file}")
        print(f"Loading validation features from: {custom_val_file}")
        
        # Load train features
        try:
            if custom_train_file.endswith('.npz'):
                train_data = np.load(custom_train_file, allow_pickle=True)
                train_features = train_data['features']
            elif custom_train_file.endswith('.pt'):
                train_data = torch.load(custom_train_file, map_location='cpu')
                train_features = train_data['features']
                if isinstance(train_features, torch.Tensor):
                    train_features = train_features.cpu().numpy()
            
            # Load val features
            if custom_val_file.endswith('.npz'):
                val_data = np.load(custom_val_file, allow_pickle=True)
                val_features = val_data['features']
            elif custom_val_file.endswith('.pt'):
                val_data = torch.load(custom_val_file, map_location='cpu')
                val_features = val_data['features']
                if isinstance(val_features, torch.Tensor):
                    val_features = val_features.cpu().numpy()
            
            # Update df with token group information if available
            if df is not None:
                # Create train dataframe
                train_df_size = train_features.shape[0]
                df_train = pd.DataFrame({'dummy': range(train_df_size)})
                
                # Add token groups for train
                if 'token_to_text_map' in train_data:
                    if isinstance(train_data['token_to_text_map'], np.ndarray):
                        df_train['token_group'] = train_data['token_to_text_map']
                    elif isinstance(train_data, dict) and 'token_to_text_map' in train_data:
                        token_map = train_data['token_to_text_map']
                        if isinstance(token_map, torch.Tensor):
                            token_map = token_map.cpu().numpy()
                        df_train['token_group'] = token_map
                
                # Create val dataframe
                val_df_size = val_features.shape[0]
                df_val = pd.DataFrame({'dummy': range(val_df_size)})
                
                # Add token groups for val
                if isinstance(val_data, dict) and 'token_to_text_map' in val_data:
                    token_map = val_data['token_to_text_map']
                    if isinstance(token_map, torch.Tensor):
                        token_map = token_map.cpu().numpy()
                    df_val['token_group'] = token_map
                
                # Combine them for the caller (assuming caller expects joint dataframe)
                # But track the split point to separate them later
                split_point = train_df_size
                df_combined = pd.concat([df_train, df_val]).reset_index(drop=True)
                
                # Replace df with our combined dataframe
                for col in df_combined.columns:
                    if col not in df.columns:
                        df[col] = None
                    df[col] = df_combined[col]
            
            print(f"Successfully loaded split features - Train: {train_features.shape}, Val: {val_features.shape}")
            
            # Return both feature sets instead of just combined
            return train_features, val_features
        
        except Exception as e:
            print(f"Error loading split features: {e}")
            print("Falling back to combined features file.")
            # Fall through to combined file loading
    
    # For backward compatibility or if split files don't exist, use combined file
    if custom_features_file is None:
        raise ValueError("custom_features_file must be provided for custom data type")
    
    print(f"Loading combined features from: {custom_features_file}")
    
    # Check if features are preprocessed
    is_prep, prep_method = is_preprocessed(custom_features_file)
    if is_prep:
        print(f"Detected preprocessed features with method: {prep_method}")
    else:
        print("Loading raw (unprocessed) features")
    
    # Load features from combined file
    try:
        if custom_features_file.endswith('.npz'):
            # Load NPZ file
            loaded_data = np.load(custom_features_file, allow_pickle=True)
            if 'features' in loaded_data:
                features = loaded_data['features']
                # Check if we need to reshape
                if len(features.shape) > 2:
                    print(f"Reshaping features from {features.shape} to 2D")
                    features = features.reshape(features.shape[0], -1)
            else:
                # Try to find the first array in the file
                for key in loaded_data:
                    if isinstance(loaded_data[key], np.ndarray) and len(loaded_data[key].shape) >= 2:
                        features = loaded_data[key]
                        # Reshape if needed
                        if len(features.shape) > 2:
                            features = features.reshape(features.shape[0], -1)
                        print(f"Using array '{key}' as features from NPZ file")
                        break
                else:
                    raise ValueError(f"Could not find features array in NPZ file: {custom_features_file}")
        elif custom_features_file.endswith('.pt'):
            # Load PyTorch file
            loaded_data = torch.load(custom_features_file, map_location='cpu')
            if isinstance(loaded_data, dict) and 'features' in loaded_data:
                features = loaded_data['features']
                # Convert to numpy if it's a tensor
                if isinstance(features, torch.Tensor):
                    features = features.cpu().numpy()
                # Check if we need to reshape
                if len(features.shape) > 2:
                    print(f"Reshaping features from {features.shape} to 2D")
                    features = features.reshape(features.shape[0], -1)
            else:
                raise ValueError(f"Could not find 'features' in PT file: {custom_features_file}")
        else:
            # Try to infer file type from content for files without proper extension
            try:
                # First try to load as NPZ
                loaded_data = np.load(custom_features_file, allow_pickle=True)
                print(f"Loaded file as NPZ despite extension: {custom_features_file}")
                if 'features' in loaded_data:
                    features = loaded_data['features']
                else:
                    # Look for arrays
                    for key in loaded_data:
                        if isinstance(loaded_data[key], np.ndarray) and len(loaded_data[key].shape) >= 2:
                            features = loaded_data[key]
                            print(f"Using array '{key}' as features")
                            break
                    else:
                        raise ValueError(f"No suitable arrays found in file")
            except:
                # Try as PyTorch file
                try:
                    loaded_data = torch.load(custom_features_file, map_location='cpu')
                    print(f"Loaded file as PT despite extension: {custom_features_file}")
                    if isinstance(loaded_data, dict) and 'features' in loaded_data:
                        features = loaded_data['features']
                        if isinstance(features, torch.Tensor):
                            features = features.cpu().numpy()
                    else:
                        raise ValueError(f"Could not find 'features' in file")
                except:
                    raise ValueError(f"Unknown file format for custom features file: {custom_features_file}")
        
        # Ensure features is a 2D array
        if len(features.shape) > 2:
            print(f"Reshaping features from {features.shape} to 2D")
            features = features.reshape(features.shape[0], -1)
        elif len(features.shape) < 2:
            print(f"Features array has wrong dimensionality: {features.shape}")
            raise ValueError(f"Features must be a 2D array, got shape {features.shape}")
            
        print(f"Loaded features with shape: {features.shape}")
        
        # Add token_to_text_map to dataframe if available
        token_to_text_map = None
        if custom_features_file.endswith('.npz') and 'token_to_text_map' in loaded_data:
            token_to_text_map = loaded_data['token_to_text_map']
            print(f"Found token_to_text_map with shape: {token_to_text_map.shape}")
        elif custom_features_file.endswith('.pt') and isinstance(loaded_data, dict) and 'token_to_text_map' in loaded_data:
            token_to_text_map = loaded_data['token_to_text_map']
            if isinstance(token_to_text_map, torch.Tensor):
                token_to_text_map = token_to_text_map.cpu().numpy()
            print(f"Found token_to_text_map with shape: {token_to_text_map.shape}")
        
        if token_to_text_map is not None and df is not None:
            # Adjust the dataframe size to match features if needed
            if len(df) != features.shape[0]:
                print(f"Adjusting dataframe size from {len(df)} to {features.shape[0]} to match features")
                df = pd.DataFrame({'dummy': range(features.shape[0])})
            
            print("Adding token_to_text_map as a label column in dataframe")
            df['token_group'] = token_to_text_map
        
        # Check if file contains train/val split indices
        train_indices = None
        val_indices = None
        
        if custom_features_file.endswith('.npz'):
            if 'train_indices' in loaded_data:
                train_indices = loaded_data['train_indices']
                print(f"Found train indices with shape: {train_indices.shape}")
            if 'val_indices' in loaded_data:
                val_indices = loaded_data['val_indices']
                print(f"Found validation indices with shape: {val_indices.shape}")
        elif custom_features_file.endswith('.pt') and isinstance(loaded_data, dict):
            if 'train_indices' in loaded_data:
                train_indices = loaded_data['train_indices']
                if isinstance(train_indices, torch.Tensor):
                    train_indices = train_indices.cpu().numpy()
                print(f"Found train indices with shape: {train_indices.shape}")
            if 'val_indices' in loaded_data:
                val_indices = loaded_data['val_indices']
                if isinstance(val_indices, torch.Tensor):
                    val_indices = val_indices.cpu().numpy()
                print(f"Found validation indices with shape: {val_indices.shape}")
        
        # Split features if indices are available
        if train_indices is not None and val_indices is not None:
            train_features = features[train_indices]
            val_features = features[val_indices]
            print(f"Split features using provided indices - Train: {train_features.shape}, Val: {val_features.shape}")
            return train_features, val_features
        
        # If no split information is available, do a simple 80/20 split
        print("No pre-defined split found. Using default 80/20 train/val split")
        train_size = int(0.8 * features.shape[0])
        
        train_features = features[:train_size]
        val_features = features[train_size:]
        
        print(f"Default split - Train: {train_features.shape}, Val: {val_features.shape}")
        return train_features, val_features
        
    except Exception as e:
        print(f"Error loading features from {custom_features_file}: {e}")
        print("Using a default random feature matrix")
        # Return a dummy feature matrix (1000 samples, 1024 features)
        import numpy as np
        features = np.random.randn(1000, 1024).astype(np.float32)
        print(f"Created dummy feature matrix with shape {features.shape}")
        
        # Split for train/val
        train_size = int(0.8 * features.shape[0])
        train_features = features[:train_size]
        val_features = features[train_size:]
        
        return train_features, val_features

=======
>>>>>>> parent of 466e95f (checkpoint)
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
        graph_file = os.path.join(graphs_dir, f"{args.model_id}_{model_name}_{args.feature_dimension}_{args.gephi_subset_size}.gexf")
        
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
<<<<<<< HEAD
    
    # Handle special case for custom dataset (GPT Neo)
    is_gptneo = (args.dataset == 'custom' and args.custom_features_file and 
                'gptneo' in args.custom_features_file.lower())
    
    if is_gptneo:
        print(f"GPT Neo features file: {args.custom_features_file}")
        
        # Check for split files
        if args.custom_train_file and args.custom_val_file:
            print(f"GPT Neo train file: {args.custom_train_file}")
            print(f"GPT Neo val file: {args.custom_val_file}")
            print("Using text-based train/val split")
        
        # Try to extract layer info
        import re
        layer_match = re.search(r'layer(\d+)_features', args.custom_features_file)
        if layer_match:
            print(f"GPT Neo layer: {layer_match.group(1)}")
        print(f"  Data Type: {args.data_type}")
    else:
        print(f"  Train: {args.train_dataset} (n={args.n_train})")
        print(f"  Val: {args.val_dataset} (n={args.n_val})")
        print(f"  Data Type: {args.data_type}")
        if args.data_type == 'text':
            print(f"  Embedding Model: {args.embedding_model} ({args.embedding_model_path})")
        if args.data_type == 'custom':
            print(f"  Custom Features File: {args.custom_features_file}")
        print(f"  Feature Column(s): {args.feature_column}")
        print(f"  Label Column: {args.label_column}")
=======
    print(f"  Train: {args.train_dataset} (n={args.n_train})")
    print(f"  Val: {args.val_dataset} (n={args.n_val})")
    print(f"  Data Type: {args.data_type}")
    if args.data_type == 'text':
        print(f"  Embedding Model: {args.embedding_model} ({args.embedding_model_path})")
    print(f"  Feature Column(s): {args.feature_column}")
    print(f"  Label Column: {args.label_column}")
    print(f"Model: {args.model_type.upper()}")
    print(f"  Input Dimension (n): {args.input_dimension}")
    print(f"  Feature Dimension (m): {args.feature_dimension or 100}")
>>>>>>> parent of 466e95f (checkpoint)
    
    # Set device
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
<<<<<<< HEAD
    
    # For custom datasets (like GPT Neo), we need to load features first to determine input dimension
    if args.data_type == 'custom' and args.custom_features_file:
        print(f"Loading custom features to determine dimensions...")
        # Get feature extraction function for custom data
        feature_extraction_fn = get_feature_extraction_fn(args.data_type)
        
        # Create a temporary dataframe for feature extraction
        temp_df = pd.DataFrame({'dummy': range(100)})
        
        # Extract features to determine dimensions
        try:
            # Check if separate train/val files are available
            if args.custom_train_file and args.custom_val_file and \
               os.path.exists(args.custom_train_file) and os.path.exists(args.custom_val_file):
                train_feature_extract, val_feature_extract = feature_extraction_fn(
                    temp_df, 
                    custom_features_file=args.custom_features_file,
                    custom_train_file=args.custom_train_file,
                    custom_val_file=args.custom_val_file
                )
            else:
                train_feature_extract, val_feature_extract = feature_extraction_fn(
                    temp_df, 
                    custom_features_file=args.custom_features_file
                )
            
            # Determine input dimension from features
            n = train_feature_extract.shape[1]  # Set input dimension from features
            print(f"Detected input dimension from features: n={n}")
            args.input_dimension = n
        except Exception as e:
            print(f"Error loading custom features to determine dimensions: {e}")
            print("Using default input dimension of 1024")
            n = 1024
            args.input_dimension = n
    else:
        # Use specified input dimension for standard datasets
        n = args.input_dimension
        
=======
>>>>>>> parent of 466e95f (checkpoint)
    # Setup feature dimensions
    n = args.input_dimension
    m = args.feature_dimension if args.feature_dimension else 100
    
    # Calculate default attention dimension to match parameter count, considering the implementation and approach
    if args.attention_dimension is None:
        # For both new and old ST implementation, respect use_memory_bank flag
        use_direct_kv = not args.use_memory_bank
        args.attention_dimension = calculate_attention_dim_for_equal_params(n, m, use_direct_kv)
        a = args.attention_dimension
        approach_type = "memory bank" if args.use_memory_bank else "direct K-V" 
        print(f"  Attention Dimension (a): {a} (auto-calculated for balanced parameter count with {approach_type} approach)")
    else:
        a = args.attention_dimension
        print(f"  Attention Dimension (a): {a} (user-specified)")
    
    if args.model_type in ['st', 'both']:
        if args.use_old_st:
            print(f"  Using original ST implementation (ST_old.py)")
        else:
            print(f"  Using new ST implementation (ST.py)")
            print(f"  ST Architecture: {'Memory Bank' if args.use_memory_bank else 'Direct K-V Matrices'}")

            print(f"Attention function: {args.attention_fn}")

    
    # Calculate optimal training steps if auto_steps is enabled
    if args.auto_steps:
        original_steps = args.target_steps
        
        # Calculate optimal steps based on model type
        if args.model_type in ["sae", "both"]:
            sae_optimal_steps = calculate_optimal_training_steps(
                feature_dimension=m,
                input_dimension=n,
                model_type='sae',
                base_steps=args.auto_steps_base,
                min_steps=args.auto_steps_min,
                max_steps=args.auto_steps_max
            )
            if args.model_type == "sae":
                args.target_steps = sae_optimal_steps
                print(f"\n=== Auto-calculated optimal steps for SAE ===")
                print(f"Feature dimension (m): {m}, Input dimension (n): {n}")
                print(f"Feature ratio (m/n): {m/n:.2f}")
                print(f"Optimal training steps: {args.target_steps:,} (was: {original_steps:,})")
                print(f"SAE activation function: {args.activation}")
                print("="*50)
        
        if args.model_type in ["st", "both"]:
            st_optimal_steps = calculate_optimal_training_steps(
                feature_dimension=m,
                input_dimension=n,
                model_type='st',
                base_steps=args.auto_steps_base,
                min_steps=args.auto_steps_min,
                max_steps=args.auto_steps_max
            )
            if args.model_type == "st":
                args.target_steps = st_optimal_steps
                print(f"\n=== Auto-calculated optimal steps for ST ===")
                print(f"Feature dimension (m): {m}, Input dimension (n): {n}")
                print(f"Feature ratio (m/n): {m/n:.2f}")
                print(f"Optimal training steps: {args.target_steps:,} (was: {original_steps:,})")
                print(f"SAE activation function: {args.activation}") 
                print("="*50)
        
        # For "both" model type, use the larger of the two
        if args.model_type == "both":
            args.target_steps = max(sae_optimal_steps, st_optimal_steps)
            print(f"\n=== Auto-calculated optimal steps for SAE+ST ===")
            print(f"Feature dimension (m): {m}, Input dimension (n): {n}")
            print(f"Feature ratio (m/n): {m/n:.2f}")
            print(f"Optimal training steps: {args.target_steps:,} (was: {original_steps:,})")
            print(f"  - SAE optimal steps: {sae_optimal_steps:,}")
            print(f"  - ST optimal steps: {st_optimal_steps:,}")
            print("="*50)
    
    print(f"Training:")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Target Steps: {args.target_steps}")
    print(f"  L1 Lambda: {args.l1_lambda}")
    print(f"  Force Retrain: {args.force_retrain}")
    print("="*50)
    
    # Model parameters
    model_params = {
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'target_steps': args.target_steps,
        'l1_lambda': args.l1_lambda,
        'force_retrain': args.force_retrain,
    }
    
<<<<<<< HEAD
    # Load datasets or custom features based on data type
    # For custom datasets (like GPT Neo), we need to load features first to determine input dimension
    if args.data_type == 'custom' and args.custom_features_file:
        print(f"Loading custom features to determine dimensions...")
        # Get feature extraction function for custom data
        feature_extraction_fn = get_feature_extraction_fn(args.data_type)
        
        # Create a temporary dataframe for feature extraction
        temp_df = pd.DataFrame({'dummy': range(100)})
        
        # Extract features to determine dimensions
        try:
            # Check if separate train/val files are available
            if args.custom_train_file and args.custom_val_file and \
               os.path.exists(args.custom_train_file) and os.path.exists(args.custom_val_file):
                train_feature_extract, val_feature_extract = feature_extraction_fn(
                    temp_df, 
                    custom_features_file=args.custom_features_file,
                    custom_train_file=args.custom_train_file,
                    custom_val_file=args.custom_val_file
                )
            else:
                train_feature_extract, val_feature_extract = feature_extraction_fn(
                    temp_df, 
                    custom_features_file=args.custom_features_file
                )
            
            # Determine input dimension from features
            n = train_feature_extract.shape[1]  # Set input dimension from features
            print(f"Detected input dimension from features: n={n}")
            args.input_dimension = n
        except Exception as e:
            print(f"Error loading custom features to determine dimensions: {e}")
            print("Using default input dimension of 1024")
            n = 1024
            args.input_dimension = n
    else:
        # Use specified input dimension for standard datasets
        n = args.input_dimension

    # The rest of the main function...
    
    # Load datasets or custom features based on data type
    if args.data_type == 'custom':
        print(f"Using custom data type with features file: {args.custom_features_file}")
        
        # For custom data type, we'll create dummy dataframes and extract features directly from the file
        train_df = pd.DataFrame({'dummy': range(100)})  # Will be populated later
        val_df = pd.DataFrame({'dummy': range(20)})     # Will be populated later
        
        # Get feature extraction function for custom data
        feature_extraction_fn = get_feature_extraction_fn(args.data_type)
        
        # Extract features from custom file
        print(f"Extracting features from custom file...")
        try:
            # Check if separate train/val files are available
            if args.custom_train_file and args.custom_val_file and \
               os.path.exists(args.custom_train_file) and os.path.exists(args.custom_val_file):
                train_feature_extract, val_feature_extract = feature_extraction_fn(
                    train_df, 
                    custom_features_file=args.custom_features_file,
                    custom_train_file=args.custom_train_file,
                    custom_val_file=args.custom_val_file
                )
            else:
                train_feature_extract, val_feature_extract = feature_extraction_fn(
                    train_df, 
                    custom_features_file=args.custom_features_file
                )
            
            # Determine input dimension from features
            if n is None or n != train_feature_extract.shape[1]:
                print(f"Setting input dimension to {train_feature_extract.shape[1]} based on loaded features")
                n = train_feature_extract.shape[1]
                args.input_dimension = n
            
            print(f"Feature matrix shapes - Train: {train_feature_extract.shape}, Val: {val_feature_extract.shape}")
            
            # Create proper train and val dataframes with the right number of rows
            train_df = pd.DataFrame({'dummy': range(len(train_feature_extract))})
            val_df = pd.DataFrame({'dummy': range(len(val_feature_extract))})
            
            # Handle token groups
            if 'token_group' in train_df.columns:
                print("Using token groups for model evaluation")
                # Also transfer token groups to validation dataframe
                if 'token_group' not in val_df.columns:
                    try:
                        val_df['token_group'] = train_df['token_group'].iloc[:len(val_df)]
                    except:
                        print("Could not transfer token groups to validation dataframe")
                        
        except Exception as e:
            print(f"Error processing custom features: {e}")
            print("Creating fallback feature matrices for training")
            
            # Create fallback feature matrices with correct dimensions
            train_size = 800
            val_size = 200
            feature_dim = n if n is not None else 1024
            
            print(f"Using fallback dimensions - n={feature_dim}, train={train_size}, val={val_size}")
            
            train_feature_extract = np.random.randn(train_size, feature_dim).astype(np.float32)
            val_feature_extract = np.random.randn(val_size, feature_dim).astype(np.float32)
            
            # Create dummy dataframes
            train_df = pd.DataFrame({'dummy': range(train_size)})
            val_df = pd.DataFrame({'dummy': range(val_size)})
            
            # Set input dimension if not already set
            if n is None:
                n = feature_dim
                args.input_dimension = n
            
        # Set train and val sample dataframes (same as original for custom data)
        train_sample_df = train_df
        val_sample_df = val_df
    else:
        # Regular dataset loading for non-custom data types
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
=======
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
>>>>>>> parent of 466e95f (checkpoint)
    
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
        print("No feature dimension specified, defaulting to 100")
        args.feature_dimension = 100
    if args.attention_dimension is None:
        a = calculate_attention_dim_for_equal_params(n, m)
        print(f"  Attention Dimension also updated to: {a} (to maintain balanced parameter count)")
    
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
        
        # Create model path
        dataset_name = os.path.splitext(os.path.basename(args.train_dataset))[0]
        model_suffix = f"{args.model_id}_{args.feature_dimension}"
        if args.data_type == 'text':
            model_suffix += f"_{args.embedding_model}"
        # For SAE model path
        if args.sae_save_path:
            sae_model_path = args.sae_save_path
        else:
            # Use hierarchical path
            sae_model_path = get_hierarchical_model_path(args, 'sae')
        print(f"SAE model will be saved to: {sae_model_path}")

        
        print(f"SAE model path: {sae_model_path}")
        
        # Initialize SAE model
        sae_model = SparseAutoencoder(
            n=n,
            m=m,
            sae_model_path=sae_model_path,
            lambda_l1=args.l1_lambda,
            device=device,
            activation=args.activation
        )
        
        # Display model information if requested
        if args.show_model_info:
            # Prepare input size for torchinfo
            input_size = (args.batch_size, n)
            display_model_info(sae_model, "SAE", input_size, verbose=args.model_info_verbosity)
        
        # Train or load model
        if args.force_retrain or not os.path.exists(sae_model_path):
            print(f"Training SAE from scratch...")
            
            # Train the model
            sae_model.train_and_validate(
                train_tensor,
                val_tensor,
                learning_rate=model_params['learning_rate'],
                batch_size=model_params['batch_size'],
                target_steps=model_params['target_steps']
            )
            
            print(f"SAE model training completed and saved to {sae_model_path}")
        else:
            print(f"Loading pre-trained SAE model from {sae_model_path}")
            checkpoint = torch.load(sae_model_path)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                sae_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                sae_model.load_state_dict(checkpoint)
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
        model_suffix = f"{args.model_id}_{args.attention_dimension}_{args.feature_dimension}_{args.attention_fn}"
        if args.data_type == 'text':
            model_suffix += f"_{args.embedding_model}"
        
        # Add suffix based on implementation and architecture approach
        if args.use_old_st:
            model_suffix += "_old"
            # Add architecture approach suffix for old implementation too
            if args.use_memory_bank:
                model_suffix += "_memory"
            else:
                model_suffix += "_direct"  # Direct K-V is default for both implementations
            if args.st_save_path:
                st_model_path = args.st_save_path
            else:
                # Use hierarchical path
                st_model_path = get_hierarchical_model_path(args, 'st')
            print(f"ST model will be saved to: {st_model_path}")

        else:
            if args.use_memory_bank:
                model_suffix += "_memory"
            else:
                model_suffix += "_direct"
            if args.st_save_path:
                st_model_path = args.st_save_path
            else:
                # Use hierarchical path
                st_model_path = get_hierarchical_model_path(args, 'st')
            print(f"ST model will be saved to: {st_model_path}")

        
        print(f"ST model path: {st_model_path}")
        model_exists = os.path.exists(st_model_path)
        print(f"Model exists: {model_exists}")
        
        # Check if the ST model file exists
        if model_exists:
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
        
        # Create the appropriate ST model implementation
        if args.use_old_st:
            print("Using original ST implementation from ST_old.py")
            # Use the same direct K-V approach flag for both implementations
            st_model = ST_old.SparseTransformer(
                X=train_feature_extract,
                n=n,
                m=m,
                a=a,
                st_model_path=st_model_path,
                lambda_l1=args.l1_lambda,
                num_heads=1,
                device=device,
                activation_threshold=args.activation_threshold,
                use_direct_kv=not args.use_memory_bank,  # Direct K-V is default unless memory bank is requested
                activation="none",  # Pass the activation function parameter
                attention_fn=args.attention_fn
            )
        else:
            print("Using new ST implementation from ST.py")
            st_model = ST.SparseTransformer(
                X=train_feature_extract,
                n=n,
                m=m,
                a=a,
                st_model_path=st_model_path,
                lambda_l1=args.l1_lambda,
                num_heads=1,
                device=device,
                activation_threshold=args.activation_threshold,
                use_mixed_precision=args.use_mixed_precision,
                use_direct_kv=not args.use_memory_bank,  # Use direct K-V by default unless memory bank is requested
                activation="none",  # Pass the activation function parameter
                attention_fn=args.attention_fn
            )
        
        # Display model information if requested
        if args.show_model_info:
            # Prepare input size for torchinfo based on the model
            input_size = (args.batch_size, n)
            display_model_info(st_model, "ST", input_size, verbose=args.model_info_verbosity)
        
        # Train or load model
        print(f"Force retrain: {args.force_retrain}, Model exists: {model_exists}")
        if args.force_retrain or not model_exists:
            print(f"Training ST from scratch...")
            
            # Different training methods based on implementation
            if args.use_old_st:
                # Old ST implementation has simpler train_and_validate
                st_model.train_and_validate(
                    train_tensor,
                    val_tensor,
                    learning_rate=model_params['learning_rate'],
                    batch_size=model_params['batch_size'],
                    target_steps=model_params['target_steps']
                )
            else:
                # New ST implementation has more parameters
                st_model.train_and_validate(
                    train_tensor,
                    val_tensor,
                    learning_rate=model_params['learning_rate'],
                    batch_size=model_params['batch_size'],
                    target_steps=model_params['target_steps'],
                    grad_accum_steps=args.grad_accum_steps,
                    eval_freq=args.eval_freq,
                    resume_from=st_model_path+".step150000" if not os.path.exists(st_model_path) else None
                )
            
            print(f"ST model training completed and saved to {st_model_path}")
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
    
    # Display comparative parameter analysis if both models are trained
    if args.model_type == 'both' and args.show_model_info:
        print("\n" + "="*50)
        print("COMPARATIVE PARAMETER ANALYSIS")
        print("="*50)
        
        # Calculate total parameters for each model
        sae_params = sum(p.numel() for p in sae_model.parameters() if p.requires_grad)
        st_params = sum(p.numel() for p in st_model.parameters() if p.requires_grad)
        
        print(f"SAE Model: {sae_params:,} parameters")
        print(f"ST Model: {st_params:,} parameters")
        print(f"Difference: {st_params - sae_params:,} parameters ({st_params/sae_params:.2f}x)")
        
        # Display key dimensions
        print("\nModel Dimensions:")
        print(f"Input dimension (n): {n}")
        print(f"Feature dimension (m): {m}")
        print(f"Attention dimension (a): {a}")
        
        # Display key component sizes
        print("\nComponent Parameter Counts:")
        print("SAE Components:")
        for name, param in sae_model.named_parameters():
            if param.requires_grad:
                print(f"  {name}: {param.numel():,} parameters ({' × '.join(str(d) for d in param.shape)})")
        
        print("\nST Components:")
        key_components = []
        for name, param in st_model.named_parameters():
            if param.requires_grad:
                key_components.append((name, param.numel(), param.shape))
        
        # Sort by parameter count (largest first)
        for name, count, shape in sorted(key_components, key=lambda x: x[1], reverse=True):
            print(f"  {name}: {count:,} parameters ({' × '.join(str(d) for d in shape)})")
        
        print("="*50)
    
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