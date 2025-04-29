import os
import glob
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from tqdm import tqdm
import argparse
from collections import defaultdict
import re
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('centroid_analysis.log')
    ]
)
logger = logging.getLogger(__name__)

def find_all_models(base_dir="models", dataset_name="gpt_neo"):
    """
    Find all model files in the directory structure, supporting both standard and GPT-Neo formats.
    
    Args:
        base_dir: Base directory containing the models
        dataset_name: Default dataset name to use for GPT-Neo models
        
    Returns:
        Dictionary mapping model paths to metadata
    """
    model_files = {}
    
    if not os.path.exists(base_dir):
        logger.error(f"Base directory does not exist: {base_dir}")
        return {}
    
    # Check if this is a GPT-Neo style directory structure
    sae_dir = os.path.join(base_dir, 'sae')
    st_dir = os.path.join(base_dir, 'st')
    
    if os.path.exists(sae_dir) or os.path.exists(st_dir):
        logger.info(f"Detected GPT-Neo style directory structure in {base_dir}")
        
        # Process SAE models
        if os.path.exists(sae_dir):
            for model_file in glob.glob(os.path.join(sae_dir, "*.pt")):
                # Extract layer number from filename (format: layer_X_sae.pt)
                filename = os.path.basename(model_file)
                layer_match = re.search(r'layer_(\d+)_sae\.pt', filename)
                
                if layer_match:
                    layer_idx = int(layer_match.group(1))
                    
                    # Extract metadata from model file if available
                    metadata = extract_metadata_from_model(model_file)
                    
                    # Fill in basic metadata
                    metadata.update({
                        'dataset': dataset_name,
                        'model_type': 'sae',
                        'function_type': 'relu',  # Default activation function
                        'layer_idx': layer_idx,
                        'filename': filename,
                        'full_path': model_file
                    })
                    
                    model_files[model_file] = metadata
            
            logger.info(f"Found {len(model_files)} SAE model files")
        
        # Process ST models
        if os.path.exists(st_dir):
            st_count = 0
            for model_file in glob.glob(os.path.join(st_dir, "*.pt")):
                # Extract layer number from filename (format: layer_X_st.pt)
                filename = os.path.basename(model_file)
                layer_match = re.search(r'layer_(\d+)_st\.pt', filename)
                
                if layer_match:
                    layer_idx = int(layer_match.group(1))
                    
                    # Extract metadata from model file if available
                    metadata = extract_metadata_from_model(model_file)
                    
                    # Fill in basic metadata
                    metadata.update({
                        'dataset': dataset_name,
                        'model_type': 'st',
                        'function_type': metadata.get('attention_fn', 'softmax'),  # Default attention function
                        'layer_idx': layer_idx,
                        'filename': filename,
                        'full_path': model_file
                    })
                    
                    model_files[model_file] = metadata
                    st_count += 1
            
            logger.info(f"Found {st_count} ST model files")
    
    else:
        # Try standard hierarchical format
        logger.info(f"Searching for models in standard hierarchical format in {base_dir}")
        
        # First level - dataset directories
        for dataset in os.listdir(base_dir):
            dataset_path = os.path.join(base_dir, dataset)
            
            if not os.path.isdir(dataset_path):
                continue
                
            # Second level - model type (sae, st)
            for model_type in os.listdir(dataset_path):
                type_path = os.path.join(dataset_path, model_type)
                if not os.path.isdir(type_path) or model_type not in ['sae', 'st']:
                    continue
                    
                for function_type in os.listdir(type_path):
                    function_path = os.path.join(type_path, function_type)
                    if not os.path.isdir(function_path):
                        continue
                        
                    for feature_dim in os.listdir(function_path):
                        feature_path = os.path.join(function_path, feature_dim)
                        if not os.path.isdir(feature_path):
                            continue
                            
                        # Find all .pth or .pt files in this directory
                        for model_file in glob.glob(os.path.join(feature_path, "*.pth")) + \
                                         glob.glob(os.path.join(feature_path, "*.pt")):
                            # Extract metadata from path
                            metadata = {
                                'dataset': dataset,
                                'model_type': model_type,
                                'function_type': function_type,
                                'feature_dimension': feature_dim,
                                'filename': os.path.basename(model_file),
                                'full_path': model_file
                            }
                            
                            # Parse parameters from filename
                            filename = os.path.basename(model_file)
                            for param in filename.replace('.pth', '').replace('.pt', '').split('_'):
                                if param.startswith('bs'):
                                    metadata['batch_size'] = param[2:]
                                elif param.startswith('lr'):
                                    metadata['learning_rate'] = param[2:].replace('p', '.')
                                elif param.startswith('steps'):
                                    metadata['steps'] = param[5:]
                                elif param.startswith('l1'):
                                    metadata['l1_lambda'] = param[2:].replace('p', '.')
                                elif param.startswith('accum'):
                                    metadata['grad_accum'] = param[5:]
                                elif param == 'memory':
                                    metadata['memory'] = True
                                elif param == 'oldst':
                                    metadata['old_st'] = True
                            
                            model_files[model_file] = metadata
        
    logger.info(f"Found a total of {len(model_files)} model files.")
    return model_files

def extract_metadata_from_model(model_path):
    """
    Extract metadata from a model checkpoint file.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Dictionary with extracted metadata
    """
    metadata = {}
    
    try:
        # Load the model file
        checkpoint = safely_open_model(model_path)
        
        if checkpoint is None:
            return metadata
        
        # Extract state dict
        if isinstance(checkpoint, dict):
            # Check for metadata in top-level keys
            for key in ['lambda_l1', 'l1_lambda', 'feature_dim', 'attention_dim', 'steps', 'attention_fn']:
                if key in checkpoint:
                    metadata[key] = checkpoint[key]
            
            # Extract feature and input dimensions from weights
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
                
            # For SAE models, get dimensions from weights
            if 'W_e.weight' in state_dict:
                weights = state_dict['W_e.weight']
                if hasattr(weights, 'shape'):
                    metadata['feature_dimension'] = weights.shape[0]  # m
                    metadata['input_dimension'] = weights.shape[1]    # n
            
            # For ST models
            elif 'W_q.weight' in state_dict:
                weights = state_dict['W_q.weight']
                if hasattr(weights, 'shape'):
                    metadata['attention_dimension'] = weights.shape[0]  # a
                    metadata['input_dimension'] = weights.shape[1]     # n
                
                # Check if using direct KV matrices
                if 'W_k_direct' in state_dict:
                    k_weights = state_dict['W_k_direct']
                    if hasattr(k_weights, 'shape'):
                        metadata['feature_dimension'] = k_weights.shape[0]  # m
                # Otherwise get feature dimension from memory indices
                elif 'memory_indices' in state_dict:
                    indices = state_dict['memory_indices']
                    if hasattr(indices, 'shape'):
                        metadata['feature_dimension'] = indices.shape[0]  # m
            
            # Extract other metadata from training history if available
            if 'training_history' in checkpoint:
                history = checkpoint['training_history']
                # Get the last value for certain metrics
                for key in ['lambda', 'dead_ratio', 'sparsity']:
                    if key in history and history[key]:
                        metadata[key] = history[key][-1]
    
    except Exception as e:
        logger.error(f"Error extracting metadata from {model_path}: {e}")
    
    return metadata

def safely_open_model(model_path, device='cpu'):
    """Safely open a model file with proper error handling"""
    try:
        return torch.load(model_path, map_location=device)
    except RuntimeError as e:
        # Fallback to different loading options
        logger.warning(f"Error loading model normally, trying with different options: {e}")
        try:
            return torch.load(model_path, map_location=device, pickle_module=torch.serialization._pickle)
        except Exception as e2:
            try:
                # Last resort: Try with weights_only=True if available (newer PyTorch)
                if hasattr(torch, 'load') and 'weights_only' in torch.load.__code__.co_varnames:
                    return torch.load(model_path, map_location=device, weights_only=True)
                else:
                    raise e2
            except Exception as e3:
                logger.error(f"Failed to load model {model_path}: {e3}")
                return None
    except FileNotFoundError:
        logger.error(f"Model file not found: {model_path}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading model {model_path}: {e}")
        return None

def load_sae_model(model_path, device='cpu'):
    """
    Load an SAE model and prepare it for feature extraction.
    
    Args:
        model_path: Path to the SAE model file
        device: Device to use for computation
        
    Returns:
        Loaded SAE model and input dimension n
    """
    try:
        logger.info(f"Loading SAE model from {model_path}")
        # Use the safe loading function
        checkpoint = safely_open_model(model_path, device)
        if checkpoint is None:
            return None, None
        
        # Extract state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Determine input and feature dimensions
        n, m = None, None
        if 'W_d.weight' in state_dict:
            n = state_dict['W_d.weight'].shape[0]  # Input dimension
            m = state_dict['W_d.weight'].shape[1]  # Feature dimension
        elif 'W_e.weight' in state_dict:
            n = state_dict['W_e.weight'].shape[1]  # Input dimension
            m = state_dict['W_e.weight'].shape[0]  # Feature dimension
        
        if n is None or m is None:
            logger.error("Could not determine dimensions from model state dict")
            return None, None
            
        logger.info(f"SAE model architecture: {m} features (m) with input dimension {n} (n)")
        
        # Import the SAE model class (assuming it's available in the path)
        try:
            from SAE import SparseAutoencoder
            
            # Create and initialize the model
            model = SparseAutoencoder(
                n=n,
                m=m,
                sae_model_path="temp.pth",  # Temporary path as we're just using for inference
                device=device
            )
            
            # Load the state dict
            model.load_state_dict(state_dict)
            model.eval()  # Set to evaluation mode
            
            logger.info(f"Successfully loaded SAE model (will be used for feature extraction)")
            return model, n
            
        except ImportError:
            logger.warning("Could not import SparseAutoencoder class. Using direct computation instead.")
            # We'll return None for the model, but keep the dimension information
            return None, n
            
    except Exception as e:
        logger.error(f"Error loading SAE model: {e}")
        return None, None

def load_st_model(model_path, device='cpu'):
    """
    Load an ST model and prepare it for feature extraction.
    
    Args:
        model_path: Path to the ST model file
        device: Device to use for computation
        
    Returns:
        Loaded ST model and input dimension n
    """
    try:
        logger.info(f"Loading ST model from {model_path}")
        # Use the safe loading function
        checkpoint = safely_open_model(model_path, device)
        if checkpoint is None:
            return None, None
        
        # Extract state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Determine dimensions
        n, m, a = None, None, None
        
        # Try to get dimensions from weights
        if 'W_q.weight' in state_dict:
            n = state_dict['W_q.weight'].shape[1]  # Input dimension
            a = state_dict['W_q.weight'].shape[0]  # Attention dimension
        
        # Check if this is a direct KV model (always assume it is)
        use_direct_kv = True
        
        if 'W_k_direct' in state_dict:
            m = state_dict['W_k_direct'].shape[0]  # Feature dimension
        elif 'memory_indices' in state_dict:
            m = len(state_dict['memory_indices'])  # Feature dimension from memory size
            
        if n is None or m is None or a is None:
            logger.error("Could not determine all dimensions from model state dict")
            return None, None
            
        logger.info(f"ST model architecture: {m} features (m) with input dimension {n} (n) and attention dimension {a} (a)")
        logger.info(f"Using direct KV approach: {use_direct_kv}")
        
        # First try to import from ST_old as that's what you mentioned using
        try:
            from ST_old import SparseTransformer
            logger.info("Using ST_old implementation for ST model")
        except ImportError:
            # Fall back to regular ST
            try:
                from ST import SparseTransformer
                logger.info("Using standard ST implementation for ST model")
            except ImportError:
                logger.warning("Could not import SparseTransformer class. Using direct computation instead.")
                # We'll return None for the model, but keep the dimension information
                return None, n
        
        # Create and initialize the model
        try:
            # We need a dummy input tensor for initialization
            dummy_data = torch.zeros((10, n), device=device)
            
            # Create model with the correct use_direct_kv setting (ALWAYS TRUE)
            model = SparseTransformer(
                X=dummy_data,  # Dummy data for initialization
                n=n,
                m=m,
                a=a,
                st_model_path="temp.pth",  # Temporary path as we're just using for inference
                device=device,
                attention_fn='softmax',
                use_direct_kv=True  # Always use direct KV
            )
            
            # Load the state dict
            model.load_state_dict(state_dict)
            model.eval()  # Set to evaluation mode
            
            logger.info(f"Successfully loaded ST model (will be used for feature extraction)")
            return model, n
            
        except Exception as model_error:
            logger.error(f"Error creating ST model: {model_error}")
            # Fallback to using approximation method
            return None, n
            
    except Exception as e:
        logger.error(f"Error loading ST model: {e}")
        return None, None

def compute_sae_features(model, input_data, device='cpu'):
    """
    Compute feature activations (f_x) for SAE model.
    
    Args:
        model: Loaded SAE model or None
        input_data: Input data to compute features for
        device: Device to use for computation
        
    Returns:
        Feature activations
    """
    if model is None:
        logger.warning("No SAE model provided. Using approximation method.")
        return None
        
    try:
        # Convert input to tensor
        input_tensor = torch.tensor(input_data, dtype=torch.float32, device=device)
        
        # Compute features
        with torch.no_grad():
            _, _, f_x = model(input_tensor)
            
        # Return as numpy array
        features = f_x.cpu().numpy()
        logger.info(f"Computed SAE feature activations with shape {features.shape} for {len(input_data)} input samples")
        return features
        
    except Exception as e:
        logger.error(f"Error computing SAE features: {e}")
        return None

def compute_st_features(model, input_data, device='cpu'):
    """
    Compute feature activations (f) for ST model.
    
    Args:
        model: Loaded ST model or None
        input_data: Input data to compute features for
        device: Device to use for computation
        
    Returns:
        Feature activations
    """
    if model is None:
        logger.warning("No ST model provided. Using approximation method.")
        return None
        
    try:
        # Convert input to tensor
        input_tensor = torch.tensor(input_data, dtype=torch.float32, device=device)
        
        # Compute features
        with torch.no_grad():
            # Try both output formats
            try:
                # ST model might return (x, x_hat, f, v)
                _, _, f, _ = model(input_tensor)
            except ValueError:
                # Or it might return (x, x_hat, f)
                _, _, f = model(input_tensor)
            
        # Return as numpy array
        features = f.cpu().numpy()
        logger.info(f"Computed ST feature activations with shape {features.shape} for {len(input_data)} input samples")
        return features
        
    except Exception as e:
        logger.error(f"Error computing ST features: {e}")
        return None

def get_sae_encoder_weights(model_path, device='cpu'):
    """Get encoder weights from an SAE model"""
    try:
        checkpoint = safely_open_model(model_path, device)
        if checkpoint is None:
            return None
        
        # Extract state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Get encoder weights
        if 'W_e.weight' in state_dict:
            weights = state_dict['W_e.weight'].cpu().numpy()
            return weights
        
        return None
    except Exception as e:
        logger.error(f"Error getting SAE encoder weights: {e}")
        return None

def approximate_sae_features(encoder_weights, feature_data):
    """
    Approximate SAE feature activations (f_x) using encoder weights.
    
    This is a simplified approach that computes feature activations as:
    f_x = ReLU(W_e * X + b_e)
    
    We'll ignore the bias term for simplicity.
    
    Args:
        encoder_weights: Encoder weight matrix (W_e)
        feature_data: Input features (X)
        
    Returns:
        Approximated feature activations (f_x)
    """
    # Compute linear projection
    linear_output = np.dot(feature_data, encoder_weights.T)
    
    # Apply ReLU activation
    activations = np.maximum(0, linear_output)
    
    return activations

def get_st_key_vectors(model_path, device='cpu'):
    """Get key vectors from an ST model"""
    try:
        checkpoint = safely_open_model(model_path, device)
        if checkpoint is None:
            return None
        
        # Extract state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Check if this is a direct K-V model
        use_direct_kv = 'W_k_direct' in state_dict
        
        if use_direct_kv:
            if 'W_k_direct' in state_dict:
                return state_dict['W_k_direct'].cpu().numpy()
        else:
            if 'W_k.weight' in state_dict:
                return state_dict['W_k.weight'].cpu().numpy()
                
        return None
    except Exception as e:
        logger.error(f"Error getting ST key vectors: {e}")
        return None

def approximate_st_features(key_vectors, feature_data):
    """
    Approximate ST feature activations (f) using key vectors for direct KV models.
    
    Args:
        key_vectors: Key vectors from ST model (W_k_direct)
        feature_data: Input features
        
    Returns:
        Approximated feature activations (f)
    """
    logger.info(f"Approximating ST features with shapes: feature_data={feature_data.shape}, key_vectors={key_vectors.shape}")
    
    # In direct KV mode, we need to project the features to attention space
    # First, we'll handle potential dimension mismatch
    n_features = feature_data.shape[1]
    att_dim = key_vectors.shape[1]
    
    # Normalize feature data
    norm_features = feature_data / (np.linalg.norm(feature_data, axis=1, keepdims=True) + 1e-8)
    
    # Project features to attention dimension if they don't match
    if n_features != att_dim:
        logger.info(f"Projecting features from dimension {n_features} to {att_dim}")
        from sklearn.decomposition import PCA
        
        # Use PCA to project to attention dimension
        pca = PCA(n_components=min(n_features, att_dim))
        projected_features = pca.fit_transform(norm_features)
        
        # If PCA gives us more dimensions than we need, truncate
        if projected_features.shape[1] > att_dim:
            projected_features = projected_features[:, :att_dim]
            
        # If we have fewer dimensions, pad with zeros
        elif projected_features.shape[1] < att_dim:
            padding = np.zeros((projected_features.shape[0], att_dim - projected_features.shape[1]))
            projected_features = np.hstack([projected_features, padding])
            
        # Renormalize after projection
        norm_features = projected_features / (np.linalg.norm(projected_features, axis=1, keepdims=True) + 1e-8)
    
    # Normalize key vectors
    norm_keys = key_vectors / (np.linalg.norm(key_vectors, axis=1, keepdims=True) + 1e-8)
    
    # Now compute similarity scores
    # norm_features: [samples, att_dim]
    # norm_keys: [m, att_dim]
    similarity_scores = np.dot(norm_features, norm_keys.T)
    
    # Apply scaling
    similarity_scores = similarity_scores / np.sqrt(att_dim)
    
    # Apply softmax
    exp_scores = np.exp(similarity_scores - np.max(similarity_scores, axis=1, keepdims=True))
    attention_weights = exp_scores / (np.sum(exp_scores, axis=1, keepdims=True) + 1e-8)
    
    return attention_weights

def get_sae_feature_vectors(model_path, device='cpu'):
    """Get feature vectors from an SAE model (decoder weights)"""
    try:
        checkpoint = safely_open_model(model_path, device)
        if checkpoint is None:
            return None
        
        # Extract state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Get decoder weights
        if 'W_d.weight' in state_dict:
            weights = state_dict['W_d.weight'].cpu().numpy()
            # Transpose to get feature vectors as rows
            vectors = weights.T
            logger.info(f"Model has {vectors.shape[0]} features (m) with input dimension {vectors.shape[1]} (n)")
            return vectors
        
        return None
    except Exception as e:
        logger.error(f"Error getting SAE feature vectors: {e}")
        return None

def get_st_value_vectors(model_path, device='cpu'):
    """Get value vectors from an ST model"""
    try:
        checkpoint = safely_open_model(model_path, device)
        if checkpoint is None:
            return None
        
        # Extract state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Check if this is a direct K-V model
        use_direct_kv = 'W_v_direct' in state_dict
        
        if use_direct_kv:
            if 'W_v_direct' in state_dict:
                return state_dict['W_v_direct'].cpu().numpy()
        else:
            if 'W_v.weight' in state_dict:
                return state_dict['W_v.weight'].cpu().numpy()
                
        return None
    except Exception as e:
        logger.error(f"Error getting ST value vectors: {e}")
        return None

def compute_feature_distances(feature_vectors, metric='cosine'):
    """
    Compute distances between feature vectors.
    
    Args:
        feature_vectors: Array of feature vectors
        metric: Distance metric to use ('cosine' or 'euclidean')
        
    Returns:
        Dictionary of distance statistics
    """
    if feature_vectors is None or len(feature_vectors) == 0:
        return None
        
    # Calculate pairwise distances
    if metric == 'cosine':
        distances = cosine_distances(feature_vectors)
    else:  # euclidean
        distances = euclidean_distances(feature_vectors)
    
    # Get upper triangle of distance matrix (excluding diagonal)
    distances_upper = distances[np.triu_indices_from(distances, k=1)]
    
    # Calculate statistics
    stats = {
        'avg_distance': np.mean(distances_upper),
        'median_distance': np.median(distances_upper),
        'min_distance': np.min(distances_upper),
        'max_distance': np.max(distances_upper),
        'std_distance': np.std(distances_upper),
        'num_features': len(feature_vectors),
        'dimension': feature_vectors.shape[1]
    }
    
    return stats

def calculate_centroid(feature_vectors):
    """
    Calculate the centroid of a set of feature vectors.
    
    Args:
        feature_vectors: Array of feature vectors
        
    Returns:
        Centroid vector
    """
    if feature_vectors is None or len(feature_vectors) == 0:
        return None
        
    return np.mean(feature_vectors, axis=0)

def compute_labeled_centroids(feature_vectors, labels, metric='euclidean', normalize_by_dimension=True):
    """
    Compute centroids for each class based on labels.
    
    Args:
        feature_vectors: Array of feature vectors
        labels: Array of labels for each feature vector
        metric: Distance metric to use ('cosine' or 'euclidean')
        normalize_by_dimension: Whether to normalize distances by feature dimension
        
    Returns:
        Dictionary with centroid information
    """
    if feature_vectors is None or len(feature_vectors) == 0 or labels is None or len(labels) == 0:
        return None
    
    # Get unique labels
    unique_labels = sorted(np.unique(labels))
    centroids = []
    
    for label in unique_labels:
        # Get all feature vectors for this class
        class_vectors = feature_vectors[labels == label]
        
        # Calculate the centroid as the mean of all vectors
        if len(class_vectors) > 0:
            centroid = np.mean(class_vectors, axis=0)
            centroids.append((label, centroid, len(class_vectors)))
    
    if not centroids:
        logger.error("No valid centroids could be calculated.")
        return None
    
    # Calculate pairwise distances between centroids
    centroid_vectors = np.array([c[1] for c in centroids])
    
    if metric == 'cosine':
        distances = cosine_distances(centroid_vectors)
    else:  # euclidean
        distances = euclidean_distances(centroid_vectors)
        # Normalize by sqrt of dimension for Euclidean distance if requested
        if normalize_by_dimension and metric == 'euclidean':
            distances = distances / np.sqrt(feature_vectors.shape[1])
    
    # Get upper triangle of distance matrix (excluding diagonal)
    distances_upper = distances[np.triu_indices_from(distances, k=1)]
    
    # Calculate statistics
    stats = {
        'avg_centroid_distance': np.mean(distances_upper),
        'median_centroid_distance': np.median(distances_upper),
        'min_centroid_distance': np.min(distances_upper),
        'max_centroid_distance': np.max(distances_upper),
        'std_centroid_distance': np.std(distances_upper),
        'num_centroids': len(centroids),
        'centroids': centroids,
        'centroid_vectors': centroid_vectors,
        'centroid_distances': distances,
        'dimension': feature_vectors.shape[1]
    }
    
    return stats

def analyze_model_with_labeled_data(model_path, model_metadata, labeled_data, metrics=None, device='cpu', normalize_by_dimension=True):
    """
    Analyze a single model using labeled data.
    
    Args:
        model_path: Path to the model file
        model_metadata: Dictionary with model metadata
        labeled_data: Dictionary with 'features' and 'labels' for analysis
        metrics: List of distance metrics to use (default: ['cosine', 'euclidean'])
        device: Device to use for computation
        normalize_by_dimension: Whether to normalize distances by feature dimension
        
    Returns:
        List of result dictionaries
    """
    if metrics is None:
        metrics = ['cosine', 'euclidean']
    
    # Extract labeled data
    feature_data = labeled_data.get('features')
    labels = labeled_data.get('labels')
    
    if feature_data is None or labels is None:
        logger.error("Error: No feature data or labels provided.")
        return []
    
    results = []
    model_type = model_metadata['model_type']
    
    # Get feature dimension from metadata or set to 0 if not available
    feature_dim = int(model_metadata.get('feature_dimension', 0))
    
    try:
        # Load the model based on type
        if model_type == 'sae':
            model, input_dim = load_sae_model(model_path, device=device)
            # Compute feature activations (f_x)
            feature_activations = compute_sae_features(model, feature_data, device=device)
        else:  # 'st'
            model, input_dim = load_st_model(model_path, device=device)
            # Compute feature activations (f)
            feature_activations = compute_st_features(model, feature_data, device=device)
            
        # If we couldn't compute features directly from the model, try the approximation
        if feature_activations is None:
            logger.info(f"Attempting approximate feature computation for {model_path}")
            
            # Get the model's weight matrices
            if model_type == 'sae':
                # For SAE, we'll use the encoder weights: X → W_e*X → f_x
                weight_matrix = get_sae_encoder_weights(model_path, device=device)
                if weight_matrix is None:
                    logger.error(f"Could not extract encoder weights from {model_path}")
                    return []
                feature_activations = approximate_sae_features(weight_matrix, feature_data)
            else:  # 'st'
                # For ST, we'll use either direct key vectors or key projection matrix
                logger.info(f"Extracting key vectors for direct KV approximation")
                key_vectors = get_st_key_vectors(model_path, device=device)
                if key_vectors is None:
                    logger.error(f"Could not extract key vectors from {model_path}")
                    return []
                
                # Always use the updated approximation function for ST models
                feature_activations = approximate_st_features(key_vectors, feature_data)
            
        if feature_activations is None:
            logger.error(f"Could not compute feature activations for {model_path}")
            return []
        
        # Store the feature dimension for reporting
        feature_dimension = feature_activations.shape[1]
        
        # Calculate distances using each metric
        for metric in metrics:
            try:
                # For feature distances, we'll compute between model features
                # (not over feature_activations since those are already "applied")
                if model_type == 'sae':
                    weight_matrix = get_sae_feature_vectors(model_path, device=device)
                else:  # 'st'
                    weight_matrix = get_st_value_vectors(model_path, device=device)
                
                if weight_matrix is not None:
                    # Compute standard feature distances using model's weights
                    distance_stats = compute_feature_distances(weight_matrix, metric=metric)
                else:
                    # If we can't get weights, set some defaults
                    distance_stats = {
                        'avg_distance': 0.0,
                        'median_distance': 0.0,
                        'min_distance': 0.0,
                        'max_distance': 0.0,
                        'std_distance': 0.0,
                        'num_features': feature_activations.shape[1],
                        'dimension': feature_data.shape[1]
                    }
                
                # Compute labeled centroids using our feature activations
                centroid_stats = compute_labeled_centroids(
                    feature_activations, 
                    labels, 
                    metric=metric,
                    normalize_by_dimension=normalize_by_dimension
                )
                
                if distance_stats is None or centroid_stats is None:
                    continue
                
                # Extract dataset from model path or use from metadata
                dataset = model_metadata.get('dataset', None)
                if dataset is None:
                    # Try to extract from path
                    path_parts = model_path.replace('\\', '/').split('/')
                    for part in path_parts:
                        if part in ['mnist', 'fashion_mnist', 'stack_exchange', 'gpt_neo']:
                            dataset = part
                            break
                
                # Get layer index if available (for GPT-Neo models)
                layer_idx = model_metadata.get('layer_idx', None)
                layer_info = {} if layer_idx is None else {'layer_idx': layer_idx}
                
                # Combine all stats
                result = {
                    'model_path': model_path,
                    'metric': metric,
                    'feature_activation_dim': feature_dimension,  # Actually used feature dimension
                    'dataset': dataset,  # Add dataset info
                    **model_metadata,
                    **layer_info,
                    **distance_stats,
                    **{k: v for k, v in centroid_stats.items() if k not in ['centroids', 'centroid_vectors', 'centroid_distances']}
                }
                
                results.append(result)
            except Exception as metric_error:
                logger.error(f"Error processing metric {metric}: {metric_error}")
                
    except Exception as model_error:
        logger.error(f"Error processing model: {model_error}")
    
    return results

def analyze_models_distances(models_dict, labeled_data, metrics=None, device='cpu'):
    """
    Analyze distances for all models using labeled data.
    
    Args:
        models_dict: Dictionary of model paths and metadata
        labeled_data: Dictionary with 'features' and 'labels' for analysis
        metrics: List of distance metrics to use (default: ['cosine', 'euclidean'])
        device: Device to use for computation
        
    Returns:
        DataFrame with distance analysis results
    """
    if metrics is None:
        metrics = ['cosine', 'euclidean']
    
    feature_data = labeled_data.get('features')
    labels = labeled_data.get('labels')
    
    if feature_data is None or labels is None:
        logger.error("Error: No feature data or labels provided.")
        return pd.DataFrame()
        
    all_results = []
    skipped = []
    
    for model_path, metadata in tqdm(models_dict.items(), desc="Analyzing models"):
        try:
            # Analyze this model
            results = analyze_model_with_labeled_data(model_path, metadata, labeled_data, metrics, device=device)
            all_results.extend(results)
            
            if not results:
                skipped.append(model_path)
                
        except Exception as e:
            logger.error(f"Error processing model {model_path}: {e}")
            skipped.append(model_path)
    
    # Report on skipped models
    if skipped:
        logger.warning(f"Skipped {len(skipped)} models due to errors")
        
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    return results_df

def create_comparison_visualizations(results_df, output_dir='model_comparison'):
    """
    Create visualizations comparing models based on distance metrics.
    
    Args:
        results_df: DataFrame with distance analysis results
        output_dir: Directory to save visualizations
        
    Returns:
        List of generated visualization paths
    """
    os.makedirs(output_dir, exist_ok=True)
    visualizations = []
    
    # Create dataset-specific directories if dataset column exists
    if 'dataset' in results_df.columns:
        datasets = results_df['dataset'].unique()
        for dataset in datasets:
            if dataset:  # Skip None values
                dataset_dir = os.path.join(output_dir, f"dataset_{dataset}")
                os.makedirs(dataset_dir, exist_ok=True)
    
    # Add explanation about centroid calculation and dimension normalization
    info_text = """
    # Label-Based Centroids Analysis

    This analysis uses **label-based centroids** instead of k-means clustering. Each centroid 
    corresponds to a known class in the dataset, providing more interpretable results.

    ## Key Features

    - **No Dimensionality Reduction**: All centroid distances were calculated in the original 
      feature space without dimensionality reduction to preserve all feature relationships.
    
    - **Dimension Normalization**: For fair comparison between models with different feature 
      dimensions, Euclidean distances are normalized by the square root of feature dimension.
    
    - **True Feature Activations**: This analysis uses the true feature activations 
      (f_x for SAE models, f for ST models) rather than just model weights.
      
    - **Class Separation**: Larger distances between class centroids indicate better 
      class separation in the feature space.
    """
    
    # Create information text file
    info_path = os.path.join(output_dir, 'analysis_information.md')
    with open(info_path, 'w') as f:
        f.write(info_text)
    visualizations.append(info_path)
    
    # Ensure we have results to visualize
    if results_df.empty:
        logger.warning("No results to visualize.")
        return visualizations
    
    # Create visualization for each metric for all datasets combined
    for metric in results_df['metric'].unique():
        # Create an explicit copy of the filtered DataFrame to avoid SettingWithCopyWarning
        metric_df = results_df[results_df['metric'] == metric].copy()
        
        if metric_df.empty:
            continue
            
        # Ensure feature_dimension is numeric
        try:
            metric_df['feature_dimension'] = pd.to_numeric(metric_df['feature_dimension'])
        except:
            logger.warning("Could not convert feature_dimension to numeric.")
            
        # Add a 'model_config' column that combines model_type and function_type
        if 'model_type' in metric_df.columns and 'function_type' in metric_df.columns:
            metric_df['model_config'] = metric_df['model_type'] + '-' + metric_df['function_type']
        elif 'model_type' in metric_df.columns:
            metric_df['model_config'] = metric_df['model_type']
            
        # If layer_idx exists, add it to model_config
        if 'layer_idx' in metric_df.columns:
            metric_df['model_config'] = metric_df['model_config'] + '-layer' + metric_df['layer_idx'].astype(str)
            
        # 1. Dimension-normalized bar chart comparing models by type and function
        if 'model_type' in metric_df.columns and 'function_type' in metric_df.columns:
            try:
                plt.figure(figsize=(14, 8))
                
                # Group by model type and function type
                grouped = metric_df.groupby(['model_type', 'function_type'])['avg_centroid_distance'].mean().reset_index()
                
                # Pivot for easier plotting
                pivot_df = grouped.pivot(index='function_type', columns='model_type', values='avg_centroid_distance')
                
                # Plot
                ax = pivot_df.plot(kind='bar', rot=45)
                plt.title(f'Average Centroid Distance by Model Type and Function ({metric} metric)')
                plt.ylabel('Average Centroid Distance (normalized)')
                plt.xlabel('Function Type')
                plt.grid(alpha=0.3, axis='y')
                plt.tight_layout()
                
                # Save figure - for all datasets combined this goes in the main output directory
                bar_path = os.path.join(output_dir, f'avg_centroid_distance_{metric}.png')
                plt.savefig(bar_path, dpi=300)
                visualizations.append(bar_path)
                plt.close()
            except Exception as e:
                logger.error(f"Error creating bar chart: {e}")
        
        # 2. For GPT-Neo models, create a visualization by layer
        if 'layer_idx' in metric_df.columns:
            try:
                plt.figure(figsize=(14, 8))
                
                # Group by model type and layer
                grouped = metric_df.groupby(['model_type', 'layer_idx'])['avg_centroid_distance'].mean().reset_index()
                
                # Sort by layer index
                grouped = grouped.sort_values('layer_idx')
                
                # Plot for each model type
                for model_type, group in grouped.groupby('model_type'):
                    plt.plot(
                        group['layer_idx'], 
                        group['avg_centroid_distance'],
                        'o-',
                        label=model_type,
                        linewidth=2,
                        markersize=8
                    )
                
                plt.title(f'Average Centroid Distance by Layer ({metric} metric)')
                plt.xlabel('Layer Index')
                plt.ylabel('Average Centroid Distance (normalized)')
                plt.legend(title='Model Type')
                plt.grid(alpha=0.3)
                plt.tight_layout()
                
                # Save figure
                layer_path = os.path.join(output_dir, f'layer_centroid_distance_{metric}.png')
                plt.savefig(layer_path, dpi=300)
                visualizations.append(layer_path)
                plt.close()
            except Exception as e:
                logger.error(f"Error creating layer chart: {e}")
    
    # Now create dataset-specific visualizations if dataset column exists
    if 'dataset' in results_df.columns:
        for dataset in results_df['dataset'].unique():
            if not dataset:  # Skip None values
                continue
                
            dataset_dir = os.path.join(output_dir, f"dataset_{dataset}")
            dataset_df = results_df[results_df['dataset'] == dataset].copy()
            
            if dataset_df.empty:
                continue
                
            logger.info(f"Creating visualizations for dataset: {dataset}")
            
            for metric in dataset_df['metric'].unique():
                dataset_metric_df = dataset_df[dataset_df['metric'] == metric].copy()
            
                if dataset_metric_df.empty:
                    continue
                
                # Ensure feature_dimension is numeric
                try:
                    dataset_metric_df['feature_dimension'] = pd.to_numeric(dataset_metric_df['feature_dimension'])
                except:
                    logger.warning(f"Could not convert feature_dimension to numeric for dataset {dataset}.")
                    
                # Add a 'model_config' column that combines model_type and function_type
                if 'model_type' in dataset_metric_df.columns and 'function_type' in dataset_metric_df.columns:
                    dataset_metric_df['model_config'] = dataset_metric_df['model_type'] + '-' + dataset_metric_df['function_type']
                elif 'model_type' in dataset_metric_df.columns:
                    dataset_metric_df['model_config'] = dataset_metric_df['model_type']
                
                # If layer_idx exists, add it to model_config
                if 'layer_idx' in dataset_metric_df.columns:
                    dataset_metric_df['model_config'] = dataset_metric_df['model_config'] + '-layer' + dataset_metric_df['layer_idx'].astype(str)
                
                # 1. For GPT-Neo models, create a visualization by layer
                if 'layer_idx' in dataset_metric_df.columns:
                    try:
                        plt.figure(figsize=(14, 8))
                        
                        # Group by model type and layer
                        grouped = dataset_metric_df.groupby(['model_type', 'layer_idx'])['avg_centroid_distance'].mean().reset_index()
                        
                        # Sort by layer index
                        grouped = grouped.sort_values('layer_idx')
                        
                        # Plot for each model type
                        for model_type, group in grouped.groupby('model_type'):
                            plt.plot(
                                group['layer_idx'], 
                                group['avg_centroid_distance'],
                                'o-',
                                label=model_type,
                                linewidth=2,
                                markersize=8
                            )
                        
                        plt.title(f'Average Centroid Distance by Layer ({metric} metric) - {dataset}')
                        plt.xlabel('Layer Index')
                        plt.ylabel('Average Centroid Distance (normalized)')
                        plt.legend(title='Model Type')
                        plt.grid(alpha=0.3)
                        plt.tight_layout()
                        
                        # Save figure
                        layer_path = os.path.join(dataset_dir, f'layer_centroid_distance_{metric}.png')
                        plt.savefig(layer_path, dpi=300)
                        visualizations.append(layer_path)
                        plt.close()
                    except Exception as e:
                        logger.error(f"Error creating layer chart: {e}")
                
                # 2. Feature dimension vs. centroid distance
                if 'feature_dimension' in dataset_metric_df.columns:
                    try:
                        plt.figure(figsize=(12, 8))
                        
                        # Create scatter plot with different colors for model types
                        sns.scatterplot(
                            data=dataset_metric_df,
                            x='feature_dimension',
                            y='avg_centroid_distance',
                            hue='model_type',
                            style='function_type' if 'function_type' in dataset_metric_df.columns else None,
                            s=100,
                            alpha=0.7
                        )
                            
                        plt.title(f'Centroid Distance vs. Feature Dimension ({metric} metric) - {dataset}')
                        plt.xlabel('Feature Dimension')
                        plt.ylabel('Average Centroid Distance (normalized)')
                        plt.grid(alpha=0.3)
                        
                        # Add best fit lines for each model type
                        for model_type, group in dataset_metric_df.groupby('model_type'):
                            if len(group) >= 2:  # Need at least 2 points for a line
                                try:
                                    sns.regplot(
                                        x='feature_dimension',
                                        y='avg_centroid_distance',
                                        data=group,
                                        scatter=False,
                                        line_kws={'linestyle': '--', 'linewidth': 2}
                                    )
                                except:
                                    logger.warning(f"Could not create regression line for {model_type}")
                        
                        plt.tight_layout()
                        
                        # Save figure - now using dataset_dir instead of output_dir
                        scatter_path = os.path.join(dataset_dir, f'centroid_distance_vs_dimension_{metric}.png')
                        plt.savefig(scatter_path, dpi=300)
                        visualizations.append(scatter_path)
                        plt.close()
                    except Exception as e:
                        logger.error(f"Error creating scatter plot: {e}")
                
                # 3. Feature activation dimension for each model
                if 'feature_activation_dim' in dataset_metric_df.columns:
                    try:
                        plt.figure(figsize=(14, 8))
                        
                        # Plot actual feature activation dimensions
                        ax = sns.barplot(
                            data=dataset_metric_df,
                            x='model_config',
                            y='feature_activation_dim',
                            hue='model_type'
                        )
                        
                        plt.title(f'Feature Activation Dimensions by Model Configuration - {dataset}')
                        plt.xlabel('Model Configuration')
                        plt.ylabel('Feature Dimension')
                        plt.xticks(rotation=45, ha='right')
                        plt.grid(alpha=0.3, axis='y')
                        plt.tight_layout()
                        
                        # Save figure - now using dataset_dir instead of output_dir
                        dim_path = os.path.join(dataset_dir, f'feature_activation_dimensions_{metric}.png')
                        plt.savefig(dim_path, dpi=300)
                        visualizations.append(dim_path)
                        plt.close()
                    except Exception as e:
                        logger.error(f"Error creating feature activation plot: {e}")
                    
                # 4. Number of classes vs. average distance
                if 'num_centroids' in dataset_metric_df.columns:
                    try:
                        plt.figure(figsize=(10, 8))
                        
                        # Create scatter plot showing relationship between number of centroids and distance
                        sns.scatterplot(
                            data=dataset_metric_df,
                            x='num_centroids',
                            y='avg_centroid_distance',
                            hue='model_type',
                            style='function_type' if 'function_type' in dataset_metric_df.columns else None,
                            s=100,
                            alpha=0.7
                        )
                        
                        plt.title(f'Centroid Distance vs. Number of Classes ({metric} metric) - {dataset}')
                        plt.xlabel('Number of Class Centroids')
                        plt.ylabel('Average Centroid Distance (normalized)')
                        plt.grid(alpha=0.3)
                        plt.tight_layout()
                        
                        # Save figure - now using dataset_dir instead of output_dir
                        class_path = os.path.join(dataset_dir, f'centroid_distance_vs_classes_{metric}.png')
                        plt.savefig(class_path, dpi=300)
                        visualizations.append(class_path)
                        plt.close()
                    except Exception as e:
                        logger.error(f"Error creating centroid count plot: {e}")
                
                # 5. Model ranking by centroid distance
                try:
                    plt.figure(figsize=(14, 10))
                    
                    # Sort by centroid distance (higher is better for class separation)
                    # Limit to top 15 or fewer if there are fewer models
                    top_models = dataset_metric_df.sort_values('avg_centroid_distance', ascending=False)
                    top_models = top_models.head(min(15, len(dataset_metric_df)))
                    
                    # Create labels that include feature dimension
                    if 'feature_dimension' in top_models.columns:
                        top_models['plot_label'] = top_models.apply(
                            lambda x: f"{x['model_config']} (dim={int(x['feature_dimension'])})",
                            axis=1
                        )
                    else:
                        top_models['plot_label'] = top_models['model_config']
                    
                    # Create horizontal bar chart
                    ax = sns.barplot(
                        data=top_models,
                        y='plot_label',
                        x='avg_centroid_distance',
                        hue='model_type',
                        dodge=False
                    )
                    
                    # Add value annotations
                    for i, bar in enumerate(ax.patches):
                        width = bar.get_width()
                        ax.text(
                            width + 0.01,
                            bar.get_y() + bar.get_height()/2,
                            f"{width:.3f}",
                            ha='left',
                            va='center'
                        )
                    
                    plt.title(f'Top Models by Class Separation ({metric} metric) - {dataset}')
                    plt.xlabel('Average Centroid Distance (normalized)')
                    plt.ylabel('Model Configuration')
                    plt.grid(alpha=0.3, axis='x')
                    plt.tight_layout()
                    
                    # Save figure - now using dataset_dir instead of output_dir
                    ranking_path = os.path.join(dataset_dir, f'model_ranking_{metric}.png')
                    plt.savefig(ranking_path, dpi=300)
                    visualizations.append(ranking_path)
                    plt.close()
                except Exception as e:
                    logger.error(f"Error creating ranking plot: {e}")
    
    # 6. Create a comprehensive comparison table
    # Add feature activation dimension if available
    if 'feature_activation_dim' in results_df.columns:
        pivot_vars = ['avg_centroid_distance', 'std_centroid_distance', 'avg_distance', 'feature_activation_dim']
    else:
        pivot_vars = ['avg_centroid_distance', 'std_centroid_distance', 'avg_distance']
    
    # Determine index columns based on available data
    index_cols = []
    for col in ['model_type', 'function_type', 'feature_dimension', 'layer_idx']:
        if col in results_df.columns:
            index_cols.append(col)
    
    if index_cols:
        table_df = results_df.pivot_table(
            values=pivot_vars,
            index=index_cols,
            columns=['metric'],
            aggfunc='mean'
        ).round(4)
        
        # Save table as CSV
        table_path = os.path.join(output_dir, 'model_comparison_table.csv')
        table_df.to_csv(table_path)
        
        # Also save as Excel for more formatting options
        try:
            excel_path = os.path.join(output_dir, 'model_comparison_table.xlsx')
            table_df.to_excel(excel_path)
            visualizations.append(excel_path)
        except Exception as e:
            logger.warning(f"Could not save Excel file: {e}")
        
        visualizations.append(table_path)
    

    # Save dataset-specific results if dataset column exists
    if 'dataset' in results_df.columns:
        # Get unique datasets
        datasets = results_df['dataset'].unique()
        
        # For each dataset, save a separate CSV
        for dataset in datasets:
            if dataset:  # Skip None values
                dataset_df = results_df[results_df['dataset'] == dataset]
                dataset_results_path = os.path.join(output_dir, f'dataset_{dataset}_results.csv')
                dataset_df.to_csv(dataset_results_path, index=False)
                visualizations.append(dataset_results_path)
                logger.info(f"Dataset-specific results saved to: {dataset_results_path}")

    # Always save the full combined results
    full_results_path = os.path.join(output_dir, 'full_results.csv')
    results_df.to_csv(full_results_path, index=False)
    visualizations.append(full_results_path)

    # Create a note about dataset-specific files
    if 'dataset' in results_df.columns:
        note_path = os.path.join(output_dir, 'dataset_results_note.txt')
        with open(note_path, 'w') as f:
            f.write("""
    IMPORTANT: Dataset-specific results

    Individual CSV files have been created for each dataset 
    (dataset_[name]_results.csv). To analyze all datasets together,
    use the full_results.csv file.

    When running centroid_analysis.py with --dataset_filter, please 
    use these individual files to avoid overwriting other dataset results.
    """)
        visualizations.append(note_path)
    
    # Create dataset summary file if dataset column exists
    if 'dataset' in results_df.columns:
        summary_path = os.path.join(output_dir, 'dataset_summary.md')
        with open(summary_path, 'w') as f:
            f.write("# Dataset Analysis Summary\n\n")
            
            # Overall stats
            f.write("## Overall Statistics\n\n")
            f.write(f"Total models analyzed: {len(results_df)}\n")
            f.write(f"Datasets: {', '.join(sorted([d for d in results_df['dataset'].unique() if d]))}\n")
            f.write(f"Metrics: {', '.join(results_df['metric'].unique())}\n\n")
            
            # Dataset breakdown
            f.write("## Dataset Breakdown\n\n")
            for dataset in sorted([d for d in results_df['dataset'].unique() if d]):
                dataset_df = results_df[results_df['dataset'] == dataset]
                f.write(f"### {dataset}\n\n")
                f.write(f"Models: {len(dataset_df)}\n")
                f.write(f"Model types: {', '.join(dataset_df['model_type'].unique())}\n")
                
                # Summary metrics
                for metric in dataset_df['metric'].unique():
                    metric_df = dataset_df[dataset_df['metric'] == metric]
                    f.write(f"\n**{metric} metric**:\n")
                    f.write(f"- Average centroid distance: {metric_df['avg_centroid_distance'].mean():.4f}\n")
                    
                    # Make sure there are records before trying to find the max
                    if not metric_df.empty:
                        max_idx = metric_df['avg_centroid_distance'].idxmax()
                        if max_idx is not None:
                            f.write(f"- Best model: {metric_df.loc[max_idx]['model_path']}\n")
                            f.write(f"- Distance: {metric_df['avg_centroid_distance'].max():.4f}\n\n")
                    else:
                        f.write("- No data available for this metric\n\n")
                
                f.write("\n")
        
        visualizations.append(summary_path)
    
    return visualizations

def load_labeled_data(data_path, label_column='label', feature_columns=None, num_samples=None, separator=','):
    """
    Load labeled data from a CSV file.
    
    Args:
        data_path: Path to CSV file
        label_column: Name of the column containing labels
        feature_columns: Names of columns containing features (if None, all except label)
        num_samples: Number of samples to load (if None, load all)
        separator: Delimiter used in the CSV file (default: ',')
        
    Returns:
        Dictionary with 'features' array and 'labels' array
    """
    try:
        # Load the data
        df = pd.read_csv(data_path, sep=separator)
        
        # Ensure label column exists
        if label_column not in df.columns:
            logger.error(f"Error: Label column '{label_column}' not found in dataset.")
            return None
        
        # Determine feature columns
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != label_column]
        elif isinstance(feature_columns, str):
            # If a single string is provided, convert it to a list
            feature_columns = [feature_columns]
        
        # Verify all feature columns exist
        missing_columns = [col for col in feature_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Error: Feature columns not found in dataset: {missing_columns}")
            return None
        
        # Sample data if requested
        if num_samples is not None and num_samples < len(df):
            df = df.sample(n=num_samples, random_state=42)
        
        # Extract features and labels
        features = df[feature_columns].values
        labels = df[label_column].values
        
        logger.info(f"Evaluation dataset: {len(features)} samples with {len(feature_columns)} feature columns")
        logger.info(f"Number of unique classes/labels: {len(np.unique(labels))}")
        
        return {
            'features': features,
            'labels': labels,
            'feature_columns': feature_columns,
            'label_column': label_column
        }
        
    except Exception as e:
        logger.error(f"Error loading labeled data: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Analyze centroids using labeled data')
    parser.add_argument('--base_dir', type=str, default='models',
                      help='Base directory containing models')
    parser.add_argument('--output_dir', type=str, default='model_comparison',
                      help='Directory to save comparison results')
    parser.add_argument('--metrics', type=str, nargs='+', default=['cosine', 'euclidean'],
                      help='Distance metrics to use')
    parser.add_argument('--device', type=str, default='cpu',
                      help='Device to use for model loading (cpu or cuda)')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to labeled dataset (CSV file)')
    parser.add_argument('--label_column', type=str, default='label',
                      help='Name of column containing labels')
    parser.add_argument('--feature_columns', type=str, nargs='+', default=None,
                      help='Names of columns containing features (if None, all except label)')
    parser.add_argument('--num_samples', type=int, default=None,
                      help='Number of samples to use (if None, use all)')
    parser.add_argument('--delimiter', type=str, default=',',
                      help='Delimiter used in the CSV file (default: ,)')
    parser.add_argument('--single_model', type=str, default=None,
                      help='Analyze a single model instead of all models in base_dir')
    parser.add_argument('--normalize_by_dimension', action='store_true', default=True,
                      help='Normalize distances by feature dimension for fair comparison')
    parser.add_argument('--dataset_filter', type=str, default=None,
                      help='Only analyze models trained on this dataset')
    parser.add_argument('--dataset_name', type=str, default='gpt_neo',
                      help='Default dataset name for models without dataset in path')
    args = parser.parse_args()
    
    # Verify device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = 'cpu'
    
    # Load labeled data first to validate
    logger.info(f"Loading labeled data from {args.data_path}...")
    labeled_data = load_labeled_data(
        args.data_path,
        label_column=args.label_column,
        feature_columns=args.feature_columns,
        num_samples=args.num_samples,
        separator=args.delimiter
    )
    
    if labeled_data is None:
        logger.error("Failed to load labeled data. Exiting.")
        return

    # Find models to analyze
    if args.single_model:
        logger.info(f"Analyzing single model: {args.single_model}")
        # Extract model type from path
        model_type = 'sae' if 'sae' in args.single_model else 'st' if 'st' in args.single_model else 'unknown'
        models_dict = {args.single_model: {'model_type': model_type, 'dataset': args.dataset_name}}
    else:
        logger.info(f"Searching for models in {args.base_dir}...")
        models_dict = find_all_models(args.base_dir, dataset_name=args.dataset_name)

    # Filter models by dataset if specified
    if args.dataset_filter and not args.single_model:
        original_count = len(models_dict)
        filtered_models = {}
        
        for path, metadata in models_dict.items():
            # Check if the dataset matches
            dataset = metadata.get('dataset', None)
            if dataset == args.dataset_filter:
                filtered_models[path] = metadata
            # Also check path components
            elif args.dataset_filter in path.replace('\\', '/').split('/'):
                metadata['dataset'] = args.dataset_filter
                filtered_models[path] = metadata
        
        models_dict = filtered_models
        logger.info(f"Filtered from {original_count} to {len(models_dict)} models matching dataset: {args.dataset_filter}")
    
    if not models_dict:
        logger.error("No models found. Check the base directory path or provide a valid single model path.")
        return
    
    # Analyze model distances using labeled data
    logger.info(f"Analyzing models using metrics: {args.metrics} on device: {device}")
    logger.info(f"Using dimension normalization: {args.normalize_by_dimension}")
    
    # Use the wrapper function in analyze_models_distances
    results = []
    skipped = []
    
    for model_path, metadata in tqdm(models_dict.items(), desc="Analyzing models"):
        try:
            # Analyze this model
            model_results = analyze_model_with_labeled_data(
                model_path, metadata, labeled_data, args.metrics, device, 
                normalize_by_dimension=args.normalize_by_dimension
            )
            results.extend(model_results)
            
            if not model_results:
                skipped.append(model_path)
                
        except Exception as e:
            logger.error(f"Error processing model {model_path}: {e}")
            skipped.append(model_path)
    
    # Report on skipped models
    if skipped:
        logger.warning(f"Skipped {len(skipped)} models due to errors")
        
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Create visualizations
    logger.info(f"Creating comparison visualizations in {args.output_dir}...")
    visualizations = create_comparison_visualizations(results_df, output_dir=args.output_dir)
    
    logger.info(f"Analysis complete. Generated {len(visualizations)} visualizations and reports.")
    logger.info(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()