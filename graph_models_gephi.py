#!/usr/bin/env python3
"""
Gephi Graph Generator for trained SAE/ST models

This script creates Gephi graph files (.gexf) from models trained using train_for_analyze_gptneo.py.
It extracts feature activations from the trained models and visualizes them as network graphs.

Example usage:
    # Graph all SAE models in the models directory
    python graph_models_gephi.py --model_dir models --model_type sae
    
    # Graph both SAE and ST models with custom sample text
    python graph_models_gephi.py --model_dir models --model_type both --sample_text "The transformer architecture revolutionized NLP."
"""

import os
import re
import argparse
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import glob

# Import the gephi.py functions for creating GEXF files
from gephi import create_gephi_graph, sanitize_name

# Try to import the specific model implementations
try:
    from SAE import SparseAutoencoder
    HAS_SAE = True
except ImportError:
    HAS_SAE = False
    print("Warning: Could not import SAE module")

try:
    from ST import SparseTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False
    print("Warning: Could not import ST module")

# Try to import ST_old as a fallback
try:
    from ST_old import SparseTransformer as SparseTransformerOld
    HAS_ST_OLD = True
except ImportError:
    HAS_ST_OLD = False
    print("Warning: Could not import ST_old module (will try regular ST only)")

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Create Gephi graphs from trained models')
    
    # Model selection
    parser.add_argument('--base_model', type=str, default='EleutherAI/gpt-neo-1.3B',
                      help='Base model name (should match what was used for training)')
    parser.add_argument('--model_dir', type=str, default='models',
                      help='Directory containing trained models')
    parser.add_argument('--model_type', type=str, choices=['sae', 'st', 'both'], default='both',
                      help='Type of models to graph')
    
    # Input and output
    parser.add_argument('--sample_text', type=str, 
                      default="The transformer architecture revolutionized natural language processing. " 
                             "Neural networks process information through layers of interconnected nodes. "
                             "Self-attention mechanisms capture long-range dependencies in sequence data.",
                      help='Sample text to use for feature extraction')
    parser.add_argument('--text_file', type=str, default=None,
                      help='Path to a text file to use instead of sample_text')
    parser.add_argument('--output_dir', type=str, default='gephi_graphs',
                      help='Directory to save Gephi graphs')
    parser.add_argument('--no_truncate', action='store_true',
                      help='Do not truncate input text, regardless of length (may cause memory issues)')
    parser.add_argument('--max_tokens', type=int, default=2048,
                      help='Maximum number of tokens to process (ignored if --no_truncate is set)')
    
    # Grouping options
    parser.add_argument('--group_by', type=str, choices=['paragraph', 'token', 'sentence'], default='paragraph',
                      help='How to group tokens for coloring in Gephi')
    parser.add_argument('--num_groups', type=int, default=None,
                      help='Number of groups to divide tokens into (for paragraph mode)')
    
    # Graph parameters
    parser.add_argument('--n_neighbors', type=int, default=4,
                      help='Number of neighbors for k-nearest neighbors graph')
    parser.add_argument('--force_cpu', action='store_true',
                      help='Force using CPU even if CUDA is available')
    parser.add_argument('--try_old_st', action='store_true',
                      help='Try loading ST models with ST_old implementation if regular loading fails')
    parser.add_argument('--use_only_old_st', action='store_true',
                      help='Always use ST_old implementation for ST models')
    parser.add_argument('--hidden_dim', type=int, default=None,
                      help='Override hidden dimension for loaded models')
    parser.add_argument('--fake_data', action='store_true',
                      help='Use fake/random data instead of actual model hidden states')
    parser.add_argument('--include_raw_activations', action='store_true',
                      help='Also create graphs from raw layer activations without decomposition')
    
    return parser.parse_args()

def get_model_paths(model_dir, model_type):
    """Get paths to trained models"""
    if model_type == 'both':
        sae_paths = get_model_paths(model_dir, 'sae') if HAS_SAE else {}
        st_paths = get_model_paths(model_dir, 'st') if HAS_ST else {}
        return {**sae_paths, **st_paths}  # Combine dictionaries
    
    model_paths = {}
    pattern = os.path.join(model_dir, model_type, f"layer_*_{model_type}.pt")
    for path in glob.glob(pattern):
        # Extract layer number from filename
        filename = os.path.basename(path)
        match = re.search(r'layer_(\d+)_', filename)
        if match:
            layer_num = int(match.group(1))
            model_paths[layer_num] = (path, model_type)
    
    return model_paths

def extract_dimensions_from_checkpoint(checkpoint_path):
    """Extract input and feature dimensions from a model checkpoint"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        hidden_dim = None
        feature_dim = None
        
        if isinstance(checkpoint, dict):
            # Try to extract dimensions directly from metadata
            if 'n' in checkpoint and 'm' in checkpoint:
                hidden_dim = checkpoint['n']
                feature_dim = checkpoint['m']
                return hidden_dim, feature_dim
            
            # Try to extract from model_state_dict keys
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                # Look for weight parameters that might contain dimensions
                if 'W_e.weight' in state_dict:
                    weight = state_dict['W_e.weight']
                    hidden_dim = weight.shape[1]  # n for SAE
                    feature_dim = weight.shape[0]  # m for SAE
                    return hidden_dim, feature_dim
                elif 'W_d.weight' in state_dict:
                    weight = state_dict['W_d.weight']
                    hidden_dim = weight.shape[0]  # n for SAE
                    feature_dim = weight.shape[1]  # m for SAE
                    return hidden_dim, feature_dim
                elif 'W_q.weight' in state_dict:
                    weight = state_dict['W_q.weight']
                    hidden_dim = weight.shape[1]  # n for ST
                    if 'W_k_direct' in state_dict:
                        feature_dim = state_dict['W_k_direct'].shape[0]  # m for ST with direct KV
                    return hidden_dim, feature_dim
                elif 'W_k_direct' in state_dict:
                    # For direct KV models
                    hidden_dim_from_v = None
                    if 'W_v_direct' in state_dict:
                        hidden_dim_from_v = state_dict['W_v_direct'].shape[1]
                    feature_dim = state_dict['W_k_direct'].shape[0]
                    attn_dim = state_dict['W_k_direct'].shape[1]
                    return hidden_dim_from_v, feature_dim
        
        # If we can't extract dimensions, return None
        return hidden_dim, feature_dim
    except Exception as e:
        print(f"Error extracting dimensions from {checkpoint_path}: {e}")
        return None, None

def load_sae_model(model_path, input_dim, feature_dim, device):
    """Load a trained SAE model"""
    if not HAS_SAE:
        raise ImportError("SAE module not available")
    
    # Initialize the SAE model with correct dimensions
    model = SparseAutoencoder(
        n=input_dim,
        m=feature_dim,
        sae_model_path=model_path,
        device=device
    )
    
    # Load the trained weights
    success = model.resume_from_checkpoint(model_path)
    if not success:
        raise ValueError(f"Failed to load SAE model from {model_path}")
    
    return model

def load_st_model(model_path, input_dim, feature_dim, device, try_old_st=False, use_only_old_st=False):
    """Load a trained ST model, falling back to ST_old if needed"""
    # Check if we should use ST_old directly
    if use_only_old_st:
        if not HAS_ST_OLD:
            raise ImportError("ST_old module requested but not available")
        return load_st_old_model(model_path, input_dim, feature_dim, device)
    
    # Try regular ST first, unless explicitly using old version
    if not HAS_ST:
        if HAS_ST_OLD and try_old_st:
            print("Regular ST module not available, trying ST_old")
            return load_st_old_model(model_path, input_dim, feature_dim, device)
        else:
            raise ImportError("ST module not available")
    
    # For ST, we need a dummy tensor to initialize
    dummy_X = torch.zeros((1, input_dim), device=device)
    attention_dim = max(20, input_dim // 8)  # Approximate attention dimension
    
    # Calculate feature dimension if not provided
    if not feature_dim:
        feature_dim = input_dim // 4  # Default to n/4
    
    # Initialize the ST model
    model = SparseTransformer(
        X=dummy_X,
        n=input_dim,
        m=feature_dim,
        a=attention_dim,
        st_model_path=model_path,
        device=device
    )
    
    # Load the trained weights
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        return model
    except Exception as e:
        if try_old_st and HAS_ST_OLD:
            print(f"Failed to load with regular ST: {e}")
            print("Trying with ST_old implementation...")
            return load_st_old_model(model_path, input_dim, feature_dim, device)
        else:
            raise ValueError(f"Failed to load ST model from {model_path}: {e}")

def load_st_old_model(model_path, input_dim, feature_dim, device):
    """Load a trained ST model using the old ST implementation"""
    if not HAS_ST_OLD:
        raise ImportError("ST_old module not available")
    
    # For ST_old, we need a dummy tensor to initialize
    dummy_X = torch.zeros((1, input_dim), device=device)
    
    # Extract dimensions from the saved model
    checkpoint = torch.load(model_path, map_location=device)
    
    # Determine if model was trained with direct KV approach
    # This is indicated by presence of W_k_direct and W_v_direct keys
    use_direct_kv = False
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        if 'W_k_direct' in state_dict and 'W_v_direct' in state_dict:
            use_direct_kv = True
            # Get feature dimension from W_k_direct
            feature_dim_from_checkpoint = state_dict['W_k_direct'].shape[0]
        elif 'memory_indices' in state_dict:
            feature_dim_from_checkpoint = state_dict['memory_indices'].shape[0]
        else:
            feature_dim_from_checkpoint = None
    else:
        if 'W_k_direct' in checkpoint and 'W_v_direct' in checkpoint:
            use_direct_kv = True
            # Get feature dimension from W_k_direct
            feature_dim_from_checkpoint = checkpoint['W_k_direct'].shape[0]
        elif 'memory_indices' in checkpoint:
            feature_dim_from_checkpoint = checkpoint['memory_indices'].shape[0]
        else:
            feature_dim_from_checkpoint = None
    
    print(f"Detected use_direct_kv={use_direct_kv} in ST model")
    
    # Override feature_dim with the one from checkpoint if available
    if feature_dim_from_checkpoint is not None:
        if feature_dim is not None and feature_dim != feature_dim_from_checkpoint:
            print(f"Warning: Overriding provided feature dimension {feature_dim} with {feature_dim_from_checkpoint} from checkpoint")
        feature_dim = feature_dim_from_checkpoint
        print(f"Using feature dimension {feature_dim} from checkpoint")
    
    # Try to extract attention dimension from the model weights
    attention_dim = None
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        if 'W_q.bias' in state_dict:
            attention_dim = state_dict['W_q.bias'].shape[0]
        elif 'W_q.weight' in state_dict:
            attention_dim = state_dict['W_q.weight'].shape[0]
        elif use_direct_kv and 'W_k_direct' in state_dict:
            attention_dim = state_dict['W_k_direct'].shape[1]
    else:
        # Check in the direct checkpoint
        if 'W_q.bias' in checkpoint:
            attention_dim = checkpoint['W_q.bias'].shape[0]
        elif 'W_q.weight' in checkpoint:
            attention_dim = checkpoint['W_q.weight'].shape[0]
        elif use_direct_kv and 'W_k_direct' in checkpoint:
            attention_dim = checkpoint['W_k_direct'].shape[1]
    
    # Default if couldn't determine
    if attention_dim is None:
        attention_dim = max(20, input_dim // 8)
    
    print(f"ST_old model dimensions - n: {input_dim}, m: {feature_dim}, a: {attention_dim}, use_direct_kv: {use_direct_kv}")
    
    # Initialize the ST_old model with the correct parameters
    model = SparseTransformerOld(
        X=dummy_X,
        n=input_dim,
        m=feature_dim,
        a=attention_dim,
        st_model_path=model_path,
        device=device,
        use_direct_kv=use_direct_kv  # This is the key parameter we need to set correctly
    )
    
    # Load the trained weights
    try:
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        raise ValueError(f"Failed to load ST_old model from {model_path}: {e}")
    
    return model

def main():
    """Main function for creating Gephi graphs"""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cpu" if args.force_cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    # Get paths to trained models
    model_paths = get_model_paths(args.model_dir, args.model_type)
    
    if not model_paths and not args.include_raw_activations:
        print(f"No {args.model_type} models found in {args.model_dir}")
        return
    
    if model_paths:
        print(f"Found {len(model_paths)} models")
    
    # Get sample text - either from file or from argument
    sample_text = args.sample_text
    if args.text_file:
        try:
            with open(args.text_file, 'r', encoding='utf-8') as f:
                sample_text = f.read()
            print(f"Loaded text from {args.text_file} ({len(sample_text)} characters)")
        except Exception as e:
            print(f"Error loading text file: {e}")
            print("Using default sample text instead")
    
    # Initialize tokenizer
    print(f"Initializing tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Error loading tokenizer from {args.base_model}: {e}")
        print("Using fallback tokenizer (gpt2)")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize the text
    if args.no_truncate:
        print(f"Tokenizing full text without truncation...")
        inputs = tokenizer(sample_text, return_tensors="pt")
    else:
        print(f"Tokenizing text with max length of {args.max_tokens} tokens...")
        inputs = tokenizer(sample_text, return_tensors="pt", truncation=True, max_length=args.max_tokens)
    
    input_tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids.squeeze().tolist())
    print(f"Tokenized text: {len(input_tokens)} tokens")
    
    # Create token ID to paragraph mapping - track which paragraph/text each token belongs to
    # We'll use this for coloring in Gephi
    print("Creating token to paragraph mapping...")
    token_to_paragraph_map = []
    
    if args.text_file:
        # If using a text file, try to split by paragraphs (double newlines)
        paragraphs = sample_text.split('\n\n')
        if len(paragraphs) <= 1:  # If no double newlines, try single newlines
            paragraphs = sample_text.split('\n')
        if len(paragraphs) <= 1:  # If still just one paragraph, split by sentences
            from nltk.tokenize import sent_tokenize
            try:
                import nltk
                nltk.download('punkt', quiet=True)
                paragraphs = sent_tokenize(sample_text)
            except:
                # Fallback to simple period splitting
                paragraphs = sample_text.split('. ')
        
        print(f"Found {len(paragraphs)} paragraphs/sections in the text")
        
        if args.num_groups is not None and args.num_groups > 0:
            # Override number of groups if specified
            if args.num_groups < len(paragraphs):
                # Combine paragraphs to reduce groups
                combined_paragraphs = []
                paragraphs_per_group = len(paragraphs) // args.num_groups
                for i in range(0, len(paragraphs), paragraphs_per_group):
                    end_idx = min(i + paragraphs_per_group, len(paragraphs))
                    combined_paragraphs.append(' '.join(paragraphs[i:end_idx]))
                paragraphs = combined_paragraphs
                print(f"Combined paragraphs into {len(paragraphs)} groups")
            elif args.num_groups > len(paragraphs):
                # Split paragraphs further
                print(f"Requested {args.num_groups} groups but only found {len(paragraphs)} paragraphs")
                print("Will create approximate groups")
                # Handle through the default group creation below
                token_len = len(input_tokens)
                n_sections = args.num_groups
                section_sizes = [token_len // n_sections] * n_sections
                for i in range(token_len % n_sections):
                    section_sizes[i] += 1
                
                section = 0
                tokens_in_section = 0
                for i in range(token_len):
                    token_to_paragraph_map.append(section)
                    tokens_in_section += 1
                    if tokens_in_section >= section_sizes[section] and section < n_sections - 1:
                        section += 1
                        tokens_in_section = 0
        
        # If we didn't create token mapping through num_groups, create it now
        if len(token_to_paragraph_map) == 0:
            # For each token, determine which paragraph it belongs to
            current_paragraph = 0
            current_paragraph_tokens = tokenizer.encode(paragraphs[0], add_special_tokens=False)
            
            for i, token_id in enumerate(inputs.input_ids.squeeze().tolist()):
                # Skip special tokens at beginning
                if i == 0 and token_id in [tokenizer.cls_token_id, tokenizer.bos_token_id]:
                    token_to_paragraph_map.append(0)
                    continue
                
                # Check if we need to move to the next paragraph
                if current_paragraph < len(paragraphs) - 1:
                    if len(current_paragraph_tokens) == 0:
                        current_paragraph += 1
                        if current_paragraph < len(paragraphs):
                            current_paragraph_tokens = tokenizer.encode(paragraphs[current_paragraph], add_special_tokens=False)
                        
                    elif token_id in current_paragraph_tokens:
                        # Remove the first occurrence of this token from the current paragraph tokens
                        idx = current_paragraph_tokens.index(token_id)
                        current_paragraph_tokens.pop(idx)
                
                # Assign current paragraph ID to this token
                token_to_paragraph_map.append(current_paragraph)
        
    else:
        # For sample text without paragraphs, create arbitrary divisions
        token_len = len(input_tokens)
        n_sections = args.num_groups if args.num_groups is not None else min(10, max(3, token_len // 50))  # Approximately 50 tokens per section
        section_sizes = [token_len // n_sections] * n_sections
        for i in range(token_len % n_sections):
            section_sizes[i] += 1
        
        section = 0
        tokens_in_section = 0
        for i in range(token_len):
            token_to_paragraph_map.append(section)
            tokens_in_section += 1
            if tokens_in_section >= section_sizes[section] and section < n_sections - 1:
                section += 1
                tokens_in_section = 0
    
    # Sanity check
    if len(token_to_paragraph_map) != len(input_tokens):
        print(f"Warning: Token to paragraph mapping length ({len(token_to_paragraph_map)}) doesn't match input tokens ({len(input_tokens)})")
        # Pad or truncate if necessary
        if len(token_to_paragraph_map) < len(input_tokens):
            token_to_paragraph_map.extend([token_to_paragraph_map[-1]] * (len(input_tokens) - len(token_to_paragraph_map)))
        else:
            token_to_paragraph_map = token_to_paragraph_map[:len(input_tokens)]
    
    # Count tokens per paragraph
    paragraph_counts = {}
    for p in token_to_paragraph_map:
        paragraph_counts[p] = paragraph_counts.get(p, 0) + 1
    
    print("Tokens per paragraph/section:")
    for p, count in sorted(paragraph_counts.items()):
        print(f"  Section {p}: {count} tokens")
    
    # Try to load the first model to determine hidden dimension
    print("Checking model dimensions from saved checkpoints...")
    model_checkpoint_path = next(iter(model_paths.values()))[0]
    hidden_dim_from_checkpoint, _ = extract_dimensions_from_checkpoint(model_checkpoint_path)
    
    # Use command line override if provided
    hidden_dim = args.hidden_dim if args.hidden_dim is not None else hidden_dim_from_checkpoint
    
    if hidden_dim is not None:
        print(f"Using hidden dimension {hidden_dim} for models")
    else:
        print("Could not determine hidden dimension from models, will use default from base model")
    
    # Decide whether to load base model or use fake data
    if args.fake_data:
        print("Using random data instead of real model hidden states")
        base_model_loaded = False
        # Create fake hidden states with the detected dimension or default 2048
        hidden_dim = hidden_dim if hidden_dim is not None else 2048
        print(f"Generating fake hidden states with dimension {hidden_dim}")
        all_hidden_states = [
            torch.randn((1, len(input_tokens), hidden_dim), device=device) 
            for _ in range(20)
        ]
    else:
        # Try to load base model if not using fake data
        print(f"Loading base model: {args.base_model}")
        try:
            config = AutoConfig.from_pretrained(args.base_model, output_hidden_states=True)
            model = AutoModelForCausalLM.from_pretrained(args.base_model, config=config)
            model.to(device)
            model.eval()
            base_model_loaded = True
            
            # Get model's hidden dimension
            model_hidden_dim = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 768
            print(f"Base model has hidden dimension {model_hidden_dim}")
            
            # Check if base model's hidden dimension matches what we need
            if hidden_dim is not None and hidden_dim != model_hidden_dim:
                print(f"Warning: Base model hidden dimension ({model_hidden_dim}) doesn't match required dimension ({hidden_dim})")
                print("This may cause issues. Consider using --fake_data option instead.")
            
            # Move inputs to device if using real model
            try:
                inputs = inputs.to(device)
            except RuntimeError as e:
                print(f"Error moving inputs to device: {e}")
                print("Input may be too large. Trying to process on CPU instead...")
                device = torch.device("cpu")
                model = model.to(device)  # Move model to CPU
                inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get hidden states
            try:
                with torch.no_grad():
                    # Process in smaller chunks if the input is very large
                    if len(input_tokens) > 1024 and device.type == 'cuda':
                        print("Large input detected, processing in chunks to avoid memory issues...")
                        # Create a dummy result structure to collect outputs
                        dummy_output = model(inputs.input_ids[:, :1])
                        all_hidden_states = [
                            torch.zeros((1, inputs.input_ids.size(1), hs.size(-1)), device=device)
                            for hs in dummy_output.hidden_states
                        ]
                        
                        # Process in chunks of 512 tokens
                        chunk_size = 512
                        for i in range(0, inputs.input_ids.size(1), chunk_size):
                            end_idx = min(i + chunk_size, inputs.input_ids.size(1))
                            print(f"Processing tokens {i}-{end_idx}...")
                            
                            chunk_inputs = {
                                'input_ids': inputs.input_ids[:, i:end_idx],
                                'attention_mask': inputs.attention_mask[:, i:end_idx] if hasattr(inputs, 'attention_mask') else None
                            }
                            # Remove None values
                            chunk_inputs = {k: v for k, v in chunk_inputs.items() if v is not None}
                            
                            chunk_outputs = model(**chunk_inputs)
                            
                            # Store the hidden states
                            for j, hs in enumerate(chunk_outputs.hidden_states):
                                all_hidden_states[j][:, i:end_idx] = hs
                    else:
                        # Process the entire input at once
                        outputs = model(**inputs)
                        all_hidden_states = outputs.hidden_states
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    print("CUDA out of memory error. Switching to fake data mode...")
                    # Switch to fake data
                    base_model_loaded = False
                    hidden_dim = hidden_dim if hidden_dim is not None else 2048
                    print(f"Generating fake hidden states with dimension {hidden_dim}")
                    all_hidden_states = [
                        torch.randn((1, len(input_tokens), hidden_dim), device=device) 
                        for _ in range(20)
                    ]
                else:
                    print(f"Error during model inference: {e}")
                    print("Creating dummy hidden states and continuing...")
                    # Create dummy hidden states
                    hidden_dim = hidden_dim if hidden_dim is not None else 2048
                    all_hidden_states = [
                        torch.randn((1, len(input_tokens), hidden_dim), device=device)
                        for _ in range(20)
                    ]
            
        except Exception as e:
            print(f"Error loading base model: {e}")
            print("Using fake data instead")
            base_model_loaded = False
            # Create fake hidden states with the detected dimension or default 2048
            hidden_dim = hidden_dim if hidden_dim is not None else 2048
            print(f"Generating fake hidden states with dimension {hidden_dim}")
            all_hidden_states = [
                torch.randn((1, len(input_tokens), hidden_dim), device=device) 
                for _ in range(20)
            ]
    
    # Process each model
    layers_processed = set()
    for layer_num, (model_path, model_type) in model_paths.items():
        print(f"\nProcessing layer {layer_num}, model: {os.path.basename(model_path)}")
        layers_processed.add(layer_num)
        
        # Get the hidden states for this layer
        layer_idx = layer_num + 1  # +1 to skip embedding layer
        if layer_idx >= len(all_hidden_states):
            print(f"Layer index {layer_idx} out of range. Model has {len(all_hidden_states)} layers (including embeddings)")
            continue
            
        hidden_states = all_hidden_states[layer_idx].squeeze()
        if hidden_states.device != device:
            hidden_states = hidden_states.to(device)
        
        # Extract dimensions from checkpoint
        checkpoint_hidden_dim, checkpoint_feature_dim = extract_dimensions_from_checkpoint(model_path)
        print(f"Detected dimensions from checkpoint - input_dim: {checkpoint_hidden_dim}, feature_dim: {checkpoint_feature_dim}")
        
        # Determine the input dimension to use (override, checkpoint, or from tensor)
        input_dim = args.hidden_dim if args.hidden_dim is not None else checkpoint_hidden_dim
        if input_dim is None:
            input_dim = hidden_states.shape[-1]  # Use hidden size from tensor
            print(f"Using input dimension {input_dim} from hidden states")
        else:
            print(f"Using input dimension {input_dim} from checkpoint/override")
        
        # Determine feature dimension
        feature_dim = checkpoint_feature_dim
        if feature_dim is None and model_type == 'sae':
            feature_dim = input_dim // 4  # Default for SAE
            print(f"Could not extract feature dimension, using {feature_dim} (input_dim/4)")
        elif feature_dim is None and model_type == 'st':
            feature_dim = input_dim // 4  # Default for ST
            print(f"Could not extract feature dimension, using {feature_dim} (input_dim/4)")
        else:
            print(f"Using feature dimension {feature_dim} from checkpoint")
        
        # Adapt hidden states if necessary
        actual_hidden_dim = hidden_states.shape[-1]
        if actual_hidden_dim != input_dim:
            print(f"Warning: Hidden states dimension ({actual_hidden_dim}) doesn't match required dimension ({input_dim})")
            print(f"Adapting hidden states to match model input dimension...")
            
            if actual_hidden_dim < input_dim:
                # Pad the hidden states
                padding = torch.zeros(hidden_states.shape[0], input_dim - actual_hidden_dim, device=device)
                print(f"Adding {padding.shape[1]} zero padding dimensions")
                hidden_states = torch.cat([hidden_states, padding], dim=1)
            else:
                # Truncate the hidden states
                print(f"Truncating hidden states from {actual_hidden_dim} to {input_dim} dimensions")
                hidden_states = hidden_states[:, :input_dim]
                
            print(f"Adapted hidden states shape: {hidden_states.shape}")
        
        # Load the trained model
        try:
            if model_type == 'sae':
                decomp_model = load_sae_model(model_path, input_dim, feature_dim, device)
            else:  # 'st'
                decomp_model = load_st_model(
                    model_path, 
                    input_dim, 
                    feature_dim, 
                    device,
                    try_old_st=args.try_old_st,
                    use_only_old_st=args.use_only_old_st
                )
            
            # Get feature activations for the hidden states
            with torch.no_grad():
                feature_activations = decomp_model.feature_activations(hidden_states)
                if isinstance(feature_activations, torch.Tensor):
                    feature_activations = feature_activations.cpu().numpy()
            
            print(f"Feature activations shape: {feature_activations.shape}")
            
            # Create DataFrame with token information
            df = pd.DataFrame()
            df['token'] = input_tokens
            df['token_id'] = inputs.input_ids.squeeze().tolist()
            df['position'] = range(len(input_tokens))
            
            # Add paragraph/section information for coloring
            df['paragraph'] = token_to_paragraph_map
            
            # Keep individual token information too
            df['token_group'] = df['token']
            
            # Determine which column to use for categorization in Gephi
            if args.group_by == 'token':
                category_column = 'token_group'
                print("Using individual tokens for coloring in Gephi")
            elif args.group_by == 'paragraph':
                category_column = 'paragraph'
                print("Using paragraphs/sections for coloring in Gephi")
            elif args.group_by == 'sentence':
                # If user requested sentence grouping but we only have paragraph info, we can approximate
                # by creating more fine-grained groups
                if 'sentence' not in df.columns:
                    # Create sentence groups (approximately 20 tokens per sentence)
                    df['sentence'] = df['position'] // 20
                    print("Creating approximate sentence grouping (20 tokens per sentence)")
                category_column = 'sentence'
                print("Using sentences for coloring in Gephi")
            else:
                # Default to paragraph
                category_column = 'paragraph'
                print("Using default paragraph grouping for coloring in Gephi")
            
            # Create a sanitized model name for the file
            model_name = sanitize_name(f"{os.path.basename(args.base_model)}_layer{layer_num}_{model_type}")
            
            # Create Gephi graph
            output_path = os.path.join(args.output_dir, f"layer_{layer_num}_{model_type}.gexf")
            create_gephi_graph(
                feature_extract=feature_activations,
                df=df,
                title_column='token',  # Use token as the title
                model_name=model_name,
                file_path=output_path,
                category_column=category_column,  # Use selected grouping for coloring
                n_neighbors=args.n_neighbors
            )
            
            print(f"Created Gephi graph: {output_path}")
            
        except Exception as e:
            print(f"Error processing model {model_path}: {e}")
            continue
    
    # Process raw activations if requested
    if args.include_raw_activations:
        print("\nProcessing raw layer activations...")
        
        # Determine which layers to process
        if not model_paths:
            # If no models were found, process all layers
            layers_to_process = list(range(len(all_hidden_states)-1))  # Skip last layer (output)
        else:
            # Process the same layers as the models
            layers_to_process = list(layers_processed)
        
        for layer_num in layers_to_process:
            layer_idx = layer_num + 1  # +1 to skip embedding layer
            if layer_idx >= len(all_hidden_states):
                print(f"Layer index {layer_idx} out of range. Model has {len(all_hidden_states)} layers (including embeddings)")
                continue
                
            print(f"\nProcessing raw activations for layer {layer_num}")
            
            # Get the hidden states for this layer
            hidden_states = all_hidden_states[layer_idx].squeeze().cpu().numpy()
            
            # Create DataFrame with token information
            df = pd.DataFrame()
            df['token'] = input_tokens
            df['token_id'] = inputs.input_ids.squeeze().tolist()
            df['position'] = range(len(input_tokens))
            
            # Add paragraph/section information for coloring
            df['paragraph'] = token_to_paragraph_map
            
            # Create a sanitized model name for the file
            model_name = sanitize_name(f"{os.path.basename(args.base_model)}_layer{layer_num}_raw")
            
            # Determine which column to use for categorization in Gephi (same as above)
            if args.group_by == 'token':
                category_column = 'token_group'
            elif args.group_by == 'paragraph':
                category_column = 'paragraph'
            elif args.group_by == 'sentence':
                if 'sentence' not in df.columns:
                    df['sentence'] = df['position'] // 20
                category_column = 'sentence'
            else:
                category_column = 'paragraph'
            
            # Create Gephi graph for raw activations
            output_path = os.path.join(args.output_dir, f"layer_{layer_num}_raw.gexf")
            create_gephi_graph(
                feature_extract=hidden_states,
                df=df,
                title_column='token',
                model_name=model_name,
                file_path=output_path,
                category_column=category_column,
                n_neighbors=args.n_neighbors
            )
            
            print(f"Created raw activation graph: {output_path}")
            
    # Print summary and instructions
    if os.path.exists(args.output_dir) and len(os.listdir(args.output_dir)) > 0:
        print("\nSuccess! Gephi graph files created:")
        for file in os.listdir(args.output_dir):
            if file.endswith('.gexf'):
                print(f"  - {os.path.join(args.output_dir, file)}")
        
        print("\nTo visualize these graphs in Gephi:")
        print("1. Open Gephi (download from https://gephi.org/ if not installed)")
        print("2. Go to File > Open and select the .gexf files")
        print("3. In the Overview tab, choose a layout algorithm (e.g., ForceAtlas2)")
        print("4. Run the layout and adjust parameters as needed")
        print("5. Use the Appearance panel to color nodes by the 'group' attribute")
    else:
        print("\nNo graph files were created. Check the error messages above.")

if __name__ == "__main__":
    main()