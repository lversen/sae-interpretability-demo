#!/usr/bin/env python3
"""
Script to train models and save them in a customized hierarchical folder structure:
- SAE: models/sae/activation_function/feature_dimension/
- ST: models/st/attention_function/feature_dimension/

Supports multiple runs of the same configuration with automatic naming:
- First run: model.pth
- Second run: model_2.pth 
- Third run: model_3.pth
- etc.
"""
import concurrent.futures
from functools import partial
import re
import os
import argparse
import itertools
import subprocess
import json
import time
import glob
import shutil
from datetime import datetime
import torch

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train models with hierarchical organization')
    
    # Model configuration parameters
    parser.add_argument('--datasets', type=str, nargs='+', default=['mnist'],
                      help='List of datasets to use')
    parser.add_argument('--model_types', type=str, nargs='+', default=['st'],
                      help='List of model types to train (sae, st, both)')
    parser.add_argument('--attention_fns', type=str, nargs='+', default=['softmax'],
                      help='List of attention functions for ST models')
    parser.add_argument('--feature_dimensions', type=int, nargs='+', default=[100],
                      help='List of feature dimensions to try')
    parser.add_argument('--attention_dimensions', type=int, nargs='+', default=[None],
                      help='List of attention dimensions to try (None for auto)')
    parser.add_argument('--activations', type=str, nargs='+', default=['relu'],
                      help='List of activation functions for SAE models')
    parser.add_argument('--learning_rates', type=float, nargs='+', default=[5e-5],
                      help='List of learning rates to try')
    parser.add_argument('--l1_lambdas', type=float, nargs='+', default=[5.0],
                      help='List of L1 regularization strengths to try')
    
    # Training parameters
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[None],
                      help='List of batch sizes to try')
    parser.add_argument('--target_steps', type=int, nargs='+', default=[None],
                      help='List of training steps to try')
    parser.add_argument('--grad_accum_steps', type=int, nargs='+', default=[1],
                      help='List of gradient accumulation steps to try')
    parser.add_argument('--eval_freqs', type=int, nargs='+', default=[None],
                      help='List of evaluation frequencies to try')
    parser.add_argument('--auto_steps', action='store_true',
                      help='Use auto step calculation')
    parser.add_argument('--auto_steps_bases', type=int, nargs='+', default=[200000],
                      help='List of base steps for auto step calculation')
    
    # Architecture options
    parser.add_argument('--use_memory_bank', action='store_true',
                      help='Use memory bank approach for ST models')
    parser.add_argument('--use_old_st', action='store_true',
                      help='Use old ST implementation')
    parser.add_argument('--use_mixed_precision', action='store_true',
                    help='Enable mixed precision training for ST model')
    
    # Multiple runs and retraining options
    parser.add_argument('--num_runs', type=int, default=1,
                      help='Number of runs for each model configuration')
    parser.add_argument('--force_retrain', action='store_true',
                      help='Force retraining even if models with the same configuration exist')
    parser.add_argument('--continue_numbering', action='store_true',
                      help='Continue run numbering from highest existing run')
    
    # Other options
    parser.add_argument('--skip_existing', action='store_true',
                      help='Skip training if model already exists')
    parser.add_argument('--summary_file', type=str, default='training_summary.md',
                      help='Path to write training summary')
    parser.add_argument('--main_args', type=str, default='',
                      help='Additional arguments to pass to main.py')
    parser.add_argument('--temp_dir', type=str, default='temp_models',
                      help='Temporary directory for model files before moving')
    parser.add_argument('--device', type=str, default='cuda',
                    help='Device to use for training (cuda or cpu)')
    parser.add_argument('--workers', type=int, default=4,
                      help='Number of parallel training jobs to run')
    return parser.parse_args()

def get_steps_version_of_path(path):
    """
    Check if a file exists with 'steps' instead of 'autosteps' in the path
    
    Args:
        path: Original path with autosteps
        
    Returns:
        Updated path with actual steps if found, original path otherwise
    """
    if path is None or 'autosteps' not in path:
        return path
        
    dir_name = os.path.dirname(path)
    file_name = os.path.basename(path)
    
    # Extract the base number from autosteps
    parts = file_name.split('_')
    for i, part in enumerate(parts):
        if part.startswith('autosteps'):
            # Create a pattern replacing just this part with steps*
            auto_base = part[9:]  # Extract base number after 'autosteps'
            parts[i] = f"steps*"
            pattern = '_'.join(parts)
            
            # Search for matching files
            matches = glob.glob(os.path.join(dir_name, pattern))
            if matches:
                # Sort matches to get the most relevant one
                matches.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                return matches[0]
    
    return path

def model_exists_with_autosteps(base_path):
    """
    Check if a model exists, accounting for auto_steps renaming.
    
    Args:
        base_path: Original path to check
        
    Returns:
        Tuple (exists, actual_path) - Boolean indicating if model exists,
        and the actual path if found (which might be different from base_path)
    """
    if base_path is None:
        return False, None
        
    # Check if the exact path exists
    if os.path.exists(base_path):
        return True, base_path
        
    # If we're looking for an autosteps file, also check for steps files
    if 'autosteps' in base_path:
        # Use the helper function to get the steps version
        steps_path = get_steps_version_of_path(base_path)
        if steps_path != base_path and os.path.exists(steps_path):
            return True, steps_path
    
    return False, base_path

def generate_combinations(args):
    """Generate all combinations of parameters to try"""
    combinations = []
    
    # Handle None values
    attention_dims = args.attention_dimensions if args.attention_dimensions != [None] else [None]
    batch_sizes = args.batch_sizes if args.batch_sizes != [None] else [None]
    target_steps = args.target_steps if args.target_steps != [None] else [None]
    eval_freqs = args.eval_freqs if args.eval_freqs != [None] else [None]
    
    # Create product of all specified parameters
    param_grid = itertools.product(
        args.datasets,
        args.model_types,
        args.attention_fns,
        args.feature_dimensions,
        attention_dims,
        args.learning_rates,
        args.l1_lambdas,
        args.activations,
        batch_sizes,
        target_steps,
        args.grad_accum_steps,
        eval_freqs,
        args.auto_steps_bases
    )
    
    for (dataset, model_type, attention_fn, feature_dim, attention_dim, lr, l1_lambda, 
         activation, batch_size, target_steps, grad_accum_steps, eval_freq, 
         auto_steps_base) in param_grid:
         
        # For SAE models, always use ReLU activation and ignore attention functions
        if model_type == 'sae':
            # Set the activation function to ReLU for SAE models
            activation = 'relu'
            
        # For ST models, make sure attention function is valid
        # No need to filter SAE models based on attention function anymore
        
        combo = {
            'dataset': dataset,
            'model_type': model_type,
            'attention_fn': attention_fn,  # Only used for ST models
            'feature_dimension': feature_dim,
            'attention_dimension': attention_dim,  # Only used for ST models
            'learning_rate': lr,
            'l1_lambda': l1_lambda,
            'activation': activation,  # Always 'relu' for SAE
            'batch_size': batch_size,
            'target_steps': target_steps,
            'grad_accum_steps': grad_accum_steps,
            'eval_freq': eval_freq,
            'auto_steps_base': auto_steps_base
        }
        combinations.append(combo)
    
    return combinations

def get_model_id(combination, args):
    """Create a unique model_id based on the combination"""
    base_id = f"{combination['dataset']}_{combination['model_type']}_{combination['feature_dimension']}"
    
    # Add attention function for ST models
    if combination['model_type'] in ['st', 'both']:
        base_id += f"_{combination['attention_fn']}"
        if combination['attention_dimension']:
            base_id += f"_a{combination['attention_dimension']}"
    # Add activation function for SAE models
    elif combination['model_type'] == 'sae':
        if combination['activation'] != 'relu':
            base_id += f"_{combination['activation']}"
    
    # Add learning rate if not default
    if combination['learning_rate'] != 5e-5:
        base_id += f"_lr{combination['learning_rate']}"
    
    # Add L1 lambda if not default
    if combination['l1_lambda'] != 5.0:
        base_id += f"_l1{combination['l1_lambda']}"
    
    # Add batch size if specified
    if combination['batch_size'] is not None:
        base_id += f"_bs{combination['batch_size']}"
    
    # Add target steps if specified
    if combination['target_steps'] is not None:
        base_id += f"_steps{combination['target_steps']}"
    
    # Add gradient accumulation steps if not 1
    if combination['grad_accum_steps'] != 1:
        base_id += f"_accum{combination['grad_accum_steps']}"
    
    # Add eval frequency if specified
    if combination['eval_freq'] is not None:
        base_id += f"_eval{combination['eval_freq']}"
    
    # Add auto_steps_base if not default and auto_steps is enabled
    if args.auto_steps and combination['auto_steps_base'] != 200000:
        base_id += f"_autobase{combination['auto_steps_base']}"
    
    if args.use_memory_bank:
        base_id += "_memory"
    
    if args.use_old_st and combination['model_type'] in ['st', 'both']:
        base_id += "_oldst"
    
    # Clean up the ID to make it a valid filename
    model_id = base_id.replace('.', 'p')  # Replace dots with 'p' for decimal points
    
    return model_id

def get_simplified_filename(model_type_prefix, combination, args):
    """Get a simplified filename focusing only on training parameters"""
    # The directory structure already includes:
    # models/[dataset]/[model_type]/[activation_or_attention]/[feature_dimension]/
    
    # So we only need to include training parameters in the filename:
    # bs{batch}_lr{learning_rate}_steps{total_steps}_[extra_params].pth
    
    parts = []
    
    # ALWAYS include batch size, learning rate, and total steps as requested
    # Handle batch size (use default if None)
    batch_size = combination['batch_size'] if combination['batch_size'] is not None else 4096
    parts.append(f"bs{batch_size}")
    
    # Always include learning rate
    lr_str = str(combination['learning_rate']).replace('.', 'p')
    parts.append(f"lr{lr_str}")
    
    # Handle total steps
    if combination['target_steps'] is not None:
        # Explicit target steps provided
        total_steps = combination['target_steps']
        parts.append(f"steps{total_steps}")
    elif args.auto_steps:
        # Auto steps based on feature dimension
        # We can't calculate the exact value here, but we can indicate auto steps is used
        parts.append(f"autosteps{combination['auto_steps_base']}")
    else:
        # Default steps (200,000)
        parts.append("steps200000")
    
    # Always include L1 lambda in the filename
    l1_str = str(combination['l1_lambda']).replace('.', 'p')
    parts.append(f"l1{l1_str}")
    
    # Add grad accumulation steps if not default
    if combination['grad_accum_steps'] != 1:
        parts.append(f"accum{combination['grad_accum_steps']}")
    
    # Add memory bank or old ST indicators for ST models
    if model_type_prefix == 'st':
        if args.use_memory_bank:
            parts.append("memory")
        
        if args.use_old_st:
            parts.append("oldst")
    
    # Combine parts into filename
    filename = f"{'_'.join(parts)}.pth"
    
    return filename

def get_hierarchical_model_path(combination, model_type_prefix, args, run_index=0):
    """Get hierarchical path for a model using the custom directory structure"""
    # For SAE: models/[dataset]/sae/activation_function/feature_dimension/bs_lr_steps.pth
    # For ST: models/[dataset]/st/attention_function/feature_dimension/bs_lr_steps.pth
    
    # Add dataset at the top level
    if model_type_prefix == 'sae':
        # Use activation function for SAE
        hierarchy_path = os.path.join(
            'models',
            combination['dataset'],  # Dataset at the top level
            'sae',                   # Model type
            combination['activation'],  # Activation function
            str(combination['feature_dimension'])  # Feature dimension
        )
    else:  # ST model
        # Use attention function for ST
        hierarchy_path = os.path.join(
            'models',
            combination['dataset'],  # Dataset at the top level
            'st',                    # Model type
            combination['attention_fn'],  # Attention function
            str(combination['feature_dimension'])  # Feature dimension
        )
    
    # Create the directory path
    os.makedirs(hierarchy_path, exist_ok=True)
    
    # Get simplified filename (now without redundant information)
    filename = get_simplified_filename(model_type_prefix, combination, args)
    
    # Modify filename to include run index if needed (only for runs 2+)
    if run_index > 0:
        # Split filename into base and extension
        base, ext = os.path.splitext(filename)
        filename = f"{base}_{run_index+1}{ext}"
    
    # Full model path
    model_path = os.path.join(hierarchy_path, filename)
    
    return model_path

def find_all_run_files(base_path):
    """Find all existing run files for a given base path"""
    if base_path is None:
        return {}
    
    # Handle the case where base_path might contain autosteps
    if 'autosteps' in base_path:
        base_path = get_steps_version_of_path(base_path)
        
    base_dir = os.path.dirname(base_path)
    base_name, ext = os.path.splitext(os.path.basename(base_path))
    
    # Look for the base file and any files with _2, _3, etc. suffixes
    pattern = os.path.join(base_dir, f"{base_name}*{ext}")
    files = glob.glob(pattern)
    
    # Map files to their run indices
    run_files = {}
    for file in files:
        filename = os.path.basename(file)
        file_base, file_ext = os.path.splitext(filename)
        
        # Check if this is the base file (run 1)
        if file_base == base_name:
            run_files[0] = file
        else:
            # Check if there's a run suffix (_2, _3, etc.)
            suffix_parts = file_base.split('_')
            if len(suffix_parts) > 1 and suffix_parts[-1].isdigit():
                run_idx = int(suffix_parts[-1]) - 1  # Convert from 1-indexed to 0-indexed
                run_files[run_idx] = file
    
    return run_files

def find_next_available_run(base_path):
    """Find the next available run index for a model path"""
    if base_path is None:
        return 0
    
    # Check if we have a renamed version (for auto_steps)
    if 'autosteps' in base_path:
        updated_path = get_steps_version_of_path(base_path)
        if updated_path != base_path:
            base_path = updated_path
        
    existing_runs = find_all_run_files(base_path)
    
    if not existing_runs:
        return 0  # First run
    
    # Return the next available run index
    max_run = max(existing_runs.keys())
    return max_run + 1

def model_exists(base_path, run_index):
    """Check if a specific run of a model exists"""
    if base_path is None:
        return False
    
    # Check for renamed files if this contains autosteps
    if 'autosteps' in base_path:
        base_path = get_steps_version_of_path(base_path)
        
    if run_index == 0:
        # Check for the base file
        return os.path.exists(base_path)
    else:
        # Check for a run with the specified index
        base_name, ext = os.path.splitext(base_path)
        run_path = f"{base_name}_{run_index+1}{ext}"
        return os.path.exists(run_path)

def find_model_files(model_id):
    """Find all model files matching the model_id with support for new naming convention"""
    # Get base training parameters from model_id
    base_params = model_id.split('_')
    
    # Original patterns to search for model files saved by main.py
    original_patterns = [
        f"models/*_model_{model_id}.pth",           # Original pattern
        f"models/*_model_{model_id}.*.pth",         # With extensions like .best or .step1000
    ]
    
    # New patterns for our custom saved files with just training parameters
    # Extract key parameters from model_id
    lr_pattern = next((p for p in base_params if p.startswith('lr')), '')
    bs_pattern = next((p for p in base_params if p.startswith('bs')), '')
    steps_pattern = next((p for p in base_params if p.startswith('steps') or p.startswith('autosteps')), '')
    
    # Create patterns for our new format
    if lr_pattern and bs_pattern and steps_pattern:
        new_patterns = [
            f"models/*/*/*/*/{bs_pattern}_{lr_pattern}_{steps_pattern}*.pth",  # Basic pattern
            f"models/*/*/*/*/{bs_pattern}_{lr_pattern}_{steps_pattern}_*.pth", # With extra params
        ]
        
        # Also search for steps versions of autosteps patterns
        if 'autosteps' in steps_pattern:
            steps_version = steps_pattern.replace('autosteps', 'steps')
            new_patterns.extend([
                f"models/*/*/*/*/{bs_pattern}_{lr_pattern}_{steps_version}*.pth",
                f"models/*/*/*/*/{bs_pattern}_{lr_pattern}_{steps_version}_*.pth",
            ])
    else:
        new_patterns = []
    
    # Combine all patterns
    all_patterns = original_patterns + new_patterns
    
    # Collect all matching files
    all_files = []
    for pattern in all_patterns:
        files = glob.glob(pattern)
        all_files.extend(files)
    
    # Remove duplicates
    all_files = list(set(all_files))
    
    return all_files

def extract_actual_steps_from_model(model_path):
    """
    Extract the actual number of steps from a trained model checkpoint
    """
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Try to extract step information
        if isinstance(checkpoint, dict):
            # Check various possible keys for step information
            if 'step' in checkpoint:
                return checkpoint['step']
            elif 'model_state_dict' in checkpoint and isinstance(checkpoint['model_state_dict'], dict):
                # Maybe steps are stored in a nested dictionary
                if 'steps' in checkpoint:
                    return checkpoint['steps']
                elif 'total_steps' in checkpoint:
                    return checkpoint['total_steps']
            elif 'training_history' in checkpoint and 'steps' in checkpoint['training_history']:
                # Get the last step recorded in training history
                if checkpoint['training_history']['steps']:
                    return checkpoint['training_history']['steps'][-1]
        
        # If we can't find step information, return None
        return None
    except Exception as e:
        print(f"Error extracting steps from model {model_path}: {e}")
        return None

def rename_with_actual_steps(file_path, actual_steps):
    """
    Rename a file to replace 'autosteps' with the actual step count
    """
    if actual_steps is None or actual_steps == 0:
        return file_path  # Don't rename if we don't have valid step information
        
    try:
        # Get the directory and filename
        dir_name, filename = os.path.split(file_path)
        
        # Check if the filename contains 'autosteps'
        if 'autosteps' in filename:
            # Create the new filename with actual steps
            new_filename = filename.replace(
                'autosteps' + filename.split('autosteps')[1].split('_')[0], 
                f'steps{actual_steps}'
            )
            
            # Create the new file path
            new_file_path = os.path.join(dir_name, new_filename)
            
            # Rename the file
            if os.path.exists(file_path):
                os.rename(file_path, new_file_path)
                print(f"Renamed file to show actual steps: {new_file_path}")
                return new_file_path
            else:
                print(f"Warning: File does not exist: {file_path}")
                return file_path
        else:
            return file_path  # No need to rename if it doesn't have 'autosteps'
    except Exception as e:
        print(f"Error renaming file: {e}")
        return file_path

def run_training(combination, args, idx, total, run_idx=0, base_run_index=0):
    """Run training for a specific combination of parameters and run index"""
    # Create model ID
    model_id = get_model_id(combination, args)
    
    # Create temporary directory for model outputs
    temp_dir = args.temp_dir
    os.makedirs(temp_dir, exist_ok=True)
    
    # Calculate the actual run index (base + offset)
    if args.continue_numbering:
        actual_run_index = base_run_index + run_idx
        print(f"Using run index {actual_run_index} = {base_run_index} (base) + {run_idx} (offset)")
    else:
        actual_run_index = run_idx
    
    # Get hierarchical paths for model outputs
    if combination['model_type'] in ['sae', 'both']:
        sae_path = get_hierarchical_model_path(combination, 'sae', args, actual_run_index)
        # Create directory for SAE
        os.makedirs(os.path.dirname(sae_path), exist_ok=True)
    else:
        sae_path = None
        
    if combination['model_type'] in ['st', 'both']:
        st_path = get_hierarchical_model_path(combination, 'st', args, actual_run_index)
        # Create directory for ST
        os.makedirs(os.path.dirname(st_path), exist_ok=True)
    else:
        st_path = None
    
    # CRITICAL FIX: Check if renamed versions of these paths exist and update them
    if args.auto_steps:
        if sae_path and 'autosteps' in sae_path:
            sae_exists, actual_sae_path = model_exists_with_autosteps(sae_path)
            if sae_exists:
                print(f"Found existing renamed SAE model: {actual_sae_path}")
                sae_path = actual_sae_path
                
        if st_path and 'autosteps' in st_path:
            st_exists, actual_st_path = model_exists_with_autosteps(st_path)
            if st_exists:
                print(f"Found existing renamed ST model: {actual_st_path}")
                st_path = actual_st_path
    
    # Check if model already exists - skip only if not forcing retrain
    if args.skip_existing and not args.force_retrain:
        # Check for existing models, handling auto_steps renaming
        sae_exists, actual_sae_path = model_exists_with_autosteps(sae_path)
        st_exists, actual_st_path = model_exists_with_autosteps(st_path)
        
        # Update paths to actual files if found
        if sae_exists:
            sae_path = actual_sae_path
            print(f"Found existing SAE model at: {sae_path}")
        if st_exists:
            st_path = actual_st_path
            print(f"Found existing ST model at: {st_path}")
        
        if sae_exists or st_exists:
            print(f"\nSkipping combination {idx+1}/{total} run {actual_run_index+1}: {model_id} (already trained)")
            return {
                'model_id': model_id,
                'returncode': 0,
                'combination': combination,
                'skipped': True,
                'sae_path': sae_path,
                'st_path': st_path,
                'run_index': actual_run_index
            }
    
    # Add run index to model_id for display purposes
    if args.num_runs > 1:
        if args.continue_numbering:
            run_display = f"run {actual_run_index+1}"
        else:
            run_display = f"run {run_idx+1}/{args.num_runs}"
    else:
        run_display = ""
    display_model_id = f"{model_id} {run_display}".strip()
    
    # Build command for running main.py
    cmd = [
        "python", "main.py",
        "--dataset", combination['dataset'],
        "--model_type", combination['model_type'],
        "--feature_dimension", str(combination['feature_dimension']),
        "--learning_rate", str(combination['learning_rate']),
        "--l1_lambda", str(combination['l1_lambda']),
        "--model_id", model_id,
    ]
    
    # Add model-specific parameters based on model type
    if combination['model_type'] in ['st', 'both']:
        # For ST models: add attention function and set activation to "none"
        cmd.extend(["--attention_fn", combination['attention_fn']])
        cmd.extend(["--activation", "none"])
        
        # Add attention dimension if specified
        if combination['attention_dimension']:
            cmd.extend(["--attention_dimension", str(combination['attention_dimension'])])
    else:
        # For SAE models: always use ReLU activation regardless of what's in combination
        cmd.extend(["--activation", "relu"])
    
    # Add optional flags
    if args.use_memory_bank:
        cmd.append("--use_memory_bank")
    
    if args.use_old_st:
        cmd.append("--use_old_st")
    if args.use_mixed_precision:
        cmd.append("--use_mixed_precision")
    
    # Add batch size if specified
    if combination['batch_size'] is not None:
        cmd.extend(["--batch_size", str(combination['batch_size'])])
    
    # Add target steps if specified
    if combination['target_steps'] is not None:
        cmd.extend(["--target_steps", str(combination['target_steps'])])
    
    # Add gradient accumulation steps if specified
    if combination['grad_accum_steps'] != 1:
        cmd.extend(["--grad_accum_steps", str(combination['grad_accum_steps'])])
    
    # Add evaluation frequency if specified
    if combination['eval_freq'] is not None:
        cmd.extend(["--eval_freq", str(combination['eval_freq'])])
    
    # Add auto_steps if specified
    if args.auto_steps:
        cmd.append("--auto_steps")
        # Add auto_steps_base if not default
        if combination['auto_steps_base'] != 200000:
            cmd.extend(["--auto_steps_base", str(combination['auto_steps_base'])])
    
    # Always add force_retrain flag when using this script with multiple runs
    if args.force_retrain:
        cmd.append("--force_retrain")
    if args.device:
        cmd.extend(["--device", args.device])
    # Add direct save paths - this makes models save directly to the correct location
    if sae_path:
        cmd.extend(["--sae_save_path", sae_path])
    
    if st_path:
        cmd.extend(["--st_save_path", st_path])
    
    # Add any additional arguments
    if args.main_args:
        cmd.extend(args.main_args.split())
    
    # Print status
    print(f"\n\n{'='*80}")
    print(f"Training combination {idx+1}/{total}: {display_model_id}")
    print(f"SAE save path: {sae_path}")
    print(f"ST save path: {st_path}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    # Record start time
    start_time = time.time()
    
    # Run the training process
    result = subprocess.run(cmd, check=False)
    
    # Record end time and duration
    end_time = time.time()
    duration = end_time - start_time
    
    # Check if models were actually saved to the intended paths
    files_saved_correctly = True
    
    if sae_path and os.path.exists(sae_path):
        print(f"Successfully saved SAE model to {sae_path}")
    elif sae_path:
        files_saved_correctly = False
        print(f"Warning: SAE model not found at {sae_path}")
    
    if st_path and os.path.exists(st_path):
        print(f"Successfully saved ST model to {st_path}")
    elif st_path:
        files_saved_correctly = False
        print(f"Warning: ST model not found at {st_path}")
    
    # Check if we're using auto steps
    if args.auto_steps and result.returncode == 0:
        # Rename files to use actual step count instead of 'autosteps'
        if sae_path and 'autosteps' in sae_path and os.path.exists(sae_path):
            # Extract actual steps from the SAE model
            actual_steps = extract_actual_steps_from_model(sae_path)
            if actual_steps:
                # Rename the file
                sae_path = rename_with_actual_steps(sae_path, actual_steps)
                print(f"Updated SAE path with actual steps: {sae_path}")
        
        if st_path and 'autosteps' in st_path and os.path.exists(st_path):
            # Extract actual steps from the ST model
            actual_steps = extract_actual_steps_from_model(st_path)
            if actual_steps:
                # Rename the file
                st_path = rename_with_actual_steps(st_path, actual_steps)
                print(f"Updated ST path with actual steps: {st_path}")
    
    # If files weren't saved correctly but the process returned 0, try to find and move them
    if not files_saved_correctly and result.returncode == 0:
        print("Models weren't saved to the hierarchical paths. Trying to find and move them...")
        
        # Find all model files
        model_files = find_model_files(model_id)
        print(f"Found {len(model_files)} model files: {model_files}")
        
        for file_path in model_files:
            filename = os.path.basename(file_path)
            
            # Determine target path based on model type
            if "sae_model_" in filename and sae_path:
                target_path = sae_path
                model_type = "sae"
            elif "st_model_" in filename and st_path:
                target_path = st_path
                model_type = "st"
            else:
                continue  # Skip unknown file types
            
            # Create directory if needed (redundant but safe)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            
            try:
                # Try to move the file
                shutil.move(file_path, target_path)
                print(f"Moved {filename} to {target_path}")
            except Exception as e:
                # If move fails, try to copy
                print(f"Move failed, trying to copy: {e}")
                try:
                    shutil.copy2(file_path, target_path)
                    os.remove(file_path)  # Remove original after successful copy
                    print(f"Copied {filename} to {target_path} and removed original")
                except Exception as e2:
                    print(f"Error moving/copying {filename}: {e2}")
    
    return {
        'model_id': model_id,
        'returncode': result.returncode,
        'combination': combination,
        'duration': duration,
        'skipped': False,
        'sae_path': sae_path,
        'st_path': st_path,
        'run_index': actual_run_index
    }

def create_summary(results, args):
    """Create a summary of training results with run information"""
    summary_path = args.summary_file
    
    # Count results by type
    successful = [r for r in results if r['returncode'] == 0 and not r.get('skipped', False)]
    skipped = [r for r in results if r.get('skipped', False)]
    failed = [r for r in results if r['returncode'] != 0]
    
    # Generate summary file with explicit UTF-8 encoding
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# Training Results Summary\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overview
        f.write("## Overview\n\n")
        f.write(f"Total configurations: {len(results) // max(1, args.num_runs)}\n")
        f.write(f"Total runs: {len(results)}\n")
        f.write(f"Runs per configuration: {args.num_runs}\n")
        f.write(f"Successful: {len(successful)}\n")
        f.write(f"Skipped: {len(skipped)}\n")
        f.write(f"Failed: {len(failed)}\n\n")
        
        # Hierarchical Structure
        f.write("## Hierarchical Organization\n\n")
        f.write("Models are organized in the following structure:\n")
        f.write("```\n")
        f.write("models/\n")
        f.write("  |-- [dataset]/\n")
        f.write("      |-- sae/\n")
        f.write("      |   |-- [activation_function]/\n")
        f.write("      |   |   |-- [feature_dimension]/\n")
        f.write("      |   |       |-- bs{batch}_lr{lr}_steps{steps}.pth         # Run 1\n")
        f.write("      |   |       |-- bs{batch}_lr{lr}_steps{steps}_2.pth       # Run 2\n")
        f.write("      |   |       |-- bs{batch}_lr{lr}_steps{steps}_3.pth       # Run 3\n")
        f.write("      |-- st/\n")
        f.write("          |-- [attention_function]/\n")
        f.write("              |-- [feature_dimension]/\n")
        f.write("                  |-- bs{batch}_lr{lr}_steps{steps}.pth         # Run 1\n")
        f.write("                  |-- bs{batch}_lr{lr}_steps{steps}_2.pth       # Run 2\n")
        f.write("                  |-- bs{batch}_lr{lr}_steps{steps}_3.pth       # Run 3\n")
        f.write("```\n\n")
        
        # Group successful models by configuration
        if successful:
            # Group by model_id (without run index)
            grouped_results = {}
            for result in successful:
                model_id = result['model_id']
                if model_id not in grouped_results:
                    grouped_results[model_id] = []
                grouped_results[model_id].append(result)
            
            f.write("## Successful Models\n\n")
            f.write("| Configuration | Run | Training Time | Path |\n")
            f.write("|--------------|-----|---------------|------|\n")
            
            for model_id, results_for_model in grouped_results.items():
                # Sort by run index
                sorted_results = sorted(results_for_model, key=lambda x: x.get('run_index', 0))
                
                # Print first row with configuration details
                first_result = sorted_results[0]
                duration_hours = first_result.get('duration', 0) / 3600
                path = first_result['sae_path'] if first_result['sae_path'] else first_result['st_path']
                f.write(f"| {model_id} | 1 | {duration_hours:.1f}h | {path} |\n")
                
                # Print additional runs if any
                for i, result in enumerate(sorted_results[1:], 2):
                    duration_hours = result.get('duration', 0) / 3600
                    path = result['sae_path'] if result['sae_path'] else result['st_path']
                    f.write(f"| | {i} | {duration_hours:.1f}h | {path} |\n")
            
            f.write("\n")
        
        # Skipped models
        if skipped:
            f.write("## Skipped Models\n\n")
            for result in skipped:
                run_info = f" (run {result.get('run_index', 0) + 1})" if args.num_runs > 1 else ""
                f.write(f"- {result['model_id']}{run_info}\n")
            f.write("\n")
        
        # Failed models
        if failed:
            f.write("## Failed Models\n\n")
            for result in failed:
                run_info = f" (run {result.get('run_index', 0) + 1})" if args.num_runs > 1 else ""
                f.write(f"- {result['model_id']}{run_info} (return code: {result['returncode']})\n")
    
    print(f"\nSummary written to {summary_path}")
    return summary_path
def main():
    """Main function with parallelization"""
    args = parse_args()
    # Generate combinations
    combinations = generate_combinations(args)
    
    print(f"Generated {len(combinations)} configurations to train")
    print(f"Number of runs per configuration: {args.num_runs}")
    print(f"Total training runs to perform: {len(combinations) * args.num_runs}")
    print(f"Using {args.workers} parallel workers")
    
    # Prepare all training tasks
    all_tasks = []
    for idx, combo in enumerate(combinations):
        # Determine base run index for continue_numbering mode
        base_run_index = 0
        if args.continue_numbering:
            base_sae_path = get_hierarchical_model_path(combo, 'sae', args, 0) if combo['model_type'] in ['sae', 'both'] else None
            base_st_path = get_hierarchical_model_path(combo, 'st', args, 0) if combo['model_type'] in ['st', 'both'] else None
            
            # Handle auto_steps renaming
            if args.auto_steps:
                if base_sae_path and 'autosteps' in base_sae_path:
                    base_sae_path = get_steps_version_of_path(base_sae_path)
                if base_st_path and 'autosteps' in base_st_path:
                    base_st_path = get_steps_version_of_path(base_st_path)
            
            # Find highest existing run index
            sae_next_run = find_next_available_run(base_sae_path) if base_sae_path else 0
            st_next_run = find_next_available_run(base_st_path) if base_st_path else 0
            base_run_index = max(sae_next_run, st_next_run)
        
        # Create tasks for each run of this combination
        for run_idx in range(args.num_runs):
            all_tasks.append((combo, idx, len(combinations), run_idx, base_run_index))
    
    # Create a partial function with fixed args
    training_func = partial(run_training_task, args=args)
    
    # Execute tasks in parallel
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(training_func, *task) for task in all_tasks]
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                print(f"Completed task for model: {result['model_id']}, run: {result['run_index']+1}")
            except Exception as e:
                print(f"Task failed with error: {e}")
    
    # Create summary
    summary_path = create_summary(results, args)
    
    print("\nTraining completed!")
    print(f"Completed {len(results)} of {len(all_tasks)} training tasks.")
    print(f"See {summary_path} for details.")

def run_training_task(combo, idx, total, run_idx, base_run_index, args):
    """Wrapper for run_training to use with concurrent.futures"""
    return run_training(combo, args, idx, total, run_idx, base_run_index)
if __name__ == "__main__":
    main()