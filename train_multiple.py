#!/usr/bin/env python3
"""
Script to train models and save them in a customized hierarchical folder structure:
- SAE: models/sae/activation_function/feature_dimension/
- ST: models/st/attention_function/feature_dimension/
"""

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
    
    # Other options
    parser.add_argument('--skip_existing', action='store_true',
                      help='Skip training if model already exists')
    parser.add_argument('--summary_file', type=str, default='training_summary.md',
                      help='Path to write training summary')
    parser.add_argument('--main_args', type=str, default='',
                      help='Additional arguments to pass to main.py')
    parser.add_argument('--temp_dir', type=str, default='temp_models',
                      help='Temporary directory for model files before moving')
    
    return parser.parse_args()

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
         
        # Skip invalid combinations (e.g., attention functions only work with ST)
        if (model_type == 'sae' and attention_fn != 'softmax'):
            continue
            
        combo = {
            'dataset': dataset,
            'model_type': model_type,
            'attention_fn': attention_fn,
            'feature_dimension': feature_dim,
            'attention_dimension': attention_dim,
            'learning_rate': lr,
            'l1_lambda': l1_lambda,
            'activation': activation,
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
    
    # Add L1 lambda if not default
    if combination['l1_lambda'] != 5.0:
        parts.append(f"l1{str(combination['l1_lambda']).replace('.', 'p')}")
    
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

def get_hierarchical_model_path(combination, model_type_prefix, args):
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
    
    # Full model path
    model_path = os.path.join(hierarchy_path, filename)
    
    return model_path

def model_exists(combination, args):
    """Check if a model already exists in the hierarchical structure"""
    if combination['model_type'] in ['sae', 'both']:
        sae_path = get_hierarchical_model_path(combination, 'sae', args)
        if os.path.exists(sae_path):
            return True
    
    if combination['model_type'] in ['st', 'both']:
        st_path = get_hierarchical_model_path(combination, 'st', args)
        if os.path.exists(st_path):
            return True
    
    return False

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
def run_training(combination, args, idx, total):
    """Run training for a specific combination of parameters"""
    # Create model ID
    model_id = get_model_id(combination, args)
    
    # Create temporary directory for model outputs
    temp_dir = args.temp_dir
    os.makedirs(temp_dir, exist_ok=True)
    
    # Check if model already exists
    if args.skip_existing and model_exists(combination, args):
        print(f"\nSkipping combination {idx+1}/{total}: {model_id} (already trained)")
        return {
            'model_id': model_id,
            'returncode': 0,
            'combination': combination,
            'skipped': True
        }
    
    # Get hierarchical paths for model outputs
    if combination['model_type'] in ['sae', 'both']:
        sae_path = get_hierarchical_model_path(combination, 'sae', args)
    else:
        sae_path = None
        
    if combination['model_type'] in ['st', 'both']:
        st_path = get_hierarchical_model_path(combination, 'st', args)
    else:
        st_path = None
    
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
    
    # Add attention function for ST models
    if combination['model_type'] in ['st', 'both']:
        cmd.extend(["--attention_fn", combination['attention_fn']])
        # Fixed activation to "none" for ST models as specified
        cmd.extend(["--activation", "none"])
        
        # Add attention dimension if specified
        if combination['attention_dimension']:
            cmd.extend(["--attention_dimension", str(combination['attention_dimension'])])
    else:
        # Add activation function for SAE models
        cmd.extend(["--activation", combination['activation']])
    
    # Add optional flags
    if args.use_memory_bank:
        cmd.append("--use_memory_bank")
    
    if args.use_old_st:
        cmd.append("--use_old_st")
    
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
    
    # Add any additional arguments
    if args.main_args:
        cmd.extend(args.main_args.split())
    
    # Print status
    print(f"\n\n{'='*80}")
    print(f"Training combination {idx+1}/{total}: {model_id}")
    # Create direct save paths
    if sae_path:
        sae_dir = os.path.dirname(sae_path)
        os.makedirs(sae_dir, exist_ok=True)
        # Add a custom argument to save SAE directly
        cmd.extend(["--sae_save_path", sae_path])
        print(f"Adding direct SAE save path: {sae_path}")
        
    if st_path:
        st_dir = os.path.dirname(st_path)
        os.makedirs(st_dir, exist_ok=True)
        # Add a custom argument to save ST directly
        cmd.extend(["--st_save_path", st_path])
        print(f"Adding direct ST save path: {st_path}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    # Record start time
    start_time = time.time()
    
    # Run the training process
    result = subprocess.run(cmd, check=False)
    
    # Record end time and duration
    end_time = time.time()
    duration = end_time - start_time
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
        
        # Also look for any additional files like .best or .stepXXX
        if sae_path and not sae_path.endswith('.best'):
            sae_best_path = sae_path.replace('.pth', '.best')
            if os.path.exists(sae_best_path) and 'autosteps' in sae_best_path:
                actual_steps = extract_actual_steps_from_model(sae_best_path)
                if actual_steps:
                    rename_with_actual_steps(sae_best_path, actual_steps)
        
        if st_path and not st_path.endswith('.best'):
            st_best_path = st_path.replace('.pth', '.best')
            if os.path.exists(st_best_path) and 'autosteps' in st_best_path:
                actual_steps = extract_actual_steps_from_model(st_best_path)
                if actual_steps:
                    rename_with_actual_steps(st_best_path, actual_steps)
    # Find and move model files to hierarchical paths if training succeeded
    if result.returncode == 0:
        # Find all model files
        model_files = find_model_files(model_id)
        print(f"Found {len(model_files)} model files: {model_files}")
        
        for file_path in model_files:
            filename = os.path.basename(file_path)
            
            # Determine target path based on model type
            if filename.startswith("sae_model_") and sae_path:
                target_path = sae_path
                model_type = "sae"
            elif filename.startswith("st_model_") and st_path:
                target_path = st_path
                model_type = "st"
            else:
                continue  # Skip unknown file types
            
            # Move the file
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
        'sae_path': sae_path if combination['model_type'] in ['sae', 'both'] else None,
        'st_path': st_path if combination['model_type'] in ['st', 'both'] else None
    }

def create_summary(results, args):
    """Create a summary of training results"""
    summary_path = args.summary_file
    
    # Count results by type
    successful = [r for r in results if r['returncode'] == 0 and not r.get('skipped', False)]
    skipped = [r for r in results if r.get('skipped', False)]
    failed = [r for r in results if r['returncode'] != 0]
    
    # Generate summary file
    with open(summary_path, 'w') as f:
        f.write("# Training Results Summary\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overview
        f.write("## Overview\n\n")
        f.write(f"Total combinations: {len(results)}\n")
        f.write(f"Successful: {len(successful)}\n")
        f.write(f"Skipped: {len(skipped)}\n")
        f.write(f"Failed: {len(failed)}\n\n")
        
        # Hierarchical Structure
        f.write("## Hierarchical Organization\n\n")
        f.write("Models are organized in the following structure:\n")
        f.write("```\n")
        f.write("models/\n")
        f.write("  ├── sae/\n")
        f.write("  │   ├── [activation_function]/\n")
        f.write("  │   │   ├── [feature_dimension]/\n")
        f.write("  │   │   │   ├── sae_[simple_name].pth\n")
        f.write("  │   │   │   └── ...\n")
        f.write("  └── st/\n")
        f.write("      ├── [attention_function]/\n")
        f.write("      │   ├── [feature_dimension]/\n")
        f.write("      │   │   ├── st_[simple_name].pth\n")
        f.write("      │   │   └── ...\n")
        f.write("```\n\n")
        
        # Successful models
        if successful:
            f.write("## Successful Models\n\n")
            f.write("| Model ID | Training Time | Path |\n")
            f.write("|----------|---------------|------|\n")
            
            for result in successful:
                model_id = result['model_id']
                duration_hours = result.get('duration', 0) / 3600
                
                # Get path(s)
                paths = []
                if result['sae_path']:
                    paths.append(result['sae_path'])
                if result['st_path']:
                    paths.append(result['st_path'])
                
                path_str = "<br>".join(paths)
                f.write(f"| {model_id} | {duration_hours:.1f}h | {path_str} |\n")
            
            f.write("\n")
        
        # Skipped models
        if skipped:
            f.write("## Skipped Models\n\n")
            for result in skipped:
                f.write(f"- {result['model_id']}\n")
            f.write("\n")
        
        # Failed models
        if failed:
            f.write("## Failed Models\n\n")
            for result in failed:
                f.write(f"- {result['model_id']} (return code: {result['returncode']})\n")
    
    print(f"\nSummary written to {summary_path}")
    return summary_path

def main():
    """Main function to run training combinations"""
    args = parse_args()
    
    # Generate combinations
    combinations = generate_combinations(args)
    
    print(f"Generated {len(combinations)} combinations to train")
    
    # Show sample of combinations
    if len(combinations) > 10:
        print("\nSample of combinations to train:")
        for i, combo in enumerate(combinations[:5]):
            model_id = get_model_id(combo, args)
            print(f"{i+1}. {model_id}")
        print("...")
    
    # Train each combination
    results = []
    for idx, combo in enumerate(combinations):
        result = run_training(combo, args, idx, len(combinations))
        results.append(result)
    
    # Create summary
    summary_path = create_summary(results, args)
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main()