#!/usr/bin/env python3
"""
Script to automate training of multiple combinations of feature models and attention functions.
Trained models are organized in folders based on their performance.
"""

import os
import argparse
import itertools
import subprocess
import json
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import time
from datetime import datetime

def parse_args():
    """Parse command line arguments for training combinations"""
    parser = argparse.ArgumentParser(description='Train multiple combinations of models and attention functions')
    
    # Base arguments
    parser.add_argument('--dataset', type=str, default='mnist', 
                        help='Dataset to use for training')
    parser.add_argument('--model_types', type=str, nargs='+', default=['st'],
                        help='List of model types to train (sae, st, both)')
    parser.add_argument('--attention_fns', type=str, nargs='+', 
                        default=['softmax'],
                        help='List of attention functions for ST models')
    parser.add_argument('--feature_dimensions', type=int, nargs='+', default=[100],
                        help='List of feature dimensions (m) to try')
    parser.add_argument('--attention_dimensions', type=int, nargs='+', default=[None],
                        help='List of attention dimensions (a) to try (None for auto)')
    parser.add_argument('--learning_rates', type=float, nargs='+', default=[5e-5],
                        help='List of learning rates to try')
    parser.add_argument('--l1_lambdas', type=float, nargs='+', default=[5.0],
                        help='List of L1 regularization strengths to try')
    parser.add_argument('--activations', type=str, nargs='+', default=['relu'],
                        help='List of activation functions to try')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Base directory for storing results')
    parser.add_argument('--sort_metric', type=str, default='val_loss',
                        choices=['val_loss', 'sparsity', 'dead_ratio', 'combined'],
                        help='Metric to use for sorting models')
    parser.add_argument('--num_groups', type=int, default=3,
                        help='Number of performance groups to create')
    
    # Additional training configuration
    parser.add_argument('--use_memory_bank', action='store_true',
                        help='Use memory bank approach for ST models (default: direct K-V)')
    parser.add_argument('--use_old_st', action='store_true',
                        help='Use old ST implementation')
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[None],
                        help='List of batch sizes to try')
    parser.add_argument('--target_steps', type=int, nargs='+', default=[None],
                        help='List of training steps to try')
    parser.add_argument('--grad_accum_steps', type=int, nargs='+', default=[1],
                        help='List of gradient accumulation steps to try')
    parser.add_argument('--auto_steps', action='store_true',
                        help='Use auto step calculation')
    parser.add_argument('--auto_steps_bases', type=int, nargs='+', default=[200000],
                        help='List of base steps for auto step calculation')
    parser.add_argument('--eval_freqs', type=int, nargs='+', default=[None],
                        help='List of evaluation frequencies to try')
    
    # Experiment organization
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name of the experiment (default: timestamp)')
    parser.add_argument('--continue_from', type=str, default=None,
                        help='Continue from a previous experiment directory')
    parser.add_argument('--skip_completed', action='store_true',
                        help='Skip combinations that have already been trained')
    
    # Arguments to pass through to main.py
    parser.add_argument('--main_args', type=str, default='',
                        help='Additional arguments to pass to main.py (in quotes)')
    
    return parser.parse_args()

def generate_combinations(args):
    """Generate all combinations of parameters to try"""
    combinations = []
    
    # Handle None values in various parameters
    attention_dims = args.attention_dimensions
    batch_sizes = args.batch_sizes
    target_steps_list = args.target_steps
    grad_accum_steps_list = args.grad_accum_steps
    eval_freqs = args.eval_freqs
    auto_steps_bases = args.auto_steps_bases
    
    # Normalize None values for proper iteration
    if attention_dims == [None]:
        attention_dims = [None]
    if batch_sizes == [None]:
        batch_sizes = [None]
    if target_steps_list == [None]:
        target_steps_list = [None]
    if eval_freqs == [None]:
        eval_freqs = [None]
    
    # Create product of all specified parameters
    param_grid = itertools.product(
        args.model_types,
        args.attention_fns,
        args.feature_dimensions,
        attention_dims,
        args.learning_rates,
        args.l1_lambdas,
        args.activations,
        batch_sizes,
        target_steps_list,
        grad_accum_steps_list,
        eval_freqs,
        auto_steps_bases
    )
    
    for (model_type, attention_fn, feature_dim, attention_dim, lr, l1_lambda, 
         activation, batch_size, target_steps, grad_accum_steps, eval_freq, 
         auto_steps_base) in param_grid:
         
        # Skip invalid combinations (e.g., attention functions only work with ST)
        if (model_type == 'sae' and attention_fn != 'softmax'):
            continue
            
        combo = {
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
    base_id = f"{args.dataset}_{combination['model_type']}_{combination['feature_dimension']}"
    
    if combination['model_type'] in ['st', 'both']:
        base_id += f"_{combination['attention_fn']}"
        if combination['attention_dimension']:
            base_id += f"_a{combination['attention_dimension']}"
    
    if combination['activation'] != 'relu':
        base_id += f"_{combination['activation']}"
    
    if combination['learning_rate'] != 5e-5:
        base_id += f"_lr{combination['learning_rate']}"
    
    if combination['l1_lambda'] != 5.0:
        base_id += f"_l1{combination['l1_lambda']}"
    
    # Add batch size if specified and not default
    if combination['batch_size'] is not None:
        base_id += f"_bs{combination['batch_size']}"
    
    # Add target steps if specified and not default
    if combination['target_steps'] is not None:
        base_id += f"_steps{combination['target_steps']}"
    
    # Add gradient accumulation steps if not default (1)
    if combination['grad_accum_steps'] != 1:
        base_id += f"_accum{combination['grad_accum_steps']}"
    
    # Add eval frequency if specified and not default
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

def check_if_trained(model_id, args, combination):
    """Check if a model has already been trained"""
    if args.continue_from and args.skip_completed:
        # Check in the experiment directory first
        experiment_dir = args.continue_from
        
        # Check for SAE model
        if combination['model_type'] in ['sae', 'both']:
            sae_path = os.path.join(experiment_dir, 'models', f"sae_model_{model_id}.pth")
            if os.path.exists(sae_path):
                return True
        
        # Check for ST model
        if combination['model_type'] in ['st', 'both']:
            st_path = os.path.join(experiment_dir, 'models', f"st_model_{model_id}.pth")
            if os.path.exists(st_path):
                return True
    
    # Also check in the default models directory
    if combination['model_type'] in ['sae', 'both']:
        sae_path = f"models/sae_model_{model_id}.pth"
        if os.path.exists(sae_path):
            return True
    
    if combination['model_type'] in ['st', 'both']:
        st_path = f"models/st_model_{model_id}.pth"
        if os.path.exists(st_path):
            return True
    
    return False

def run_training(combination, args, idx, total, experiment_dir):
    """Run training for a specific combination of parameters"""
    # Create a unique model_id
    model_id = get_model_id(combination, args)
    
    # Check if this model has already been trained
    if check_if_trained(model_id, args, combination):
        print(f"\nSkipping combination {idx+1}/{total}: {model_id} (already trained)")
        return {
            'model_id': model_id,
            'returncode': 0,  # Pretend it succeeded
            'combination': combination,
            'skipped': True
        }
    
    # Build command for running main.py
    cmd = [
        "python", "main.py",
        "--dataset", args.dataset,
        "--model_type", combination['model_type'],
        "--feature_dimension", str(combination['feature_dimension']),
        "--learning_rate", str(combination['learning_rate']),
        "--l1_lambda", str(combination['l1_lambda']),
        "--model_id", model_id,
        "--activation", combination['activation'],
        "--save_config"  # Save configuration for later reference
    ]
    
    # Add attention function if using ST model
    if combination['model_type'] in ['st', 'both']:
        cmd.extend(["--attention_fn", combination['attention_fn']])
        
        # Add attention dimension if specified
        if combination['attention_dimension']:
            cmd.extend(["--attention_dimension", str(combination['attention_dimension'])])
    
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
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    # Record start time
    start_time = time.time()
    
    # Run the training process
    result = subprocess.run(cmd, check=False)
    
    # Record end time and duration
    end_time = time.time()
    duration = end_time - start_time
    
    # Move model files to experiment directory if specified
    if experiment_dir:
        exp_models_dir = os.path.join(experiment_dir, 'models')
        os.makedirs(exp_models_dir, exist_ok=True)
        
        # Copy model files
        for model_type in ['sae', 'st']:
            src_path = f"models/{model_type}_model_{model_id}.pth"
            if os.path.exists(src_path):
                dst_path = os.path.join(exp_models_dir, os.path.basename(src_path))
                shutil.copy2(src_path, dst_path)
        
        # Copy config file
        if os.path.exists("last_config.json"):
            dst_path = os.path.join(experiment_dir, f"config_{model_id}.json")
            shutil.copy2("last_config.json", dst_path)
    
    return {
        'model_id': model_id,
        'returncode': result.returncode,
        'combination': combination,
        'duration': duration,
        'skipped': False
    }

def extract_metrics(model_id, args, experiment_dir=None):
    """Extract metrics from the trained model's checkpoint file"""
    metrics = {
        'val_loss': float('inf'),
        'dead_ratio': 1.0,
        'sparsity': 0.0,
        'step': 0,
        'found': False
    }
    
    # Check in experiment directory first if provided
    if experiment_dir:
        st_path = os.path.join(experiment_dir, 'models', f"st_model_{model_id}.pth")
        sae_path = os.path.join(experiment_dir, 'models', f"sae_model_{model_id}.pth")
    else:
        st_path = f"models/st_model_{model_id}.pth"
        sae_path = f"models/sae_model_{model_id}.pth"
    
    # Try to load the model file
    try:
        if os.path.exists(st_path):
            checkpoint = torch.load(st_path, map_location='cpu')
            if isinstance(checkpoint, dict):
                metrics['val_loss'] = checkpoint.get('val_loss', float('inf'))
                metrics['dead_ratio'] = checkpoint.get('dead_ratio', 1.0)
                metrics['sparsity'] = checkpoint.get('sparsity', 0.0)
                metrics['step'] = checkpoint.get('step', 0)
                metrics['found'] = True
                
                # Get training history if available
                if 'training_history' in checkpoint:
                    history = checkpoint['training_history']
                    if 'val_loss' in history and len(history['val_loss']) > 0:
                        metrics['final_val_loss'] = history['val_loss'][-1]
                        metrics['best_val_loss'] = min(history['val_loss'])
        
        elif os.path.exists(sae_path):
            checkpoint = torch.load(sae_path, map_location='cpu')
            if isinstance(checkpoint, dict):
                metrics['val_loss'] = checkpoint.get('val_loss', float('inf'))
                metrics['dead_ratio'] = checkpoint.get('dead_ratio', 1.0)
                metrics['step'] = checkpoint.get('step', 0)
                metrics['found'] = True
    except Exception as e:
        print(f"Error extracting metrics for {model_id}: {e}")
    
    return metrics

def calculate_combined_score(metrics):
    """Calculate a combined score based on multiple metrics"""
    # Normalize values to [0, 1] range for each metric
    # Lower is better for val_loss and dead_ratio
    # Higher is better for sparsity
    
    # Weight factors (can be adjusted)
    val_loss_weight = 0.6
    dead_ratio_weight = 0.2
    sparsity_weight = 0.2
    
    # Cap values to reasonable bounds
    val_loss = min(metrics.get('val_loss', float('inf')), 100)
    dead_ratio = min(metrics.get('dead_ratio', 1.0), 1.0)
    sparsity = max(min(metrics.get('sparsity', 0.0), 1.0), 0.0)
    
    # For val_loss, use a sigmoid-like scaling (assuming typical values are between 0-10)
    val_loss_score = 1.0 - (val_loss / (val_loss + 5.0))
    
    # For dead_ratio, 0 is best and 1 is worst
    dead_ratio_score = 1.0 - dead_ratio
    
    # For sparsity, higher is generally better but extremely high values might indicate issues
    # Optimal range might be around 0.8-0.95
    if sparsity > 0.95:
        sparsity_score = 0.9 - (sparsity - 0.95) * 2  # Penalty for too high sparsity
    else:
        sparsity_score = sparsity
    
    # Calculate weighted score
    combined_score = (
        val_loss_weight * val_loss_score + 
        dead_ratio_weight * dead_ratio_score +
        sparsity_weight * sparsity_score
    )
    
    return combined_score

def organize_results(results, args):
    """Organize trained models into folders based on their performance and hierarchical structure"""
    if not args.experiment_name and not args.continue_from:
        # Create timestamp-based experiment name if not provided
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"experiment_{timestamp}"
    else:
        experiment_name = args.experiment_name or os.path.basename(args.continue_from)
    
    # Use continue_from directory if provided, otherwise create new
    if args.continue_from and os.path.isdir(args.continue_from):
        experiment_dir = args.continue_from
    else:
        experiment_dir = os.path.join(args.output_dir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join(experiment_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Create hierarchical directory structure
    hierarchy_dir = os.path.join(experiment_dir, 'hierarchy')
    os.makedirs(hierarchy_dir, exist_ok=True)
    
    # Extract metrics for all runs
    print("\nExtracting performance metrics for all models...")
    for result in results:
        metrics = extract_metrics(result['model_id'], args, experiment_dir)
        result.update(metrics)
        
        if args.sort_metric == 'combined':
            result['combined_score'] = calculate_combined_score(metrics)
    
    # Create folders for organized results by performance
    group_names = ['top_performers', 'medium_performers', 'low_performers']
    if args.num_groups != 3:
        # Create generic names if not using the default 3 groups
        group_names = [f"group_{i+1}" for i in range(args.num_groups)]
    
    for name in group_names:
        group_dir = os.path.join(experiment_dir, name)
        os.makedirs(group_dir, exist_ok=True)
    
    # Remove failed and not found runs
    valid_results = [r for r in results if r['returncode'] == 0 and r.get('found', False)]
    skipped_results = [r for r in results if r.get('skipped', False)]
    failed_results = [r for r in results if r['returncode'] != 0]
    not_found_results = [r for r in results if r['returncode'] == 0 and not r.get('found', False)]
    
    if not valid_results:
        print("\nNo valid models found to organize!")
        
        # Still create a summary file for what happened
        summary_path = os.path.join(experiment_dir, "training_summary.md")
        with open(summary_path, 'w') as f:
            f.write("# Training Results Summary\n\n")
            f.write(f"Experiment: {experiment_name}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"Total combinations: {len(results)}\n")
            f.write(f"Skipped: {len(skipped_results)}\n")
            f.write(f"Failed: {len(failed_results)}\n")
            f.write(f"Not found: {len(not_found_results)}\n")
            
            if skipped_results:
                f.write("\n## Skipped Combinations\n\n")
                for r in skipped_results:
                    f.write(f"- {r['model_id']}\n")
            
            if failed_results:
                f.write("\n## Failed Combinations\n\n")
                for r in failed_results:
                    f.write(f"- {r['model_id']} (return code: {r['returncode']})\n")
        
        print(f"Summary written to {summary_path}")
        return experiment_dir
    
    # Sort by the chosen metric
    if args.sort_metric == 'val_loss':
        valid_results.sort(key=lambda x: x.get('val_loss', float('inf')))
    elif args.sort_metric == 'sparsity':
        # Higher sparsity is better, so sort in descending order
        valid_results.sort(key=lambda x: -x.get('sparsity', 0.0))
    elif args.sort_metric == 'dead_ratio':
        # Lower dead_ratio is better
        valid_results.sort(key=lambda x: x.get('dead_ratio', 1.0))
    elif args.sort_metric == 'combined':
        # Higher score is better
        valid_results.sort(key=lambda x: -x.get('combined_score', 0.0))
    
    # Divide into performance groups
    group_size = max(1, len(valid_results) // args.num_groups)
    performance_groups = []
    
    for i in range(args.num_groups):
        start_idx = i * group_size
        end_idx = (i + 1) * group_size if i < args.num_groups - 1 else len(valid_results)
        group = valid_results[start_idx:end_idx]
        
        if group:  # Only add non-empty groups
            performance_groups.append(group)
    
    # Create a summary file
    summary_path = os.path.join(experiment_dir, "training_summary.md")
    metrics_df_rows = []
    
    with open(summary_path, 'w') as f:
        f.write("# Training Results Summary\n\n")
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Sorted by: {args.sort_metric}\n\n")
        
        # Overview
        f.write("## Overview\n\n")
        f.write(f"Total combinations: {len(results)}\n")
        f.write(f"Successful: {len(valid_results)}\n")
        f.write(f"Skipped: {len(skipped_results)}\n")
        f.write(f"Failed: {len(failed_results)}\n")
        f.write(f"Not found: {len(not_found_results)}\n\n")
        
        # Hierarchical Structure Information
        f.write("## Hierarchical Organization\n\n")
        f.write("Models are organized in the following hierarchical structure:\n")
        f.write("```\n")
        f.write("hierarchy/\n")
        f.write("  ├── [model_type]/\n")
        f.write("  │   ├── [dataset]/\n")
        f.write("  │   │   ├── [activation_function]/\n")
        f.write("  │   │   │   ├── [feature_dimension]/\n")
        f.write("  │   │   │   │   ├── model_file_1.pth\n")
        f.write("  │   │   │   │   ├── model_file_2.pth\n")
        f.write("  │   │   │   │   └── ...\n")
        f.write("```\n\n")
        
        for i, (group, name) in enumerate(zip(performance_groups, group_names[:len(performance_groups)])):
            group_dir = os.path.join(experiment_dir, name)
            
            f.write(f"## Group {i+1}: {name}\n\n")
            f.write("| Model ID | Val Loss | Dead Ratio | Sparsity | Training Time |\n")
            f.write("|----------|----------|------------|----------|---------------|\n")
            
            for result in group:
                model_id = result['model_id']
                val_loss = result.get('val_loss', float('inf'))
                dead_ratio = result.get('dead_ratio', 1.0)
                sparsity = result.get('sparsity', 0.0)
                duration_hours = result.get('duration', 0) / 3600
                
                # Write to summary
                f.write(f"| {model_id} | {val_loss:.4f} | {dead_ratio:.2f} | {sparsity:.2f} | {duration_hours:.1f}h |\n")
                
                # Add to metrics dataframe
                # Create a comprehensive row with all parameters
                metrics_row = {
                    'model_id': model_id,
                    'group': name,
                    'val_loss': val_loss,
                    'dead_ratio': dead_ratio,
                    'sparsity': sparsity,
                    'training_time_hours': duration_hours,
                    'feature_dimension': result['combination']['feature_dimension'],
                    'model_type': result['combination']['model_type'],
                    'attention_fn': result['combination'].get('attention_fn', ''),
                    'activation': result['combination']['activation'],
                    'learning_rate': result['combination']['learning_rate'],
                    'l1_lambda': result['combination']['l1_lambda']
                }
                
                # Add all additional parameters
                for param, value in result['combination'].items():
                    if param not in metrics_row:
                        metrics_row[param] = value
                        
                metrics_df_rows.append(metrics_row)
                
                # Create symbolic links in the group directory
                for model_type_prefix in ['st', 'sae']:
                    source_model_path = os.path.join(models_dir, f"{model_type_prefix}_model_{model_id}.pth")
                    if os.path.exists(source_model_path):
                        # 1. Create a link in the performance group folder
                        dest_path = os.path.join(group_dir, f"{model_type_prefix}_model_{model_id}.pth")
                        # Create relative symlink
                        rel_path = os.path.relpath(source_model_path, os.path.dirname(dest_path))
                        if os.path.exists(dest_path):
                            os.remove(dest_path)
                        os.symlink(rel_path, dest_path)
                        
                        # 2. Create hierarchical directory structure and link
                        model_type = result['combination']['model_type']
                        dataset = args.dataset
                        activation = result['combination']['activation']
                        feature_dim = str(result['combination']['feature_dimension'])
                        
                        # Create hierarchical path
                        hier_path = os.path.join(
                            hierarchy_dir, 
                            model_type,
                            dataset,
                            activation,
                            feature_dim
                        )
                        os.makedirs(hier_path, exist_ok=True)
                        
                        # Create symbolic link in hierarchy
                        hier_dest_path = os.path.join(hier_path, f"{model_type_prefix}_model_{model_id}.pth")
                        # Create relative symlink
                        hier_rel_path = os.path.relpath(source_model_path, os.path.dirname(hier_dest_path))
                        if os.path.exists(hier_dest_path):
                            os.remove(hier_dest_path)
                        os.symlink(hier_rel_path, hier_dest_path)
            
            f.write("\n")
        
        # Add sections for skipped/failed if any
        if skipped_results:
            f.write("\n## Skipped Combinations\n\n")
            for r in skipped_results:
                f.write(f"- {r['model_id']}\n")
        
        if failed_results:
            f.write("\n## Failed Combinations\n\n")
            for r in failed_results:
                f.write(f"- {r['model_id']} (return code: {r['returncode']})\n")
    
    # Create a CSV file with metrics for easier analysis
    metrics_df = pd.DataFrame(metrics_df_rows)
    metrics_csv_path = os.path.join(experiment_dir, "metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    
    # Generate plots if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        # Create plots directory
        plots_dir = os.path.join(experiment_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot val_loss vs. feature_dimension
        plt.figure(figsize=(10, 6))
        for model_type in metrics_df['model_type'].unique():
            df_subset = metrics_df[metrics_df['model_type'] == model_type]
            plt.scatter(
                df_subset['feature_dimension'], 
                df_subset['val_loss'],
                label=model_type,
                alpha=0.7
            )
        plt.xlabel('Feature Dimension')
        plt.ylabel('Validation Loss')
        plt.title('Validation Loss vs Feature Dimension')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(plots_dir, 'val_loss_vs_dimension.png'), dpi=150)
        
        # Box plot of val_loss by model_type
        plt.figure(figsize=(10, 6))
        metrics_df.boxplot(column='val_loss', by='model_type', grid=False)
        plt.title('Validation Loss by Model Type')
        plt.suptitle('')  # Remove pandas default title
        plt.ylabel('Validation Loss')
        plt.savefig(os.path.join(plots_dir, 'val_loss_by_model_type.png'), dpi=150)
        
        # If there are ST models, plot by attention function
        if 'st' in metrics_df['model_type'].values:
            st_df = metrics_df[metrics_df['model_type'].isin(['st', 'both'])]
            if len(st_df['attention_fn'].unique()) > 1:
                plt.figure(figsize=(12, 6))
                st_df.boxplot(column='val_loss', by='attention_fn', grid=False)
                plt.title('Validation Loss by Attention Function')
                plt.suptitle('')
                plt.ylabel('Validation Loss')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'val_loss_by_attention_fn.png'), dpi=150)
    
    except Exception as e:
        print(f"Could not generate plots: {e}")
    
    print(f"\nResults organized into {len(performance_groups)} performance groups in {experiment_dir}")
    print(f"Hierarchical organization available in {os.path.join(experiment_dir, 'hierarchy')}")
    print(f"Summary written to {summary_path}")
    print(f"Metrics saved to {metrics_csv_path}")
    
    return experiment_dir

def main():
    """Main function to run training combinations"""
    args = parse_args()
    
    # Set up experiment directory
    if args.experiment_name and not args.continue_from:
        # Create new experiment directory
        experiment_dir = os.path.join(args.output_dir, args.experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
    elif args.continue_from:
        # Use existing directory
        experiment_dir = args.continue_from
        if not os.path.isdir(experiment_dir):
            print(f"Error: Specified continue_from directory {experiment_dir} does not exist")
            return
    else:
        # Will be created in organize_results
        experiment_dir = None
    
    # Save experiment configuration
    if experiment_dir:
        config_path = os.path.join(experiment_dir, "experiment_config.json")
        with open(config_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    
    # Generate all combinations to train
    combinations = generate_combinations(args)
    
    print(f"Generated {len(combinations)} combinations to train")
    
    # Show sample of combinations if there are many
    if len(combinations) > 10:
        print("\nSample of combinations to train:")
        for i, combo in enumerate(combinations[:5]):
            model_id = get_model_id(combo, args)
            print(f"{i+1}. {model_id}")
        print("...")
    
    # Train each combination
    results = []
    for idx, combo in enumerate(combinations):
        result = run_training(combo, args, idx, len(combinations), experiment_dir)
        results.append(result)
    
    # Organize results into performance-based folders
    final_dir = organize_results(results, args)
    
    print("\nExperiment completed!")
    print(f"Results available in: {final_dir}")

if __name__ == "__main__":
    main()