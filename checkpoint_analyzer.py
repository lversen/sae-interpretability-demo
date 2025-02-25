import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from model_visualization import analyze_model_trajectory
import re
from typing import List, Dict, Tuple, Optional, Union, Any


def scan_checkpoints(model_dir: str, pattern: str = "*model*") -> Dict[str, List[str]]:
    """
    Scan directory for model checkpoints
    
    Args:
        model_dir: Directory containing model checkpoints
        pattern: Glob pattern to match checkpoint files
        
    Returns:
        Dictionary mapping model types to lists of checkpoint paths
    """
    # Get all files matching pattern
    checkpoint_files = glob.glob(os.path.join(model_dir, pattern))
    
    # Organize by model type
    checkpoints = {
        'sae': [],
        'st': [],
        'other': []
    }
    
    for filepath in checkpoint_files:
        if 'sae' in filepath.lower():
            checkpoints['sae'].append(filepath)
        elif 'st' in filepath.lower():
            checkpoints['st'].append(filepath)
        else:
            checkpoints['other'].append(filepath)
    
    # Sort by step number if possible
    for model_type in checkpoints:
        # Try to extract step number from filenames
        def get_step(filepath):
            match = re.search(r'step(\d+)', filepath)
            if match:
                return int(match.group(1))
            return 0
        
        checkpoints[model_type].sort(key=get_step)
    
    # Print summary
    print(f"Found {len(checkpoint_files)} checkpoint files:")
    for model_type, files in checkpoints.items():
        if files:
            print(f"  {model_type.upper()}: {len(files)} checkpoints")
            for file in files[:3]:
                print(f"    - {os.path.basename(file)}")
            if len(files) > 3:
                print(f"    - ... and {len(files)-3} more")
    
    return checkpoints


def extract_metrics_from_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """
    Extract training metrics from a model checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Dictionary of metrics
    """
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Check if it's a full checkpoint or just model state
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Extract metrics
            metrics = {
                'step': checkpoint.get('step', 0),
                'epoch': checkpoint.get('epoch', 0),
                'train_loss': checkpoint.get('train_loss', None),
                'val_loss': checkpoint.get('val_loss', None),
                'lambda_l1': checkpoint.get('lambda_l1', None),
                'dead_ratio': checkpoint.get('dead_ratio', None),
                'sparsity': None
            }
            
            # Extract sparsity if available
            if 'training_history' in checkpoint and 'sparsity' in checkpoint['training_history']:
                history = checkpoint['training_history']
                if len(history['sparsity']) > 0:
                    metrics['sparsity'] = history['sparsity'][-1][1]  # Last recorded sparsity
            
            return metrics
        else:
            # Just model state dict, try to extract step from filename
            match = re.search(r'step(\d+)', checkpoint_path)
            step = int(match.group(1)) if match else 0
            
            return {
                'step': step,
                'epoch': None,
                'train_loss': None,
                'val_loss': None,
                'lambda_l1': None,
                'dead_ratio': None,
                'sparsity': None
            }
    
    except Exception as e:
        print(f"Error extracting metrics from {checkpoint_path}: {e}")
        return {
            'step': 0,
            'epoch': None,
            'train_loss': None,
            'val_loss': None,
            'lambda_l1': None,
            'dead_ratio': None,
            'sparsity': None
        }


def analyze_training_progress(checkpoints: List[str], output_dir: Optional[str] = None):
    """
    Analyze training progress from a list of checkpoints
    
    Args:
        checkpoints: List of checkpoint paths
        output_dir: Directory to save plots to (optional)
    """
    if not checkpoints:
        print("No checkpoints to analyze.")
        return
    
    # Extract metrics from each checkpoint
    metrics_list = []
    
    for ckpt_path in checkpoints:
        metrics = extract_metrics_from_checkpoint(ckpt_path)
        metrics['path'] = ckpt_path
        metrics_list.append(metrics)
    
    # Create DataFrame
    metrics_df = pd.DataFrame(metrics_list)
    
    # Sort by step
    metrics_df = metrics_df.sort_values('step')
    
    # Print summary
    print("\nTraining Progress Summary:")
    if 'train_loss' in metrics_df and metrics_df['train_loss'].notna().any():
        print(f"Initial train loss: {metrics_df['train_loss'].iloc[0]:.4f}")
        print(f"Final train loss: {metrics_df['train_loss'].iloc[-1]:.4f}")
    
    if 'val_loss' in metrics_df and metrics_df['val_loss'].notna().any():
        print(f"Initial val loss: {metrics_df['val_loss'].iloc[0]:.4f}")
        print(f"Final val loss: {metrics_df['val_loss'].iloc[-1]:.4f}")
    
    if 'dead_ratio' in metrics_df and metrics_df['dead_ratio'].notna().any():
        print(f"Initial dead ratio: {metrics_df['dead_ratio'].iloc[0]:.2%}")
        print(f"Final dead ratio: {metrics_df['dead_ratio'].iloc[-1]:.2%}")
    
    # Create metrics dictionary for plotting
    plot_metrics = {
        'steps': metrics_df['step'].tolist(),
        'loss': metrics_df['train_loss'].tolist() if 'train_loss' in metrics_df else [],
        'val_loss': metrics_df['val_loss'].tolist() if 'val_loss' in metrics_df else [],
        'dead_ratio': metrics_df['dead_ratio'].tolist() if 'dead_ratio' in metrics_df else [],
        'lambda': metrics_df['lambda_l1'].tolist() if 'lambda_l1' in metrics_df else [],
    }
    
    if 'sparsity' in metrics_df and metrics_df['sparsity'].notna().any():
        plot_metrics['sparsity'] = metrics_df['sparsity'].tolist()
    
    # Create plots
    model_type = 'sae' if 'sae' in checkpoints[0].lower() else 'st'
    analyze_model_trajectory(checkpoints, is_sae=(model_type == 'sae'), metrics=plot_metrics)
    
    # Save DataFrame if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        metrics_df.to_csv(os.path.join(output_dir, f"{model_type}_training_metrics.csv"), index=False)
        print(f"Saved metrics to {os.path.join(output_dir, f'{model_type}_training_metrics.csv')}")
    
    return metrics_df


def compare_checkpoints(checkpoint_paths: List[str], sample_data, is_sae: bool = True):
    """
    Compare multiple checkpoint states on sample data
    
    Args:
        checkpoint_paths: List of checkpoint paths to compare
        sample_data: Sample data for evaluation
        is_sae: Whether the checkpoints are SAE (True) or ST (False) models
    """
    # Load each checkpoint and evaluate
    from model_visualization import visualize_model_ensemble
    
    # First, determine the model class and parameters
    if is_sae:
        from SAE import SparseAutoencoder as ModelClass
        model_params = {
            'n': sample_data.shape[1],
            'm': 8 * sample_data.shape[1],
            'sae_model_path': 'temp.pth',
            'lambda_l1': 5.0
        }
    else:
        from ST import SparseTransformer as ModelClass
        model_params = {
            'X': sample_data[:100],  # Just use a small subset for reference
            'n': sample_data.shape[1],
            'm': 8 * sample_data.shape[1],
            'a': sample_data.shape[1] // 2,
            'st_model_path': 'temp.pth',
            'lambda_l1': 5.0
        }
    
    # Create models dictionary
    models_dict = {}
    
    for ckpt_path in checkpoint_paths:
        # Extract step number from filename
        match = re.search(r'step(\d+)', ckpt_path)
        step = int(match.group(1)) if match else 0
        
        # Create model instance
        model = ModelClass(**model_params)
        
        try:
            # Try loading as state dict first
            state_dict = torch.load(ckpt_path)
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                model.load_state_dict(state_dict['model_state_dict'])
            else:
                model.load_state_dict(state_dict)
            
            # Add to models dictionary
            model_name = f"Step {step}"
            models_dict[model_name] = (model, is_sae)
        
        except Exception as e:
            print(f"Error loading checkpoint {ckpt_path}: {e}")
    
    # Visualize all models
    if models_dict:
        visualize_model_ensemble(models_dict, sample_data)
    else:
        print("No models could be loaded for comparison.")


def main():
    """Main function for checkpoint analysis"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze model checkpoints')
    parser.add_argument('--model_dir', type=str, default='models',
                       help='Directory containing model checkpoints')
    parser.add_argument('--pattern', type=str, default='*model*.step*',
                       help='Glob pattern to match checkpoint files')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save outputs to')
    parser.add_argument('--model_type', type=str, choices=['sae', 'st', 'both'],
                       default='both', help='Type of model to analyze')
    parser.add_argument('--compare', action='store_true',
                       help='Compare checkpoints on sample data')
    parser.add_argument('--max_checkpoints', type=int, default=5,
                       help='Maximum number of checkpoints to compare')
    
    args = parser.parse_args()
    
    # Scan for checkpoints
    checkpoints = scan_checkpoints(args.model_dir, args.pattern)
    
    # Analyze checkpoints
    if args.model_type == 'both':
        for model_type in ['sae', 'st']:
            if checkpoints[model_type]:
                print(f"\nAnalyzing {model_type.upper()} checkpoints:")
                metrics_df = analyze_training_progress(
                    checkpoints[model_type], args.output_dir)
                
                # Compare on sample data if requested
                if args.compare:
                    # Load sample data (for example, MNIST)
                    import pandas as pd
                    import numpy as np
                    
                    print(f"\nLoading sample data for {model_type.upper()} comparison...")
                    try:
                        # Try to load MNIST data
                        test_df = pd.read_csv('data/mnist_test.csv')
                        sample_data = test_df.iloc[:, 1:].values.astype(np.float32)
                        
                        # Select subset of checkpoints for comparison
                        compare_paths = checkpoints[model_type][-args.max_checkpoints:]
                        
                        print(f"Comparing {len(compare_paths)} checkpoints on sample data...")
                        compare_checkpoints(compare_paths, sample_data, is_sae=(model_type == 'sae'))
                    
                    except Exception as e:
                        print(f"Error loading sample data: {e}")
    else:
        if checkpoints[args.model_type]:
            print(f"\nAnalyzing {args.model_type.upper()} checkpoints:")
            metrics_df = analyze_training_progress(
                checkpoints[args.model_type], args.output_dir)
            
            # Compare on sample data if requested
            if args.compare:
                # Load sample data (for example, MNIST)
                import pandas as pd
                import numpy as np
                
                print(f"\nLoading sample data for {args.model_type.upper()} comparison...")
                try:
                    # Try to load MNIST data
                    test_df = pd.read_csv('data/mnist_test.csv')
                    sample_data = test_df.iloc[:, 1:].values.astype(np.float32)
                    
                    # Select subset of checkpoints for comparison
                    compare_paths = checkpoints[args.model_type][-args.max_checkpoints:]
                    
                    print(f"Comparing {len(compare_paths)} checkpoints on sample data...")
                    compare_checkpoints(compare_paths, sample_data, is_sae=(args.model_type == 'sae'))
                
                except Exception as e:
                    print(f"Error loading sample data: {e}")
        else:
            print(f"No {args.model_type.upper()} checkpoints found.")


if __name__ == "__main__":
    main()