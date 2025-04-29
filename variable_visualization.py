import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import glob
import re
import torch
from typing import List, Optional, Dict, Union, Tuple
import warnings

def extract_model_info_from_files(model_dir: str, variable: str) -> pd.DataFrame:
    """
    Extract model information from trained model files and create a results dataframe.
    
    Args:
        model_dir: Base directory containing model files (models/sae, models/st)
        variable: Variable to extract (e.g., 'lambda_l1', 'feature_dim')
        
    Returns:
        DataFrame with model info and extracted variables
    """
    print(f"Extracting model information from directory: {model_dir}")
    
    results = []
    sae_dir = os.path.join(model_dir, 'sae')
    st_dir = os.path.join(model_dir, 'st')
    
    # First check if these directories exist
    dirs_to_check = []
    if os.path.exists(sae_dir):
        dirs_to_check.append(('sae', sae_dir))
    if os.path.exists(st_dir):
        dirs_to_check.append(('st', st_dir))
    
    if not dirs_to_check:
        print(f"No model directories found in {model_dir}. Please check the path.")
        # Try to find any model files in the directory itself
        model_files = glob.glob(os.path.join(model_dir, "layer_*_*.pt"))
        if model_files:
            print(f"Found {len(model_files)} model files directly in {model_dir}")
            dirs_to_check = [('unknown', model_dir)]
        else:
            print("No model files found.")
            return pd.DataFrame()
    
    # Process each directory (sae and st)
    for model_type, dir_path in dirs_to_check:
        model_files = glob.glob(os.path.join(dir_path, "layer_*_*.pt"))
        if not model_files:
            print(f"No model files found in {dir_path}")
            continue
        
        print(f"Found {len(model_files)} {model_type} model files")
        
        for model_file in model_files:
            try:
                # Extract layer number and model subtype from filename
                filename = os.path.basename(model_file)
                layer_match = re.search(r'layer_(\d+)_([a-z]+)\.pt', filename)
                
                if layer_match:
                    layer_idx = int(layer_match.group(1))
                    model_subtype = layer_match.group(2)
                else:
                    layer_idx = -1
                    model_subtype = "unknown"
                
                # Extract variable value from model checkpoint
                checkpoint = torch.load(model_file, map_location='cpu')
                
                # Different structures for different model types
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    # Enhanced format with full checkpoint info
                    value = None
                    
                    # Try to extract the variable from top-level keys
                    if variable in checkpoint:
                        value = checkpoint[variable]
                    # Try to extract from training_history if available
                    elif 'training_history' in checkpoint and variable in checkpoint['training_history']:
                        # Get the last value in the history
                        history = checkpoint['training_history'][variable]
                        if history and len(history) > 0:
                            value = history[-1]
                    
                    # Calculate or extract metrics
                    if value is not None:
                        # Get validation loss if available
                        val_loss = checkpoint.get('val_loss', 0)
                        dead_ratio = checkpoint.get('dead_ratio', 0)
                        
                        results.append({
                            'model_type': model_type,
                            'layer_idx': layer_idx,
                            'function_type': model_subtype,
                            variable: value,
                            'avg_centroid_distance': val_loss,  # Using val_loss as a proxy metric
                            'dead_ratio': dead_ratio,
                            'model_file': model_file,
                            'metric': 'val_loss'  # Setting metric to val_loss
                        })
                        
                elif isinstance(checkpoint, dict) and 'lambda_l1' in checkpoint:
                    # Simple format with direct parameter access
                    value = None
                    if variable == 'lambda_l1':
                        value = checkpoint['lambda_l1']
                    
                    # Try to extract common parameters from state dict
                    if value is None:
                        # For dimension parameters, try to infer from weights
                        if 'W_e.weight' in checkpoint:
                            if variable == 'feature_dim' or variable == 'm':
                                value = checkpoint['W_e.weight'].shape[0]
                            elif variable == 'input_dim' or variable == 'n':
                                value = checkpoint['W_e.weight'].shape[1]
                    
                    if value is not None:
                        results.append({
                            'model_type': model_type,
                            'layer_idx': layer_idx,
                            'function_type': model_subtype,
                            variable: value,
                            'avg_centroid_distance': 0,  # No metric available
                            'model_file': model_file,
                            'metric': 'unknown'
                        })
                
                else:
                    # State dictionary only format
                    # Try to extract structural parameters from weights
                    value = None
                    
                    # Common parameter name mappings
                    param_mappings = {
                        'lambda_l1': ['lambda_l1', 'l1_lambda'],
                        'feature_dim': ['m', 'feature_dim', 'feature_dimension'],
                        'input_dim': ['n', 'input_dim', 'input_dimension'],
                        'attention_dim': ['a', 'attention_dim', 'attention_dimension']
                    }
                    
                    # Try to find the variable in the checkpoint keys
                    var_options = [variable]
                    # Add alternate names if we know them
                    for key, options in param_mappings.items():
                        if variable in options:
                            var_options = options
                            break
                    
                    for var_name in var_options:
                        if var_name in checkpoint:
                            value = checkpoint[var_name]
                            break
                    
                    # For SAE models, try to infer dimensions from weight matrices
                    if value is None and model_type == 'sae':
                        if ('W_e.weight' in checkpoint) and (variable in ['feature_dim', 'm']):
                            value = checkpoint['W_e.weight'].shape[0]
                        elif ('W_e.weight' in checkpoint) and (variable in ['input_dim', 'n']):
                            value = checkpoint['W_e.weight'].shape[1]
                    
                    # For ST models, try to infer dimensions from weight matrices
                    if value is None and model_type == 'st':
                        if ('W_q.weight' in checkpoint) and (variable in ['attention_dim', 'a']):
                            value = checkpoint['W_q.weight'].shape[0]
                        elif ('W_q.weight' in checkpoint) and (variable in ['input_dim', 'n']):
                            value = checkpoint['W_q.weight'].shape[1]
                    
                    if value is not None:
                        results.append({
                            'model_type': model_type,
                            'layer_idx': layer_idx,
                            'function_type': model_subtype,
                            variable: value,
                            'avg_centroid_distance': 0,  # No metric available
                            'model_file': model_file,
                            'metric': 'structure'  # Metric based on model structure
                        })
            
            except Exception as e:
                print(f"Error processing {model_file}: {e}")
    
    # Convert to DataFrame
    if results:
        df = pd.DataFrame(results)
        print(f"Created results dataframe with {len(df)} rows and columns: {df.columns.tolist()}")
        return df
    else:
        print("No results extracted from model files.")
        return pd.DataFrame()

def load_and_combine_results(results_csv: str, variable: str = None, model_dir: str = None) -> pd.DataFrame:
    """
    Load results from a CSV file or directory with multiple dataset CSVs.
    If no CSV is found, try to extract info directly from model files.
    
    Args:
        results_csv: Path to CSV file or directory containing dataset CSVs
        variable: Variable to extract if generating from model files
        model_dir: Model directory to use if generating from model files
        
    Returns:
        Combined DataFrame with all results
    """
    # First try to load from CSV
    csv_loaded = False
    combined_df = None
    
    # Check if the path is a directory or file and try to load CSVs
    if os.path.exists(results_csv):
        if os.path.isdir(results_csv):
            print(f"Looking for results CSVs in directory: {results_csv}")
            # Look for dataset-specific CSV files
            dataset_csvs = glob.glob(os.path.join(results_csv, "dataset_*_results.csv"))
            
            # If no dataset CSVs found, check for any CSV files
            if not dataset_csvs:
                # Check for full_results.csv or any other CSV
                full_results = os.path.join(results_csv, "full_results.csv")
                if os.path.exists(full_results):
                    print(f"Loading full_results.csv")
                    combined_df = pd.read_csv(full_results)
                    csv_loaded = True
                else:
                    # Try to find any CSV file
                    any_csvs = glob.glob(os.path.join(results_csv, "*.csv"))
                    if any_csvs:
                        print(f"Found CSV file: {any_csvs[0]}")
                        combined_df = pd.read_csv(any_csvs[0])
                        csv_loaded = True
            else:
                # Load and combine all dataset CSVs
                print(f"Found {len(dataset_csvs)} dataset-specific CSV files")
                all_dfs = []
                
                for csv_path in dataset_csvs:
                    try:
                        dataset_name = os.path.basename(csv_path).replace("dataset_", "").replace("_results.csv", "")
                        print(f"Loading data for dataset: {dataset_name}")
                        df = pd.read_csv(csv_path)
                        
                        # Ensure dataset column exists and is correctly set
                        if 'dataset' not in df.columns:
                            df['dataset'] = dataset_name
                        
                        all_dfs.append(df)
                    except Exception as e:
                        print(f"Error loading {csv_path}: {e}")
                
                if all_dfs:
                    # Combine all dataframes
                    combined_df = pd.concat(all_dfs, ignore_index=True)
                    print(f"Combined {len(combined_df)} rows from {len(all_dfs)} datasets")
                    csv_loaded = True
                
        elif os.path.isfile(results_csv) and results_csv.endswith('.csv'):
            # Direct CSV file
            print(f"Loading results from file: {results_csv}")
            combined_df = pd.read_csv(results_csv)
            csv_loaded = True
    
    # If we couldn't load from CSV and have a model directory, try to generate from models
    if not csv_loaded and model_dir and variable:
        print(f"No valid CSV found. Trying to extract information from model files in {model_dir}...")
        combined_df = extract_model_info_from_files(model_dir, variable)
        
        # If we generated data, save it to a CSV for future use
        if not combined_df.empty:
            if os.path.isdir(results_csv):
                save_path = os.path.join(results_csv, f"{variable}_generated_results.csv")
            else:
                # Create a directory for the CSV if specified path doesn't exist
                os.makedirs(os.path.dirname(results_csv), exist_ok=True)
                save_path = results_csv
            
            combined_df.to_csv(save_path, index=False)
            print(f"Saved generated results to {save_path}")
    
    # If we still don't have data, raise an error
    if combined_df is None or combined_df.empty:
        # Just return an empty DataFrame instead of raising an error
        print("WARNING: No data could be loaded or generated. Creating empty DataFrame.")
        return pd.DataFrame()
    
    return combined_df

def create_variable_visualizations(results_csv: str, 
                                  variable: str, 
                                  output_dir: Optional[str] = None,
                                  secondary_variable: Optional[str] = None,
                                  metrics: Optional[List[str]] = None,
                                  dataset_filter: Optional[str] = None,
                                  model_dir: Optional[str] = None,
                                  default_dataset: str = "gpt_neo") -> List[str]:
    """
    Create visualizations comparing model centroid distances in relation to a specified variable.
    
    Args:
        results_csv: Path to CSV file with analysis results
        variable: The main variable to analyze (e.g., 'l1_lambda', 'feature_dimension')
        output_dir: Directory to save visualizations (default: visualization/{variable})
        secondary_variable: Optional second variable for more complex comparisons
        metrics: Which distance metrics to visualize (default: all available)
        dataset_filter: Optional dataset to filter by
        model_dir: Directory containing model files to extract info from if CSV not found
        default_dataset: Default dataset name to use if none is in the data
        
    Returns:
        List of generated visualization paths
    """
    # Load and combine results from CSV(s) or model files
    df = load_and_combine_results(results_csv, variable, model_dir)
    
    if df.empty:
        print("No data to visualize. Please check your inputs.")
        return []
    
    # Validate that the variable exists
    if variable not in df.columns:
        print(f"Error: Variable '{variable}' not found in the results data.")
        print(f"Available variables: {', '.join(df.columns)}")
        return []
    
    # Check secondary variable if provided
    if secondary_variable and secondary_variable not in df.columns:
        print(f"Warning: Secondary variable '{secondary_variable}' not found. Ignoring it.")
        secondary_variable = None
        
    # Determine metrics if not specified
    if metrics is None:
        if 'metric' in df.columns:
            metrics = df['metric'].unique().tolist()
        else:
            metrics = ['unknown']
    
    # Check if dataset column exists, otherwise add default
    has_dataset_column = 'dataset' in df.columns
    if not has_dataset_column:
        print(f"No dataset column found. Using default dataset name: {default_dataset}")
        df['dataset'] = default_dataset
        has_dataset_column = True
    
    visualizations = []
    
    # Handle output directory structure based on dataset information
    if has_dataset_column:
        # Create the base visualization directory
        base_dir = "visualization"
        if output_dir:
            base_dir = output_dir
        os.makedirs(base_dir, exist_ok=True)
        
        # Get unique datasets from dataframe
        all_datasets = sorted([d for d in df['dataset'].unique() if d])
        
        # Add default dataset if needed
        if not all_datasets:
            all_datasets = [default_dataset]
            df['dataset'] = default_dataset
            
        print(f"All available datasets: {', '.join(all_datasets)}")
        
        # Create root level info file
        root_info_path = os.path.join(base_dir, 'analysis_info.txt')
        with open(root_info_path, 'w') as f:
            f.write(f"""
Variable Analysis for: {variable}
===============================

The visualizations are organized in dataset-specific directories.
Please navigate to each dataset directory to see the visualizations.

Source data: {'Generated from model files' if model_dir else results_csv}
Metrics included: {', '.join(metrics)}
Secondary variable (if used): {secondary_variable or 'None'}
Datasets available: {', '.join(all_datasets)}
Analysis date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
""")
        visualizations.append(root_info_path)
        
        # If dataset filter is provided, analyze only that dataset
        if dataset_filter:
            if dataset_filter in all_datasets:
                datasets_to_analyze = [dataset_filter]
                print(f"Filtering to single dataset: {dataset_filter}")
            else:
                print(f"Dataset {dataset_filter} not found in the data. Available datasets: {all_datasets}")
                datasets_to_analyze = all_datasets
        else:
            datasets_to_analyze = all_datasets
            
        # Process each dataset
        for dataset in datasets_to_analyze:
            # Extract dataset data
            dataset_df = df[df['dataset'] == dataset].copy()
            if dataset_df.empty:
                print(f"No data for dataset: {dataset}. Creating placeholder.")
                # Create dataset directory structure: visualization/dataset/variable
                dataset_dir = os.path.join(base_dir, dataset)
                os.makedirs(dataset_dir, exist_ok=True)
                
                variable_dir = os.path.join(dataset_dir, variable)
                os.makedirs(variable_dir, exist_ok=True)
                
                # Create placeholder info file
                placeholder_path = os.path.join(variable_dir, "no_data.txt")
                with open(placeholder_path, 'w') as f:
                    f.write(f"No data available for dataset: {dataset}")
                visualizations.append(placeholder_path)
                continue
                
            # Create dataset directory structure: visualization/dataset/variable
            dataset_dir = os.path.join(base_dir, dataset)
            os.makedirs(dataset_dir, exist_ok=True)
            
            variable_dir = os.path.join(dataset_dir, variable)
            os.makedirs(variable_dir, exist_ok=True)
            
            # Create dataset-specific info file
            dataset_info_path = os.path.join(variable_dir, 'dataset_info.txt')
            with open(dataset_info_path, 'w') as f:
                f.write(f"""
Dataset: {dataset} - Variable: {variable}
=======================================

This directory contains visualizations showing the relationship between 
{variable} and centroid distances for the {dataset} dataset.

Number of models: {len(dataset_df)}
Metrics included: {', '.join(dataset_df['metric'].unique().tolist()) if 'metric' in dataset_df.columns else 'N/A'}
Analysis date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
""")
            visualizations.append(dataset_info_path)
            
            # Determine if variable is numeric or categorical for this dataset
            try:
                dataset_df[variable] = pd.to_numeric(dataset_df[variable])
                variable_type = 'numeric'
            except:
                variable_type = 'categorical'
                
            print(f"\nAnalyzing dataset: {dataset} - Variable: {variable} ({variable_type})")
            
            # Filter out any rows with NaN in the variable column
            dataset_df = dataset_df.dropna(subset=[variable])
            
            if dataset_df.empty:
                print(f"No valid data found for dataset {dataset} after filtering.")
                continue
                
            # Generate dataset-specific visualizations
            if variable_type == 'numeric':
                dataset_vis = create_numeric_variable_plots(
                    dataset_df, variable, variable_dir, metrics, 
                    secondary_variable, dataset_name=dataset)
                visualizations.extend(dataset_vis)
            else:
                dataset_vis = create_categorical_variable_plots(
                    dataset_df, variable, variable_dir, metrics, 
                    secondary_variable, dataset_name=dataset)
                visualizations.extend(dataset_vis)
            
            # Create general analysis plots for this dataset
            general_vis = create_general_analysis_plots(
                dataset_df, variable, variable_dir, metrics, 
                secondary_variable, dataset_name=dataset)
            visualizations.extend(general_vis)
            
    else:
        # No dataset column - use traditional structure
        if output_dir is None:
            output_dir = f"visualization/all_datasets/{variable}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create information file for single dataset case
        info_text = f"""
Variable Analysis for: {variable}
===============================

This directory contains visualizations showing the relationship between 
{variable} and centroid distances in your models.

The visualizations were generated from: {'Model files' if model_dir else results_csv}

Metrics included: {', '.join(metrics)}
Secondary variable (if used): {secondary_variable or 'None'}
Analysis date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        info_path = os.path.join(output_dir, 'analysis_info.txt')
        with open(info_path, 'w') as f:
            f.write(info_text)
        visualizations.append(info_path)
        
        # Determine if variable is numeric or categorical
        try:
            df[variable] = pd.to_numeric(df[variable])
            variable_type = 'numeric'
        except:
            variable_type = 'categorical'
        
        print(f"Analyzing relationship between '{variable}' ({variable_type}) and centroid distances")
        print(f"Using metrics: {metrics}")
        if secondary_variable:
            print(f"Using secondary variable: {secondary_variable}")
        
        # Filter out any rows with NaN in the variable column
        df = df.dropna(subset=[variable])
        
        if df.empty:
            print("No valid data found after filtering.")
            return visualizations
        
        # Generate visualizations based on variable type
        if variable_type == 'numeric':
            visualizations.extend(
                create_numeric_variable_plots(df, variable, output_dir, metrics, secondary_variable)
            )
        else:
            visualizations.extend(
                create_categorical_variable_plots(df, variable, output_dir, metrics, secondary_variable)
            )
        
        # Create additional general analysis plots (works for both types)
        visualizations.extend(
            create_general_analysis_plots(df, variable, output_dir, metrics, secondary_variable)
        )
    
    return visualizations

def create_numeric_variable_plots(df: pd.DataFrame, 
                                 variable: str, 
                                 output_dir: str, 
                                 metrics: List[str], 
                                 secondary_variable: Optional[str] = None,
                                 dataset_name: Optional[str] = None) -> List[str]:
    """
    Create plots specifically for numeric variables.
    """
    visualizations = []
    
    # Add dataset suffix for titles if provided
    dataset_suffix = f" - {dataset_name}" if dataset_name else ""
    
    # Add dataset prefix for filenames if provided
    file_prefix = f"{dataset_name}_" if dataset_name else ""
    
    # Process for each metric
    for metric in metrics:
        try:
            # Filter data for this metric
            if 'metric' in df.columns:
                metric_df = df[df['metric'] == metric].copy()
                metric_suffix = f"_{metric}"
            else:
                metric_df = df.copy()
                metric_suffix = ""
            
            if metric_df.empty:
                continue
                
            # 1. Scatter plot of variable vs centroid distance
            plt.figure(figsize=(12, 8))
            
            if secondary_variable and secondary_variable in metric_df.columns:
                # Use secondary variable for coloring
                scatter = plt.scatter(
                    metric_df[variable],
                    metric_df['avg_centroid_distance'],
                    c=pd.to_numeric(metric_df[secondary_variable], errors='ignore'),
                    cmap='viridis',
                    alpha=0.7,
                    s=80
                )
                plt.colorbar(label=secondary_variable)
            elif 'model_type' in metric_df.columns:
                # Use model_type for coloring
                sns.scatterplot(
                    data=metric_df,
                    x=variable,
                    y='avg_centroid_distance',
                    hue='model_type',
                    style='model_type' if metric_df['model_type'].nunique() <= 5 else None,
                    s=80,
                    alpha=0.7
                )
                plt.legend(title='Model Type')
            else:
                # Simple scatter plot
                plt.scatter(
                    metric_df[variable],
                    metric_df['avg_centroid_distance'],
                    alpha=0.7,
                    s=80
                )
            
            plt.title(f'Average Centroid Distance vs {variable.replace("_", " ").title()}{metric_suffix}{dataset_suffix}')
            plt.xlabel(variable.replace('_', ' ').title())
            plt.ylabel('Average Centroid Distance')
            plt.grid(alpha=0.3)
            
            scatter_path = os.path.join(output_dir, f'{file_prefix}scatter_{variable}{metric_suffix}.png')
            plt.savefig(scatter_path, dpi=300)
            visualizations.append(scatter_path)
            plt.close()
            
            # 2. Line plot showing trends by model/function type
            plt.figure(figsize=(14, 8))
            
            if 'model_type' in metric_df.columns and 'function_type' in metric_df.columns:
                # Group by model_type, function_type, and the variable
                grouped = metric_df.groupby(['model_type', 'function_type', variable])['avg_centroid_distance'].mean().reset_index()
                
                # Plot each model/function combination
                for (model, function), group in grouped.groupby(['model_type', 'function_type']):
                    # Sort by variable for proper line plotting
                    group = group.sort_values(variable)
                    plt.plot(
                        group[variable],
                        group['avg_centroid_distance'],
                        'o-',
                        label=f"{model} - {function}",
                        linewidth=2,
                        markersize=8,
                        alpha=0.8
                    )
                
                plt.legend(title='Model - Function')
            else:
                # Simple trend line
                grouped = metric_df.groupby(variable)['avg_centroid_distance'].mean().reset_index()
                plt.plot(
                    grouped[variable],
                    grouped['avg_centroid_distance'],
                    'o-',
                    linewidth=2,
                    markersize=8
                )
            
            plt.title(f'Centroid Distance Trend by {variable.replace("_", " ").title()}{metric_suffix}{dataset_suffix}')
            plt.xlabel(variable.replace('_', ' ').title())
            plt.ylabel('Average Centroid Distance')
            plt.grid(alpha=0.3)
            
            trend_path = os.path.join(output_dir, f'{file_prefix}trend_{variable}{metric_suffix}.png')
            plt.savefig(trend_path, dpi=300)
            visualizations.append(trend_path)
            plt.close()
            
            # 3. Box plot showing distribution by variable
            plt.figure(figsize=(14, 8))
            
            # Check if the variable has too many unique values for a good boxplot
            unique_values = metric_df[variable].nunique()
            
            if unique_values <= 10:  # Only do boxplot if reasonable number of groups
                # Create box plot
                if 'model_type' in metric_df.columns:
                    sns.boxplot(
                        x=variable,
                        y='avg_centroid_distance',
                        hue='model_type',
                        data=metric_df
                    )
                    plt.legend(title='Model Type')
                else:
                    sns.boxplot(
                        x=variable,
                        y='avg_centroid_distance',
                        data=metric_df
                    )
                
                plt.title(f'Distribution of Centroid Distances by {variable.replace("_", " ").title()}{metric_suffix}{dataset_suffix}')
                plt.xlabel(variable.replace('_', ' ').title())
                plt.ylabel('Average Centroid Distance')
                plt.grid(alpha=0.3, axis='y')
                
                box_path = os.path.join(output_dir, f'{file_prefix}boxplot_{variable}{metric_suffix}.png')
                plt.savefig(box_path, dpi=300)
                visualizations.append(box_path)
            else:
                # For too many unique values, create a binned histogram instead
                plt.close()
            
            plt.close()
            
            # 4. Regression plot if seaborn's regplot is available
            try:
                plt.figure(figsize=(12, 8))
                
                if 'model_type' in metric_df.columns:
                    # Create separate regression lines for each model type
                    model_types = metric_df['model_type'].unique()
                    for i, model_type in enumerate(model_types):
                        model_data = metric_df[metric_df['model_type'] == model_type]
                        sns.regplot(
                            x=variable,
                            y='avg_centroid_distance',
                            data=model_data,
                            scatter=True,
                            label=model_type,
                            scatter_kws={'s': 80, 'alpha': 0.6},
                            line_kws={'linewidth': 3}
                        )
                    plt.legend(title='Model Type')
                else:
                    # Single regression line
                    sns.regplot(
                        x=variable,
                        y='avg_centroid_distance',
                        data=metric_df,
                        scatter=True,
                        scatter_kws={'s': 80, 'alpha': 0.6},
                        line_kws={'linewidth': 3}
                    )
                
                plt.title(f'Regression: Centroid Distance vs {variable.replace("_", " ").title()}{metric_suffix}{dataset_suffix}')
                plt.xlabel(variable.replace('_', ' ').title())
                plt.ylabel('Average Centroid Distance')
                plt.grid(alpha=0.3)
                
                reg_path = os.path.join(output_dir, f'{file_prefix}regression_{variable}{metric_suffix}.png')
                plt.savefig(reg_path, dpi=300)
                visualizations.append(reg_path)
                plt.close()
            except Exception as e:
                print(f"Error creating regression plot: {e}")
        
        except Exception as e:
            print(f"Error creating plots for metric {metric}: {e}")
    
    return visualizations

def create_categorical_variable_plots(df: pd.DataFrame, 
                                     variable: str, 
                                     output_dir: str, 
                                     metrics: List[str],
                                     secondary_variable: Optional[str] = None,
                                     dataset_name: Optional[str] = None) -> List[str]:
    """
    Create plots specifically for categorical variables.
    """
    visualizations = []
    
    # Add dataset suffix for titles if provided
    dataset_suffix = f" - {dataset_name}" if dataset_name else ""
    
    # Add dataset prefix for filenames if provided
    file_prefix = f"{dataset_name}_" if dataset_name else ""
    
    # Process for each metric
    for metric in metrics:
        try:
            # Filter data for this metric
            if 'metric' in df.columns:
                metric_df = df[df['metric'] == metric].copy()
                metric_suffix = f"_{metric}"
            else:
                metric_df = df.copy()
                metric_suffix = ""
            
            if metric_df.empty:
                continue
                
            # 1. Bar plot of average centroid distance by variable categories
            plt.figure(figsize=(14, 8))
            
            if 'model_type' in metric_df.columns:
                # Group by model_type and the variable
                grouped = metric_df.groupby(['model_type', variable])['avg_centroid_distance'].mean().reset_index()
                
                # Plot grouped bar chart
                sns.barplot(
                    x=variable,
                    y='avg_centroid_distance',
                    hue='model_type',
                    data=grouped,
                    alpha=0.7
                )
                plt.legend(title='Model Type')
            else:
                # Simple bar chart
                grouped = metric_df.groupby(variable)['avg_centroid_distance'].mean().reset_index()
                sns.barplot(
                    x=variable,
                    y='avg_centroid_distance',
                    data=grouped,
                    alpha=0.7
                )
            
            plt.title(f'Average Centroid Distance by {variable.replace("_", " ").title()}{metric_suffix}{dataset_suffix}')
            plt.xlabel(variable.replace('_', ' ').title())
            plt.ylabel('Average Centroid Distance')
            plt.grid(alpha=0.3, axis='y')
            
            # Rotate x-axis labels if needed
            if len(metric_df[variable].unique()) > 3:
                plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            bar_path = os.path.join(output_dir, f'{file_prefix}barplot_{variable}{metric_suffix}.png')
            plt.savefig(bar_path, dpi=300)
            visualizations.append(bar_path)
            plt.close()
            
            # 2. Box plot showing distribution by variable
            plt.figure(figsize=(14, 8))
            
            if secondary_variable and secondary_variable in metric_df.columns:
                # Include secondary variable in the visualization
                sns.boxplot(
                    x=variable,
                    y='avg_centroid_distance',
                    hue=secondary_variable,
                    data=metric_df
                )
                plt.legend(title=secondary_variable.replace('_', ' ').title())
            else:
                # Simple box plot
                sns.boxplot(
                    x=variable,
                    y='avg_centroid_distance',
                    data=metric_df
                )
            
            plt.title(f'Distribution of Centroid Distances by {variable.replace("_", " ").title()}{metric_suffix}{dataset_suffix}')
            plt.xlabel(variable.replace('_', ' ').title())
            plt.ylabel('Average Centroid Distance')
            plt.grid(alpha=0.3, axis='y')
            
            # Rotate x-axis labels if needed
            if len(metric_df[variable].unique()) > 3:
                plt.xticks(rotation=45, ha='right')
                
            plt.tight_layout()
            box_path = os.path.join(output_dir, f'{file_prefix}boxplot_{variable}{metric_suffix}.png')
            plt.savefig(box_path, dpi=300)
            visualizations.append(box_path)
            plt.close()
            
            # 3. Violin plot for more detailed distribution visualization
            try:
                plt.figure(figsize=(14, 8))
                
                if 'model_type' in metric_df.columns:
                    # Split by model type
                    sns.violinplot(
                        x=variable,
                        y='avg_centroid_distance',
                        hue='model_type',
                        data=metric_df,
                        split=True if metric_df['model_type'].nunique() == 2 else False,
                        inner='quartile'
                    )
                    plt.legend(title='Model Type')
                else:
                    # Simple violin plot
                    sns.violinplot(
                        x=variable,
                        y='avg_centroid_distance',
                        data=metric_df,
                        inner='quartile'
                    )
                
                plt.title(f'Distribution Detail of Centroid Distances by {variable.replace("_", " ").title()}{metric_suffix}{dataset_suffix}')
                plt.xlabel(variable.replace('_', ' ').title())
                plt.ylabel('Average Centroid Distance')
                plt.grid(alpha=0.3, axis='y')
                
                # Rotate x-axis labels if needed
                if len(metric_df[variable].unique()) > 3:
                    plt.xticks(rotation=45, ha='right')
                    
                plt.tight_layout()
                violin_path = os.path.join(output_dir, f'{file_prefix}violin_{variable}{metric_suffix}.png')
                plt.savefig(violin_path, dpi=300)
                visualizations.append(violin_path)
                plt.close()
            except Exception as e:
                print(f"Error creating violin plot: {e}")
                
        except Exception as e:
            print(f"Error creating plots for metric {metric}: {e}")
    
    return visualizations

def create_general_analysis_plots(df: pd.DataFrame, 
                                 variable: str, 
                                 output_dir: str, 
                                 metrics: List[str],
                                 secondary_variable: Optional[str] = None,
                                 dataset_name: Optional[str] = None) -> List[str]:
    """
    Create general analysis plots that work for both numeric and categorical variables.
    """
    visualizations = []
    
    # Add dataset suffix for titles if provided
    dataset_suffix = f" - {dataset_name}" if dataset_name else ""
    
    # Add dataset prefix for filenames if provided
    file_prefix = f"{dataset_name}_" if dataset_name else ""
    
    # Process for each metric
    for metric in metrics:
        try:
            # Filter data for this metric
            if 'metric' in df.columns:
                metric_df = df[df['metric'] == metric].copy()
                metric_suffix = f"_{metric}"
            else:
                metric_df = df.copy()
                metric_suffix = ""
            
            if metric_df.empty:
                continue
            
            # 1. Heatmap of variable and function_type/model_type if available
            if 'function_type' in metric_df.columns:
                plt.figure(figsize=(12, 10))
                
                # Create pivot table for heatmap
                try:
                    pivot_table = metric_df.pivot_table(
                        values='avg_centroid_distance',
                        index=['model_type', 'function_type'] if 'model_type' in metric_df.columns else 'function_type',
                        columns=[variable],
                        aggfunc='mean'
                    )
                    
                    # Create heatmap
                    sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt='.3f')
                    plt.title(f'Centroid Distance by Function Type and {variable.replace("_", " ").title()}{metric_suffix}{dataset_suffix}')
                    plt.tight_layout()
                    
                    heatmap_path = os.path.join(output_dir, f'{file_prefix}heatmap_{variable}{metric_suffix}.png')
                    plt.savefig(heatmap_path, dpi=300)
                    visualizations.append(heatmap_path)
                    plt.close()
                except Exception as e:
                    print(f"Error creating heatmap: {e}")
            
            # 2. Top and bottom models by variable value
            if 'model_type' in metric_df.columns and 'function_type' in metric_df.columns:
                plt.figure(figsize=(14, 10))
                
                # Get top 5 and bottom 5 models by centroid distance
                top_models = metric_df.sort_values('avg_centroid_distance', ascending=False).head(5)
                bottom_models = metric_df.sort_values('avg_centroid_distance').head(5)
                
                # Combine into one DataFrame
                compare_models = pd.concat([top_models, bottom_models])
                
                # Create model labels
                compare_models['model_label'] = compare_models.apply(
                    lambda x: f"{x['model_type']}-{x['function_type']} ({x[variable]})",
                    axis=1
                )
                
                # Create color mapping based on model type
                colors = ['#1f77b4' if m == 'sae' else '#2ca02c' for m in compare_models['model_type']]
                
                # Horizontal bar chart
                bars = plt.barh(
                    compare_models['model_label'],
                    compare_models['avg_centroid_distance'],
                    color=colors,
                    alpha=0.7
                )
                
                # Add value annotations
                for i, bar in enumerate(bars):
                    plt.text(
                        bar.get_width() + 0.01,
                        bar.get_y() + bar.get_height()/2,
                        f"{compare_models['avg_centroid_distance'].iloc[i]:.3f}",
                        va='center'
                    )
                
                plt.title(f'Top and Bottom Models by Centroid Distance ({variable} comparison){metric_suffix}{dataset_suffix}')
                plt.xlabel('Average Centroid Distance')
                plt.ylabel('Model Configuration')
                plt.grid(alpha=0.3, axis='x')
                plt.tight_layout()
                
                # Add legend for model types
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='#1f77b4', alpha=0.7, label='SAE'),
                    Patch(facecolor='#2ca02c', alpha=0.7, label='ST')
                ]
                plt.legend(handles=legend_elements, loc='lower right')
                
                # Save
                ranking_path = os.path.join(output_dir, f'{file_prefix}top_bottom_{variable}{metric_suffix}.png')
                plt.savefig(ranking_path, dpi=300)
                visualizations.append(ranking_path)
                plt.close()
                
            # 3. Variable impact summary
            if len(metric_df) >= 5:  # Only create if we have enough data
                try:
                    plt.figure(figsize=(10, 8))
                    
                    # Group by the variable and calculate average centroid distance
                    if metric_df[variable].dtype.name != 'category':
                        impact_df = metric_df.groupby(variable)['avg_centroid_distance'].mean().reset_index()
                        impact_df = impact_df.sort_values('avg_centroid_distance', ascending=False)
                        
                        # Create bar chart of variable impact
                        sns.barplot(
                            data=impact_df,
                            x=variable,
                            y='avg_centroid_distance',
                            palette='viridis'
                        )
                        
                        plt.title(f'Impact of {variable.replace("_", " ").title()} on Class Separation{metric_suffix}{dataset_suffix}')
                        plt.xlabel(variable.replace('_', ' ').title())
                        plt.ylabel('Average Centroid Distance')
                        plt.grid(alpha=0.3, axis='y')
                        
                        # Rotate x-axis labels if needed
                        if len(impact_df) > 3:
                            plt.xticks(rotation=45, ha='right')
                            
                        plt.tight_layout()
                        impact_path = os.path.join(output_dir, f'{file_prefix}impact_{variable}{metric_suffix}.png')
                        plt.savefig(impact_path, dpi=300)
                        visualizations.append(impact_path)
                    plt.close()
                except Exception as e:
                    print(f"Error creating impact summary: {e}")
                    plt.close()
                
        except Exception as e:
            print(f"Error creating general plots for metric {metric}: {e}")
    
    return visualizations

def main():
    parser = argparse.ArgumentParser(description='Create visualizations for a specific variable')
    parser.add_argument('variable', type=str, 
                      help='Variable to analyze (e.g., l1_lambda, feature_dimension)')
    parser.add_argument('--results_csv', type=str, default='model_comparison/full_results.csv',
                      help='Path to CSV file with analysis results')
    parser.add_argument('--output_dir', type=str, default=None,
                      help='Directory to save visualizations (default: visualization/dataset/variable)')
    parser.add_argument('--metrics', type=str, nargs='+', default=None,
                      help='Distance metrics to use (default: all available)')
    parser.add_argument('--secondary', type=str, default=None,
                      help='Secondary variable for additional insight')
    parser.add_argument('--dataset', type=str, default=None,
                      help='Filter to a specific dataset')
    parser.add_argument('--model_dir', type=str, default='models',
                      help='Directory containing trained models')
    parser.add_argument('--default_dataset', type=str, default='gpt_neo',
                      help='Default dataset name to use if none is provided')
    
    args = parser.parse_args()
    
    # Suppress common warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    # Create visualizations
    print(f"Creating visualizations for variable: {args.variable}")
    visualizations = create_variable_visualizations(
        args.results_csv, 
        args.variable,
        args.output_dir,
        args.secondary,
        args.metrics,
        args.dataset,
        args.model_dir,
        args.default_dataset
    )
    
    # Print results summary
    base_dir = args.output_dir or "visualization"
    if args.dataset:
        output_dir = f"{base_dir}/{args.dataset}/{args.variable}"
    else:
        output_dir = f"{base_dir}/gpt_neo/{args.variable}"
    
    print(f"\nGenerated {len(visualizations)} visualizations for {args.variable}.")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()