#!/usr/bin/env python3
"""
Enhanced Organize Best Models from Centroid Analysis

This script finds the top-performing models based on centroid distances,
copies them to a dedicated folder structure, and provides detailed metrics information.
"""

import os
import shutil
import pandas as pd
import argparse
from tabulate import tabulate
import json
from pathlib import Path
import sys

def generate_gephi_graphs(best_models_df, output_dir, dataset_path=None, n_random_labels=10, 
                          gephi_subset_size=1000, graph_neighbors=4):
    """
    Generate Gephi graph visualizations for the best models.
    
    Args:
        best_models_df: DataFrame with the best models information
        output_dir: Base directory where models are copied
        dataset_path: Path to the dataset file for feature extraction
        n_random_labels: Number of random labels for the graph visualization
        gephi_subset_size: Size of the subset for Gephi visualization
        graph_neighbors: Number of neighbors for graph creation
        
    Returns:
        List of paths to generated graph files
    """
    if best_models_df is None or best_models_df.empty:
        print("No models to create graphs for.")
        return []
    
    # Create graphs subfolder
    graphs_dir = os.path.join(output_dir, "graphs")
    os.makedirs(graphs_dir, exist_ok=True)
    
    # Import necessary modules
    try:
        from sample_handler import get_consistent_samples
        from gephi import create_gephi_graph, select_random_labels
        import torch
        import pandas as pd
        from SAE import SparseAutoencoder
        from ST import SparseTransformer
    except ImportError as e:
        print(f"Error importing required modules for graph generation: {e}")
        print("Skipping graph generation.")
        return []
    
    # Load dataset if provided
    if dataset_path and os.path.exists(dataset_path):
        try:
            print(f"Loading dataset from {dataset_path}")
            df = pd.read_csv(dataset_path)
            
            # Determine label column
            label_column = None
            for col in ['label', 'labels', 'target', 'class']:
                if col in df.columns:
                    label_column = col
                    break
            
            if not label_column:
                print("Could not determine label column. Using first column.")
                label_column = df.columns[0]
                
            # Determine feature columns (all except label column)
            feature_columns = [col for col in df.columns if col != label_column]
            
            # Get a consistent sample
            sample_df, _ = get_consistent_samples(df, min(gephi_subset_size*2, len(df)), 
                                               "graph_sample", "graph_generation")
            
            # Extract features
            features = sample_df[feature_columns].values
            
            print(f"Loaded {len(sample_df)} samples with {len(feature_columns)} features")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Skipping graph generation.")
            return []
    else:
        print("No valid dataset provided. Skipping graph generation.")
        return []
    
    # Track generated graph files
    generated_graphs = []
    
    # Device selection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device} for graph generation")
    
    # Process each model
    for idx, row in best_models_df.iterrows():
        model_path = row['model_path']
        model_type = row['model_type']
        feature_dim = int(row.get('feature_dimension', 100))
        
        # Skip if model file doesn't exist
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            continue
        
        try:
            print(f"Processing model: {model_path}")
            
            # First, load the checkpoint to inspect dimensions
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Create feature tensor for model initialization
            X_tensor = torch.tensor(features, dtype=torch.float32).to(device)
            
            # Load the appropriate model type
            if model_type.lower() == 'sae':
                # Load SAE model
                model = SparseAutoencoder(
                    n=features.shape[1],
                    m=feature_dim,
                    sae_model_path=model_path,
                    device=device
                )
                
                # Load weights
                model.load_state_dict(state_dict)
                
                # Extract features
                with torch.no_grad():
                    _, _, feature_activations = model(X_tensor)
                    feature_extract = feature_activations.cpu().numpy()
            else:  # ST model
                # Extract the correct attention dimension from checkpoint
                attention_dim = None
                
                # Try to determine from weights in the checkpoint
                if 'W_q.weight' in state_dict:
                    attention_dim = state_dict['W_q.weight'].shape[0]
                    print(f"Detected attention dimension from checkpoint: {attention_dim}")
                elif 'W_k_direct' in state_dict and state_dict['W_k_direct'].dim() > 1:
                    attention_dim = state_dict['W_k_direct'].shape[1]
                    print(f"Detected attention dimension from W_k_direct: {attention_dim}")
                
                # If we still don't have a valid dimension, use a fallback
                if attention_dim is None or attention_dim <= 0:
                    attention_dim = feature_dim // 4
                    print(f"Using default attention dimension: {attention_dim}")
                
                # Load ST model with correct attention dimension
                model = SparseTransformer(
                    X=X_tensor,
                    n=features.shape[1],
                    m=feature_dim,
                    a=attention_dim,  # Use the extracted attention dimension
                    st_model_path=model_path,
                    device=device
                )
                
                # Load weights
                model.load_state_dict(state_dict)
                
                # Extract features
                with torch.no_grad():
                    _, _, feature_activations, _ = model(X_tensor)
                    feature_extract = feature_activations.cpu().numpy()
            
            # Create descriptive filename for the graph with more distinguishing information
            if 'dataset' in row:
                # Start with basic info
                base_name = f"{row['dataset']}_{model_type}_{row['function_type']}_{feature_dim}"
                
                # Add learning rate if available
                if 'learning_rate' in row:
                    lr_str = str(row['learning_rate']).replace('.', 'p')
                    base_name += f"_lr{lr_str}"
                
                # Add L1 lambda if available
                if 'l1_lambda' in row:
                    l1_str = str(row['l1_lambda']).replace('.', 'p')
                    base_name += f"_l1{l1_str}"
                
                # Add steps or avg_centroid_distance for uniqueness
                if 'steps' in row:
                    base_name += f"_steps{row['steps']}"
                elif 'avg_centroid_distance' in row:
                    dist_str = f"{row['avg_centroid_distance']:.4f}".replace('.', 'p')
                    base_name += f"_dist{dist_str}"
                
                # Add index from DataFrame for further uniqueness
                base_name += f"_idx{idx}"
            else:
                # Fallback with similar pattern
                base_name = f"{model_type}_{row['function_type']}_{feature_dim}_idx{idx}"

            # Create full graph file path
            graph_file = os.path.join(graphs_dir, f"{base_name}.gexf")

            # Ensure filename isn't too long (filesystem limits)
            if len(os.path.basename(graph_file)) > 240:  # Leave room for path
                # Create a hash of the original name for uniqueness
                import hashlib
                name_hash = hashlib.md5(base_name.encode()).hexdigest()[:10]
                short_base_name = f"{row.get('dataset', 'model')}_{model_type}_{feature_dim}_idx{idx}_{name_hash}"
                graph_file = os.path.join(graphs_dir, f"{short_base_name}.gexf")
            
            # Select random labels for visualization
            selected_labels = select_random_labels(
                sample_df, 
                title_column=label_column, 
                n_random_labels=n_random_labels,
                category_column=label_column
            )
            
            print(f"Creating graph visualization for {base_name}...")
            
            # Create the graph
            create_gephi_graph(
                feature_extract=feature_extract[:gephi_subset_size],
                df=sample_df[:gephi_subset_size],
                title_column=label_column,
                model_name=base_name,
                file_path=graph_file,
                selected_labels=selected_labels,
                category_column=label_column,
                n_neighbors=graph_neighbors
            )
            
            print(f"Graph saved to: {graph_file}")
            generated_graphs.append(graph_file)
            
            # Clean up memory
            del model, feature_extract, feature_activations
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"Error generating graph for {model_path}: {e}")
            import traceback
            traceback.print_exc()
    
    # Create a readme for the graphs folder
    readme_path = os.path.join(graphs_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write("# Gephi Graph Visualizations\n\n")
        f.write("This folder contains graph visualizations of the selected best models.\n\n")
        f.write("## Created Graphs\n\n")
        
        if generated_graphs:
            for graph_path in generated_graphs:
                graph_name = os.path.basename(graph_path)
                f.write(f"- {graph_name}\n")
        else:
            f.write("No graphs were generated.\n")
    
    print(f"\nGenerated {len(generated_graphs)} graph files in {graphs_dir}")
    return generated_graphs

# Import functionality from find_best_models if available, or define it here
try:
    from find_best_models import find_best_models, display_model_info
except ImportError:
    # Define the functions here as fallback
    def find_best_models(results_path='model_comparison/full_results.csv', top_n=5, metrics=None, 
                    sort_by='avg_centroid_distance', ascending=False, filter_params=None):
        """
        Find the best models from centroid analysis results.
        
        Args:
            results_path: Path to the full_results.csv file
            top_n: Number of top models to return
            metrics: List of metrics to consider (if None, use all available)
            sort_by: Metric to sort by (default: avg_centroid_distance)
            ascending: Whether to sort in ascending order (default: False, meaning higher is better)
            filter_params: Dict of column-value pairs to filter by (e.g., {'model_type': 'sae'})
            
        Returns:
            DataFrame with top models for each metric
        """
        # Check if results file exists
        if not os.path.exists(results_path):
            print(f"Error: Results file not found at {results_path}")
            return None
        
        # Load results
        try:
            results_df = pd.read_csv(results_path)
            print(f"Loaded {len(results_df)} model evaluation results")
        except Exception as e:
            print(f"Error loading results file: {e}")
            return None
        
        # Ensure the required columns exist
        required_columns = ['model_path', 'model_type', 'function_type', 'feature_dimension', 'metric', sort_by]
        missing_columns = [col for col in required_columns if col not in results_df.columns]
        if missing_columns:
            print(f"Error: Required columns missing in results: {missing_columns}")
            return None
        
        # Filter by requested metrics
        if metrics is not None:
            results_df = results_df[results_df['metric'].isin(metrics)]
            print(f"Filtered to {len(results_df)} results with metrics: {metrics}")
        
        # Apply additional filters
        if filter_params:
            for column, value in filter_params.items():
                if column in results_df.columns:
                    # Handle lists of values
                    if isinstance(value, list):
                        results_df = results_df[results_df[column].isin(value)]
                        print(f"Filtered by {column} in {value}: {len(results_df)} results remaining")
                    else:
                        results_df = results_df[results_df[column] == value]
                        print(f"Filtered by {column} = {value}: {len(results_df)} results remaining")
            
        if results_df.empty:
            print("No results match the specified criteria.")
            return None
        
        # Group by metric and get top models for each
        top_models = []
        
        # First group by dataset if dataset is in the columns (NEW)
        if 'dataset' in results_df.columns:
            # Group by dataset and metric
            for (dataset_name, metric_name), group in results_df.groupby(['dataset', 'metric']):
                # Sort within each dataset-metric group
                sorted_group = group.sort_values(sort_by, ascending=ascending)
                
                # Get top N models for this dataset and metric
                group_top = sorted_group.head(top_n)
                print(f"Found {len(group_top)} top models for dataset: {dataset_name}, metric: {metric_name}")
                top_models.append(group_top)
        else:
            # Original logic - group only by metric
            for metric_name, metric_group in results_df.groupby('metric'):
                # Sort within each metric group
                sorted_group = metric_group.sort_values(sort_by, ascending=ascending)
                
                # Get top N models
                metric_top = sorted_group.head(top_n)
                print(f"Found {len(metric_top)} top models for metric: {metric_name}")
                top_models.append(metric_top)
        
        # Combine all top models
        if top_models:
            combined_top = pd.concat(top_models)
            return combined_top
        else:
            return None
            
    def display_model_info(model_df, output_format='text', output_file=None):
        """
        Display model information in a readable format.
        
        Args:
            model_df: DataFrame with model information
            output_format: Format to display results ('text', 'csv', 'json', or 'markdown')
            output_file: Optional file path to save the output
            
        Returns:
            None (prints to console or saves to file)
        """
        if model_df is None or model_df.empty:
            print("No models to display.")
            return
        
        # Select key columns for display
        display_columns = [
            'dataset',  # Added dataset column to display (NEW)
            'model_type', 'function_type', 'feature_dimension', 
            'metric', 'avg_centroid_distance', 'std_centroid_distance',
            'model_path'
        ]
        
        # Add additional columns if they exist
        optional_columns = ['avg_distance', 'l1_lambda', 'num_centroids', 'dead_ratio', 'sparsity']
        for col in optional_columns:
            if col in model_df.columns:
                display_columns.append(col)
        
        # Filter columns that exist in the DataFrame
        display_columns = [col for col in display_columns if col in model_df.columns]
        
        # Create a display DataFrame
        display_df = model_df[display_columns].copy()
        
        # Round numeric columns
        numeric_columns = display_df.select_dtypes(include=['float']).columns
        display_df[numeric_columns] = display_df[numeric_columns].round(4)
        
        # Format feature_dimension as integer if it's numeric
        if 'feature_dimension' in display_df.columns:
            try:
                display_df['feature_dimension'] = display_df['feature_dimension'].astype(int)
            except:
                pass
        
        # Format model path to be more readable
        if 'model_path' in display_df.columns:
            display_df['model_path'] = display_df['model_path'].apply(lambda p: os.path.relpath(p) if os.path.isabs(p) else p)
        
        # Generate output
        output_content = None
        if output_format.lower() == 'text':
            output_content = "\nBest Models by Centroid Distance:\n"
            output_content += tabulate(display_df, headers='keys', tablefmt='pretty', showindex=False)
        elif output_format.lower() == 'csv':
            output_content = display_df.to_csv(index=False)
        elif output_format.lower() == 'json':
            output_content = display_df.to_json(orient='records', indent=2)
        elif output_format.lower() == 'markdown':
            output_content = "## Best Models by Centroid Distance\n\n"
            output_content += display_df.to_markdown(index=False)
        else:
            output_content = "Unsupported output format. Using text format:\n"
            output_content += tabulate(display_df, headers='keys', tablefmt='pretty', showindex=False)
        
        # Save to file or print to console
        if output_file:
            with open(output_file, 'w') as f:
                f.write(output_content)
            print(f"Results saved to {output_file}")
        else:
            print(output_content)
        
        return display_df

def create_folder_name(args):
    """Create a descriptive folder name based on the filtering criteria"""
    parts = []
    
    # Always include top_n in the name
    parts.append(f"top{args.top_n}")
    
    # Add dataset if specified (NEW)
    if args.dataset:
        dataset_str = "_".join(args.dataset)
        parts.append(dataset_str)
    
    # Add metrics if specified
    if args.metrics:
        metric_str = "_".join(args.metrics)
        parts.append(metric_str)
    
    # Add model type if filtered
    if args.model_type:
        model_type_str = "_".join(args.model_type)
        parts.append(model_type_str)
    
    # Add function type if filtered
    if args.function_type:
        fn_type_str = "_".join(args.function_type)
        parts.append(fn_type_str)
    
    # Add feature dimensions if filtered
    if args.feature_dimension:
        dim_str = "dim" + "_".join(str(d) for d in args.feature_dimension)
        parts.append(dim_str)
    
    # Add sorting information
    sort_str = args.sort_by
    if args.ascending:
        sort_str += "_asc"
    else:
        sort_str += "_desc"
    parts.append(sort_str)
    
    # Combine all parts
    folder_name = "-".join(parts)
    
    return folder_name

def copy_model_files(best_models_df, output_dir, verify=True):
    """
    Copy the model files to the specified directory structure.
    
    Args:
        best_models_df: DataFrame with the best models information
        output_dir: Base directory to create the organized structure
        verify: Whether to verify that the model files exist
        
    Returns:
        Dictionary with information about the copied models
    """
    if best_models_df is None or best_models_df.empty:
        print("No models to copy.")
        return None
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare information for summary
    summary = {
        "total_models": len(best_models_df),
        "metrics": best_models_df['metric'].unique().tolist(),
        "copied_models": [],
        "errors": []
    }
    
    # Add datasets to summary if available (NEW)
    if 'dataset' in best_models_df.columns:
        summary["datasets"] = best_models_df['dataset'].unique().tolist()
    
    # Track metrics and datasets for organizing in subdirectories
    organization = {}
    
    # Organize by dataset if available, then by metric (NEW)
    if 'dataset' in best_models_df.columns:
        for dataset in best_models_df['dataset'].unique():
            dataset_dir = os.path.join(output_dir, f"dataset_{dataset}")
            os.makedirs(dataset_dir, exist_ok=True)
            
            dataset_models = best_models_df[best_models_df['dataset'] == dataset]
            
            for metric in dataset_models['metric'].unique():
                metric_dir = os.path.join(dataset_dir, f"metric_{metric}")
                os.makedirs(metric_dir, exist_ok=True)
                
                if dataset not in organization:
                    organization[dataset] = {}
                
                organization[dataset][metric] = dataset_models[dataset_models['metric'] == metric]
    else:
        # Original organization by metric only
        for metric in best_models_df['metric'].unique():
            metric_dir = os.path.join(output_dir, f"metric_{metric}")
            os.makedirs(metric_dir, exist_ok=True)
            organization[metric] = best_models_df[best_models_df['metric'] == metric]
    
    # Process each model
    for idx, row in best_models_df.iterrows():
        model_path = row['model_path']
        metric = row['metric']
        model_type = row['model_type']
        function_type = row['function_type']
        
        # Get dataset if available (NEW)
        dataset = row.get('dataset', None)
        
        # Check if model file exists
        if verify and not os.path.exists(model_path):
            error = f"Model file not found: {model_path}"
            summary["errors"].append(error)
            print(f"Error: {error}")
            continue
        
        try:
            # Create a descriptive name for the model copy
            model_basename = os.path.basename(model_path)
            feature_dim = str(row.get('feature_dimension', ''))
            avg_distance = f"{row.get('avg_centroid_distance', 0):.4f}"
            
            # Include dataset in filename if available (NEW)
            if dataset:
                new_filename = f"{dataset}_{model_type}_{function_type}_{feature_dim}_{avg_distance}_{model_basename}"
                new_filename_with_rank = f"{idx+1:02d}_{dataset}_{model_type}_{function_type}_{feature_dim}_{avg_distance}_{model_basename}"
            else:
                new_filename = f"{model_type}_{function_type}_{feature_dim}_{avg_distance}_{model_basename}"
                new_filename_with_rank = f"{idx+1:02d}_{model_type}_{function_type}_{feature_dim}_{avg_distance}_{model_basename}"
            
            # Copy to main directory
            main_copy_path = os.path.join(output_dir, new_filename_with_rank)
            shutil.copy2(model_path, main_copy_path)
            
            # Copy to appropriate subdirectory
            if dataset:
                # Copy to dataset-metric specific directory (NEW)
                metric_copy_path = os.path.join(output_dir, f"dataset_{dataset}", f"metric_{metric}", new_filename)
            else:
                # Copy to metric-specific directory (original)
                metric_copy_path = os.path.join(output_dir, f"metric_{metric}", new_filename)
                
            if not os.path.exists(metric_copy_path):  # Avoid duplicate copies
                shutil.copy2(model_path, metric_copy_path)
            
            # Add to successful models
            model_info = {
                "original_path": model_path,
                "copy_path": main_copy_path,
                "model_type": model_type,
                "function_type": function_type,
                "feature_dimension": str(row.get('feature_dimension', '')),
                "metric": metric,
                "avg_centroid_distance": avg_distance
            }
            
            # Add dataset if available (NEW)
            if dataset:
                model_info["dataset"] = dataset
                
            summary["copied_models"].append(model_info)
            
        except Exception as e:
            error = f"Error processing {model_path}: {str(e)}"
            summary["errors"].append(error)
            print(f"Error: {error}")
    
    # Create an index file with the model information
    index_file = os.path.join(output_dir, "index.json")
    with open(index_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create a more human-readable summary
    summary_file = os.path.join(output_dir, "README.md")
    with open(summary_file, 'w') as f:
        f.write("# Best Models Summary\n\n")
        f.write(f"Total models: {summary['total_models']}\n\n")
        f.write(f"Metrics: {', '.join(summary['metrics'])}\n\n")
        
        # Add datasets if available (NEW)
        if 'datasets' in summary:
            f.write(f"Datasets: {', '.join(summary['datasets'])}\n\n")
        
        if summary["errors"]:
            f.write("## Errors\n\n")
            for error in summary["errors"]:
                f.write(f"- {error}\n")
            f.write("\n")
        
        f.write("## Copied Models\n\n")
        
        # Convert the copied models to a DataFrame for tabulate
        models_df = pd.DataFrame(summary["copied_models"])
        if not models_df.empty:
            # Include dataset in display columns if available (NEW)
            if 'dataset' in models_df.columns:
                display_cols = ['dataset', 'model_type', 'function_type', 'feature_dimension', 
                              'metric', 'avg_centroid_distance', 'original_path', 'copy_path']
            else:
                display_cols = ['model_type', 'function_type', 'feature_dimension', 
                              'metric', 'avg_centroid_distance', 'original_path', 'copy_path']
                
            # Filter columns that exist
            display_cols = [col for col in display_cols if col in models_df.columns]
            
            f.write(models_df[display_cols].to_markdown(index=False))
    
    # Create a CSV file with the data
    csv_file = os.path.join(output_dir, "best_models.csv")
    best_models_df.to_csv(csv_file, index=False)
    
    # Create a detailed model metrics file
    metrics_file = os.path.join(output_dir, "model_metrics.txt")
    display_model_info(best_models_df, output_format='text', output_file=metrics_file)
    
    print(f"\nSuccessfully copied {len(summary['copied_models'])} models to {output_dir}")
    if summary["errors"]:
        print(f"Encountered {len(summary['errors'])} errors. See {summary_file} for details.")
    
    print(f"\nCreated the following files:")
    print(f"- {index_file} (JSON with full model information)")
    print(f"- {summary_file} (Markdown with human-readable summary)")
    print(f"- {csv_file} (CSV with model data)")
    print(f"- {metrics_file} (Detailed model metrics in text format)")
    
    # Create dataset-specific and metric-specific readme files (UPDATED)
    if 'dataset' in best_models_df.columns:
        # Organize by dataset first, then by metric
        for dataset in best_models_df['dataset'].unique():
            dataset_dir = os.path.join(output_dir, f"dataset_{dataset}")
            dataset_readme = os.path.join(dataset_dir, "README.md")
            
            dataset_models = best_models_df[best_models_df['dataset'] == dataset]
            
            with open(dataset_readme, 'w') as f:
                f.write(f"# Best Models - {dataset} dataset\n\n")
                f.write(f"Total models: {len(dataset_models)}\n\n")
                
                # Sort by avg_centroid_distance
                sorted_df = dataset_models.sort_values('avg_centroid_distance', ascending=False)
                
                # Display in markdown table
                display_cols = ['model_type', 'function_type', 'feature_dimension', 
                              'metric', 'avg_centroid_distance', 'model_path']
                # Filter columns that exist
                display_cols = [col for col in display_cols if col in sorted_df.columns]
                
                f.write(sorted_df[display_cols].to_markdown(index=False))
            
            # Create metric-specific readme files within each dataset
            for metric in dataset_models['metric'].unique():
                metric_dir = os.path.join(dataset_dir, f"metric_{metric}")
                metric_readme = os.path.join(metric_dir, "README.md")
                
                metric_models = dataset_models[dataset_models['metric'] == metric]
                
                with open(metric_readme, 'w') as f:
                    f.write(f"# Best Models - {dataset} dataset, {metric} metric\n\n")
                    f.write(f"Total models: {len(metric_models)}\n\n")
                    
                    # Sort by avg_centroid_distance
                    sorted_df = metric_models.sort_values('avg_centroid_distance', ascending=False)
                    
                    # Display in markdown table
                    display_cols = ['model_type', 'function_type', 'feature_dimension', 
                                  'avg_centroid_distance', 'model_path']
                    # Filter columns that exist
                    display_cols = [col for col in display_cols if col in sorted_df.columns]
                    
                    f.write(sorted_df[display_cols].to_markdown(index=False))
                
                # Create a detailed metrics file for this dataset-metric combination
                metric_metrics_file = os.path.join(metric_dir, "model_metrics.txt")
                display_model_info(metric_models, output_format='text', output_file=metric_metrics_file)
                
                print(f"- {metric_readme} (Dataset-metric specific README)")
                print(f"- {metric_metrics_file} (Dataset-metric specific detailed metrics)")
    else:
        # Original organization by metric only
        for metric in best_models_df['metric'].unique():
            metric_dir = os.path.join(output_dir, f"metric_{metric}")
            metric_readme = os.path.join(metric_dir, "README.md")
            
            metric_df = best_models_df[best_models_df['metric'] == metric]
            
            with open(metric_readme, 'w') as f:
                f.write(f"# Best Models - {metric} metric\n\n")
                f.write(f"Total models: {len(metric_df)}\n\n")
                
                # Sort by avg_centroid_distance
                sorted_df = metric_df.sort_values('avg_centroid_distance', ascending=False)
                
                # Display in markdown table
                display_cols = ['model_type', 'function_type', 'feature_dimension', 
                              'avg_centroid_distance', 'model_path']
                # Filter columns that exist
                display_cols = [col for col in display_cols if col in sorted_df.columns]
                
                f.write(sorted_df[display_cols].to_markdown(index=False))
            
            # Create a detailed metrics file for this metric
            metric_metrics_file = os.path.join(metric_dir, "model_metrics.txt")
            display_model_info(metric_df, output_format='text', output_file=metric_metrics_file)
            
            print(f"- {metric_readme} (Metric-specific README)")
            print(f"- {metric_metrics_file} (Metric-specific detailed metrics)")
    
    # Display the tabular output directly in the console as well
    print("\nDetailed Metrics for Selected Models:")
    display_model_info(best_models_df, output_format='text')
    
    return summary

def main():
    parser = argparse.ArgumentParser(description='Copy best models from centroid analysis results')
    parser.add_argument('--results_path', type=str, default='model_comparison/full_results.csv',
                      help='Path to full_results.csv file from centroid analysis')
    parser.add_argument('--top_n', type=int, default=10,
                      help='Number of top models to organize for each metric')
    parser.add_argument('--metrics', type=str, nargs='+', default=None,
                      help='Metrics to consider (if None, use all available)')
    parser.add_argument('--sort_by', type=str, default='avg_centroid_distance',
                      help='Metric to sort by (default: avg_centroid_distance)')
    parser.add_argument('--ascending', action='store_true', default=False,
                      help='Sort in ascending order (default: False, higher values are better)')
    parser.add_argument('--model_type', type=str, nargs='+', default=None,
                      help='Filter by model type (e.g., sae, st)')
    parser.add_argument('--function_type', type=str, nargs='+', default=None,
                      help='Filter by function type (e.g., softmax, relu)')
    parser.add_argument('--feature_dimension', type=int, nargs='+', default=None,
                      help='Filter by feature dimension')
    # Add new dataset filter argument (NEW)
    parser.add_argument('--dataset', type=str, nargs='+', default=None,
                      help='Filter by dataset (e.g., mnist, fashion_mnist)')
    parser.add_argument('--output_dir', type=str, default=None,
                      help='Directory to store copied models (default: auto-generated)')
    parser.add_argument('--no_verify', action='store_true', default=False,
                      help='Skip verification of model file existence')
    # Add an option to separate top models by dataset (NEW)
    parser.add_argument('--separate_datasets', action='store_true', default=False,
                      help='Find top N models for each dataset independently')
    #graph
    parser.add_argument('--generate_graphs', action='store_true', default=False,
                    help='Generate Gephi graph visualizations for best models')
    parser.add_argument('--dataset_path', type=str, default=None,
                      help='Path to dataset for graph visualization')
    parser.add_argument('--n_random_labels', type=int, default=10,
                      help='Number of random labels for graph visualization')
    parser.add_argument('--gephi_subset_size', type=int, default=1000,
                      help='Size of subset for Gephi visualization')
    parser.add_argument('--graph_neighbors', type=int, default=4,
                      help='Number of neighbors for graph creation')
    
    args = parser.parse_args()
    
    # Build filter parameters
    filter_params = {}
    if args.model_type:
        filter_params['model_type'] = args.model_type
    if args.function_type:
        filter_params['function_type'] = args.function_type
    if args.feature_dimension:
        filter_params['feature_dimension'] = args.feature_dimension
    # Add dataset to filter parameters (NEW)
    if args.dataset:
        filter_params['dataset'] = args.dataset
    

    # Find best models
    print("Finding best models based on criteria...")
    best_models = find_best_models(
        results_path=args.results_path,
        top_n=args.top_n,
        metrics=args.metrics,
        sort_by=args.sort_by,
        ascending=args.ascending,
        filter_params=filter_params
    )
    
    if best_models is None or best_models.empty:
        print("No models found matching the criteria. Exiting.")
        sys.exit(1)
    
    # Create auto-generated output directory name if not provided
    if args.output_dir is None:
        folder_name = create_folder_name(args)
        # Create in a 'best_models' subdirectory
        args.output_dir = os.path.join('best_models', folder_name)
    
    print(f"\nCopying {len(best_models)} models to {args.output_dir}")
    
    # Copy the models
    copy_model_files(
        best_models,
        args.output_dir,
        verify=not args.no_verify
    )

    if args.generate_graphs:
        print("\nGenerating Gephi graphs for best models...")
        generate_gephi_graphs(
            best_models,
            args.output_dir,
            dataset_path=args.dataset_path,
            n_random_labels=args.n_random_labels,
            gephi_subset_size=args.gephi_subset_size,
            graph_neighbors=args.graph_neighbors
        )
if __name__ == "__main__":
    main()