import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from typing import List, Optional
import warnings

def create_variable_visualizations(results_csv: str, 
                                  variable: str, 
                                  output_dir: Optional[str] = None,
                                  secondary_variable: Optional[str] = None,
                                  metrics: Optional[List[str]] = None) -> List[str]:
    """
    Create visualizations comparing model centroid distances in relation to a specified variable.
    
    Args:
        results_csv: Path to CSV file with analysis results
        variable: The main variable to analyze (e.g., 'l1_lambda', 'feature_dimension')
        output_dir: Directory to save visualizations (default: visualization/{variable})
        secondary_variable: Optional second variable for more complex comparisons
        metrics: Which distance metrics to visualize (default: all available)
        
    Returns:
        List of generated visualization paths
    """
    # Set up output directory
    if output_dir is None:
        output_dir = f"visualization/{variable}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    df = pd.read_csv(results_csv)
    
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
    
    # Track generated visualizations
    visualizations = []
    
    # Create information file
    info_text = f"""
Variable Analysis for: {variable}
===============================

This directory contains visualizations showing the relationship between 
{variable} and centroid distances in your models.

The visualizations were generated from: {results_csv}

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
    
    # Print information about the analysis
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
                                 secondary_variable: Optional[str] = None) -> List[str]:
    """
    Create plots specifically for numeric variables.
    """
    visualizations = []
    
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
            
            plt.title(f'Average Centroid Distance vs {variable.replace("_", " ").title()}{metric_suffix}')
            plt.xlabel(variable.replace('_', ' ').title())
            plt.ylabel('Average Centroid Distance')
            plt.grid(alpha=0.3)
            
            scatter_path = os.path.join(output_dir, f'scatter_{variable}{metric_suffix}.png')
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
            
            plt.title(f'Centroid Distance Trend by {variable.replace("_", " ").title()}{metric_suffix}')
            plt.xlabel(variable.replace('_', ' ').title())
            plt.ylabel('Average Centroid Distance')
            plt.grid(alpha=0.3)
            
            trend_path = os.path.join(output_dir, f'trend_{variable}{metric_suffix}.png')
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
                
                plt.title(f'Distribution of Centroid Distances by {variable.replace("_", " ").title()}{metric_suffix}')
                plt.xlabel(variable.replace('_', ' ').title())
                plt.ylabel('Average Centroid Distance')
                plt.grid(alpha=0.3, axis='y')
                
                box_path = os.path.join(output_dir, f'boxplot_{variable}{metric_suffix}.png')
                plt.savefig(box_path, dpi=300)
                visualizations.append(box_path)
            else:
                # For too many unique values, create a binned histogram instead
                plt.title(f'Too many unique values ({unique_values}) for boxplot. See scatter plot instead.')
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
                
                plt.title(f'Regression: Centroid Distance vs {variable.replace("_", " ").title()}{metric_suffix}')
                plt.xlabel(variable.replace('_', ' ').title())
                plt.ylabel('Average Centroid Distance')
                plt.grid(alpha=0.3)
                
                reg_path = os.path.join(output_dir, f'regression_{variable}{metric_suffix}.png')
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
                                     secondary_variable: Optional[str] = None) -> List[str]:
    """
    Create plots specifically for categorical variables.
    """
    visualizations = []
    
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
            
            plt.title(f'Average Centroid Distance by {variable.replace("_", " ").title()}{metric_suffix}')
            plt.xlabel(variable.replace('_', ' ').title())
            plt.ylabel('Average Centroid Distance')
            plt.grid(alpha=0.3, axis='y')
            
            # Rotate x-axis labels if needed
            if len(metric_df[variable].unique()) > 3:
                plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            bar_path = os.path.join(output_dir, f'barplot_{variable}{metric_suffix}.png')
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
            
            plt.title(f'Distribution of Centroid Distances by {variable.replace("_", " ").title()}{metric_suffix}')
            plt.xlabel(variable.replace('_', ' ').title())
            plt.ylabel('Average Centroid Distance')
            plt.grid(alpha=0.3, axis='y')
            
            # Rotate x-axis labels if needed
            if len(metric_df[variable].unique()) > 3:
                plt.xticks(rotation=45, ha='right')
                
            plt.tight_layout()
            box_path = os.path.join(output_dir, f'boxplot_{variable}{metric_suffix}.png')
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
                
                plt.title(f'Distribution Detail of Centroid Distances by {variable.replace("_", " ").title()}{metric_suffix}')
                plt.xlabel(variable.replace('_', ' ').title())
                plt.ylabel('Average Centroid Distance')
                plt.grid(alpha=0.3, axis='y')
                
                # Rotate x-axis labels if needed
                if len(metric_df[variable].unique()) > 3:
                    plt.xticks(rotation=45, ha='right')
                    
                plt.tight_layout()
                violin_path = os.path.join(output_dir, f'violin_{variable}{metric_suffix}.png')
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
                                 secondary_variable: Optional[str] = None) -> List[str]:
    """
    Create general analysis plots that work for both numeric and categorical variables.
    """
    visualizations = []
    
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
                    plt.title(f'Centroid Distance by Function Type and {variable.replace("_", " ").title()}{metric_suffix}')
                    plt.tight_layout()
                    
                    heatmap_path = os.path.join(output_dir, f'heatmap_{variable}{metric_suffix}.png')
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
                
                plt.title(f'Top and Bottom Models by Centroid Distance ({variable} comparison){metric_suffix}')
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
                ranking_path = os.path.join(output_dir, f'top_bottom_{variable}{metric_suffix}.png')
                plt.savefig(ranking_path, dpi=300)
                visualizations.append(ranking_path)
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
                      help='Directory to save visualizations (default: visualization/{variable})')
    parser.add_argument('--metrics', type=str, nargs='+', default=None,
                      help='Distance metrics to use (default: all available)')
    parser.add_argument('--secondary', type=str, default=None,
                      help='Secondary variable for additional insight')
    
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
        args.metrics
    )
    
    # Print results summary
    output_dir = args.output_dir or f"visualization/{args.variable}"
    print(f"\nGenerated {len(visualizations)} visualizations for {args.variable}.")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()