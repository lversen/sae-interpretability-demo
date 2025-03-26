import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from matplotlib.colors import LinearSegmentedColormap

def create_enhanced_visualizations(results_csv, output_dir='enhanced_visualizations'):
    """
    Create enhanced visualizations from analysis results.
    
    Args:
        results_csv: Path to CSV file with analysis results
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    df = pd.read_csv(results_csv)
    
    if df.empty:
        print("No results to visualize.")
        return []
    
    visualizations = []
    
    # 1. Combined grid plot for both SAE and ST using catplot
    for metric in df['metric'].unique():
        metric_df = df[df['metric'] == metric].copy()
        
        # Ensure feature_dimension is numeric
        try:
            metric_df['feature_dimension'] = pd.to_numeric(metric_df['feature_dimension'])
        except:
            pass
        
        # Create a catplot for cleaner model type comparison
        g = sns.catplot(
            data=metric_df,
            kind="bar",
            x="function_type", 
            y="avg_centroid_distance",
            hue="model_type",
            palette=["blue", "green"],
            alpha=0.7,
            height=6,
            aspect=2
        )
        
        g.set_axis_labels("Function Type (Activation/Attention)", "Average Centroid Distance")
        g.legend.set_title("Model Type")
        plt.title(f'Average Centroid Distance Comparison ({metric} metric)')
        plt.grid(alpha=0.3)
        
        # Rotate x-tick labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Adjust layout and save
        plt.tight_layout()
        grid_path = os.path.join(output_dir, f'model_comparison_{metric}.png')
        plt.savefig(grid_path, dpi=300)
        visualizations.append(grid_path)
        plt.close()
    
    # 2. Feature dimension impact visualization
    plt.figure(figsize=(15, 10))
    
    # Set up a 2x2 grid for different metrics and model types
    for i, metric in enumerate(df['metric'].unique()):
        metric_df = df[df['metric'] == metric].copy()
        
        for j, model_type in enumerate(['sae', 'st']):
            plt.subplot(2, 2, i*2 + j + 1)
            
            model_data = metric_df[metric_df['model_type'] == model_type].copy()
            if model_data.empty:
                plt.text(0.5, 0.5, f"No {model_type.upper()} data for {metric} metric",
                        ha='center', va='center', transform=plt.gca().transAxes)
                continue
            
            # Ensure feature_dimension is numeric
            try:
                model_data['feature_dimension'] = pd.to_numeric(model_data['feature_dimension'])
            except:
                pass
            
            # Group by function type and feature dimension
            grouped = model_data.groupby(['function_type', 'feature_dimension'])['avg_centroid_distance'].mean().reset_index()
            
            # Plot
            for function, group in grouped.groupby('function_type'):
                plt.plot(
                    group['feature_dimension'], 
                    group['avg_centroid_distance'], 
                    'o-',
                    label=function,
                    linewidth=2,
                    markersize=8,
                    alpha=0.8
                )
            
            plt.title(f'{model_type.upper()} - {metric.capitalize()} Metric')
            plt.xlabel('Feature Dimension')
            plt.ylabel('Average Centroid Distance')
            plt.grid(alpha=0.3)
            plt.legend(title=f"{'Activation' if model_type=='sae' else 'Attention'} Function")
    
    plt.tight_layout()
    dimension_path = os.path.join(output_dir, 'feature_dimension_impact.png')
    plt.savefig(dimension_path, dpi=300)
    visualizations.append(dimension_path)
    plt.close()
    
    # 3. Create radar charts
    create_radar_charts(df, output_dir, visualizations)
    
    # 4. Create model ranking visualization
    create_model_ranking(df, output_dir, visualizations)
    
    # 5. Feature correlation analysis
    create_feature_correlation_analysis(df, output_dir, visualizations)
    
    # 6. Model architecture comparison 
    create_architecture_comparison(df, output_dir, visualizations)
    
    return visualizations

def create_radar_charts(df, output_dir, visualizations):
    """Create radar charts comparing different models"""
    if df.empty:
        return
    
    # We'll create radar charts for the top models by type
    for metric in df['metric'].unique():
        metric_df = df[df['metric'] == metric].copy()
        
        # Create categories for radar chart
        categories = ['avg_centroid_distance', 'avg_distance', 
                     'std_centroid_distance', 'min_centroid_distance',
                     'max_centroid_distance']
        
        # Get top model for each type (lowest avg_centroid_distance)
        top_models = metric_df.sort_values('avg_centroid_distance').groupby('model_type').first()
        
        if len(top_models) == 0:
            continue
            
        # Normalize values for radar chart (0-1 scale)
        radar_data = top_models[categories].copy()
        
        # Invert some metrics where lower is better
        for col in ['avg_centroid_distance', 'avg_distance', 'std_centroid_distance']:
            max_val = metric_df[col].max()
            min_val = metric_df[col].min()
            if max_val > min_val:
                radar_data[col] = 1 - ((top_models[col] - min_val) / (max_val - min_val))
        
        # Create radar chart
        plt.figure(figsize=(10, 8))
        
        # Number of variables
        N = len(categories)
        
        # What will be the angle of each axis in the plot
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Initialize the plot
        ax = plt.subplot(111, polar=True)
        
        # Draw one axis per variable and add labels
        plt.xticks(angles[:-1], [c.replace('_', ' ').title() for c in categories], size=10)
        
        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=8)
        plt.ylim(0, 1)
        
        # Plot each model
        for model_type, row in radar_data.iterrows():
            values = row[categories].values.flatten().tolist()
            values += values[:1]  # Close the loop
            
            # Plot
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_type.upper())
            ax.fill(angles, values, alpha=0.1)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.title(f'Model Comparison Radar Chart ({metric} metric)')
        
        # Save
        radar_path = os.path.join(output_dir, f'radar_chart_{metric}.png')
        plt.savefig(radar_path, dpi=300)
        visualizations.append(radar_path)
        plt.close()

def create_model_ranking(df, output_dir, visualizations):
    """Create model ranking visualization"""
    if df.empty:
        return
    
    # Rank models by average centroid distance (lower is better)
    for metric in df['metric'].unique():
        metric_df = df[df['metric'] == metric].copy()
        
        # Define a composite score (lower is better)
        metric_df['composite_score'] = (
            metric_df['avg_centroid_distance'] * 0.6 +  # Higher weight to centroid distance
            metric_df['std_centroid_distance'] * 0.2 +  # Reward consistency
            metric_df['avg_distance'] * 0.2             # Consider general feature distance
        )
        
        # Sort by score
        top_models = metric_df.sort_values('composite_score').head(15)
        
        # Create model labels with relevant info
        top_models['model_label'] = top_models.apply(
            lambda x: f"{x['model_type'].upper()} - {x['function_type']} - {x['feature_dimension']}",
            axis=1
        )
        
        # Create bar chart
        plt.figure(figsize=(12, 8))
        
        # Create color mapping based on model type
        colors = ['#1f77b4' if m == 'sae' else '#2ca02c' for m in top_models['model_type']]
        
        # Horizontal bar chart
        bars = plt.barh(
            top_models['model_label'],
            top_models['composite_score'],
            color=colors,
            alpha=0.7
        )
        
        # Add value annotations
        for i, bar in enumerate(bars):
            plt.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height()/2,
                f"{top_models['composite_score'].iloc[i]:.3f}",
                va='center'
            )
        
        plt.title(f'Top 15 Models Ranked by Composite Score ({metric} metric)')
        plt.xlabel('Composite Score (lower is better)')
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
        ranking_path = os.path.join(output_dir, f'model_ranking_{metric}.png')
        plt.savefig(ranking_path, dpi=300)
        visualizations.append(ranking_path)
        plt.close()

def create_feature_correlation_analysis(df, output_dir, visualizations):
    """Create correlation analysis between feature metrics"""
    if df.empty:
        return
        
    # Add explanation about centroid calculation without dimensionality reduction
    info_text = """
    Note: All centroid distances were calculated in the original feature space.
    No dimensionality reduction was applied before centroid calculation to preserve
    all feature relationships and avoid information loss.
    """
    
    # Create information text file
    info_path = os.path.join(output_dir, 'analysis_information.txt')
    with open(info_path, 'w') as f:
        f.write(info_text)
    visualizations.append(info_path)
    
    # Select numeric columns for correlation
    numeric_cols = ['avg_centroid_distance', 'std_centroid_distance',
                   'min_centroid_distance', 'max_centroid_distance',
                   'avg_distance', 'std_distance', 'min_distance',
                   'max_distance', 'num_features', 'dimension']
    
    # Add feature_dimension if available and numeric
    if 'feature_dimension' in df.columns:
        try:
            df_copy = df.copy()
            df_copy['feature_dimension'] = pd.to_numeric(df_copy['feature_dimension'])
            numeric_cols.append('feature_dimension')
        except:
            pass
    
    # For each metric
    for metric in df['metric'].unique():
        metric_df = df[df['metric'] == metric].copy()
        
        # Calculate correlation matrix
        corr_matrix = metric_df[numeric_cols].corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        
        # Create a custom diverging colormap (blue-white-red)
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        # Create heatmap
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap=cmap,
            center=0,
            square=True,
            linewidths=.5,
            cbar_kws={"shrink": .8}
        )
        
        plt.title(f'Feature Metrics Correlation Matrix ({metric} metric)')
        plt.tight_layout()
        
        # Save
        corr_path = os.path.join(output_dir, f'correlation_matrix_{metric}.png')
        plt.savefig(corr_path, dpi=300)
        visualizations.append(corr_path)
        plt.close()
        
        # Create scatter plot matrix for key metrics
        key_metrics = ['avg_centroid_distance', 'avg_distance', 
                      'std_centroid_distance', 'num_features']
        
        if 'feature_dimension' in numeric_cols:
            key_metrics.append('feature_dimension')
        
        try:
            # Add model_type as hue
            g = sns.pairplot(
                metric_df,
                vars=key_metrics,
                hue='model_type',
                diag_kind='kde',
                plot_kws={'alpha': 0.6},
                height=2.5
            )
            
            plt.suptitle(f'Relationships Between Key Metrics ({metric} metric)', y=1.02)
            
            # Save
            pairplot_path = os.path.join(output_dir, f'metric_relationships_{metric}.png')
            plt.savefig(pairplot_path, dpi=300, bbox_inches='tight')
            visualizations.append(pairplot_path)
            plt.close()
        except Exception as e:
            print(f"Error creating pairplot: {e}")

def create_architecture_comparison(df, output_dir, visualizations):
    """Compare model architectures (SAE vs ST, memory vs direct)"""
    if df.empty:
        return
    
    # Create comparison boxplots
    plt.figure(figsize=(14, 10))
    
    # Set up a grid for different metrics
    for i, metric in enumerate(df['metric'].unique()):
        metric_df = df[df['metric'] == metric].copy()
        
        plt.subplot(len(df['metric'].unique()), 1, i+1)
        
        # Try to add 'memory_type' column based on 'memory' flag if it exists
        try:
            if 'memory' in metric_df.columns:
                metric_df['memory_type'] = metric_df.apply(
                    lambda x: 'Memory Bank' if x['memory'] else 'Direct K-V',
                    axis=1
                )
            else:
                # Default to 'Unknown' if no memory column
                metric_df['memory_type'] = 'Unknown'
                
            # Create boxplot without overlapping label problems
            ax = sns.boxplot(
                x='model_type',
                y='avg_centroid_distance',
                hue='model_type',  # Use the same column for hue
                legend=False,      # Don't create a redundant legend
                data=metric_df,
                palette='Set3'
            )
            
            plt.title(f'Model Architecture Comparison ({metric} metric)')
            plt.xlabel('Model Type')
            plt.ylabel('Average Centroid Distance')
            plt.grid(alpha=0.3, axis='y')
            
            # Add swarm plot for individual points - separate to avoid label conflicts
            sns.swarmplot(
                x='model_type',
                y='avg_centroid_distance',
                data=metric_df,
                alpha=0.7,
                size=4,
                color='.3'
            )
        except Exception as e:
            print(f"Error creating architecture comparison: {e}")
    
    plt.tight_layout()
    
    # Save
    arch_path = os.path.join(output_dir, 'architecture_comparison.png')
    plt.savefig(arch_path, dpi=300)
    visualizations.append(arch_path)
    plt.close()
    
    # Create architecture histogram comparison
    for metric in df['metric'].unique():
        metric_df = df[df['metric'] == metric].copy()
        
        # Create a figure with kernel density estimates
        plt.figure(figsize=(12, 8))
        
        # Create data subsets
        sae_data = metric_df[metric_df['model_type'] == 'sae']['avg_centroid_distance']
        
        # For ST data, separate by memory type if possible
        st_data = metric_df[metric_df['model_type'] == 'st']
        
        try:
            if 'memory' in st_data.columns:
                st_direct_data = st_data[~st_data['memory']]['avg_centroid_distance']
                st_memory_data = st_data[st_data['memory']]['avg_centroid_distance']
            else:
                # If no memory column, just use all ST data
                st_direct_data = st_data['avg_centroid_distance']
                st_memory_data = pd.Series()  # Empty series
        except:
            # Fallback
            st_direct_data = st_data['avg_centroid_distance']
            st_memory_data = pd.Series()  # Empty series
        
        # Plot density curves
        if not sae_data.empty:
            sns.kdeplot(sae_data, label='SAE', fill=True, alpha=0.3)
        if not st_direct_data.empty:
            sns.kdeplot(st_direct_data, label='ST (Direct K-V)', fill=True, alpha=0.3)
        if not st_memory_data.empty:
            sns.kdeplot(st_memory_data, label='ST (Memory Bank)', fill=True, alpha=0.3)
        
        plt.title(f'Distribution of Centroid Distances by Architecture ({metric} metric)')
        plt.xlabel('Average Centroid Distance')
        plt.ylabel('Density')
        plt.grid(alpha=0.3)
        plt.legend()
        
        # Save
        dist_path = os.path.join(output_dir, f'distance_distribution_{metric}.png')
        plt.savefig(dist_path, dpi=300)
        visualizations.append(dist_path)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Create enhanced visualizations from analysis results')
    parser.add_argument('--results_csv', type=str, default='model_comparison/full_results.csv',
                      help='Path to CSV file with analysis results')
    parser.add_argument('--output_dir', type=str, default='enhanced_visualizations',
                      help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # Create visualizations
    print(f"Creating enhanced visualizations from {args.results_csv}...")
    visualizations = create_enhanced_visualizations(args.results_csv, output_dir=args.output_dir)
    
    print(f"Generated {len(visualizations)} enhanced visualizations.")
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()