import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, List, Tuple, Optional
from thought_vectors import *

def analyze_latent_feature_importance_per_class(latent_reps, labels, n_features=5):
    """
    Analyze which latent features are most important for distinguishing each class
    
    Args:
        latent_reps: Latent representations
        labels: Labels for each sample
        n_features: Number of top features to return per class
        
    Returns:
        Dictionary mapping class label to tuple of (feature_importance, top_feature_indices)
    """
    try:
        unique_labels = np.unique(labels)
        importance_per_class = {}
        
        for label in unique_labels:
            # Create binary labels (one-vs-rest)
            binary_labels = (labels == label).astype(int)
            
            # Train a random forest for this class
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(latent_reps, binary_labels)
            
            # Get feature importances
            feature_importance = clf.feature_importances_
            
            # Get top features
            top_feature_indices = np.argsort(feature_importance)[::-1][:n_features]
            
            # Store results for this class
            importance_per_class[label] = (feature_importance, top_feature_indices)
        
        return importance_per_class
    
    except Exception as e:
        print(f"Error analyzing feature importance per class: {e}")
        # Fallback: use variance but still organize by class
        importance_per_class = {}
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            # Get samples for this class
            class_latent = latent_reps[labels == label]
            
            # Calculate variance for each feature
            feature_variance = np.var(class_latent, axis=0)
            
            # Get top features by variance
            top_feature_indices = np.argsort(feature_variance)[::-1][:n_features]
            
            # Store results for this class
            importance_per_class[label] = (feature_variance, top_feature_indices)
        
        return importance_per_class

def generate_feature_importance_markdown_tables(
    importance_per_class: Dict[int, Tuple[np.ndarray, np.ndarray]], 
    label_names: Optional[List[str]] = None,
    n_top_features: int = 10,
    output_file: Optional[str] = None
) -> str:
    """
    Generate markdown tables summarizing feature importance per class
    
    Args:
        importance_per_class: Dictionary from analyze_latent_feature_importance_per_class
        label_names: Names of the classes (optional)
        n_top_features: Number of top features to include in detailed tables
        output_file: Optional file path to save the markdown output
        
    Returns:
        String containing markdown formatted tables
    """
    markdown_content = []
    
    # Header
    markdown_content.append("# Feature Importance Analysis Per Class\n")
    markdown_content.append("This analysis shows which latent features are most important for distinguishing each class.\n")
    
    # Get all unique classes
    classes = sorted(importance_per_class.keys())
    
    # 1. Summary table of all classes
    markdown_content.append("## Summary Table - Top 5 Features Per Class\n")
    
    summary_table = []
    summary_headers = ["Class", "Top Feature", "Importance", "2nd Feature", "Importance", 
                      "3rd Feature", "Importance", "4th Feature", "Importance", "5th Feature", "Importance"]
    
    for class_label in classes:
        feature_importance, top_feature_indices = importance_per_class[class_label]
        class_name = label_names[class_label] if label_names else f"Class {class_label}"
        
        row = [class_name]
        for i in range(min(5, len(top_feature_indices))):
            feature_idx = top_feature_indices[i]
            importance = feature_importance[feature_idx]
            row.extend([f"F{feature_idx}", f"{importance:.4f}"])
        
        # Pad row if fewer than 5 features
        while len(row) < len(summary_headers):
            row.extend(["-", "-"])
            
        summary_table.append(row)
    
    # Create summary table markdown
    summary_df = pd.DataFrame(summary_table, columns=summary_headers)
    markdown_content.append(summary_df.to_markdown(index=False))
    markdown_content.append("\n")
    
    # 2. Feature overlap analysis
    markdown_content.append("## Feature Overlap Analysis\n")
    
    # Find which features appear in top N for multiple classes
    feature_class_mapping = {}
    for class_label in classes:
        _, top_feature_indices = importance_per_class[class_label]
        class_name = label_names[class_label] if label_names else f"Class {class_label}"
        
        for i, feature_idx in enumerate(top_feature_indices[:5]):  # Top 5 features
            if feature_idx not in feature_class_mapping:
                feature_class_mapping[feature_idx] = []
            feature_class_mapping[feature_idx].append((class_name, i+1))  # rank starting from 1
    
    # Create overlap table
    overlap_data = []
    for feature_idx, class_info in feature_class_mapping.items():
        if len(class_info) > 1:  # Feature appears in multiple classes
            classes_str = ", ".join([f"{cls}({rank})" for cls, rank in class_info])
            overlap_data.append([f"F{feature_idx}", len(class_info), classes_str])
    
    if overlap_data:
        overlap_df = pd.DataFrame(overlap_data, columns=["Feature", "Num Classes", "Classes (Rank)"])
        overlap_df = overlap_df.sort_values("Num Classes", ascending=False)
        markdown_content.append("Features that appear in top 5 for multiple classes:\n")
        markdown_content.append(overlap_df.to_markdown(index=False))
        markdown_content.append("\n")
    else:
        markdown_content.append("No features appear in the top 5 for multiple classes.\n")
    
    # 3. Detailed tables for each class
    markdown_content.append("## Detailed Feature Rankings Per Class\n")
    
    for class_label in classes:
        feature_importance, top_feature_indices = importance_per_class[class_label]
        class_name = label_names[class_label] if label_names else f"Class {class_label}"
        
        markdown_content.append(f"### {class_name}\n")
        
        # Create detailed table for this class
        detailed_data = []
        for i, feature_idx in enumerate(top_feature_indices[:n_top_features]):
            importance = feature_importance[feature_idx]
            detailed_data.append([i+1, f"F{feature_idx}", f"{importance:.6f}", f"{importance/feature_importance[top_feature_indices[0]]*100:.1f}%"])
        
        detailed_df = pd.DataFrame(detailed_data, columns=["Rank", "Feature", "Importance", "% of Top"])
        markdown_content.append(detailed_df.to_markdown(index=False))
        markdown_content.append("\n")
    
    # 4. Statistical summary
    markdown_content.append("## Statistical Summary\n")
    
    stats_data = []
    for class_label in classes:
        feature_importance, top_feature_indices = importance_per_class[class_label]
        class_name = label_names[class_label] if label_names else f"Class {class_label}"
        
        top_5_importance = feature_importance[top_feature_indices[:5]]
        stats_data.append([
            class_name,
            f"{np.max(feature_importance):.4f}",
            f"{np.mean(top_5_importance):.4f}",
            f"{np.std(top_5_importance):.4f}",
            f"{np.sum(top_5_importance):.4f}",
            len(top_feature_indices)
        ])
    
    stats_df = pd.DataFrame(stats_data, columns=[
        "Class", "Max Importance", "Mean Top 5", "Std Top 5", "Sum Top 5", "Total Features"
    ])
    markdown_content.append(stats_df.to_markdown(index=False))
    markdown_content.append("\n")
    
    # Join all content
    full_markdown = "\n".join(markdown_content)
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_markdown)
        print(f"Detailed feature importance tables saved to {output_file}")
    
    return full_markdown

def generate_feature_comparison_heatmap_table(
    importance_per_class: Dict[int, Tuple[np.ndarray, np.ndarray]], 
    label_names: Optional[List[str]] = None,
    top_n_features: int = 20
) -> str:
    """
    Generate a markdown table showing feature importance as a heatmap-style table
    
    Args:
        importance_per_class: Dictionary from analyze_latent_feature_importance_per_class
        label_names: Names of the classes (optional)
        top_n_features: Number of top features to include across all classes
        
    Returns:
        String containing markdown formatted heatmap table
    """
    classes = sorted(importance_per_class.keys())
    
    # Collect all top features across classes
    all_top_features = set()
    for class_label in classes:
        _, top_feature_indices = importance_per_class[class_label]
        all_top_features.update(top_feature_indices[:top_n_features])
    
    # Sort features by their maximum importance across all classes
    feature_max_importance = {}
    for feature_idx in all_top_features:
        max_imp = 0
        for class_label in classes:
            feature_importance, _ = importance_per_class[class_label]
            if feature_idx < len(feature_importance):
                max_imp = max(max_imp, feature_importance[feature_idx])
        feature_max_importance[feature_idx] = max_imp
    
    sorted_features = sorted(feature_max_importance.keys(), 
                           key=lambda x: feature_max_importance[x], reverse=True)[:top_n_features]
    
    # Create heatmap data
    heatmap_data = []
    headers = ["Feature"] + [label_names[c] if label_names else f"Class {c}" for c in classes]
    
    for feature_idx in sorted_features:
        row = [f"F{feature_idx}"]
        for class_label in classes:
            feature_importance, _ = importance_per_class[class_label]
            if feature_idx < len(feature_importance):
                importance = feature_importance[feature_idx]
                # Create visual indicator based on importance level
                if importance > 0.1:
                    row.append(f"HIGH {importance:.3f}")
                elif importance > 0.05:
                    row.append(f"MED {importance:.3f}")
                elif importance > 0.01:
                    row.append(f"LOW {importance:.3f}")
                else:
                    row.append(f"- {importance:.3f}")
            else:
                row.append("- 0.000")
        heatmap_data.append(row)
    
    heatmap_df = pd.DataFrame(heatmap_data, columns=headers)
    
    markdown_content = []
    markdown_content.append("# Feature Importance Heatmap\n")
    markdown_content.append("HIGH = High importance (>0.1), MED = Medium importance (>0.05), LOW = Low importance (>0.01), - = Very low\n")
    markdown_content.append(heatmap_df.to_markdown(index=False))
    
    return "\n".join(markdown_content)

def generate_samples_by_modifying_features_per_class(model, latent_reps, labels, importance_per_class, 
                                                   model_type='sae', input_shape=(28, 28), 
                                                   min_val=-3, max_val=3, n_steps=5, device='cuda',
                                                   use_class_mean=True):
    """
    Generate samples by modifying top features in latent space for each class
    
    Args:
        model: Trained model
        latent_reps: Latent representations
        labels: Labels for each sample
        importance_per_class: Dictionary from analyze_latent_feature_importance_per_class
        model_type: Type of model ('sae' or 'st')
        input_shape: Shape of input images
        min_val: Minimum value for feature modification
        max_val: Maximum value for feature modification
        n_steps: Number of steps between min and max
        device: Device to use for computation
        use_class_mean: Whether to use class-specific mean vectors as baselines
        
    Returns:
        Dictionary mapping class labels to dictionaries of {feature_idx: list of samples}
    """
    # Create dictionary to store results for each class
    all_class_samples = {}
    
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        # Get samples and importance info for this class
        class_mask = labels == label
        class_latent = latent_reps[class_mask]
        feature_importance, top_feature_indices = importance_per_class[label]
        
        # Get baseline latent representation for this class
        if use_class_mean:
            # Use the mean of this class
            base_latent = np.mean(class_latent, axis=0)
        else:
            # Use zeros
            base_latent = np.zeros(latent_reps.shape[1])
        
        # Dictionary to store results for this class
        modified_samples = {}
        
        # For each top feature
        for feature_idx in top_feature_indices:
            # Create a range of values to set for this feature
            feature_values = np.linspace(min_val, max_val, n_steps)
            
            # Generate samples with modified feature
            feature_samples = []
            
            for value in feature_values:
                # Copy the baseline latent representation
                modified_latent = base_latent.copy()
                
                # Set the feature to the current value
                modified_latent[feature_idx] = value
                
                # Reshape for batch dimension
                modified_latent = np.expand_dims(modified_latent, axis=0)
                
                # Reconstruct from modified latent
                reconstruction = reconstruct_from_feature_activations(
                    model, modified_latent, model_type=model_type, device=device)
                
                # Add to samples
                feature_samples.append(reconstruction[0])
            
            # Add to results
            modified_samples[feature_idx] = feature_samples
        
        # Store results for this class
        all_class_samples[label] = modified_samples
    
    return all_class_samples

def create_feature_importance_grid(model, latent_reps, labels, label_names, importance_per_class,
                                 model_type='sae', input_shape=(28, 28), 
                                 min_val=-3, max_val=3, n_steps=5, device='cuda',
                                 n_features_per_class=5, rows_per_class=3, output_dir='feature_grids'):
    """
    Create grid visualizations of the most important features for each class
    
    Args:
        model: Trained model
        latent_reps: Latent representations
        labels: Labels for each sample
        label_names: Names of labels
        importance_per_class: Output from analyze_latent_feature_importance_per_class
        model_type: Type of model ('sae' or 'st')
        input_shape: Shape of input images
        min_val: Minimum value for feature modification
        max_val: Maximum value for feature modification 
        n_steps: Number of steps between min and max
        device: Device to use for computation
        n_features_per_class: Number of top features to show per class
        rows_per_class: Number of rows/samples to show in grid for each class
        output_dir: Base directory to save output
        
    Returns:
        None (saves files to output_dir)
    """
    unique_labels = np.unique(labels)
    
    # Generate samples for each class and its top features
    all_class_samples = generate_samples_by_modifying_features_per_class(
        model, latent_reps, labels, importance_per_class, model_type, input_shape, 
        min_val, max_val, n_steps, device, use_class_mean=True)
    
    # Process each class
    for label in unique_labels:
        # Get class name or label
        class_name = label_names[label] if label_names is not None else f"Class_{label}"
        print(f"Creating visualizations for {class_name}...")
        
        # Create directory for this class
        class_dir = os.path.join(output_dir, f"{class_name}")
        os.makedirs(class_dir, exist_ok=True)
        
        # Get feature importance for this class
        feature_importance, top_features = importance_per_class[label]
        
        # Limit to the requested number of features
        top_features = top_features[:n_features_per_class]
        
        # Get samples for this class
        modified_samples = all_class_samples[label]
        
        # For each top feature, create a row in the grid
        for feature_idx in top_features:
            importance_score = feature_importance[feature_idx]
            samples = modified_samples[feature_idx]
            
            titles = [f"Value: {val:.2f}" for val in np.linspace(min_val, max_val, len(samples))]
            
            fig = create_grid_of_samples(
                samples, titles, input_shape, n_cols=len(samples),
                main_title=f"Feature {feature_idx} (Importance: {importance_score:.4f})"
            )
            
            # Save with importance score in filename for easier sorting
            fig.savefig(os.path.join(class_dir, f'feature_{feature_idx}_imp_{importance_score:.4f}.png'), dpi=300)
            plt.close(fig)
        
        # Create a combined grid with all top features
        combined_grid_rows = []
        feature_labels = []
        
        # Sort features by importance for better visualization
        sorted_features = sorted([(feature_idx, feature_importance[feature_idx]) 
                                 for feature_idx in top_features], 
                                key=lambda x: x[1], reverse=True)
        
        # Get just the sorted feature indices
        sorted_feature_indices = [f[0] for f in sorted_features]
        
        # Limit to rows_per_class if specified
        if rows_per_class > 0 and rows_per_class < len(sorted_feature_indices):
            sorted_feature_indices = sorted_feature_indices[:rows_per_class]
        
        for feature_idx in sorted_feature_indices:
            importance_score = feature_importance[feature_idx]
            samples = modified_samples[feature_idx]
            
            # Create a single row for this feature
            row_width = input_shape[1] * len(samples)
            row_height = input_shape[0]
            row = np.zeros((row_height, row_width))
            
            for j, sample in enumerate(samples):
                img = sample.reshape(input_shape)
                row[:, j*input_shape[1]:(j+1)*input_shape[1]] = img
            
            combined_grid_rows.append(row)
            feature_labels.append(f"Feature {feature_idx}: {importance_score:.4f}")
        
        # Stack rows to create the full grid
        if combined_grid_rows:
            full_grid = np.vstack(combined_grid_rows)
            
            plt.figure(figsize=(12, 2 * len(sorted_feature_indices)))
            plt.imshow(full_grid, cmap='gray')
            plt.title(f"Top Features for {class_name} (by Importance)")
            
            # Add labels for each row
            for i, label_text in enumerate(feature_labels):
                plt.text(-5, i * input_shape[0] + input_shape[0]/2, 
                        label_text, 
                        verticalalignment='center', horizontalalignment='right',
                        fontsize=9)
            
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(class_dir, f'all_top_features.png'), dpi=300)
            plt.close()
            
            # Create a summary file
            with open(os.path.join(class_dir, 'feature_importance_summary.txt'), 'w', encoding='utf-8') as f:
                f.write(f"Feature Importance Summary for {class_name}\n")
                f.write("="*50 + "\n\n")
                f.write("Features in descending order of importance:\n")
                for feature_idx, importance in sorted_features:
                    f.write(f"Feature {feature_idx}: {importance:.6f}\n")

def create_thought_vectors_dashboard_with_class_features(
    model, data, labels, model_type='sae', 
    input_shape=(28, 28), device='cuda',
    reduction_method='tsne', n_components=2,
    label_names=None, output_dir='thought_vectors_output', 
    n_features=5, n_features_per_class=5, min_val=-3, max_val=3, n_steps=5, 
    zero_baseline=True, samples_per_class=10, rows_per_class=3):
    """
    Create a complete dashboard for thought vectors visualization with class-specific feature analysis
    
    Args:
        model: Trained model
        data: Input data tensor
        labels: Labels tensor
        model_type: Type of model ('sae' or 'st')
        input_shape: Shape of input images
        device: Device to use for computation
        reduction_method: Dimensionality reduction method
        n_components: Number of components for reduction
        label_names: Names of the labels
        output_dir: Directory to save output
        n_features: Number of top features to analyze overall
        n_features_per_class: Number of top features to analyze per class
        min_val: Minimum value for feature variation
        max_val: Maximum value for feature variation
        n_steps: Number of steps for feature variation
        zero_baseline: Whether to use zero vector for overall baseline
        samples_per_class: Number of samples to show per class
        rows_per_class: Number of rows (features) to show in grid for each class
        
    Returns:
        None (saves files to output_dir)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Compute latent representations
    print("Computing latent representations...")
    latent_reps = compute_latent_representations(model, data, model_type, device=device)
    
    # 2. Reduce dimensions for visualization
    print(f"Reducing dimensions with {reduction_method}...")
    reduced_reps = reduce_dimensions(latent_reps, method=reduction_method, n_components=n_components)
    
    # 3. Visualize latent space with static plot
    print("Creating static visualization...")
    static_fig = visualize_latent_space_static(
        reduced_reps, labels.numpy(), label_names, 
        title=f"Latent Space Visualization ({model_type.upper()})")
    static_fig.savefig(os.path.join(output_dir, 'latent_space_static.png'), dpi=300)
    
    # 4. Create interactive visualization
    print("Creating interactive visualization...")
    interactive_fig, recon_fig = visualize_latent_space_interactive(
        reduced_reps, labels.numpy(), data, model, latent_reps, 
        input_shape, model_type, label_names)
    
    interactive_fig.write_html(os.path.join(output_dir, 'latent_space_interactive.html'))
    if recon_fig is not None:
        recon_fig.write_html(os.path.join(output_dir, 'sample_reconstructions.html'))
    
    # 5. Analyze overall feature importance
    print("Analyzing overall feature importance...")
    feature_importance, top_feature_indices = analyze_latent_feature_importance(
        latent_reps, labels.numpy(), n_features=n_features)
    
    # Save feature importance plot
    plt.figure(figsize=(10, 6))
    sorted_indices = np.argsort(feature_importance)[::-1]
    plt.bar(range(len(sorted_indices[:20])), feature_importance[sorted_indices[:20]])
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.title('Top 20 Feature Importance (All Classes)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance_overall.png'), dpi=300)
    plt.close()
    
    # 6. Analyze feature importance per class
    print("Analyzing feature importance per class...")
    importance_per_class = analyze_latent_feature_importance_per_class(
        latent_reps, labels.numpy(), n_features=n_features_per_class)
    
    # 6.5. Generate markdown tables for feature importance analysis
    print("Generating feature importance markdown tables...")
    
    # Generate detailed tables
    detailed_report = generate_feature_importance_markdown_tables(
        importance_per_class, 
        label_names=label_names, 
        n_top_features=n_features_per_class,
        output_file=os.path.join(output_dir, "feature_importance_detailed.md")
    )
    
    # Generate heatmap table
    heatmap_report = generate_feature_comparison_heatmap_table(
        importance_per_class, 
        label_names=label_names, 
        top_n_features=15
    )
    
    # Save heatmap report
    with open(os.path.join(output_dir, "feature_importance_heatmap.md"), 'w', encoding='utf-8') as f:
        f.write(heatmap_report)
    print(f"Feature importance heatmap table saved to {os.path.join(output_dir, 'feature_importance_heatmap.md')}")
    
    # Save combined report
    combined_report = detailed_report + "\n\n---\n\n" + heatmap_report
    with open(os.path.join(output_dir, "feature_importance_complete.md"), 'w', encoding='utf-8') as f:
        f.write(combined_report)
    print(f"Complete feature importance report saved to {os.path.join(output_dir, 'feature_importance_complete.md')}")
    
    # 7. Create class-specific feature visualizations
    print("Creating class-specific feature visualizations...")
    class_features_dir = os.path.join(output_dir, 'class_features')
    create_feature_importance_grid(
        model, latent_reps, labels.numpy(), label_names, importance_per_class,
        model_type, input_shape, min_val, max_val, n_steps, device,
        n_features_per_class, rows_per_class, class_features_dir
    )
    
    # 8. Generate overall samples by modifying features
    print("Generating overall samples by modifying features...")
    modified_samples = generate_samples_by_modifying_features(
        model, latent_reps, top_feature_indices[:n_features], model_type, input_shape, 
        min_val=min_val, max_val=max_val, n_steps=n_steps, device=device, 
        zero_baseline=zero_baseline)
    
    # Create a grid of samples for each top feature
    for feature_idx, samples in modified_samples.items():
        importance_score = feature_importance[feature_idx]
        titles = [f"Value: {val:.2f}" for val in np.linspace(min_val, max_val, len(samples))]
        fig = create_grid_of_samples(samples, titles, input_shape, n_cols=len(samples), 
                                    main_title=f"Feature {feature_idx} Variation (Importance: {importance_score:.4f})")
        # Save with importance score in filename for easier sorting
        fig.savefig(os.path.join(output_dir, f'feature_{feature_idx}_imp_{importance_score:.4f}_variation.png'), dpi=300)
        plt.close(fig)
    
    # 9. Create feature grid visualization
    print("Creating feature grid visualization...")
    try:
        feature_grid = create_sample_grid_from_features(
            model, latent_reps, top_feature_indices[:2], model_type, input_shape, device,
            min_val=min_val, max_val=max_val, n_steps=n_steps, zero_baseline=zero_baseline)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(feature_grid, cmap='gray')
        plt.title(f"Grid of Samples - Features {top_feature_indices[0]} vs {top_feature_indices[1]}")
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, 'feature_grid.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error creating feature grid: {e}")
    
    # 10. Generate interpolation animation
    print("Creating interpolation animation...")
    interpolation_fig = create_interpolation_animation(
        model, data, latent_reps, labels.numpy(), n_samples_per_class=samples_per_class, 
        model_type=model_type, input_shape=input_shape, device=device)
    
    if interpolation_fig is not None:
        interpolation_fig.write_html(os.path.join(output_dir, 'interpolation_animation.html'))
    
    print(f"Thought Vectors dashboard created in {output_dir}")
    print("Markdown reports generated:")
    print(f"  • Detailed analysis: {os.path.join(output_dir, 'feature_importance_detailed.md')}")
    print(f"  • Heatmap table: {os.path.join(output_dir, 'feature_importance_heatmap.md')}")
    print(f"  • Complete report: {os.path.join(output_dir, 'feature_importance_complete.md')}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Thought Vectors Visualization for Image Datasets')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained model file')
    parser.add_argument('--model_type', type=str, default='sae', choices=['sae', 'st'],
                      help='Type of model')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='mnist', 
                      help='Dataset name or path to custom dataset')
    parser.add_argument('--dataset_path', type=str, default=None,
                      help='Path to dataset file (default depends on dataset name)')
    parser.add_argument('--n_samples', type=int, default=1000,
                      help='Number of samples to use (default: 1000)')
    parser.add_argument('--input_shape', type=str, default='28,28',
                      help='Input shape (default: 28,28 for MNIST)')
    
    # Feature analysis parameters
    parser.add_argument('--n_features', type=int, default=5,
                      help='Number of top features to analyze overall (default: 5)')
    parser.add_argument('--n_features_per_class', type=int, default=5,
                      help='Number of top features to analyze per class (default: 5)')
    parser.add_argument('--samples_per_class', type=int, default=2,
                      help='Number of samples per class for interpolation (default: 2)')
    parser.add_argument('--rows_per_class', type=int, default=3,
                      help='Number of feature rows to show in grid for each class (default: 3)')
    
    # Feature variation parameters
    parser.add_argument('--min_val', type=float, default=-3.0,
                      help='Minimum value for feature variation (default: -3.0)')
    parser.add_argument('--max_val', type=float, default=3.0,
                      help='Maximum value for feature variation (default: 3.0)')
    parser.add_argument('--n_steps', type=int, default=5,
                      help='Number of steps for feature variation (default: 5)')
    parser.add_argument('--use_mean_baseline', action='store_true',
                      help='Use mean vector as baseline instead of zero vector')
    
    # Visualization parameters
    parser.add_argument('--reduction_method', type=str, default='tsne',
                      choices=['tsne', 'pca', 'umap'],
                      help='Dimensionality reduction method')
    parser.add_argument('--n_components', type=int, default=2,
                      help='Number of components for dimensionality reduction')
    parser.add_argument('--output_dir', type=str, default='thought_vectors_output',
                      help='Directory to save output')
    
    # Device parameters
    parser.add_argument('--device', type=str, default=None,
                      help='Device to use for computation (default: cuda if available, else cpu)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process input_shape
    args.input_shape = tuple(map(int, args.input_shape.split(',')))
    
    # Set device if not specified
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Set dataset path if not specified
    if args.dataset_path is None:
        if args.dataset.lower() == 'mnist':
            args.dataset_path = 'data/mnist_train.csv'
        elif args.dataset.lower() == 'fashion_mnist':
            args.dataset_path = 'data/fashion_mnist_train.csv'
        else:
            args.dataset_path = args.dataset
    
    return args

def main():
    """Main function"""
    args = parse_args()
    
    print(f"Using device: {args.device}")
    print(f"Model path: {args.model_path}")
    print(f"Model type: {args.model_type}")
    print(f"Dataset: {args.dataset}")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Feature variation: min={args.min_val}, max={args.max_val}, steps={args.n_steps}")
    print(f"Using {'mean' if args.use_mean_baseline else 'zero'} vector as baseline")
    print(f"Class-specific analysis: {args.n_features_per_class} features per class")
    print(f"Showing {args.rows_per_class} rows per class in grid visualizations")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = load_model(args.model_path, args.model_type, args.device)
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Load dataset
    if args.dataset.lower() == 'mnist':
        data, labels, label_names = load_mnist_dataset(args.dataset_path, args.n_samples, args.input_shape)
    else:
        data, labels, label_names = load_image_dataset(args.dataset_path, args.n_samples, args.input_shape)
    
    # Create the enhanced thought vectors dashboard
    create_thought_vectors_dashboard_with_class_features(
        model, data, labels, args.model_type, args.input_shape, args.device,
        args.reduction_method, args.n_components, label_names, args.output_dir, 
        args.n_features, args.n_features_per_class, args.min_val, args.max_val, args.n_steps, 
        not args.use_mean_baseline,  # zero_baseline is True when use_mean_baseline is False
        args.samples_per_class, args.rows_per_class
    )

if __name__ == "__main__":
    main()