import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.ensemble import RandomForestClassifier
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
            with open(os.path.join(class_dir, 'feature_importance_summary.txt'), 'w') as f:
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