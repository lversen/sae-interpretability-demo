# Train Combinations Script

A comprehensive tool for training, comparing, and organizing machine learning models with different parameter configurations.

## Key Features

- **Multi-parameter Grid Search**: Specify lists of values for any training parameter
- **Dual Organization System**:
  - **Performance-based Groups**: Models sorted by validation loss, sparsity, or other metrics
  - **Hierarchical Structure**: Models organized by model_type/dataset/activation/feature_dimension
- **Comprehensive Reporting**: Detailed summaries, metrics, and visualizations
- **Experiment Management**: Resume interrupted experiments and skip completed models

## Usage

### Basic Usage

```bash
python train_combinations.py \
  --model_types st \
  --batch_sizes 2048 4096 \
  --learning_rates 5e-5 1e-3 \
  --feature_dimensions 100 256
```

### Available Parameters

#### Model Configuration
- `--model_types`: Model types to train (`sae`, `st`, `both`)
- `--attention_fns`: Attention functions for ST models
- `--feature_dimensions`: Feature dimensions to try
- `--attention_dimensions`: Attention dimensions to try
- `--activations`: Activation functions to try
- `--l1_lambdas`: L1 regularization strengths to try

#### Training Configuration
- `--batch_sizes`: Batch sizes to try
- `--learning_rates`: Learning rates to try
- `--target_steps`: Training steps to try
- `--grad_accum_steps`: Gradient accumulation steps to try
- `--eval_freqs`: Evaluation frequencies to try
- `--auto_steps`: Use auto step calculation
- `--auto_steps_bases`: Base steps for auto step calculation

#### Architecture Options
- `--use_memory_bank`: Use memory bank approach for ST models
- `--use_old_st`: Use old ST implementation

#### Experiment Organization
- `--experiment_name`: Name of the experiment
- `--output_dir`: Base directory for storing results
- `--sort_metric`: Metric to use for sorting models
- `--num_groups`: Number of performance groups to create
- `--continue_from`: Continue from a previous experiment directory
- `--skip_completed`: Skip combinations that have already been trained

## Output Structure

```
experiments/
  └── my_experiment/
      ├── models/                 # All trained model files
      ├── hierarchy/              # Hierarchical organization
      │   ├── st/                 # Model type
      │   │   ├── mnist/          # Dataset
      │   │   │   ├── relu/       # Activation function
      │   │   │   │   ├── 100/    # Feature dimension
      │   │   │   │   │   └── st_model_*.pth
      │   │   │   │   └── 256/
      │   │   │   │       └── st_model_*.pth
      │   │   │   └── gelu/
      │   │   │       └── .../
      │   │   └── fashion_mnist/
      │   └── sae/
      │       └── .../
      ├── top_performers/         # Best performing models
      ├── medium_performers/      # Average performing models
      ├── low_performers/         # Low performing models
      ├── plots/                  # Visualization plots
      ├── metrics.csv             # Complete metrics for all models
      └── training_summary.md     # Detailed summary report
```

## Examples

### Comprehensive Parameter Sweep

```bash
python train_combinations.py \
  --model_types st \
  --batch_sizes 1024 2048 4096 \
  --learning_rates 1e-5 5e-5 1e-4 \
  --feature_dimensions 64 128 256 \
  --attention_fns softmax sparsemax \
  --activations relu gelu \
  --l1_lambdas 1.0 5.0 \
  --grad_accum_steps 1 2 \
  --experiment_name "full_parameter_sweep"
```

### Compare Memory vs Direct KV

```bash
# First run with direct KV (default)
python train_combinations.py \
  --model_types st \
  --batch_sizes 2048 4096 \
  --feature_dimensions 128 \
  --experiment_name "memory_vs_direct"

# Then run with memory bank
python train_combinations.py \
  --model_types st \
  --batch_sizes 2048 4096 \
  --feature_dimensions 128 \
  --experiment_name "memory_vs_direct" \
  --use_memory_bank \
  --continue_from experiments/memory_vs_direct \
  --skip_completed
```