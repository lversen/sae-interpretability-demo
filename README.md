# Sparse Neural Network Interpretability: A Comparative Analysis of Autoencoders and Transformers

**Master's Thesis in Applied Physics and Mathematics (Machine Learning & Statistics)**
UiT – The Arctic University of Norway, 2025

**Author:** Sebastian Iversen
**Contact:** sebive98@gmail.com
**Thesis:** [Read the full thesis (PDF)](thesis.pdf)

---

## Overview

This repository contains the complete implementation and experiments for my master's thesis investigating interpretability in large language models (LLMs) through sparse representation learning. The thesis compares two fundamental approaches to understanding internal representations in neural networks:

1. **Sparse Autoencoders (SAE)** — Traditional dimensionality reduction with sparsity constraints
2. **Sparse Transformers with Attention (ST)** — Attention-based memory mechanisms for sparse feature extraction

Both methods aim to decompose neural network activations into interpretable, sparse feature sets that reveal how models internally represent information.

---

## Key Concepts

### Sparse Autoencoders (SAE)
- Learn overcomplete sparse representations of neural activations
- Use L1 regularization to enforce sparsity
- Decoder weights reveal interpretable features
- Standard baseline for mechanistic interpretability

### Sparse Transformers (ST)
- Leverage attention mechanisms to select relevant memory vectors
- Learn which features to attend to for reconstruction
- Support different attention functions (softmax, ReLU+softmax)
- Investigate whether attention provides better interpretability than standard autoencoders

---

## Research Questions

- Can attention-based methods outperform autoencoders for interpretability?
- How do sparse features learned by SAE vs ST differ qualitatively?
- What role does the choice of attention function play?
- Can we identify interpretable features in both MNIST and GPT-Neo activations?

---

## Tech Stack

- **Python 3.11+**
- **PyTorch** — Deep learning framework
- **PyTorch AMP** — Mixed precision training
- **NumPy, Pandas** — Data manipulation
- **Matplotlib** — Visualization
- **scikit-learn** — Evaluation metrics
- **Gephi** — Graph visualization
- **sentence-transformers, UMAP** — Additional utilities

---

## Project Structure

```
Master-thesis/
│
├── src/                          # Source code
│   ├── models/                   # Core model implementations
│   │   ├── SAE.py               # Sparse Autoencoder
│   │   ├── ST.py                # Sparse Transformer (optimized)
│   │   ├── ST_old.py            # Legacy ST implementation
│   │   └── deadfeatures.py      # Dead feature tracking utility
│   │
│   ├── data/                     # Dataset loading and handling
│   │   ├── mnist.py             # MNIST dataset loader
│   │   ├── fashion_mnist.py     # Fashion-MNIST loader
│   │   ├── sample_handler.py    # Consistent sampling utilities
│   │   └── test_dataset.py      # Test dataset utilities
│   │
│   ├── analysis/                 # Analysis and evaluation scripts
│   │   ├── analyze_gptneo.py    # GPT-Neo activation analysis
│   │   ├── checkpoint_analyzer.py
│   │   ├── centroid_analysis.py
│   │   ├── clustering_centroids.py
│   │   ├── cluster_connectivity.py
│   │   ├── find_best_models.py
│   │   └── feature_extraction_with_store.py
│   │
│   ├── visualization/            # Plotting and visualization
│   │   ├── visualization.py
│   │   ├── model_visualization.py
│   │   ├── mnist_visualization_examples.py
│   │   ├── plot_decoder.py      # Decoder weight visualization
│   │   ├── plot_decoder_weights.py
│   │   ├── thought_vectors.py   # Feature activation analysis
│   │   └── ...
│   │
│   ├── graph/                    # Graph-based analysis
│   │   ├── gephi.py             # Gephi export utilities
│   │   ├── graph_models_gephi.py
│   │   └── create_model_graph.py
│   │
│   └── utils/                    # Utility functions
│       ├── download_model.py    # Model download utilities
│       └── generate_labels.py   # Label generation
│
├── scripts/                      # Training and execution scripts
│   ├── main.py                  # Main training script
│   ├── train_multiple.py        # Multi-configuration training
│   └── train_for_analyze_gptneo.py
│
├── data/                         # Datasets (CSV files)
│   ├── final_data.csv
│   ├── stack_exchange_train.csv
│   └── stack_exchange_val.csv
│
├── configs/                      # Configuration files
│   └── last_config.json         # Last used training config
│
├── docs/                         # Documentation and references
│   ├── training_summary.md      # Training results summary
│   ├── train_combinations.md    # Experiment configurations
│   └── references/              # Research papers
│       ├── scaling_laws.pdf
│       └── softmax_attention_is_a_fluke.pdf
│
├── .gitignore                    # Git ignore rules
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Getting Started

### Prerequisites

- Python 3.11 or higher
- CUDA-capable GPU (recommended for training)
- 16GB+ RAM recommended

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/lversen/Master-thesis.git
   cd Master-thesis
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Additional dependencies (if needed):
   ```bash
   pip install torch torchvision matplotlib pandas scikit-learn
   ```

3. **Prepare datasets:**
   Datasets are expected in the `data/` directory. For GPT-Neo experiments, activation CSV files should follow the naming convention:
   ```
   data/gptneo_gpt_neo_1.3B_layer{N}_train.csv
   data/gptneo_gpt_neo_1.3B_layer{N}_val.csv
   ```

---

## Running Experiments

### Training a Single Model

```bash
python scripts/main.py --dataset mnist --model sae --feature_dim 1600 --lambda_l1 5.0
```

### Training Multiple Configurations

```bash
python scripts/train_multiple.py
```

This will train models across various hyperparameter configurations (feature dimensions, learning rates, attention functions, etc.).

### Analyzing GPT-Neo Activations

```bash
python scripts/train_for_analyze_gptneo.py
```

### Visualizing Decoder Weights

```python
from src.visualization.plot_decoder import plot_decoder_matrix_columns
from src.models.SAE import SparseAutoencoder

# Load model
model = SparseAutoencoder.load("path/to/model.pth")

# Plot decoder weights
plot_decoder_matrix_columns(model, X, num_cols=10)
```

---

## Key Results

The thesis trained **252 model configurations** across:
- **Datasets:** MNIST, GPT-Neo layer activations (layers 8-14)
- **Models:** SAE and ST with varying feature dimensions (20, 50, 100, 200, 400, 800, 1600)
- **Attention functions:** Softmax, ReLU+Softmax
- **Regularization:** L1 penalties (2.5, 5.0, 10.0)

Results are summarized in [`docs/training_summary.md`](docs/training_summary.md).

---

## References

Key papers explored in this thesis:
- Scaling Laws for Neural Language Models ([`docs/references/scaling_laws.pdf`](docs/references/scaling_laws.pdf))
- Softmax Attention Analysis ([`docs/references/softmax_attention_is_a_fluke.pdf`](docs/references/softmax_attention_is_a_fluke.pdf))

---

## Contributing

This is a thesis project and not actively maintained for external contributions. However, if you find bugs or have suggestions, feel free to open an issue or contact me directly.

---

## Contact

**Sebastian Iversen**
sebive98@gmail.com
MSc Applied Physics and Mathematics (Machine Learning & Statistics)
UiT – The Arctic University of Norway

---

## License

This repository is made available for academic and educational purposes. No license is currently specified. If you use this code in your research, please cite the thesis.

---

## Acknowledgments

Thanks to my supervisors at UiT – The Arctic University of Norway for their guidance throughout this project.

---

**Last Updated:** April 2026
