#!/usr/bin/env python
"""
Enhanced GPT Neo Layer Analyzer (with reason.ipynb capabilities)

This script analyzes intermediate layer activations from transformer models,
using the exact approach from reason.ipynb, and supports sparse decomposition.

Features:
- EXACT same token concatenation and tracking as reason.ipynb
- Direct hidden state extraction without hooks
- Precise UMAP configurations to match reason.ipynb
- All cluster analysis capabilities from reason.ipynb
- Support for multiple model types (GPT-Neo, GPT2, OPT)
- Sparse decomposition (SAE or ST) capabilities

Example usage:
    python analyze_gptneo.py --model EleutherAI/gpt-neo-125m --visualize
    python analyze_gptneo.py --text_file my_texts.txt --analyze_clusters
    python analyze_gptneo.py --texts "Neural networks" "Language models" --create_gif
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import torch
import seaborn as sns
from tqdm import tqdm
import random
import time
import glob
from PIL import Image
from transformers import (
    GPTNeoForCausalLM, GPTNeoConfig, AutoTokenizer,
    GPT2LMHeadModel, GPT2Config, GPT2Tokenizer,
    AutoConfig, AutoModelForCausalLM
)
import umap.umap_ as umap
import networkx as nx
from sklearn.neighbors import kneighbors_graph

# Check if SAE and ST are available
try:
    from SAE import SparseAutoencoder
    from ST import SparseTransformer
    DECOMP_AVAILABLE = True
except ImportError:
    print("Warning: SAE or ST modules not found. Will use simplified decomposition.")
    DECOMP_AVAILABLE = False

# Create a simplified SAE if the original is not available
class SimplifiedSAE:
    """Simplified SAE implementation when the original is not available"""
    
    def __init__(self, n, m, device='cuda', lambda_l1=1.0):
        self.n = n  # Input dimension
        self.m = m  # Feature dimension
        self.device = device
        self.lambda_l1 = lambda_l1
        
        # Initialize weights
        self.encoder = torch.nn.Linear(n, m, bias=True).to(device)
        self.decoder = torch.nn.Linear(m, n, bias=True).to(device)
        
        # Initialize with sensible values
        torch.nn.init.xavier_uniform_(self.encoder.weight)
        torch.nn.init.xavier_uniform_(self.decoder.weight)
        
    def forward(self, x):
        # Encoder - ReLU activation
        h = torch.relu(self.encoder(x))
        # Decoder
        x_hat = self.decoder(h)
        return x_hat, h
    
    def train_and_validate(self, train_tensor, val_tensor, learning_rate=1e-3, 
                          batch_size=64, target_steps=5000):
        """Simple training loop"""
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=learning_rate
        )
        
        # Create simple dataloader
        train_dataset = torch.utils.data.TensorDataset(train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        print(f"Training simplified SAE for {target_steps} steps...")
        step = 0
        while step < target_steps:
            for batch in train_loader:
                x = batch[0]
                
                # Forward pass
                x_hat, h = self.forward(x)
                
                # Compute loss
                recon_loss = torch.mean((x_hat - x) ** 2)
                l1_loss = self.lambda_l1 * torch.mean(torch.abs(h))
                loss = recon_loss + l1_loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Log progress
                step += 1
                if step % 100 == 0:
                    print(f"Step {step}/{target_steps}, Loss: {loss.item():.6f}")
                
                if step >= target_steps:
                    break
        
        print("Training completed!")
    
    def feature_activations(self, x):
        """Get feature activations for input x"""
        with torch.no_grad():
            h = torch.relu(self.encoder(x))
        return h

# Create a simplified ST if the original is not available
class SimplifiedST:
    """Simplified ST implementation when the original is not available"""
    
    def __init__(self, X, n, m, a, device='cuda', lambda_l1=1.0):
        self.n = n  # Input dimension
        self.m = m  # Feature dimension
        self.a = a  # Attention dimension
        self.device = device
        self.lambda_l1 = lambda_l1
        
        # Initialize weights
        self.W_q = torch.nn.Linear(n, a, bias=True).to(device)
        self.W_k = torch.nn.Linear(n, a, bias=True).to(device)
        self.W_v = torch.nn.Linear(n, n, bias=True).to(device)
        
        # Initialize memory
        self.memory_values = torch.nn.Parameter(torch.randn(m, n).to(device))
        self.memory_keys = torch.nn.Parameter(torch.randn(m, a).to(device))
        
        # Initialize with sensible values
        torch.nn.init.xavier_uniform_(self.W_q.weight)
        torch.nn.init.xavier_uniform_(self.W_k.weight)
        torch.nn.init.xavier_uniform_(self.W_v.weight)
        torch.nn.init.xavier_uniform_(self.memory_values)
        torch.nn.init.xavier_uniform_(self.memory_keys)
    
    def forward(self, x):
        # Calculate queries
        queries = self.W_q(x)
        
        # Calculate attention scores
        attention_scores = torch.matmul(queries, self.memory_keys.t())
        
        # Apply softmax
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # Get output
        output = torch.matmul(attention_weights, self.memory_values)
        return output, attention_weights
    
    def train_and_validate(self, train_tensor, val_tensor, learning_rate=1e-3, 
                          batch_size=64, target_steps=5000):
        """Simple training loop"""
        optimizer = torch.optim.Adam([
            {'params': self.W_q.parameters()},
            {'params': self.W_k.parameters()},
            {'params': self.W_v.parameters()},
            {'params': self.memory_values},
            {'params': self.memory_keys}
        ], lr=learning_rate)
        
        # Create simple dataloader
        train_dataset = torch.utils.data.TensorDataset(train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        print(f"Training simplified ST for {target_steps} steps...")
        step = 0
        while step < target_steps:
            for batch in train_loader:
                x = batch[0]
                
                # Forward pass
                output, attention_weights = self.forward(x)
                
                # Compute loss
                recon_loss = torch.mean((output - x) ** 2)
                l1_loss = self.lambda_l1 * torch.mean(torch.abs(attention_weights))
                loss = recon_loss + l1_loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Log progress
                step += 1
                if step % 100 == 0:
                    print(f"Step {step}/{target_steps}, Loss: {loss.item():.6f}")
                
                if step >= target_steps:
                    break
        
        print("Training completed!")
    
    def feature_activations(self, x):
        """Get feature activations for input x"""
        with torch.no_grad():
            _, attention_weights = self.forward(x)
        return attention_weights

# New class with functions from reason.ipynb
class TokenClusteringUtils:
    """
    Utility class with clustering and visualization methods from reason.ipynb
    """
    @staticmethod
    def get_group_fractions(group_ids):
        """
        Calculates fractional positions within each group (for gradient coloring)
        
        Args:
            group_ids: Array of group IDs
            
        Returns:
            Array of same shape with fractional positions (0-1) within each group
        """
        fractions = np.zeros_like(group_ids, dtype=float)
        unique_groups = np.unique(group_ids)
        
        for g in unique_groups:
            g_indices = np.where(group_ids == g)[0]
            if len(g_indices) <= 1:
                fractions[g_indices] = 0.0
                continue
            local_positions = np.arange(len(g_indices))
            local_fractions = local_positions / (len(g_indices) - 1)
            fractions[g_indices] = local_fractions
        return fractions
    
    @staticmethod
    def blend_color_and_alpha(base_color, frac, 
                              start_alpha=0.2, end_alpha=1.0, 
                              light_factor=0.5):
        """
        Blend a base color with white based on fractional position and adjust alpha
        
        Args:
            base_color: RGB tuple (r, g, b)
            frac: Fractional position (0-1)
            start_alpha: Starting alpha value
            end_alpha: Ending alpha value
            light_factor: Light blending factor
            
        Returns:
            RGBA tuple with blended color and alpha
        """
        alpha = start_alpha + frac * (end_alpha - start_alpha)
        
        r0, g0, b0 = base_color
        white_blend = light_factor * (1.0 - frac)
        r = (1.0 * white_blend) + r0 * (1.0 - white_blend)
        g = (1.0 * white_blend) + g0 * (1.0 - white_blend)
        b = (1.0 * white_blend) + b0 * (1.0 - white_blend)
        
        return (r, g, b, alpha)
    
    @staticmethod
    def create_final_colors(group_ids, base_colors_rgb,
                            start_alpha=0.2, end_alpha=1.0, 
                            light_factor=0.5):
        """
        Create an array of colors for visualization with gradient effect within groups
        
        Args:
            group_ids: Array of group IDs
            base_colors_rgb: List of base colors for each group
            start_alpha: Starting alpha value
            end_alpha: Ending alpha value
            light_factor: Light blending factor
            
        Returns:
            Array of RGBA values for each point
        """
        fractions = TokenClusteringUtils.get_group_fractions(group_ids)
        final_colors = np.zeros((len(group_ids), 4), dtype=float)
        
        for i, g in enumerate(group_ids):
            base_color = base_colors_rgb[g % len(base_colors_rgb)]
            frac = fractions[i]
            rgba = TokenClusteringUtils.blend_color_and_alpha(
                base_color, frac,
                start_alpha=start_alpha,
                end_alpha=end_alpha,
                light_factor=light_factor
            )
            final_colors[i] = rgba
        return final_colors
    
    @staticmethod
    def compute_cluster_centroids(layer, group_ids):
        """
        Compute the mean hidden state (centroid) for each group
        
        Args:
            layer: 2D numpy array of shape [seq_len, hidden_size]
            group_ids: 1D array of length seq_len, each token's group id
            
        Returns:
            Dictionary {group_id: centroid vector}
        """
        unique_groups = np.unique(group_ids)
        centroids = {}
        
        for g in unique_groups:
            indices = np.where(group_ids == g)[0]
            centroids[g] = layer[indices].mean(axis=0)
            
        return centroids
    
    @staticmethod
    def compute_umap_centroids(umap_result, group_ids):
        """
        Compute centroids in UMAP reduced space
        
        Args:
            umap_result: UMAP reduced embeddings
            group_ids: Group IDs for each point
            
        Returns:
            Dictionary {group_id: centroid in UMAP space}
        """
        unique_groups = np.unique(group_ids)
        umap_centroids = {}
        
        for g in unique_groups:
            indices = np.where(group_ids == g)[0]
            umap_centroids[g] = umap_result[indices].mean(axis=0)
            
        return umap_centroids
    
    @staticmethod
    def compute_distance_matrix(centroids):
        """
        Compute a pairwise Euclidean distance matrix between centroids
        
        Args:
            centroids: Dictionary {group_id: centroid vector}
            
        Returns:
            Tuple with (groups list, distance matrix, average distance)
        """
        groups = sorted(centroids.keys())
        num_groups = len(groups)
        dist_matrix = np.zeros((num_groups, num_groups))
        
        for i, g1 in enumerate(groups):
            for j, g2 in enumerate(groups):
                if i == j:
                    dist_matrix[i, j] = 0.0
                else:
                    dist_matrix[i, j] = np.linalg.norm(centroids[g1] - centroids[g2])
        
        mean_rows = dist_matrix.mean(0)  # mean of each row
        avg_distance = mean_rows.mean(0)  # mean of all columns -> single value
        
        return groups, dist_matrix, avg_distance
    
    @staticmethod
    def create_graph(matrix, group_ids, k=5):
        """
        Creates a graph from embeddings using k-nearest neighbors
        
        Args:
            matrix: numpy array of shape [n_points, dim]
            group_ids: 1D array for each node's group ID
            k: number of neighbors for each node
            
        Returns:
            NetworkX Graph with node attributes
        """
        n_points = matrix.shape[0]

        adjacency = kneighbors_graph(
            matrix,
            n_neighbors=k,
            metric='euclidean',
            mode='distance',
            include_self=False
        ).tolil()

        G = nx.Graph()
        for i in range(n_points):
            G.add_node(i,
                    pos=(matrix[i, 0], matrix[i, 1]),
                    group=group_ids[i])

        for i in range(n_points):
            for j in adjacency.rows[i]:
                if not G.has_edge(i, j):
                    G.add_edge(i, j)

        return G
    
    @staticmethod
    def compute_modularity(G):
        """
        Compute the modularity of the graph based on node groups
        
        Args:
            G: NetworkX Graph where each node has a 'group' attribute
            
        Returns:
            Modularity value (float)
        """
        # Dictionary grouping nodes by their group id
        communities_dict = {}
        for node in G.nodes():
            group = G.nodes[node]['group']
            if group not in communities_dict:
                communities_dict[group] = set()
            communities_dict[group].add(node)
        
        # List of communities
        communities = list(communities_dict.values())
        
        # Modularity using networkx function
        mod = nx.algorithms.community.quality.modularity(G, communities)
        return mod
    
    @staticmethod
    def visualize_graph(G, output_path, layer_nr, base_colors_rgb):
        """
        Visualize the graph with node colors based on group IDs
        
        Args:
            G: NetworkX Graph with 'pos' and 'group' node attributes
            output_path: Directory to save the image
            layer_nr: Layer number (for filename)
            base_colors_rgb: List of RGB tuples for group colors
        """
        pos = nx.get_node_attributes(G, 'pos')
        
        node_groups = [G.nodes[node]['group'] for node in G.nodes()]
        node_colors = [base_colors_rgb[group % len(base_colors_rgb)] for group in node_groups]
        
        plt.figure(figsize=(8, 8))
        
        nx.draw_networkx_edges(G, pos, edge_color='black', alpha=0.9, width=1)
        
        nx.draw_networkx_nodes(
            G, pos,
            node_size=10,
            node_color=node_colors
        )
        
        unique_groups = sorted(set(node_groups))
        legend_handles = []
        for group in unique_groups:
            color = base_colors_rgb[group % len(base_colors_rgb)]
            patch = mpatches.Patch(color=color, label=f"Group {group}")
            legend_handles.append(patch)
        plt.legend(handles=legend_handles, title="Group ID", loc="best")
        
        plt.title(f"Graph from 2D UMAP - Layer {layer_nr}")
        plt.axis("off")
        plt.tight_layout()
        
        save_path = f"{output_path}/graph_layer{layer_nr}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def create_gif_from_images(image_folder, output_path, duration=1000):
        """
        Create an animated GIF from a series of images
        
        Args:
            image_folder: Folder containing images
            output_path: Path to save the resulting GIF
            duration: Frame duration in milliseconds
        """
        png_files = glob.glob(os.path.join(image_folder, "layer*.png"))
        png_files = sorted(png_files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].replace("layer", "")))
        
        frames = []
        for file in png_files:
            frame = Image.open(file)
            frames.append(frame)
        
        if frames:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:], 
                duration=duration,  
                loop=1
            )
            print(f"GIF successfully saved to {output_path}")
        else:
            print("No images found in the folder!")


class ModelAnalyzer:
    """
    Enhanced analyzer for extracting and analyzing intermediate layer activations from transformer models.
    This class follows the exact approach from reason.ipynb.
    """
    
    def __init__(self, model_name="EleutherAI/gpt-neo-125m", device=None, local_model_path=None):
        """
        Initialize the analyzer
        
        Args:
            model_name: HuggingFace model name or path
            device: Computing device ('cuda' or 'cpu')
            local_model_path: Path to a locally downloaded model
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        # Set extended timeouts via environment variables
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # 5 minutes
        os.environ["TRANSFORMERS_REQUEST_TIMEOUT"] = "300"  # 5 minutes
        
        # Determine model path and type
        self.using_local_model = local_model_path is not None
        model_path = local_model_path if self.using_local_model else model_name
        self.model_path = model_path
        
        # Detect model type from name (just like in reason.ipynb)
        if 'gpt-neo' in model_name.lower():
            self.model_type = 'gpt-neo'
        elif 'gpt2' in model_name.lower():
            self.model_type = 'gpt2'
        elif 'opt' in model_name.lower():
            self.model_type = 'opt'
        else:
            self.model_type = 'auto'  # Try to auto-detect
        
        print(f"Loading model '{model_name}' (type: {self.model_type}) on {device}...")
        
        try:
            # Load tokenizer
            if self.using_local_model:
                # Check for tokenizer in different possible locations
                if os.path.exists(os.path.join(local_model_path, "tokenizer")):
                    tokenizer_path = os.path.join(local_model_path, "tokenizer")
                else:
                    tokenizer_path = local_model_path
                
                print(f"Loading tokenizer from {tokenizer_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path,
                    local_files_only=True
                )
            else:
                # Select appropriate tokenizer based on model type
                if self.model_type == 'gpt-neo' or self.model_type == 'gpt2':
                    self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
                else:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Ensure padding token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with proper configuration for getting all hidden states
            if self.using_local_model:
                # Check for model in different possible locations
                if os.path.exists(os.path.join(local_model_path, "model")):
                    model_weights_path = os.path.join(local_model_path, "model")
                else:
                    model_weights_path = local_model_path
                
                print(f"Loading model from {model_weights_path}")
                # Load configuration first to set output_hidden_states
                if self.model_type == 'gpt-neo':
                    config = GPTNeoConfig.from_pretrained(
                        model_weights_path, 
                        output_hidden_states=True
                    )
                    self.model = GPTNeoForCausalLM.from_pretrained(
                        model_weights_path,
                        config=config,
                        local_files_only=True,
                        low_cpu_mem_usage=True,
                        torch_dtype=torch.float16 if device == 'cuda' else torch.float32
                    )
                elif self.model_type == 'gpt2':
                    config = GPT2Config.from_pretrained(
                        model_weights_path, 
                        output_hidden_states=True
                    )
                    self.model = GPT2LMHeadModel.from_pretrained(
                        model_weights_path,
                        config=config,
                        local_files_only=True,
                        low_cpu_mem_usage=True,
                        torch_dtype=torch.float16 if device == 'cuda' else torch.float32
                    )
                else:
                    config = AutoConfig.from_pretrained(
                        model_weights_path, 
                        output_hidden_states=True
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_weights_path,
                        config=config,
                        local_files_only=True,
                        low_cpu_mem_usage=True,
                        torch_dtype=torch.float16 if device == 'cuda' else torch.float32
                    )
            else:
                # Load from HuggingFace with appropriate configuration
                if self.model_type == 'gpt-neo':
                    config = GPTNeoConfig.from_pretrained(model_name, output_hidden_states=True)
                    self.model = GPTNeoForCausalLM.from_pretrained(
                        model_name,
                        config=config,
                        low_cpu_mem_usage=True,
                        torch_dtype=torch.float16 if device == 'cuda' else torch.float32
                    )
                elif self.model_type == 'gpt2':
                    config = GPT2Config.from_pretrained(model_name, output_hidden_states=True)
                    self.model = GPT2LMHeadModel.from_pretrained(
                        model_name,
                        config=config,
                        low_cpu_mem_usage=True,
                        torch_dtype=torch.float16 if device == 'cuda' else torch.float32
                    )
                else:
                    config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        config=config,
                        low_cpu_mem_usage=True,
                        torch_dtype=torch.float16 if device == 'cuda' else torch.float32
                    )
            
            self.model = self.model.to(device)
            self.model.eval()  # Set to evaluation mode
            
            # Get number of layers
            try:
                if self.model_type == 'gpt-neo':
                    self.num_layers = len(self.model.transformer.h)
                elif self.model_type == 'gpt2':
                    self.num_layers = len(self.model.transformer.h)
                elif self.model_type == 'opt':
                    self.num_layers = len(self.model.model.decoder.layers)
                else:
                    # Try to identify model structure
                    if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                        self.num_layers = len(self.model.transformer.h)
                    elif hasattr(self.model, 'model') and hasattr(self.model.model, 'decoder') and hasattr(self.model.model.decoder, 'layers'):
                        self.num_layers = len(self.model.model.decoder.layers)
                    else:
                        self.num_layers = 12  # Default fallback
                print(f"Model has {self.num_layers} layers")
            except AttributeError:
                # For different model architectures
                print("Couldn't determine number of layers. Using default range of 12 layers.")
                self.num_layers = 12
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("\nTROUBLESHOOTING SUGGESTIONS:")
            print("1. Download the model locally first using download_model.py:")
            print("   python download_model.py --model EleutherAI/gpt-neo-125m --output models/gpt-neo-125m")
            print("   Then use: --local_model_path models/gpt-neo-125m")
            print("2. Try a smaller model like 'gpt2' or 'distilgpt2'")
            print("3. Check your internet connection")
            print("4. If on a corporate network, set proxy environment variables:")
            print("   export HTTPS_PROXY=http://proxy-server:port")
            raise
        
        # Class variables for color and visualization
        self.base_colors_hex = [
            "#1f77b4",  # blue
            "#ff7f0e",  # orange
            "#2ca02c",  # green
            "#d62728",  # red
            "#9467bd",  # purple
            "#8c564b",  # brown
            "#e377c2",  # pink
            "#bcbd22",  # olive
        ]
        self.base_colors_rgb = [mcolors.to_rgb(h) for h in self.base_colors_hex]
        
        # Store layer graphs for latent analysis
        self.layer_graphs = {}
    
    def process_texts(self, texts, layer_indices=None):
        """
        Process multiple texts and extract all hidden states.
        Uses reason.ipynb approach: concatenate all texts with a space,
        and use cumulative sums to track token positions.
        
        Args:
            texts: List of texts to analyze
            layer_indices: List of layer indices to extract (defaults to all layers)
            
        Returns:
            Tuple of (hidden_states, token_to_text_map, input_texts_token_lengths, input_text)
        """
        # Concatenate all texts with a space, exactly like reason.ipynb
        input_text = " ".join(texts)
        
        # Tokenize and get token lengths for each paragraph
        input_texts_token_lengths = [
            len(self.tokenizer.encode(paragraph))
            for paragraph in texts
        ]
        print("Token lengths per paragraph:", input_texts_token_lengths)
        
        # Calculate cumulative sums for tracking token positions
        cumulative_lengths = np.cumsum(input_texts_token_lengths)
        print("Cumulative sums:", cumulative_lengths)
        
        # Encode the entire text at once
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        
        # Run the model with output_hidden_states=True to get all layer states
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Extract all hidden states
        all_hidden_states = outputs.hidden_states
        
        # Get total sequence length
        seq_len = all_hidden_states[0].size(1)
        token_indices = np.arange(seq_len)
        
        # Assign text group IDs to tokens using searchsorted (exact match to reason.ipynb)
        group_ids = np.searchsorted(cumulative_lengths, token_indices, side="right")
        
        # Filter hidden states if layer_indices is specified
        if layer_indices is not None:
            # Filter to only include requested layers
            hidden_states = {
                f"layer_{i}": all_hidden_states[i].cpu().squeeze().numpy()
                for i in layer_indices if i < len(all_hidden_states)
            }
        else:
            # Include all layers
            hidden_states = {
                f"layer_{i}": all_hidden_states[i].cpu().squeeze().numpy()
                for i in range(len(all_hidden_states))
            }
        
        return hidden_states, group_ids, input_texts_token_lengths, input_text
    
    def apply_sparse_decomposition(self, activations, decomposition_type="sae", 
                                  feature_dim=None, **kwargs):
        """
        Apply sparse decomposition to layer activations
        
        Args:
            activations: Activation matrix (n_samples, hidden_dim)
            decomposition_type: Type of decomposition ('sae' or 'st')
            feature_dim: Feature dimension, if None uses hidden_dim / 4
            **kwargs: Additional arguments for decomposition model
            
        Returns:
            Decomposition model and feature activations
        """
        # Get dimensions
        n_samples, hidden_dim = activations.shape
        
        # Default feature dimension to hidden_dim / 4 if not specified
        if feature_dim is None:
            feature_dim = max(100, hidden_dim // 4)
            print(f"Using feature dimension: {feature_dim}")
        
        # Convert to PyTorch tensor
        activation_tensor = torch.from_numpy(activations).float().to(self.device)
        
        # Create decomposition model
        if decomposition_type.lower() == "sae":
            # Create SAE model - use original if available, otherwise use simplified
            if DECOMP_AVAILABLE:
                model = SparseAutoencoder(
                    n=hidden_dim,
                    m=feature_dim,
                    device=self.device,
                    **kwargs
                )
            else:
                print("Using simplified SAE implementation")
                model = SimplifiedSAE(
                    n=hidden_dim,
                    m=feature_dim,
                    device=self.device,
                    lambda_l1=kwargs.get('lambda_l1', 1.0)
                )
        elif decomposition_type.lower() == "st":
            # Calculate attention dimension to be balanced with SAE param count
            a = max(20, hidden_dim // 8)
            print(f"Using attention dimension: {a}")
            
            # Create ST model - use original if available, otherwise use simplified
            if DECOMP_AVAILABLE:
                model = SparseTransformer(
                    X=activations,
                    n=hidden_dim,
                    m=feature_dim,
                    a=a,
                    device=self.device,
                    **kwargs
                )
            else:
                print("Using simplified ST implementation")
                model = SimplifiedST(
                    X=activations,
                    n=hidden_dim,
                    m=feature_dim,
                    a=a,
                    device=self.device,
                    lambda_l1=kwargs.get('lambda_l1', 1.0)
                )
        else:
            raise ValueError(f"Unknown decomposition type: {decomposition_type}")
        
        print(f"Training {decomposition_type.upper()} model...")
        # Split for training and validation
        split_idx = max(1, int(n_samples * 0.8))
        train_tensor = activation_tensor[:split_idx]
        val_tensor = activation_tensor[split_idx:]
        
        # Train model
        model.train_and_validate(
            train_tensor,
            val_tensor,
            batch_size=min(64, n_samples),
            learning_rate=1e-4,
            target_steps=5000,  # Reduced for demonstration
        )
        
        # Get feature activations
        with torch.no_grad():
            feature_activations = model.feature_activations(activation_tensor)
            feature_activations = feature_activations.cpu().numpy()
        
        print(f"Feature activations shape: {feature_activations.shape}")
        return model, feature_activations
    
    def visualize_activations(self, activations, title="Layer Activations", 
                            figsize=(12, 8), plot_type="umap", 
                            token_to_text_map=None, texts=None,
                            output_path=None):
        """
        Visualize activations using various plot types with color coding by source text
        
        Args:
            activations: Activation matrix
            title: Plot title
            figsize: Figure size
            plot_type: Type of plot ('heatmap', 'umap', 'histogram')
            token_to_text_map: Array mapping each token to its source text index
            texts: List of original texts (for legend labels)
            output_path: Path to save the visualization (if None, just displays)
        """
        plt.figure(figsize=figsize)
        
        # Set up colors - create a color map that's visually distinct
        if token_to_text_map is not None:
            num_texts = len(np.unique(token_to_text_map))
            cmap = plt.cm.get_cmap('tab20' if num_texts <= 20 else 'tab20b')
            colors = [cmap(i % 20) for i in range(num_texts)]
            
            # Create legend labels
            if texts:
                legend_labels = [f"Text {i+1}: {text[:30]}..." for i, text in enumerate(texts)]
            else:
                legend_labels = [f"Text {i+1}" for i in range(num_texts)]
        
        if plot_type == "heatmap":
            # For heatmap, rearrange tokens by text source for clearer visualization
            if token_to_text_map is not None:
                # Sort by text source
                sort_indices = np.argsort(token_to_text_map)
                sorted_activations = activations[sort_indices]
                
                # Plot heatmap with horizontal lines between texts
                ax = sns.heatmap(sorted_activations[:min(500, sorted_activations.shape[0])], 
                            cmap='viridis', 
                            robust=True)
                
                # Add lines between different texts
                text_boundaries = np.where(np.diff(token_to_text_map[sort_indices]) != 0)[0]
                for boundary in text_boundaries:
                    if boundary < 500:  # Only show boundaries within the displayed range
                        plt.axhline(y=boundary + 0.5, color='red', linestyle='-', linewidth=1)
            else:
                # Standard heatmap without color coding
                sns.heatmap(activations[:min(500, activations.shape[0])], 
                            cmap='viridis', 
                            robust=True)
            
            plt.title(f"{title} Heatmap")
            plt.xlabel("Dimension")
            plt.ylabel("Token")
                
        elif plot_type == "umap":
            # Apply UMAP for dimensionality reduction with exact reason.ipynb parameters
            reducer = umap.UMAP(
                n_neighbors=5,
                n_components=2,
                metric='euclidean',
                repulsion_strength=2,
                random_state=42
            )
            embedding = reducer.fit_transform(activations)
            
            if token_to_text_map is not None:
                # Create gradient colors within each group
                final_colors = TokenClusteringUtils.create_final_colors(
                    token_to_text_map,
                    self.base_colors_rgb,
                    start_alpha=0.2,
                    end_alpha=1.0,
                    light_factor=0.5
                )
                
                # Plot with enhanced colors
                plt.scatter(embedding[:, 0], embedding[:, 1], 
                        s=20, c=final_colors)
                
                # For legend, we need solid colors for each group
                unique_groups = np.unique(token_to_text_map)
                legend_handles = []
                for i, g in enumerate(unique_groups):
                    color = self.base_colors_rgb[i % len(self.base_colors_rgb)]
                    patch = mpatches.Patch(color=color, label=legend_labels[i])
                    legend_handles.append(patch)
                
                # Add legend with appropriate size
                if len(unique_groups) <= 10:
                    plt.legend(handles=legend_handles, fontsize='small')
                else:
                    # Create a separate legend figure if too many texts
                    leg = plt.legend(handles=legend_handles, fontsize='x-small',
                                  loc='upper left', bbox_to_anchor=(1.05, 1))
                
                # Get centroids in UMAP space
                umap_centroids = TokenClusteringUtils.compute_umap_centroids(embedding, token_to_text_map)
                
                # Add text labels at centroids
                for g, centroid in umap_centroids.items():
                    label_color = self.base_colors_rgb[g % len(self.base_colors_rgb)]
                    plt.text(centroid[0], centroid[1], f"{g}",
                            fontsize=14, fontweight='bold',
                            color="black")
                
                # Compute distance matrix between centroids
                groups, dist_matrix, avg_dist = TokenClusteringUtils.compute_distance_matrix(umap_centroids)
                
                # Create graph for modularity calculation
                G = TokenClusteringUtils.create_graph(embedding, token_to_text_map, k=5)
                modularity = TokenClusteringUtils.compute_modularity(G)
                
                # Store graph for later use
                layer_name = title.replace(" Raw Activations", "").replace(" ", "_").lower()
                self.layer_graphs[layer_name] = G
                
                # Add metrics to plot title
                plt.title(f"{title} UMAP - Modularity: {modularity:.3f}, Avg Distance: {avg_dist:.3f}")
                
            else:
                plt.scatter(embedding[:, 0], embedding[:, 1], s=5, alpha=0.5)
                plt.title(f"{title} UMAP")
            
            plt.xlabel("UMAP 1")
            plt.ylabel("UMAP 2")
                
        elif plot_type == "histogram":
            # For histogram, use stacked or grouped histograms by text
            if token_to_text_map is not None:
                # Create stacked histograms
                num_texts = len(np.unique(token_to_text_map))
                for i in range(num_texts):
                    mask = token_to_text_map == i
                    plt.hist(activations[mask].flatten(), bins=50, alpha=0.3, 
                            label=legend_labels[i], color=self.base_colors_rgb[i % len(self.base_colors_rgb)])
                plt.legend(fontsize='small')
            else:
                # Standard histogram without color coding
                plt.hist(activations.flatten(), bins=50, alpha=0.7)
            
            plt.title(f"{title} Histogram")
            plt.xlabel("Activation Value")
            plt.ylabel("Frequency")
                
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
        
        plt.tight_layout()
        
        # Save if output path is specified
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def save_activations(self, activations, filename):
        """Save activations to file"""
        directory = "activations"
        os.makedirs(directory, exist_ok=True)
        full_path = os.path.join(directory, filename)
        np.save(full_path, activations)
        print(f"Saved activations to {full_path}")
        
    def analyze_feature_correlations(self, feature_activations, threshold=0.7):
        """
        Analyze correlations between features
        
        Args:
            feature_activations: Feature activation matrix
            threshold: Correlation threshold
            
        Returns:
            Correlation matrix and highly correlated features
        """
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(feature_activations.T)
        
        # Find highly correlated features
        high_corr = []
        for i in range(corr_matrix.shape[0]):
            for j in range(i+1, corr_matrix.shape[1]):
                if abs(corr_matrix[i, j]) > threshold:
                    high_corr.append((i, j, corr_matrix[i, j]))
        
        # Sort by correlation strength
        high_corr.sort(key=lambda x: abs(x[2]), reverse=True)
        
        return corr_matrix, high_corr
    
    def analyze_clusters(self, layer_activations, token_to_text_map, texts=None, 
                        output_dir="cluster_analysis"):
        """
        Perform extensive cluster analysis exactly like reason.ipynb
        
        Args:
            layer_activations: Dictionary of layer activations
            token_to_text_map: Array mapping tokens to their source texts
            texts: Original input texts
            output_dir: Directory to save results
            
        Returns:
            Dictionary with analysis metrics
        """
        print("\n" + "="*50)
        print("PERFORMING CLUSTER ANALYSIS")
        print("="*50)
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        graphs_path = os.path.join(output_dir, "graphs")
        layers_path = os.path.join(output_dir, "layers")
        latent_graphs_path = os.path.join(output_dir, "latent_graphs")
        
        os.makedirs(graphs_path, exist_ok=True)
        os.makedirs(layers_path, exist_ok=True)
        os.makedirs(latent_graphs_path, exist_ok=True)
        
        # Store metrics for each layer
        avg_dists_full = []
        avg_dists_umap = []
        layer_modularities = []
        
        # For tracking layers
        layer_names = sorted(layer_activations.keys())
        
        # Process each layer
        for layer_name in layer_names:
            layer_nr = int(layer_name.split('_')[1])  # Extract layer number
            print(f"\nProcessing {layer_name}...")
            
            # Get activations
            activations = layer_activations[layer_name]
            
            # Calculate centroids and distances in full embedding space
            orig_centroids = TokenClusteringUtils.compute_cluster_centroids(activations, token_to_text_map)
            groups, orig_dist_matrix, avg_dist_full = TokenClusteringUtils.compute_distance_matrix(orig_centroids)
            avg_dists_full.append(avg_dist_full)
            
            # Apply UMAP for dimensionality reduction with exact reason.ipynb parameters
            reducer = umap.UMAP(
                n_neighbors=5,
                n_components=2,
                metric='euclidean',
                repulsion_strength=2,
                random_state=42
            )
            umap_result = reducer.fit_transform(activations)
            
            # Create graph in latent space (full dimensionality)
            G_latent = TokenClusteringUtils.create_graph(activations, token_to_text_map, k=5)
            self.layer_graphs[layer_name] = G_latent
            modularity = TokenClusteringUtils.compute_modularity(G_latent)
            layer_modularities.append(modularity)
            
            # Create graph in UMAP space for visualization
            G_umap = TokenClusteringUtils.create_graph(umap_result, token_to_text_map, k=5)
            
            # Calculate centroids and distances in UMAP space
            umap_centroids = TokenClusteringUtils.compute_umap_centroids(umap_result, token_to_text_map)
            umap_groups, umap_dist_matrix, avg_dist_umap = TokenClusteringUtils.compute_distance_matrix(umap_centroids)
            avg_dists_umap.append(avg_dist_umap)
            
            # Create enhanced colors for visualization
            final_colors = TokenClusteringUtils.create_final_colors(
                token_to_text_map,
                self.base_colors_rgb,
                start_alpha=0.2,
                end_alpha=1.0,
                light_factor=0.5
            )
            
            # Create multi-panel visualization (like in reason.ipynb)
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))
            
            # Plot UMAP scatter with color-coded tokens
            ax1.scatter(umap_result[:, 0], umap_result[:, 1], c=final_colors, s=20)
            
            # Add centroid labels
            for g, centroid in umap_centroids.items():
                label_color = self.base_colors_rgb[g % len(self.base_colors_rgb)]
                ax1.text(centroid[0], centroid[1], f"{g}",
                        fontsize=14, fontweight='bold',
                        color="black")
            
            # Add legend
            legend_handles = []
            unique_groups = sorted(np.unique(token_to_text_map))
            for g in unique_groups:
                label_color = self.base_colors_rgb[g % len(self.base_colors_rgb)]
                if texts:
                    label_text = f"Text {g+1}: {texts[g][:20]}..."
                else:
                    label_text = f"Group {g}"
                patch = mpatches.Patch(color=label_color, label=label_text)
                legend_handles.append(patch)
            ax1.legend(handles=legend_handles, title="Groups", bbox_to_anchor=(1.05, 1), loc="upper left")
            
            # Set title
            model_name = self.model_type.upper()
            ax1.set_title(f"{model_name} UMAP – Layer: {layer_nr}")
            ax1.axis("off")
            
            # Plot distance heatmaps
            sns.heatmap(orig_dist_matrix, annot=True, fmt=".2f", cmap="viridis",
                        xticklabels=groups, yticklabels=groups, ax=ax2)
            ax2.set_title(f"Embedded Space Centroid Distances – Layer: {layer_nr}")
            
            sns.heatmap(umap_dist_matrix, annot=True, fmt=".2f", cmap="viridis",
                        xticklabels=umap_groups, yticklabels=umap_groups, ax=ax3)
            ax3.set_title(f"UMAP (2D) Centroid Distances – Layer: {layer_nr}")
            
            # Save the figure
            plt.tight_layout()
            save_path = f"{layers_path}/layer{layer_nr}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Create and save graph visualization
            TokenClusteringUtils.visualize_graph(G_umap, graphs_path, layer_nr, self.base_colors_rgb)
            
            # Export latent graphs for Gephi
            for node in G_latent.nodes():
                # Convert position attributes to strings for GEXF compatibility
                pos = G_latent.nodes[node].get("pos")
                if isinstance(pos, (list, tuple)):
                    G_latent.nodes[node]["pos"] = ",".join(str(x) for x in pos)
            
            nx.write_gexf(G_latent, f"{latent_graphs_path}/layer{layer_nr}.gexf")
        
        # Create summary plots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12))
        
        # Plot average distances in full space
        x_values_full = list(range(len(avg_dists_full)))
        ax1.plot(x_values_full, avg_dists_full, marker='o', linestyle='-', linewidth=2, color='blue')
        ax1.set_title("Average Distances (Full Embedding Space)")
        ax1.set_xlabel("Layer")
        ax1.set_ylabel("Average Distance")
        ax1.set_xticks(x_values_full)
        ax1.set_xticklabels([str(x) for x in x_values_full])
        ax1.grid(True)
        
        # Plot average distances in UMAP space
        x_values_umap = list(range(len(avg_dists_umap)))
        ax2.plot(x_values_umap, avg_dists_umap, marker='o', linestyle='-', linewidth=2, color='green')
        ax2.set_title("Average Distances (UMAP Space)")
        ax2.set_xlabel("Layer")
        ax2.set_ylabel("Average Distance")
        ax2.set_xticks(x_values_umap)
        ax2.set_xticklabels([str(x) for x in x_values_umap])
        ax2.grid(True)
        
        # Plot modularity values
        x_values_mod = list(range(len(layer_modularities)))
        ax3.plot(x_values_mod, layer_modularities, marker='o', linestyle='-', linewidth=2, color='purple')
        ax3.set_title("Layer Modularities")
        ax3.set_xlabel("Layer")
        ax3.set_ylabel("Modularity")
        ax3.set_xticks(x_values_mod)
        ax3.set_xticklabels([str(x) for x in x_values_mod])
        ax3.grid(True)
        
        plt.tight_layout()
        save_path = f"{output_dir}/measurements.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create animated GIF from layer visualizations
        TokenClusteringUtils.create_gif_from_images(
            layers_path,
            f"{output_dir}/layer_progression.gif",
            duration=1000
        )
        
        # Return analysis summary
        return {
            "avg_dists_full": avg_dists_full,
            "avg_dists_umap": avg_dists_umap,
            "modularities": layer_modularities,
            "best_layer_idx": np.argmax(layer_modularities),
            "best_modularity": np.max(layer_modularities)
        }
    
    def create_gephi_visualization(self, feature_activations, output_path,
                                  n_neighbors=10, min_edge_weight=0.5):
        """
        Create Gephi visualization from feature activations
        
        Args:
            feature_activations: Feature activation matrix
            output_path: Path to save GEXF file
            n_neighbors: Number of neighbors for each node
            min_edge_weight: Minimum correlation for edge creation
        """
        # Create DataFrames needed for gephi graph creation
        n_features = feature_activations.shape[1]
        feature_df = pd.DataFrame({
            'feature_id': [f"feature_{i}" for i in range(n_features)],
            'value': [i for i in range(n_features)]
        })
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(feature_activations.T)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes with attributes
        for i in range(n_features):
            G.add_node(i, 
                      id=f"feature_{i}",
                      label=f"Feature {i}",
                      category=i % 10)  # Assign category for coloring
        
        # Add edges based on correlation
        edges_added = 0
        
        # First approach: Add edges for top N neighbors of each node
        for i in range(n_features):
            # Get correlations for this feature with all others
            correlations = [(j, abs(corr_matrix[i, j])) for j in range(n_features) if i != j]
            # Sort by correlation strength
            correlations.sort(key=lambda x: x[1], reverse=True)
            # Add edges for top neighbors
            for j, weight in correlations[:n_neighbors]:
                if weight >= min_edge_weight:
                    G.add_edge(i, j, weight=float(weight))
                    edges_added += 1
        
        print(f"Added {edges_added} edges to graph")
        
        # Export to GEXF
        nx.write_gexf(G, output_path)
        print(f"Created Gephi graph at {output_path}")


def parse_args():
    """Parse command-line arguments for GPT Neo layer analysis"""
    parser = argparse.ArgumentParser(description='Enhanced Transformer Layer Analyzer with reason.ipynb capabilities')
    
    # Model selection
    parser.add_argument('--model', type=str, default='EleutherAI/gpt-neo-125m',
                      help='Model to analyze (HuggingFace model name or path)')
    parser.add_argument('--local_model_path', type=str, default=None,
                      help='Path to a locally downloaded model (prioritized over --model if provided)')
    parser.add_argument('--device', type=str, default=None,
                      help='Device to use for computation (cuda or cpu)')
    parser.add_argument('--use_smaller_model', action='store_true',
                      help='Use a smaller model (gpt2) for faster download and analysis')
    
    # Input texts
    input_group = parser.add_argument_group('Input Options')
    input_group.add_argument('--text_file', type=str, default=None,
                      help='File containing texts to analyze (one per line)')
    input_group.add_argument('--texts', type=str, nargs='+', default=None,
                      help='Direct text inputs to analyze')
    input_group.add_argument('--generate_samples', action='store_true',
                      help='Generate sample texts covering different domains')
    input_group.add_argument('--duplicate_text', action='store_true',
                      help='Duplicate the same text multiple times to analyze clustering of identical content')
    
    # Layer selection
    layer_group = parser.add_argument_group('Layer Selection')
    layer_group.add_argument('--layers', type=int, nargs='+', default=None,
                      help='Layer indices to analyze (default: all layers)')
    layer_group.add_argument('--all_layers', action='store_true',
                      help='Analyze all layers in the model (default behavior)')
    
    # Decomposition options
    decomp_group = parser.add_argument_group('Decomposition Options')
    decomp_group.add_argument('--decomposition', type=str, default='sae',
                      choices=['sae', 'st', 'both', 'none'],
                      help='Type of sparse decomposition to apply')
    decomp_group.add_argument('--feature_dim', type=int, default=None,
                      help='Feature dimension for decomposition (default: input_dim/4)')
    decomp_group.add_argument('--l1_lambda', type=float, default=5.0,
                      help='L1 regularization strength')
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--visualize', action='store_true',
                      help='Visualize activations and decomposition results')
    output_group.add_argument('--save_activations', action='store_true',
                      help='Save activations to file')
    output_group.add_argument('--create_graph', action='store_true',
                      help='Create Gephi graph visualization')
    output_group.add_argument('--create_gif', action='store_true',
                      help='Create animated GIF of layer progression')
    output_group.add_argument('--output_dir', type=str, default="analysis_results",
                      help='Directory to save results')
    
    # Analysis options
    analysis_group = parser.add_argument_group('Analysis Options')
    analysis_group.add_argument('--analyze_features', action='store_true',
                      help='Perform detailed analysis on learned features')
    analysis_group.add_argument('--analyze_clusters', action='store_true',
                      help='Perform cluster analysis like in reason.ipynb')
    
    # Timeout and connection options
    connection_group = parser.add_argument_group('Connection Options')
    connection_group.add_argument('--timeout', type=int, default=300,
                      help='Request timeout in seconds (default: 300)')
    connection_group.add_argument('--max_retries', type=int, default=3,
                      help='Maximum number of retries for failed downloads (default: 3)')
    
    args = parser.parse_args()
    
    # Set environment variables for timeouts
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(args.timeout)
    os.environ["TRANSFORMERS_REQUEST_TIMEOUT"] = str(args.timeout)
    
    # Use smaller model if requested
    if args.use_smaller_model:
        print("Using smaller gpt2 model for faster download and analysis")
        args.model = "gpt2"
    
    # Set device if not specified
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return args


def main():
    """
    Main function for enhanced transformer layer analysis tool.
    Implements the exact same approach as reason.ipynb.
    """
    # Parse command-line arguments
    args = parse_args()
    
    # Set up output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*50)
    print("ENHANCED TRANSFORMER LAYER ANALYSIS")
    print("="*50)
    
    if args.local_model_path:
        print(f"Local model path: {args.local_model_path}")
    else:
        print(f"Model: {args.model}")
    
    print(f"Decomposition: {args.decomposition}")
    print(f"Device: {args.device}")
    print(f"Output directory: {args.output_dir}")
    
    # Create model analyzer with robust error handling
    try:
        analyzer = ModelAnalyzer(
            model_name=args.model, 
            device=args.device,
            local_model_path=args.local_model_path
        )
    except Exception as e:
        print(f"\nERROR: Failed to initialize model: {str(e)}")
        print("\nRECOMMENDED SOLUTION:")
        print("1. First download the model locally using snapshot_download:")
        print("   python download_model.py --model EleutherAI/gpt-neo-125m")
        print("\n2. Then use the downloaded model path:")
        print("   python analyze_gptneo.py --local_model_path models/gpt-neo-125m --text_file your_texts.txt")
        print("\n3. Or try a smaller model:")
        print("   python analyze_gptneo.py --use_smaller_model --text_file your_texts.txt")
        return
    
    # Get input texts
    texts = []
    
    if args.text_file:
        # Load texts from file
        try:
            print(f"Loading texts from {args.text_file}")
            with open(args.text_file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(texts)} texts")
        except Exception as e:
            print(f"Error loading text file: {e}")
            print("Using default sample text instead")
            texts = ["The transformer architecture revolutionized natural language processing."]
    elif args.texts:
        # Use provided texts
        texts = args.texts
        print(f"Using {len(texts)} provided texts")
    elif args.generate_samples:
        # Use diverse sample texts
        texts = [
            # Technical content
            "Neural networks process information through layers of interconnected nodes, each applying weights and activation functions to transform input data.",
            # Creative writing
            "The old oak tree stood sentinel at the edge of the forest, its gnarled branches reaching skyward like ancient fingers.",
            # News article
            "Scientists announced today the discovery of a new exoplanet that may contain liquid water, raising hopes for finding extraterrestrial life.",
            # Casual conversation
            "Hey, did you catch that new movie last weekend? I thought the special effects were amazing but the plot was predictable.",
            # Academic writing
            "The experiment yielded statistically significant results (p<0.01), suggesting a strong correlation between the variables under investigation.",
            # Historical text
            "In 1776, representatives from the thirteen colonies signed the Declaration of Independence, formally announcing their separation from Britain.",
            # Business writing
            "The quarterly financial report indicates a 12% increase in revenue, driven primarily by strong performance in emerging markets.",
            # Medical text
            "Patients presenting with these symptoms should be evaluated for possible autoimmune disorders, particularly those affecting the thyroid.",
            # Legal text
            "The plaintiff alleges that the defendant breached the terms of the contract by failing to deliver the specified goods within the timeframe.",
            # Technical documentation
            "To install the package, run 'pip install library-name' and import the required modules into your Python script."
        ]
        print(f"Using {len(texts)} generated sample texts")
    else:
        # Use default text
        texts = ["The transformer architecture revolutionized natural language processing by introducing self-attention mechanisms that capture long-range dependencies in text data."]
        print("Using default sample text")
    
    # Duplicate text if requested (for analyzing identical content)
    if args.duplicate_text and len(texts) > 0:
        first_text = texts[0]
        texts = [first_text] * 8  # Create 8 copies
        print(f"Duplicated the same text {len(texts)} times")
    
    # Limit number of texts if too many (to avoid memory issues)
    max_texts = 20
    if len(texts) > max_texts:
        print(f"Warning: Limited to {max_texts} texts to avoid memory issues")
        texts = texts[:max_texts]
    
    print(f"\nProcessing {len(texts)} texts through model...")
    
    try:
        # If all_layers is true or no layers specified, use all layers
        if args.all_layers or args.layers is None:
            # When using all layers, use range from 0 to num_layers
            layer_indices = list(range(analyzer.num_layers + 1))  # +1 to include the final layer
            print(f"Analyzing all {len(layer_indices)} layers (including embedding layer)")
        else:
            layer_indices = args.layers
            print(f"Analyzing {len(layer_indices)} specified layers: {layer_indices}")
            
        # Process texts using reason.ipynb approach (concatenate all texts)
        hidden_states, token_to_text_map, token_lengths, full_text = analyzer.process_texts(
            texts, layer_indices)
    
        # Print token information
        total_tokens = sum(token_lengths)
        print(f"\nAnalyzed {total_tokens} tokens across {len(texts)} texts")
        
        print(f"\nExtracted hidden states from {len(hidden_states)} layers")
        for layer_name, activations in hidden_states.items():
            print(f"  {layer_name}: shape {activations.shape}")
    except Exception as e:
        print(f"\nError during activation extraction: {e}")
        print("Try using a smaller model or fewer/shorter texts")
        return
    
    # Save raw activations if requested
    if args.save_activations:
        activations_dir = os.path.join(args.output_dir, "activations")
        os.makedirs(activations_dir, exist_ok=True)
        
        print("\nSaving raw activations...")
        for layer_name, activations in hidden_states.items():
            output_path = f"{activations_dir}/{layer_name}_raw.npy"
            np.save(output_path, activations)
            print(f"  Saved to {output_path}")
    
    # Visualize raw activations if requested
    if args.visualize:
        try:
            viz_dir = os.path.join(args.output_dir, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            
            print("\nVisualizing raw activations...")
            for layer_name, activations in hidden_states.items():
                print(f"  Visualizing {layer_name}")
                
                # UMAP visualization with exact reason.ipynb parameters
                output_path = f"{viz_dir}/{layer_name}_umap.png"
                analyzer.visualize_activations(
                    activations, 
                    f"{layer_name} Raw Activations",
                    plot_type="umap",
                    token_to_text_map=token_to_text_map,
                    texts=texts,
                    output_path=output_path
                )
                
                # Heatmap
                output_path = f"{viz_dir}/{layer_name}_heatmap.png"
                analyzer.visualize_activations(
                    activations, 
                    f"{layer_name} Raw Activations",
                    plot_type="heatmap",
                    token_to_text_map=token_to_text_map,
                    texts=texts,
                    output_path=output_path
                )
                
                # Histogram
                output_path = f"{viz_dir}/{layer_name}_histogram.png"
                analyzer.visualize_activations(
                    activations, 
                    f"{layer_name} Raw Activations",
                    plot_type="histogram",
                    token_to_text_map=token_to_text_map,
                    texts=texts,
                    output_path=output_path
                )
        except Exception as e:
            print(f"Error during visualization: {e}")
            print("Continuing with analysis...")
    
    # Apply sparse decomposition if requested
    decomposed_features = {}
    if args.decomposition != 'none':
        print("\n" + "="*50)
        print(f"Applying {args.decomposition.upper()} decomposition at token level")
        print("="*50)
        
        decomp_dir = os.path.join(args.output_dir, "decomposition")
        os.makedirs(decomp_dir, exist_ok=True)
        
        # Process each layer
        for layer_name, activations in hidden_states.items():
            print(f"\nProcessing layer: {layer_name} with {activations.shape[0]} tokens")
            
            # Calculate feature dimension if not specified
            feature_dim = args.feature_dim
            if feature_dim is None:
                # Default to input_dim / 4 with a minimum of 100 features
                feature_dim = max(100, activations.shape[1] // 4)
                print(f"Auto-selected feature dimension: {feature_dim}")
            
            try:
                if args.decomposition in ['sae', 'both']:
                    print("\n  Applying SAE decomposition...")
                    sae_model, sae_features = analyzer.apply_sparse_decomposition(
                        activations,
                        decomposition_type="sae",
                        feature_dim=feature_dim,
                        lambda_l1=args.l1_lambda
                    )
                    
                    # Store the features
                    decomposed_features[f"{layer_name}_sae"] = sae_features
                    
                    # Calculate sparsity
                    sparsity = np.mean(sae_features == 0) * 100
                    print(f"  SAE features shape: {sae_features.shape}, sparsity: {sparsity:.2f}%")
                    
                    # Save features if requested
                    if args.save_activations:
                        output_path = f"{decomp_dir}/{layer_name}_sae_features.npy"
                        np.save(output_path, sae_features)
                        print(f"  Saved SAE features to {output_path}")
                    
                    # Visualize features
                    if args.visualize:
                        print("  Visualizing SAE features...")
                        
                        # UMAP visualization with exact reason.ipynb parameters
                        output_path = f"{decomp_dir}/{layer_name}_sae_umap.png"
                        analyzer.visualize_activations(
                            sae_features, 
                            f"{layer_name} SAE Features",
                            plot_type="umap",
                            token_to_text_map=token_to_text_map,
                            texts=texts,
                            output_path=output_path
                        )
                        
                        # Heatmap
                        output_path = f"{decomp_dir}/{layer_name}_sae_heatmap.png"
                        analyzer.visualize_activations(
                            sae_features, 
                            f"{layer_name} SAE Features",
                            plot_type="heatmap",
                            token_to_text_map=token_to_text_map,
                            texts=texts,
                            output_path=output_path
                        )
                    
                    # Create graph visualization
                    if args.create_graph:
                        graph_dir = os.path.join(args.output_dir, "graphs")
                        os.makedirs(graph_dir, exist_ok=True)
                        
                        output_path = f"{graph_dir}/{layer_name}_sae_graph.gexf"
                        print(f"  Creating SAE graph visualization: {output_path}")
                        analyzer.create_gephi_visualization(
                            sae_features,
                            output_path,
                            n_neighbors=10,
                            min_edge_weight=0.5
                        )
                
                if args.decomposition in ['st', 'both']:
                    print("\n  Applying ST decomposition...")
                    st_model, st_features = analyzer.apply_sparse_decomposition(
                        activations,
                        decomposition_type="st",
                        feature_dim=feature_dim,
                        lambda_l1=args.l1_lambda
                    )
                    
                    # Store the features
                    decomposed_features[f"{layer_name}_st"] = st_features
                    
                    # Calculate sparsity
                    sparsity = np.mean(st_features == 0) * 100
                    print(f"  ST features shape: {st_features.shape}, sparsity: {sparsity:.2f}%")
                    
                    # Save features if requested
                    if args.save_activations:
                        output_path = f"{decomp_dir}/{layer_name}_st_features.npy"
                        np.save(output_path, st_features)
                        print(f"  Saved ST features to {output_path}")
                    
                    # Visualize features
                    if args.visualize:
                        print("  Visualizing ST features...")
                        
                        # UMAP with reason.ipynb parameters
                        output_path = f"{decomp_dir}/{layer_name}_st_umap.png"
                        analyzer.visualize_activations(
                            st_features, 
                            f"{layer_name} ST Features",
                            plot_type="umap",
                            token_to_text_map=token_to_text_map,
                            texts=texts,
                            output_path=output_path
                        )
                        
                        # Heatmap
                        output_path = f"{decomp_dir}/{layer_name}_st_heatmap.png"
                        analyzer.visualize_activations(
                            st_features, 
                            f"{layer_name} ST Features",
                            plot_type="heatmap",
                            token_to_text_map=token_to_text_map,
                            texts=texts,
                            output_path=output_path
                        )
                    
                    # Create graph visualization
                    if args.create_graph:
                        graph_dir = os.path.join(args.output_dir, "graphs")
                        os.makedirs(graph_dir, exist_ok=True)
                        
                        output_path = f"{graph_dir}/{layer_name}_st_graph.gexf"
                        print(f"  Creating ST graph visualization: {output_path}")
                        analyzer.create_gephi_visualization(
                            st_features,
                            output_path,
                            n_neighbors=10,
                            min_edge_weight=0.5
                        )
                        
            except Exception as e:
                print(f"Error during decomposition of {layer_name}: {e}")
                print("  Skipping this layer and continuing with others...")
                continue
    
    # Perform feature analysis if we have decomposed features
    if decomposed_features and args.analyze_features:
        print("\n" + "="*50)
        print("FEATURE ANALYSIS")
        print("="*50)
        
        feature_dir = os.path.join(args.output_dir, "feature_analysis")
        os.makedirs(feature_dir, exist_ok=True)
        
        for name, features in decomposed_features.items():
            print(f"\nAnalyzing {name} features:")
            
            try:
                # Find highly correlated features
                corr_matrix, high_corr = analyzer.analyze_feature_correlations(features, threshold=0.7)
                print(f"Found {len(high_corr)} highly correlated feature pairs (threshold=0.7)")
                
                if len(high_corr) > 0:
                    print("Top 5 correlated feature pairs:")
                    for i, (feat1, feat2, corr) in enumerate(high_corr[:5]):
                        print(f"  {i+1}. Features {feat1} and {feat2}: {corr:.4f}")
                
                # Analyze feature activations across texts
                print("\nFeature activation patterns:")
                # Get the top 5 most activated features across all texts
                mean_activations = np.mean(features, axis=0)
                top_features = np.argsort(-mean_activations)[:5]
                
                print("Top 5 most activated features:")
                for i, feat_idx in enumerate(top_features):
                    print(f"  {i+1}. Feature {feat_idx}: {mean_activations[feat_idx]:.4f}")
                
                # Calculate percentage of dead features (never activated)
                dead_features = np.sum(np.max(features, axis=0) == 0)
                print(f"Dead features: {dead_features} ({dead_features/features.shape[1]*100:.2f}%)")
                
                # Plot correlation matrix
                plt.figure(figsize=(10, 10))
                sns.heatmap(corr_matrix, cmap="coolwarm", center=0, xticklabels=False, yticklabels=False)
                plt.title(f"Feature Correlation Matrix - {name}")
                plt.tight_layout()
                plt.savefig(f"{feature_dir}/{name}_correlation_matrix.png", dpi=150, bbox_inches='tight')
                plt.close()
                
                # Plot mean feature activations
                plt.figure(figsize=(12, 6))
                plt.bar(range(len(mean_activations)), mean_activations)
                plt.title(f"Mean Feature Activations - {name}")
                plt.xlabel("Feature Index")
                plt.ylabel("Mean Activation")
                plt.tight_layout()
                plt.savefig(f"{feature_dir}/{name}_mean_activations.png", dpi=150, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                print(f"Error during feature analysis: {e}")
                print("  Skipping and continuing...")
    
    # Perform cluster analysis if requested
    if args.analyze_clusters:
        cluster_dir = os.path.join(args.output_dir, "cluster_analysis")
        
        # Run detailed cluster analysis similar to reason.ipynb
        cluster_metrics = analyzer.analyze_clusters(
            hidden_states,
            token_to_text_map,
            texts=texts,
            output_dir=cluster_dir
        )
        
        # Print summary of cluster analysis
        print("\n" + "="*50)
        print("CLUSTER ANALYSIS SUMMARY")
        print("="*50)
        print(f"Best layer: layer_{cluster_metrics['best_layer_idx']} with modularity {cluster_metrics['best_modularity']:.4f}")
        print(f"Results saved to {cluster_dir}")
        
        # Create GIF if requested
        if args.create_gif:
            gif_path = os.path.join(args.output_dir, "layer_progression.gif")
            TokenClusteringUtils.create_gif_from_images(
                os.path.join(cluster_dir, "layers"),
                gif_path
            )
            print(f"Created layer progression GIF at {gif_path}")
    
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)
    print(f"Processed {len(texts)} texts through {len(hidden_states)} layers")
    if args.decomposition != 'none':
        print(f"Applied {args.decomposition} decomposition to all layers")
    
    print("\nResults saved to:")
    print(f"  - {args.output_dir}/")
    
    if args.create_graph:
        print("\nTo visualize the graphs:")
        print("1. Open Gephi (https://gephi.org/)")
        print("2. File > Open > [select .gexf file from the graphs directory]")
        print("3. In Overview tab, apply a layout (e.g., ForceAtlas 2)")
        print("4. In Preview tab, adjust visualization settings and export")
    
    print("\nNext steps:")
    print("- Try a different model or decomposition method")
    print("- Analyze more/different text inputs")
    print("- Explore other layers (use --layers option)")
    print("- Try --duplicate_text to analyze how identical texts cluster")
    print("- Use --analyze_clusters for detailed clustering metrics")
    print("- Create animated GIFs with --create_gif")

if __name__ == "__main__":
    main()