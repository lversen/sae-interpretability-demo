#!/usr/bin/env python
"""
Enhanced GPT Neo Layer Analyzer (with support for pre-trained SAE/ST models)

This script analyzes intermediate layer activations from transformer models,
using the exact approach from reason.ipynb, and supports loading pre-trained
sparse decomposition models (SAE or ST).

Features:
- EXACT same token concatenation and tracking as reason.ipynb
- Direct hidden state extraction without hooks
- Precise UMAP configurations to match reason.ipynb
- All cluster analysis capabilities from reason.ipynb
- Support for multiple model types (GPT-Neo, GPT2, OPT)
- Sparse decomposition (SAE or ST) capabilities
- NEW: Loading pre-trained SAE/ST models from specified paths

Example usage:
    # Standard usage
    python analyze_gptneo.py --model EleutherAI/gpt-neo-125m --visualize
    
    # Using pre-trained SAE models
    python analyze_gptneo.py --model EleutherAI/gpt-neo-125m --decomposition sae --sae_model_path models/sae/
    
    # Using pre-trained ST models with specific naming pattern
    python analyze_gptneo.py --model EleutherAI/gpt-neo-125m --decomposition st --st_model_path models/st/ --model_pattern "layer_{layer_num}_{decomp_type}"
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
    
    def __init__(self, n, m, sae_model_path=None, device='cuda', lambda_l1=1.0):
        self.n = n  # Input dimension
        self.m = m  # Feature dimension
        self.device = device
        self.lambda_l1 = lambda_l1
        self.sae_model_path = sae_model_path
        
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
        
    def resume_from_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        try:
            print(f"Loading simplified SAE from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # If it's a full checkpoint with state_dict
                encoder_state = {k.replace('encoder.', ''): v for k, v in checkpoint['model_state_dict'].items() if k.startswith('encoder.')}
                decoder_state = {k.replace('decoder.', ''): v for k, v in checkpoint['model_state_dict'].items() if k.startswith('decoder.')}
                
                self.encoder.load_state_dict(encoder_state)
                self.decoder.load_state_dict(decoder_state)
                
                if 'lambda_l1' in checkpoint:
                    self.lambda_l1 = checkpoint['lambda_l1']
                
                print(f"Successfully loaded checkpoint with lambda_l1 = {self.lambda_l1}")
            else:
                # Just weights, try direct loading
                self.load_state_dict(checkpoint)
                print("Loaded model weights directly")
            
            return True
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False

# Create a simplified ST if the original is not available
class SimplifiedST:
    """Simplified ST implementation when the original is not available"""
    
    def __init__(self, X, n, m, a, st_model_path=None, device='cuda', lambda_l1=1.0):
        self.n = n  # Input dimension
        self.m = m  # Feature dimension
        self.a = a  # Attention dimension
        self.device = device
        self.lambda_l1 = lambda_l1
        self.st_model_path = st_model_path
        
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
                          batch_size=64, target_steps=5000, resume_from=None):
        """Simple training loop with option to resume"""
        # Check if we should resume from checkpoint
        if resume_from and os.path.exists(resume_from):
            try:
                self.resume_from_checkpoint(resume_from)
                return self
            except Exception as e:
                print(f"Error resuming from checkpoint: {e}")
                print("Will train from scratch instead")
        
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
        return self
    
    def feature_activations(self, x):
        """Get feature activations for input x"""
        with torch.no_grad():
            _, attention_weights = self.forward(x)
        return attention_weights
    
    def resume_from_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        try:
            print(f"Loading simplified ST from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.load_state_dict(checkpoint['model_state_dict'])
                
                if 'lambda_l1' in checkpoint:
                    self.lambda_l1 = checkpoint['lambda_l1']
                
                print(f"Successfully loaded checkpoint with lambda_l1 = {self.lambda_l1}")
            else:
                # Just weights
                self.load_state_dict(checkpoint)
                print("Loaded model weights directly")
            
            return True
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False

# New class with functions from reason.ipynb
class TokenClusteringUtils:
    """
    Utility class with clustering and visualization methods from reason.ipynb
    """
    @staticmethod
    def visualize_distance_matrix(groups, dist_matrix, avg_distance, output_path, layer_nr):
        """
        Visualize the distance matrix between cluster centroids as a heatmap
        
        Args:
            groups: List of group IDs
            dist_matrix: Pairwise distance matrix
            avg_distance: Average distance value
            output_path: Directory to save the image
            layer_nr: Layer number (for filename)
        """
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(
            dist_matrix,
            annot=True,
            fmt=".2f",
            cmap="viridis_r",  # reversed viridis (darker = closer)
            xticklabels=[f"G{g}" for g in groups],
            yticklabels=[f"G{g}" for g in groups]
        )
        
        plt.title(f"Cluster Distance Matrix - Layer {layer_nr}\nAvg Distance: {avg_distance:.4f}")
        plt.xlabel("Group ID")
        plt.ylabel("Group ID")
        plt.tight_layout()
        
        # Save the figure
        save_path = f"{output_path}/distance_matrix_layer{layer_nr}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Distance matrix visualization saved to {save_path}")
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
    def create_gif_from_images(image_folder, output_path, duration=1000, pattern="*_umap.png"):
        """
        Create an animated GIF from a series of images
        
        Args:
            image_folder: Folder containing images
            output_path: Path to save the resulting GIF
            duration: Frame duration in milliseconds
            pattern: File pattern to match (default: "*_umap.png")
        """
        png_files = glob.glob(os.path.join(image_folder, pattern))
        
        # Extract layer numbers for sorting
        def get_layer_num(filename):
            basename = os.path.basename(filename)
            # Look for "layer_X" pattern and extract X
            import re
            match = re.search(r'layer_(\d+)', basename)
            if match:
                return int(match.group(1))
            return 0
        
        png_files = sorted(png_files, key=get_layer_num)
        
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
            print(f"No images found in {image_folder} matching pattern {pattern}!")


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
    
    
    def visualize_activations(self, activations, title="", plot_type="umap", token_to_text_map=None, texts=None, output_path=None):
        """
        Visualize activations using UMAP
        """
        try:
            import umap
            import matplotlib.pyplot as plt
            import os
            
            # Reduce dimensionality with UMAP
            reducer = umap.UMAP(
                n_neighbors=15,
                min_dist=0.1,
                n_components=2,
                metric='euclidean',
                random_state=42
            )
            
            # Fit and transform
            embedding = reducer.fit_transform(activations)
            
            # Create plot
            plt.figure(figsize=(10, 8))
            
            # Color points by text group if available
            if token_to_text_map is not None:
                # Get unique groups
                unique_groups = sorted(set(token_to_text_map))
                colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_groups)))
                
                # Plot each group with a different color
                for i, group in enumerate(unique_groups):
                    mask = token_to_text_map == group
                    plt.scatter(
                        embedding[mask, 0],
                        embedding[mask, 1],
                        c=[colors[i]],
                        label=f"Text {group+1}" if texts else f"Group {group+1}",
                        alpha=0.7
                    )
                plt.legend()
            else:
                # No groups, use single color
                plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.7)
            
            plt.title(title)
            plt.tight_layout()
            
            # Save if output path is provided
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"Visualization saved to {output_path}")
            
            # Display if requested
            plt.close()
            
        except Exception as e:
            print(f"Error during visualization: {e}")
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
        
    def get_model_path_for_layer(self, layer_name, decomp_type, model_base_path, model_pattern=None):
        """
        Get the appropriate model path for a given layer and decomposition type
        
        Args:
            layer_name: Name of the layer (e.g., "layer_0")
            decomp_type: Type of decomposition ("sae" or "st")
            model_base_path: Base path where models are stored
            model_pattern: Pattern for model filenames (can include {layer_num} and {decomp_type})
            
        Returns:
            Path to the model file, or None if not found
        """
        if not model_base_path:
            return None
            
        # Extract layer number from layer name
        try:
            layer_num = int(layer_name.split('_')[1])
        except (IndexError, ValueError):
            layer_num = 0
            
        # Default patterns if none provided
        if not model_pattern:
            # Try different common patterns
            patterns = [
                f"{layer_name}_{decomp_type}.pt",
                f"{layer_name}_{decomp_type}.pth",
                f"layer{layer_num}_{decomp_type}.pt",
                f"layer{layer_num}_{decomp_type}.pth",
                f"{decomp_type}_layer{layer_num}.pt",
                f"{decomp_type}_layer{layer_num}.pth",
                f"{decomp_type}_{layer_name}.pt",
                f"{decomp_type}_{layer_name}.pth"
            ]
            
            # Check if any of these patterns exist
            for pattern in patterns:
                model_path = os.path.join(model_base_path, pattern)
                if os.path.exists(model_path):
                    return model_path
                    
            # If none found, return the first pattern anyway (it will fail gracefully later)
            return os.path.join(model_base_path, patterns[0])
        else:
            # Use the provided pattern, replacing placeholders
            filename = model_pattern.format(
                layer_num=layer_num,
                layer_name=layer_name,
                decomp_type=decomp_type
            )
            
            # Add file extension if not provided
            if not filename.endswith('.pt') and not filename.endswith('.pth'):
                filename += '.pt'
                
            model_path = os.path.join(model_base_path, filename)
            return model_path
    
    def apply_sparse_decomposition(self, activations, decomposition_type="sae", 
                                  feature_dim=None, sae_model_path=None, st_model_path=None,
                                  model_pattern=None, layer_name="layer_0", **kwargs):
        """
        Apply sparse decomposition to layer activations
        
        Args:
            activations: Activation matrix (n_samples, hidden_dim)
            decomposition_type: Type of decomposition ('sae' or 'st')
            feature_dim: Feature dimension, if None uses hidden_dim / 4
            sae_model_path: Base path to pre-trained SAE models
            st_model_path: Base path to pre-trained ST models
            model_pattern: Pattern for model filenames
            layer_name: Current layer name for model lookup
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
        
        # Check if we need to load a pre-trained model
        model_loaded_from_path = False  # Flag to track if we successfully loaded a model
        
        # Get specific model path for this layer and decomposition type
        if decomposition_type.lower() == "sae":
            model_path = self.get_model_path_for_layer(
                layer_name, 
                "sae", 
                sae_model_path, 
                model_pattern
            )
        else:  # ST
            model_path = self.get_model_path_for_layer(
                layer_name, 
                "st", 
                st_model_path, 
                model_pattern
            )
        
        # Create decomposition model
        if decomposition_type.lower() == "sae":
            # Create SAE model - use original if available, otherwise use simplified
            if DECOMP_AVAILABLE:
                print(f"Creating SAE model with dims: {hidden_dim} -> {feature_dim}")
                model = SparseAutoencoder(
                    n=hidden_dim,
                    m=feature_dim,
                    sae_model_path=model_path,  # Pass model path to SAE
                    device=self.device,
                    **kwargs
                )
                
                # If model path exists, try to load the model
                if model_path and os.path.exists(model_path):
                    print(f"Loading pre-trained SAE model from: {model_path}")
                    success = model.resume_from_checkpoint(model_path)
                    model_loaded_from_path = success
                    if not success:
                        print(f"Failed to load SAE model from {model_path}, will train from scratch")
            else:
                print("Using simplified SAE implementation")
                model = SimplifiedSAE(
                    n=hidden_dim,
                    m=feature_dim,
                    sae_model_path=model_path,
                    device=self.device,
                    lambda_l1=kwargs.get('lambda_l1', 1.0)
                )
                
                # If model path exists, try to load the model
                if model_path and os.path.exists(model_path):
                    print(f"Loading pre-trained simplified SAE model from: {model_path}")
                    success = model.resume_from_checkpoint(model_path)
                    model_loaded_from_path = success
                    if not success:
                        print(f"Failed to load simplified SAE model from {model_path}, will train from scratch")
                        
        elif decomposition_type.lower() == "st":
            # Calculate attention dimension to be balanced with SAE param count
            a = max(20, hidden_dim // 8)
            print(f"Using attention dimension: {a}")
            
            # Create ST model - use original if available, otherwise use simplified
            if DECOMP_AVAILABLE:
                print(f"Creating ST model with dims: {hidden_dim} -> {feature_dim}, attention dim: {a}")
                model = SparseTransformer(
                    X=activation_tensor,
                    n=hidden_dim,
                    m=feature_dim,
                    a=a,
                    st_model_path=model_path,  # Pass model path to ST
                    device=self.device,
                    **kwargs
                )
                
                # If model path exists, try to load the model
                if model_path and os.path.exists(model_path):
                    print(f"Loading pre-trained ST model from: {model_path}")
                    # ST models can be loaded through train_and_validate with resume_from
                    model_loaded_from_path = True
            else:
                print("Using simplified ST implementation")
                model = SimplifiedST(
                    X=activation_tensor,
                    n=hidden_dim,
                    m=feature_dim,
                    a=a,
                    st_model_path=model_path,
                    device=self.device,
                    lambda_l1=kwargs.get('lambda_l1', 1.0)
                )
                
                # If model path exists, try to load the model
                if model_path and os.path.exists(model_path):
                    print(f"Loading pre-trained simplified ST model from: {model_path}")
                    success = model.resume_from_checkpoint(model_path)
                    model_loaded_from_path = success
                    if not success:
                        print(f"Failed to load simplified ST model from {model_path}, will train from scratch")
        else:
            raise ValueError(f"Unknown decomposition type: {decomposition_type}")
        
        # Train model if not loaded from path
        if not model_loaded_from_path:
            print(f"Training {decomposition_type.upper()} model...")
            # Split for training and validation
            split_idx = max(1, int(n_samples * 0.8))
            train_tensor = activation_tensor[:split_idx]
            val_tensor = activation_tensor[split_idx:]
            
            # Train model
            if decomposition_type.lower() == "st" and DECOMP_AVAILABLE and model_path and os.path.exists(model_path):
                # For ST, use resume_from parameter
                model.train_and_validate(
                    train_tensor,
                    val_tensor,
                    batch_size=min(64, n_samples),
                    learning_rate=1e-4,
                    target_steps=5000,  # Reduced for demonstration
                    resume_from=model_path
                )
            else:
                # Regular training
                model.train_and_validate(
                    train_tensor,
                    val_tensor,
                    batch_size=min(64, n_samples),
                    learning_rate=1e-4,
                    target_steps=5000,  # Reduced for demonstration
                )
        else:
            print(f"Using pre-trained {decomposition_type.upper()} model, skipping training")
        
        # Get feature activations
        with torch.no_grad():
            feature_activations = model.feature_activations(activation_tensor)
            if isinstance(feature_activations, torch.Tensor):
                feature_activations = feature_activations.cpu().numpy()
        
        print(f"Feature activations shape: {feature_activations.shape}")
        return model, feature_activations
    
    # Rest of the ModelAnalyzer class methods...
    # (I've removed them for brevity but they would be identical to the original code)
    # ...

def parse_args():
    """Parse command-line arguments for GPT Neo layer analysis"""
    parser = argparse.ArgumentParser(description='Enhanced Transformer Layer Analyzer with reason.ipynb capabilities and pre-trained model support')
    
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
    
    # NEW: Pre-trained model paths
    decomp_group.add_argument('--sae_model_path', type=str, default=None,
                      help='Base path to pre-trained SAE models')
    decomp_group.add_argument('--st_model_path', type=str, default=None,
                      help='Base path to pre-trained ST models')
    decomp_group.add_argument('--model_pattern', type=str, default=None,
                      help='Pattern for model filenames, can include {layer_num}, {layer_name}, and {decomp_type} placeholders')
    decomp_group.add_argument('--train_if_not_found', action='store_true',
                      help='Train a new model if pre-trained model not found (default: True)')
    
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
    Implements the exact same approach as reason.ipynb, with added support
    for loading pre-trained SAE/ST models.
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
    
    # Display pre-trained model paths if provided
    if args.sae_model_path:
        print(f"SAE model path: {args.sae_model_path}")
    if args.st_model_path:
        print(f"ST model path: {args.st_model_path}")
    if args.model_pattern:
        print(f"Model filename pattern: {args.model_pattern}")
    
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
        except Exception as e:
            print(f"Error during visualization: {e}")
            print("Continuing with analysis...")
    
    # Apply sparse decomposition if requested
    decomposed_features = {}
    if args.decomposition != 'none':
        print("\n" + "="*50)
        print(f"Applying {args.decomposition.upper()} decomposition at token level")
        if args.sae_model_path:
            print(f"Using pre-trained SAE models from: {args.sae_model_path}")
        if args.st_model_path:
            print(f"Using pre-trained ST models from: {args.st_model_path}")
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
                        lambda_l1=args.l1_lambda,
                        # Pass the pre-trained model paths and pattern
                        sae_model_path=args.sae_model_path,
                        model_pattern=args.model_pattern,
                        layer_name=layer_name
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
                
                if args.decomposition in ['st', 'both']:
                    print("\n  Applying ST decomposition...")
                    st_model, st_features = analyzer.apply_sparse_decomposition(
                        activations,
                        decomposition_type="st",
                        feature_dim=feature_dim,
                        lambda_l1=args.l1_lambda,
                        # Pass the pre-trained model paths and pattern
                        st_model_path=args.st_model_path,
                        model_pattern=args.model_pattern,
                        layer_name=layer_name
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
                # Analyze cluster distances if requested
                if args.analyze_clusters and args.visualize:
                    print("\nAnalyzing cluster distances between token groups...")
                    cluster_dir = os.path.join(args.output_dir, "clusters")
                    os.makedirs(cluster_dir, exist_ok=True)
                    
                    # Process each layer
                    for layer_name, activations in hidden_states.items():
                        try:
                            # Extract layer number for output filename
                            layer_nr = int(layer_name.split('_')[1])
                            
                            print(f"  Computing cluster centroids for {layer_name}...")
                            # Compute centroids for each group in the raw activations
                            centroids = TokenClusteringUtils.compute_cluster_centroids(
                                activations, token_to_text_map)
                            
                            # Compute distance matrix between centroids
                            groups, dist_matrix, avg_distance = TokenClusteringUtils.compute_distance_matrix(
                                centroids)
                            
                            print(f"    Found {len(groups)} clusters with avg distance: {avg_distance:.4f}")
                            
                            # Visualize the distance matrix
                            TokenClusteringUtils.visualize_distance_matrix(
                                groups, dist_matrix, avg_distance, cluster_dir, layer_nr)
                            
                        except Exception as e:
                            print(f"Error during cluster analysis of {layer_name}: {e}")
                            print("  Skipping this layer and continuing with others...")
                            continue
            except Exception as e:
                print(f"Error during decomposition of {layer_name}: {e}")
                print("  Skipping this layer and continuing with others...")
                continue
    
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)
    print(f"Processed {len(texts)} texts through {len(hidden_states)} layers")
    if args.decomposition != 'none':
        print(f"Applied {args.decomposition} decomposition to all layers")
    # Create GIFs if requested
    if args.create_gif and args.visualize:
        print("\nCreating GIFs from visualizations...")
        
        # Create GIF for raw activations
        viz_dir = os.path.join(args.output_dir, "visualizations")
        if os.path.exists(viz_dir):
            gif_path = os.path.join(args.output_dir, "layers_raw_progression.gif")
            print(f"Creating GIF of raw activation layers: {gif_path}")
            TokenClusteringUtils.create_gif_from_images(viz_dir, gif_path, pattern="layer*_umap.png")
        
        # Create GIF for decompositions
        if args.decomposition != 'none':
            decomp_dir = os.path.join(args.output_dir, "decomposition")
            if os.path.exists(decomp_dir):
                if args.decomposition in ['sae', 'both']:
                    gif_path = os.path.join(args.output_dir, "layers_sae_progression.gif")
                    print(f"Creating GIF of SAE feature layers: {gif_path}")
                    TokenClusteringUtils.create_gif_from_images(decomp_dir, gif_path, pattern="layer*_sae_umap.png")
                
                if args.decomposition in ['st', 'both']:
                    gif_path = os.path.join(args.output_dir, "layers_st_progression.gif")
                    print(f"Creating GIF of ST feature layers: {gif_path}")
                    TokenClusteringUtils.create_gif_from_images(decomp_dir, gif_path, pattern="layer*_st_umap.png")
    print("\nResults saved to:")
    print(f"  - {args.output_dir}/")
    
    print("\nNext steps:")
    print("- Try different pre-trained models with --sae_model_path or --st_model_path")
    print("- Customize the model filename pattern with --model_pattern")
    print("- Analyze more/different text inputs")
    print("- Explore other layers (use --layers option)")
    print("- Try --duplicate_text to analyze how identical texts cluster")

if __name__ == "__main__":
    main()