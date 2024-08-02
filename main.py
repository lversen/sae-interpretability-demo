import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from feature_extraction_with_store import feature_extraction_with_store
from gephi import node_attributes, gephi_export
from language_classification import language_classifier
from sample_handler import get_consistent_samples

np.set_printoptions(suppress=True)

class SparseAutoencoder(nn.Module):
    def __init__(self, D, F, lambda_l1=0.1):
        super(SparseAutoencoder, self).__init__()
        self.D = D  # Input dimension (residual stream dimension)
        self.F = F  # Number of features
        self.lambda_l1 = lambda_l1

        # Encoder: W^enc ∈ ℝ^(F×D)
        self.encoder = nn.Linear(D, F)
        # Decoder: W^dec ∈ ℝ^(D×F)
        self.decoder = nn.Linear(F, D)

        # Initialize weights
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)

        # Learned biases
        self.b_enc = nn.Parameter(torch.zeros(F))
        self.b_dec = nn.Parameter(torch.zeros(D))

    def forward(self, x):
        # Encoder: calculate feature activations (f_i)
        f_i = torch.relu(self.encoder(x) + self.b_enc)
        # Decoder
        x_hat = self.decoder(f_i) + self.b_dec
        return x_hat, f_i

    def loss_function(self, x, x_hat, f_i):
        mse_loss = torch.mean(torch.sum((x - x_hat)**2, dim=1))
        l1_loss = self.lambda_l1 * torch.sum(torch.abs(f_i) * torch.norm(self.decoder.weight, dim=0))
        return mse_loss + l1_loss

    def train_model(self, feature_extract, batch_size=32, num_epochs=100, learning_rate=5e-5, patience=10, min_delta=1e-4):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # Convert to PyTorch tensor if necessary
        if isinstance(feature_extract, np.ndarray):
            feature_extract = torch.from_numpy(feature_extract).float()
        
        # Normalize input
        feature_extract = feature_extract / torch.sqrt(torch.mean(feature_extract**2, dim=1, keepdim=True))
        
        # Create DataLoader for batching
        dataset = TensorDataset(feature_extract)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Early stopping setup
        best_loss = float('inf')
        no_improve_epochs = 0
        loss_history = deque(maxlen=patience)
        
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in dataloader:
                x = batch[0]
                
                x_hat, f_i = self(x)
                loss = self.loss_function(x, x_hat, f_i)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            loss_history.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
            
            # Check for improvement
            if avg_loss < best_loss - min_delta:
                best_loss = avg_loss
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
            
            # Early stopping check
            if no_improve_epochs >= patience:
                print(f"Early stopping triggered. No improvement for {patience} epochs.")
                break
        
        # Print final loss and number of epochs
        print(f"Training completed. Final loss: {avg_loss:.4f}, Epochs: {epoch+1}")

    def get_feature_activations(self, x, batch_size=32):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        x = x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True))
        
        dataset = TensorDataset(x)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        feature_activations = []
        
        with torch.no_grad():
            for batch in dataloader:
                _, f_i = self(batch[0])
                feature_activations.append(f_i)
        
        return torch.cat(feature_activations, dim=0)

    @property
    def W_dec(self):
        return self.decoder.weight.data.T

def split_categories(category_string: str, delimiter: str = ',') -> List[str]:
    return [cat.strip() for cat in category_string.split(delimiter) if cat.strip()]

def select_and_assign_exact_n_categories(df: pd.DataFrame, category: str, n: int, delimiter: str = ',') -> Tuple[pd.DataFrame, List[str]]:
    if category not in df.columns:
        raise ValueError(f"{category} is not a column in the dataset")
    
    all_categories = [cat.strip() for cats in df[category].apply(lambda x: split_categories(str(x), delimiter)) for cat in cats]
    category_counts = Counter(all_categories)
    selected_categories = [cat for cat, _ in category_counts.most_common(n)]
    
    print(f"Selected top {n} categories: {selected_categories}")
    
    def assign_category(cat_string):
        cats = split_categories(str(cat_string), delimiter)
        matching_cats = [cat for cat in cats if cat in selected_categories]
        return random.choice(matching_cats) if matching_cats else None
    
    df['assigned_category'] = df[category].apply(assign_category)
    filtered_df = df.dropna(subset=['assigned_category'])
    
    assigned_categories = filtered_df['assigned_category'].value_counts(normalize=True) * 100
    print("Distribution of assigned categories:")
    print(assigned_categories)
    
    return filtered_df, selected_categories

def run_all(
    datasets: List[str],
    models: List[str],
    n: int,
    feature_column: List[str],
    label_column: List[str],
    create_graph: bool = False,
    force_new_embeddings: bool = False,
    classify_language: List[str] = [],
    top_n_category: Optional[Dict[str, Dict[str, Any]]] = None,
    batch_size: int = 32,
    patience: int = 10,
    min_delta: float = 1e-4
):
    
    for i, dataset in enumerate(datasets):
        print(f"Processing dataset: {dataset}")        
        full_df = pd.read_csv(dataset)
        print(f"Full dataset shape: {full_df.shape}")

        category_column = None
        if top_n_category:
            category_info = top_n_category.get(dataset, {})
            if category_info:
                category_column = category_info['column']
                n_top = category_info['n']
                delimiter = category_info.get('delimiter', ',')
                full_df, selected_categories = select_and_assign_exact_n_categories(full_df, category_column, n_top, delimiter)
                print(f"Filtered dataset shape after selecting and assigning exactly {n_top} categories: {full_df.shape}")

        original_n = n
        n = min(n, len(full_df))
        if n < original_n:
            print(f"Warning: Requested sample size ({original_n}) is larger than the available data ({n}). Using {n} samples.")

        for model in models:
            print(f"Processing model: {model}")
            
            df, indices = get_consistent_samples(full_df, n, dataset, model)
            print(f"Sample shape: {df.shape}")
            
            feature_extract = feature_extraction_with_store(
                df, full_df, model, n, dataset, feature_column[i], 
                force_new_embeddings=force_new_embeddings
            )
            
            # Implement SAE with batching and early stopping
            D = feature_extract.shape[1]  # Input dimension (residual stream dimension)
            F = 16  # Number of features in the SAE (you can adjust this)
            sae = SparseAutoencoder(D, F)
            sae.train_model(feature_extract, batch_size=batch_size, patience=patience, min_delta=min_delta, num_epochs=100_000)
            
            # Get feature activations using batching
            feature_activations = sae.get_feature_activations(feature_extract, batch_size=batch_size)
            print("Feature activations shape:", feature_activations)
            
            # Get decoder weights (feature vectors)
            feature_vectors = sae.W_dec
            print("Feature vectors shape:", feature_vectors.shape)
            
            if len(classify_language) != 0:
                indices = np.array(indices, dtype=np.int32)
                language_classifier(df, indices, classify_language, dataset)
            
            if create_graph:
                mapping, attributes = node_attributes(df, label_column[i], model, 'assigned_category')
                print(f"Exporting Gephi graph for {dataset} with model {model}")
                gephi_export(feature_extract, dataset, model, mapping, attributes)
    
    print("Processing complete for all datasets and models.")

if __name__ == "__main__":
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
    
    datasets = ["data/final_data.csv"]
    feature_column = ["Description"]
    label_column = ["Name"]
    models = ['whaleloops/phrase-bert']
    n = 20_000
    top_n_category = {"data/final_data.csv": {"column": "Genres", "n": 10, "delimiter": ","}}

    run_all(
        datasets=datasets,
        models=models,
        n=n,
        feature_column=feature_column,
        label_column=label_column,
        batch_size=n,  # You can adjust this value
        patience=1000,    # Number of epochs with no improvement after which training will be stopped
        min_delta=1e-5  # Minimum change in loss to qualify as an improvement
    )