import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

# Load the tokenizer from the same model you trained on
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

# Read the 5 paragraphs
with open('text.txt', 'r', encoding="utf-8") as f:
    paragraphs = [line.strip() for line in f.readlines()]

# Create features by tokenizing and getting embeddings
all_tokens = []
all_labels = []

# Process each paragraph
for para_idx, paragraph in enumerate(paragraphs):
    # Tokenize the paragraph (skip special tokens to get individual normal tokens)
    tokens = tokenizer.encode(paragraph, add_special_tokens=False)
    
    # Add each token with its paragraph label
    for token in tokens:
        all_tokens.append(token)
        all_labels.append(para_idx)  # Use paragraph number as the label

# Get embeddings for these tokens
# Note: This creates a simple feature vector for each token using the embedding layer
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
embeddings = model.transformer.wte.weight.detach().cpu().numpy()

# Create features from token embeddings
features = np.array([embeddings[token] for token in all_tokens])

# Create a DataFrame with the label column
df = pd.DataFrame({'label': all_labels})

# Add the feature columns
for i in range(features.shape[1]):
    df[f'feature_{i}'] = features[:, i]

# Save to CSV
df.to_csv('paragraph_features.csv', index=False)