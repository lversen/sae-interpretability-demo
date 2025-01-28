import torch
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
import os

print("Downloading and processing MNIST dataset...")

# Create data directory if it doesn't exist
os.makedirs('./data', exist_ok=True)

# Download MNIST dataset
mnist_train = datasets.MNIST(root='./data', train=True, download=True)
mnist_test = datasets.MNIST(root='./data', train=False, download=True)

# Process training data
print("Processing training data...")
X_train = pd.DataFrame((mnist_train.data.float() / 255.0).view(-1, 28*28).numpy())
y_train = pd.DataFrame(mnist_train.targets.numpy(), columns=['label'])

# Process test data
print("Processing test data...")
X_test = pd.DataFrame((mnist_test.data.float() / 255.0).view(-1, 28*28).numpy())
y_test = pd.DataFrame(mnist_test.targets.numpy(), columns=['label'])

# Combine features and labels
train_df = pd.concat([y_train, X_train], axis=1)
test_df = pd.concat([y_test, X_test], axis=1)

# Save to CSV
print("Saving training data...")
train_df.to_csv('./data/mnist_train.csv', index=False)
print(f"Training data saved: {train_df.shape[0]} samples, {train_df.shape[1]} columns")

print("Saving test data...")
test_df.to_csv('./data/mnist_test.csv', index=False)
print(f"Test data saved: {test_df.shape[0]} samples, {test_df.shape[1]} columns")