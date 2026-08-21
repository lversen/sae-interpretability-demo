"""Train a small sparse autoencoder on GPT-2 residual-stream activations.

Usage:
    python demo/train.py

Produces demo/artifacts/sae.pt and demo/artifacts/corpus_features.npy,
both consumed by demo/app.py.
"""
import os

import numpy as np
import torch
from huggingface_hub import HfApi
from transformers import GPT2Model, GPT2Tokenizer

from corpus import SENTENCES
from sae import SparseAutoencoder

HF_REPO_ID = "Sebastian-T-Iversen/sae-interpretability-demo"  # must match app.py's HF_REPO_ID
MODEL_NAME = "gpt2"
LAYER = 6           # middle layer of gpt2-small's 12 transformer blocks
N_IN = 768           # gpt2-small hidden size
N_FEATURES = 3072    # 4x overcomplete
STEPS = 2000
LR = 1e-3
L1_COEF = 2e-3

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")


def get_activations(sentences, model, tokenizer, device):
    """Mean-pooled hidden state at LAYER for each sentence."""
    vecs = []
    model.eval()
    with torch.no_grad():
        for sent in sentences:
            inputs = tokenizer(sent, return_tensors="pt").to(device)
            out = model(**inputs, output_hidden_states=True)
            hidden = out.hidden_states[LAYER][0]  # (seq_len, n_in)
            vecs.append(hidden.mean(dim=0).cpu())
    return torch.stack(vecs)


def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    gpt2 = GPT2Model.from_pretrained(MODEL_NAME).to(device)

    print(f"Extracting activations for {len(SENTENCES)} sentences...")
    X = get_activations(SENTENCES, gpt2, tokenizer, device).to(device)

    print(f"Training SAE: n_in={N_IN}, n_features={N_FEATURES}, steps={STEPS}")
    sae = SparseAutoencoder(N_IN, N_FEATURES).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=LR)

    sae.train()
    for step in range(STEPS):
        optimizer.zero_grad()
        x_hat, f = sae(X)
        loss, recon, sparsity = sae.loss(X, x_hat, f, l1_coef=L1_COEF)
        loss.backward()
        optimizer.step()

        if step % 200 == 0 or step == STEPS - 1:
            active = (f > 0).float().mean().item()
            print(f"step {step:4d} | loss {loss.item():.4f} | recon {recon.item():.4f} "
                  f"| sparsity {sparsity.item():.4f} | active-frac {active:.3f}")

    sae.eval()
    with torch.no_grad():
        _, corpus_features = sae(X)

    torch.save({
        "state_dict": sae.state_dict(),
        "n_in": N_IN,
        "n_features": N_FEATURES,
        "layer": LAYER,
        "model_name": MODEL_NAME,
    }, os.path.join(ARTIFACT_DIR, "sae.pt"))

    np.save(os.path.join(ARTIFACT_DIR, "corpus_features.npy"), corpus_features.cpu().numpy())

    print(f"\nSaved artifacts to {ARTIFACT_DIR}")

    print(f"Pushing artifacts to https://huggingface.co/{HF_REPO_ID} ...")
    api = HfApi()
    api.create_repo(repo_id=HF_REPO_ID, repo_type="model", exist_ok=True)
    for filename in ("sae.pt", "corpus_features.npy"):
        api.upload_file(
            path_or_fileobj=os.path.join(ARTIFACT_DIR, filename),
            path_in_repo=filename,
            repo_id=HF_REPO_ID,
        )
    print("Pushed. Now run: streamlit run demo/app.py")


if __name__ == "__main__":
    main()
