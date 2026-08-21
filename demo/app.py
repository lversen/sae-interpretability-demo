"""Interactive dense-vs-sparse feature explorer.

Type a sentence and see two things side by side: the raw GPT-2 activation
(dense, every dimension has some value, none of them individually mean
anything) versus the SAE's sparse features (a handful fire, and each one
can be named by the other corpus sentences that fire it too) - a concrete,
clickable version of what the thesis measures in prose.

Run with: streamlit run demo/app.py
The dense side works immediately. The sparse side needs
demo/artifacts/sae.pt from demo/train.py.
"""
import os

import numpy as np
import streamlit as st
import torch
from huggingface_hub import hf_hub_download
from transformers import GPT2Model, GPT2Tokenizer

from corpus import SENTENCES
from sae import SparseAutoencoder

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
MODEL_NAME = "gpt2"
DEFAULT_LAYER = 6  # matches train.py; overridden by the checkpoint's own layer once trained
HF_REPO_ID = "Sebastian-T-Iversen/sae-interpretability-demo"  # must match train.py's HF_REPO_ID


def resolve_artifact(filename):
    """Local copy first (e.g. right after training), else pull from the HF Hub registry."""
    local_path = os.path.join(ARTIFACT_DIR, filename)
    if os.path.exists(local_path):
        return local_path
    try:
        return hf_hub_download(repo_id=HF_REPO_ID, filename=filename)
    except Exception:
        return None


@st.cache_resource
def load_gpt2():
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    gpt2 = GPT2Model.from_pretrained(MODEL_NAME)
    gpt2.eval()
    return tokenizer, gpt2


@st.cache_resource
def load_sae(sae_path, features_path):
    checkpoint = torch.load(sae_path, map_location="cpu", weights_only=True)
    sae = SparseAutoencoder(checkpoint["n_in"], checkpoint["n_features"])
    sae.load_state_dict(checkpoint["state_dict"])
    sae.eval()
    corpus_features = np.load(features_path)
    return sae, corpus_features, checkpoint["layer"]


def get_activation(sentence, tokenizer, gpt2, layer):
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        out = gpt2(**inputs, output_hidden_states=True)
    hidden = out.hidden_states[layer][0]
    return hidden.mean(dim=0)


def to_heatmap(values, rows=24, cols=32, cell_size=18, clip_percentile=99):
    """Grid of colored squares, one per dimension, diverging blue/white/red.

    A rows x cols grid (768 = 24 x 32, exact) reads better than a thin 1D strip -
    each dimension gets an actual visible square. Color is clipped at a
    percentile of |values| rather than the true max, so a couple of huge
    outlier values (see "rogue dimensions" below) saturate to solid color
    instead of washing out every other square's contrast.
    """
    vmax = max(np.percentile(np.abs(values), clip_percentile), 1e-6)
    n = np.clip(values / vmax, -1, 1).reshape(rows, cols)
    rgb = np.full((rows, cols, 3), 255, dtype=np.uint8)
    pos = n > 0
    rgb[pos, 1] = ((1 - n[pos]) * 255).astype(np.uint8)
    rgb[pos, 2] = ((1 - n[pos]) * 255).astype(np.uint8)
    neg = ~pos
    rgb[neg, 0] = ((1 + n[neg]) * 255).astype(np.uint8)
    rgb[neg, 1] = ((1 + n[neg]) * 255).astype(np.uint8)
    return np.repeat(np.repeat(rgb, cell_size, axis=0), cell_size, axis=1)


st.set_page_config(page_title="Sparse Feature Explorer", layout="centered")
st.title("Dense vs. Sparse: Why Interpretability Needs Sparsity")
st.caption(
    "Companion demo to my master's thesis on sparse neural network interpretability. "
    "Type a sentence and compare GPT-2's raw activation to the sparse features "
    "a small autoencoder extracts from it."
)

tokenizer, gpt2 = load_gpt2()
sae_path = resolve_artifact("sae.pt")
features_path = resolve_artifact("corpus_features.npy")
sae_available = sae_path is not None and features_path is not None

text = st.text_input("Enter a sentence", value="The team celebrated after winning the game.")

if text:
    dense_col, sparse_col = st.columns(2)

    with dense_col:
        st.subheader("Dense (raw GPT-2 activation)")
        x = get_activation(text, tokenizer, gpt2, DEFAULT_LAYER)
        x = x.numpy()
        st.caption(
            "Every one of 768 dimensions has some value. None of them, alone, means anything - "
            "just noise, dimension to dimension. (A couple of 'rogue dimensions' run far larger "
            "than the rest on every input regardless of content, a known transformer artifact - "
            "color here is capped at the 99th percentile so they don't wash out everything else.)"
        )
        st.image(to_heatmap(x), width="stretch")
        st.metric("Nonzero dimensions", f"{int((np.abs(x) > 1e-6).sum())} / {len(x)}")

    with sparse_col:
        st.subheader("Sparse (SAE features)")
        if not sae_available:
            st.info("Train the SAE first: `python demo/train.py`")
        else:
            sae, corpus_features, layer = load_sae(sae_path, features_path)
            x_sae = get_activation(text, tokenizer, gpt2, layer)
            with torch.no_grad():
                _, f = sae(x_sae.unsqueeze(0))
            f = f.squeeze(0).numpy()

            top_idx = np.argsort(-f)[:10]
            top_vals = f[top_idx]

            st.caption("Only a handful of features turn on. Each one you can point to and name.")
            st.bar_chart({"activation": top_vals}, x_label=None, height=250)
            st.metric("Active features", f"{int((f > 0).sum())} / {len(f)}")

    if sae_available:
        best_feature = int(top_idx[0])
        st.subheader(f"Other sentences that activate feature {best_feature} most strongly")
        st.caption(
            "If this feature is monosemantic, these should share a common theme with "
            "your input sentence - that's the interpretability payoff sparsity buys you."
        )
        feature_col = corpus_features[:, best_feature]
        top_corpus_idx = np.argsort(-feature_col)[:5]
        for idx in top_corpus_idx:
            st.write(f"- {SENTENCES[idx]}  (activation: {feature_col[idx]:.3f})")
