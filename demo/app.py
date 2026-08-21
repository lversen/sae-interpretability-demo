"""Interactive sparse-feature explorer.

Type a sentence, see which sparse features fire, and see which other
sentences in the corpus fire the same top feature most strongly -
a concrete, clickable version of what the thesis measures in prose.

Run with: streamlit run demo/app.py
(requires demo/artifacts/sae.pt from demo/train.py first)
"""
import os

import numpy as np
import streamlit as st
import torch
from transformers import GPT2Model, GPT2Tokenizer

from corpus import SENTENCES
from sae import SparseAutoencoder

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")


@st.cache_resource
def load_everything():
    checkpoint = torch.load(os.path.join(ARTIFACT_DIR, "sae.pt"), map_location="cpu", weights_only=True)
    sae = SparseAutoencoder(checkpoint["n_in"], checkpoint["n_features"])
    sae.load_state_dict(checkpoint["state_dict"])
    sae.eval()

    tokenizer = GPT2Tokenizer.from_pretrained(checkpoint["model_name"])
    gpt2 = GPT2Model.from_pretrained(checkpoint["model_name"])
    gpt2.eval()

    corpus_features = np.load(os.path.join(ARTIFACT_DIR, "corpus_features.npy"))

    return sae, tokenizer, gpt2, corpus_features, checkpoint["layer"]


def get_activation(sentence, tokenizer, gpt2, layer):
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        out = gpt2(**inputs, output_hidden_states=True)
    hidden = out.hidden_states[layer][0]
    return hidden.mean(dim=0)


st.set_page_config(page_title="Sparse Feature Explorer", layout="centered")
st.title("Sparse Autoencoder Feature Explorer")
st.caption(
    "Companion demo to my master's thesis on sparse neural network interpretability. "
    "Type a sentence, and see which sparse features in a small GPT-2 sparse autoencoder "
    "fire in response."
)

if not os.path.exists(os.path.join(ARTIFACT_DIR, "sae.pt")):
    st.error("No trained model found. Run `python demo/train.py` first.")
    st.stop()

sae, tokenizer, gpt2, corpus_features, layer = load_everything()

text = st.text_input("Enter a sentence", value="The team celebrated after winning the game.")

if text:
    x = get_activation(text, tokenizer, gpt2, layer)
    with torch.no_grad():
        _, f = sae(x.unsqueeze(0))
    f = f.squeeze(0).numpy()

    top_idx = np.argsort(-f)[:10]
    top_vals = f[top_idx]

    st.subheader("Top 10 activated features")
    st.bar_chart({"activation": top_vals}, x_label=None)
    st.write({f"feature {i}": round(float(v), 3) for i, v in zip(top_idx, top_vals)})

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
