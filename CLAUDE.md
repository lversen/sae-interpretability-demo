# Repo notes for Claude Code

## demo/

Written by an AI assistant (2026-08-17) that had no Python/PyTorch available
to actually run it. **The code has never been executed end to end.** Treat
it as a first draft, not verified working code.

Likely failure points, in rough order of probability:
- `transformers` API drift: `GPT2Model`/`GPT2Tokenizer` import paths and the
  `output_hidden_states` return shape have changed across versions before.
  If `demo/train.py` fails on the GPT-2 load or the `hidden_states[LAYER]`
  indexing, check the installed `transformers` version's docs first.
- Tensor shape mismatches between `demo/train.py` (produces `corpus_features.npy`)
  and `demo/app.py` (loads it) — they must agree on `N_FEATURES` and `LAYER`.
- `demo/requirements.txt` has no pinned versions — if something breaks after
  a `pip install`, suspect a version bump before suspecting the logic.

**How to debug:** run `python demo/train.py` standalone first, before touching
the Streamlit app — it prints loss/sparsity every 200 steps, so you can see
immediately if activations, shapes, or the training loop itself are broken.
Only move to `streamlit run demo/app.py` once training completes and
`demo/artifacts/` has both files.

The rest of the repo (`src/`, `scripts/`, `thesis.pdf`) is the original,
working master's thesis codebase — treat that as trusted, unlike `demo/`.
