"""
Word2Vec — Skip-gram with Negative Sampling (pure NumPy)
=========================================================

Mathematical background
-----------------------
Skip-gram: given a center word c, predict each surrounding context word o.

Negative Sampling loss for one (center c, context o) pair with K negatives:

    J = -log σ(u_o · v_c) - Σ_{k=1}^{K} log σ(-u_k · v_c)

where
  v_c  = W_in[c]   — center-word embedding  (shape: d)
  u_o  = W_out[o]  — context-word embedding (shape: d)
  u_k  = W_out[k]  — negative sample k      (shape: d)
  σ(x) = 1 / (1 + exp(-x))

Gradients (derived by chain rule):

  ∂J/∂v_c  = (σ(u_o·v_c) − 1)·u_o  +  Σ_k σ(u_k·v_c)·u_k
  ∂J/∂u_o  = (σ(u_o·v_c) − 1)·v_c
  ∂J/∂u_k  =  σ(u_k·v_c)·v_c         for each negative k

Noise distribution for negative sampling:
    P_n(w) ∝ freq(w)^(3/4)   (from the original word2vec paper)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid: avoids overflow for large negative x."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Word2Vec:
    """
    Skip-gram Word2Vec model trained with Negative Sampling.

    Two weight matrices are maintained:
      W_in  (vocab_size × embed_dim) — embeddings looked up for the center word.
      W_out (vocab_size × embed_dim) — embeddings looked up for context / negatives.

    After training, W_in (or the average of W_in and W_out) is used as word vectors.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        learning_rate: float = 0.025,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        # Xavier-style init for W_in; zeros for W_out (common choice)
        self.W_in  = (rng.random((vocab_size, embed_dim)) - 0.5) / embed_dim
        self.W_out = np.zeros((vocab_size, embed_dim))
        self.lr = learning_rate

    # ------------------------------------------------------------------
    # Single training step
    # ------------------------------------------------------------------

    def train_step(
        self,
        center_id: int,
        context_id: int,
        neg_ids: np.ndarray,
    ) -> float:
        """
        One forward + backward pass for a (center, context, negatives) triple.

        Parameters
        ----------
        center_id  : index of the center word
        context_id : index of the positive context word
        neg_ids    : array of shape (K,) with negative sample indices

        Returns
        -------
        loss : scalar float (for monitoring)
        """
        v_c   = self.W_in[center_id]       # shape (d,)
        u_o   = self.W_out[context_id]     # shape (d,)
        u_neg = self.W_out[neg_ids]        # shape (K, d)

        # ---- Forward pass ------------------------------------------------
        pos_score  = np.dot(u_o, v_c)      # scalar:  u_o · v_c
        neg_scores = u_neg @ v_c           # shape (K,): u_k · v_c for each k

        pos_sig = sigmoid(pos_score)       # σ(u_o · v_c)
        neg_sig = sigmoid(neg_scores)      # σ(u_k · v_c)  shape (K,)

        # ---- Loss (for logging only, not used in gradient) ---------------
        loss = -np.log(pos_sig + 1e-10) - np.sum(np.log(sigmoid(-neg_scores) + 1e-10))

        # ---- Gradients ---------------------------------------------------
        # ∂J/∂v_c = (σ(u_o·v_c) − 1)·u_o + Σ_k σ(u_k·v_c)·u_k
        grad_v_c = (pos_sig - 1.0) * u_o + neg_sig @ u_neg  # shape (d,)

        # ∂J/∂u_o = (σ(u_o·v_c) − 1)·v_c
        grad_u_o = (pos_sig - 1.0) * v_c  # shape (d,)

        # ∂J/∂u_k = σ(u_k·v_c)·v_c   for each k
        grad_u_neg = np.outer(neg_sig, v_c)  # shape (K, d)

        # ---- Parameter updates (SGD) ------------------------------------
        self.W_in[center_id]  -= self.lr * grad_v_c
        self.W_out[context_id] -= self.lr * grad_u_o
        # np.add.at handles repeated indices in neg_ids correctly
        np.add.at(self.W_out, neg_ids, -self.lr * grad_u_neg)

        return float(loss)

    # ------------------------------------------------------------------
    # Cosine similarity helpers
    # ------------------------------------------------------------------

    @property
    def vectors(self) -> np.ndarray:
        """L2-normalised W_in embeddings — ready for cosine similarity."""
        norms = np.linalg.norm(self.W_in, axis=1, keepdims=True)
        return self.W_in / np.maximum(norms, 1e-10)

    def most_similar(self, word: str, vocab: "Vocabulary", topn: int = 10):
        """Return the topn most similar words by cosine similarity."""
        if word not in vocab.word2id:
            return []
        idx = vocab.word2id[word]
        query = self.vectors[idx]                      # (d,)
        sims  = self.vectors @ query                   # (V,)
        sims[idx] = -1.0                               # exclude query itself
        top_ids = np.argpartition(sims, -topn)[-topn:]
        top_ids = top_ids[np.argsort(sims[top_ids])[::-1]]
        return [(vocab.id2word[i], float(sims[i])) for i in top_ids]
