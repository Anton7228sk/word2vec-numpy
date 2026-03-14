import numpy as np


def sigmoid(x):
    # Numerically stable: avoids overflow for large negative x
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


class Word2Vec:
    def __init__(self, vocab_size, embed_dim):
        scale = 0.5 / embed_dim
        # Input (center word) and output (context word) embedding matrices
        self.W_in  = np.random.uniform(-scale, scale, (vocab_size, embed_dim))
        self.W_out = np.zeros((vocab_size, embed_dim))

    def nearest_neighbors(self, word, word_to_idx, idx_to_word, k=10):
        if word not in word_to_idx:
            return []
        vec   = self.W_in[word_to_idx[word]]
        norms = np.linalg.norm(self.W_in, axis=1)
        sims  = self.W_in @ vec / (norms + 1e-10) / (np.linalg.norm(vec) + 1e-10)
        top_k = np.argsort(-sims)[1:k + 1]
        return [(idx_to_word[i], float(sims[i])) for i in top_k]
