import re
import random
import numpy as np
from collections import Counter


def load_text(path):
    with open(path, encoding='utf-8') as f:
        return f.read()


def tokenize(text):
    return re.findall(r'[a-z]+', text.lower())


def build_vocab(tokens, min_count=5):
    counts = Counter(tokens)
    # Sort by frequency so high-frequency words get low indices
    words = [w for w, c in counts.most_common() if c >= min_count]
    word_to_idx = {w: i for i, w in enumerate(words)}
    return word_to_idx, words, counts


def subsample(tokens, word_to_idx, counts, t=1e-3):
    # Mikolov et al. subsampling: frequent words are dropped with higher probability
    total = sum(counts[w] for w in word_to_idx)
    keep_prob = {}
    for w in word_to_idx:
        f = counts[w] / total
        keep_prob[w] = (np.sqrt(f / t) + 1) * (t / f)
    return [w for w in tokens if w in word_to_idx and random.random() < min(keep_prob[w], 1.0)]


def make_noise_dist(words, counts):
    # Unigram distribution raised to 3/4 power, as proposed in the original paper
    freq = np.array([counts[w] ** 0.75 for w in words])
    return freq / freq.sum()


def generate_pairs(tokens, word_to_idx, window_size=5):
    indices = [word_to_idx[t] for t in tokens]
    pairs = []
    n = len(indices)
    for i, center in enumerate(indices):
        # Sample window size uniformly, as in the original word2vec
        w = random.randint(1, window_size)
        for j in range(max(0, i - w), min(n, i + w + 1)):
            if j != i:
                pairs.append((center, indices[j]))
    return pairs
