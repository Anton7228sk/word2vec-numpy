"""
Data loading, vocabulary building, and training-pair generation.

Subsampling of frequent words
------------------------------
Following the original word2vec paper, each word w is discarded with probability

    P_discard(w) = 1 − sqrt(t / freq(w))

where freq(w) is the word's relative frequency and t is a threshold (typically 1e-5).
This reduces the influence of very common words (the, a, …) and speeds up training.

Noise distribution for negative sampling
-----------------------------------------
    P_n(w) ∝ freq(w)^(3/4)

Raising to the 3/4 power smooths the distribution: rare words are sampled more
often than with unigram sampling, but less often than with a uniform distribution.
"""

import re
import zipfile
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Iterator

import numpy as np


class Vocabulary:
    """Maps words ↔ integer IDs; stores word frequencies."""

    def __init__(self, min_count: int = 5) -> None:
        self.min_count = min_count
        self.word2id: dict[str, int] = {}
        self.id2word: dict[int, str] = {}
        self.freq: np.ndarray = np.array([])   # raw counts, aligned with IDs

    def build(self, tokens: List[str]) -> "Vocabulary":
        counter = Counter(tokens)
        # Keep only words that appear at least min_count times
        vocab = [(w, c) for w, c in counter.most_common() if c >= self.min_count]
        for idx, (word, count) in enumerate(vocab):
            self.word2id[word] = idx
            self.id2word[idx]  = word
        self.freq = np.array([c for _, c in vocab], dtype=np.float64)
        return self

    def __len__(self) -> int:
        return len(self.word2id)


def load_text8(path: str) -> List[str]:
    """
    Load the text8 dataset.
    Accepts either a plain .txt file or a .zip containing text8.
    Download: http://mattmahoney.net/dc/text8.zip
    """
    p = Path(path)
    if p.suffix == ".zip":
        with zipfile.ZipFile(p) as z:
            name = z.namelist()[0]
            text = z.read(name).decode("utf-8")
    else:
        text = p.read_text(encoding="utf-8")
    return text.split()


def load_plain_text(path: str) -> List[str]:
    """
    Minimal tokenizer for any raw text file.
    Lowercases and keeps only alphabetic tokens.
    """
    text = Path(path).read_text(encoding="utf-8", errors="ignore").lower()
    return re.findall(r"[a-z]+", text)


# ---------------------------------------------------------------------------
# Subsampling
# ---------------------------------------------------------------------------

def subsample(tokens: List[str], vocab: Vocabulary, t: float = 1e-5) -> List[str]:
    """
    Discard frequent tokens with probability P = 1 − sqrt(t / freq(w)).
    Returns a new token list.
    """
    total = vocab.freq.sum()
    keep_prob = {}
    for word, idx in vocab.word2id.items():
        f = vocab.freq[idx] / total
        keep_prob[word] = min(1.0, np.sqrt(t / f) + t / f)

    rng = np.random.default_rng(0)
    return [
        w for w in tokens
        if w in vocab.word2id and rng.random() < keep_prob[w]
    ]


# ---------------------------------------------------------------------------
# Training-pair generation
# ---------------------------------------------------------------------------

def generate_pairs(
    token_ids: np.ndarray,
    window_size: int = 5,
) -> List[Tuple[int, int]]:
    """
    Yield (center_id, context_id) pairs using a dynamic window.

    For each position t the actual window w is drawn uniformly from [1, window_size].
    This gives closer words a higher expected sampling weight — same as the
    original C implementation.
    """
    rng = np.random.default_rng(1)
    pairs = []
    n = len(token_ids)
    for t in range(n):
        w = rng.integers(1, window_size + 1)
        start = max(0, t - w)
        end   = min(n, t + w + 1)
        center = token_ids[t]
        for j in range(start, end):
            if j != t:
                pairs.append((center, int(token_ids[j])))
    return pairs


# ---------------------------------------------------------------------------
# Negative sampler
# ---------------------------------------------------------------------------

class NegativeSampler:
    """
    Efficient negative sampler using P_n(w) ∝ freq(w)^(3/4).

    Implements the alias method so each draw is O(1) after O(V) precomputation.
    """

    def __init__(self, vocab: Vocabulary, num_negatives: int = 5) -> None:
        self.num_negatives = num_negatives
        probs = vocab.freq ** 0.75
        probs /= probs.sum()
        self._build_alias(probs)

    # ---- Alias method setup ----

    def _build_alias(self, probs: np.ndarray) -> None:
        n = len(probs)
        prob  = probs * n
        alias = np.zeros(n, dtype=np.int64)
        small, large = [], []
        for i, p in enumerate(prob):
            (small if p < 1.0 else large).append(i)
        while small and large:
            s, l = small.pop(), large.pop()
            alias[s] = l
            prob[l]  = prob[l] + prob[s] - 1.0
            (small if prob[l] < 1.0 else large).append(l)
        self._prob  = prob
        self._alias = alias

    def sample(self, size: int, exclude: Tuple[int, ...] = ()) -> np.ndarray:
        """Draw `size` negative indices (re-sampling any that hit `exclude`)."""
        rng    = np.random.default_rng()
        result = np.empty(size, dtype=np.int64)
        filled = 0
        exclude_set = set(exclude)
        while filled < size:
            # Vectorised alias draw
            i = rng.integers(0, len(self._prob), size=size - filled)
            u = rng.random(size - filled)
            drawn = np.where(u < self._prob[i], i, self._alias[i])
            for d in drawn:
                if d not in exclude_set:
                    result[filled] = d
                    filled += 1
                    if filled == size:
                        break
        return result
