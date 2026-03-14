"""
Evaluation utilities for Word2Vec embeddings.

Supported evaluations
---------------------
1. Nearest neighbours   — qualitative check via cosine similarity
2. Word analogy         — king − man + woman ≈ queen  (3CosAdd)
3. SimLex-999 / WS-353  — Spearman correlation with human similarity judgements
                          (needs the dataset files; skipped gracefully if absent)

Usage (standalone)
------------------
python evaluate.py --vectors vectors.npy --vocab vectors.vocab.json
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Nearest-neighbour probe (printed during training)
# ---------------------------------------------------------------------------

def evaluate_similarity(model, vocab, probes: List[str], topn: int = 5) -> None:
    """Quick qualitative check: print top-{topn} neighbours for each probe word."""
    for word in probes:
        neighbours = model.most_similar(word, vocab, topn=topn)
        if neighbours:
            nb_str = ", ".join(f"{w} ({s:.3f})" for w, s in neighbours)
            print(f"  {word:15s} → {nb_str}")
        else:
            print(f"  {word:15s} → (not in vocabulary)")


# ---------------------------------------------------------------------------
# Analogy evaluation  (3CosAdd)
# ---------------------------------------------------------------------------

def analogy(vectors: np.ndarray, word2id: dict, id2word: dict,
            a: str, b: str, c: str, topn: int = 5) -> List[Tuple[str, float]]:
    """
    Find word d such that  a : b  ::  c : d
    using the 3CosAdd method:
        d* = argmax_d [ cos(d, b) − cos(d, a) + cos(d, c) ]

    Parameters
    ----------
    a, b, c : the three query words
    Returns list of (word, score) for the topn candidates (excluding a, b, c).
    """
    missing = [w for w in (a, b, c) if w not in word2id]
    if missing:
        return []

    # L2-normalised vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    V = vectors / np.maximum(norms, 1e-10)

    # query vector
    query = V[word2id[b]] - V[word2id[a]] + V[word2id[c]]

    scores = V @ query  # (vocab_size,)
    for w in (a, b, c):
        scores[word2id[w]] = -np.inf

    top_ids = np.argpartition(scores, -topn)[-topn:]
    top_ids = top_ids[np.argsort(scores[top_ids])[::-1]]
    return [(id2word[i], float(scores[i])) for i in top_ids]


# ---------------------------------------------------------------------------
# Human-similarity benchmarks
# ---------------------------------------------------------------------------

def spearman_correlation(xs: np.ndarray, ys: np.ndarray) -> float:
    """Spearman ρ without scipy dependency."""
    n = len(xs)
    rank = lambda v: np.argsort(np.argsort(v)).astype(float)
    rx, ry = rank(xs), rank(ys)
    d2 = np.sum((rx - ry) ** 2)
    return 1.0 - 6.0 * d2 / (n * (n ** 2 - 1))


def evaluate_wordsim(vectors: np.ndarray, word2id: dict, dataset_path: str) -> float:
    """
    Evaluate on WordSim-353 or SimLex-999 (tab-separated: word1 word2 score).
    Returns Spearman ρ between model cosine similarities and human scores.
    """
    p = Path(dataset_path)
    if not p.exists():
        print(f"  Benchmark file not found: {dataset_path} — skipping.")
        return float("nan")

    human, model_scores = [], []
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    V = vectors / np.maximum(norms, 1e-10)

    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                parts = line.split()
            if len(parts) < 3:
                continue
            w1, w2, score = parts[0].lower(), parts[1].lower(), float(parts[2])
            if w1 in word2id and w2 in word2id:
                cos = float(V[word2id[w1]] @ V[word2id[w2]])
                human.append(score)
                model_scores.append(cos)

    if len(human) < 2:
        print("  Not enough pairs found in vocabulary.")
        return float("nan")

    rho = spearman_correlation(np.array(model_scores), np.array(human))
    print(f"  Spearman ρ = {rho:.4f}  ({len(human)} pairs evaluated)")
    return rho


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate saved Word2Vec vectors")
    parser.add_argument("--vectors", default="vectors.npy",          help="Path to saved W_in (.npy)")
    parser.add_argument("--vocab",   default="vectors.vocab.json",    help="Path to vocab JSON")
    parser.add_argument("--wordsim", default="",                      help="Optional WordSim-353 / SimLex-999 file")
    args = parser.parse_args()

    vectors  = np.load(args.vectors)
    with open(args.vocab) as f:
        word2id = json.load(f)
    id2word = {v: k for k, v in word2id.items()}

    # Normalise
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    V = vectors / np.maximum(norms, 1e-10)

    # ---- Nearest neighbours for sample words ----
    probe_words = ["king", "man", "woman", "paris", "france", "dog", "computer", "science"]
    print("\n=== Nearest Neighbours ===")
    for word in probe_words:
        if word not in word2id:
            continue
        idx   = word2id[word]
        sims  = V @ V[idx]
        sims[idx] = -1.0
        top   = np.argsort(sims)[::-1][:8]
        nb    = ", ".join(f"{id2word[i]} ({sims[i]:.3f})" for i in top)
        print(f"  {word:15s} → {nb}")

    # ---- Analogy ----
    print("\n=== Analogy (king − man + woman = ?) ===")
    results = analogy(vectors, word2id, id2word, "man", "king", "woman")
    if results:
        print("  Top candidates:", ", ".join(f"{w} ({s:.3f})" for w, s in results))
    else:
        print("  One or more analogy words not in vocabulary.")

    more_analogies = [
        ("paris", "france", "berlin"),
        ("good",  "better", "bad"),
        ("walk",  "walked", "run"),
    ]
    for a, b, c in more_analogies:
        res = analogy(vectors, word2id, id2word, a, b, c)
        if res:
            print(f"  {a} : {b}  ::  {c} : {res[0][0]}  (score {res[0][1]:.3f})")

    # ---- Similarity benchmark ----
    if args.wordsim:
        print("\n=== Human Similarity Benchmark ===")
        evaluate_wordsim(vectors, word2id, args.wordsim)


if __name__ == "__main__":
    main()
