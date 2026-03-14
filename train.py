"""
Training script for Word2Vec (skip-gram + negative sampling).

Usage
-----
# Download text8 first:
#   wget http://mattmahoney.net/dc/text8.zip

python train.py --data text8.zip --epochs 5 --embed-dim 100

# Or with any plain text file:
python train.py --data my_corpus.txt --epochs 3 --embed-dim 50
"""

import argparse
import time
from pathlib import Path

import numpy as np

from dataset import (
    Vocabulary, NegativeSampler,
    load_text8, load_plain_text,
    subsample, generate_pairs,
)
from word2vec import Word2Vec
from evaluate import evaluate_similarity


def parse_args():
    p = argparse.ArgumentParser(description="Train Word2Vec (skip-gram + NS)")
    p.add_argument("--data",        default="text8.zip", help="Path to text8.zip or any .txt file")
    p.add_argument("--embed-dim",   type=int,   default=100,   help="Embedding dimensionality")
    p.add_argument("--window",      type=int,   default=5,     help="Max context window size")
    p.add_argument("--negatives",   type=int,   default=5,     help="Number of negative samples")
    p.add_argument("--min-count",   type=int,   default=5,     help="Min word frequency")
    p.add_argument("--subsample-t", type=float, default=1e-5,  help="Subsampling threshold")
    p.add_argument("--lr",          type=float, default=0.025, help="Initial learning rate")
    p.add_argument("--epochs",      type=int,   default=5,     help="Number of training epochs")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--save",        default="vectors.npy",     help="Where to save W_in")
    p.add_argument("--log-every",   type=int,   default=100_000, help="Print loss every N steps")
    return p.parse_args()


def linear_lr_decay(initial_lr: float, step: int, total_steps: int) -> float:
    """Linearly decay lr to 1e-4 * initial_lr over training."""
    progress = step / total_steps
    return max(initial_lr * (1.0 - progress), initial_lr * 1e-4)


def main():
    args = parse_args()
    np.random.seed(args.seed)

    # ------------------------------------------------------------------
    # 1. Load corpus
    # ------------------------------------------------------------------
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found.\n"
            "Download text8: wget http://mattmahoney.net/dc/text8.zip"
        )
    print(f"Loading corpus from {data_path} …")
    if data_path.suffix == ".zip" or "text8" in data_path.name:
        tokens = load_text8(str(data_path))
    else:
        tokens = load_plain_text(str(data_path))
    print(f"  Raw tokens: {len(tokens):,}")

    # ------------------------------------------------------------------
    # 2. Build vocabulary
    # ------------------------------------------------------------------
    vocab = Vocabulary(min_count=args.min_count).build(tokens)
    print(f"  Vocabulary size: {len(vocab):,} (min_count={args.min_count})")

    # ------------------------------------------------------------------
    # 3. Subsample frequent words
    # ------------------------------------------------------------------
    tokens = subsample(tokens, vocab, t=args.subsample_t)
    print(f"  Tokens after subsampling: {len(tokens):,}")

    # Convert to integer IDs (drop unknowns)
    token_ids = np.array(
        [vocab.word2id[w] for w in tokens if w in vocab.word2id],
        dtype=np.int64,
    )

    # ------------------------------------------------------------------
    # 4. Generate skip-gram pairs
    # ------------------------------------------------------------------
    print("Generating skip-gram pairs …")
    pairs = generate_pairs(token_ids, window_size=args.window)
    print(f"  Total pairs: {len(pairs):,}")

    # ------------------------------------------------------------------
    # 5. Initialise model and negative sampler
    # ------------------------------------------------------------------
    model   = Word2Vec(len(vocab), args.embed_dim, args.lr, seed=args.seed)
    sampler = NegativeSampler(vocab, num_negatives=args.negatives)
    pairs_arr = np.array(pairs, dtype=np.int64)

    # ------------------------------------------------------------------
    # 6. Training loop
    # ------------------------------------------------------------------
    total_steps = len(pairs) * args.epochs
    step        = 0
    initial_lr  = args.lr
    t0          = time.time()

    print(f"\nStarting training — {args.epochs} epoch(s), {total_steps:,} steps total\n")

    for epoch in range(1, args.epochs + 1):
        # Shuffle pairs each epoch
        idx = np.random.permutation(len(pairs_arr))
        shuffled = pairs_arr[idx]

        running_loss = 0.0
        log_count    = 0

        for center_id, context_id in shuffled:
            neg_ids = sampler.sample(args.negatives, exclude=(int(center_id), int(context_id)))

            # Update learning rate
            model.lr = linear_lr_decay(initial_lr, step, total_steps)

            loss = model.train_step(int(center_id), int(context_id), neg_ids)
            running_loss += loss
            log_count    += 1
            step         += 1

            if step % args.log_every == 0:
                avg_loss  = running_loss / log_count
                elapsed   = time.time() - t0
                progress  = step / total_steps * 100
                print(
                    f"  epoch {epoch}  |  progress {progress:5.1f}%  "
                    f"|  loss {avg_loss:.4f}  |  lr {model.lr:.6f}  "
                    f"|  {elapsed:.0f}s elapsed"
                )
                running_loss = 0.0
                log_count    = 0

        # Epoch-level quick evaluation
        print(f"\n--- Epoch {epoch} complete ---")
        evaluate_similarity(model, vocab, probes=["king", "man", "woman", "paris", "france"])
        print()

    # ------------------------------------------------------------------
    # 7. Save vectors
    # ------------------------------------------------------------------
    save_path = Path(args.save)
    np.save(save_path, model.W_in)
    print(f"Saved W_in vectors to {save_path}")

    # Optionally save vocabulary
    import json
    vocab_path = save_path.with_suffix(".vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(vocab.word2id, f)
    print(f"Saved vocabulary to {vocab_path}")


if __name__ == "__main__":
    main()
