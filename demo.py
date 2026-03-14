"""
Self-contained demo that trains Word2Vec on a tiny built-in corpus
(no external dataset required). Runs in < 10 seconds.

python demo.py
"""

import textwrap
import numpy as np
from dataset import Vocabulary, NegativeSampler, subsample, generate_pairs
from word2vec import Word2Vec
from evaluate import analogy


# ---------------------------------------------------------------------------
# Tiny corpus — enough to sanity-check the implementation
# ---------------------------------------------------------------------------

CORPUS = textwrap.dedent("""
    the king rules the kingdom the king is powerful the king has a crown
    the queen rules the kingdom the queen is powerful the queen has a crown
    the man went to the city the man is strong the man works hard
    the woman went to the city the woman is strong the woman works hard
    paris is the capital of france france is a country in europe
    berlin is the capital of germany germany is a country in europe
    london is the capital of england england is a country in europe
    the dog runs in the park the cat sleeps at home
    the dog is a pet the cat is a pet dogs and cats are animals
    the computer processes data the machine learning model trains on data
    science studies the natural world mathematics is the language of science
    the king of france lives in paris the queen of england lives in london
    a man and a woman walked to the park the dog followed them
    the powerful king ruled the large kingdom for many years
    the young queen studied science and mathematics in the city
    france and germany are neighbors in europe england is an island
""").strip()


def tokenize(text: str):
    import re
    return re.findall(r"[a-z]+", text.lower())


def main():
    tokens = tokenize(CORPUS) * 200     # repeat to get more signal

    # Build vocab (no min_count here — corpus is tiny)
    vocab = Vocabulary(min_count=2).build(tokens)
    print(f"Vocabulary: {len(vocab)} words")

    # Subsample & encode
    tokens_sub = subsample(tokens, vocab, t=1e-4)
    ids = np.array([vocab.word2id[w] for w in tokens_sub if w in vocab.word2id])

    # Generate pairs
    pairs = generate_pairs(ids, window_size=5)
    print(f"Training pairs: {len(pairs):,}")

    # Model
    model   = Word2Vec(len(vocab), embed_dim=50, learning_rate=0.05, seed=42)
    sampler = NegativeSampler(vocab, num_negatives=5)

    # Training loop
    pairs_arr   = np.array(pairs, dtype=np.int64)
    total_steps = len(pairs_arr) * 3
    step        = 0

    for epoch in range(1, 4):
        idx  = np.random.permutation(len(pairs_arr))
        loss_acc = 0.0
        for center, context in pairs_arr[idx]:
            neg = sampler.sample(5, exclude=(int(center), int(context)))
            # Linear LR decay
            model.lr = max(0.05 * (1 - step / total_steps), 0.05 * 1e-4)
            loss_acc += model.train_step(int(center), int(context), neg)
            step     += 1
        print(f"Epoch {epoch}  avg loss: {loss_acc / len(pairs_arr):.4f}")

    # ---- Qualitative evaluation ----
    print("\n=== Nearest Neighbours ===")
    for word in ["king", "queen", "man", "woman", "france", "paris", "dog"]:
        neighbours = model.most_similar(word, vocab, topn=5)
        if neighbours:
            nb = ", ".join(f"{w}({s:.2f})" for w, s in neighbours)
            print(f"  {word:10s} : {nb}")

    print("\n=== Analogy: king - man + woman ===")
    res = analogy(model.W_in, vocab.word2id, vocab.id2word, "man", "king", "woman")
    if res:
        print("  Top:", ", ".join(f"{w}({s:.2f})" for w, s in res))

    print("\n=== Analogy: france - paris + berlin ===")
    res = analogy(model.W_in, vocab.word2id, vocab.id2word, "paris", "france", "berlin")
    if res:
        print("  Top:", ", ".join(f"{w}({s:.2f})" for w, s in res))


if __name__ == "__main__":
    main()
