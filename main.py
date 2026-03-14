import os
import zipfile
import urllib.request
import numpy as np

from data  import load_text, tokenize, build_vocab, subsample, make_noise_dist, generate_pairs
from model import Word2Vec
from train import train

# Hyperparameters
EMBED_DIM   = 100
WINDOW_SIZE = 5
NUM_NEG     = 5
EPOCHS      = 3
LR          = 0.025
MIN_COUNT   = 5
MAX_TOKENS  = 5_000_000   # cap for manageable runtime; use None for full dataset

DATA_URL  = 'http://mattmahoney.net/dc/text8.zip'
DATA_ZIP  = 'text8.zip'
DATA_FILE = 'text8.txt'


def maybe_download():
    if os.path.exists(DATA_FILE):
        return
    if os.path.exists(DATA_ZIP) and not zipfile.is_zipfile(DATA_ZIP):
        print("Removing corrupted zip ...")
        os.remove(DATA_ZIP)

    if not os.path.exists(DATA_ZIP):
        print(f"Downloading {DATA_URL} ...")
        req = urllib.request.urlopen(DATA_URL, timeout=30)
        total = int(req.headers.get('Content-Length', 0))
        downloaded = 0
        chunk_size = 1 << 15  # 32 KB
        with open(DATA_ZIP, 'wb') as f:
            while True:
                chunk = req.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB  ({pct:.0f}%)", end='', flush=True)
        print()
    print("Extracting ...")
    with zipfile.ZipFile(DATA_ZIP) as z:
        with z.open('text8') as src, open(DATA_FILE, 'w', encoding='utf-8') as dst:
            dst.write(src.read().decode('utf-8'))


if __name__ == '__main__':
    maybe_download()

    print("Preprocessing ...")
    text   = load_text(DATA_FILE)
    tokens = tokenize(text)
    if MAX_TOKENS:
        tokens = tokens[:MAX_TOKENS]

    word_to_idx, vocab, counts = build_vocab(tokens, min_count=MIN_COUNT)
    tokens = subsample(tokens, word_to_idx, counts)
    vocab_size = len(word_to_idx)
    print(f"Vocab: {vocab_size:,}  tokens after subsampling: {len(tokens):,}")

    noise_dist = make_noise_dist(vocab, counts)

    print("Generating skip-gram pairs ...")
    pairs = generate_pairs(tokens, word_to_idx, window_size=WINDOW_SIZE)
    print(f"Pairs: {len(pairs):,}")

    model = Word2Vec(vocab_size, EMBED_DIM)

    print("Training ...")
    train(model, pairs, noise_dist,
          epochs=EPOCHS, lr=LR, num_neg=NUM_NEG)

    np.save('embeddings_in.npy',  model.W_in)
    np.save('embeddings_out.npy', model.W_out)
    print("Saved embeddings_in.npy  embeddings_out.npy")

    # Quick sanity check: nearest neighbors for a few probe words
    probes = ['king', 'paris', 'python', 'science']
    print()
    for word in probes:
        neighbors = model.nearest_neighbors(word, word_to_idx, vocab, k=7)
        if neighbors:
            nn_str = ', '.join(f"{w} ({s:.3f})" for w, s in neighbors)
            print(f"{word}: {nn_str}")
