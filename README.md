# Word2Vec — pure NumPy implementation

Skip-gram with Negative Sampling, implemented from scratch using only NumPy.

## Mathematical background

### Skip-gram objective

Given a corpus of tokens, for each center word **c** we want to maximise the likelihood of observing the surrounding context words **o** within a window of size **w**.

### Negative Sampling loss

For one `(center c, context o)` pair, with **K** negatives sampled from a noise distribution:

$$J = -\log \sigma(\mathbf{u}_o \cdot \mathbf{v}_c) \;-\; \sum_{k=1}^{K} \log \sigma(-\mathbf{u}_k \cdot \mathbf{v}_c)$$

where
- $\mathbf{v}_c = W_{\text{in}}[c]$ — center-word embedding
- $\mathbf{u}_o = W_{\text{out}}[o]$ — context-word embedding
- $\sigma(x) = \frac{1}{1+e^{-x}}$

### Gradients (derived by chain rule)

$$\frac{\partial J}{\partial \mathbf{v}_c} = (\sigma(\mathbf{u}_o \cdot \mathbf{v}_c) - 1)\,\mathbf{u}_o + \sum_{k} \sigma(\mathbf{u}_k \cdot \mathbf{v}_c)\,\mathbf{u}_k$$

$$\frac{\partial J}{\partial \mathbf{u}_o} = (\sigma(\mathbf{u}_o \cdot \mathbf{v}_c) - 1)\,\mathbf{v}_c$$

$$\frac{\partial J}{\partial \mathbf{u}_k} = \sigma(\mathbf{u}_k \cdot \mathbf{v}_c)\,\mathbf{v}_c \quad \text{for each negative } k$$

Updates use plain SGD with linear learning-rate decay.

### Noise distribution

Negative words are drawn from $P_n(w) \propto \text{freq}(w)^{3/4}$, implemented with the **alias method** for O(1) sampling.

### Subsampling of frequent words

Each token is discarded before training with probability

$$P_{\text{discard}}(w) = 1 - \sqrt{\frac{t}{\text{freq}(w)}}$$

(threshold $t = 10^{-5}$ by default), reducing the dominance of stop-words.

---

## Project structure

```
word2vec.py    — Word2Vec class: forward pass, gradients, parameter update
dataset.py     — Vocabulary, subsampling, pair generation, negative sampler
train.py       — Full training script (text8 or any .txt corpus)
evaluate.py    — Nearest-neighbour, analogy, and WordSim-353 evaluation
demo.py        — Self-contained demo on a tiny built-in corpus (no download needed)
requirements.txt
```

---

## Quick start

### Run the demo (no download needed)

```bash
pip install numpy
python demo.py
```

### Train on text8

```bash
# Download the corpus (~100 MB)
wget http://mattmahoney.net/dc/text8.zip

pip install numpy
python train.py --data text8.zip --epochs 5 --embed-dim 100

# Evaluate
python evaluate.py --vectors vectors.npy --vocab vectors.vocab.json
```

### Train on any text file

```bash
python train.py --data my_corpus.txt --epochs 3 --embed-dim 50
```

### All training options

```
--data         path to text8.zip or plain .txt file  (default: text8.zip)
--embed-dim    embedding dimensionality               (default: 100)
--window       maximum context window size            (default: 5)
--negatives    number of negative samples per pair    (default: 5)
--min-count    minimum word frequency                 (default: 5)
--subsample-t  subsampling threshold                  (default: 1e-5)
--lr           initial learning rate                  (default: 0.025)
--epochs       training epochs                        (default: 5)
--save         output path for W_in vectors (.npy)   (default: vectors.npy)
```

---

## Design decisions

| Choice | Rationale |
|--------|-----------|
| Two weight matrices `W_in`, `W_out` | Faithful to the original paper; reduces hubness in the embedding space |
| Alias method for negative sampling | O(1) per draw after O(V) pre-computation |
| Dynamic window (uniform draw in [1, w]) | Same as the original C implementation — closer words receive higher weight implicitly |
| Linear LR decay | Simple and effective; avoids the need for momentum or adaptive optimisers |
| `np.add.at` for negative updates | Correctly accumulates gradients when the same word appears multiple times in `neg_ids` |

---

## References

- Mikolov et al., *Efficient Estimation of Word Representations in Vector Space* (2013)
- Mikolov et al., *Distributed Representations of Words and Phrases and their Compositionality* (2013)
- Goldberg & Levy, *word2vec Explained* (2014)
