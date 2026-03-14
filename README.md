# Word2Vec — NumPy Implementation

Skip-gram with negative sampling, implemented from scratch using NumPy only (no PyTorch / TensorFlow).

## Method

**Skip-gram** predicts surrounding context words given a center word.
**Negative sampling** replaces the expensive full-softmax with a binary classification task: distinguish the true context word from *K* noise words sampled from a smoothed unigram distribution.

Loss for a single (center, context) pair with *K* negatives:

```
L = -log σ(v_pos · u_c) - Σ_k log σ(-v_neg_k · u_c)
```

Gradients (derived analytically, using d/dx[-log σ(x)] = σ(x) - 1):

```
∂L/∂u_c     = (σ(v_pos · u_c) - 1) · v_pos  +  Σ_k σ(v_neg_k · u_c) · v_neg_k
∂L/∂v_pos   = (σ(v_pos · u_c) - 1) · u_c
∂L/∂v_neg_k =  σ(v_neg_k · u_c) · u_c
```

Parameters are updated with vanilla SGD with linear learning rate decay.

## Project structure

```
data.py       tokenization, vocabulary, subsampling, skip-gram pair generation
model.py      Word2Vec class (embedding matrices + nearest-neighbor search)
train.py      training loop with mini-batch SGD
main.py       entry point: downloads data, runs training, prints nearest neighbors
```

## Key implementation details

- **Subsampling** — frequent words (the, a, …) are randomly discarded before pair generation, following Mikolov et al. (2013).
- **Noise distribution** — negative samples are drawn from the unigram distribution raised to the 3/4 power.
- **Noise table** — 10 M negative indices are pre-sampled once per run instead of calling `np.random.choice(p=...)` on every step, which is O(vocab\_size) and prohibitively slow.
- **Mini-batch training** — pairs are processed in batches of 512 using vectorised `einsum` operations, reducing Python overhead by ~50×.
- **Dynamic context window** — window size is sampled uniformly from [1, max\_window] for each center word, as in the original C implementation.

## Setup

```bash
pip install numpy
python main.py
```

The script downloads and extracts the [text8](http://mattmahoney.net/dc/text8.zip) corpus (~100 MB) automatically on first run.

Default hyperparameters (editable at the top of `main.py`):

| Parameter | Value |
|-----------|-------|
| Embedding dim | 100 |
| Window size | 5 |
| Negative samples | 5 |
| Epochs | 3 |
| Learning rate | 0.025 |
| Min word count | 5 |
| Max tokens | 5 000 000 |

## Sample output

Nearest neighbors by cosine similarity after 3 epochs on 5 M tokens:

```
king:    isabella (0.738), wessex (0.732), vii (0.722), sancho (0.717), vassal (0.715)
paris:   rodin (0.732), villa (0.711), bologna (0.695), nuremberg (0.693), leipzig (0.685)
python:  monty (0.823), clampett (0.811), rhino (0.808)   ← text8 is Wikipedia 2006; Monty Python dominates
science: fiction (0.718), psychology (0.718), humanities (0.707), zoology (0.700)
```

## References

Mikolov, T. et al. (2013). *Distributed Representations of Words and Phrases and their Compositionality*. NeurIPS.
