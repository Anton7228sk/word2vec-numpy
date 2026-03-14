import numpy as np

from model import sigmoid


def make_noise_table(noise_dist, size=10_000_000):
    # Pre-sample once per run; ~10x faster than calling np.random.choice(p=...) each step
    return np.random.choice(len(noise_dist), size=size, p=noise_dist).astype(np.int32)


def train(model, pairs, noise_dist,
          epochs=5, lr=0.025, num_neg=5, batch_size=512, log_every=500_000):

    pairs_arr = np.array(pairs, dtype=np.int32)
    n = len(pairs_arr)

    noise_table = make_noise_table(noise_dist)
    noise_ptr   = 0

    for epoch in range(epochs):
        np.random.shuffle(pairs_arr)
        total_loss = 0.0
        lr_epoch   = lr * (1.0 - epoch / (epochs + 1))

        for start in range(0, n, batch_size):
            batch = pairs_arr[start:start + batch_size]
            B     = len(batch)

            center_idx  = batch[:, 0]   # (B,)
            context_idx = batch[:, 1]   # (B,)

            # Pull negatives from pre-built table, wrap around if exhausted
            end = noise_ptr + B * num_neg
            if end <= len(noise_table):
                neg_flat = noise_table[noise_ptr:end]
            else:
                noise_table[:] = make_noise_table(noise_dist, len(noise_table))
                noise_ptr = 0
                neg_flat  = noise_table[:B * num_neg]
                end       = B * num_neg
            noise_ptr = end % len(noise_table)
            neg_indices = neg_flat.reshape(B, num_neg)   # (B, K)

            # Forward
            u_c   = model.W_in[center_idx]               # (B, D)
            v_pos = model.W_out[context_idx]             # (B, D)
            v_neg = model.W_out[neg_indices]             # (B, K, D)

            pos_scores = np.einsum('bd,bd->b', v_pos, u_c)          # (B,)
            neg_scores = np.einsum('bkd,bd->bk', v_neg, u_c)        # (B, K)

            pos_sig = sigmoid(pos_scores)   # (B,)
            neg_sig = sigmoid(neg_scores)   # (B, K)

            total_loss += (-np.sum(np.log(pos_sig + 1e-10))
                           - np.sum(np.log(np.maximum(1.0 - neg_sig, 1e-10))))

            # Backward
            d_pos      = pos_sig - 1.0                                           # (B,)
            grad_u_c   = d_pos[:, None] * v_pos + np.einsum('bk,bkd->bd', neg_sig, v_neg)
            grad_v_pos = d_pos[:, None] * u_c                                    # (B, D)
            grad_v_neg = neg_sig[:, :, None] * u_c[:, None, :]                  # (B, K, D)

            # np.add.at correctly accumulates when indices repeat within a batch
            np.add.at(model.W_in,  center_idx,  -lr_epoch * grad_u_c)
            np.add.at(model.W_out, context_idx, -lr_epoch * grad_v_pos)
            np.add.at(model.W_out, neg_indices, -lr_epoch * grad_v_neg)

            done = start + B
            if done % log_every < batch_size:
                print(f"  step {done:>9,}/{n:,}  loss {total_loss / done:.4f}")

        print(f"Epoch {epoch + 1}/{epochs}  avg loss {total_loss / n:.4f}  lr {lr_epoch:.5f}")
