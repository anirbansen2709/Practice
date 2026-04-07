import numpy as np

def vectorized_causal_attention(X, W_Q, W_K, W_V):
    """
    Computes causal attention for a sequence using vectorized operations.
    
    X: (seq_len, embed_dim)
    W_Q, W_K, W_V: (embed_dim, head_dim)
    """
    seq_len, d_k = X.shape[0], W_Q.shape[1]

    # 1. Linear Projections
    Q = X @ W_Q  # (seq_len, head_dim)
    K = X @ W_K  # (seq_len, head_dim)
    V = X @ W_V  # (seq_len, head_dim)

    # 2. Scaled Dot-Product: (seq_len, seq_len)
    # This matrix contains all possible pairings (including the future)
    logits = (Q @ K.T) / np.sqrt(d_k)

    # 3. Vectorized Causal Logic
    # We use np.tril_indices to find the "future" indices and set them to -inf
    # Alternatively, use np.triu (upper triangle) with k=1 to target the future
    future_mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
    logits[future_mask] = -1e9

    # 4. Softmax and Output
    # Subtract max for numerical stability (along rows)
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    # Output: (seq_len, head_dim)
    return probs @ V

# --- Example ---
X = np.random.randn(10, 32)  # 10 tokens, 32-dim embedding
W_Q = np.random.randn(32, 16)
W_K = np.random.randn(32, 16)
W_V = np.random.randn(32, 16)

context_vectors = vectorized_causal_attention(X, W_Q, W_K, W_V)
