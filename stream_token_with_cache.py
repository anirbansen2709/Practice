import numpy as np

def stream_token_with_cache(x, W_Q, W_K, W_V, kv_cache=None):
    """
    Processes a single token and updates the KV cache.
    
    x: (1, embed_dim) - The current token embedding
    W_Q, W_K, W_V: (embed_dim, head_dim) - Learned weights
    kv_cache: Tuple (K_past, V_past) or None
              Each element in tuple is (seq_len, head_dim)
    """
    # 1. Project the single current token
    q_current = x @ W_Q  # (1, head_dim)
    k_current = x @ W_K  # (1, head_dim)
    v_current = x @ W_V  # (1, head_dim)

    # 2. Update or Initialize Cache
    if kv_cache is None:
        # First token in the sequence
        k_total = k_current
        v_total = v_current
    else:
        k_past, v_past = kv_cache
        # Concatenate current K, V to the past: (seq_len + 1, head_dim)
        k_total = np.concatenate([k_past, k_current], axis=0)
        v_total = np.concatenate([v_past, v_current], axis=0)

    # 3. Scaled Dot-Product Attention
    d_k = q_current.shape[-1]
    
    # Scores: q_current (1, d_k) @ k_total.T (d_k, seq_len) -> (1, seq_len)
    logits = (q_current @ k_total.T) / np.sqrt(d_k)

    # 4. Softmax (Only 1 row, so axis=-1)
    # Note: No mask needed here! q_current is the latest token.
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)

    # 5. Output: probs (1, seq_len) @ v_total (seq_len, head_dim) -> (1, head_dim)
    output = probs @ v_total
    
    # Return context vector and the updated cache tuple
    return output, (k_total, v_total)

# --- Simulation of a Generation Loop ---
embed_dim, head_dim = 32, 16
W_Q = np.random.randn(embed_dim, head_dim)
W_K = np.random.randn(embed_dim, head_dim)
W_V = np.random.randn(embed_dim, head_dim)

cache = None
sequence_to_generate = 5

for i in range(sequence_to_generate):
    # Simulate a single input token
    x_input = np.random.randn(1, embed_dim)
    
    # Process token
    out, cache = stream_token_with_cache(x_input, W_Q, W_K, W_V, cache)
    
    print(f"Step {i+1}: Cache size is {cache[0].shape[0]} tokens")
