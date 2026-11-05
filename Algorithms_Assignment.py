import numpy as np

# ---------------------------------------------------------
#  Algorithm 1: Token embedding
# ---------------------------------------------------------
def token_embedding(v, W_e):
    return W_e[:, v]

# ---------------------------------------------------------
# Algorithm 6: layer_norm
# ---------------------------------------------------------
def layer_norm(e, gamma, beta, epsilon=1e-5):
    d_e = e.shape[0]
    m = np.sum(e) / d_e
    v = np.sum((e - m) ** 2) / d_e
    return ((e - m) / np.sqrt(v + epsilon)) * gamma + beta

# ---------------------------------------------------------
#  Algorithm 4: ATTENTION
# ---------------------------------------------------------
def attention(X, Z, W_qkv, Mask):
    W_q, b_q = W_qkv['W_q'], W_qkv['b_q']
    W_k, b_k = W_qkv['W_k'], W_qkv['b_k']
    W_v, b_v = W_qkv['W_v'], W_qkv['b_v']
    
    d_attn = W_q.shape[0]
    
    b_q, b_k, b_v = b_q.reshape(-1, 1), b_k.reshape(-1, 1), b_v.reshape(-1, 1)
    
    Q = W_q @ X + b_q
    K = W_k @ Z + b_k
    V = W_v @ Z + b_v
    
    S = (K.T @ Q) / np.sqrt(d_attn)
    S = np.where(Mask == 1, S, -np.inf)
    
    S_max = np.max(S, axis=0, keepdims=True)
    exp_S = np.exp(S - S_max)
    softmax_S = exp_S / np.sum(exp_S, axis=0, keepdims=True)
    
    return V @ softmax_S

# ---------------------------------------------------------
#  Algorithm 5: MHATTENTION
# ---------------------------------------------------------
def mhattention(X, Z, W, Mask, H):
    W_o, b_o = W['W_o'], W['b_o']
    b_o = b_o.reshape(-1, 1)
    
    Y_heads = []
    for h in range(H):
        Y_h = attention(X, Z, W['heads'][h], Mask)
        Y_heads.append(Y_h)
        
    Y = np.concatenate(Y_heads, axis=0)
    return W_o @ Y + b_o


if __name__ == "__main__":
    np.random.seed(42) # For reproducible output

    # GPT-2 Small Hyperparameters
    d_e = 768           # Model width
    N_V = 50257         # Vocabulary size
    H = 12              # Attention heads
    d_attn = 64         # Dimension per head
    l = 4               # Let's simulate a short input sequence of 4 tokens

    print("--- 1. Token Embedding ---")
    W_e = np.random.randn(d_e, N_V) * 0.02
    token_id = 1540  # arbitrary token ID
    e = token_embedding(token_id, W_e)
    print(f"Token {token_id} embedding shape: {e.shape}")
    print(f"First 3 values: {e[:3]}")

    print("\n--- 2. Layer Normalization ---")
    gamma = np.ones(d_e)
    beta = np.zeros(d_e)
    e_norm = layer_norm(e, gamma, beta)
    print(f"Normalized e shape: {e_norm.shape}")
    print(f"Mean (should be ~0): {np.mean(e_norm):.4f}")
    print(f"Variance (should be ~1): {np.var(e_norm):.4f}")

    print("\n--- 3. Setup for Attention (Generating a 4-token sequence matrix) ---")
    X = np.random.randn(d_e, l)  # Primary sequence
    Z = X                        # Context sequence (Self-Attention)
    
    # Causal Mask (t_z <= t_x -> context position <= primary position)
    Mask = np.array([[1 if tz <= tx else 0 for tx in range(l)] for tz in range(l)])
    print("Causal Mask:")
    print(Mask)

    print("\n--- 4. Single-Head Attention ---")
    W_qkv_single = {
        'W_q': np.random.randn(d_attn, d_e) * 0.02, 'b_q': np.zeros(d_attn),
        'W_k': np.random.randn(d_attn, d_e) * 0.02, 'b_k': np.zeros(d_attn),
        'W_v': np.random.randn(d_attn, d_e) * 0.02, 'b_v': np.zeros(d_attn),
    }
    V_tilde_single = attention(X, Z, W_qkv_single, Mask)
    print(f"Single Head Output Shape (d_out, sequence_length): {V_tilde_single.shape}")

    print("\n--- 5. Multi-Head Attention (12 Heads) ---")
    # Generate random parameters for all 12 heads
    heads_params = []
    for _ in range(H):
        heads_params.append({
            'W_q': np.random.randn(d_attn, d_e) * 0.02, 'b_q': np.zeros(d_attn),
            'W_k': np.random.randn(d_attn, d_e) * 0.02, 'b_k': np.zeros(d_attn),
            'W_v': np.random.randn(d_attn, d_e) * 0.02, 'b_v': np.zeros(d_attn),
        })
    
    W_mh = {
        'W_o': np.random.randn(d_e, H * d_attn) * 0.02, 
        'b_o': np.zeros(d_e),
        'heads': heads_params
    }
    
    V_tilde_mh = mhattention(X, Z, W_mh, Mask, H)
    print(f"Multi-Head Output Shape (d_e, sequence_length): {V_tilde_mh.shape}")
    print(f"Output for first token (first 3 values): {V_tilde_mh[:3, 0]}")