import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def linear_attention_forward_intra(
    Q_ptr, K_ptr, V_ptr, 
    stride_q, stride_k, stride_v,
    N, D,
    trunk_size: tl.constexpr
):
    # Each program processes one trunk
    trunk_idx = tl.program_id(0)
    start_row = trunk_idx * trunk_size
    row_idx = start_row + tl.arange(0, trunk_size)
    row_mask = row_idx < N
    
    # Load entire trunk for K and V (trunk_size x D)
    k_ptrs = K_ptr + row_idx[:, None] * stride_k + tl.arange(0, D)[None, :]
    v_ptrs = V_ptr + row_idx[:, None] * stride_v + tl.arange(0, D)[None, :]
    
    k_tile = tl.load(k_ptrs, mask=row_mask[:, None], other=0.0)
    v_tile = tl.load(v_ptrs, mask=row_mask[:, None], other=0.0)
    
    # Compute trunk-level attention matrix (D x D)
    attn_matrix = tl.dot(tl.trans(v_tile), k_tile)
    
    # Load Q trunk (trunk_size x D)
    q_ptrs = Q_ptr + row_idx[:, None] * stride_q + tl.arange(0, D)[None, :]
    q_tile = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0)
    
    # Compute trunk-level output
    output_trunk = tl.dot(q_tile, tl.trans(attn_matrix))
    
    # Store updated Q (intra-trunk contribution)
    tl.store(q_ptrs, output_trunk, mask=row_mask[:, None])

@triton.jit
def linear_attention_forward_inter(
    S_ptr,  # Global attention matrix (D x D)
    K_ptr, V_ptr,
    stride_k, stride_v, stride_s,
    N, D,
    trunk_size: tl.constexpr
):
    # Each program processes one trunk
    trunk_idx = tl.program_id(0)
    start_row = trunk_idx * trunk_size
    row_idx = start_row + tl.arange(0, trunk_size)
    row_mask = row_idx < N
    
    # Load entire trunk for K and V (trunk_size x D)
    k_ptrs = K_ptr + row_idx[:, None] * stride_k + tl.arange(0, D)[None, :]
    v_ptrs = V_ptr + row_idx[:, None] * stride_v + tl.arange(0, D)[None, :]
    
    k_tile = tl.load(k_ptrs, mask=row_mask[:, None], other=0.0)
    v_tile = tl.load(v_ptrs, mask=row_mask[:, None], other=0.0)
    
    # Compute trunk contribution to global attention matrix
    trunk_attn = tl.dot(tl.trans(v_tile), k_tile)
    
    # Atomically add to global matrix (D x D)
    s_ptrs = S_ptr + tl.arange(0, D)[:, None] * stride_s + tl.arange(0, D)[None, :]
    tl.atomic_add(s_ptrs, trunk_attn)

class LinearAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V):
        N, D = Q.shape
        device = Q.device
        dtype = Q.dtype
        
        # Configuration parameters
        trunk_size = 128  # Tune based on GPU characteristics
        
        # Initialize global attention matrix
        S = torch.zeros((D, D), device=device, dtype=torch.float32)
        
        # Compute number of trunks
        num_trunks = (N + trunk_size - 1) // trunk_size
        
        # Intra-trunk processing (updates Q in-place)
        grid_intra = (num_trunks,)
        linear_attention_forward_intra[grid_intra](
            Q, K, V,
            Q.stride(0), K.stride(0), V.stride(0),
            N, D,
            trunk_size=trunk_size
        )
        
        # Inter-trunk processing (accumulates to S)
        grid_inter = (num_trunks,)
        linear_attention_forward_inter[grid_inter](
            S, K, V,
            K.stride(0), V.stride(0), S.stride(0),
            N, D,
            trunk_size=trunk_size
        )
        
        # Apply global attention to Q
        Q = Q @ S.to(dtype)
        
        ctx.save_for_backward(Q, K, V, S)
        return Q

class LinearAttentionTriton(nn.Module):
    def forward(self, Q, K, V):
        return LinearAttention.apply(Q, K, V)

# Test code
if __name__ == "__main__":
    torch.manual_seed(0)
    N = 1024
    D = 768
    device = torch.device('cuda')
    
    Q = torch.randn(N, D, device=device)
    K = torch.randn(N, D, device=device)
    V = torch.randn(N, D, device=device)
    
    # Create module
    tla = LinearAttentionTriton()
    
    # Run forward pass
    output = tla(Q, K, V)
    print("Output shape:", output.shape)
    print("Output dtype:", output.dtype)