import torch
import custom_extension
import math
import matplotlib.pyplot as plt
import numpy as np

def create_sinusoidal_positions(seq_length, dim, base=10000.0):
    """Reference implementation of position encodings"""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    pos_seq = torch.arange(seq_length).float()
    sinusoid = torch.einsum('i,j->ij', pos_seq, inv_freq)
    return torch.cat([torch.sin(sinusoid), torch.cos(sinusoid)], dim=-1)

def rotate_every_two(x):
    """Rotate every two elements in the last dimension"""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack([-x2, x1], dim=-1)
    return x.flatten(-2)

def apply_rotary_pos_emb_reference(q, k, cos, sin):
    """Reference implementation of RoPE"""
    # Reshape q and k to match the complex rotation
    q_real, q_imag = q[..., ::2], q[..., 1::2]
    k_real, k_imag = k[..., ::2], k[..., 1::2]
    
    # Apply complex multiplication
    q_out_real = q_real * cos - q_imag * sin
    q_out_imag = q_real * sin + q_imag * cos
    k_out_real = k_real * cos - k_imag * sin
    k_out_imag = k_real * sin + k_imag * cos
    
    # Stack and flatten
    q_out = torch.stack([q_out_real, q_out_imag], dim=-1).flatten(-2)
    k_out = torch.stack([k_out_real, k_out_imag], dim=-1).flatten(-2)
    
    return q_out, k_out

def test_rotary_embeddings():
    print("\n=== Testing Rotary Position Embeddings ===")
    
    # Test parameters
    batch_size = 2
    seq_length = 32
    num_heads = 4
    head_dim = 64
    rope_base = 10000.0
    
    # Create test tensors
    query = torch.randn(batch_size, num_heads, seq_length, head_dim)
    key = torch.randn(batch_size, num_heads, seq_length, head_dim)
    
    # Get rotary matrices from our implementation
    device = query.device
    dtype = query.dtype
    cos, sin = custom_extension.create_rope_rotary_matrices(
        seq_length=seq_length,
        dim=head_dim,
        base=rope_base,
        options=torch.TensorOptions().device(device).dtype(dtype)
    )
    
    # Apply our implementation
    custom_q = custom_extension.apply_rotary_embeddings(query, cos, sin)
    custom_k = custom_extension.apply_rotary_embeddings(key, cos, sin)
    
    # Apply reference implementation
    ref_q, ref_k = apply_rotary_pos_emb_reference(query, key, cos, sin)
    
    # Compare results
    q_diff = (custom_q - ref_q).abs().max().item()
    k_diff = (custom_k - ref_k).abs().max().item()
    
    print(f"\nMaximum differences:")
    print(f"Query difference: {q_diff:.6f}")
    print(f"Key difference: {k_diff:.6f}")
    
    # Visualize position encodings
    plt.figure(figsize=(15, 5))
    
    # Plot original vs rotated attention patterns
    plt.subplot(131)
    attn_orig = torch.matmul(query[0, 0], key[0, 0].transpose(-2, -1))
    plt.imshow(attn_orig.detach().numpy())
    plt.title("Original Attention Pattern")
    plt.colorbar()
    
    plt.subplot(132)
    attn_rope = torch.matmul(custom_q[0, 0], custom_k[0, 0].transpose(-2, -1))
    plt.imshow(attn_rope.detach().numpy())
    plt.title("RoPE Attention Pattern")
    plt.colorbar()
    
    # Plot relative position sensitivity
    plt.subplot(133)
    rel_pos = []
    for i in range(seq_length):
        for j in range(seq_length):
            rel_pos.append(abs(i - j))
    rel_pos = np.array(rel_pos)
    attn_flat = attn_rope.flatten().detach().numpy()
    
    plt.scatter(rel_pos, attn_flat, alpha=0.5)
    plt.xlabel("Relative Position")
    plt.ylabel("Attention Score")
    plt.title("Position Sensitivity")
    
    plt.tight_layout()
    plt.savefig('rope_analysis.png')
    plt.close()
    
    # Test attention computation
    print("\nTesting attention computation with RoPE...")
    
    # Create attention inputs
    query_layer = torch.randn(batch_size, seq_length, head_dim * num_heads)
    key_layer = torch.randn(batch_size, seq_length, head_dim * num_heads)
    value_layer = torch.randn(batch_size, seq_length, head_dim * num_heads)
    
    # Compute attention with RoPE
    output, weights = custom_extension.rope_attention(
        query_layer,
        key_layer,
        value_layer,
        num_heads,
        torch.Tensor(),  # no mask
        0.0,  # no dropout
        True,  # need weights
        False,  # don't average weights
        rope_base,
        False  # not training
    )
    
    print("\nOutput shape:", output.shape)
    print("Attention weights shape:", weights.shape)
    
    # Verify rotary properties
    print("\nVerifying rotary properties...")
    
    # 1. Translation equivariance
    shift = 3
    shifted_q = torch.roll(query_layer, shifts=shift, dims=1)
    shifted_k = torch.roll(key_layer, shifts=shift, dims=1)
    shifted_v = torch.roll(value_layer, shifts=shift, dims=1)
    
    shifted_output, shifted_weights = custom_extension.rope_attention(
        shifted_q, shifted_k, shifted_v,
        num_heads, torch.Tensor(), 0.0, True, False, rope_base, False
    )
    
    # The attention pattern should be similarly shifted
    weights_shift_diff = (
        torch.roll(weights, shifts=shift, dims=-1) - 
        shifted_weights
    ).abs().mean().item()
    
    print(f"Average difference after shifting: {weights_shift_diff:.6f}")
    
    # 2. Relative position sensitivity
    print("\nRelative position correlations:")
    for pos in [1, 2, 4, 8]:
        correlation = torch.corrcoef(
            torch.stack([
                weights[0, 0, :, :].diagonal(offset=pos).flatten(),
                weights[0, 0, :, :].diagonal(offset=-pos).flatten()
            ])
        )[0, 1].item()
        print(f"Position ±{pos}: {correlation:.3f}")

if __name__ == "__main__":
    test_rotary_embeddings()