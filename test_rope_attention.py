import torch
import custom_extension
import math
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

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
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create test tensors with smaller values to avoid numerical instability
    query = torch.randn(batch_size, num_heads, seq_length, head_dim) * 0.1
    key = torch.randn(batch_size, num_heads, seq_length, head_dim) * 0.1
    
    # Get rotary matrices from our implementation
    cos, sin = custom_extension.create_rope_rotary_matrices(
        seq_length=seq_length,
        dim=head_dim,
        base=rope_base,
        options=None
    )
    
    # Move tensors to match query device/dtype if needed
    cos = cos.to(device=query.device, dtype=query.dtype)
    sin = sin.to(device=query.device, dtype=query.dtype)
    
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
    
    # Test attention computation
    print("\nTesting attention computation with RoPE...")
    
    # Create attention inputs with smaller values
    query_layer = torch.randn(batch_size, seq_length, head_dim * num_heads) * 0.1
    key_layer = torch.randn(batch_size, seq_length, head_dim * num_heads) * 0.1
    value_layer = torch.randn(batch_size, seq_length, head_dim * num_heads) * 0.1
    
    # Create proper causal attention mask with better numerical stability
    attn_mask = torch.triu(
        torch.zeros(seq_length, seq_length) - 65504,  # Use float16 max negative instead of -1e4
        diagonal=1
    ).to(device=query_layer.device, dtype=query_layer.dtype)
    
    # Scale inputs for better numerical stability
    query_scale = math.sqrt(1.0 / (head_dim * num_heads))
    query_layer = query_layer * query_scale
    key_layer = key_layer * query_scale
    value_layer = value_layer * query_scale
    
    print("\nInput stats:")
    print(f"Query mean: {query_layer.mean():.6f}, std: {query_layer.std():.6f}")
    print(f"Key mean: {key_layer.mean():.6f}, std: {key_layer.std():.6f}")
    print(f"Value mean: {value_layer.mean():.6f}, std: {value_layer.std():.6f}")
    print(f"Mask min: {attn_mask.min():.1f}, max: {attn_mask.max():.1f}")
    print(f"Scale factor: {query_scale:.6f}")
    
    # Normalize weights before any shifting or analysis
    def normalize_attention(attn_weights):
        # Remove any extreme values
        attn_weights = torch.clamp(attn_weights, min=-100, max=100)
        # Apply softmax per head
        return torch.nn.functional.softmax(attn_weights, dim=-1)
    
    # Get attention outputs
    output, weights = custom_extension.rope_attention(
        query_layer,
        key_layer,
        value_layer,
        num_heads,
        attn_mask,
        0.0,
        True,
        False,
        rope_base,
        False
    )
    
    # Normalize weights immediately
    weights = normalize_attention(weights)
    
    print("\nOutput stats:")
    print(f"Output mean: {output.mean():.6f}, std: {output.std():.6f}")
    print(f"Normalized weights mean: {weights.mean():.6f}, std: {weights.std():.6f}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    
    # Visualize attention patterns
    plt.figure(figsize=(15, 5))
    
    # Plot attention pattern for first head
    plt.subplot(131)
    plt.imshow(weights[0, 0].detach().numpy(), cmap='viridis')
    plt.title("Attention Pattern (Head 0)")
    plt.colorbar()
    
    # Plot attention pattern for second head
    plt.subplot(132)
    plt.imshow(weights[0, 1].detach().numpy(), cmap='viridis')
    plt.title("Attention Pattern (Head 1)")
    plt.colorbar()
    
    # Plot average attention pattern across heads
    plt.subplot(133)
    plt.imshow(weights[0].mean(0).detach().numpy(), cmap='viridis')
    plt.title("Average Attention Pattern")
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('attention_patterns.png')
    plt.close()
    
    # Verify rotary properties
    print("\nVerifying rotary properties...")
    
    # 1. Translation equivariance
    shift = 3
    shifted_q = torch.roll(query_layer, shifts=shift, dims=1)
    shifted_k = torch.roll(key_layer, shifts=shift, dims=1)
    shifted_v = torch.roll(value_layer, shifts=shift, dims=1)
    
    _, shifted_weights = custom_extension.rope_attention(
        shifted_q, 
        shifted_k, 
        shifted_v,
        num_heads, 
        attn_mask,
        0.0, 
        True, 
        False, 
        rope_base, 
        False
    )
    
    # Normalize shifted weights
    shifted_weights = normalize_attention(shifted_weights)
    
    # Compare attention patterns
    weights_shift_diff = (
        torch.roll(weights, shifts=shift, dims=-1) - 
        shifted_weights
    ).abs().mean().item()
    
    print(f"Average difference after shifting: {weights_shift_diff:.6f}")
    
    # 2. Analyze relative position sensitivity
    for head in range(num_heads):
        head_weights = weights[0, head]
        
        # Calculate average attention weight for each relative position
        rel_pos_weights = {}
        for pos in range(-seq_length+1, seq_length):
            diag_vals = head_weights.diagonal(offset=pos)
            if len(diag_vals) > 0:
                avg_weight = diag_vals.mean().item()
                rel_pos_weights[pos] = avg_weight
        
        # Print strongest attention positions
        sorted_pos = sorted(rel_pos_weights.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # Plot relative position sensitivity
    plt.figure(figsize=(10, 5))
    positions = list(rel_pos_weights.keys())
    weights_list = [rel_pos_weights[p] for p in positions]
    
    plt.plot(positions, weights_list, 'b-', label='Average attention')
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Relative Position')
    plt.ylabel('Average Attention Weight')
    plt.title('Attention Weight vs Relative Position')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('position_sensitivity.png')
    plt.close()

if __name__ == "__main__":
    test_rotary_embeddings()