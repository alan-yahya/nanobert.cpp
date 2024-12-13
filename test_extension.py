import torch
import custom_extension
import time
from typing import Callable
from torch.nn import functional as F

def benchmark_fn(fn: Callable, *args, num_runs: int = 100, warmup: int = 10):
    """Helper function to benchmark a function"""
    # Warmup runs
    for _ in range(warmup):
        fn(*args)
    
    # Actual timing
    start = time.perf_counter()
    for _ in range(num_runs):
        fn(*args)
    end = time.perf_counter()
    
    return (end - start) / num_runs

def test_matmul():
    print("\n=== Testing Matrix Multiplication ===")
    
    # Create 2D test tensors
    size = 512
    a = torch.randn(size, size)
    b = torch.randn(size, size)
    
    # Compare results
    custom_result = custom_extension.custom_matmul(a, b)
    torch_result = torch.mm(a, b)  # Use mm for 2D matrix multiplication
    
    if torch.allclose(custom_result, torch_result, rtol=1e-4, atol=1e-4):
        print("✓ Results match between custom and PyTorch implementations")
    else:
        print("✗ Results differ between implementations")
        max_diff = (custom_result - torch_result).abs().max().item()
        print(f"Max difference: {max_diff}")
    
    # Benchmark
    custom_time = benchmark_fn(lambda: custom_extension.custom_matmul(a, b))
    torch_time = benchmark_fn(lambda: torch.mm(a, b))
    
    print(f"\nBenchmark results (average over 100 runs):")
    print(f"Custom implementation: {custom_time*1000:.3f} ms")
    print(f"PyTorch built-in: {torch_time*1000:.3f} ms")
    print(f"Speed ratio (PyTorch/Custom): {torch_time/custom_time:.2f}x")

def pytorch_attention(query, key, value, num_heads, attn_mask=None, dropout_p=0.0):
    """PyTorch's implementation of multi-head attention for comparison"""
    batch_size, seq_length, embed_dim = query.size()
    head_dim = embed_dim // num_heads
    scaling = float(1.0 / (head_dim ** 0.5))
    
    # Reshape for multi-head attention
    query = query.reshape(batch_size, seq_length, num_heads, head_dim).transpose(1, 2)
    key = key.reshape(batch_size, seq_length, num_heads, head_dim).transpose(1, 2)
    value = value.reshape(batch_size, seq_length, num_heads, head_dim).transpose(1, 2)
    
    # Compute attention scores
    attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scaling
    
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask
    
    attn_weights = F.softmax(attn_weights, dim=-1)
    if dropout_p > 0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)
    
    # Apply attention to values
    output = torch.matmul(attn_weights, value)
    
    # Reshape output
    output = output.transpose(1, 2).contiguous()
    output = output.reshape(batch_size, seq_length, embed_dim)
    
    return output, attn_weights

def mlm_attention(query, key, value, num_heads, attn_mask=None, dropout_p=0.0):
    """HuggingFace's MLM attention implementation for comparison"""
    from transformers.models.bert.modeling_bert import BertSelfAttention
    import torch.nn as nn
    
    # Create a temporary config object
    class Config:
        def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
            self.hidden_size = hidden_size
            self.num_attention_heads = num_attention_heads
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.is_decoder = False  # Add this line to fix the error
    
    batch_size, seq_length, embed_dim = query.size()
    config = Config(
        hidden_size=embed_dim,
        num_attention_heads=num_heads,
        attention_probs_dropout_prob=dropout_p
    )
    
    # Create BERT attention layer
    attention = BertSelfAttention(config)
    
    # Set weights to identity matrices for fair comparison
    attention.query = nn.Linear(embed_dim, embed_dim, bias=False)
    attention.key = nn.Linear(embed_dim, embed_dim, bias=False)
    attention.value = nn.Linear(embed_dim, embed_dim, bias=False)
    
    with torch.no_grad():
        attention.query.weight.copy_(torch.eye(embed_dim))
        attention.key.weight.copy_(torch.eye(embed_dim))
        attention.value.weight.copy_(torch.eye(embed_dim))
    
    # Prepare attention mask
    if attn_mask is not None:
        attention_mask = attn_mask.unsqueeze(0).unsqueeze(0)
    else:
        attention_mask = None
    
    # Forward pass
    output = attention(
        hidden_states=query,
        attention_mask=attention_mask,
        output_attentions=True
    )
    
    return output[0], output[1]  # (output, attention_weights)

def test_attention():
    print("\n=== Testing Attention Implementations ===")
    
    # Test parameters
    batch_size = 2
    seq_length = 32
    embed_dim = 64
    num_heads = 2
    head_dim = embed_dim // num_heads
    
    print(f"Parameters: batch={batch_size}, seq_len={seq_length}, "
          f"embed_dim={embed_dim}, heads={num_heads}, head_dim={head_dim}")
    
    # Create input tensors
    query = torch.randn(batch_size, seq_length, embed_dim, requires_grad=True)
    key = torch.randn(batch_size, seq_length, embed_dim, requires_grad=True)
    value = torch.randn(batch_size, seq_length, embed_dim, requires_grad=True)
    
    # Create attention mask (causal mask for testing)
    attn_mask = torch.triu(torch.ones(seq_length, seq_length) * float('-inf'), diagonal=1)
    
    print("\nInput shapes:")
    print(f"Query: {query.shape}")
    print(f"Key: {key.shape}")
    print(f"Value: {value.shape}")
    print(f"Mask: {attn_mask.shape}")
    
    try:
        print("\n--- Testing RoPE Attention ---")
        rope_output, rope_weights = custom_extension.rope_attention(
            query=query,
            key=key,
            value=value,
            num_heads=num_heads,
            attn_mask=attn_mask,
            dropout_p=0.0,
            need_weights=True,
            average_attn_weights=True,
            rope_base=10000.0,
            training=False
        )
        
        print("✓ RoPE forward pass successful")
        print(f"Output shape: {rope_output.shape}")
        print(f"Attention weights shape: {rope_weights.shape}")
        
        # Test backward pass
        grad_output = torch.randn_like(rope_output)
        rope_output.backward(grad_output)
        print("✓ RoPE backward pass successful")
        
        print("\n--- Testing Standard Attention ---")
        # Reset gradients
        query.grad = None
        key.grad = None
        value.grad = None
        
        # Forward pass with standard attention
        std_output, std_weights = custom_extension.full_attention(
            query=query,
            key=key,
            value=value,
            num_heads=num_heads,
            attn_mask=attn_mask,
            dropout_p=0.0,
            need_weights=True,
            average_attn_weights=True,
            training=False
        )
        
        print("✓ Standard attention forward pass successful")
        print(f"Output shape: {std_output.shape}")
        print(f"Attention weights shape: {std_weights.shape}")
        
        # Test backward pass
        std_output.backward(grad_output)
        print("✓ Standard attention backward pass successful")
        
        print("\n--- Testing MLM Attention ---")
        # Reset gradients
        query.grad = None
        key.grad = None
        value.grad = None
        
        # Forward pass with MLM attention
        mlm_output, mlm_weights = mlm_attention(
            query=query,
            key=key,
            value=value,
            num_heads=num_heads,
            attn_mask=attn_mask,
            dropout_p=0.0
        )
        
        print("✓ MLM attention forward pass successful")
        print(f"Output shape: {mlm_output.shape}")
        print(f"Attention weights shape: {mlm_weights.shape}")
        
        # Test backward pass
        mlm_output.backward(grad_output)
        print("✓ MLM attention backward pass successful")
        
        # Compare outputs
        print("\n--- Comparing Implementations ---")
        rope_std_diff = (rope_output - std_output).abs().max().item()
        rope_mlm_diff = (rope_output - mlm_output).abs().max().item()
        std_mlm_diff = (std_output - mlm_output).abs().max().item()
        
        print(f"Max differences:")
        print(f"- RoPE vs Standard: {rope_std_diff:.6f}")
        print(f"- RoPE vs MLM: {rope_mlm_diff:.6f}")
        print(f"- Standard vs MLM: {std_mlm_diff:.6f}")
        
        # Benchmark
        print("\n--- Benchmarking ---")
        rope_time = benchmark_fn(
            lambda: custom_extension.rope_attention(
                query, key, value, num_heads, attn_mask, 0.0, True, True, 10000.0, False
            )[0]
        )
        
        std_time = benchmark_fn(
            lambda: custom_extension.full_attention(
                query, key, value, num_heads, attn_mask, 0.0, True, True, False
            )[0]
        )
        
        mlm_time = benchmark_fn(
            lambda: mlm_attention(
                query, key, value, num_heads, attn_mask, 0.0
            )[0]
        )
        
        print(f"RoPE attention: {rope_time*1000:.3f} ms")
        print(f"Standard attention: {std_time*1000:.3f} ms")
        print(f"MLM attention: {mlm_time*1000:.3f} ms")
        print(f"Speed ratios:")
        print(f"- RoPE vs Standard: {std_time/rope_time:.2f}x")
        print(f"- RoPE vs MLM: {mlm_time/rope_time:.2f}x")
        print(f"- Standard vs MLM: {mlm_time/std_time:.2f}x")
        
    except Exception as e:
        print(f"\nError in attention test: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    print("Running CPU benchmarks\n")
    torch.manual_seed(42)
    
    try:
        test_matmul()
        test_attention()
    except Exception as e:
        print(f"Error in tests: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
