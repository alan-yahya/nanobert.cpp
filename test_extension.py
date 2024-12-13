import torch
import custom_extension
import time
from typing import Callable
from torch.nn import functional as F
import matplotlib.pyplot as plt

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
    num_heads = 4
    head_dim = 64
    embed_dim = head_dim * num_heads
    
    # Create test tensors with requires_grad=True
    query = torch.randn(batch_size, seq_length, embed_dim, requires_grad=True)
    key = torch.randn(batch_size, seq_length, embed_dim, requires_grad=True)
    value = torch.randn(batch_size, seq_length, embed_dim, requires_grad=True)
    
    # Create gradient tensor for backward pass
    grad_output = torch.randn(batch_size, seq_length, embed_dim)
    
    try:
        print("\n--- Testing RoPE Attention ---")
        rope_output, rope_weights = custom_extension.rope_attention(
            query=query,
            key=key,
            value=value,
            num_heads=num_heads,
            attn_mask=torch.zeros(seq_length, seq_length),  # bidirectional attention
            dropout_p=0.0,
            need_weights=True,
            average_attn_weights=False,  # Keep separate head weights
            rope_base=10000.0,
            training=True  # Set to True for training mode
        )
        
        print("✓ RoPE attention forward pass successful")
        print(f"Output shape: {rope_output.shape}")
        print(f"Attention weights shape: {rope_weights.shape}")
        
        # Visualize RoPE attention patterns for each head
        plt.figure(figsize=(15, 5))
        for head in range(min(3, num_heads)):  # Show first 3 heads
            plt.subplot(1, 3, head + 1)
            plt.imshow(rope_weights[0, head].detach().numpy(), cmap='viridis')
            plt.title(f"RoPE Head {head}")
            plt.colorbar()
        plt.tight_layout()
        plt.savefig('rope_attention_heads.png')
        plt.close()
        
        # Test backward pass
        rope_output.backward(grad_output)
        print("✓ RoPE attention backward pass successful")
        
        print("\n--- Testing Standard Attention ---")
        std_output, std_weights = custom_extension.full_attention(
            query=query.detach().requires_grad_(),  # Create new gradient history
            key=key.detach().requires_grad_(),
            value=value.detach().requires_grad_(),
            num_heads=num_heads,
            attn_mask=torch.zeros(seq_length, seq_length),
            dropout_p=0.0,
            need_weights=True,
            average_attn_weights=False,
            training=True  # Set to True for training mode
        )
        
        print("✓ Standard attention forward pass successful")
        print(f"Output shape: {std_output.shape}")
        print(f"Attention weights shape: {std_weights.shape}")
        
        # Visualize standard attention patterns for each head
        plt.figure(figsize=(15, 5))
        for head in range(min(3, num_heads)):  # Show first 3 heads
            plt.subplot(1, 3, head + 1)
            plt.imshow(std_weights[0, head].detach().numpy(), cmap='viridis')
            plt.title(f"Standard Head {head}")
            plt.colorbar()
        plt.tight_layout()
        plt.savefig('standard_attention_heads.png')
        plt.close()
        
        # Test backward pass
        std_output.backward(grad_output)
        print("✓ Standard attention backward pass successful")
        
        # Compare outputs and attention patterns
        print("\n--- Comparing Implementations ---")
        output_diff = (rope_output - std_output).abs().max().item()
        print(f"Max output difference: {output_diff:.6f}")
        
        # Compare attention patterns per head
        for head in range(num_heads):
            head_diff = (rope_weights[0, head] - std_weights[0, head]).abs().max().item()
            print(f"Head {head} max attention difference: {head_diff:.6f}")
        
        # Benchmark
        print("\n--- Benchmarking ---")
        def benchmark_fn(fn):
            start = time.perf_counter()
            for _ in range(100):
                fn()
            return (time.perf_counter() - start) / 100
        
        rope_time = benchmark_fn(
            lambda: custom_extension.rope_attention(
                query, key, value, num_heads, torch.zeros(seq_length, seq_length),
                0.0, True, False, 10000.0, False
            )[0]
        )
        
        std_time = benchmark_fn(
            lambda: custom_extension.full_attention(
                query, key, value, num_heads, torch.zeros(seq_length, seq_length),
                0.0, True, False, False
            )[0]
        )
        
        print(f"RoPE attention: {rope_time*1000:.3f} ms")
        print(f"Standard attention: {std_time*1000:.3f} ms")
        print(f"Speed ratio: {std_time/rope_time:.2f}x")
        
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
