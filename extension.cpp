#include <torch/extension.h>
#include <vector>
#include <cmath>

// Custom tensor operations

// Custom attention score computation
torch::Tensor custom_attention_scores(torch::Tensor query, torch::Tensor key, float scale_factor) {
    // Check dimensions (batch_size, num_heads, seq_length, head_dim)
    TORCH_CHECK(query.dim() == 4 && key.dim() == 4, "Query and Key tensors must be 4D");
    TORCH_CHECK(query.size(3) == key.size(3), "Head dimensions must match");

    // Compute scaled dot product attention
    // (batch_size, num_heads, seq_length_q, head_dim) x (batch_size, num_heads, head_dim, seq_length_k)
    auto scores = torch::matmul(query, key.transpose(-2, -1));
    
    // Apply scaling
    scores = scores * scale_factor;
    
    return scores;
}

// Custom autograd function for full attention
class FullAttentionFunction : public torch::autograd::Function<FullAttentionFunction> {
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor query,
        torch::Tensor key,
        torch::Tensor value,
        int64_t num_heads,
        torch::Tensor attn_mask,
        float dropout_p,
        bool need_weights,
        bool average_attn_weights,
        bool training = false
    ) {
        const auto batch_size = query.size(0);
        const auto tgt_len = query.size(1);
        const auto src_len = key.size(1);
        const auto embed_dim = query.size(2);
        const auto head_dim = embed_dim / num_heads;
        
        auto scaling = float(1.0 / std::sqrt(head_dim));
        
        // Reshape query, key, value for multi-head attention
        auto query_reshaped = query.view({batch_size, tgt_len, num_heads, head_dim}).transpose(1, 2);
        auto key_reshaped = key.view({batch_size, src_len, num_heads, head_dim}).transpose(1, 2);
        auto value_reshaped = value.view({batch_size, src_len, num_heads, head_dim}).transpose(1, 2);

        // Calculate attention scores
        auto attn_weights = torch::matmul(query_reshaped, key_reshaped.transpose(-2, -1)) * scaling;

        // Apply attention mask if provided
        if (attn_mask.defined()) {
            if (attn_mask.dim() == 2) {
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0);
            } else if (attn_mask.dim() == 3) {
                attn_mask = attn_mask.unsqueeze(1);
            }
            attn_weights += attn_mask;
        }

        // Apply softmax
        auto attn_probs = torch::softmax(attn_weights, -1);
        
        // Store intermediate values for backward pass
        ctx->save_for_backward({query_reshaped, key_reshaped, value_reshaped, attn_probs});
        ctx->saved_data["scaling"] = scaling;
        ctx->saved_data["num_heads"] = num_heads;

        // Apply dropout
        if (dropout_p > 0.0 && training) {
            attn_probs = torch::dropout(attn_probs, dropout_p, true);
        }

        // Calculate attention output
        auto attn_output = torch::matmul(attn_probs, value_reshaped);
        
        // Reshape output back to original dimensions
        attn_output = attn_output.transpose(1, 2).contiguous();
        attn_output = attn_output.view({batch_size, tgt_len, embed_dim});
        
        // Handle attention weights for return
        if (need_weights && average_attn_weights) {
            attn_weights = attn_weights.mean(1);
        }

        return {attn_output, attn_weights};
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto query = saved[0];
        auto key = saved[1];
        auto value = saved[2];
        auto attn_probs = saved[3];
        
        auto scaling = ctx->saved_data["scaling"].toDouble();
        auto num_heads = ctx->saved_data["num_heads"].toInt();
        
        auto grad_output = grad_outputs[0];  // gradient of output
        
        // Reshape grad_output to match attention shape
        auto batch_size = grad_output.size(0);
        auto tgt_len = grad_output.size(1);
        auto embed_dim = grad_output.size(2);
        grad_output = grad_output.reshape({batch_size, tgt_len, num_heads, embed_dim / num_heads}).transpose(1, 2);
        
        // Compute gradients
        auto grad_value = torch::matmul(attn_probs.transpose(-2, -1), grad_output);
        auto grad_attn_probs = torch::matmul(grad_output, value.transpose(-2, -1));
        
        // Compute gradient of softmax
        auto grad_attn_weights = grad_attn_probs * attn_probs;
        grad_attn_weights -= attn_probs * grad_attn_weights.sum(-1, true);
        
        // Compute gradients for query and key
        auto grad_query = torch::matmul(grad_attn_weights, key) * scaling;
        auto grad_key = torch::matmul(grad_attn_weights.transpose(-2, -1), query) * scaling;
        
        // Reshape gradients back to original dimensions
        grad_query = grad_query.transpose(1, 2).contiguous().reshape({batch_size, tgt_len, embed_dim});
        grad_key = grad_key.transpose(1, 2).contiguous().reshape({batch_size, tgt_len, embed_dim});
        grad_value = grad_value.transpose(1, 2).contiguous().reshape({batch_size, tgt_len, embed_dim});
        
        // Return gradients for all inputs (in the same order as forward inputs)
        return {
            grad_query,                // grad_query
            grad_key,                  // grad_key
            grad_value,                // grad_value
            torch::Tensor(),          // grad_num_heads (integer, no gradient)
            torch::Tensor(),          // grad_attn_mask
            torch::Tensor(),          // grad_dropout_p (float, no gradient)
            torch::Tensor(),          // grad_need_weights (bool, no gradient)
            torch::Tensor(),          // grad_average_attn_weights (bool, no gradient)
            torch::Tensor(),          // grad_training (bool, no gradient)
            torch::Tensor()           // grad_rope_base (float, no gradient)
        };
    }
};

// Python-facing function that uses the autograd function
std::vector<torch::Tensor> full_attention(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    int64_t num_heads,
    torch::Tensor attn_mask,
    float dropout_p,
    bool need_weights,
    bool average_attn_weights,
    bool training = false
) {
    return FullAttentionFunction::apply(
        query, key, value, num_heads, attn_mask,
        dropout_p, need_weights, average_attn_weights, training
    );
}

// Matrix multiplication
torch::Tensor custom_matmul(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "Both tensors must be 2D");
    TORCH_CHECK(a.size(1) == b.size(0), "Incompatible matrix dimensions");
    return torch::mm(a, b);
}

// Helper function to create rotation matrices for RoPE
std::vector<torch::Tensor> create_rope_rotary_matrices(
    int64_t seq_length,
    int64_t dim,
    float base = 10000.0,
    torch::TensorOptions options = torch::TensorOptions()
) {
    auto inv_freq = 1.0 / torch::pow(base, 
        torch::arange(0, dim, 2, options).div(dim));
    
    auto t = torch::arange(seq_length, options);
    auto freqs = torch::outer(t, inv_freq);
    
    auto cos = torch::cos(freqs);  // [seq_length, dim/2]
    auto sin = torch::sin(freqs);  // [seq_length, dim/2]
    
    return {cos, sin};
}

// Apply rotary embeddings to a tensor
torch::Tensor apply_rotary_embeddings(
    torch::Tensor x,        // [batch_size, num_heads, seq_length, head_dim]
    torch::Tensor cos,      // [seq_length, head_dim/2]
    torch::Tensor sin       // [seq_length, head_dim/2]
) {
    auto shape = x.sizes();
    auto head_dim = shape[3];
    
    // Split last dimension into pairs for rotation
    auto x_reshape = x.reshape({-1, shape[2], head_dim}); // [batch_size * num_heads, seq_length, head_dim]
    auto x_split = x_reshape.chunk(2, -1);  // Split along head_dim
    auto x1 = x_split[0];  // [..., head_dim/2]
    auto x2 = x_split[1];  // [..., head_dim/2]
    
    // Apply rotation using complex multiplication
    auto out1 = x1 * cos.unsqueeze(0) - x2 * sin.unsqueeze(0);
    auto out2 = x1 * sin.unsqueeze(0) + x2 * cos.unsqueeze(0);
    
    // Concatenate and reshape back
    return torch::cat({out1, out2}, -1).view(shape);
}

// Custom autograd function for RoPE attention
class RoPEAttentionFunction : public torch::autograd::Function<RoPEAttentionFunction> {
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor query,           // [batch_size, tgt_len, embed_dim]
        torch::Tensor key,             // [batch_size, src_len, embed_dim]
        torch::Tensor value,           // [batch_size, src_len, embed_dim]
        int64_t num_heads,
        torch::Tensor attn_mask,
        float dropout_p,
        bool need_weights,
        bool average_attn_weights,
        float rope_base = 10000.0,
        bool training = false
    ) {
        const auto batch_size = query.size(0);
        const auto tgt_len = query.size(1);
        const auto src_len = key.size(1);
        const auto embed_dim = query.size(2);
        const auto head_dim = embed_dim / num_heads;
        
        auto scaling = float(1.0 / std::sqrt(head_dim));
        
        // Reshape for multi-head attention
        auto query_reshaped = query.reshape({batch_size, tgt_len, num_heads, head_dim}).transpose(1, 2);
        auto key_reshaped = key.reshape({batch_size, src_len, num_heads, head_dim}).transpose(1, 2);
        auto value_reshaped = value.reshape({batch_size, src_len, num_heads, head_dim}).transpose(1, 2);

        // Create and apply rotary position embeddings
        auto rotary_matrices = create_rope_rotary_matrices(
            std::max(tgt_len, src_len), 
            head_dim, 
            rope_base,
            query.options()
        );
        
        auto cos = rotary_matrices[0];
        auto sin = rotary_matrices[1];
        
        query_reshaped = apply_rotary_embeddings(query_reshaped, cos, sin);
        key_reshaped = apply_rotary_embeddings(key_reshaped, cos, sin);

        // Calculate attention scores
        auto attn_weights = torch::matmul(query_reshaped, key_reshaped.transpose(-2, -1)) * scaling;

        // Apply attention mask if provided
        if (attn_mask.defined()) {
            if (attn_mask.dim() == 2) {
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0);
            } else if (attn_mask.dim() == 3) {
                attn_mask = attn_mask.unsqueeze(1);
            }
            attn_weights += attn_mask;
        }

        // Apply softmax
        auto attn_probs = torch::softmax(attn_weights, -1);
        
        // Store for backward pass
        ctx->save_for_backward({query_reshaped, key_reshaped, value_reshaped, attn_probs, cos, sin});
        ctx->saved_data["scaling"] = scaling;
        ctx->saved_data["num_heads"] = num_heads;

        // Apply dropout during training
        if (dropout_p > 0.0 && training) {
            attn_probs = torch::dropout(attn_probs, dropout_p, true);
        }

        // Calculate attention output
        auto attn_output = torch::matmul(attn_probs, value_reshaped);
        
        // Reshape output back
        attn_output = attn_output.transpose(1, 2).contiguous();
        attn_output = attn_output.reshape({batch_size, tgt_len, embed_dim});
        
        if (need_weights && average_attn_weights) {
            attn_weights = attn_weights.mean(1);
        }

        return {attn_output, attn_weights};
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs
    ) {
        // Retrieve saved tensors and values
        auto saved = ctx->get_saved_variables();
        auto query = saved[0];
        auto key = saved[1];
        auto value = saved[2];
        auto attn_probs = saved[3];
        auto cos = saved[4];
        auto sin = saved[5];
        
        auto num_heads = ctx->saved_data["num_heads"].toInt();
        auto scaling = ctx->saved_data["scaling"].toDouble();
        
        auto grad_output = grad_outputs[0];
        
        // Compute gradients
        auto batch_size = grad_output.size(0);
        auto tgt_len = grad_output.size(1);
        auto embed_dim = grad_output.size(2);
        grad_output = grad_output.reshape({batch_size, tgt_len, num_heads, embed_dim / num_heads}).transpose(1, 2);
        
        // Compute gradients
        auto grad_value = torch::matmul(attn_probs.transpose(-2, -1), grad_output);
        auto grad_attn_probs = torch::matmul(grad_output, value.transpose(-2, -1));
        
        // Compute gradient of softmax
        auto grad_attn_weights = grad_attn_probs * attn_probs;
        grad_attn_weights -= attn_probs * grad_attn_weights.sum(-1, true);
        
        // Compute gradients for query and key with RoPE consideration
        auto grad_query = torch::matmul(grad_attn_weights, key) * scaling;
        auto grad_key = torch::matmul(grad_attn_weights.transpose(-2, -1), query) * scaling;
        
        // Apply inverse rotary embeddings for gradients
        grad_query = apply_rotary_embeddings(grad_query, cos, -sin);  // Note the negative sin for inverse
        grad_key = apply_rotary_embeddings(grad_key, cos, -sin);
        
        // Reshape gradients back
        grad_query = grad_query.transpose(1, 2).contiguous().reshape({batch_size, tgt_len, embed_dim});
        grad_key = grad_key.transpose(1, 2).contiguous().reshape({batch_size, tgt_len, embed_dim});
        grad_value = grad_value.transpose(1, 2).contiguous().reshape({batch_size, tgt_len, embed_dim});
        
        // Return gradients for all inputs in the same order as forward
        return {
            grad_query,                // grad_query
            grad_key,                  // grad_key
            grad_value,                // grad_value
            torch::Tensor(),          // grad_num_heads (integer, no gradient)
            torch::Tensor(),          // grad_attn_mask
            torch::Tensor(),          // grad_dropout_p (float, no gradient)
            torch::Tensor(),          // grad_need_weights (bool, no gradient)
            torch::Tensor(),          // grad_average_attn_weights (bool, no gradient)
            torch::Tensor(),          // grad_rope_base (float, no gradient)
            torch::Tensor()           // grad_training (bool, no gradient)
        };
    }
};

// Python-facing function
std::vector<torch::Tensor> rope_attention(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    int64_t num_heads,
    torch::Tensor attn_mask,
    float dropout_p,
    bool need_weights,
    bool average_attn_weights,
    float rope_base = 10000.0,
    bool training = false
) {
    return RoPEAttentionFunction::apply(
        query, key, value, num_heads, attn_mask,
        dropout_p, need_weights, average_attn_weights, rope_base, training
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_matmul", &custom_matmul, "Custom Matrix Multiplication");
    m.def("custom_attention_scores", &custom_attention_scores, "Custom Attention Score Computation");
    m.def("full_attention", &full_attention, "Full Multi-Head Attention Implementation",
          py::arg("query"),
          py::arg("key"),
          py::arg("value"),
          py::arg("num_heads"),
          py::arg("attn_mask") = torch::Tensor(),
          py::arg("dropout_p") = 0.0,
          py::arg("need_weights") = true,
          py::arg("average_attn_weights") = true,
          py::arg("training") = false);
    m.def("rope_attention", &rope_attention, "RoPE Multi-Head Attention",
          py::arg("query"),
          py::arg("key"),
          py::arg("value"),
          py::arg("num_heads"),
          py::arg("attn_mask") = torch::Tensor(),
          py::arg("dropout_p") = 0.0,
          py::arg("need_weights") = true,
          py::arg("average_attn_weights") = true,
          py::arg("rope_base") = 10000.0,
          py::arg("training") = false);
} 
