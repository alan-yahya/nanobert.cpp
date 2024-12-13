import torch
from transformers import AutoModel, AutoTokenizer
import custom_extension

def load_nanobert():
    # Load model and tokenizer
    model = AutoModel.from_pretrained("alan-yahya/NanoBERT-V2")
    tokenizer = AutoTokenizer.from_pretrained("alan-yahya/NanoBERT-V2")
    return model, tokenizer

def custom_forward_pass(model, input_ids, attention_mask=None):
    # Get the model's config
    config = model.config
    
    # Get embeddings
    embeddings = model.embeddings(input_ids)
    
    # Process through each layer with our custom attention
    hidden_states = embeddings
    
    for layer in model.encoder.layer:
        # Get query, key projections
        query = layer.attention.self.query(hidden_states)
        key = layer.attention.self.key(hidden_states)
        value = layer.attention.self.value(hidden_states)
        
        # Reshape for attention
        batch_size = query.size(0)
        query = query.view(batch_size, -1, config.num_attention_heads, 
                         config.hidden_size // config.num_attention_heads).transpose(1, 2)
        key = key.view(batch_size, -1, config.num_attention_heads,
                      config.hidden_size // config.num_attention_heads).transpose(1, 2)
        
        # Use our custom attention score computation
        scale_factor = 1.0 / torch.sqrt(torch.tensor(config.hidden_size // config.num_attention_heads))
        attention_scores = custom_extension.custom_attention_scores(query, key, scale_factor)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask.unsqueeze(1).unsqueeze(2)
        
        # Apply softmax
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        
        # Continue with the rest of the layer operations...
        # (This is simplified - you'd need to complete the attention computation
        # and feed forward network steps)
    
    return hidden_states

def main():
    # Load model and tokenizer
    model, tokenizer = load_nanobert()
    
    # Prepare input
    text = "Testing custom attention with NanoBERT"
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Run custom forward pass
    with torch.no_grad():
        outputs = custom_forward_pass(model, inputs["input_ids"], 
                                    attention_mask=inputs["attention_mask"])
    
    print("Output shape:", outputs.shape)
    print("First token representation:", outputs[0][0][:5])  # Print first 5 values

if __name__ == "__main__":
    main() 