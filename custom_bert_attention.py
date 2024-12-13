import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertSelfAttention
import custom_extension

class CustomBertAttention(BertSelfAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor] | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        
        batch_size, seq_length = hidden_states.size()[:2]
        
        # Do the linear projections
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Use custom RoPE attention
        attention_output, attention_weights = custom_extension.rope_attention(
            query=query_layer,
            key=key_layer,
            value=value_layer,
            num_heads=self.num_attention_heads,
            attn_mask=attention_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            need_weights=output_attentions,
            average_attn_weights=True,
            training=self.training
        )
        
        outputs = (attention_output,)
        if output_attentions:
            outputs += (attention_weights,)
            
        return outputs 