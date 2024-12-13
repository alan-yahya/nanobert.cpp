from transformers.models.bert.modeling_bert import BertModel, BertForMaskedLM
from .custom_bert_attention import CustomBertAttention

class CustomBertModel(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        # Replace all attention layers with custom attention
        for layer in self.encoder.layer:
            layer.attention.self = CustomBertAttention(config)

class CustomBertForMLM(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.bert = CustomBertModel(config)
        # Initialize weights and apply final processing
        self.post_init() 