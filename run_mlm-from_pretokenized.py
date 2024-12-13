import transformers
from custom_bert import CustomBertForMLM
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    is_torch_xla_available,
    set_seed,
)
from dataclasses import dataclass, field
from typing import Optional

# ... (keep existing code until model creation)

@dataclass
class ModelArguments:
    rope_base: float = field(
        default=10000.0,
        metadata={"help": "Base value for RoPE position embeddings"}
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
        },
    )

if model_args.model_name_or_path:
    model = CustomBertForMLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        rope_base=model_args.rope_base,
    )
else:
    logger.info("Training new model from scratch")
    model = CustomBertForMLM(config)
 