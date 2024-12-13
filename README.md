# C++ Extensions for NanoBERT

This project implements a custom attention mechanism with optimized C++ extensions for a BERT model from the HuggingFace hub trained using Pytorch (in this case, https://huggingface.co/alan-yahya/NanoBERT-V2).

For info on C++ extensions for Pytorch see https://pytorch.org/tutorials/advanced/cpp_extension.html
Primarily custom model architecture implementations, not inference/deployment implementations (for that see llama.cpp or unsloth.ai).

Integrating RoPE for BERT model run_mlm from Pytorch scripts.

## Features

- Optimized C++ attention implementation
- Custom matrix multiplication optimizations
- Integration with HuggingFace's BERT models
- Support for pre-tokenized datasets

## Project Structure

### Core Implementation
- `extension.cpp`
  - Custom attention mechanism implementation
  - Optimized matrix multiplication
  - CUDA-compatible computations

- `custom_bert_attention.py`
  - Custom attention layer integration
  - HuggingFace compatibility layer
  - C++ extension bindings

- `custom_bert.py`
  - Modified BERT model implementations
  - `CustomBertModel`: Base model
  - `CustomBertForMLM`: MLM-specific model

### Training
- `run_mlm-from_pretokenized.py`
  - Pre-tokenized dataset support
  - HuggingFace Trainer integration
  - MLM training configuration

### Build System
- `build.py`
  - C++ extension compilation
  - MSVC/ninja build configuration

- `pyproject.toml`
  - Build dependencies
  - Package configuration

- `setup.py`
  - Extension module setup
  - Package metadata

## Installation

1. Install build dependencies:
```bash
pip install -r requirements.txt
```

2. Build the C++ extension:
```bash
python build.py
```

3. Install the package:
```bash
pip install -e .
```

## Usage

```python
from custom_bert import CustomBertModel

# Initialize model
model = CustomBertModel.from_pretrained('bert-base-uncased')

# Forward pass with optimized attention
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask
)
```

## Training

To train using the custom attention mechanism:

```bash
python run_mlm-from_pretokenized.py \
    --model_name_or_path bert-base-uncased \
    --train_file path/to/train.txt \
    --validation_file path/to/val.txt \
    --do_train \
    --do_eval \
    --output_dir ./output
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
