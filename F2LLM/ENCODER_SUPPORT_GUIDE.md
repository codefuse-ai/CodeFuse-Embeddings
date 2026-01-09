# F2LLM Encoder-Only Model Support: Complete Guide

## Overview

F2LLM now supports both decoder-only and encoder-only model architectures for training embedding models. This enhancement provides architectural flexibility, allowing users to leverage the bidirectional attention of encoder models (BERT, RoBERTa) for improved representation learning in tasks like code retrieval and similarity detection.

## Implementation Details

### Model Architecture Detection

The `model.py` file includes automatic architecture detection:

```python
# Determine if model is encoder-only (e.g., BERT, RoBERTa) or decoder-only (e.g., GPT, Qwen)
self.is_encoder_only = any(arch in config.architectures for arch in [
    'BertModel', 'RobertaModel', 'DebertaModel', 
    'ElectraModel', 'AlbertModel', 'DistilBertModel'
])
```

### Embedding Extraction Strategies

**Encoder-only models** (e.g., BERT):
- Use [CLS] token (index 0) as sequence representation
- Bidirectional attention captures context from both directions
- Well-suited for classification and retrieval tasks

**Decoder-only models** (e.g., Qwen, GPT):
- Use last non-padded token as sequence representation
- Causal attention appropriate for generation tasks
- Can handle longer context windows

### Tokenization Differences

The `tokenize_data_general.py` script handles both architectures:
- **Encoder models**: Automatically adds [CLS] and [SEP] tokens, handles special token types
- **Decoder models**: Manually adds EOS tokens, handles masking appropriately

## Supported Architectures

### Encoder-Only Models
- BERT (`BertModel`)
- RoBERTa (`RobertaModel`)
- DeBERTa (`DebertaModel`)
- ELECTRA (`ElectraModel`)
- ALBERT (`AlbertModel`)
- DistilBERT (`DistilBertModel`)

### Decoder-Only Models (existing support)
- Qwen models
- GPT models
- LLaMA/Mistral models

## Usage Guide

### Prerequisites

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

Make sure you have `transformers>=4.51.0` for full compatibility.

### Step 1: Prepare Your Data

Prepare your training data in the required format. The data should be in JSON format with the following structure:

```json
[
  {
    "query": "What is the capital of France?",
    "pos": ["Paris"],
    "neg": ["London", "Berlin", "Madrid"]
  }
]
```

Where:
- `query`: The input query text
- `pos`: Array of positive (relevant) documents
- `neg`: Array of negative (irrelevant) documents

### Step 2: Tokenize Your Data

For encoder models, use the general tokenization script:

```bash
# Tokenize for BERT-based models
python tokenize_data_general.py \
    --model_path bert-base-uncased \
    --data_dir path/to/your/training_data \
    --output_dir data_tokenized_bert \
    --max_seq_length 512 \
    --num_processes 8
```

For decoder models, you can continue using the existing approach:

```bash
# Tokenize for decoder models (e.g., Qwen)
python tokenize_data_general.py \
    --model_path Qwen/Qwen2-7B \
    --data_dir path/to/your/training_data \
    --output_dir data_tokenized_qwen \
    --max_seq_length 1024 \
    --num_processes 8
```

### Step 3: Configure Training

#### For Encoder Models

Create or modify your configuration file for encoder models. Here's an example for BERT:

```json
{
  "model_path": "bert-base-uncased",
  "experiment_id": "bert-base-uncased+lr.2e-5+bs.16x32+context.512+2epochs",
  "train_data_path": "path/to/data_tokenized_bert",
  "output_dir": "output",
  "tb_dir": "output/tb",
  "cache_dir": "cache",
  "train_batch_size": 16,
  "checkpointing_steps": 5000,
  "validation_steps": 5000,
  "max_seq_length": 512,
  "learning_rate": 2e-5,
  "min_lr": 1e-7,
  "weight_decay": 0.01,
  "warmup_steps": 500,
  "train_epochs": 2,
  "log_interval": 100,
  "num_hard_neg": 7
}
```

Key parameters for encoder models:
- Use higher learning rates (typically 2e-5 to 5e-5)
- Max sequence length usually 512 for BERT-like models
- Model path should point to an encoder-only model

#### For Decoder Models

The existing configuration works unchanged for decoder models. Here's an example for Qwen:

```json
{
  "model_path": "Qwen/Qwen2-7B",
  "experiment_id": "qwen2-7b+lr.8e-6+bs.16x32+context.1024+2epochs",
  "train_data_path": "path/to/data_tokenized_qwen",
  "output_dir": "output",
  "tb_dir": "output/tb",
  "cache_dir": "cache",
  "train_batch_size": 16,
  "checkpointing_steps": 5000,
  "validation_steps": 5000,
  "max_seq_length": 1024,
  "learning_rate": 8e-6,
  "min_lr": 1e-7,
  "weight_decay": 0.01,
  "warmup_steps": 500,
  "train_epochs": 2,
  "log_interval": 100,
  "num_hard_neg": 7
}
```

### Step 4: Initialize Accelerate Configuration

First, generate the accelerate configuration file:

```bash
accelerate config
```

Or copy the example config:

```bash
cp configs/accelerate_config.yaml accelerate_config.yaml
```

### Step 5: Start Training

#### For Encoder Models

```bash
accelerate launch \
    --config_file accelerate_config.yaml \
    run.py \
    --config configs/config_bert.json
```

#### For Decoder Models

```bash
accelerate launch \
    --config_file accelerate_config.yaml \
    run.py \
    --config configs/config.json
```

### Configuration Differences

| Parameter | Encoder Models | Decoder Models |
|-----------|----------------|----------------|
| `learning_rate` | 2e-5 to 5e-5 | 1e-6 to 1e-5 |
| `max_seq_length` | 512 (typical) | 1024+ (typical) |
| `attn_implementation` | 'eager' | 'flash_attention_2' |

## Advantages of Encoder Models

Encoder-only models with bidirectional attention offer several advantages for embedding tasks:

1. **Better Context Understanding**: Each token sees both left and right context
2. **Superior Performance on Retrieval Tasks**: Excellent for semantic similarity
3. **Efficient Processing**: No causal mask needed during inference
4. **Established Pretraining**: Extensive pretraining on large corpora

## Best Practices

1. **Learning Rate**: Use higher learning rates (2e-5 to 5e-5) for encoder models
2. **Sequence Length**: Most encoder models have 512 token max length
3. **Task Suitability**: Encoder models excel at retrieval, classification, and similarity tasks
4. **Memory Management**: Encoder models may have different memory patterns than decoders

## Migration Guide

To switch from decoder-only to encoder-only models:
1. Change `model_path` to an encoder model
2. Update `max_seq_length` (typically 512 for encoders)
3. Use `tokenize_data_general.py` (handles both)
4. Increase learning rate appropriately
5. Update data path to encoder-tokenized data

## Example Use Cases

**Encoder Models (BERT, RoBERTa)**:
- Code search and retrieval
- Similarity detection
- Classification tasks
- Clustering applications

**Decoder Models (Qwen, GPT)**:
- Code completion
- Generation tasks
- Sequential modeling

## Files Created/Modified

- `ENCODER_SUPPORT_GUIDE.md` - guide for encoder models
- Updated `README.md` with encoder support details
- `tokenize_data_general.py` - Unified tokenization script
- Enhanced `model.py` with architecture detection and handling
- `test_encoder_support.py`  - Test scripts

Run tests with:
```bash
python test_encoder_support.py
```

For more detailed information, check out the main README and the specific documentation files in the repository.
