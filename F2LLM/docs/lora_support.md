# LoRA Support in F2LLM

## Overview

Low-Rank Adaptation (LoRA) is a parameter-efficient fine-tuning technique that significantly reduces the number of trainable parameters while maintaining model performance. F2LLM provides built-in support for LoRA, allowing users to fine-tune large language models efficiently without requiring full model updates.

## Key Benefits

- **Memory Efficiency**: Dramatically reduces memory requirements during training
- **Computational Efficiency**: Faster training with fewer parameters to update
- **Storage Efficiency**: Smaller adapter files compared to full model checkpoints
- **Modularity**: Easy to switch between different LoRA adapters for various tasks

## Configuration

LoRA can be enabled by setting the appropriate parameters in your configuration file or through command line arguments.

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_lora` | bool | `false` | Enable or disable LoRA |
| `lora_r` | int | `8` | The rank of the LoRA decomposition |
| `lora_alpha` | int | `16` | Scaling factor for LoRA |
| `lora_dropout` | float | `0.05` | Dropout rate applied to LoRA layers |
| `lora_target_modules` | str | `"all-linear"` | Target modules to apply LoRA to |

### Target Modules

The `lora_target_modules` parameter specifies which layers to apply LoRA to:

- **"all-linear"** (default): Applies LoRA to all linear projection layers including:
  - `q_proj`: Query projections
  - `v_proj`: Value projections
  - `k_proj`: Key projections
  - `o_proj`: Output projections
  - `gate_proj`: Gate projections (in feed-forward networks)
  - `up_proj`: Up projections (in feed-forward networks)
  - `down_proj`: Down projections (in feed-forward networks)
  - `lm_head`: Language model head

- **Custom list**: Comma-separated module names (e.g., `"q_proj,v_proj"`)

## Example Configuration

```json
{
    "model_path": "models/qwen3-0.6b",
    "experiment_id": "f2llm_lora_example",
    "output_dir": "output",
    "tb_dir": "tb_logs",
    "cache_dir": "cache",
    "train_data_path": "data_tokenized_qwen",
    "train_batch_size": 4,
    "max_seq_length": 1024,
    "learning_rate": 1e-4,
    "min_lr": 1e-6,
    "weight_decay": 1e-2,
    "warmup_steps": 100,
    "num_hard_neg": 7,
    "train_steps": 1000,
    "train_epochs": 3,
    "log_interval": 20,
    "checkpointing_steps": 100,
    "validation_steps": 100,
    "use_lora": true,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "lora_target_modules": "all-linear"
}
```

## Implementation Details

### Model Initialization

When `use_lora` is set to `true`, the model automatically applies LoRA during initialization in the `F2LLM.__init__()` method:

1. The base model is loaded from the specified `model_path`
2. LoRA configuration is created with the provided parameters
3. The PEFT (Parameter-Efficient FineTuning) library applies the LoRA adapters

### Parameter Efficiency

With LoRA enabled, only a fraction of the model's parameters are trainable:

- **Full model parameters**: All model weights
- **Trainable parameters**: Only LoRA adapter weights and biases
- **Memory savings**: Often 90%+ reduction in trainable parameters

## Usage Examples

### Training with LoRA

1. Create a configuration file with LoRA enabled
2. Run the training script:

```bash
python run.py --config config_lora_example.json
```

### Loading Models with LoRA Adapters

Use the `lora_utils.py` module to load models with previously trained adapters:

```python
from lora_utils import load_model_with_lora

model, tokenizer = load_model_with_lora(
    base_model_path="path/to/base/model",
    lora_adapter_path="path/to/lora/adapter"
)
```

### Merging LoRA Weights

To permanently merge LoRA weights with the base model:

```python
from lora_utils import merge_lora_weights

merged_model = merge_lora_weights(model, save_path="path/to/merged/model")
```

## Utilities

### lora_utils.py

This module provides several utility functions for LoRA operations:

- `load_model_with_lora()`: Load a base model with optional LoRA adapter
- `merge_lora_weights()`: Merge LoRA weights with the base model
- `get_lora_model_info()`: Get information about a LoRA model configuration
- `count_parameters()`: Count model parameters (trainable vs total)

## Best Practices

1. **Start with default parameters**: Use r=8, alpha=16, dropout=0.05 as a starting point
2. **Adjust r value**: Higher r values (16, 32) may improve performance but increase memory
3. **Tune alpha**: Alpha/r ratio often around 2 is effective (e.g., r=8, alpha=16)
4. **Monitor parameter count**: Check the trainable vs total parameter ratio during initialization
5. **Use appropriate target modules**: "all-linear" covers most important layers, but task-specific modules might be more efficient

## Troubleshooting

### Common Issues

- **PEFT library not found**: Install with `pip install peft`
- **Memory issues**: Reduce LoRA rank (`lora_r`) to further decrease memory usage
- **Performance degradation**: Try increasing `lora_r` or `lora_alpha` values

### Performance Considerations

- Lower ranks (r=4, 8) use less memory but may underperform
- Higher ranks (r=32, 64) approach full fine-tuning performance but use more memory
- The alpha/ratio is often kept around 2 for optimal performance