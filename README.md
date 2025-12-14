## CodeFuse Embeddings

<p align="center">
    <img src="https://modelscope.cn/api/v1/models/codefuse-ai/CodeFuse-QWen-14B/repo?Revision=master&FilePath=LOGO.jpg&View=true" width="800"/>
<p>

Embedding-related repos from CodeFuse, including:

- [CGE](./CGE/README.md)
- [D2LLM](https://github.com/codefuse-ai/D2LLM)
- [F2LLM](./F2LLM/README.md)

### Encoder-Only Model Support

You can now fine-tune encoder-only (BERT-style) models in F2LLM for embedding tasks. Configure via `model_arch` and `pooling` in the F2LLM config:

- **`model_arch`**: `encoder` for encoder-only models; defaults to `decoder`.
- **`pooling`**: `cls` (recommended for encoders), `mean`, or `last_token`.

Example:

```
{
    "model_path": "bert-base-uncased",
    "experiment_id": "bert-enc-embeds",
    "output_dir": "output",
    "tb_dir": "output/tb",
    "cache_dir": "cache",
    "train_data_path": "F2LLM/training_data/data_tokenized",
    "train_batch_size": 16,
    "max_seq_length": 512,
    "learning_rate": 1e-4,
    "model_arch": "encoder",
    "pooling": "cls"
}
```

Notes:
- Encoder defaults to `[CLS]` pooling; `mean` averages non-pad tokens.
- Decoder uses `last_token` pooling (existing behavior).
- Hugging Face tokenizers will handle special tokens for encoders automatically.

**Star History**

[![Star History Chart](https://api.star-history.com/svg?repos=codefuse-ai/CodeFuse-Embeddings&type=date&legend=top-left)](https://www.star-history.com/#codefuse-ai/CodeFuse-Embeddings&type=date&legend=top-left)
