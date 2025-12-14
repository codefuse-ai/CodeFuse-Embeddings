## CodeFuse Embeddings

<p align="center">
    <img src="https://modelscope.cn/api/v1/models/codefuse-ai/CodeFuse-QWen-14B/repo?Revision=master&FilePath=LOGO.jpg&View=true" width="800"/>
### Gradient Accumulation

To train with larger effective batch sizes on limited GPU memory, we added gradient accumulation.

- New config key: `gradient_accumulation_steps` (default: 1)
- Effective global batch size: `train_batch_size * gradient_accumulation_steps * num_processes`
- `train_steps` represent optimization steps (after accumulation). When not set, they are computed as `total_micro_batches * train_epochs // gradient_accumulation_steps`.

Usage:

1. Set in your config JSON:
    - `"gradient_accumulation_steps": 8`
2. Run training as usual with `F2LLM/run.py`.

Quick Tests (no real data required):

```bash
python F2LLM/test_gradient_accumulation.py
python F2LLM/smoke_test_accumulation.py
```

The first verifies optimizer step counts; the second runs a small synthetic pipeline on CPU with accumulation.

<p>

Embedding-related repos from CodeFuse, including:

- [CGE](./CGE/README.md)
- [D2LLM](https://github.com/codefuse-ai/D2LLM)
- [F2LLM](./F2LLM/README.md)

**Star History**

[![Star History Chart](https://api.star-history.com/svg?repos=codefuse-ai/CodeFuse-Embeddings&type=date&legend=top-left)](https://www.star-history.com/#codefuse-ai/CodeFuse-Embeddings&type=date&legend=top-left)
