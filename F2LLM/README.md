## F2LLM

F2LLMs (Foundation-to-Feature Large Language Models) are foundation models directly finetuned on 6 million high-quality query-document pairs, striking a strong balance between model size, training cost, and embedding performance:

<p align="center">
    <img src="imgs/overview.png" width="700"/>
<p>

On the MTEB leaderboard, F2LLM-4B ranks 2nd among models of ~4B size, and 7th overall, while F2LLM-1.7B ranks 1st among models of 1B-2B size.

<p align="center">
    <img src="imgs/mteb_leaderboard.png" width="700"/>
<p>

F2LLMs are fully open. Model checkpoints are available at:

- [F2LLM 0.6B](https://huggingface.co/codefuse-ai/F2LLM-0.6B)
- [F2LLM 1.7B](https://huggingface.co/codefuse-ai/F2LLM-1.7B)
- [F2LLM 4B](https://huggingface.co/codefuse-ai/F2LLM-4B)

Training data is available at [F2LLM data](https://huggingface.co/datasets/codefuse-ai/F2LLM).

### Train

In this repo we provide a streamlined and efficient script for training embedding models. The framework now supports **13 popular base models** across 6 different families (Qwen3, LLaMA 2/3, Mistral, Phi, Code-LLaMA, and Gemma).

#### Quick Start with Different Models

```python
from model import F2LLM

# Load any of 13 supported models
model = F2LLM('meta-llama/Llama-2-7b', model_id='llama-2-7b')
model = F2LLM('mistralai/Mistral-7B-v0.1', model_id='mistral-7b')
model = F2LLM('microsoft/Phi-3-mini-4k-instruct', model_id='phi-3-mini')
model = F2LLM('meta-llama/CodeLlama-7b', model_id='code-llama-7b')
```

#### Training Steps

To train embedding models with any supported base model:

- Setup environment following `requirements.txt`. We note that transformers>=4.51.0 is required.
- Download data and backbone models from Hugging Face.
- Run `tokenize_data_generic.py` to tokenize data for any model (replaces `tokenize_data_qwen.py`):
  ```bash
  python tokenize_data_generic.py \
    --model_path meta-llama/Llama-2-7b \
    --model_id llama-2-7b \
    --root_dir training_data \
    --output_dir data_tokenized \
    --hf_token "$HF_TOKEN"   # optional; required for gated models
  ```
  If you encounter a 401/GatedRepoError, login with `huggingface-cli login` or set `export HF_TOKEN=hf_xxx`. Alternatively, try an open model such as `mistralai/Mistral-7B-v0.1` or `microsoft/Phi-3-mini-4k-instruct`.
- Choose a model configuration from `configs/` (e.g., `llama2-7b.json`, `mistral-7b.json`, `phi3-mini.json`)
- Start training with `accelerate launch --config_file configs/accelerate_config.yaml run.py --config configs/llama2-7b.json`.

Note: we recommend setting `num_processes` to 1 in `configs/accelerate_config.yaml` and launch the training code once to generate cache for training data before starting the actual training.

For multi-node training, run on the main node:

```
accelerate launch --config_file configs/accelerate_config.yaml --num_machines N_NODE --num_processes N_PROCESSES --machine_rank 0 --main_process_ip MASTER_IP --main_process_port MASTER_PORT run.py --config configs/config.json
```

where N_NODE is the number of machines; N_PROCESSES is N_NODE\*8; MASTER_IP is the IP address of your master node, and MASTER_PORT is a port available on your machine (e.g. 6379).

On worker nodes, also run the above commmand but modify `machine_rank` accordingly.

### Citation

If you use the F2LLM models, data, or code, please cite the following technical report.

```
@article{2025F2LLM,
  title={F2LLM Technical Report: Matching SOTA Embedding Performance with 6 Million Open-Source Data},
  author={Ziyin Zhang and Zihan Liao and Hang Yu and Peng Di and Rui Wang},
  journal      = {CoRR},
  volume       = {abs/2510.02294},
  year         = {2025},
  url          = {https://doi.org/10.48550/arXiv.2510.02294},
  doi          = {10.48550/ARXIV.2510.02294},
  eprinttype    = {arXiv},
  eprint       = {2510.02294}
}
```
