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

In this repo we provide a streamlined and efficient script for training embedding models. To reproduce the training of F2LLMs, please:

- Setup environment following `requirements.txt`. We note that transformers>=4.51.0 is required for training Qwen3 models.
- Download data and backbone models from Hugging Face (we use Qwen3 models).
- Run `tokenize_data_qwen.py` to tokenize the downloaded data
- Modify model path, data path, and other arguments in `configs/config.json`.
- Start training with `accelerate launch --config_file configs/accelerate_config.yaml run.py --config configs/config.json`.

Note: we recommend setting `num_processes` to 1 in `configs/accelerate_config.yaml` and launch the training code once to generate cache for training data before starting the actual training.

For multi-node training, run on the main node:

```
accelerate launch --config_file configs/accelerate_config.yaml --num_machines N_NODE --num_processes N_PROCESSES --machine_rank 0 --main_process_ip MASTER_IP --main_process_port MASTER_PORT run.py --config configs/config.json
```

where N_NODE is the number of machines; N_PROCESSES is N_NODE\*8; MASTER_IP is the IP address of your master node, and MASTER_PORT is a port available on your machine (e.g. 6379).

On worker nodes, also run the above commmand but modify `machine_rank` accordingly.

### Train with LoRA

For efficient fine-tuning with reduced computational costs, we support **LoRA (Low-Rank Adaptation)** via PEFT (Parameter-Efficient Fine-Tuning). LoRA allows you to adapt base models with minimal parameter updates, making it ideal for resource-constrained environments.

#### LoRA Configuration

Add the following parameters to `configs/config.json` to enable LoRA training:

```json
{
  "use_lora": true,
  "lora_r": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
}
```

#### LoRA Parameters Explanation

- `use_lora` (bool): Enable LoRA fine-tuning. Default: `false`
- `lora_r` (int): LoRA rank (lower values = more efficient, typically 8-32). Default: `16`
- `lora_alpha` (int): LoRA scaling factor. Typically set to 2× `lora_r`. Default: `32`
- `lora_dropout` (float): Dropout probability for LoRA layers. Default: `0.05`
- `lora_target_modules` (list): Transformer modules to apply LoRA to. Default targets query, key, value, output projections and feed-forward gates.

#### LoRA Training Example

```bash
# Start LoRA training with the same command
accelerate launch --config_file configs/accelerate_config.yaml run.py --config configs/config.json
```

#### LoRA Training Benefits

- **Parameter Efficiency**: Only ~1-5% of original model parameters are trainable
- **Reduced Memory**: Significantly lower GPU memory requirements
- **Faster Training**: Quicker convergence due to fewer parameters
- **Portable Adapters**: Save only LoRA weights (~10-100MB) instead of full models
- **Composability**: Combine multiple LoRA adapters for different tasks

#### Loading LoRA Fine-tuned Models

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

# Load the base model and LoRA adapters
model = AutoPeftModelForCausalLM.from_pretrained("path/to/lora/checkpoint")
tokenizer = AutoTokenizer.from_pretrained("path/to/lora/checkpoint")

# For inference, convert to single model file (optional)
model = model.merge_and_unload()
```

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
