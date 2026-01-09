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
- Run `python tokenize_data_general.py --model_path <path_to_model>` to tokenize the downloaded data for both decoder and encoder models
- Modify model path, data path, and other arguments in `configs/config.json` (for decoder models) or `configs/config_bert.json` (for encoder models).
- Start training with `accelerate launch --config_file configs/accelerate_config.yaml run.py --config configs/config.json`.

Note: we recommend setting `num_processes` to 1 in `configs/accelerate_config.yaml` and launch the training code once to generate cache for training data before starting the actual training.

For multi-node training, run on the main node:

```
accelerate launch --config_file configs/accelerate_config.yaml --num_machines N_NODE --num_processes N_PROCESSES --machine_rank 0 --main_process_ip MASTER_IP --main_process_port MASTER_PORT run.py --config configs/config.json
```

where N_NODE is the number of machines; N_PROCESSES is N_NODE*8; MASTER_IP is the IP address of your master node, and MASTER_PORT is a port available on your machine (e.g. 6379).

On worker nodes, also run the above commmand but modify `machine_rank` accordingly.

### Support for Encoder-Only Models

Starting from this update, the framework now supports both decoder-only (e.g., Qwen, GPT) and encoder-only (e.g., BERT, RoBERTa) architectures:

- **Decoder-only models**: Use the last non-padded token as the sequence representation
- **Encoder-only models**: Use the [CLS] token (first token) as the sequence representation
- **Automatic detection**: The system automatically detects architecture type based on the model's configuration
- **Tokenization**: Different tokenization strategies for encoder vs. decoder models
- **Config files**: Separate example configs provided for both architectures

#### Quick Start with Encoder Models

To train with encoder models like BERT:

1. **Tokenize your data**:
   ```bash
   python tokenize_data_general.py \
       --model_path bert-base-uncased \
       --data_dir training_data \
       --output_dir data_tokenized_bert \
       --max_seq_length 512 \
       --num_processes 8
   ```

2. **Configure training** (use `configs/config_bert.json` as template):
   ```json
   {
     "model_path": "bert-base-uncased",
     "train_data_path": "data_tokenized_bert",
     "max_seq_length": 512,
     "learning_rate": 2e-5,
     "train_batch_size": 16
   }
   ```

3. **Start training**:
   ```bash
   accelerate launch --config_file configs/accelerate_config.yaml run.py --config configs/config_bert.json
   ```

For complete documentation on encoder model support, see [ENCODER_SUPPORT_GUIDE.md](ENCODER_SUPPORT_GUIDE.md).

#### Architecture-Specific Details

| Aspect | Encoder-Only | Decoder-Only |
|--------|--------------|--------------|
| Embedding Strategy | [CLS] token (first) | Last non-padded token |
| Tokenization | Auto special tokens | Manual EOS token |
| Attention | Bidirectional | Causal (unidirectional) |
| Typical Max Length | 512 tokens | Up to 8192+ tokens |
| Learning Rate | 2e-5 to 5e-5 | 1e-6 to 1e-5 |

**Supported Encoder Architectures**:
- BERT (`BertModel`)
- RoBERTa (`RobertaModel`)
- DeBERTa (`DebertaModel`)
- ELECTRA (`ElectraModel`)
- ALBERT (`AlbertModel`)
- DistilBERT (`DistilBertModel`)

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
