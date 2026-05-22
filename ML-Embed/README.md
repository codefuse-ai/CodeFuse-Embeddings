## ML-Embed

<div align="center">

[![ICML 2026](https://img.shields.io/badge/ICML-2026-blue.svg)](https://icml.cc/Conferences/2026)
[![arXiv](https://img.shields.io/badge/arXiv-2605.15081-b31b1b.svg)](https://arxiv.org/abs/2605.15081)
[![Hugging Face](https://img.shields.io/badge/🤗%20HuggingFace-Models%20%26%20Data-yellow)](https://huggingface.co/collections/codefuse-ai/codefuse-embeddings)
[![GitHub](https://img.shields.io/badge/GitHub-Code-green)](https://github.com/codefuse-ai/CodeFuse-Embeddings)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**Ziyin Zhang, Zihan Liao, Hang Yu, Peng Di, Rui Wang**

_Shanghai Jiao Tong University & Ant Group_

</div>

## 🌍 Overview

The development of high-quality text embeddings has been drifting toward an **exclusionary future**, defined by three critical barriers:

1. 🔒 **Prohibitive computational costs** — training and inference of massive decoder-based models
2. 🗺️ **Narrow linguistic focus** — neglecting the vast majority of the world's languages
3. 🚫 **Lack of transparency** — closed-source or open-weight-only models that stifle research

**ML-Embed** dismantles these barriers. It is a suite of inclusive, efficient, and fully open text embedding models built upon our novel **3-Dimensional Matryoshka Learning (3D-ML)** framework, trained on a massively multilingual dataset spanning **282 natural languages** and **40+ programming languages**.

---

## ✨ Key Contributions

### 🪆 3-Dimensional Matryoshka Learning (3D-ML)

A unified framework providing **end-to-end efficiency** across the entire model lifecycle:

| Dimension          | Technique                                | Benefit                                                           |
| ------------------ | ---------------------------------------- | ----------------------------------------------------------------- |
| **Parameters**     | Matryoshka Embedding Learning (MEL)      | Efficient training & inference via low-rank factorized embeddings |
| **Depth**          | Matryoshka Layer Learning (MLL)          | Flexible inference-time depth without retraining                  |
| **Representation** | Matryoshka Representation Learning (MRL) | Variable-size embeddings for efficient storage                    |

### 🌐 Massively Multilingual Dataset

- **60 million** training samples aggregated from **157** public sources
- **282** natural languages (ISO-639-3) and **40+** programming languages
- Driven by real-world data availability, not benchmark optimization
- Substantially more linguistically diverse than comparable open datasets

### 🔓 Full Transparency

We release **everything**: model weights, training data, and training code - a fully reproducible blueprint for globally equitable AI.

## 🤗 Model & Data

- The entire suite of baseline models (including additional 80M & 14B models not described in the paper) and training data are released under the name [F2LLM-v2](https://huggingface.co/collections/codefuse-ai/f2llm).
- The 0.6B model trained with 3D-ML is available at [codefuse-ai/ML-Embed-0.6B](https://huggingface.co/codefuse-ai/ML-Embed-0.6B).

## 🛠️ Quick Start

To train embedding models with 3D-ML, please:

- Download data and backbone models from Hugging Face (we use Qwen3 models).
- Run `tokenize_data_qwen.py` to tokenize the downloaded data, and then cancatenate all corpus files into a single `corpus.parquet` file.
- Modify model path, data path, and other arguments in `configs/config.json`.
- Start training with `accelerate launch --config_file configs/accelerate_config.yaml run.py --config configs/config.json`.

Note: we recommend setting `num_processes` to 1 in `configs/accelerate_config.yaml` and launch the training code once to generate cache for training data before starting the actual training.

For multi-node training, run on the main node:

```
accelerate launch --config_file configs/accelerate_config.yaml --num_machines N_NODE --num_processes N_PROCESSES --machine_rank 0 --main_process_ip MASTER_IP --main_process_port MASTER_PORT run.py --config configs/config.json
```

where N_NODE is the number of machines; N_PROCESSES is N_NODE\*8; MASTER_IP is the IP address of your master node, and MASTER_PORT is a port available on your machine (e.g. 6379).

On worker nodes, also run the above commmand but modify `machine_rank` accordingly.

## Citation

Please cite the following paper for 3D-ML training method and data:

```
@article{2026ML-Embed,
  author       = {Ziyin Zhang and
                  Zihan Liao and
                  Hang Yu and
                  Peng Di and
                  Rui Wang},
  title        = {ML-Embed: Inclusive and Efficient Embeddings for a Multilingual World},
  journal      = {CoRR},
  volume       = {abs/2605.15081},
  year         = {2026},
  url          = {https://doi.org/10.48550/arXiv.2605.15081},
  doi          = {10.48550/ARXIV.2605.15081},
  eprinttype   = {arXiv},
  eprint       = {2605.15081}
}
```

If you use or refer to the baseline models (which are released and submitted to MTEB leaderboard under the name F2LLM-v2), you are also welcome to cite the following technical report:

```
@article{2026F2LLM-v2,
  title={F2LLM-v2: Inclusive, Performant, and Efficient Embeddings for a Multilingual World},
  author={Ziyin Zhang and Zihan Liao and Hang Yu and Peng Di and Rui Wang},
  journal      = {CoRR},
  volume       = {abs/2603.19223},
  year         = {2026},
  url          = {https://doi.org/10.48550/arXiv.2603.19223},
  doi          = {10.48550/ARXIV.2603.19223},
  eprinttype    = {arXiv},
  eprint       = {2603.19223}
}
```
