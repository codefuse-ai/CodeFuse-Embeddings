# Encoder-Only 模型使用指南

本文档详细介绍了如何在CodeFuse-Embeddings项目中使用Encoder-Only模型（如BERT）进行训练和推理。

## 目录

1. [模型架构支持](#模型架构支持)
2. [数据准备](#数据准备)
3. [配置文件说明](#配置文件说明)
4. [训练步骤](#训练步骤)
5. [模型特点](#模型特点)
6. [常见问题](#常见问题)

## 模型架构支持

当前支持的Encoder-Only模型架构包括：

- BERT系列（bert-base-uncased, bert-base-cased等）
- RoBERTa系列
- ELECTRA系列
- DeBERTa系列
- ALBERT系列
- DistilBERT系列

模型会自动检测架构类型并应用相应的处理逻辑。

## 数据准备

### 数据格式

训练数据需要以Parquet格式存储，每个文件包含以下列：

| 列名 | 类型 | 描述 |
|------|------|------|
| query | string | 查询文本 |
| passage | string | 正样本文本 |
| negative_N | string | 负样本文本（N为序号，如negative_1, negative_2等）|
| query_input_ids | list of int | 查询文本的tokenized ID序列 |
| passage_input_ids | list of int | 正样本文本的tokenized ID序列 |
| negative_N_input_ids | list of int | 负样本文本的tokenized ID序列 |

### 示例数据结构

```python
{
    "query": "What is artificial intelligence?",
    "passage": "Artificial intelligence is intelligence demonstrated by machines...",
    "negative_1": "The weather is nice today.",
    "query_input_ids": [101, 2054, 2003, 7976, 4454, 1029, 102],
    "passage_input_ids": [101, 7976, 4454, 2003, 4454, 7645, 2011, 6681...],
    "negative_1_input_ids": [101, 1996, 4633, 2003, 3835, 2651, 1012, 102]
}
```

### 数据预处理

1. 准备原始文本数据（query, passage, negatives）
2. 使用tokenizer对文本进行编码
3. 保存为Parquet格式文件

示例预处理脚本可参考 `tokenize_data_bert.py`：

```bash
python tokenize_data_bert.py --input_file training_data/sample.json --output_file data_tokenized_bert/sample_data.parquet --model_name models/bert-base-uncased
```

### 数据目录结构

```
F2LLM/
├── data_tokenized_bert/
│   ├── dataset1.parquet
│   ├── dataset2.parquet
│   └── sample_data.parquet
```

## 配置文件说明

### 主配置文件 (config_bert.json)

```json
{
  "model_path": "models/bert-base-uncased",
  "model_type": "encoder",
  "experiment_id": "bert-base+lr.2e-5+bs.32x32+context.512+3epochs",
  "train_data_path": "data_tokenized_bert",
  "output_dir": "output",
  "tb_dir": "output/tb",
  "cache_dir": "cache",
  "train_batch_size": 32,
  "checkpointing_steps": 5000,
  "validation_steps": 5000,
  "max_seq_length": 512,
  "learning_rate": 2e-5,
  "min_lr": 1e-7,
  "weight_decay": 0.01,
  "warmup_steps": 500,
  "train_epochs": 3,
  "log_interval": 100,
  "num_hard_neg": 7
}
```

配置参数说明：

| 参数 | 默认值 | 描述 |
|------|--------|------|
| model_path | "models/bert-base-uncased" | 预训练模型路径 |
| model_type | "encoder" | 模型类型，必须设置为"encoder" |
| experiment_id | "bert-base+..." | 实验标识符 |
| train_data_path | "data_tokenized_bert" | 训练数据目录 |
| output_dir | "output" | 模型输出目录 |
| tb_dir | "output/tb" | TensorBoard日志目录 |
| cache_dir | "cache" | 缓存目录 |
| train_batch_size | 32 | 训练批次大小 |
| checkpointing_steps | 5000 | 检查点保存步数 |
| validation_steps | 5000 | 验证步数 |
| max_seq_length | 512 | 最大序列长度 |
| learning_rate | 2e-5 | 学习率 |
| min_lr | 1e-7 | 最小学习率 |
| weight_decay | 0.01 | 权重衰减 |
| warmup_steps | 500 | 学习率预热步数 |
| train_epochs | 3 | 训练轮数 |
| log_interval | 100 | 日志记录间隔 |
| num_hard_neg | 7 | 硬负样本数量 |

### Accelerate配置文件 (accelerate_config.yaml)

```yaml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: NO
downcast_bf16: "no"
machine_rank: 0
main_training_function: main
mixed_precision: "no"
num_machines: 1
num_processes: 1
use_cpu: false
```

## 训练步骤

### 1. 环境准备

确保已安装所有依赖：

```bash
pip install -r requirements.txt
```

### 2. 数据准备

准备训练数据并将其放置在 `data_tokenized_bert/` 目录中。

### 3. 模型准备

确保预训练模型已下载并放置在 `models/` 目录中。

### 4. 启动训练

使用以下命令启动训练：

```bash
# 激活虚拟环境
source .venv/bin/activate

# 启动训练
accelerate launch --config_file configs/accelerate_config.yaml run.py --config configs/config_bert.json
```

### 5. 监控训练

使用TensorBoard监控训练过程：

```bash
tensorboard --logdir=output/tb
```

### 6. 模型输出

训练完成后，模型将保存在以下目录结构中：

```
output/
├── bert-base+lr.2e-5+bs.32x32+context.512+3epochs/
│   ├── epoch_1/
│   ├── epoch_2/
│   ├── epoch_3/
│   └── args.json
└── tb/
    └── bert-base+lr.2e-5+bs.32x32+context.512+3epochs/
```

## 模型特点

### 1. 改进的池化策略

Encoder-Only模型使用masked-mean pooling而非传统的[CLS] token pooling：

```python
def _masked_mean_pool(self, last_hidden, attention_mask):
    # last_hidden: [N, seq_len, dim], attention_mask: [N, seq_len]
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)  # [N, seq_len, 1]
    summed = (last_hidden * mask).sum(dim=1)  # [N, dim]
    counts = mask.sum(dim=1).clamp(min=1e-9)  # [N, 1]
    return summed / counts  # [N, dim]
```

优势：
- 利用序列中所有有效token的信息
- 避免[CLS] token可能存在的信息不足问题
- 对长序列有更好的表示能力

### 2. 自动模型类型检测

模型会自动检测架构类型并应用相应的处理逻辑：

```python
encoder_architectures = [
    'Bert', 'Roberta', 'Electra', 'Deberta', 'Albert', 'DistilBert'
]

if hasattr(config, 'architectures'):
    archs = config.architectures
    for arch in archs:
        if any(encoder_arch in arch for encoder_arch in encoder_architectures):
            return 'encoder'
```

### 3. 特殊token处理

对于Encoder模型，确保[CLS]和[SEP] token可用：

```python
# For encoder models, ensure [CLS] token is available
if args.model_type == 'encoder' or (args.model_type == 'auto' and 'bert' in args.model_path.lower()):
    if tokenizer.cls_token is None:
        tokenizer.add_special_tokens({'cls_token': '[CLS]'})
    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({'sep_token': '[SEP]'})
```

## 常见问题

### 1. 训练过程中出现梯度错误

**问题**：`element 0 of tensors does not require grad and does not have a grad_fn`

**解决方案**：确保所有损失张量都支持梯度计算：

```python
# 对于非检索数据集，创建支持梯度的张量
loss = torch.tensor(0.0, device=outputs['query_passage_features'].device, requires_grad=True)
```

### 2. 模型类型检测失败

**问题**：模型被错误识别为decoder类型

**解决方案**：在配置文件中明确指定模型类型：

```json
{
  "model_type": "encoder"
}
```

### 3. 内存不足

**问题**：训练时出现内存不足错误

**解决方案**：
1. 减小batch size
2. 减小序列长度
3. 启用混合精度训练

```json
{
  "train_batch_size": 16,
  "max_seq_length": 256
}
```

### 4. 数据加载错误

**问题**：数据加载时出现维度不匹配

**解决方案**：检查数据格式是否符合要求，特别是input_ids的维度。

### 5. TensorBoard无数据

**问题**：TensorBoard显示"No dashboards are active"

**解决方案**：
1. 检查日志目录路径是否正确
2. 确保训练已运行足够长时间以写入日志
3. 使用正确的日志目录启动TensorBoard

```bash
tensorboard --logdir=output/tb/bert-base+lr.2e-5+bs.32x32+context.512+3epochs/
```

## 性能优化建议

1. **批处理大小**：根据GPU内存调整batch size
2. **序列长度**：根据数据特点调整max_seq_length
3. **学习率**：BERT类模型通常使用2e-5到5e-5的学习率
4. **负样本数量**：根据数据集特点调整num_hard_neg参数

## 贡献

如有任何问题或建议，请提交issue或pull request。
