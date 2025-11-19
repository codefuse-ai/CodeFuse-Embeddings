# Ray分布式训练配置文件说明

## 1. 配置文件概述

Ray分布式训练使用多种配置文件来管理训练参数、集群设置和资源分配。主要配置文件包括：

1. `configs/ray_config.yaml` - 主配置文件，定义集群、训练和检查点配置
2. `configs/ray_cpu_test_config.json` - CPU测试配置文件
3. `configs/ray_mac_test_config.json` - Mac测试配置文件

## 2. YAML配置文件说明

文件路径：`configs/ray_config.yaml`

### 2.1 集群配置 (cluster)

```yaml
cluster:
  mode: "auto"  # 集群模式：auto(自动连接) 或 manual(手动指定)
  address: "ray://head-node-ip:10001"  # 仅在manual模式下使用，指定集群地址
  num_workers: 4  # 工作节点数量
  num_gpus_per_worker: 2  # 每个工作节点的GPU数量
  resources_per_worker:   # 每个工作节点的资源分配
    CPU: 2
    GPU: 2
```

参数说明：
- `mode`: 集群连接模式
  - `auto`: 自动连接到集群
  - `manual`: 手动指定集群地址
- `address`: 集群地址，仅在manual模式下使用
- `num_workers`: 工作节点数量，决定分布式训练的并行度
- `num_gpus_per_worker`: 每个工作节点使用的GPU数量
- `resources_per_worker`: 每个工作节点的资源分配，包括CPU和GPU

### 2.2 训练配置 (training)

```yaml
training:
  use_gpu: true           # 是否使用GPU进行训练
  batch_size_per_worker: 16  # 每个工作节点的批次大小
  num_epochs: 2           # 训练轮数
```

参数说明：
- `use_gpu`: 是否使用GPU进行训练
- `batch_size_per_worker`: 每个工作节点处理的批次大小
- `num_epochs`: 训练轮数

### 2.3 检查点配置 (checkpoint)

```yaml
checkpoint:
  frequency: 5000         # 检查点保存频率（步数）
  keep_checkpoints_num: 3 # 保留的检查点数量
```

参数说明：
- `frequency`: 检查点保存频率，以训练步数为单位
- `keep_checkpoints_num`: 保留的检查点数量，用于节省存储空间

## 3. JSON配置文件说明

### 3.1 CPU测试配置文件

文件路径：`configs/ray_cpu_test_config.json`

```json
{
  "model_path": "/Users/zhaojie/Project/CodeFuse-Embeddings/F2LLM/models/bert-base-uncased",
  "experiment_id": "ray_cpu_test",
  "train_data_path": "/Users/zhaojie/Project/CodeFuse-Embeddings/F2LLM/data_tokenized_bert",
  "output_dir": "/Users/zhaojie/Project/CodeFuse-Embeddings/F2LLM/output/ray_cpu_test",
  "tb_dir": "/Users/zhaojie/Project/CodeFuse-Embeddings/F2LLM/output/tb/ray_cpu_test",
  "cache_dir": "/Users/zhaojie/Project/CodeFuse-Embeddings/F2LLM/cache/ray_cpu_test",
  "train_batch_size": 2,
  "checkpointing_steps": 10,
  "validation_steps": 10,
  "max_seq_length": 64,
  "learning_rate": 2e-5,
  "min_lr": 1e-7,
  "weight_decay": 0.01,
  "warmup_steps": 5,
  "train_epochs": 1,
  "log_interval": 5,
  "num_hard_neg": 1,
  "train_steps": -1,
  "local_files_only": true
}
```

参数说明：
- `model_path`: 模型文件路径
- `experiment_id`: 实验ID，用于区分不同实验
- `train_data_path`: 训练数据路径
- `output_dir`: 模型输出目录
- `tb_dir`: TensorBoard日志目录
- `cache_dir`: 缓存目录
- `train_batch_size`: 训练批次大小
- `checkpointing_steps`: 检查点保存步数间隔
- `validation_steps`: 验证步数间隔
- `max_seq_length`: 最大序列长度
- `learning_rate`: 学习率
- `min_lr`: 最小学习率
- `weight_decay`: 权重衰减
- `warmup_steps`: 学习率预热步数
- `train_epochs`: 训练轮数
- `log_interval`: 日志记录间隔
- `num_hard_neg`: 硬负样本数量
- `train_steps`: 训练步数，-1表示自动计算
- `local_files_only`: 是否仅从本地文件加载模型，true表示只使用本地文件，false表示允许从HuggingFace下载

### 3.2 Mac测试配置文件

文件路径：`configs/ray_mac_test_config.json`

```json
{
  "model_path": "/Users/zhaojie/Project/CodeFuse-Embeddings/F2LLM/models/bert-base-uncased",
  "experiment_id": "ray_mac_test",
  "train_data_path": "/Users/zhaojie/Project/CodeFuse-Embeddings/F2LLM/data_tokenized_bert",
  "output_dir": "/Users/zhaojie/Project/CodeFuse-Embeddings/F2LLM/output/ray_mac_test",
  "tb_dir": "/Users/zhaojie/Project/CodeFuse-Embeddings/F2LLM/output/tb/ray_mac_test",
  "cache_dir": "/Users/zhaojie/Project/CodeFuse-Embeddings/F2LLM/cache/ray_mac_test",
  "train_batch_size": 2,
  "checkpointing_steps": 50,
  "validation_steps": 50,
  "max_seq_length": 64,
  "learning_rate": 2e-5,
  "min_lr": 1e-7,
  "weight_decay": 0.01,
  "warmup_steps": 10,
  "train_epochs": 3,
  "log_interval": 10,
  "num_hard_neg": 1,
  "train_steps": -1,
  "cluster": {
    "mode": "auto",
    "address": "ray://head-node-ip:10001",
    "num_workers": 4,
    "num_gpus_per_worker": 2
  }
}
```

参数说明：
- 与CPU测试配置文件相同的基础参数
- `cluster`: 集群配置参数，包含模式、地址、工作节点数量和GPU数量，用于支持集群训练模式

## 4. 配置使用建议

### 4.1 CPU训练配置

对于仅使用CPU的训练环境，建议：
1. 设置`use_gpu`为false
2. 适当减小`train_batch_size`以适应内存限制
3. 减少`num_workers`以降低资源消耗
4. 使用较小的`max_seq_length`以提高训练速度

### 4.2 GPU训练配置

对于使用GPU的训练环境，建议：
1. 设置`use_gpu`为true
2. 根据GPU内存调整`train_batch_size`
3. 合理设置`num_workers`和`num_gpus_per_worker`以充分利用硬件资源
4. 使用较大的`max_seq_length`以提高模型性能

### 4.3 集群训练配置

对于多节点集群训练，建议：
1. 根据集群规模设置`num_workers`
2. 根据节点硬件配置设置`num_gpus_per_worker`
3. 合理分配`resources_per_worker`中的CPU和GPU资源
4. 根据网络环境选择合适的集群连接模式

### 4.4 检查点恢复训练配置

对于需要检查点恢复训练的场景，建议：
1. 合理设置`checkpointing_steps`参数以平衡存储空间和恢复粒度
2. 保留足够的检查点数量以应对不同的恢复需求
3. 在中断后恢复训练时，确保使用与原始训练相同的配置文件
4. 检查点路径应指向包含完整模型、优化器和调度器状态的目录
5. 确保检查点目录包含以下文件：
   - `model.pth`：模型状态字典
   - `optimizer.pth`：优化器状态字典
   - `lr_scheduler.pth`：学习率调度器状态字典
   - `args.pkl`：训练参数
6. 定期清理旧的检查点以节省存储空间
