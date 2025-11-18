# Ray分布式训练使用说明

## 1. 环境准备

### 1.1 依赖安装

确保已安装Ray依赖：

```bash
# 使用pip安装
pip install 'ray[train]' torch transformers datasets

# 或使用uv安装（推荐）
uv pip install 'ray[train]' torch transformers datasets
```

### 1.2 环境变量设置

在运行训练之前，建议设置以下环境变量以减少警告信息：

```bash
export TOKENIZERS_PARALLELISM=false
export TORCH_DISTRIBUTED_ELASTIC_LOG_LEVEL=ERROR
```

## 2. 单机训练部署方式

### 2.1 CPU训练

#### 使用uv运行（推荐）：

```bash
PYTHONPATH=. uv run python scripts/ray_train.py --config configs/ray_cpu_test_config.json --num-workers 1
```

#### 使用标准Python运行：

```bash
PYTHONPATH=. python scripts/ray_train.py --config configs/ray_cpu_test_config.json --num-workers 1
```

### 2.2 GPU训练

#### 使用uv运行（推荐）：

```bash
PYTHONPATH=. uv run python scripts/ray_train.py --config configs/ray_mac_test_config.json --num-workers 1 --use-gpu
```

#### 使用标准Python运行：

```bash
PYTHONPATH=. python scripts/ray_train.py --config configs/ray_mac_test_config.json --num-workers 1 --use-gpu
```

## 3. 集群训练部署方式

### 3.1 启动Ray集群

在主节点上启动Ray集群：

```bash
# 启动主节点
ray start --head --port=6379

# 获取连接信息
ray nodes
```

在工作节点上连接到主节点：

```bash
# 在工作节点上运行
ray start --address=<head-node-ip>:6379
```

### 3.2 集群训练命令

#### 使用uv运行（推荐）：

```bash
# 自动连接模式
PYTHONPATH=. uv run python scripts/ray_train.py --config configs/ray_mac_test_config.json --num-workers 4

# 手动指定集群地址
PYTHONPATH=. uv run python scripts/ray_train.py --config configs/ray_mac_test_config.json --num-workers 4 --cluster-mode manual --cluster-address ray://<head-node-ip>:10001
```

#### 使用标准Python运行：

```bash
# 自动连接模式
PYTHONPATH=. python scripts/ray_train.py --config configs/ray_mac_test_config.json --num-workers 4

# 手动指定集群地址
PYTHONPATH=. python scripts/ray_train.py --config configs/ray_mac_test_config.json --num-workers 4 --cluster-mode manual --cluster-address ray://<head-node-ip>:10001
```

## 4. CPU与GPU训练部署方式

### 4.1 CPU训练

CPU训练适用于没有GPU或GPU资源不足的环境。

#### 配置要求：
1. 设置`--num-workers`参数控制并行度
2. 适当减小`train_batch_size`以适应内存限制
3. 使用较小的`max_seq_length`以提高训练速度

#### 运行命令：

```bash
# uv方式（推荐）
PYTHONPATH=. uv run python scripts/ray_train.py --config configs/ray_cpu_test_config.json --num-workers 1

# 标准Python方式
PYTHONPATH=. python scripts/ray_train.py --config configs/ray_cpu_test_config.json --num-workers 1
```

### 4.2 GPU训练

GPU训练可显著提高训练速度，适用于有GPU资源的环境。

#### 配置要求：
1. 添加`--use-gpu`参数启用GPU训练
2. 根据GPU内存调整`train_batch_size`
3. 合理设置`--num-workers`和`--num-gpus-per-worker`参数

#### 运行命令：

```bash
# uv方式（推荐）
PYTHONPATH=. uv run python scripts/ray_train.py --config configs/ray_mac_test_config.json --num-workers 1 --use-gpu --num-gpus-per-worker 1

# 标准Python方式
PYTHONPATH=. python scripts/ray_train.py --config configs/ray_mac_test_config.json --num-workers 1 --use-gpu --num-gpus-per-worker 1
```

## 5. 多GPU训练部署方式

### 5.1 单机多GPU训练

在单台机器上使用多个GPU进行训练。

#### 配置要求：
1. 确保机器有多个可用GPU
2. 设置`--use-gpu`参数启用GPU训练
3. 设置`--num-gpus-per-worker`参数指定每个工作节点使用的GPU数量

#### 运行命令：

```bash
# uv方式（推荐）
PYTHONPATH=. uv run python scripts/ray_train.py --config configs/ray_mac_test_config.json --num-workers 2 --use-gpu --num-gpus-per-worker 2

# 标准Python方式
PYTHONPATH=. python scripts/ray_train.py --config configs/ray_mac_test_config.json --num-workers 2 --use-gpu --num-gpus-per-worker 2
```

### 5.2 多节点多GPU训练

在多个节点上使用多个GPU进行训练。

#### 配置要求：
1. 确保所有节点都有可用GPU
2. 正确配置Ray集群
3. 合理设置工作节点数和每个节点的GPU数

#### 运行命令：

```bash
# uv方式（推荐）
PYTHONPATH=. uv run python scripts/ray_train.py --config configs/ray_mac_test_config.json --num-workers 4 --use-gpu --num-gpus-per-worker 2

# 标准Python方式
PYTHONPATH=. python scripts/ray_train.py --config configs/ray_mac_test_config.json --num-workers 4 --use-gpu --num-gpus-per-worker 2
```

## 6. 输出文件说明

训练输出保存在`output/`目录中，包括：

1. **模型检查点**：包含模型权重和配置文件
2. **训练日志**：包含训练过程中的日志信息
3. **TensorBoard日志**：可用于可视化训练过程

### 6.1 检查点文件结构

```
output/
└── <experiment_id>/
    └── <experiment_id>/
        ├── epoch_1/
        │   ├── config.json
        │   ├── model.safetensors
        │   ├── tokenizer.json
        │   ├── tokenizer_config.json
        │   └── vocab.txt
        ├── epoch_2/
        └── step_<n>/
```

### 6.2 查看训练结果

```bash
# 查看生成的检查点文件
ls -la output/ray_cpu_test/ray_cpu_test/

# 查看特定epoch的检查点
ls -la output/ray_cpu_test/ray_cpu_test/epoch_1/
```

## 7. 常见问题与解决方案

### 7.1 模块导入问题

**问题**：出现`ModuleNotFoundError`错误
**解决方案**：确保设置`PYTHONPATH=.`环境变量

```bash
PYTHONPATH=. uv run python scripts/ray_train.py ...
```

### 7.2 文件大小限制问题

**问题**：出现工作目录大小超过512MB限制的错误
```
RuntimeEnvSetupError: Failed to set up runtime environment.
Failed to upload working_dir to the Ray cluster: Package size exceeds the maximum size of 512.00MiB.
```

**解决方案**：系统已自动配置运行时环境排除大型文件

如果仍然遇到此问题，可以手动指定排除规则：
```bash
# 在代码中配置运行时环境
runtime_env = {
    "excludes": [
        "output/**",
        "models/**",
        "cache/**",
        "data_tokenized_bert/**",
        "*.safetensors",
        "*.bin",
        "*.h5",
        "*.msgpack"
    ]
}
ray.init(runtime_env=runtime_env)
```

### 7.3 资源配置问题

**问题**：出现资源配置错误
**解决方案**：根据是否使用GPU动态配置资源

### 7.4 GPU相关问题

**问题**：GPU训练时出现错误
**解决方案**：
1. 确保安装了正确的CUDA驱动
2. 检查GPU是否被其他进程占用
3. 适当调整批次大小以适应GPU内存

### 7.5 恢复训练问题

**问题**：恢复训练时检查点路径不存在或损坏
**解决方案**：
1. 确保指定的检查点路径存在且包含完整的检查点文件
2. 检查检查点目录是否包含`model.pth`、`optimizer.pth`等文件
3. 确保恢复训练时使用的配置文件与原始训练保持一致

## 8. 检查点恢复训练

### 8.1 恢复训练的作用

检查点恢复训练允许在训练中断后从最近保存的检查点继续训练，而不是从头开始。这在以下场景中特别有用：

1. 训练过程中意外中断（如系统崩溃、资源不足等）
2. 需要延长训练时间
3. 分阶段训练模型

### 8.2 检查点文件存储位置

Ray分布式训练使用两种不同的检查点机制，它们的存储位置也不同：

1. **Ray检查点**：
   - 存储位置：Ray默认结果目录（如`/Users/<username>/ray_results/`）
   - 包含文件：
     - `model.pth`：模型状态字典
     - `optimizer.pth`：优化器状态字典
     - `lr_scheduler.pth`：学习率调度器状态字典
     - `args.pkl`：训练参数
   - 用途：主要用于训练状态的保存和恢复

2. **模型检查点**：
   - 存储位置：项目目录的`output/`文件夹下
   - 包含文件：
     - `model.safetensors`：模型权重文件
     - `config.json`：模型配置文件
     - `tokenizer.json`：分词器文件
     - 其他模型相关文件
   - 用途：主要用于模型的持久化存储和部署

### 8.3 恢复训练命令

使用`--resume-from-checkpoint`和`--resume-checkpoint-path`参数来恢复训练：

```bash
# 从指定step检查点恢复训练
PYTHONPATH=. uv run python scripts/ray_train.py --config configs/ray_cpu_test_config.json --num-workers 1 --resume-from-checkpoint --resume-checkpoint-path output/ray_cpu_test/ray_cpu_test/step_10

# 从指定epoch检查点恢复训练
PYTHONPATH=. uv run python scripts/ray_train.py --config configs/ray_cpu_test_config.json --num-workers 1 --resume-from-checkpoint --resume-checkpoint-path output/ray_cpu_test/ray_cpu_test/epoch_1

# GPU训练从step检查点恢复
PYTHONPATH=. uv run python scripts/ray_train.py --config configs/ray_mac_test_config.json --num-workers 1 --use-gpu --resume-from-checkpoint --resume-checkpoint-path output/ray_mac_test/ray_mac_test/step_50

# GPU训练从epoch检查点恢复
PYTHONPATH=. uv run python scripts/ray_train.py --config configs/ray_mac_test_config.json --num-workers 1 --use-gpu --resume-from-checkpoint --resume-checkpoint-path output/ray_mac_test/ray_mac_test/epoch_2
```

### 8.4 查找可用检查点

在恢复训练前，首先需要找到可用的检查点：

```bash
# 查看所有检查点
ls -la output/ray_cpu_test/ray_cpu_test/

# 查看按步骤保存的检查点
ls -la output/ray_cpu_test/ray_cpu_test/step_*/

# 查看按轮次保存的检查点
ls -la output/ray_cpu_test/ray_cpu_test/epoch_*/

# 查看Ray检查点
ls -la ~/ray_results/
```

**检查点类型说明**：
- `step_*`目录：按训练步数保存的检查点，包含模型在特定训练步数的状态
- `epoch_*`目录：按训练轮次保存的检查点，包含模型在完成特定训练轮次后的状态

选择哪种类型的检查点取决于您的恢复需求：
- 如果需要更精细的恢复点，可以选择step检查点
- 如果希望按完整的训练轮次恢复，可以选择epoch检查点

### 8.5 恢复训练示例

1. 从step检查点恢复训练示例：
假设训练在第50步中断，可以从该检查点恢复训练：

```bash
# 查看可用检查点
ls -la output/ray_cpu_test/ray_cpu_test/step_*/

# 从第50步检查点恢复训练
PYTHONPATH=. uv run python scripts/ray_train.py --config configs/ray_cpu_test_config.json --num-workers 1 --resume-from-checkpoint --resume-checkpoint-path output/ray_cpu_test/ray_cpu_test/step_50
```

2. 从epoch检查点恢复训练示例：
假设训练在第2轮次中断，可以从该检查点恢复训练：

```bash
# 查看可用检查点
ls -la output/ray_cpu_test/ray_cpu_test/epoch_*/

# 从第2轮次检查点恢复训练
PYTHONPATH=. uv run python scripts/ray_train.py --config configs/ray_cpu_test_config.json --num-workers 1 --resume-from-checkpoint --resume-checkpoint-path output/ray_cpu_test/ray_cpu_test/epoch_2
```

### 8.6 注意事项

1. 恢复训练时使用的配置文件应与原始训练保持一致
2. 恢复训练会继续使用原始训练的学习率调度器状态
3. 检查点路径应指向包含模型、优化器和调度器状态的目录
4. 如果检查点路径不存在或损坏，训练将从头开始并记录警告信息
5. Ray检查点文件存储在用户主目录下的`ray_results`文件夹中，而不是项目目录中

## 9. 性能优化建议

### 9.1 资源配置优化

1. 根据硬件资源合理设置工作节点数
2. 根据内存大小调整批次大小
3. 使用多个GPU以提高训练速度

### 9.2 训练参数优化

1. 根据数据集大小调整训练轮数
2. 合理设置学习率和预热步数
3. 根据需要调整检查点保存频率

### 9.3 集群配置优化

1. 确保网络连接稳定
2. 合理分配各节点资源
3. 监控集群状态及时发现性能瓶颈