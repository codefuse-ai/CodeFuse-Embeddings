# 为F2LLM添加Encoder-only模型支持

## 需求概述
当前F2LLM项目仅支持Decoder-only LLM（如Qwen系列），需要扩展支持Encoder-only模型（BERT风格架构），以提供更多架构选择并利用双向注意力的优势。

## 实现方案

### 1. 修改模型架构适配层
- 创建新的模型类支持Encoder-only架构
- 修改现有的`F2LLM`类以支持两种架构类型
- 添加模型类型检测和适配逻辑

### 2. 修改特征提取逻辑
- 调整`forward`方法以适配Encoder-only模型的特征提取
- Encoder-only模型使用[CLS]标记或平均池化获取句向量
- 保持与现有Decoder-only模型输出接口一致

### 3. 修改配置和参数处理
- 在参数配置中添加模型类型标识
- 根据模型类型调整序列处理方式
- 支持不同的tokenizer处理方式

### 4. 修改数据处理逻辑
- 调整tokenization脚本以支持Encoder-only模型
- 添加对[CLS]标记的支持
- 保持数据格式兼容性

### 5. 验证和测试
- 验证Encoder-only模型训练和推理流程
- 对比Encoder-only和Decoder-only模型的性能
- 确保向后兼容性

## 实现步骤

### 步骤1：创建Encoder-only模型适配类
- 在`model.py`中添加新的模型类或扩展现有类
- 实现Encoder-only模型的特征提取逻辑

### 步骤2：修改特征提取接口
- 更新`forward`方法以支持两种架构
- 确保输出格式一致性

### 步骤3：添加模型类型检测
- 添加模型类型自动检测逻辑
- 根据模型类型选择合适的处理流程

### 步骤4：调整配置参数
- 在`arguments.py`中添加模型类型参数
- 修改配置文件以支持新参数

### 步骤5：修改数据处理脚本
- 更新`tokenize_data_qwen.py`以支持Encoder-only模型
- 添加对[CLS]标记的处理

### 步骤6：准备BERT模型文件
- 创建BERT模型目录：
  ```bash
  mkdir -p F2LLM/models/bert-base-uncased
  ```
- 从Hugging Face镜像站点下载BERT模型文件：
  - config.json
  - tokenizer.json
  - tokenizer_config.json
  - vocab.txt
- 将下载的文件放置到`F2LLM/models/bert-base-uncased`目录中

### 步骤7：验证实现
- 使用BERT类模型进行训练测试
- 验证模型输出和性能
- 确保现有Decoder-only模型不受影响

## 使用BERT类模型进行训练测试的操作步骤

### 1. 准备训练数据
首先需要获取训练数据：
1. 下载F2LLM训练数据集：`https://huggingface.co/datasets/codefuse-ai/F2LLM`
2. 将数据放置在`training_data`目录下

### 2. 配置模型参数
项目已经为您提供了BERT模型的配置文件`configs/config_bert.json`，其中包含：
- `"model_path": "models/bert-base-uncased"` - 指定BERT模型路径
- `"model_type": "encoder"` - 指定模型类型为encoder
- 其他超参数如学习率、批次大小、序列长度等

### 3. 数据预处理
使用专门的脚本对数据进行预处理：
```bash
python tokenize_data_bert.py
```

这个脚本会：
- 自动加载BERT tokenizer
- 确保[CLS]和[SEP]特殊标记可用
- 处理查询和文档数据，添加适当的标记
- 将处理后的数据保存在`data_tokenized_bert`目录中

### 4. 启动训练
使用以下命令启动BERT模型的训练：
```bash
accelerate launch --config_file configs/accelerate_config.yaml run.py --config configs/config_bert.json
```

### 5. 监控训练过程
训练过程中可以通过以下方式监控：
- 查看终端输出的日志信息
- 使用TensorBoard查看训练指标：
  ```bash
  tensorboard --logdir output/tb
  ```

## 如何判断训练结果是否符合预期

在使用BERT类模型进行训练时，可以通过以下几个方面来判断输出结果是否符合预期：

### 1. 训练过程指标
- **损失值（Loss）**：训练损失应逐渐下降，验证损失也应保持稳定或下降
- **学习率**：应按照设定的调度策略（如余弦退火）正确变化
- **训练速度**：每个step的处理时间应在合理范围内

### 2. 模型输出检查
- **特征维度**：确保输出的嵌入向量维度符合预期（如768维用于BERT-base）
- **特征质量**：通过相似度计算检查嵌入向量是否能正确区分相关和不相关的查询-文档对

### 3. 验证指标
- **验证集性能**：在验证集上的损失应保持稳定或持续下降
- **收敛性**：训练足够轮次后，损失应趋于收敛

### 4. 文件输出检查
- **模型检查点**：检查`output`目录中是否正确保存了模型检查点
- **日志文件**：确认训练参数和过程日志正确记录在`args.json`中
- **TensorBoard日志**：确保`output/tb`目录中有正确的日志文件用于可视化

如果您需要实际运行训练来测试这些步骤，请确保您有足够计算资源（推荐至少8GB GPU内存）和训练数据。