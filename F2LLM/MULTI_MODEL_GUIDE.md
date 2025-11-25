# F2LLM 多模型支持使用指南

## 概述

修改后的F2LLM现在支持多种decoder-only模型，包括Qwen、LLaMA、Baichuan、ChatGLM等系列模型。

## 支持的模型

### 已测试模型
- **Qwen系列**: Qwen-7B, Qwen-14B, Qwen3-4B等
- **LLaMA系列**: LLaMA-7B, LLaMA2-13B等  
- **Baichuan系列**: Baichuan-13B, Baichuan2-13B等
- **ChatGLM系列**: ChatGLM-6B, ChatGLM2-6B等

### 理论支持的模型
任何基于transformers库的decoder-only模型都应该可以工作，包括：
- GPT系列
- CodeT5+
- CodeGen
- StarCoder
- 以及其他自定义decoder-only模型

## 使用方法

### 1. 模型配置

修改配置文件 `configs/config.json`：

```json
{
  "model_path": "your-model-path",
  "model_type": "auto",  // 可选: auto, qwen, llama, baichuan等
  "attn_implementation": "flash_attention_2", // flash_attention_2, sdpa, null
  "use_flash_attention": true,
  // ... 其他配置
}
```

#### 配置说明

- **model_path**: 模型路径或HuggingFace模型名称
- **model_type**: 模型类型，用于自动适配特殊处理
- **attn_implementation**: 注意力实现方式
  - `"flash_attention_2"`: 使用Flash Attention 2（最快，但需要支持）
  - `"sdpa"`: 使用PyTorch的Scaled Dot Product Attention
  - `null`: 不使用特殊注意力实现
- **use_flash_attention**: 是否尝试使用flash attention

### 2.获取训练数据
#### 方案1：使用huggingface-cli

如果您想使用原始的huggingface-cli命令：

```bash
# 安装huggingface-hub
pip install huggingface-hub

# 从huggingface中下载训练数据，若遇网络问题，可以考虑使用镜像
export HF_ENDPOINT=https://hf-mirror.com
python -m huggingface_hub.cli download codefuse-ai/F2LLM --repo-type dataset --local-dir training_data --include "*.parquet"
```

#### 方案2：手动下载

1. 访问网站：https://huggingface.co/datasets/codefuse-ai/F2LLM
2. 手动下载.parquet文件
3. 保存到 `training_data/` 目录

### 3. 数据预处理

使用通用分词脚本处理数据：

```bash
# 基础用法
python tokenize_data.py --model_path "meta-llama/Llama-2-7b-hf" --max_seq_length 1023

# 完整参数
python tokenize_data.py \
    --model_path "baichuan-inc/Baichuan2-13B-Base" \
    --max_seq_length 1023 \
    --data_dir "training_data" \
    --output_dir "data_tokenized" \
    --num_processes 16
```

### 4. 训练

```bash
# 单GPU训练
accelerate launch --config_file configs/accelerate_config.yaml run.py --config configs/config.json

# 多GPU训练
accelerate launch --config_file configs/accelerate_config.yaml --num_processes 8 run.py --config configs/config.json
```

## 模型特定配置

### LLaMA模型
```json
{
  "model_path": "meta-llama/Llama-2-7b-hf",
  "model_type": "llama",
  "attn_implementation": "sdpa",
  "use_flash_attention": true,
  "max_seq_length": 2048
}
```

### Baichuan模型
```json
{
  "model_path": "baichuan-inc/Baichuan2-13B-Base",
  "model_type": "baichuan", 
  "attn_implementation": "flash_attention_2",
  "use_flash_attention": true,
  "max_seq_length": 2048
}
```

### ChatGLM模型
```json
{
  "model_path": "THUDM/chatglm3-6b-base",
  "model_type": "chatglm",
  "attn_implementation": null,
  "use_flash_attention": false,
  "max_seq_length": 2048
}
```

## 故障排除

### 常见问题

1. **Flash Attention不支持**
   - 错误信息: `FlashAttention only supports Ampere GPUs or newer.`
   - 解决: 设置 `"use_flash_attention": false` 或 `"attn_implementation": "sdpa"`

2. **内存不足**
   - 减小 `train_batch_size`
   - 减小 `max_seq_length`
   - 使用梯度累积

3. **模型加载失败**
   - 确保模型路径正确
   - 检查网络连接（如果是HF模型）
   - 查看具体的错误信息，调整注意力配置

### 调试建议

1. **逐步测试**
   ```bash
   # 先测试模型加载
   python -c "from transformers import AutoModel; model = AutoModel.from_pretrained('your-model')"
   
   # 再测试分词
   python tokenize_data.py --model_path "your-model" --num_processes 1
   ```

2. **查看日志**
   - 修改后的代码会输出详细的加载信息
   - 关注警告信息，它们通常包含有用的回退信息

3. **性能优化**
   - 优先使用Flash Attention 2（如果硬件支持）
   - 使用SDPA作为第二选择
   - 禁用特殊注意力实现作为最后手段

## 性能对比

| 模型 | 注意力实现 | 训练速度 | 内存使用 | 兼容性 |
|------|------------|----------|----------|---------|
| Qwen3-4B | flash_attention_2 | ★★★★★ | ★★★★★ | ★★★★☆ |
| LLaMA2-7B | sdpa | ★★★★☆ | ★★★★☆ | ★★★★★ |
| Baichuan2-13B | flash_attention_2 | ★★★★★ | ★★★★☆ | ★★★☆☆ |
| ChatGLM3-6B | default | ★★★☆☆ | ★★★☆☆ | ★★★★★ |

## 扩展支持

如果需要支持新的模型类型，可以：

1. 在 `model.py` 中添加模型特定的处理逻辑
2. 在配置文件中添加相应的模型类型标识
3. 测试并验证兼容性

## 注意事项

1. **模型许可**: 确保你有权使用指定的模型
2. **硬件要求**: 大型模型需要更多GPU内存
3. **数据格式**: 确保训练数据格式与模型要求一致
4. **分词器兼容性**: 不同模型可能使用不同的分词器

## 技术支持

如遇到问题，请提供以下信息：
- 模型名称和版本
- 完整的错误日志
- 硬件配置（GPU型号、内存等）
- 配置文件内容
