# LoRA支持需求文档

## 1. 需求背景

在当前的CodeFuse-Embeddings项目中，F2LLM模块支持将Decoder-only LLMs转换为Embedding模型，主要通过全模型微调或转换的方式实现。为了提高训练效率并降低计算成本，我们引入了LoRA (Low-Rank Adaptation) PEFT (Parameter-Efficient Fine-Tuning) 方法的支持，使用户能够通过最小参数更新来适配基础模型。

## 2. 需求目标

- 实现LoRA微调方法的支持，提高训练效率
- 减少模型训练时的内存使用和计算成本
- 保持模型性能的同时显著减少可训练参数数量
- 提供灵活的LoRA配置选项，满足不同场景需求

## 3. 功能实现

### 3.1 核心实现

1. **LoRA集成**
   - 使用Hugging Face的PEFT库实现LoRA功能
   - 支持在指定模型层应用LoRA适配器
   - 实现LoRA参数的配置化管理

2. **模型支持**
   - 支持Qwen系列模型的LoRA微调
   - 可扩展支持其他Decoder-only架构的LLMs

### 3.2 配置参数

LoRA功能通过以下参数进行配置：

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `use_lora` | bool | false | 是否启用LoRA |
| `lora_r` | int | 8 | LoRA矩阵的秩 |
| `lora_alpha` | int | 32 | LoRA的缩放因子 |
| `lora_dropout` | float | 0.1 | LoRA层的Dropout率 |
| `lora_target_modules` | list | ["q_proj", "v_proj"] | 应用LoRA的模块列表 |

### 3.3 配置文件示例

```json
{
  "model_path": "models/qwen3-0.6b",
  "experiment_id": "0.6b_lora_test",
  "train_data_path": "training_data/data_tokenized_qwen",
  "output_dir": "output",
  "tb_dir": "output/tb",
  "cache_dir": "cache",
  "train_batch_size": 16,
  "checkpointing_steps": 5000,
  "validation_steps": 5000,
  "max_seq_length": 1024,
  "learning_rate": 8e-6,
  "min_lr": 1e-7,
  "weight_decay": 0.1,
  "warmup_steps": 500,
  "train_epochs": 2,
  "log_interval": 100,
  "num_hard_neg": 7,
  "use_lora": true,
  "lora_r": 8,
  "lora_alpha": 32,
  "lora_dropout": 0.1,
  "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
}
```

### 3.4 代码实现

#### 3.4.1 模型初始化

在`F2LLM/model.py`中，通过以下代码集成LoRA支持：

```python
# 检查是否启用LoRA
if args and getattr(args, 'use_lora', False):
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules
    )
    self.lm = get_peft_model(self.lm, peft_config)
    print("LoRA enabled")
    self.lm.print_trainable_parameters()
```

#### 3.4.2 模型保存

在`F2LLM/utils.py`中，实现了针对LoRA模型的特殊保存逻辑：

```python
# Handle LoRA model saving
if getattr(args, 'use_lora', False):
    # For LoRA models, we only save the adapter weights
    unwrapped_model.save_pretrained(
        output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save
    )
else:
    # For full fine-tuning, save the entire model
    unwrapped_model.save_pretrained(
        output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model.lm),
    )
```

### 3.5 使用方法

1. 创建LoRA配置文件，设置`use_lora: true`及其他相关参数
2. 运行训练脚本：
   ```bash
   cd F2LLM
   python run.py --config configs/config_lora.json
   ```
3. 训练完成后，LoRA适配器权重将保存在输出目录中

### 3.6 模型加载

对于LoRA模型，可以使用PEFT库加载：

```python
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel

base_model = AutoModel.from_pretrained('base_model_path')
model = PeftModel.from_pretrained(base_model, 'output/{experiment_id}')
tokenizer = AutoTokenizer.from_pretrained('base_model_path')
```

## 4. 优势与效果

### 4.1 训练效率提升

- 显著减少可训练参数数量（通常减少99%以上）
- 降低内存使用，支持在资源受限设备上训练更大模型
- 缩短训练时间

### 4.2 性能保持

- 在保持模型性能的同时实现参数高效微调
- 支持与全参数微调相当的模型质量

### 4.3 灵活性

- 可配置的LoRA参数满足不同场景需求
- 支持指定不同的模型层应用LoRA

## 5. 注意事项

1. LoRA适配器需要与基础模型配合使用，单独加载适配器权重无法工作
2. 不同的`lora_r`值会影响模型性能和训练效率的平衡
3. `lora_target_modules`需要根据具体模型架构进行调整

## 6. 后续优化方向

1. 支持更多类型的PEFT方法（如AdaLoRA、IA³等）
2. 提供自动化的LoRA参数搜索功能
3. 增加对更多模型架构的支持
4. 优化LoRA在推理阶段的性能
