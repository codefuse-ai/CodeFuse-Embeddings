# Matryoshka Representation Learning (MRL) 支持需求文档

## 1. 背景

在训练CodeFuse-Embeddings模型时，为了提供更大的灵活性以适应不同的下游应用和计算预算，我们实现了Matryoshka Representation Learning (MRL)支持。MRL是一种"俄罗斯套娃"式的训练方法，允许单个模型在推理时产生不同维度的高质量嵌入（例如64、128、256、512、1024等），从而为不同的应用场景提供显著的灵活性。

## 2. 需求目标

为CodeFuse-Embeddings模型增加MRL支持，使得模型能够在训练时学习多个嵌入维度的表示，并在推理时根据需要选择合适的嵌入维度，以平衡性能和计算效率。

## 3. 技术实现

### 3.1 MRL核心概念

Matryoshka Representation Learning是一种训练方法，它允许模型在不同维度上产生嵌入表示，其中较低维度的嵌入是较高维度嵌入的子集。这种方法通过以下方式实现：

1. 在训练过程中，模型同时学习多个目标维度的表示
2. 使用投影层将完整维度的嵌入映射到目标维度
3. 在损失计算时，同时考虑所有目标维度的损失

### 3.2 实现细节

#### 3.2.1 配置参数

在`F2LLM/configs/config.json`中增加了以下MRL相关配置参数：

- `mrl_enabled`: 是否启用MRL（布尔值，默认为false）
- `mrl_dims`: MRL目标维度列表（数组，默认为[128, 256, 512, 1024]）
- `mrl_loss_weights`: 每个维度的损失权重（数组，默认为[1.0, 1.0, 1.0, 1.0]）

#### 3.2.2 模型修改

在`F2LLM/model.py`中实现了以下MRL相关功能：

1. `F2LLM`类中增加了MRL支持：
   - 添加了`mrl_enabled`标志来控制是否启用MRL
   - 创建了针对每个目标维度的投影层(`mrl_projections`)
   - 实现了`get_mrl_embeddings`方法用于获取特定维度的嵌入
   - 实现了`get_all_mrl_embeddings`方法用于获取所有维度的嵌入

2. `forward`方法修改：
   - 增加了`target_dim`参数用于指定目标嵌入维度
   - 根据是否启用MRL返回不同维度的嵌入表示
   - 支持返回所有维度的嵌入字典

#### 3.2.3 训练过程修改

在`F2LLM/utils.py`中实现了以下MRL相关功能：

1. 增加了MRL损失计算函数：
   - `mrl_inbatch_loss`: 计算MRL的批次内负采样损失
   - `mrl_hard_loss`: 计算MRL的硬负样本损失

2. 修改了训练循环：
   - 在每个训练批次中随机选择一个目标维度
   - 根据选择的维度计算相应的损失

3. 修改了验证过程：
   - 在验证时使用完整维度进行评估

#### 3.2.4 参数定义

在`F2LLM/arguments.py`中增加了MRL相关参数定义：

- `mrl_enabled`: 是否启用MRL
- `mrl_dims`: MRL目标维度列表
- `mrl_loss_weights`: 每个维度的损失权重

## 4. 使用方法

### 4.1 启用MRL训练

在配置文件`F2LLM/configs/config.json`中设置：

```json
{
  "mrl_enabled": true,
  "mrl_dims": [128, 256, 512, 1024],
  "mrl_loss_weights": [1.0, 1.0, 1.0, 1.0]
}
```

### 4.2 启动训练

```bash
cd F2LLM
python run.py --config configs/config.json
```

### 4.3 推理时使用不同维度

在推理时，可以通过指定`target_dim`参数来获取特定维度的嵌入：

```python
# 获取特定维度的嵌入
outputs = model.forward(batch, target_dim=512)

# 获取所有维度的嵌入
outputs = model.forward(batch)
```

## 5. 优势

1. **灵活性**: 单个模型可以生成多种维度的嵌入，适应不同应用场景
2. **效率**: 在推理时可以选择较低维度以提高速度，或选择较高维度以获得更好性能
3. **资源优化**: 根据计算预算和性能要求选择合适的嵌入维度
4. **兼容性**: 保持与现有代码的兼容性，通过配置参数控制是否启用MRL

## 6. 验证

变更后，模型能够：

1. 成功训练启用MRL的模型
2. 在推理时生成不同维度的嵌入表示
3. 保持与未启用MRL模型相当的性能
4. 在不同维度之间提供平滑的性能权衡
