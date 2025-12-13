#!/bin/bash

# F2LLM Ray 分布式训练启动脚本

# 设置环境变量
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # 如果不使用 InfiniBand

# 默认配置文件
CONFIG_FILE=${1:-"configs/config_ray.json"}

echo "=========================================="
echo "F2LLM Ray 分布式训练"
echo "=========================================="
echo "配置文件: $CONFIG_FILE"
echo "=========================================="

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件 $CONFIG_FILE 不存在!"
    exit 1
fi

# 检查 Ray 是否安装
python -c "import ray" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "错误: Ray 未安装,请运行: pip install ray[train]"
    exit 1
fi

# 启动训练
echo "开始训练..."
python run_ray.py --config "$CONFIG_FILE"

echo "=========================================="
echo "训练完成!"
echo "=========================================="
