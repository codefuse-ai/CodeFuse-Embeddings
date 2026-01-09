## Ray Distributed Training

This directory contains the Ray-based distributed training implementation for F2LLM embedding models, providing scalable, fault-tolerant training capabilities with automatic resource management and seamless scaling from single-node to multi-node clusters.

### Usage

#### Single-Node Training
```bash
python ray_distributed_run.py --config configs/ray_config.json --num_workers 4 --num_gpus_per_worker 1.0
```

#### Multi-Node Training

1. On the head node:
```bash
ray start --head --port=6379
python ray_distributed_run.py --config configs/ray_config.json --num_workers 8 --num_gpus_per_worker 1.0 --ray_head_address HEAD_NODE_IP
```

2. On worker nodes:
```bash
ray start --address=HEAD_NODE_IP:6379
```

### Configuration

The Ray-specific configuration extends the original config with these additional parameters:

- `num_workers`: Number of Ray workers (processes) to use
- `num_gpus_per_worker`: Number of GPUs per worker
- `num_cpus_per_worker`: Number of CPUs per worker

### Requirements

Install Ray-specific dependencies:

```bash
pip install -r ray_requirements.txt
```
