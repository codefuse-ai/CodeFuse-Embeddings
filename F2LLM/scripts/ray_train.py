import ray
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import json
import random
from datasets import load_dataset
from arguments import parse_args
from model import F2LLM
from utils import CLASSIFICATION_DATASETS
import argparse
import logging

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Suppress PyTorch redirect warnings on Mac
os.environ["TORCH_DISTRIBUTED_ELASTIC_LOG_LEVEL"] = "ERROR"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def monitor_cluster_status():
    """Monitor cluster status and report metrics"""
    try:
        # Get cluster resources
        resources = ray.available_resources()
        logger.info(f"Available resources: {resources}")
        
        # Get node information
        nodes = ray.nodes()
        logger.info(f"Number of nodes: {len(nodes)}")
        
        return resources, nodes
    except Exception as e:
        logger.error(f"Failed to monitor cluster status: {e}")
        return None, None


def _stack(input_ids, max_len):
    """Stack input ids with padding"""
    data = [ids[:max_len] for ids in input_ids]
    lens = [len(x) for x in data]
    tensor = torch.tensor(sum(data, []))
    return tensor.split(lens)


def collate_fn(batch_raw, args, tokenizer, num_hard_neg=None):
    '''
    Collate function for DataLoader
    length of input_ids: bs * (2 + num_hard_neg)
    0 - bs-1: query input ids
    bs - 2*bs-1: passage input ids
    2*bs - 2*bs+num_hard_neg-1: hard neg for sample 1
    2*bs+num_hard_neg*(i-1) - 2*bs+num_hard_neg*i-1: hard neg for sample i (i from 1 to bs)
    '''
    if num_hard_neg is None:
        num_hard_neg = 1 if batch_raw[0]['dataset_name'] in CLASSIFICATION_DATASETS else args.num_hard_neg
    
    # select args.num_hard_neg hard negatives from a total of 24
    hard_neg_indices = [0] if num_hard_neg == 1 else random.sample(list(range(24)), num_hard_neg)
    input_ids = _stack(
        [s['query_input_ids'] for s in batch_raw] + \
        [s['passage_input_ids'] for s in batch_raw] + \
        [s[f'negative_{i+1}_input_ids'] for s in batch_raw for i in hard_neg_indices],
        args.max_seq_length
    )
    seqlens = torch.tensor([ids.size(0) for ids in input_ids])
    # pad input ids to [bs, max_len]
    input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long()
    
    return {'input_ids': input_ids, 'seq_lens': seqlens, 'attention_mask': attention_masks, 'bs': len(batch_raw), 'dataset_name': batch_raw[0]['dataset_name']}


def load_data(args, tokenizer, worker_index=0, num_workers=1):
    """Load training and validation datasets with distributed support"""
    logger.info(f"Loading data for worker {worker_index}/{num_workers}")
    
    train_datasets, valid_datasets = [], []
    
    # 获取当前工作节点的文件列表
    all_files = sorted(os.listdir(args.train_data_path))
    
    # 根据工作节点索引分发文件
    worker_files = all_files[worker_index::num_workers] if num_workers > 1 else all_files
    
    logger.info(f"Worker {worker_index} processing files: {worker_files}")
    
    for f in worker_files:
        dataset_name = f.split('.parquet')[0]
        dataset = load_dataset("parquet", data_files=os.path.join(args.train_data_path, f), cache_dir=args.cache_dir)['train']
        dataset = dataset.add_column("dataset_name", [dataset_name]*len(dataset))
        dataset = dataset.train_test_split(train_size=0.99, shuffle=True, seed=0)
        train_datasets.append((dataset_name, dataset['train']))
        valid_datasets.append((dataset_name, dataset['test']))

    train_loaders = {
        name: DataLoader(ds, shuffle=True, batch_size=args.train_batch_size, 
                        collate_fn=lambda batch: collate_fn(batch, args, tokenizer))
        for name, ds in train_datasets
    }
    valid_loaders = {
        name: DataLoader(ds, shuffle=False, batch_size=args.train_batch_size, 
                        collate_fn=lambda batch: collate_fn(batch, args, tokenizer))
        for name, ds in valid_datasets
    }
    
    return train_loaders, valid_loaders


class MultiLoader:
    """
    Iterates over a dict(name -> DataLoader) and returns complete batches.
    At every __iter__ a new random order is created;
    the epoch ends when every loader is exhausted once.
    """
    def __init__(self, loader_dict, accelerator=None):
        self.loader_dict = loader_dict
        # For Ray, we don't need to prepare the loaders with accelerator
        # The preparation will be done by Ray Train

    def __len__(self):
        return sum(len(v) for v in self.loader_dict.values())
    
    def reset_epoch(self, epoch):
        self.rng = random.Random(epoch)
        self.iters = {k: iter(v) for k, v in self.loader_dict.items()}
        self.names = list(self.iters.keys())
        self.weights = [len(self.loader_dict[k]) for k in self.names]

    def __iter__(self):
        while self.names:                           # until every DataLoader is empty
            name = self.rng.choices(self.names, weights=self.weights)[0] # pick a data-source at random
            try:
                batch = next(self.iters[name])
                yield batch
            except StopIteration:
                idx = self.names.index(name)
                self.names.pop(idx)                 # this dataset has no batch left
                self.weights.pop(idx)


def train_func(config):
    """Main training function for Ray Train with multi-GPU support"""
    # Import required modules
    from arguments import Args
    from transformers import AutoTokenizer, set_seed, get_scheduler
    from torch.optim import AdamW
    import ray
    from ray.train import get_context
    import torch.distributed as dist
    import torch
    
    # Convert config to Args object
    args = Args(**config)
    args.output_dir = f"{args.output_dir}/{args.experiment_id}"
    args.tb_dir = f"{args.tb_dir}/{args.experiment_id}"
    
    # Set seed for reproducibility
    set_seed(0)
    
    # Get the Ray Train context
    train_context = get_context()
    
    # Get worker information
    world_rank = train_context.get_world_rank()
    world_size = train_context.get_world_size()
    
    # Get GPU information
    local_rank = train_context.get_local_rank()
    local_world_size = train_context.get_local_world_size()
    
    logger.info(f"Worker {world_rank}/{world_size}, GPU {local_rank}/{local_world_size} started")
    
    # Monitor cluster status on head node
    if world_rank == 0:
        monitor_cluster_status()
    
    # Load tokenizer
    # Use local_files_only setting from args if available, otherwise default to True
    local_files_only = getattr(args, 'local_files_only', True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=local_files_only)
    
    # Load data with distributed support
    train_loaders, valid_loaders = load_data(args, tokenizer, world_rank, world_size)
    
    # Create model with multi-GPU support
    # Get use_gpu from the training loop config
    use_gpu = ray.train.get_context().get_world_size() > 0 and torch.cuda.is_available()
    model = F2LLM(args.model_path, args.max_seq_length, args=args, use_multi_gpu=use_gpu)
    model.lm.gradient_checkpointing_enable()
    
    # Set seed again to make sure that different models share the same seed
    set_seed(0)
    
    # Move model to GPU if needed
    if use_gpu:
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        model.set_device(device)
        logger.info(f"Model moved to device: {device}")
    
    # Create optimizer
    optimizer = AdamW(model.lm.parameters(),
                      weight_decay=args.weight_decay,
                      lr=args.learning_rate,
                      betas=(0.9, 0.98))
    
    # Create learning rate scheduler
    lr_scheduler = get_scheduler("cosine",
                                optimizer=optimizer,
                                num_warmup_steps=args.warmup_steps,
                                num_training_steps=args.train_steps)
    
    # Resume from checkpoint if specified
    if hasattr(args, 'resume_from_checkpoint') and args.resume_from_checkpoint and hasattr(args, 'resume_checkpoint_path') and args.resume_checkpoint_path:
        try:
            checkpoint_data = load_ray_checkpoint(args.resume_checkpoint_path)
            if checkpoint_data["model_state_dict"]:
                model.lm.load_state_dict(checkpoint_data["model_state_dict"])
            if checkpoint_data["optimizer_state_dict"]:
                optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
            if checkpoint_data["lr_scheduler_state_dict"]:
                lr_scheduler.load_state_dict(checkpoint_data["lr_scheduler_state_dict"])
            logger.info(f"Resumed training from checkpoint: {args.resume_checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to resume from checkpoint: {e}")
    
    # Prepare model for distributed training
    if use_gpu and local_world_size > 1:
        model.lm = ray.train.torch.prepare_model(model.lm, parallelism="ddp")
        logger.info("Model prepared for distributed training with DDP")
    else:
        model.lm = ray.train.torch.prepare_model(model.lm)
    
    # Prepare data loaders
    for k, v in train_loaders.items():
        train_loaders[k] = ray.train.torch.prepare_data_loader(v)
    for k, v in valid_loaders.items():
        valid_loaders[k] = ray.train.torch.prepare_data_loader(v)
    
    # Create train dataloader
    train_dataloader = MultiLoader(train_loaders)
    
    # Determine training steps
    override_train_step = False
    if args.train_steps < 0:
        args.train_steps = sum(len(v) for v in train_loaders.values()) * args.train_epochs
        override_train_step = True
    
    logger.info(f"******************************** Training step before prepare: {args.train_steps} ********************************")
    
    # If training on multiple GPUs, length of dataloader would have changed
    if override_train_step:
        args.train_steps = len(train_dataloader) * args.train_epochs
    logger.info(f"******************************** Training step after prepare: {args.train_steps} ********************************")
    
    # Import accelerate_train from utils and adapt it for Ray
    # from utils import accelerate_train
    from ray_utils import ray_train
    
    # Get number of training samples
    num_train_samples = sum(len(ds) for _, ds in train_loaders.items())
    
    # Call the adapted training function
    # accelerate_train(args, None, model, train_dataloader, valid_loaders,
    #                  optimizer, lr_scheduler, num_train_samples)
    ray_train(args, model, train_dataloader, valid_loaders,
              optimizer, lr_scheduler, num_train_samples)


def main():
    """Main function to start Ray training with multi-GPU support"""
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU for training")
    parser.add_argument("--cluster-mode", type=str, default="auto", help="Cluster mode: auto or manual")
    parser.add_argument("--cluster-address", type=str, default=None, help="Cluster address for manual mode")
    parser.add_argument("--num-gpus-per-worker", type=int, default=1, help="Number of GPUs per worker")
    parser.add_argument("--resume-from-checkpoint", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--resume-checkpoint-path", type=str, default=None, help="Path to checkpoint for resuming")
    args = parser.parse_args()
    
    # Parse the config file
    with open(args.config) as f:
        config = json.load(f)
    
    # Initialize Ray cluster with runtime environment to exclude large files
    if args.cluster_mode == "manual" and args.cluster_address:
        logger.info(f"Connecting to Ray cluster at {args.cluster_address}")
        ray.init(address=args.cluster_address)
    else:
        logger.info("Connecting to local Ray cluster")
        # Configure runtime environment to exclude large files
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
    
    # Configure Ray Train with dynamic resource allocation
    resources_per_worker = {"CPU": 2}
    if args.use_gpu:
        resources_per_worker["GPU"] = args.num_gpus_per_worker
    
    scaling_config = ScalingConfig(
        num_workers=args.num_workers,
        use_gpu=args.use_gpu,
        resources_per_worker=resources_per_worker
    )
    
    # Create Ray TorchTrainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=config,
        scaling_config=scaling_config,
    )
    
    # Start training
    result = trainer.fit()
    
    logger.info("Training completed successfully!")
    return result


if __name__ == "__main__":
    main()