from arguments import parse_args
from utils_ray import ray_train, CLASSIFICATION_DATASETS
from transformers import (
    AutoTokenizer,
    set_seed,
    get_scheduler
)
import os, json, random
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from model import F2LLM
import ray
from ray import train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
import deepspeed
import yaml

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_deepspeed_config_from_accelerate(accelerate_config_path, args, world_size):
    """
    从 accelerate_config.yaml 加载 DeepSpeed 配置并转换为 DeepSpeed 格式
    与 run.py 中 Accelerate 的行为保持一致，只读取配置文件中实际存在的配置项
    """
    with open(accelerate_config_path, 'r') as f:
        accelerate_config = yaml.safe_load(f)
    
    # 检查是否使用 DeepSpeed
    if accelerate_config.get('distributed_type') != 'DEEPSPEED':
        return None
    
    deepspeed_config = accelerate_config.get('deepspeed_config', {})
    
    # 构建 DeepSpeed 配置 - 只包含 accelerate_config.yaml 中存在的配置项
    ds_config = {
        "train_batch_size": args.train_batch_size * world_size,
        "train_micro_batch_size_per_gpu": args.train_batch_size,
    }
    
    # 从 accelerate_config.yaml 读取的配置项
    if 'gradient_accumulation_steps' in deepspeed_config:
        ds_config["gradient_accumulation_steps"] = deepspeed_config['gradient_accumulation_steps']
    
    if 'gradient_clipping' in deepspeed_config:
        ds_config["gradient_clipping"] = deepspeed_config['gradient_clipping']
    
    # ZeRO 优化配置
    if 'zero_stage' in deepspeed_config:
        zero_stage = deepspeed_config['zero_stage']
        ds_config["zero_optimization"] = {"stage": zero_stage}
        
        # offload 配置
        if 'offload_optimizer_device' in deepspeed_config:
            ds_config["zero_optimization"]["offload_optimizer"] = {
                "device": deepspeed_config['offload_optimizer_device']
            }
        
        if 'offload_param_device' in deepspeed_config:
            ds_config["zero_optimization"]["offload_param"] = {
                "device": deepspeed_config['offload_param_device']
            }
    
    # 混合精度配置
    mixed_precision = accelerate_config.get('mixed_precision', 'no')
    if mixed_precision == "fp16":
        ds_config["fp16"] = {"enabled": True}
    elif mixed_precision == "bf16":
        ds_config["bf16"] = {"enabled": True}
    
    print(f"Loaded DeepSpeed config: {ds_config}")
    
    return ds_config


def _stack(input_ids, max_len):
    data = [ids[:max_len] for ids in input_ids]     # input_ids: list of lists
    lens = [len(x) for x in data]
    tensor = torch.tensor(sum(data, []))            # (total_tokens,)
    return tensor.split(lens)                       # list of 1-d tensors


def get_collate_fn(tokenizer, args):
    def collate_fn(batch_raw):
        '''
            length of input_ids: bs * (2 + num_hard_neg)
            0 - bs-1: query input ids
            bs - 2*bs-1: passage input ids
            2*bs - 2*bs+num_hard_neg-1: hard neg for sample 1
            2*bs+num_hard_neg*(i-1) - 2*bs+num_hard_neg*i-1: hard neg for sample i (i from 1 to bs)
        '''
        num_hard_neg = 1 if batch_raw[0]['dataset_name'] in CLASSIFICATION_DATASETS else args.num_hard_neg
        # select args.num_hard_neg hard negatives from a total of 24
        hard_neg_indices = [0] if num_hard_neg == 1 else random.sample(list(range(24)), num_hard_neg)
        input_ids = _stack(
            [s['query_input_ids'] for s in batch_raw]+\
            [s['passage_input_ids'] for s in batch_raw]+\
            [s[f'negative_{i+1}_input_ids'] for s in batch_raw for i in hard_neg_indices],
            args.max_seq_length
        )
        seqlens = torch.tensor([ids.size(0) for ids in input_ids])
        # pad input ids to [bs, max_len]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_masks = input_ids.ne(tokenizer.pad_token_id).long()
        
        return {'input_ids': input_ids, 'seq_lens': seqlens, 'attention_mask': attention_masks, 'bs': len(batch_raw), 'dataset_name': batch_raw[0]['dataset_name']}
    
    return collate_fn


class MultiLoader:
    """
    Iterates over a dict(name -> DataLoader) and returns complete batches.
    At every __iter__ a new random order is created;
    the epoch ends when every loader is exhausted once.
    """
    def __init__(self, loader_dict):
        self.loader_dict = loader_dict

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
    """Ray Train training function"""
    # Get configuration
    args = config["args"]
    
    # Set seed
    set_seed(0)
    
    # Get world info
    world_size = train.get_context().get_world_size()
    world_rank = train.get_context().get_world_rank()
    local_rank = train.get_context().get_local_rank()
    
    # Set device
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    if world_rank == 0:
        os.makedirs(f"{args.output_dir}", exist_ok=True)
        with open(os.path.join(args.output_dir, "args.json"), "w") as f:   
            json.dump(args.dict(), f, indent=2)
    
    # Load datasets
    train_datasets, valid_datasets = [], []
    for f in sorted(os.listdir(args.train_data_path)):
        dataset_name = f.split('.parquet')[0]
        dataset = load_dataset("parquet", data_files=os.path.join(args.train_data_path, f), cache_dir=args.cache_dir)['train']
        dataset = dataset.add_column("dataset_name", [dataset_name]*len(dataset))
        dataset = dataset.train_test_split(train_size=0.99, shuffle=True, seed=0)
        train_datasets.append((dataset_name, dataset['train']))
        valid_datasets.append((dataset_name, dataset['test']))
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    collate_fn = get_collate_fn(tokenizer, args)
    
    # Create data loaders
    train_loaders = {
        name: DataLoader(ds, shuffle=True, batch_size=args.train_batch_size, collate_fn=collate_fn)
        for name, ds in train_datasets
    }
    valid_loaders = {
        name: DataLoader(ds, shuffle=False, batch_size=args.train_batch_size, collate_fn=collate_fn)
        for name, ds in valid_datasets
    }
    
    # Prepare Ray Train DataLoader
    from ray.train.torch import prepare_data_loader
    for k in train_loaders.keys():
        train_loaders[k] = prepare_data_loader(train_loaders[k])
    for k in valid_loaders.keys():
        valid_loaders[k] = prepare_data_loader(valid_loaders[k])
    
    train_dataloader = MultiLoader(train_loaders)
    
    # Determine training steps
    override_train_step = False
    if args.train_steps < 0:
        args.train_steps = sum(len(v) for v in train_loaders.values()) * args.train_epochs
        override_train_step = True
    
    if world_rank == 0:
        print(f"******************************** Training step before prepare: {args.train_steps} ********************************")
    
    # Initialize model
    model = F2LLM(args.model_path, args.max_seq_length, args=args)
    model.lm.gradient_checkpointing_enable()
    
    # Set seed again
    set_seed(0)
    
    # Create optimizer and scheduler (like run.py)
    optimizer = AdamW(model.lm.parameters(),
                      weight_decay=args.weight_decay,
                      lr=args.learning_rate,
                      betas=(0.9, 0.98))
    
    lr_scheduler = get_scheduler("cosine",
                                optimizer=optimizer,
                                num_warmup_steps=args.warmup_steps,
                                num_training_steps=args.train_steps)
    
    # Load DeepSpeed configuration from accelerate_config.yaml
    accelerate_config_path = os.path.join(os.path.dirname(__file__), "configs", "accelerate_config.yaml")
    ds_config = load_deepspeed_config_from_accelerate(accelerate_config_path, args, world_size)
    
    if ds_config is None:
        raise ValueError("DeepSpeed configuration not found in accelerate_config.yaml")
    
    if world_rank == 0:
        print("=" * 80)
        print("DeepSpeed Configuration loaded from accelerate_config.yaml:")
        print(json.dumps(ds_config, indent=2))
        print("=" * 80)
    
    # Initialize DeepSpeed with external optimizer and scheduler
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model.lm,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=ds_config,
        dist_init_required=False  # Ray Train handles distributed initialization
    )
    
    # Update model reference
    model.lm = model_engine
    model.set_device()
    
    # Update training steps if needed
    if override_train_step:
        args.train_steps = len(train_dataloader) * args.train_epochs
    
    if world_rank == 0:
        print(f"******************************** Training step after prepare: {args.train_steps} ********************************")
    
    # Get total number of samples
    num_train_samples = sum(len(ds) for _, ds in train_datasets)
    
    # Start training
    ray_train(args, model, train_dataloader, valid_loaders,
              optimizer, lr_scheduler, num_train_samples, 
              world_size, world_rank, device)


def main():
    args = parse_args()
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()
    
    # Configure scaling
    scaling_config = ScalingConfig(
        num_workers=args.num_processes if hasattr(args, 'num_processes') and args.num_processes > 0 else 2,
        use_gpu=True,
        resources_per_worker={"CPU": 8, "GPU": 1},
    )
    
    # Create Ray TorchTrainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={"args": args},
        scaling_config=scaling_config,
    )
    
    # Start training
    result = trainer.fit()
    
    print("Training completed!")
    print(f"Results: {result}")


if __name__ == "__main__":
    main()
