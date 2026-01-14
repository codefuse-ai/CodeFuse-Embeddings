"""
Ray Train Integration for F2LLM

This module provides Ray Train distributed training capabilities for the F2LLM
embedding model, maintaining compatibility with the existing Accelerate-based training.
"""

from ray_config import RayTrainConfig
from utils import (
    DistributedContext, inbatch_loss, hard_loss, validate,
    write_tensorboard, save_checkpoint,
    CLASSIFICATION_DATASETS, RETRIEVAL_DATASETS, CLUSTERING_DATASETS
)
from transformers import (
    AutoTokenizer,
    set_seed,
    get_scheduler
)
import os
import json
import random
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from model import F2LLM
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

import ray.train
from ray.train import ScalingConfig, RunConfig, CheckpointConfig, FailureConfig
from ray.train.torch import TorchTrainer, TorchConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _stack(input_ids, max_len):
    """Stack and truncate input IDs"""
    data = [ids[:max_len] for ids in input_ids]
    lens = [len(x) for x in data]
    tensor = torch.tensor(sum(data, []))
    return tensor.split(lens)


def train_func(config):
    """
    Main training function executed on each Ray worker

    Args:
        config: Training configuration dictionary
    """
    # Convert config dict to RayTrainConfig object
    args = RayTrainConfig(**config)

    # Initialize distributed context (Ray backend)
    distributed_ctx = DistributedContext(backend='ray')

    # Set random seed for reproducibility
    set_seed(0)

    # Create output directories (main process only)
    if distributed_ctx.is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            json.dump(args.dict(), f, indent=2)

    # Load datasets
    distributed_ctx.print("Loading datasets...")
    train_datasets, valid_datasets = [], []
    for f in sorted(os.listdir(args.train_data_path)):
        if not f.endswith('.parquet'):
            continue
        dataset_name = f.split('.parquet')[0]
        dataset_path = os.path.join(args.train_data_path, f)
        dataset = load_dataset("parquet", data_files=dataset_path, cache_dir=args.cache_dir)['train']
        dataset = dataset.add_column("dataset_name", [dataset_name]*len(dataset))
        dataset = dataset.train_test_split(train_size=0.99, shuffle=True, seed=0)
        train_datasets.append((dataset_name, dataset['train']))
        valid_datasets.append((dataset_name, dataset['test']))

    distributed_ctx.print(f"Loaded {len(train_datasets)} datasets")

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    def collate_fn(batch_raw):
        """Collate function for data loading"""
        num_hard_neg = 1 if batch_raw[0]['dataset_name'] in CLASSIFICATION_DATASETS else args.num_hard_neg
        hard_neg_indices = [0] if num_hard_neg == 1 else random.sample(list(range(24)), num_hard_neg)

        input_ids = _stack(
            [s['query_input_ids'] for s in batch_raw] +
            [s['passage_input_ids'] for s in batch_raw] +
            [s[f'negative_{i+1}_input_ids'] for s in batch_raw for i in hard_neg_indices],
            args.max_seq_length
        )
        seqlens = torch.tensor([ids.size(0) for ids in input_ids])
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_masks = input_ids.ne(tokenizer.pad_token_id).long()

        return {
            'input_ids': input_ids,
            'seq_lens': seqlens,
            'attention_mask': attention_masks,
            'bs': len(batch_raw),
            'dataset_name': batch_raw[0]['dataset_name']
        }

    # Create data loaders
    train_loaders = {
        name: DataLoader(ds, shuffle=True, batch_size=args.train_batch_size, collate_fn=collate_fn)
        for name, ds in train_datasets
    }
    valid_loaders = {
        name: DataLoader(ds, shuffle=False, batch_size=args.train_batch_size, collate_fn=collate_fn)
        for name, ds in valid_datasets
    }

    # Prepare data loaders for Ray Train
    from ray.train.torch import prepare_data_loader
    train_loaders = {
        name: prepare_data_loader(loader)
        for name, loader in train_loaders.items()
    }
    valid_loaders = {
        name: prepare_data_loader(loader)
        for name, loader in valid_loaders.items()
    }

    class MultiLoader:
        """Multi-dataset loader with weighted sampling"""
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
            while self.names:
                name = self.rng.choices(self.names, weights=self.weights)[0]
                try:
                    batch = next(self.iters[name])
                    yield batch
                except StopIteration:
                    idx = self.names.index(name)
                    self.names.pop(idx)
                    self.weights.pop(idx)

    # Determine training steps
    override_train_step = False
    if args.train_steps < 0:
        args.train_steps = sum(len(v) for v in train_loaders.values()) * args.train_epochs
        override_train_step = True

    distributed_ctx.print(f"Training steps before prepare: {args.train_steps}")

    # Create model
    distributed_ctx.print("Creating model...")
    model = F2LLM(args.model_path, args.max_seq_length, args=args)
    model.lm.gradient_checkpointing_enable()

    # Set seed again for consistent initialization
    set_seed(0)

    # Create optimizer and scheduler
    optimizer = AdamW(
        model.lm.parameters(),
        weight_decay=args.weight_decay,
        lr=args.learning_rate,
        betas=(0.9, 0.98)
    )

    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.train_steps
    )

    # Prepare model and optimizer for Ray Train
    from ray.train.torch import prepare_model, prepare_optimizer
    model.lm = prepare_model(model.lm)
    optimizer = prepare_optimizer(optimizer)

    # Set model device
    model.set_device()

    # Create MultiLoader
    train_dataloader = MultiLoader(train_loaders)

    # Adjust training steps if needed
    if override_train_step:
        args.train_steps = len(train_dataloader) * args.train_epochs
    distributed_ctx.print(f"Training steps after prepare: {args.train_steps}")

    # Start training
    distributed_ctx.print("=" * 80)
    distributed_ctx.print("Starting training")
    distributed_ctx.print(f"  Num train samples = {sum(len(ds) for _, ds in train_datasets)}")
    distributed_ctx.print(f"  Num epochs = {args.train_epochs}")
    distributed_ctx.print(f"  Per device batch size = {args.train_batch_size}")
    distributed_ctx.print(f"  Global batch size = {args.train_batch_size * distributed_ctx.world_size}")
    distributed_ctx.print(f"  Steps per epoch = {len(train_dataloader)}")
    distributed_ctx.print(f"  Total training steps = {args.train_steps}")
    distributed_ctx.print("=" * 80)

    # Filter datasets
    global RETRIEVAL_DATASETS, CLASSIFICATION_DATASETS, CLUSTERING_DATASETS
    RETRIEVAL_DATASETS = [ds for ds in RETRIEVAL_DATASETS if ds in train_dataloader.loader_dict.keys()]
    CLASSIFICATION_DATASETS = [ds for ds in CLASSIFICATION_DATASETS if ds in train_dataloader.loader_dict.keys()]
    CLUSTERING_DATASETS = [ds for ds in CLUSTERING_DATASETS if ds in train_dataloader.loader_dict.keys()]

    # Initialize TensorBoard writer
    summary_writer = SummaryWriter(log_dir=args.tb_dir) if distributed_ctx.is_main_process() else None

    # Training loop
    criterion = CrossEntropyLoss(reduction='none')
    pbar = tqdm(range(args.train_steps), disable=not distributed_ctx.is_local_main_process())
    completed_steps = 0

    # Initialize loss tracking
    loss_dict = {ds_name: torch.tensor(0.0, device=model.lm.device) for ds_name in RETRIEVAL_DATASETS}
    loss_hard_dict = {ds_name: torch.tensor(0.0, device=model.lm.device) for ds_name in train_dataloader.loader_dict.keys()}
    count_dict = {ds_name: torch.tensor(0, device=model.lm.device) for ds_name in RETRIEVAL_DATASETS}
    count_hard_dict = {ds_name: torch.tensor(0, device=model.lm.device) for ds_name in train_dataloader.loader_dict.keys()}

    model.lm.train()

    for epoch in range(args.train_epochs):
        distributed_ctx.print(f"Starting epoch {epoch+1}")
        train_dataloader.reset_epoch(epoch)

        for batch in train_dataloader:
            # Forward pass
            outputs = model.forward(batch)

            # Compute losses
            loss_hard = hard_loss(
                outputs['query_passage_features'].squeeze(1),
                outputs['passage_passage_features'].squeeze(1),
                outputs['negative_passage_features'],
                criterion,
                distributed_ctx
            )

            dataset_name = batch['dataset_name']
            count_hard_dict[dataset_name] += 1
            loss_hard_dict[dataset_name] += loss_hard.detach().float()

            if dataset_name in RETRIEVAL_DATASETS:
                loss = inbatch_loss(
                    outputs['query_passage_features'].squeeze(1),
                    outputs['passage_passage_features'].squeeze(1),
                    criterion,
                    distributed_ctx
                )
                count_dict[dataset_name] += 1
                loss_dict[dataset_name] += loss.detach().float()
            else:
                loss = 0.0

            loss_total = loss + loss_hard

            # Backward pass
            loss_total.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Enforce minimum learning rate
            if optimizer.param_groups[0]['lr'] < args.min_lr:
                for i in range(len(optimizer.param_groups)):
                    optimizer.param_groups[i]['lr'] = args.min_lr

            # Logging
            completed_steps += 1
            if completed_steps % args.log_interval == 0:
                pbar.update(args.log_interval)

                train_log_dict = {"lr": optimizer.param_groups[0]['lr']}

                # Aggregate losses across GPUs
                for k in loss_dict.keys():
                    count = distributed_ctx.gather(count_dict[k]).sum()
                    if count > 0:
                        train_log_dict[f"{k}/training_loss_in_batch"] = distributed_ctx.gather(loss_dict[k]).sum() / count

                for k in loss_hard_dict.keys():
                    count = distributed_ctx.gather(count_hard_dict[k]).sum()
                    if count > 0:
                        train_log_dict[f"{k}/training_loss_hard"] = distributed_ctx.gather(loss_hard_dict[k]).sum() / count

                # Compute averages
                train_log_dict['Avg/retrieval/training_loss_in_batch'] = torch.tensor([v for k, v in train_log_dict.items() if k.split('/')[0] in RETRIEVAL_DATASETS and k.endswith('training_loss_in_batch')]).mean() if any(k.endswith('training_loss_in_batch') for k in train_log_dict.keys()) else torch.tensor(0.0)
                train_log_dict['Avg/retrieval/training_loss_hard'] = torch.tensor([v for k, v in train_log_dict.items() if k.split('/')[0] in RETRIEVAL_DATASETS and k.endswith('training_loss_hard')]).mean() if any(k.split('/')[0] in RETRIEVAL_DATASETS and k.endswith('training_loss_hard') for k in train_log_dict.keys()) else torch.tensor(0.0)
                train_log_dict['Avg/classification/training_loss_hard'] = torch.tensor([v for k, v in train_log_dict.items() if k.split('/')[0] in CLASSIFICATION_DATASETS]).mean() if any(k.split('/')[0] in CLASSIFICATION_DATASETS for k in train_log_dict.keys()) else torch.tensor(0.0)
                train_log_dict['Avg/clustering/training_loss_hard'] = torch.tensor([v for k, v in train_log_dict.items() if k.split('/')[0] in CLUSTERING_DATASETS]).mean() if any(k.split('/')[0] in CLUSTERING_DATASETS for k in train_log_dict.keys()) else torch.tensor(0.0)

                distributed_ctx.print(f"[Train] Step = {completed_steps}")
                if distributed_ctx.is_main_process():
                    write_tensorboard(summary_writer, train_log_dict, completed_steps)

                # Reset counters
                loss_dict = {ds_name: torch.tensor(0.0, device=model.lm.device) for ds_name in RETRIEVAL_DATASETS}
                loss_hard_dict = {ds_name: torch.tensor(0.0, device=model.lm.device) for ds_name in train_dataloader.loader_dict.keys()}
                count_dict = {ds_name: torch.tensor(0, device=model.lm.device) for ds_name in RETRIEVAL_DATASETS}
                count_hard_dict = {ds_name: torch.tensor(0, device=model.lm.device) for ds_name in train_dataloader.loader_dict.keys()}

            # Validation
            if completed_steps % args.validation_steps == 0:
                model.lm.eval()
                validate(args, distributed_ctx, model, valid_loaders, criterion, completed_steps, summary_writer)
                model.lm.train()

            # Checkpoint saving
            if args.checkpointing_steps and completed_steps % args.checkpointing_steps == 0:
                output_dir = os.path.join(args.output_dir, f"step_{completed_steps}")
                save_checkpoint(args, distributed_ctx, model, output_dir, lr_scheduler)

                # Report checkpoint to Ray Train
                if distributed_ctx.is_main_process():
                    ray.train.report(
                        metrics={"step": completed_steps},
                        checkpoint=ray.train.Checkpoint.from_directory(output_dir)
                    )

            if completed_steps >= args.train_steps:
                break

        # Epoch checkpoint
        output_dir = os.path.join(args.output_dir, f"epoch_{epoch+1}")
        save_checkpoint(args, distributed_ctx, model, output_dir, lr_scheduler)

        if completed_steps % args.validation_steps != 0:
            model.lm.eval()
            validate(args, distributed_ctx, model, valid_loaders, criterion, completed_steps, summary_writer)
            model.lm.train()

    if summary_writer:
        summary_writer.close()

    distributed_ctx.print("Training completed!")


class RayF2LLMTrainer:
    """Ray Train trainer wrapper for F2LLM"""

    def __init__(self, config: RayTrainConfig):
        """
        Initialize Ray trainer

        Args:
            config: RayTrainConfig instance
        """
        self.config = config

        # Create ScalingConfig
        scaling_config = ScalingConfig(
            num_workers=config.num_workers,
            use_gpu=config.use_gpu,
            resources_per_worker=config.resources_per_worker,
        )

        # Create TorchConfig
        torch_config = TorchConfig(
            backend=config.backend,
        )

        # Create CheckpointConfig
        checkpoint_config = CheckpointConfig(
            num_to_keep=config.checkpoint_num_to_keep,
            checkpoint_score_attribute=config.checkpoint_score_attribute,
            checkpoint_score_order=config.checkpoint_score_order,
        )

        # Create FailureConfig if fault tolerance enabled
        failure_config = None
        if config.enable_fault_tolerance:
            failure_config = FailureConfig(max_failures=config.max_retries)

        # Create RunConfig
        run_config = RunConfig(
            name=config.experiment_id,
            storage_path=os.path.dirname(config.output_dir),
            checkpoint_config=checkpoint_config,
            failure_config=failure_config,
        )

        # Create TorchTrainer
        self.trainer = TorchTrainer(
            train_loop_per_worker=train_func,
            train_loop_config=config.dict(),
            scaling_config=scaling_config,
            torch_config=torch_config,
            run_config=run_config,
        )

    def fit(self):
        """Start training"""
        result = self.trainer.fit()
        return result


if __name__ == "__main__":
    # Example usage for testing
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to ray_config.yaml")
    args = parser.parse_args()

    config = RayTrainConfig.from_yaml(args.config)
    trainer = RayF2LLMTrainer(config)
    result = trainer.fit()
    print(f"Training completed: {result}")
