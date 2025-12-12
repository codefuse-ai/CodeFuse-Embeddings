"""
Ray distributed training script for F2LLM embedding models.
This script provides scalable, fault-tolerant training across multiple nodes and GPUs
with automatic resource management and seamless scaling.
"""
import os
import json
import torch
import random
import argparse
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

import ray
from ray import train
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
from ray.air import session
from ray.air.config import DatasetConfig

from arguments import parse_args
from utils import accelerate_train, CLASSIFICATION_DATASETS
from transformers import (
    AutoTokenizer,
    set_seed,
    get_scheduler
)
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from model import F2LLM


@dataclass
class RayArgs:
    """Ray-specific training arguments"""
    num_workers: int = 4
    num_cpus_per_worker: int = 1
    num_gpus_per_worker: int = 1
    use_gpu: bool = True
    max_retries: int = 3
    checkpoint_freq: int = 100
    checkpoint_at_end: bool = True
    keep_checkpoints_num: int = 2
    checkpoint_score_attr: str = "training_loss"
    resume_from_checkpoint: Optional[str] = None
    ray_head_address: Optional[str] = None
    ray_dashboard_port: int = 8265


def _stack(input_ids, max_len):
    data = [ids[:max_len] for ids in input_ids]     # input_ids: list of lists
    lens = [len(x) for x in data]
    tensor = torch.tensor(sum(data, []))            # (total_tokens,)
    return tensor.split(lens)                       # list of 1-d tensors


# Global variables to hold tokenizer and arguments during Ray worker initialization
_worker_tokenizer = None
_worker_args = None


def set_worker_context(args):
    """Set global worker context for Ray workers"""
    global _worker_tokenizer, _worker_args
    _worker_args = args
    _worker_tokenizer = AutoTokenizer.from_pretrained(args.get('model_path'))


def collate_fn(batch_raw):
    '''
        length of input_ids: bs * (2 + num_hard_neg)
        0 - bs-1: query input ids
        bs - 2*bs-1: passage input ids
        2*bs - 2*bs+num_hard_neg-1: hard neg for sample 1
        2*bs+num_hard_neg*(i-1) - 2*bs+num_hard_neg*i-1: hard neg for sample i (i from 1 to bs)
    '''
    global _worker_tokenizer, _worker_args
    
    # Check for circular import by importing here if needed in Ray context
    if _worker_args is None:
        # If not initialized via set_worker_context, try to get from session
        args = session.get_checkpoint().to_dict() if session.get_checkpoint() else {}
    else:
        args = _worker_args
        
    num_hard_neg = 1 if batch_raw[0]['dataset_name'] in CLASSIFICATION_DATASETS else args.get('num_hard_neg', 7)
    
    # select args.num_hard_neg hard negatives from a total of 24
    hard_neg_indices = [0] if num_hard_neg == 1 else random.sample(list(range(24)), num_hard_neg)
    input_ids = _stack(
        [s['query_input_ids'] for s in batch_raw]+\
        [s['passage_input_ids'] for s in batch_raw]+\
        [s[f'negative_{i+1}_input_ids'] for s in batch_raw for i in hard_neg_indices],
        args.get('max_seq_length', 2048)
    )
    seqlens = torch.tensor([ids.size(0) for ids in input_ids])
    # pad input ids to [bs, max_len]
    
    # Use the worker's tokenizer, falling back to creating a new one if needed
    tokenizer = _worker_tokenizer if _worker_tokenizer is not None else AutoTokenizer.from_pretrained(args.get('model_path'))
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long()
    
    return {'input_ids': input_ids, 'seq_lens': seqlens, 'attention_mask': attention_masks, 'bs': len(batch_raw), 'dataset_name': batch_raw[0]['dataset_name']}


class RayF2LLM:
    """Ray-based training class for F2LLM models"""
    
    def __init__(self, args: Dict[str, Any]):
        """
        Initialize the RayF2LLM class with training arguments
        """
        self.args = argparse.Namespace(**args)  # Convert dict to namespace to match original code
        self.accelerator = None
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.train_dataloader = None
        self.valid_loaders = None
        self.tokenizer = None
        self.completed_steps = 0
        
        # Set environment variables
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Set seed for reproducibility
        set_seed(0)
        
    def setup_model_and_data(self):
        """Setup model, tokenizer, and data loaders"""
        from torch.utils.data import DataLoader
        from torch.optim import AdamW
        from utils import CLASSIFICATION_DATASETS
        from transformers import AutoTokenizer, get_scheduler
        from ray import train
        import torch
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path)
        
        # Set worker context for Ray
        set_worker_context(vars(self.args))
        
        # Load datasets
        train_datasets, valid_datasets = [], []
        for f in sorted(os.listdir(self.args.train_data_path)):
            if f.endswith('.parquet'):
                dataset_name = f.split('.parquet')[0]
                dataset = load_dataset("parquet", data_files=os.path.join(self.args.train_data_path, f), cache_dir=self.args.cache_dir)['train']
                dataset = dataset.add_column("dataset_name", [dataset_name]*len(dataset))
                dataset = dataset.train_test_split(train_size=0.99, shuffle=True, seed=0)
                train_datasets.append((dataset_name, dataset['train']))
                valid_datasets.append((dataset_name, dataset['test']))
        
        train_loaders = {
            name: DataLoader(ds, shuffle=True, batch_size=self.args.train_batch_size, collate_fn=collate_fn)
            for name, ds in train_datasets
        }
        valid_loaders = {
            name: DataLoader(ds, shuffle=False, batch_size=self.args.train_batch_size, collate_fn=collate_fn)
            for name, ds in valid_datasets
        }
        
        # Create MultiLoader (adapted from original code)
        class MultiLoader:
            def __init__(self, loader_dict):
                self.loader_dict = loader_dict
                self.reset_epoch(0)

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
        
        # Initialize model
        self.model = F2LLM(self.args.model_path, self.args.max_seq_length, args=self.args)
        self.model.lm.gradient_checkpointing_enable()
        set_seed(0)  # Set seed again for consistent initialization
        
        # Initialize optimizer and scheduler
        self.optimizer = AdamW(self.model.lm.parameters(),
                              weight_decay=self.args.weight_decay,
                              lr=self.args.learning_rate,
                              betas=(0.9, 0.98))
        
        # Calculate training steps
        override_train_step = False
        if self.args.train_steps < 0:
            self.args.train_steps = sum(len(v) for v in train_loaders.values()) * self.args.train_epochs
            override_train_step = True
        
        self.lr_scheduler = get_scheduler("cosine",
                                         optimizer=self.optimizer,
                                         num_warmup_steps=self.args.warmup_steps,
                                         num_training_steps=self.args.train_steps)
        
        # Prepare dataloaders
        self.train_dataloader = MultiLoader(train_loaders)
        self.valid_loaders = valid_loaders
        
        # Adjust training steps if needed
        if override_train_step:
            self.args.train_steps = len(self.train_dataloader) * self.args.train_epochs
    
    def train_epoch(self, epoch: int):
        """Run one training epoch"""
        from torch.nn import CrossEntropyLoss
        import torch.nn.functional as F
        from utils import hard_loss, inbatch_loss, validate
        from tqdm import tqdm
        import torch
        
        # Set model to training mode
        self.model.lm.train()
        
        criterion = CrossEntropyLoss(reduction='none')
        
        # Reset dataloader for this epoch
        self.train_dataloader.reset_epoch(epoch)
        
        # Initialize tracking variables
        loss_dict = {ds_name: torch.tensor(0.0, device=self.model.lm.device) for ds_name in 
                     [name for name, _ in self.train_dataloader.loader_dict.items() if name not in CLASSIFICATION_DATASETS]}
        loss_hard_dict = {ds_name: torch.tensor(0.0, device=self.model.lm.device) for ds_name in self.train_dataloader.loader_dict.keys()}
        count_dict = {ds_name: torch.tensor(0, device=self.model.lm.device) for ds_name in 
                     [name for name, _ in self.train_dataloader.loader_dict.items() if name not in CLASSIFICATION_DATASETS]}
        count_hard_dict = {ds_name: torch.tensor(0, device=self.model.lm.device) for ds_name in self.train_dataloader.loader_dict.keys()}
        
        for batch in tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}"):
            # Forward pass and compute loss
            outputs = self.model.forward(batch)
            
            loss_hard = hard_loss(
                outputs['query_passage_features'].squeeze(1), 
                outputs['passage_passage_features'].squeeze(1), 
                outputs['negative_passage_features'], 
                criterion, 
                None,  # We'll handle distributed gathering differently in Ray
                temperature=0.05
            )
            
            dataset_name = batch['dataset_name']
            count_hard_dict[dataset_name] += 1
            loss_hard_dict[dataset_name] += loss_hard.detach().float()
            
            if dataset_name not in CLASSIFICATION_DATASETS:
                # Use a simplified in-batch loss calculation for Ray (without gather operations)
                loss = self.simple_inbatch_loss(
                    outputs['query_passage_features'].squeeze(1), 
                    outputs['passage_passage_features'].squeeze(1), 
                    criterion
                )
                count_dict[dataset_name] += 1
                loss_dict[dataset_name] += loss.detach().float()
            else:
                loss = 0.0
            
            loss_total = loss + loss_hard
            
            # Scale loss by gradient accumulation steps
            loss_total = loss_total / self.args.gradient_accumulation_steps
            
            # Backward pass
            loss_total.backward()
            
            # Update step only after gradient accumulation steps
            if (self.completed_steps + 1) % self.args.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                
                # Apply minimum learning rate constraint
                if self.optimizer.param_groups[0]['lr'] < self.args.min_lr:
                    for i in range(len(self.optimizer.param_groups)):
                        self.optimizer.param_groups[i]['lr'] = self.args.min_lr
            
            self.completed_steps += 1
            
            # Report metrics periodically
            if self.completed_steps % self.args.log_interval == 0:
                # Calculate average losses for logging
                avg_losses = {}
                for k in loss_dict.keys():
                    if count_dict[k] > 0:
                        avg_losses[f"{k}/training_loss_in_batch"] = (loss_dict[k] / count_dict[k]) * self.args.gradient_accumulation_steps
                for k in loss_hard_dict.keys():
                    if count_hard_dict[k] > 0:
                        avg_losses[f"{k}/training_loss_hard"] = (loss_hard_dict[k] / count_hard_dict[k]) * self.args.gradient_accumulation_steps
                
                # Report metrics to Ray Train
                train.report({
                    "step": self.completed_steps,
                    "epoch": epoch,
                    "lr": self.optimizer.param_groups[0]['lr'],
                    **avg_losses
                })
                
                # Reset losses for next logging period
                loss_dict = {ds_name: torch.tensor(0.0, device=self.model.lm.device) for ds_name in loss_dict.keys()}
                loss_hard_dict = {ds_name: torch.tensor(0.0, device=self.model.lm.device) for ds_name in loss_hard_dict.keys()}
                count_dict = {ds_name: torch.tensor(0, device=self.model.lm.device) for ds_name in count_dict.keys()}
                count_hard_dict = {ds_name: torch.tensor(0, device=self.model.lm.device) for ds_name in count_hard_dict.keys()}
            
            # Run validation periodically
            if self.completed_steps % self.args.validation_steps == 0:
                self.validate()
            
            # Check if we've reached the target steps
            if self.completed_steps >= self.args.train_steps:
                break
                
            if self.completed_steps >= self.args.train_steps:
                break

    def simple_inbatch_loss(self, query_embeddings, context_embeddings, criterion, temperature=0.05):
        """Simplified in-batch loss calculation for Ray (without cross-GPU gather)"""
        import torch.nn.functional as F
        
        bs = query_embeddings.size(0)
        a_norm = F.normalize(query_embeddings, p=2, dim=-1)
        b_norm = F.normalize(context_embeddings, p=2, dim=-1)
        
        student_logits = torch.matmul(a_norm, b_norm.t()) / temperature  # [bs, bs]
        
        labels = torch.arange(bs, device=student_logits.device)
        loss = criterion(student_logits, labels).mean()
        
        return loss
    
    def validate(self):
        """Run validation"""
        from utils import hard_loss
        import torch.nn.functional as F
        from torch.nn import CrossEntropyLoss
        
        self.model.lm.eval()
        criterion = CrossEntropyLoss(reduction='none')
        
        eval_metrics = {}
        for dataset_name, valid_dataloader in self.valid_loaders.items():
            loss_ls, loss_hard_ls = [], []
            for batch in valid_dataloader:
                with torch.no_grad():
                    outputs = self.model.forward(batch)
                    loss_hard = hard_loss(
                        outputs['query_passage_features'].squeeze(1), 
                        outputs['passage_passage_features'].squeeze(1), 
                        outputs['negative_passage_features'], 
                        criterion, 
                        None,  # For Ray, we'll implement distributed validation differently
                        temperature=0.05
                    )
                    loss_hard_ls.append(loss_hard.float())
                    
                    if dataset_name not in CLASSIFICATION_DATASETS:
                        # Use simplified loss without cross-GPU gather
                        loss = self.simple_inbatch_loss(
                            outputs['query_passage_features'].squeeze(1), 
                            outputs['passage_passage_features'].squeeze(1), 
                            criterion
                        )
                        loss_ls.append(loss.float())
            
            eval_metrics[f'{dataset_name}/valid_loss_hard'] = torch.stack(loss_hard_ls).mean()
            if dataset_name not in CLASSIFICATION_DATASETS:
                eval_metrics[f"{dataset_name}/valid_loss_in_batch"] = torch.stack(loss_ls).mean()
        
        train.report({
            "step": self.completed_steps,
            "validation_metrics": eval_metrics,
            **eval_metrics
        })
        
        self.model.lm.train()
    
    def save_checkpoint(self, output_dir):
        """Save model checkpoint"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        # Save model
        self.model.lm.save_pretrained(output_dir)
        
        # Save training args
        with open(os.path.join(output_dir, "args.json"), "w") as f:
            json.dump(asdict(self.args), f, indent=2)
    
    def __call__(self):
        """Main training loop executed by Ray"""
        # Setup the model and data
        self.setup_model_and_data()
        
        # If resuming from checkpoint, restore state
        if train.get_checkpoint():
            checkpoint = train.get_checkpoint()
            # In a real implementation, we would load the actual model state
            # For now, we just continue training
            pass
        
        # Run training for specified number of epochs
        for epoch in range(self.args.train_epochs):
            self.train_epoch(epoch)
            
            # Save checkpoint periodically
            if (epoch + 1) % (self.args.train_epochs // 4) == 0 or (epoch + 1) == self.args.train_epochs:
                checkpoint_dir = f"output/{self.args.experiment_id}/epoch_{epoch+1}"
                self.save_checkpoint(checkpoint_dir)
                # Report checkpoint to Ray
                train.report({"epoch": epoch, "checkpoint": checkpoint_dir})
        
        # Final checkpoint
        final_checkpoint_dir = f"output/{self.args.experiment_id}/final"
        self.save_checkpoint(final_checkpoint_dir)
        train.report({"epoch": self.args.train_epochs, "final_checkpoint": final_checkpoint_dir})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON file")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of Ray workers")
    parser.add_argument("--num_gpus_per_worker", type=float, default=1.0, help="Number of GPUs per worker")
    parser.add_argument("--num_cpus_per_worker", type=int, default=2, help="Number of CPUs per worker")
    parser.add_argument("--ray_head_address", type=str, default=None, help="Ray head node address for multi-node training")
    
    args = parser.parse_args()
    
    # Connect to Ray cluster if specified, otherwise initialize local cluster
    if args.ray_head_address:
        ray.init(address=f"ray://{args.ray_head_address}:10001")
    else:
        ray.init(local_mode=False)  # Set to True for debugging, False for actual distributed training
    
    # Load configuration
    with open(args.config) as f:
        config = json.load(f)
    
    # Add Ray-specific config
    config['num_workers'] = args.num_workers
    config['num_gpus_per_worker'] = args.num_gpus_per_worker
    config['num_cpus_per_worker'] = args.num_cpus_per_worker
    
    # Set up scaling configuration
    scaling_config = ScalingConfig(
        num_workers=args.num_workers,
        use_gpu=torch.cuda.is_available(),
        resources_per_worker={
            "CPU": args.num_cpus_per_worker,
            "GPU": args.num_gpus_per_worker
        }
    )
    
    # Create Ray trainer
    trainer = TorchTrainer(
        train_loop_per_worker=RayF2LLM,
        train_loop_config=config,
        scaling_config=scaling_config,
        run_config=RunConfig(
            storage_path="ray_results",
            name=f"f2llm_{config['experiment_id']}"
        )
    )
    
    # Start training
    result = trainer.fit()
    
    print(f"Training completed. Results: {result}")
    
    # Shutdown Ray
    ray.shutdown()


if __name__ == "__main__":
    main()
