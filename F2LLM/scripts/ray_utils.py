from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import os

# Set environment variables to suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_DISTRIBUTED_ELASTIC_LOG_LEVEL"] = "ERROR"

# Import dataset constants from the original utils
from utils import CLASSIFICATION_DATASETS, CLUSTERING_DATASETS, RETRIEVAL_DATASETS


def write_tensorboard(summary_writer: SummaryWriter, log_dict: dict, completed_steps):
    for key, value in log_dict.items():
        summary_writer.add_scalar(key, value, completed_steps)


def save_checkpoint(args, accelerator, model, output_dir, lr_scheduler):
    # For Ray, we don't have accelerator, so we need to adapt this function
    print(f"Saving checkpoint to {output_dir}")
    
    # Save tokenizer
    model.tokenizer.save_pretrained(output_dir)
    
    # Save model
    model.lm.save_pretrained(output_dir)
    
    print(f"Checkpoint saved to {output_dir}")


def load_checkpoint(checkpoint_dir, model, optimizer, lr_scheduler):
    """Load checkpoint from directory"""
    print(f"Loading checkpoint from {checkpoint_dir}")
    
    # Load model
    model.lm = model.lm.from_pretrained(checkpoint_dir)
    
    print(f"Checkpoint loaded from {checkpoint_dir}")
    return model


def save_ray_checkpoint(args, model, optimizer, lr_scheduler, checkpoint_dir):
    """Save checkpoint using Ray's checkpointing mechanism"""
    import ray.train
    import os
    import tempfile
    import ray.cloudpickle as pickle
    
    # Create a temporary directory for the checkpoint
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save model state dict
        model_path = os.path.join(temp_dir, 'model.pth')
        torch.save(model.lm.state_dict(), model_path)
        
        # Save optimizer state dict
        optimizer_path = os.path.join(temp_dir, 'optimizer.pth')
        torch.save(optimizer.state_dict(), optimizer_path)
        
        # Save learning rate scheduler state dict
        lr_scheduler_path = os.path.join(temp_dir, 'lr_scheduler.pth')
        torch.save(lr_scheduler.state_dict(), lr_scheduler_path)
        
        # Save args
        args_path = os.path.join(temp_dir, 'args.pkl')
        with open(args_path, 'wb') as f:
            pickle.dump(args, f)
        
        # Create Ray checkpoint from directory
        from ray.train import Checkpoint
        checkpoint = Checkpoint.from_directory(temp_dir)
        
        # Report checkpoint to Ray
        ray.train.report({}, checkpoint=checkpoint)
    
    print(f"Ray checkpoint saved using directory method")
    return checkpoint


def load_ray_checkpoint(checkpoint_dir):
    """Load checkpoint using Ray's checkpointing mechanism"""
    import ray.train
    import os
    import ray.cloudpickle as pickle
    import torch
    
    # Load checkpoint from directory
    checkpoint = ray.train.Checkpoint.from_directory(checkpoint_dir)
    
    # Get the checkpoint directory
    with checkpoint.as_directory() as checkpoint_path:
        # Load model state dict
        model_path = os.path.join(checkpoint_path, 'model.pth')
        if os.path.exists(model_path):
            model_state_dict = torch.load(model_path)
        else:
            model_state_dict = None
        
        # Load optimizer state dict
        optimizer_path = os.path.join(checkpoint_path, 'optimizer.pth')
        if os.path.exists(optimizer_path):
            optimizer_state_dict = torch.load(optimizer_path)
        else:
            optimizer_state_dict = None
        
        # Load learning rate scheduler state dict
        lr_scheduler_path = os.path.join(checkpoint_path, 'lr_scheduler.pth')
        if os.path.exists(lr_scheduler_path):
            lr_scheduler_state_dict = torch.load(lr_scheduler_path)
        else:
            lr_scheduler_state_dict = None
        
        # Load args
        args_path = os.path.join(checkpoint_path, 'args.pkl')
        if os.path.exists(args_path):
            with open(args_path, 'rb') as f:
                args = pickle.load(f)
        else:
            args = None
    
    checkpoint_data = {
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "lr_scheduler_state_dict": lr_scheduler_state_dict,
        "args": args
    }
    
    print(f"Ray checkpoint loaded from {checkpoint_dir}")
    return checkpoint_data


def inbatch_loss(
        query_embeddings, # [bs, d]
        context_embeddings, # [bs, d]
        criterion,
        accelerator=None,  # For Ray compatibility
        temperature=0.05,
    ):
    
    bs = query_embeddings.size(0)
    a_norm = F.normalize(query_embeddings, p=2, dim=-1)
    
    # For Ray, we don't gather across GPUs, so we use context_embeddings directly
    b_norm = F.normalize(context_embeddings, p=2, dim=-1)

    student_logits = torch.matmul(a_norm, b_norm.t()) / temperature # [bs, bs]

    labels = torch.arange(bs, device=student_logits.device)
    loss_bs = criterion(student_logits, labels) # (bs)

    loss = loss_bs.mean()

    return loss

def hard_loss(
        query_embeddings, # [bs, d]
        context_embeddings, # [bs, d]
        hard_neg_embeddings, # [bs, num, d]
        criterion,
        accelerator=None,  # For Ray compatibility
        temperature=0.05,
    ):

    if hard_neg_embeddings is None:
        return 0.0

    bs = query_embeddings.size(0)
    a_norm = F.normalize(query_embeddings, p=2, dim=-1)

    hard_neg_embeddings = torch.concat([
        context_embeddings.unsqueeze(1),
        hard_neg_embeddings
    ], dim=1) # [bs, num_hard+1, d]
    
    hard_norm = F.normalize(hard_neg_embeddings, p=2, dim=-1)
    logits = (a_norm.unsqueeze(1) * hard_norm).sum(-1) / temperature # [bs, num_hard+1]

    loss_hard = criterion(logits, torch.zeros((bs), dtype=torch.long, device=logits.device)).mean()

    return loss_hard


def validate(args, model, valid_loader_dict, criterion, completed_steps, summary_writer):
    eval_log_dict = {}
    for dataset_name, valid_dataloader in valid_loader_dict.items():
        loss_ls, loss_hard_ls = [], []
        for batch in valid_dataloader:
            with torch.no_grad():
                outputs = model.forward(batch)
                loss_hard = hard_loss(outputs['query_passage_features'].squeeze(1), outputs['passage_passage_features'].squeeze(1), outputs['negative_passage_features'], criterion)
                loss_hard_ls.append(loss_hard.float())
                if dataset_name in RETRIEVAL_DATASETS:
                    loss = inbatch_loss(outputs['query_passage_features'].squeeze(1), outputs['passage_passage_features'].squeeze(1), criterion)
                    loss_ls.append(loss.float())
        
        loss_hard_ls = torch.stack(loss_hard_ls)
        eval_log_dict[f'{dataset_name}/valid_loss_hard'] = loss_hard_ls.mean()
        if dataset_name in RETRIEVAL_DATASETS:
            loss_ls = torch.stack(loss_ls)
            eval_log_dict[f"{dataset_name}/valid_loss_in_batch"] = loss_ls.mean()
    
    eval_log_dict['Avg/retrieval/valid_loss_in_batch'] = torch.tensor([v for k, v in eval_log_dict.items() if k.split('/')[0] in RETRIEVAL_DATASETS and k.endswith('valid_loss_in_batch')]).mean()
    eval_log_dict['Avg/retrieval/valid_loss_hard'] = torch.tensor([v for k, v in eval_log_dict.items() if k.split('/')[0] in RETRIEVAL_DATASETS and k.endswith('valid_loss_hard')]).mean()
    eval_log_dict['Avg/classification/valid_loss_hard'] = torch.tensor([v for k, v in eval_log_dict.items() if k.split('/')[0] in CLASSIFICATION_DATASETS]).mean()
    eval_log_dict['Avg/clustering/valid_loss_hard'] = torch.tensor([v for k, v in eval_log_dict.items() if k.split('/')[0] in CLUSTERING_DATASETS]).mean()
    
    # In Ray, every worker can write to tensorboard
    write_tensorboard(summary_writer, eval_log_dict, completed_steps)
    print(f"[Validation] Step = {completed_steps}")
        

def ray_train(args,
              model, 
              train_dataloader,
              valid_loader_dict,
              optimizer,
              lr_scheduler,
              num_train_samples):
    print("**************************************** Start training ****************************************")
    print(f" Num train samples = {num_train_samples}")
    print(f" Num epochs = {args.train_epochs}")
    print(f" Per device batch size = {args.train_batch_size}")
    
    # Calculate global batch size
    import ray
    from ray.train import get_context
    train_context = get_context()
    local_world_size = train_context.get_local_world_size()
    global_batch_size = args.train_batch_size * local_world_size
    
    print(f" Global batch size = {global_batch_size}")
    print(f" Step per epoch = {len(train_dataloader)}")
    print(f" Total training steps = {args.train_steps}")
    print("************************************************************************************************")
    
    # Filter datasets based on what's available in the dataloader
    global RETRIEVAL_DATASETS, CLASSIFICATION_DATASETS, CLUSTERING_DATASETS
    RETRIEVAL_DATASETS = [ds for ds in RETRIEVAL_DATASETS if ds in train_dataloader.loader_dict.keys()]
    CLASSIFICATION_DATASETS = [ds for ds in CLASSIFICATION_DATASETS if ds in train_dataloader.loader_dict.keys()]
    CLUSTERING_DATASETS = [ds for ds in CLUSTERING_DATASETS if ds in train_dataloader.loader_dict.keys()]

    summary_writer = SummaryWriter(log_dir=args.tb_dir)
    criterion = CrossEntropyLoss(reduction='none')
    pbar = tqdm(range(args.train_steps))
    completed_steps = 0
    loss_dict = {ds_name: torch.tensor(0.0) for ds_name in RETRIEVAL_DATASETS}
    loss_hard_dict = {ds_name: torch.tensor(0.0) for ds_name in train_dataloader.loader_dict.keys()}
    count_dict = {ds_name: torch.tensor(0) for ds_name in RETRIEVAL_DATASETS}
    count_hard_dict = {ds_name: torch.tensor(0) for ds_name in train_dataloader.loader_dict.keys()}

    model.lm.train()
    for epoch in range(args.train_epochs):
        print(f"*************** Starting epoch {epoch+1} ***************")
        train_dataloader.reset_epoch(epoch)
        for batch in train_dataloader:
            # forward and compute loss
            outputs = model.forward(batch)
            # passage features: [bs, 1, d]
            # hard_neg_features: [bs, num_hard_neg, d]

            loss_hard = hard_loss(outputs['query_passage_features'].squeeze(1), outputs['passage_passage_features'].squeeze(1), outputs['negative_passage_features'], criterion)
            dataset_name = batch['dataset_name']
            count_hard_dict[dataset_name] += 1
            loss_hard_dict[dataset_name] += loss_hard.detach().float()
            if dataset_name in RETRIEVAL_DATASETS:
                loss = inbatch_loss(outputs['query_passage_features'].squeeze(1), outputs['passage_passage_features'].squeeze(1), criterion)
                count_dict[dataset_name] += 1
                loss_dict[dataset_name] += loss.detach().float()
            else:
                loss = 0.0
            
            loss_total = loss + loss_hard

            # backward, optimizer, scheduler
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            lr_scheduler.step()
            
            if optimizer.param_groups[0]['lr'] < args.min_lr:
                for i in range(len(optimizer.param_groups)):
                    optimizer.param_groups[i]['lr'] = args.min_lr
            
            # log
            completed_steps += 1
            if completed_steps % args.log_interval == 0:
                pbar.update(min(args.log_interval, args.train_steps - pbar.n))

                train_log_dict = {"lr": optimizer.param_groups[0]['lr']}
                for k in loss_dict.keys():
                    count = count_dict[k]
                    if count > 0:
                        train_log_dict[f"{k}/training_loss_in_batch"] = loss_dict[k] / count
                for k in loss_hard_dict.keys():
                    count = count_hard_dict[k]
                    if count > 0:
                        train_log_dict[f"{k}/training_loss_hard"] = loss_hard_dict[k] / count
                train_log_dict['Avg/retrieval/training_loss_in_batch'] = torch.tensor([v for k, v in train_log_dict.items() if k.split('/')[0] in RETRIEVAL_DATASETS and k.endswith('training_loss_in_batch')]).mean()
                train_log_dict['Avg/retrieval/training_loss_hard'] = torch.tensor([v for k, v in train_log_dict.items() if k.split('/')[0] in RETRIEVAL_DATASETS and k.endswith('training_loss_hard')]).mean()
                train_log_dict['Avg/classification/training_loss_hard'] = torch.tensor([v for k, v in train_log_dict.items() if k.split('/')[0] in CLASSIFICATION_DATASETS]).mean()
                train_log_dict['Avg/clustering/training_loss_hard'] = torch.tensor([v for k, v in train_log_dict.items() if k.split('/')[0] in CLUSTERING_DATASETS]).mean()

                print(f"[Train] Step = {completed_steps}")
                write_tensorboard(summary_writer, train_log_dict, completed_steps)
                loss_dict = {ds_name: torch.tensor(0.0) for ds_name in RETRIEVAL_DATASETS}
                loss_hard_dict = {ds_name: torch.tensor(0.0) for ds_name in train_dataloader.loader_dict.keys()}
                count_dict = {ds_name: torch.tensor(0) for ds_name in RETRIEVAL_DATASETS}
                count_hard_dict = {ds_name: torch.tensor(0) for ds_name in train_dataloader.loader_dict.keys()}

            # validation
            if completed_steps % args.validation_steps == 0:
                model.lm.eval()
                validate(args, model, valid_loader_dict, criterion, completed_steps, summary_writer)
                model.lm.train()

            # step checkpoint
            if args.checkpointing_steps and completed_steps % args.checkpointing_steps == 0:
                output_dir = os.path.join(args.output_dir, f"step_{completed_steps}")
                save_checkpoint(args, None, model, output_dir, lr_scheduler)
                
                # Save Ray checkpoint
                try:
                    save_ray_checkpoint(args, model, optimizer, lr_scheduler, output_dir)
                except Exception as e:
                    print(f"Failed to save Ray checkpoint: {e}")

            if completed_steps >= args.train_steps:
                break

        # epoch checkpoint
        output_dir = os.path.join(args.output_dir, f"epoch_{epoch+1}")
        save_checkpoint(args, None, model, output_dir, lr_scheduler)
        
        # Save Ray checkpoint
        try:
            save_ray_checkpoint(args, model, optimizer, lr_scheduler, output_dir)
        except Exception as e:
            print(f"Failed to save Ray checkpoint: {e}")
            
        if completed_steps % args.validation_steps != 0:
            model.lm.eval()
            validate(args, model, valid_loader_dict, criterion, completed_steps, summary_writer)
            model.lm.train()
    
    # Ensure progress bar is updated to completion
    if pbar.n < args.train_steps:
        pbar.update(args.train_steps - pbar.n)
    pbar.close()
    
    summary_writer.close()
