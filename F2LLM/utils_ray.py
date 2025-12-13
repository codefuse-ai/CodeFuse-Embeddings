from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import os
import torch.distributed as dist
from ray import train

CLASSIFICATION_DATASETS = ['amazon_counterfactual', 'amazon_polarity', 'imdb', 'toxic_conversations', 'cola']
CLUSTERING_DATASETS = ['amazon_reviews', 'banking77', 'emotion', 'mtop_intent', 'mtop_domain', 'massive_scenario', 'massive_intent', 'tweet_sentiment_extraction', 'arxiv_clustering_p2p', 'arxiv_clustering_s2s', 'biorxiv_clustering_p2p', 'biorxiv_clustering_s2s', 'medrxiv_clustering_p2p', 'medrxiv_clustering_s2s', 'reddit_clustering_p2p', 'reddit_clustering_s2s', 'stackexchange_clustering_p2p', 'stackexchange_clustering_s2s', 'twentynewsgroups']
RETRIEVAL_DATASETS = ['arguana', 'snli', 'mnli', 'anli', 'paq', 'squad', 'stackexchange', 'msmarco', 'natural_questions', 'hotpotqa', 'fever', 'eli5', 'fiqa', 'bioasq', 'nfcorpus', 'miracl', 'mrtidy', 'scifact', 'qqp', 'stackoverflowdupquestions', 'sts12', 'sts22', 'stsbenchmark', 'amazon_qa', 'cnn_dm', 'coliee', 'paq_part2', 'pubmedqa', 's2orc_abstract_citation', 's2orc_title_abstract', 's2orc_title_citation', 'sentence_compression', 'specter', 'triviaqa', 'xsum', 'stackexchange_part2', 'stackexchangedupquestions_s2s', 'stackexchangedupquestions_p2p']


def write_tensorboard(summary_writer: SummaryWriter, log_dict: dict, completed_steps):
    for key, value in log_dict.items():
        summary_writer.add_scalar(key, value, completed_steps)


def save_checkpoint(args, model, output_dir, lr_scheduler, world_rank):
    """Save checkpoint using Ray Train"""
    # Wait for all processes
    if dist.is_initialized():
        dist.barrier()
    
    if world_rank == 0:
        print(f"Saving checkpoint to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    
    if world_rank == 0:
        model.tokenizer.save_pretrained(output_dir)
    
    # Check if using DeepSpeed
    if hasattr(model.lm, 'save_checkpoint'):
        # DeepSpeed model
        model.lm.save_checkpoint(output_dir)
    else:
        # Unwrap DDP model
        unwrapped_model = model.lm.module if hasattr(model.lm, 'module') else model.lm
        
        if world_rank == 0:
            unwrapped_model.save_pretrained(output_dir)
    
    # Wait for all processes
    if dist.is_initialized():
        dist.barrier()


def all_gather_tensor(tensor, world_size):
    """Gather tensors from all processes"""
    if not dist.is_initialized() or world_size == 1:
        return tensor
    
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    return torch.cat(tensor_list, dim=0)


def inbatch_loss(
        query_embeddings, # [bs, d]
        context_embeddings, # [bs, d]
        criterion,
        world_size,
        world_rank,
        temperature=0.05,
    ):
    
    bs = query_embeddings.size(0)
    a_norm = F.normalize(query_embeddings, p=2, dim=-1)
    
    # Gather context embeddings from all processes
    b_cross_gpus = all_gather_tensor(context_embeddings, world_size) # [bs*world_size, d]
    b_norm_cross_gpus = F.normalize(b_cross_gpus, p=2, dim=-1)

    student_logits = torch.matmul(a_norm, b_norm_cross_gpus.t()) / temperature # [bs, bs*world_size]

    labels = torch.arange(bs, device=student_logits.device) + bs * world_rank
    loss_bs = criterion(student_logits, labels) # (bs)

    loss = loss_bs.mean()

    return loss


def hard_loss(
        query_embeddings, # [bs, d]
        context_embeddings, # [bs, d]
        hard_neg_embeddings, # [bs, num, d]
        criterion,
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


def all_reduce_mean(tensor, world_size):
    """All reduce and compute mean across processes"""
    if not dist.is_initialized() or world_size == 1:
        return tensor
    
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor / world_size


def validate(args, model, valid_loader_dict, criterion, completed_steps, summary_writer, world_size, world_rank):
    eval_log_dict = {}
    for dataset_name, valid_dataloader in valid_loader_dict.items():
        loss_ls, loss_hard_ls = [], []
        for batch in valid_dataloader:
            with torch.no_grad():
                outputs = model.forward(batch)
                loss_hard = hard_loss(outputs['query_passage_features'].squeeze(1), 
                                     outputs['passage_passage_features'].squeeze(1), 
                                     outputs['negative_passage_features'], 
                                     criterion)
                
                # Gather loss from all processes
                loss_hard_gathered = all_gather_tensor(loss_hard.unsqueeze(0), world_size)
                loss_hard_ls.append(loss_hard_gathered.float())
                
                if dataset_name in RETRIEVAL_DATASETS:
                    loss = inbatch_loss(outputs['query_passage_features'].squeeze(1), 
                                       outputs['passage_passage_features'].squeeze(1), 
                                       criterion, world_size, world_rank)
                    loss_gathered = all_gather_tensor(loss.unsqueeze(0), world_size)
                    loss_ls.append(loss_gathered.float())
        
        # Wait for all processes
        if dist.is_initialized():
            dist.barrier()
        
        loss_hard_ls = torch.cat(loss_hard_ls)
        eval_log_dict[f'{dataset_name}/valid_loss_hard'] = loss_hard_ls.mean()
        if dataset_name in RETRIEVAL_DATASETS:
            loss_ls = torch.cat(loss_ls)
            eval_log_dict[f"{dataset_name}/valid_loss_in_batch"] = loss_ls.mean()
    
    eval_log_dict['Avg/retrieval/valid_loss_in_batch'] = torch.tensor([v for k, v in eval_log_dict.items() if k.split('/')[0] in RETRIEVAL_DATASETS and k.endswith('valid_loss_in_batch')]).mean()
    eval_log_dict['Avg/retrieval/valid_loss_hard'] = torch.tensor([v for k, v in eval_log_dict.items() if k.split('/')[0] in RETRIEVAL_DATASETS and k.endswith('valid_loss_hard')]).mean()
    eval_log_dict['Avg/classification/valid_loss_hard'] = torch.tensor([v for k, v in eval_log_dict.items() if k.split('/')[0] in CLASSIFICATION_DATASETS]).mean()
    eval_log_dict['Avg/clustering/valid_loss_hard'] = torch.tensor([v for k, v in eval_log_dict.items() if k.split('/')[0] in CLUSTERING_DATASETS]).mean()
    
    if world_rank == 0:
        write_tensorboard(summary_writer, eval_log_dict, completed_steps)
        print(f"[Validation] Step = {completed_steps}")
    
    # Report metrics to Ray Train
    eval_report_dict = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in eval_log_dict.items()}
    train.report(eval_report_dict)


def ray_train(args,
              model, 
              train_dataloader,
              valid_loader_dict,
              optimizer,
              lr_scheduler,
              num_train_samples,
              world_size,
              world_rank,
              device):
    
    if world_rank == 0:
        print("**************************************** Start training ****************************************")
        print(f" Num train samples = {num_train_samples}")
        print(f" Num epochs = {args.train_epochs}")
        print(f" Per device batch size = {args.train_batch_size}")
        print(f" Global batch size = {args.train_batch_size * world_size}")
        print(f" Step per epoch = {len(train_dataloader)}")
        print(f" Total training steps = {args.train_steps}")
        print("************************************************************************************************")
    
    global RETRIEVAL_DATASETS, CLASSIFICATION_DATASETS, CLUSTERING_DATASETS
    RETRIEVAL_DATASETS = [ds for ds in RETRIEVAL_DATASETS if ds in train_dataloader.loader_dict.keys()]
    CLASSIFICATION_DATASETS = [ds for ds in CLASSIFICATION_DATASETS if ds in train_dataloader.loader_dict.keys()]
    CLUSTERING_DATASETS = [ds for ds in CLUSTERING_DATASETS if ds in train_dataloader.loader_dict.keys()]

    summary_writer = SummaryWriter(log_dir=args.tb_dir) if world_rank == 0 else None
    criterion = CrossEntropyLoss(reduction='none')
    pbar = tqdm(range(args.train_steps), disable=(world_rank != 0))
    completed_steps = 0
    
    # Check if using DeepSpeed
    is_deepspeed = hasattr(model.lm, 'backward')
    
    loss_dict = {ds_name: torch.tensor(0.0, device=device) for ds_name in RETRIEVAL_DATASETS}
    loss_hard_dict = {ds_name: torch.tensor(0.0, device=device) for ds_name in train_dataloader.loader_dict.keys()}
    count_dict = {ds_name: torch.tensor(0, device=device) for ds_name in RETRIEVAL_DATASETS}
    count_hard_dict = {ds_name: torch.tensor(0, device=device) for ds_name in train_dataloader.loader_dict.keys()}

    model.lm.train()
    for epoch in range(args.train_epochs):
        if world_rank == 0:
            print(f"*************** Starting epoch {epoch+1} ***************")
        
        train_dataloader.reset_epoch(epoch)
        for batch in train_dataloader:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # forward and compute loss
            outputs = model.forward(batch)

            loss_hard = hard_loss(outputs['query_passage_features'].squeeze(1), 
                                 outputs['passage_passage_features'].squeeze(1), 
                                 outputs['negative_passage_features'], 
                                 criterion)
            
            dataset_name = batch['dataset_name']
            count_hard_dict[dataset_name] += 1
            loss_hard_dict[dataset_name] += loss_hard.detach().float()
            
            if dataset_name in RETRIEVAL_DATASETS:
                loss = inbatch_loss(outputs['query_passage_features'].squeeze(1), 
                                   outputs['passage_passage_features'].squeeze(1), 
                                   criterion, world_size, world_rank)
                count_dict[dataset_name] += 1
                loss_dict[dataset_name] += loss.detach().float()
            else:
                loss = 0.0
            
            loss_total = loss + loss_hard

            # backward, optimizer, scheduler - DeepSpeed compatible
            if is_deepspeed:
                # DeepSpeed handles backward, optimizer step, and scheduler step
                model.lm.backward(loss_total)
                model.lm.step()
            else:
                # Standard PyTorch training
                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()
                lr_scheduler.step()
            
            # Get current learning rate
            if is_deepspeed:
                current_lr = model.lm.get_lr()[0]
            else:
                current_lr = optimizer.param_groups[0]['lr']
            
            # Apply minimum learning rate constraint
            if current_lr < args.min_lr:
                if is_deepspeed:
                    # For DeepSpeed, we need to update the lr in the config
                    for param_group in model.lm.optimizer.param_groups:
                        param_group['lr'] = args.min_lr
                else:
                    for i in range(len(optimizer.param_groups)):
                        optimizer.param_groups[i]['lr'] = args.min_lr
            
            # log
            completed_steps += 1
            if completed_steps % args.log_interval == 0:
                pbar.update(args.log_interval)

                train_log_dict = {"lr": current_lr}
                
                for k in loss_dict.keys():
                    count = all_reduce_mean(count_dict[k].clone(), world_size)
                    if count > 0:
                        loss_sum = all_reduce_mean(loss_dict[k].clone(), world_size)
                        train_log_dict[f"{k}/training_loss_in_batch"] = loss_sum / count
                
                for k in loss_hard_dict.keys():
                    count = all_reduce_mean(count_hard_dict[k].clone(), world_size)
                    if count > 0:
                        loss_sum = all_reduce_mean(loss_hard_dict[k].clone(), world_size)
                        train_log_dict[f"{k}/training_loss_hard"] = loss_sum / count
                
                train_log_dict['Avg/retrieval/training_loss_in_batch'] = torch.tensor([v for k, v in train_log_dict.items() if k.split('/')[0] in RETRIEVAL_DATASETS and k.endswith('training_loss_in_batch')]).mean()
                train_log_dict['Avg/retrieval/training_loss_hard'] = torch.tensor([v for k, v in train_log_dict.items() if k.split('/')[0] in RETRIEVAL_DATASETS and k.endswith('training_loss_hard')]).mean()
                train_log_dict['Avg/classification/training_loss_hard'] = torch.tensor([v for k, v in train_log_dict.items() if k.split('/')[0] in CLASSIFICATION_DATASETS]).mean()
                train_log_dict['Avg/clustering/training_loss_hard'] = torch.tensor([v for k, v in train_log_dict.items() if k.split('/')[0] in CLUSTERING_DATASETS]).mean()

                if world_rank == 0:
                    print(f"[Train] Step = {completed_steps}")
                    write_tensorboard(summary_writer, train_log_dict, completed_steps)
                
                # Report metrics to Ray Train
                report_dict = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in train_log_dict.items()}
                train.report(report_dict)
                
                loss_dict = {ds_name: torch.tensor(0.0, device=device) for ds_name in RETRIEVAL_DATASETS}
                loss_hard_dict = {ds_name: torch.tensor(0.0, device=device) for ds_name in train_dataloader.loader_dict.keys()}
                count_dict = {ds_name: torch.tensor(0, device=device) for ds_name in RETRIEVAL_DATASETS}
                count_hard_dict = {ds_name: torch.tensor(0, device=device) for ds_name in train_dataloader.loader_dict.keys()}

            # validation
            if completed_steps % args.validation_steps == 0:
                model.lm.eval()
                validate(args, model, valid_loader_dict, criterion, completed_steps, summary_writer, world_size, world_rank)
                model.lm.train()

            # step checkpoint
            if args.checkpointing_steps and completed_steps % args.checkpointing_steps == 0:
                output_dir = os.path.join(args.output_dir, f"step_{completed_steps}")
                save_checkpoint(args, model, output_dir, lr_scheduler, world_rank)

            if completed_steps >= args.train_steps:
                break

        # epoch checkpoint
        output_dir = os.path.join(args.output_dir, f"epoch_{epoch+1}")
        save_checkpoint(args, model, output_dir, lr_scheduler, world_rank)
        
        if completed_steps % args.validation_steps != 0:
            model.lm.eval()
            validate(args, model, valid_loader_dict, criterion, completed_steps, summary_writer, world_size, world_rank)
            model.lm.train()
    
    if summary_writer:
        summary_writer.close()
