from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import os
import gc

CLASSIFICATION_DATASETS = ['amazon_counterfactual', 'amazon_polarity', 'imdb', 'toxic_conversations', 'cola']
CLUSTERING_DATASETS = ['amazon_reviews', 'banking77', 'emotion', 'mtop_intent', 'mtop_domain', 'massive_scenario', 'massive_intent', 'tweet_sentiment_extraction', 'arxiv_clustering_p2p', 'arxiv_clustering_s2s', 'biorxiv_clustering_p2p', 'biorxiv_clustering_s2s', 'medrxiv_clustering_p2p', 'medrxiv_clustering_s2s', 'reddit_clustering_p2p', 'reddit_clustering_s2s', 'stackexchange_clustering_p2p', 'stackexchange_clustering_s2s', 'twentynewsgroups']
RETRIEVAL_DATASETS = ['arguana', 'snli', 'mnli', 'anli', 'paq', 'squad', 'stackexchange', 'msmarco', 'natural_questions', 'hotpotqa', 'fever', 'eli5', 'fiqa', 'bioasq', 'nfcorpus', 'miracl', 'mrtidy', 'scifact', 'qqp', 'stackoverflowdupquestions', 'sts12', 'sts22', 'stsbenchmark', 'amazon_qa', 'cnn_dm', 'coliee', 'paq_part2', 'pubmedqa', 's2orc_abstract_citation', 's2orc_title_abstract', 's2orc_title_citation', 'sentence_compression', 'specter', 'triviaqa', 'xsum', 'stackexchange_part2', 'stackexchangedupquestions_s2s', 'stackexchangedupquestions_p2p']


def write_tensorboard(summary_writer: SummaryWriter, log_dict: dict, completed_steps):
    for key, value in log_dict.items():
        summary_writer.add_scalar(key, value, completed_steps)


def save_checkpoint(args, accelerator, model, output_dir, lr_scheduler):
    accelerator.wait_for_everyone()
    accelerator.print(f"Saving checkpoint to {output_dir}")
    
    if accelerator.is_main_process:
        model.tokenizer.save_pretrained(output_dir)
    unwrapped_model = accelerator.unwrap_model(model.lm)
    unwrapped_model.save_pretrained(
        output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model.lm), # this is required for zero 3
    )
    accelerator.wait_for_everyone()


def inbatch_loss(
        query_embeddings, # [bs, d]
        context_embeddings, # [bs, d]
        criterion,
        accelerator,
        temperature=0.05,
    ):
    
    bs = query_embeddings.size(0)
    a_norm = F.normalize(query_embeddings, p=2, dim=-1)
    b_cross_gpus = accelerator.gather(context_embeddings) # [bs*process, d]
    b_norm_cross_gpus = F.normalize(b_cross_gpus, p=2, dim=-1) # ()

    student_logits = torch.matmul(a_norm, b_norm_cross_gpus.t()) / temperature # [bs, bs*process]

    labels = torch.arange(bs, device=student_logits.device) + bs * accelerator.process_index
    loss_bs = criterion(student_logits, labels) # (bs)

    loss = loss_bs.mean()

    return loss


def hard_loss(
        query_embeddings, # [bs, d]
        context_embeddings, # [bs, d]
        hard_neg_embeddings, # [bs, num, d]
        criterion,
        accelerator,
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


def validate(args, accelerator, model, valid_loader_dict, criterion, completed_steps, summary_writer):
    eval_log_dict = {}
    for dataset_name, valid_dataloader in valid_loader_dict.items():
        loss_ls, loss_hard_ls = [], []
        for batch in valid_dataloader:
            with torch.no_grad():
                outputs = model.forward(batch)
                loss_hard = hard_loss(outputs['query_passage_features'].squeeze(1), outputs['passage_passage_features'].squeeze(1), outputs['negative_passage_features'], criterion, accelerator)
                # 确保loss_hard是一个标量张量
                if isinstance(loss_hard, torch.Tensor) and loss_hard.dim() == 0:
                    loss_hard_ls.append(accelerator.gather(loss_hard.unsqueeze(0)).float())
                elif isinstance(loss_hard, torch.Tensor):
                    loss_hard_ls.append(accelerator.gather(loss_hard).float())
                else:
                    loss_hard_ls.append(accelerator.gather(torch.tensor(loss_hard, device=model.lm.device).unsqueeze(0)).float())
                
                if dataset_name in RETRIEVAL_DATASETS:
                    loss = inbatch_loss(outputs['query_passage_features'].squeeze(1), outputs['passage_passage_features'].squeeze(1), criterion, accelerator)
                    # 确保loss是一个标量张量
                    if isinstance(loss, torch.Tensor) and loss.dim() == 0:
                        loss_ls.append(accelerator.gather(loss.unsqueeze(0)).float())
                    elif isinstance(loss, torch.Tensor):
                        loss_ls.append(accelerator.gather(loss).float())
                    else:
                        loss_ls.append(accelerator.gather(torch.tensor(loss, device=model.lm.device).unsqueeze(0)).float())
        
        accelerator.wait_for_everyone()
        if loss_hard_ls:
            loss_hard_ls = torch.cat(loss_hard_ls)
            eval_log_dict[f'{dataset_name}/valid_loss_hard'] = loss_hard_ls.mean()
        if dataset_name in RETRIEVAL_DATASETS and loss_ls:
            loss_ls = torch.cat(loss_ls)
            eval_log_dict[f"{dataset_name}/valid_loss_in_batch"] = loss_ls.mean()
    
    # 计算平均损失
    retrieval_loss_in_batch = [v for k, v in eval_log_dict.items() if k.split('/')[0] in RETRIEVAL_DATASETS and k.endswith('valid_loss_in_batch')]
    if retrieval_loss_in_batch:
        eval_log_dict['Avg/retrieval/valid_loss_in_batch'] = torch.stack(retrieval_loss_in_batch).mean()
    
    retrieval_loss_hard = [v for k, v in eval_log_dict.items() if k.split('/')[0] in RETRIEVAL_DATASETS and k.endswith('valid_loss_hard')]
    if retrieval_loss_hard:
        eval_log_dict['Avg/retrieval/valid_loss_hard'] = torch.stack(retrieval_loss_hard).mean()
    
    classification_loss_hard = [v for k, v in eval_log_dict.items() if k.split('/')[0] in CLASSIFICATION_DATASETS]
    if classification_loss_hard:
        eval_log_dict['Avg/classification/valid_loss_hard'] = torch.stack(classification_loss_hard).mean()
    
    clustering_loss_hard = [v for k, v in eval_log_dict.items() if k.split('/')[0] in CLUSTERING_DATASETS]
    if clustering_loss_hard:
        eval_log_dict['Avg/clustering/valid_loss_hard'] = torch.stack(clustering_loss_hard).mean()
    
    if accelerator.is_main_process and eval_log_dict:
        write_tensorboard(summary_writer, eval_log_dict, completed_steps)
    accelerator.print(f"[Validation] Step = {completed_steps}")


def accelerate_train(args,
                     accelerator, 
                     model, 
                     train_dataloader,
                     valid_loader_dict,
                     optimizer,
                     lr_scheduler,
                     num_train_samples):
    # 计算有效批次大小和步数
    effective_batch_size = args.train_batch_size * args.gradient_accumulation_steps * accelerator.num_processes
    effective_train_steps = args.train_steps // args.gradient_accumulation_steps if args.train_steps > 0 else -1
    
    accelerator.print("**************************************** Start training ****************************************")
    accelerator.print(f" Num train samples = {num_train_samples}")
    accelerator.print(f" Num epochs = {args.train_epochs}")
    accelerator.print(f" Per device batch size = {args.train_batch_size}")
    accelerator.print(f" Gradient accumulation steps = {args.gradient_accumulation_steps}")
    accelerator.print(f" Effective batch size = {effective_batch_size}")
    accelerator.print(f" Global batch size = {args.train_batch_size * accelerator.num_processes}")
    accelerator.print(f" Step per epoch = {len(train_dataloader)}")
    accelerator.print(f" Total training steps = {args.train_steps}")
    accelerator.print(f" Effective training steps = {effective_train_steps if effective_train_steps > 0 else 'auto'}")
    accelerator.print("************************************************************************************************")
    
    global RETRIEVAL_DATASETS, CLASSIFICATION_DATASETS, CLUSTERING_DATASETS
    RETRIEVAL_DATASETS = [ds for ds in RETRIEVAL_DATASETS if ds in train_dataloader.loader_dict.keys()]
    CLASSIFICATION_DATASETS = [ds for ds in CLASSIFICATION_DATASETS if ds in train_dataloader.loader_dict.keys()]
    CLUSTERING_DATASETS = [ds for ds in CLUSTERING_DATASETS if ds in train_dataloader.loader_dict.keys()]

    summary_writer = SummaryWriter(log_dir=args.tb_dir) if accelerator.is_main_process else None
    criterion = CrossEntropyLoss(reduction='none')
    
    # 调整进度条和步数计算
    effective_total_steps = args.train_steps if args.train_steps > 0 else len(train_dataloader) * args.train_epochs
    pbar = tqdm(range(effective_total_steps), disable=not accelerator.is_local_main_process)
    
    completed_steps = 0
    effective_completed_steps = 0
    
    # 损失累积
    loss_dict = {ds_name: torch.tensor(0.0, device=model.lm.device) for ds_name in RETRIEVAL_DATASETS}
    loss_hard_dict = {ds_name: torch.tensor(0.0, device=model.lm.device) for ds_name in train_dataloader.loader_dict.keys()}
    count_dict = {ds_name: torch.tensor(0, device=model.lm.device) for ds_name in RETRIEVAL_DATASETS}
    count_hard_dict = {ds_name: torch.tensor(0, device=model.lm.device) for ds_name in train_dataloader.loader_dict.keys()}
    
    # 梯度累积状态
    accumulated_loss = 0.0
    accumulated_loss_hard = 0.0
    grad_norm = 0.0

    model.lm.train()
    for epoch in range(args.train_epochs):
        accelerator.print(f"*************** Starting epoch {epoch+1} ***************")
        train_dataloader.reset_epoch(epoch)
        
        for step, batch in enumerate(train_dataloader):
            # forward and compute loss
            outputs = model.forward(batch)
            
            loss_hard = hard_loss(outputs['query_passage_features'].squeeze(1), 
                                outputs['passage_passage_features'].squeeze(1), 
                                outputs['negative_passage_features'], 
                                criterion, accelerator)
            dataset_name = batch['dataset_name']
            
            if dataset_name in RETRIEVAL_DATASETS:
                loss = inbatch_loss(outputs['query_passage_features'].squeeze(1), 
                                  outputs['passage_passage_features'].squeeze(1), 
                                  criterion, accelerator)
            else:
                loss = 0.0
            
            # 累积损失（按梯度累积步数缩放）
            loss_total = (loss + loss_hard) / args.gradient_accumulation_steps
            accumulated_loss += loss / args.gradient_accumulation_steps
            accumulated_loss_hard += loss_hard / args.gradient_accumulation_steps
            
            # 累积梯度
            accelerator.backward(loss_total)
            
            # 更新统计信息
            count_hard_dict[dataset_name] += 1
            loss_hard_dict[dataset_name] += loss_hard.detach().float()
            if dataset_name in RETRIEVAL_DATASETS:
                count_dict[dataset_name] += 1
                loss_dict[dataset_name] += loss.detach().float()
            
            # 检查是否达到梯度累积步数
            is_update_step = ((step + 1) % args.gradient_accumulation_steps == 0) or (step + 1 == len(train_dataloader))
            
            if is_update_step:
                # 梯度裁剪
                if args.max_grad_norm > 0:
                    grad_norm = accelerator.clip_grad_norm_(model.lm.parameters(), args.max_grad_norm)
                
                # 优化器步骤
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                # 确保学习率不低于最小值
                if optimizer.param_groups[0]['lr'] < args.min_lr:
                    for i in range(len(optimizer.param_groups)):
                        optimizer.param_groups[i]['lr'] = args.min_lr
                
                effective_completed_steps += 1
                
                # 内存清理
                if step % 100 == 0:
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # 重置累积损失
                accumulated_loss = 0.0
                accumulated_loss_hard = 0.0
            
            # 更新进度条
            completed_steps += 1
            if completed_steps % args.log_interval == 0:
                pbar.update(args.log_interval)
                
                # 计算平均损失
                train_log_dict = {
                    "lr": optimizer.param_groups[0]['lr'],
                    "grad_norm": grad_norm if isinstance(grad_norm, (int, float)) else grad_norm.item() if hasattr(grad_norm, 'item') else 0.0
                }
                
                for k in loss_dict.keys():
                    count = accelerator.gather(count_dict[k]).sum()
                    if count > 0:
                        train_log_dict[f"{k}/training_loss_in_batch"] = accelerator.gather(loss_dict[k]).sum() / count
                for k in loss_hard_dict.keys():
                    count = accelerator.gather(count_hard_dict[k]).sum()
                    if count > 0:
                        train_log_dict[f"{k}/training_loss_hard"] = accelerator.gather(loss_hard_dict[k]).sum() / count
                
                # 计算平均损失
                avg_keys = ['Avg/retrieval/training_loss_in_batch', 'Avg/retrieval/training_loss_hard', 
                           'Avg/classification/training_loss_hard', 'Avg/clustering/training_loss_hard']
                for avg_key in avg_keys:
                    relevant_keys = [k for k in train_log_dict.keys() if avg_key.split('/')[1] in k and k.endswith(avg_key.split('/')[-1])]
                    if relevant_keys:
                        values = [train_log_dict[k] for k in relevant_keys]
                        train_log_dict[avg_key] = torch.tensor(values).mean()
                
                accelerator.print(f"[Train] Step = {effective_completed_steps} (effective)")
                if accelerator.is_main_process:
                    write_tensorboard(summary_writer, train_log_dict, effective_completed_steps)
                
                # 重置统计信息
                loss_dict = {ds_name: torch.tensor(0.0, device=model.lm.device) for ds_name in RETRIEVAL_DATASETS}
                loss_hard_dict = {ds_name: torch.tensor(0.0, device=model.lm.device) for ds_name in train_dataloader.loader_dict.keys()}
                count_dict = {ds_name: torch.tensor(0, device=model.lm.device) for ds_name in RETRIEVAL_DATASETS}
                count_hard_dict = {ds_name: torch.tensor(0, device=model.lm.device) for ds_name in train_dataloader.loader_dict.keys()}
            
            # 验证（基于有效步数）
            if effective_completed_steps > 0 and effective_completed_steps % args.validation_steps == 0:
                model.lm.eval()
                validate(args, accelerator, model, valid_loader_dict, criterion, effective_completed_steps, summary_writer)
                model.lm.train()
            
            # 检查点保存（基于有效步数）
            if args.checkpointing_steps and effective_completed_steps > 0 and effective_completed_steps % args.checkpointing_steps == 0:
                output_dir = os.path.join(args.output_dir, f"step_{effective_completed_steps}")
                save_checkpoint(args, accelerator, model, output_dir, lr_scheduler)
            
            if effective_completed_steps >= args.train_steps and args.train_steps > 0:
                break
        
        # epoch checkpoint（基于有效步数）
        if effective_completed_steps > 0:
            output_dir = os.path.join(args.output_dir, f"epoch_{epoch+1}")
            save_checkpoint(args, accelerator, model, output_dir, lr_scheduler)
            if effective_completed_steps % args.validation_steps != 0:
                model.lm.eval()
                validate(args, accelerator, model, valid_loader_dict, criterion, effective_completed_steps, summary_writer)
                model.lm.train()
    
    if summary_writer:
        summary_writer.close()