from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import os

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
    # b_norm = torch.nn.functional.normalize(context_embeddings, p=2, dim=-1)
    b_cross_gpus = accelerator.gather(context_embeddings) # [bs*process, d]
    # print((context_embeddings - b_cross_gpus[bs * accelerator.process_index : bs * accelerator.process_index+bs]).abs().sum())
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
                loss_hard_ls.append(accelerator.gather(loss_hard).float())
                if dataset_name in RETRIEVAL_DATASETS:
                    loss = inbatch_loss(outputs['query_passage_features'].squeeze(1), outputs['passage_passage_features'].squeeze(1), criterion, accelerator)
                    loss_ls.append(accelerator.gather(loss).float())
        
        accelerator.wait_for_everyone()
        loss_hard_ls = torch.cat(loss_hard_ls)
        eval_log_dict[f'{dataset_name}/valid_loss_hard'] = loss_hard_ls.mean()
        if dataset_name in RETRIEVAL_DATASETS:
            loss_ls = torch.cat(loss_ls)
            eval_log_dict[f"{dataset_name}/valid_loss_in_batch"] = loss_ls.mean()
    
    eval_log_dict['Avg/retrieval/valid_loss_in_batch'] = torch.tensor([v for k, v in eval_log_dict.items() if k.split('/')[0] in RETRIEVAL_DATASETS and k.endswith('valid_loss_in_batch')]).mean()
    eval_log_dict['Avg/retrieval/valid_loss_hard'] = torch.tensor([v for k, v in eval_log_dict.items() if k.split('/')[0] in RETRIEVAL_DATASETS and k.endswith('valid_loss_hard')]).mean()
    eval_log_dict['Avg/classification/valid_loss_hard'] = torch.tensor([v for k, v in eval_log_dict.items() if k.split('/')[0] in CLASSIFICATION_DATASETS]).mean()
    eval_log_dict['Avg/clustering/valid_loss_hard'] = torch.tensor([v for k, v in eval_log_dict.items() if k.split('/')[0] in CLUSTERING_DATASETS]).mean()
    if accelerator.is_main_process:
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
    accelerator.print("**************************************** Start training ****************************************")
    accelerator.print(f" Num train samples = {num_train_samples}")
    accelerator.print(f" Num epochs = {args.train_epochs}")
    accelerator.print(f" Per device batch size = {args.train_batch_size}")
    accelerator.print(f" Global batch size = {args.train_batch_size * accelerator.num_processes}")
    accelerator.print(f" Step per epoch = {len(train_dataloader)}")
    accelerator.print(f" Total training steps = {args.train_steps}")
    accelerator.print("************************************************************************************************")
    global RETRIEVAL_DATASETS, CLASSIFICATION_DATASETS, CLUSTERING_DATASETS
    RETRIEVAL_DATASETS = [ds for ds in RETRIEVAL_DATASETS if ds in train_dataloader.loader_dict.keys()]
    CLASSIFICATION_DATASETS = [ds for ds in CLASSIFICATION_DATASETS if ds in train_dataloader.loader_dict.keys()]
    CLUSTERING_DATASETS = [ds for ds in CLUSTERING_DATASETS if ds in train_dataloader.loader_dict.keys()]

    summary_writer = SummaryWriter(log_dir=args.tb_dir) if accelerator.is_main_process else None
    criterion = CrossEntropyLoss(reduction='none')
    pbar = tqdm(range(args.train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    loss_dict = {ds_name: torch.tensor(0.0, device=model.lm.device) for ds_name in RETRIEVAL_DATASETS}
    loss_hard_dict = {ds_name: torch.tensor(0.0, device=model.lm.device) for ds_name in train_dataloader.loader_dict.keys()}
    count_dict = {ds_name: torch.tensor(0, device=model.lm.device) for ds_name in RETRIEVAL_DATASETS}
    count_hard_dict = {ds_name: torch.tensor(0, device=model.lm.device) for ds_name in train_dataloader.loader_dict.keys()}

    # -----验证梯度累积生效-----
    # step1:
    unwrapped_model = accelerator.unwrap_model(model.lm)
    if accelerator.is_main_process:
        try:
            initial_weight = unwrapped_model.get_parameter("transformer.h.0.attn.c_attn.weight").clone().detach()
        except AttributeError:
            for name, param in unwrapped_model.named_parameters():
                if param.requires_grad:
                    initial_weight = param.clone().detach()
                    weight_name = name
                    break

    model.lm.train()
    for epoch in range(args.train_epochs):
        accelerator.print(f"*************** Starting epoch {epoch + 1} ***************")
        train_dataloader.reset_epoch(epoch)
        for batch in train_dataloader:
            with accelerator.accumulate(model.lm):
                # forward and compute loss
                outputs = model.forward(batch)
                # passage features: [bs, 1, d]
                # hard_neg_features: [bs, num_hard_neg, d]

                loss_hard = hard_loss(outputs['query_passage_features'].squeeze(1),
                                      outputs['passage_passage_features'].squeeze(1),
                                      outputs['negative_passage_features'], criterion, accelerator)
                dataset_name = batch['dataset_name']
                count_hard_dict[dataset_name] += 1
                loss_hard_dict[dataset_name] += loss_hard.detach().float()
                if dataset_name in RETRIEVAL_DATASETS:
                    loss = inbatch_loss(outputs['query_passage_features'].squeeze(1),
                                        outputs['passage_passage_features'].squeeze(1), criterion, accelerator)
                    count_dict[dataset_name] += 1
                    loss_dict[dataset_name] += loss.detach().float()
                else:
                    loss = 0.0

                loss_total = loss + loss_hard

                # backward, optimizer, scheduler
                accelerator.backward(loss_total)
                # optimizer.step()
            lr_scheduler.step()
            # optimizer.zero_grad()
            if optimizer.param_groups[0]['lr'] < args.min_lr:
                for i in range(len(optimizer.param_groups)):
                    optimizer.param_groups[i]['lr'] = args.min_lr

            # log
            completed_steps += 1

            # step2: 每 2 步检查权重变化
            if completed_steps % 2 == 0 and accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model.lm)
                try:
                    current_weight = unwrapped_model.get_parameter("transformer.h.0.attn.c_attn.weight")
                except AttributeError:
                    current_weight = dict(unwrapped_model.named_parameters())[weight_name]

                weight_changed = not torch.equal(initial_weight, current_weight)
                accelerator.print(f"[DEBUG] Step {completed_steps}: Weight changed = {weight_changed}")
                if weight_changed:
                    initial_weight = current_weight.clone().detach()

            if completed_steps % args.log_interval == 0:
                pbar.update(args.log_interval)

                train_log_dict = {"lr": optimizer.param_groups[0]['lr']}
                for k in loss_dict.keys():
                    count = accelerator.gather(count_dict[k]).sum()
                    if count > 0:
                        train_log_dict[f"{k}/training_loss_in_batch"] = accelerator.gather(loss_dict[k]).sum() / count
                for k in loss_hard_dict.keys():
                    count = accelerator.gather(count_hard_dict[k]).sum()
                    if count > 0:
                        train_log_dict[f"{k}/training_loss_hard"] = accelerator.gather(loss_hard_dict[k]).sum() / count
                train_log_dict['Avg/retrieval/training_loss_in_batch'] = torch.tensor([v for k, v in train_log_dict.items() if k.split('/')[0] in RETRIEVAL_DATASETS and k.endswith('training_loss_in_batch')]).mean()
                train_log_dict['Avg/retrieval/training_loss_hard'] = torch.tensor([v for k, v in train_log_dict.items() if k.split('/')[0] in RETRIEVAL_DATASETS and k.endswith('training_loss_hard')]).mean()
                train_log_dict['Avg/classification/training_loss_hard'] = torch.tensor([v for k, v in train_log_dict.items() if k.split('/')[0] in CLASSIFICATION_DATASETS]).mean()
                train_log_dict['Avg/clustering/training_loss_hard'] = torch.tensor([v for k, v in train_log_dict.items() if k.split('/')[0] in CLUSTERING_DATASETS]).mean()

                accelerator.print(f"[Train] Step = {completed_steps}")
                if accelerator.is_main_process:
                    write_tensorboard(summary_writer, train_log_dict, completed_steps)
                loss_dict = {ds_name: torch.tensor(0.0, device=model.lm.device) for ds_name in RETRIEVAL_DATASETS}
                loss_hard_dict = {ds_name: torch.tensor(0.0, device=model.lm.device) for ds_name in train_dataloader.loader_dict.keys()}
                count_dict = {ds_name: torch.tensor(0, device=model.lm.device) for ds_name in RETRIEVAL_DATASETS}
                count_hard_dict = {ds_name: torch.tensor(0, device=model.lm.device) for ds_name in train_dataloader.loader_dict.keys()}

            # validation
            if completed_steps % args.validation_steps == 0:
                model.lm.eval()
                validate(args, accelerator, model, valid_loader_dict, criterion, completed_steps, summary_writer)
                model.lm.train()

            # step checkpoint
            if args.checkpointing_steps and completed_steps % args.checkpointing_steps == 0:
                output_dir = os.path.join(args.output_dir, f"step_{completed_steps}")
                save_checkpoint(args, accelerator, model, output_dir, lr_scheduler)

            if completed_steps >= args.train_steps:
                break

        # epoch checkpoint
        output_dir = os.path.join(args.output_dir, f"epoch_{epoch+1}")
        save_checkpoint(args, accelerator, model, output_dir, lr_scheduler)
        if completed_steps % args.validation_steps != 0:
            model.lm.eval()
            validate(args, accelerator, model, valid_loader_dict, criterion, completed_steps, summary_writer)
            model.lm.train()
    
    if summary_writer:
        summary_writer.close()