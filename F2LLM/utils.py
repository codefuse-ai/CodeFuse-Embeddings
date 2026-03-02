from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import os, math

CLASSIFICATION_DATASETS = ['amazon_counterfactual', 'amazon_polarity', 'imdb', 'toxic_conversations', 'cola', 'mela', 'waimai', 'nordic_classification_danish', 'nordic_classification_swedish', 'nordic_classification_norwegian']
CLUSTERING_DATASETS = ['amazon_reviews', 'banking77', 'emotion', 'mtop_intent', 'mtop_domain', 'massive_scenario', 'massive_intent', 'tweet_sentiment_extraction', 'arxiv_clustering_p2p', 'arxiv_clustering_s2s', 'biorxiv_clustering_p2p', 'biorxiv_clustering_s2s', 'medrxiv_clustering_p2p', 'medrxiv_clustering_s2s', 'reddit_clustering_p2p', 'reddit_clustering_s2s', 'stackexchange_clustering_p2p', 'stackexchange_clustering_s2s', 'twentynewsgroups', 'sib200', 'mlsum_clustering_de', 'mlsum_clustering_es', 'mlsum_clustering_fr', 'mlsum_clustering_ru', 'bactrianx_language_classification', 'bactrianx_translation', 'europarl', 'xcodeeval_translation', 'tnews', 'thucnews', 'csl', 'cedr', 'ru_sentiment']
RETRIEVAL_DATASETS = ['arguana', 'snli', 'mnli', 'anli', 'paq', 'squad', 'stackexchange', 'msmarco', 'natural_questions', 'hotpotqa', 'fever', 'eli5', 'fiqa', 'bioasq', 'nfcorpus', 'miracl', 'mrtidy', 'scifact', 'qqp', 'stackoverflowdupquestions', 'sts12', 'sts22', 'stsbenchmark', 'amazon_qa', 'cnn_dm', 'coliee', 'paq_part2', 'pubmedqa', 's2orc_abstract_citation', 's2orc_title_abstract', 's2orc_title_citation', 'sentence_compression', 'specter', 'triviaqa', 'xsum', 'stackexchange_part2', 'stackexchangedupquestions_s2s', 'stackexchangedupquestions_p2p', 'webfaq_ara', 'webfaq_aze', 'webfaq_ben', 'webfaq_bul', 'webfaq_cat', 'webfaq_ces', 'webfaq_dan', 'webfaq_deu', 'webfaq_ell', 'webfaq_eng', 'webfaq_est', 'webfaq_fas', 'webfaq_fin', 'webfaq_fra', 'webfaq_heb', 'webfaq_hin', 'webfaq_hrv', 'webfaq_hun', 'webfaq_ind', 'webfaq_isl', 'webfaq_ita', 'webfaq_jpn', 'webfaq_kat', 'webfaq_kaz', 'webfaq_kor', 'webfaq_lav', 'webfaq_lit', 'webfaq_mar', 'webfaq_msa', 'webfaq_nld', 'webfaq_nor', 'webfaq_pol', 'webfaq_por', 'webfaq_ron', 'webfaq_rus', 'webfaq_slk', 'webfaq_slv', 'webfaq_spa', 'webfaq_sqi', 'webfaq_srp', 'webfaq_swe', 'webfaq_tgl', 'webfaq_tha', 'webfaq_tur', 'webfaq_ukr', 'webfaq_urd', 'webfaq_uzb', 'webfaq_vie', 'webfaq_zho', 'mmarco_arabic', 'mmarco_chinese', 'mmarco_dutch', 'mmarco_french', 'mmarco_german', 'mmarco_hindi', 'mmarco_indonesian', 'mmarco_italian', 'mmarco_japanese', 'mmarco_portuguese', 'mmarco_russian', 'mmarco_spanish', 'mmarco_vietnamese', 'miracl_ar', 'miracl_bn', 'miracl_en', 'miracl_es', 'miracl_fa', 'miracl_fi', 'miracl_fr', 'miracl_hi', 'miracl_id', 'miracl_ja', 'miracl_ko', 'miracl_ru', 'miracl_sw', 'miracl_te', 'miracl_th', 'miracl_zh', 'mrtidy_arabic', 'mrtidy_bengali', 'mrtidy_finnish', 'mrtidy_indonesian', 'mrtidy_japanese', 'mrtidy_korean', 'mrtidy_russian', 'mrtidy_swahili', 'mrtidy_telugu', 'mrtidy_thai', 'mldr_ar', 'mldr_de', 'mldr_en', 'mldr_es', 'mldr_fr', 'mldr_hi', 'mldr_it', 'mldr_ja', 'mldr_ko', 'mldr_pt', 'mldr_ru', 'mldr_th', 'mldr_zh', 'mkqa_ar', 'mkqa_da', 'mkqa_de', 'mkqa_en', 'mkqa_es', 'mkqa_fi', 'mkqa_fr', 'mkqa_he', 'mkqa_hu', 'mkqa_it', 'mkqa_ja', 'mkqa_km', 'mkqa_ko', 'mkqa_ms', 'mkqa_nl', 'mkqa_no', 'mkqa_pl', 'mkqa_pt', 'mkqa_ru', 'mkqa_sv', 'mkqa_th', 'mkqa_tr', 'mkqa_vi', 'mkqa_zh_cn', 'mkqa_zh_hk', 'mkqa_zh_tw', 'sts22_multilingual', 'mlsum_retrieval_de', 'mlsum_retrieval_es', 'mlsum_retrieval_fr', 'mlsum_retrieval_ru', 'mlsum_retrieval_tr', 'aya', 'muri', 'muri_part2', 'unpc_ar2en', 'unpc_ar2es', 'unpc_ar2fr', 'unpc_ar2ru', 'unpc_ar2zh', 'unpc_en2ar', 'unpc_en2es', 'unpc_en2fr', 'unpc_en2ru', 'unpc_en2zh', 'unpc_es2ar', 'unpc_es2en', 'unpc_es2fr', 'unpc_es2ru', 'unpc_es2zh', 'unpc_fr2ar', 'unpc_fr2en', 'unpc_fr2es', 'unpc_fr2ru', 'unpc_fr2zh', 'unpc_ru2ar', 'unpc_ru2en', 'unpc_ru2es', 'unpc_ru2fr', 'unpc_ru2zh', 'unpc_zh2ar', 'unpc_zh2en', 'unpc_zh2es', 'unpc_zh2fr', 'unpc_zh2ru', 'xnli_ar', 'xnli_bg', 'xnli_de', 'xnli_el', 'xnli_es', 'xnli_fr', 'xnli_hi', 'xnli_ru', 'xnli_sw', 'xnli_th', 'xnli_tr', 'xnli_ur', 'xnli_vi', 'xnli_zh', 'pawsx_de', 'pawsx_en', 'pawsx_es', 'pawsx_fr', 'pawsx_ja', 'pawsx_ko', 'pawsx_zh', 'csn_ccr_go', 'csn_ccr_java', 'csn_ccr_javascript', 'csn_ccr_php', 'csn_ccr_python', 'csn_ccr_ruby', 'csn_go', 'csn_java', 'csn_javascript', 'csn_php', 'csn_python', 'csn_ruby', 'cosqa', 'synthetic_text2sql', 'codefeedback_mt', 'codefeedback_st', 'stackoverflow_qa', 'ocgi', 'ocgi_part2', 'ocr2_cpp', 'ocr2_python', 'xcodeeval_code2code_C#', 'xcodeeval_code2code_C++', 'xcodeeval_code2code_C', 'xcodeeval_code2code_D', 'xcodeeval_code2code_Go', 'xcodeeval_code2code_Haskell', 'xcodeeval_code2code_Java', 'xcodeeval_code2code_Javascript', 'xcodeeval_code2code_Kotlin', 'xcodeeval_code2code_Ocaml', 'xcodeeval_code2code_PHP', 'xcodeeval_code2code_Pascal', 'xcodeeval_code2code_Perl', 'xcodeeval_code2code_Python', 'xcodeeval_code2code_Ruby', 'xcodeeval_code2code_Rust', 'xcodeeval_code2code_Scala', 'xcodeeval_nl2code_C#', 'xcodeeval_nl2code_C++', 'xcodeeval_nl2code_C', 'xcodeeval_nl2code_D', 'xcodeeval_nl2code_Go', 'xcodeeval_nl2code_Haskell', 'xcodeeval_nl2code_Java', 'xcodeeval_nl2code_Javascript', 'xcodeeval_nl2code_Kotlin', 'xcodeeval_nl2code_Ocaml', 'xcodeeval_nl2code_PHP', 'xcodeeval_nl2code_Pascal', 'xcodeeval_nl2code_Perl', 'xcodeeval_nl2code_Python', 'xcodeeval_nl2code_Ruby', 'xcodeeval_nl2code_Rust', 'xcodeeval_nl2code_Scala', 'procqa', 'paracrawl_so-en', 'paracrawl_en-sw', 'paracrawl_sw-en', 'paracrawl_en-tg', 'paracrawl_en-azj', 'paracrawl_tg-en', 'paracrawl_en-th', 'paracrawl_azj-en', 'paracrawl_th-en', 'paracrawl_en-tl', 'paracrawl_en-hi', 'paracrawl_tl-en', 'paracrawl_en-uk', 'paracrawl_hi-en', 'paracrawl_uk-en', 'paracrawl_en-vi', 'paracrawl_en-hy', 'paracrawl_vi-en', 'paracrawl_en-zh', 'paracrawl_zh-en', 'paracrawl_hy-en', 'paracrawl_es-ca', 'paracrawl_ca-es', 'paracrawl_en-id', 'paracrawl_es-eu', 'paracrawl_id-en', 'paracrawl_eu-es', 'paracrawl_es-gl', 'paracrawl_gl-es', 'paracrawl_en-km', 'paracrawl_nl-fr', 'paracrawl_fr-nl', 'paracrawl_pl-cs', 'paracrawl_km-en', 'paracrawl_cs-pl', 'paracrawl_pl-de', 'paracrawl_en-ko', 'paracrawl_de-pl', 'paracrawl_ko-en', 'paracrawl_en-lo', 'paracrawl_lo-en', 'paracrawl_en-my', 'paracrawl_my-en', 'paracrawl_en-ne', 'paracrawl_ne-en', 'paracrawl_en-ps', 'paracrawl_ps-en', 'paracrawl_en-ru', 'paracrawl_ru-en', 'paracrawl_en-si', 'paracrawl_si-en', 'paracrawl_en-so', 'natural_reasoning', 'natural_reasoning_part2', 'multialpaca', 'coig', 'oasst2', 'wildchat', 'wildchat_part2', 'wildchat_part3', 'm2lingual', 'infinstruct', 'infinstruct_part2', 'cmedqav2', 'medinstruct', 'huatuo_kgqa', 'huatuo_encqa', 'clirmatrix_af-bg', 'clirmatrix_cv-fa', 'clirmatrix_fi-ht', 'clirmatrix_ku-mk', 'clirmatrix_pnb-sk', 'clirmatrix_th-uk', 'clirmatrix_ar-fr', 'clirmatrix_en', 'clirmatrix_hu-ko', 'clirmatrix_ml-nl', 'clirmatrix_ru-it', 'clirmatrix_ur-yo', 'clirmatrix_bn-cs', 'clirmatrix_es-pt', 'clirmatrix_ja-de', 'clirmatrix_nn-pms', 'clirmatrix_sl-tg', 'clirmatrix_zh', 'bq', 'yahoo_answers', 'dureader', 't2ranking', 'gooaq', 'medicalqa_ru', 'medi2', 'medi2_part2', 'openorca', 'openorca_part2', 'medical_instruction', 'multicpr_medical', 'healthcaremagic', 'multicpr_ecom', 'ocnli', 'esci', 'cord19', 'cmcqa', 'cmnli', 'dbpedia', 'fever_nl', 'hotpotqa_nl', 'koalpaca', 'koalpaca_realqa', 'komagpie', 'lawzhidao', 'lcqmc', 'lcsts', 'llm_retrieval_data', 'mailruqa', 'medical_flashcards', 'medmcqa', 'medqa_en', 'medqa_zh', 'medquad', 'mscinli', 'nordic_retrieval_danish', 'nordic_retrieval_norwegian', 'nordic_retrieval_swedish', 'nordic_sts_danish', 'nordic_sts_norwegian', 'nordic_sts_swedish', 'nordic_text_matching_danish', 'nordic_text_matching_norwegian', 'nordic_text_matching_swedish', 'parsquad', 'persian_qa', 'pquad', 'qbqtc', 'refgpt', 'ru_instruct', 'siberian_dataset', 'simclue', 'webmedqa', 'wikiomnia', 'mqa_ca', 'mqa_lt', 'mqa_nl', 'mqa_sr', 'mqa_es', 'mqa_he', 'mqa_id', 'mqa_de', 'mqa_lv', 'mqa_vi', 'mqa_it', 'mqa_sk', 'mqa_ms', 'mqa_hi', 'mqa_zh', 'mqa_hr', 'mqa_th', 'mqa_no', 'mqa_is', 'mqa_bg', 'mqa_ru', 'mqa_pl', 'mqa_ja', 'mqa_ro', 'mqa_da', 'mqa_sv', 'mqa_uk', 'mqa_el', 'mqa_pt', 'mqa_et', 'mqa_cs', 'mqa_tr', 'mqa_fr', 'mqa_fa', 'mqa_hu', 'mqa_ko', 'mqa_ar', 'mqa_fi']


def write_tensorboard(summary_writer: SummaryWriter, log_dict: dict, completed_steps):
    for key, value in log_dict.items():
        summary_writer.add_scalar(key, value, completed_steps)


def save_checkpoint(args, accelerator, model, output_dir, lr_scheduler):
    accelerator.wait_for_everyone()
    accelerator.print(f"Saving checkpoint to {output_dir}")
    
    if accelerator.is_main_process:
        model.tokenizer.save_pretrained(output_dir)
    unwrapped_model = accelerator.unwrap_model(model.lm)
    if accelerator.is_main_process:
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
        use_mrl=False,
        mrl_dims=None
    ):
    
    def _calculate_loss(q_emb, c_emb, labels):
        a_norm = F.normalize(q_emb, p=2, dim=-1)
        b_norm_cross_gpus = F.normalize(c_emb, p=2, dim=-1) # ()

        student_logits = torch.matmul(a_norm, b_norm_cross_gpus.t()) / temperature # [bs, bs*process]

        loss_bs = criterion(student_logits, labels) # (bs)

        return loss_bs.mean()

    bs = query_embeddings.size(0)
    b_cross_gpus = accelerator.gather(context_embeddings) # [bs*process, d]
    labels = torch.arange(bs, device=b_cross_gpus.device) + bs * accelerator.process_index
    total_loss = 0.0
    dims = (mrl_dims + [query_embeddings.shape[-1]]) if use_mrl else [query_embeddings.shape[-1]]
    d = query_embeddings.shape[-1]
    
    n = 1
    for dim in sorted(dims)[::-1]:
        if dim > query_embeddings.shape[-1]:
            continue
        q_emb_trunc = query_embeddings[..., :dim]
        c_emb_trunc = b_cross_gpus[..., :dim]
        total_loss += _calculate_loss(q_emb_trunc, c_emb_trunc, labels) / (n * math.sqrt(d/dim))
        n += 1
    return total_loss# / n



def hard_loss(
        query_embeddings, # [bs, d]
        context_embeddings, # [bs, d]
        hard_neg_embeddings, # [bs, num, d]
        criterion,
        accelerator,
        temperature=0.05,
        use_mrl=False,
        mrl_dims=None
    ):
    
    if hard_neg_embeddings is None:
        return 0.0

    def _calculate_loss(q_emb, passage_emb):
        bs = q_emb.size(0)
        a_norm = F.normalize(q_emb, p=2, dim=-1)
        
        hard_norm = F.normalize(passage_emb, p=2, dim=-1)
        logits = (a_norm.unsqueeze(1) * hard_norm).sum(-1) / temperature # [bs, num_hard+1]

        loss_hard = criterion(logits, torch.zeros((bs), dtype=torch.long, device=logits.device)).mean()

        return loss_hard

    total_loss = 0.0
    dims = (mrl_dims + [query_embeddings.shape[-1]]) if use_mrl else [query_embeddings.shape[-1]]
    d = query_embeddings.shape[-1]
    hard_neg_embeddings = torch.concat([
        context_embeddings.unsqueeze(1),
        hard_neg_embeddings
    ], dim=1) # [bs, num_hard+1, d]

    n = 1
    for dim in sorted(dims)[::-1]:
        if dim > query_embeddings.shape[-1]:
            continue
        q_emb_trunc = query_embeddings[..., :dim]
        all_passages_trunc = hard_neg_embeddings[..., :dim]
        total_loss += _calculate_loss(q_emb_trunc, all_passages_trunc) / (n * math.sqrt(d/dim))
        n += 1
    return total_loss# / n


def validate(args, accelerator, model, valid_loader_dict, criterion, completed_steps, summary_writer):
    eval_log_dict = {}
    for dataset_name, valid_dataloader in valid_loader_dict.items():
        loss_ls, loss_hard_ls = [], []
        for batch in valid_dataloader:
            with torch.no_grad():
                output = model.forward(batch)[0]
                loss_hard = hard_loss(output['query_passage_features'].squeeze(1), output['passage_passage_features'].squeeze(1), output['negative_passage_features'], criterion, accelerator)
                loss_hard_ls.append(accelerator.gather(loss_hard).float())
                if dataset_name in RETRIEVAL_DATASETS:
                    loss = inbatch_loss(output['query_passage_features'].squeeze(1), output['passage_passage_features'].squeeze(1), criterion, accelerator)
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
    if any(ds not in RETRIEVAL_DATASETS+CLASSIFICATION_DATASETS+CLUSTERING_DATASETS for ds in train_dataloader.loader_dict.keys()):
        raise ValueError("Unknown dataset")

    summary_writer = SummaryWriter(log_dir=args.tb_dir) if accelerator.is_main_process else None
    criterion = CrossEntropyLoss(reduction='none')
    pbar = tqdm(range(args.train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    loss_dict = {ds_name: torch.tensor(0.0, device=model.lm.device) for ds_name in RETRIEVAL_DATASETS}
    loss_hard_dict = {ds_name: torch.tensor(0.0, device=model.lm.device) for ds_name in train_dataloader.loader_dict.keys()}
    count_dict = {ds_name: torch.tensor(0, device=model.lm.device) for ds_name in RETRIEVAL_DATASETS}
    count_hard_dict = {ds_name: torch.tensor(0, device=model.lm.device) for ds_name in train_dataloader.loader_dict.keys()}

    # initialize mrl dimensions
    mrl_dims = [2**i for i in range(3, 15)]
    mrl_dims = [d for d in mrl_dims if d < model.hidden_size]

    model.lm.train()
    for epoch in range(args.train_epochs):
        accelerator.print(f"*************** Starting epoch {epoch+1} ***************")
        train_dataloader.reset_epoch(epoch)
        for batch in train_dataloader:
            # forward and compute loss
            outputs = model.forward(batch, accelerator)
            # passage features: [bs, 1, d]
            # hard_neg_features: [bs, num_hard_neg, d]

            loss, loss_hard = 0.0, 0.0
            for i, output in enumerate(outputs):
                loss_hard += hard_loss(output['query_passage_features'].squeeze(1), output['passage_passage_features'].squeeze(1), output['negative_passage_features'], criterion, accelerator, use_mrl=args.use_mrl, mrl_dims=mrl_dims)

            dataset_name = batch['dataset_name']
            count_hard_dict[dataset_name] += 1
            loss_hard_dict[dataset_name] += loss_hard.detach().float()

            if dataset_name in RETRIEVAL_DATASETS:
                for i, output in enumerate(outputs):
                    loss += inbatch_loss(output['query_passage_features'].squeeze(1), output['passage_passage_features'].squeeze(1), criterion, accelerator, use_mrl=args.use_mrl, mrl_dims=mrl_dims)
                count_dict[dataset_name] += 1
                loss_dict[dataset_name] += loss.detach().float()
            
            loss_total = (loss + loss_hard) / len(outputs)

            # backward, optimizer, scheduler
            accelerator.backward(loss_total)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if optimizer.param_groups[0]['lr'] < args.min_lr:
                optimizer.param_groups[0]['lr'] = args.min_lr
            
            # log
            completed_steps += 1
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

            # step checkpoint (and for ckpt merging)
            if completed_steps == 5 or args.checkpointing_steps and (completed_steps % args.checkpointing_steps == 0 or (completed_steps % 500 == 0 and completed_steps > args.train_steps - 2500)):
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