import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPast
from functools import partial
import random


class F2LLM(nn.Module):
    def __init__(self,
                 model_path,
                 max_seq_length=512,
                 args=None,
                 accelerator=None
                 ):
        super().__init__()
        self.args = args
        self.dtype = torch.bfloat16
        self.device = None # accelerator.prepare后设置
        self.lm = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=self.dtype, attn_implementation='flash_attention_2')
        self.lm.config.use_cache = False
        self.hidden_size = self.lm.config.hidden_size
        self.num_hidden_layers = self.lm.config.num_hidden_layers
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_seq_length = max_seq_length
        # mll
        if args.use_mll:
            mll_layers = [1, 2, 4, 8, 16, 24, 32, 40, 48] # layer 0 is the embedding layer
            mll_layers = [l for l in mll_layers if l <= self.num_hidden_layers]
            if self.num_hidden_layers not in mll_layers:
                mll_layers = mll_layers + [self.num_hidden_layers]
            self.mll_layers = mll_layers[::-1]
        else:
            self.mll_layers = None
        # embedding decomposition
        if args.r > 0:
            with torch.no_grad():
                U, S, V = torch.linalg.svd(self.lm.embed_tokens.weight.float().to(f'cuda:{accelerator.local_process_index}'), full_matrices=False)
                U = U[:,:args.r] @ torch.diag(S[:args.r])
                V = V[:args.r,:]
            self.U = nn.Parameter(U.to(self.dtype).cpu())
            self.V = nn.Parameter(V.to(self.dtype).cpu())
            if args.use_mel:
                embedding_ranks = [2**i for i in range(6, 15)]
                embedding_ranks = [r for r in embedding_ranks if r <= args.r]
                if args.r not in embedding_ranks:
                    embedding_ranks = embedding_ranks + [args.r]
                self.embedding_ranks = embedding_ranks[::-1]
                ranks_weights = [r / self.embedding_ranks[0] for r in self.embedding_ranks]
                weights_sum = sum(ranks_weights)
                self.embedding_rank_weights = [r / weights_sum for r in ranks_weights]
            else:
                self.embedding_ranks = None
                self.embedding_rank_weights = None

    def set_device(self):
        self.device = self.lm.device
    
    def forward(self, batch, accelerator=None):
        bs = batch['bs']
        num_hard_neg = int((len(batch['input_ids']) - 2*bs) / bs)

        ################################################# Qwen3 forward start
        output_hidden_states = self.args.use_mll
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        if self.args.r > 0:
            if self.args.use_mel:
                # method 1: dynamic rank sampling
                r = random.choices(self.embedding_ranks, weights=self.embedding_rank_weights)[0]
                if accelerator is not None:
                    accelerator.print(f"Embedding rank: {r}")
                inputs_embeds = (self.U[:, :r] @ self.V[:r,:])[input_ids]

                # method 2: summing over all ranks in each step
                # embedding_matrix = 0.0
                # for r, w in zip(self.embedding_ranks, self.embedding_rank_weights):
                #     embedding_matrix += (w / sum(self.embedding_rank_weights)) * (self.U[:, :r] @ self.V[:r,:])
                # inputs_embeds = embedding_matrix[input_ids]
            else:
                inputs_embeds = (self.U @ self.V)[input_ids]
        else:
            inputs_embeds = self.lm.embed_tokens(input_ids)

        cache_position = torch.arange(
            0, inputs_embeds.shape[1], device=inputs_embeds.device
        )
        position_ids = cache_position.unsqueeze(0)

        causal_mask = self.lm._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, None
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.lm.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None

        for l, decoder_layer in enumerate(self.lm.layers[: self.num_hidden_layers]):
            # print(f"Layer: {l+1}")
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            # regular forward
            if self.lm.gradient_checkpointing and self.lm.training:
                layer_outputs = self.lm._gradient_checkpointing_func(
                    partial(decoder_layer.__call__),
                    hidden_states,
                    causal_mask,
                    position_ids,
                    None,
                    False,
                    False,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

        hidden_states = self.lm.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        outputs = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )
        ################################################# Qwen3 forward end

        # outputs = self.lm(batch['input_ids'],
        #                 batch['attention_mask'],
        #                 )

        if not self.args.use_mll:
            passage_features_all_tokens = outputs.last_hidden_state
            return [{
                'query_passage_features': passage_features_all_tokens[torch.arange(bs), batch['seq_lens'][:bs] - 1].unsqueeze(1),
                'passage_passage_features': passage_features_all_tokens[torch.arange(bs)+bs, batch['seq_lens'][bs:2*bs] - 1].unsqueeze(1),
                'negative_passage_features': None if num_hard_neg == 0 else passage_features_all_tokens[torch.arange(bs*num_hard_neg)+2*bs, batch['seq_lens'][2*bs:len(batch['seq_lens'])] - 1].unsqueeze(1).view(bs, num_hard_neg, -1)
            }]
        else:
            embeddings = []
            for l in self.mll_layers:
                hidden_states = outputs.hidden_states[l] if l == self.num_hidden_layers else self.lm.norm(outputs.hidden_states[l])
            
                embeddings.append({
                    'query_passage_features': hidden_states[torch.arange(bs), batch['seq_lens'][:bs] - 1].unsqueeze(1),
                    'passage_passage_features': hidden_states[torch.arange(bs)+bs, batch['seq_lens'][bs:2*bs] - 1].unsqueeze(1),
                    'negative_passage_features': None if num_hard_neg == 0 else hidden_states[torch.arange(bs*num_hard_neg)+2*bs, batch['seq_lens'][2*bs:len(batch['seq_lens'])] - 1].unsqueeze(1).view(bs, num_hard_neg, -1)
                })

            return embeddings