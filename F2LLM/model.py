import torch
from transformers import AutoModel, AutoTokenizer


class F2LLM:
    def __init__(self,
                 model_path,
                 max_seq_length=512,
                 args=None
                 ):

        self.args = args
        self.dtype = torch.bfloat16
        self.device = None # set after accelerator.prepare
        
        # Determine model type based on args or auto-detection
        self.model_type = self._get_model_type(model_path, args)
        print("mode_type:" + self.model_type)
        
        # Load model based on type
        if self.model_type == 'encoder':
            # For encoder-only models like BERT
            self.lm = AutoModel.from_pretrained(model_path, trust_remote_code=True, dtype=self.dtype)
        else:
            # For decoder-only models like Qwen (default)
            self.lm = AutoModel.from_pretrained(model_path, trust_remote_code=True, dtype=self.dtype, attn_implementation='flash_attention_2')
            self.lm.config.use_cache = False
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_seq_length = max_seq_length

    def _get_model_type(self, model_path, args):
        """更精确的模型类型检测"""
        if args and hasattr(args, 'model_type') and args.model_type != 'auto':
            return args.model_type

        # 基于模型架构自动检测
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

            encoder_architectures = [
                'Bert', 'Roberta', 'Electra', 'Deberta', 'Albert', 'DistilBert'
            ]

            if hasattr(config, 'architectures'):
                archs = config.architectures
                for arch in archs:
                    if any(encoder_arch in arch for encoder_arch in encoder_architectures):
                        return 'encoder'

            # 检查是否有encoder_attention_heads但没有decoder_layers
            if (hasattr(config, 'encoder_attention_heads') and
                    not hasattr(config, 'decoder_layers')):
                return 'encoder'

        except Exception as e:
            print(f"Model type detection failed: {e}")

        return 'decoder'

    def set_device(self):
        self.device = self.lm.device

    def _masked_mean_pool(self, last_hidden, attention_mask):
        # last_hidden: [N, seq_len, dim], attention_mask: [N, seq_len]
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden)  # [N, seq_len, 1]
        summed = (last_hidden * mask).sum(dim=1)  # [N, dim]
        counts = mask.sum(dim=1).clamp(min=1e-9)  # [N, 1]
        return summed / counts  # [N, dim]

    def forward(self, batch):
        """
        Expects batch to contain:
          - 'input_ids': LongTensor [N, seq_len]
          - 'attention_mask': LongTensor [N, seq_len]
          - 'bs': int (number of queries per microbatch)
        Optional:
          - 'num_hard': int (number of hard negatives per query)
          - 'seq_lens': list or tensor of lengths per example (required for decoder-last-token pooling)
        Layout convention assumed:
          index 0..bs-1      => queries
          index bs..2*bs-1   => positive passages
          index 2*bs..       => negatives flattened (if any), expected bs * num_hard entries
        Returns dict with:
          - 'query_passage_features': Tensor [bs, 1, dim]
          - 'passage_passage_features': Tensor [bs, 1, dim]
          - 'negative_passage_features': None or Tensor [bs, num_hard, dim]
        """
        # move inputs to model device
        device = next(self.lm.parameters()).device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        bs = int(batch['bs'])
        num_hard = int(batch.get('num_hard', 0))

        # forward through backbone
        outputs = self.lm(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden = outputs.last_hidden_state  # [N, seq_len, dim]

        if self.model_type == 'encoder' or self.model_type == 'encoder-decoder':
            # Use masked-mean pooling (recommended for encoder)
            pooled = self._masked_mean_pool(last_hidden, attention_mask)  # [N, dim]
        else:
            # decoder-only: use last non-pad token via seq_lens (seq_lens must be provided)
            if 'seq_lens' not in batch:
                raise ValueError("For decoder model_type you must provide batch['seq_lens'] (list/tensor of sequence lengths).")
            seq_lens = torch.tensor(batch['seq_lens'], device=device, dtype=torch.long)  # [N]
            # clamp seq_lens (they are 1-based lengths); index = seq_len-1
            idx = (seq_lens - 1).clamp(min=0)
            # gather last token hidden states
            # gather requires expansion
            dim = last_hidden.size(-1)
            idx_exp = idx.view(-1, 1, 1).expand(-1, 1, dim)  # [N,1,dim]
            pooled = last_hidden.gather(1, idx_exp).squeeze(1)  # [N, dim]

        N = pooled.size(0)
        if N < 2*bs:
            raise ValueError(f"Batch size N={N} smaller than 2*bs={2*bs}. Check collator/layout.")

        q = pooled[0:bs]       # [bs, dim]
        p = pooled[bs:2*bs]    # [bs, dim]

        if num_hard == 0:
            neg = None
        else:
            expected = bs * num_hard
            neg_all = pooled[2*bs:2*bs + expected]
            if neg_all.size(0) != expected:
                raise ValueError(f"Expected {expected} negative vectors but got {neg_all.size(0)}. Check collator.")
            neg = neg_all.view(bs, num_hard, -1)  # [bs, num_hard, dim]

        # match training code expectation: return [bs,1,dim] for query/passage
        return {
            'query_passage_features': q.unsqueeze(1),    # [bs,1,dim]
            'passage_passage_features': p.unsqueeze(1),  # [bs,1,dim]
            'negative_passage_features': neg            # None or [bs,num_hard,dim]
        }

