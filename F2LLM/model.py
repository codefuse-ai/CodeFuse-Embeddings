import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig


class F2LLM:
    def __init__(self,
                 model_path,
                 max_seq_length=512,
                 args=None
                 ):

        self.args = args
        self.dtype = torch.bfloat16
        self.device = None # set after accelerator.prepare
        config = AutoConfig.from_pretrained(model_path)
        encoder_archs = ['BertModel', 'RobertaModel', 'DebertaModel', 'ElectraModel', 'AlbertModel', 'DistilBertModel']

        # Allow explicit override via args.model_arch; otherwise infer from config
        if self.args and getattr(self.args, 'model_arch', None):
            arch_flag = self.args.model_arch.lower()
            self.is_encoder_only = arch_flag == 'encoder'
        else:
            self.is_encoder_only = any(arch in getattr(config, 'architectures', []) for arch in encoder_archs)

        # Choose attention impl: prefer flash_attention_2 when available on CUDA for decoders; otherwise fallback to eager
        if not self.is_encoder_only and torch.cuda.is_available():
            try:
                import flash_attn  # noqa: F401
                attn_impl = 'flash_attention_2'
            except Exception:
                attn_impl = 'eager'
        else:
            attn_impl = 'eager'
        self.lm = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            attn_implementation=attn_impl
        )
        if not self.is_encoder_only:
            self.lm.config.use_cache = False
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_seq_length = max_seq_length

    def set_device(self):
        self.device = self.lm.device
    
    def forward(self, batch):
        bs = batch['bs']
        num_hard_neg = int((len(batch['input_ids']) - 2*bs) / bs)

        outputs = self.lm(
            batch['input_ids'],
            batch['attention_mask'],
        )

        hidden = outputs.last_hidden_state  # [total_bs, seq_len, dim]

        # Pooling per-architecture
        if self.is_encoder_only:
            pooling = getattr(self.args, 'pooling', 'cls') if self.args else 'cls'
            if pooling == 'mean':
                mask = batch['attention_mask'].unsqueeze(-1)  # [B, L, 1]
                summed = (hidden * mask).sum(dim=1, keepdim=True)
                lengths = mask.sum(dim=1, keepdim=True).clamp_min(1)
                pooled = summed / lengths
            elif pooling == 'cls_mean':
                mask = batch['attention_mask'].unsqueeze(-1)
                summed = (hidden * mask).sum(dim=1, keepdim=True)
                lengths = mask.sum(dim=1, keepdim=True).clamp_min(1)
                mean_pooled = summed / lengths
                pooled = 0.5 * (hidden[:, 0:1, :] + mean_pooled)
            else:  # default CLS
                pooled = hidden[:, 0:1, :]
        else:
            # decoder-style: last non-pad token representation
            pooled = torch.stack([hidden[i, [batch['seq_lens'][i]-1]] for i in range(len(batch['seq_lens']))])

        return {
            'query_passage_features': pooled[:bs],
            'passage_passage_features': pooled[bs:2*bs],
            'negative_passage_features': None if num_hard_neg == 0 else pooled[2*bs:].view(bs, num_hard_neg, -1)
        }

