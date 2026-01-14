import torch
from transformers import AutoModel, AutoTokenizer
from utils import detect_model_type, extract_cls_embeddings, extract_mean_pooling_embeddings, extract_last_token_embeddings
import flash_attn
from packaging import version


class F2LLM:
    def __init__(self,
                 model_path,
                 max_seq_length=512,
                 args=None
                 ):

        self.args = args
        self.dtype = torch.bfloat16
        self.device = None # set after accelerator.prepare
        self.model_type = detect_model_type(model_path)

        flash_attn_version = getattr(flash_attn, '__version__', '0.0.0')
        no_support_deterministic = version.parse(flash_attn_version) < version.parse("2.6.0")
        
        if self.model_type == 'encoder_only' and no_support_deterministic:
            attn_implementation = 'eager'
        else:
            attn_implementation = 'flash_attention_2'
        
        print(f"Model Type: {self.model_type} | Attention Implementation: {attn_implementation}")
        self.lm = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            attn_implementation=attn_implementation
        )

        self.lm.config.use_cache = False
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_seq_length = max_seq_length
        self.use_cls_pooling = self._should_use_cls_pooling()
    
    def _should_use_cls_pooling(self):
        if self.model_type != 'encoder_only':
            return False
        return not hasattr(self.lm, 'pooler') or self.lm.pooler is None
    
    def set_device(self):
        self.device = self.lm.device
    
    def forward(self, batch):
        bs = batch['bs']
        num_hard_neg = int((len(batch['input_ids']) - 2*bs) / bs)

        outputs = self.lm(batch['input_ids'],
                        batch['attention_mask'],
                        )
        
        if self.model_type == 'decoder_only':
            return extract_last_token_embeddings(bs, num_hard_neg, outputs.last_hidden_state, batch)
        else:
            if self.use_cls_pooling:
                return extract_cls_embeddings(bs, num_hard_neg, outputs.last_hidden_state, batch)
            else:
                return extract_mean_pooling_embeddings(bs, num_hard_neg, outputs.last_hidden_state, batch)
