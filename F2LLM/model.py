import torch
from transformers import AutoModel, AutoTokenizer
from lora_config import LoRAConfig, apply_lora_to_model


class F2LLM:
    def __init__(self,
                 model_path,
                 max_seq_length=512,
                 args=None,
                 use_lora=False,
                 lora_config=None
                 ):

        self.args = args
        self.dtype = torch.bfloat16
        self.device = None # set after accelerator.prepare
        self.use_lora = use_lora
        
        # Try flash_attention_2 first, fall back to eager if not available
        try:
            self.lm = AutoModel.from_pretrained(
                model_path, 
                trust_remote_code=True, 
                torch_dtype=self.dtype, 
                attn_implementation='flash_attention_2'
            )
        except (ImportError, ValueError):
            # Flash attention not available, use default
            self.lm = AutoModel.from_pretrained(
                model_path, 
                trust_remote_code=True, 
                torch_dtype=self.dtype
            )
        
        self.lm.config.use_cache = False
        
        # Apply LoRA if enabled
        if self.use_lora:
            if lora_config is None:
                lora_config = LoRAConfig()
            self.lm = apply_lora_to_model(self.lm, lora_config)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_seq_length = max_seq_length

    def set_device(self):
        self.device = self.lm.device
    
    def forward(self, batch):
        bs = batch['bs']
        num_hard_neg = int((len(batch['input_ids']) - 2*bs) / bs)

        outputs = self.lm(batch['input_ids'],
                        batch['attention_mask'],
                        )

        passage_features_all_tokens = outputs.last_hidden_state
        return {
            'query_passage_features': torch.stack([passage_features_all_tokens[i, [batch['seq_lens'][i]-1]] for i in range(bs)]),
            'passage_passage_features': torch.stack([passage_features_all_tokens[i, [batch['seq_lens'][i]-1]] for i in range(bs, 2*bs)]),
            'negative_passage_features': None if num_hard_neg == 0 else torch.stack([passage_features_all_tokens[i, [batch['seq_lens'][i]-1]] for i in range(2*bs, len(batch['seq_lens']))]).view(bs, num_hard_neg, -1)
        }

