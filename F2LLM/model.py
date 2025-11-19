import torch
from transformers import AutoModel, AutoTokenizer


class F2LLM:
    def __init__(self,
                 model_path,
                 max_seq_length=512,
                 args=None,
                 use_multi_gpu=False
                 ):

        self.args = args
        self.dtype = torch.bfloat16
        self.device = None # set after accelerator.prepare
        self.use_multi_gpu = use_multi_gpu
        
        # Check if the model supports flash attention
        # BERT models don't support flash attention 2.0
        if 'bert' in model_path.lower():
            self.lm = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=self.dtype)
        else:
            # For multi-GPU, we might need to adjust attention implementation
            try:
                self.lm = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=self.dtype, attn_implementation='flash_attention_2')
            except:
                # Fallback to default attention if flash attention is not available
                self.lm = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=self.dtype)
        
        self.lm.config.use_cache = False
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_seq_length = max_seq_length

    def set_device(self, device=None):
        if device is not None:
            self.device = device
            self.lm.to(device)
        else:
            self.device = next(self.lm.parameters()).device
    
    def forward(self, batch):
        bs = batch['bs']
        num_hard_neg = int((len(batch['input_ids']) - 2*bs) / bs)

        # Move batch to device if needed
        if self.device is not None:
            batch['input_ids'] = batch['input_ids'].to(self.device)
            batch['attention_mask'] = batch['attention_mask'].to(self.device)

        outputs = self.lm(batch['input_ids'],
                        batch['attention_mask'],
                        )

        passage_features_all_tokens = outputs.last_hidden_state
        return {
            'query_passage_features': torch.stack([passage_features_all_tokens[i, [batch['seq_lens'][i]-1]] for i in range(bs)]),
            'passage_passage_features': torch.stack([passage_features_all_tokens[i, [batch['seq_lens'][i]-1]] for i in range(bs, 2*bs)]),
            'negative_passage_features': None if num_hard_neg == 0 else torch.stack([passage_features_all_tokens[i, [batch['seq_lens'][i]-1]] for i in range(2*bs, len(batch['seq_lens']))]).view(bs, num_hard_neg, -1)
        }

