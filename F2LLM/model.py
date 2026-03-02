import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


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

    def set_device(self):
        self.device = self.lm.device
    
    def forward(self, batch, accelerator=None):
        bs = batch['bs']
        num_hard_neg = int((len(batch['input_ids']) - 2*bs) / bs)

        outputs = self.lm(batch['input_ids'],
                        batch['attention_mask'],
                        )

        passage_features_all_tokens = outputs.last_hidden_state
        return [{
            'query_passage_features': passage_features_all_tokens[torch.arange(bs), batch['seq_lens'][:bs] - 1].unsqueeze(1),
            'passage_passage_features': passage_features_all_tokens[torch.arange(bs)+bs, batch['seq_lens'][bs:2*bs] - 1].unsqueeze(1),
            'negative_passage_features': None if num_hard_neg == 0 else passage_features_all_tokens[torch.arange(bs*num_hard_neg)+2*bs, batch['seq_lens'][2*bs:len(batch['seq_lens'])] - 1].unsqueeze(1).view(bs, num_hard_neg, -1)
        }]
