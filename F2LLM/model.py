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
        self.lm = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=self.dtype, attn_implementation='flash_attention_2')
        self.lm.config.use_cache = False
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_seq_length = max_seq_length

    def set_device(self, device=None):
        """
        Set device - compatible with both Accelerate and Ray Train

        Args:
            device: Specific device to use. If None, auto-detect from model.
        """
        if device is not None:
            self.device = device
        elif hasattr(self.lm, 'device'):
            # Accelerate path: model already has device attribute
            self.device = self.lm.device
        else:
            # Ray Train path: get device from model parameters
            self.device = next(self.lm.parameters()).device
    
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

