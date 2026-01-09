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
        
        # Load model config to determine architecture type
        config = AutoConfig.from_pretrained(model_path)
        
        # Determine if model is encoder-only (e.g., BERT, RoBERTa) or decoder-only (e.g., GPT, Qwen)
        self.is_encoder_only = any(arch in config.architectures for arch in ['BertModel', 'RobertaModel', 'DebertaModel', 'ElectraModel', 'AlbertModel', 'DistilBertModel'])
        
        # Load the model
        self.lm = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=self.dtype, attn_implementation='flash_attention_2' if not self.is_encoder_only else 'eager')
        
        # For decoder-only models, disable cache; for encoder-only models, no cache to disable
        if not self.is_encoder_only:
            self.lm.config.use_cache = False
            
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
        
        if self.is_encoder_only:
            # For encoder-only models, use [CLS] token (index 0) as sequence representation
            return {
                'query_passage_features': passage_features_all_tokens[0:bs, [0], :],  # [bs, 1, d]
                'passage_passage_features': passage_features_all_tokens[bs:2*bs, [0], :],  # [bs, 1, d]
                'negative_passage_features': None if num_hard_neg == 0 else passage_features_all_tokens[2*bs:, [0], :].view(bs, num_hard_neg, -1)  # [bs, num_hard_neg, d]
            }
        else:
            # For decoder-only models, use last non-padded token as sequence representation
            return {
                'query_passage_features': torch.stack([passage_features_all_tokens[i, [batch['seq_lens'][i]-1]] for i in range(bs)]),
                'passage_passage_features': torch.stack([passage_features_all_tokens[i, [batch['seq_lens'][i]-1]] for i in range(bs, 2*bs)]),
                'negative_passage_features': None if num_hard_neg == 0 else torch.stack([passage_features_all_tokens[i, [batch['seq_lens'][i]-1]] for i in range(2*bs, len(batch['seq_lens']))]).view(bs, num_hard_neg, -1)
            }

