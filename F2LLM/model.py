import torch
from transformers import AutoModel, AutoTokenizer
from utils import detect_encoder_only_model


class F2LLM:
    def __init__(self,
                 model_path,
                 max_seq_length=512,
                 args=None
                 ):

        self.args = args
        self.dtype = torch.bfloat16
        self.device = None # set after accelerator.prepare

        try:
            self.lm = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=self.dtype,
                attn_implementation='flash_attention_2'
            )
        except Exception as e:
            print(f"Flash Attention 2不可用,使用默认attention实现: {e}")
            self.lm = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=self.dtype
            )

        self.is_encoder_only = detect_encoder_only_model(self.lm.config)
        print(f"模型类型: {'Encoder-only' if self.is_encoder_only else 'Decoder-only'}")

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

        passage_features_all_tokens = outputs.last_hidden_state

        if self.is_encoder_only:
            query_features = passage_features_all_tokens[:bs, 0, :]
            passage_features = passage_features_all_tokens[bs:2*bs, 0, :]
            negative_features = None if num_hard_neg == 0 else \
                passage_features_all_tokens[2*bs:, 0, :].view(bs, num_hard_neg, -1)
        else:
            query_features = torch.stack([
                passage_features_all_tokens[i, batch['seq_lens'][i]-1]
                for i in range(bs)
            ])
            passage_features = torch.stack([
                passage_features_all_tokens[i, batch['seq_lens'][i]-1]
                for i in range(bs, 2*bs)
            ])
            negative_features = None if num_hard_neg == 0 else torch.stack([
                passage_features_all_tokens[i, batch['seq_lens'][i]-1]
                for i in range(2*bs, len(batch['seq_lens']))
            ]).view(bs, num_hard_neg, -1)

        return {
            'query_passage_features': query_features,
            'passage_passage_features': passage_features,
            'negative_passage_features': negative_features
        }

