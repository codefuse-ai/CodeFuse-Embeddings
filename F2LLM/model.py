import torch
from transformers import AutoModel, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType


class F2LLM:
    def __init__(self,
                 model_path,
                 max_seq_length=512,
                 args=None
                 ):

        self.args = args
        self.dtype = torch.bfloat16
        self.device = None # set after accelerator.prepare

        # Check if CUDA is available and set the attention implementation accordingly
        # Only use flash_attention_2 if CUDA is available and flash_attn is installed
        attn_implementation = None
        if torch.cuda.is_available():
            try:
                import flash_attn
                attn_implementation = 'flash_attention_2'
            except ImportError:
                attn_implementation = 'eager'  # or 'sdpa' if available

        self.lm = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=self.dtype, attn_implementation=attn_implementation)
        self.lm.config.use_cache = False
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_seq_length = max_seq_length

        # Add LoRA support
        if args and getattr(args, 'use_lora', False):
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=args.lora_target_modules
            )
            self.lm = get_peft_model(self.lm, peft_config)
            print("LoRA enabled")
            self.lm.print_trainable_parameters()

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

