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
        self.lm = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=self.dtype, attn_implementation='flash_attention_2')
        self.lm.config.use_cache = False
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_seq_length = max_seq_length
        
        # Apply LoRA if enabled
        if args and args.use_lora:
            self._apply_lora()
            
        # Enable gradient requirements for LoRA with flash attention
        if hasattr(self.lm, 'enable_input_require_grads'):
            self.lm.enable_input_require_grads()

    def _apply_lora(self):
        """Apply LoRA adaptation to the model"""
        # Print LoRA training message
        print("Using LoRA training, optimizing only LoRA parameters")
        
        target_modules = self.args.lora_target_modules.split(",") if self.args.lora_target_modules else None
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,  # For decoder-only models
            inference_mode=False,
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            target_modules=target_modules
        )
        
        self.lm = get_peft_model(self.lm, peft_config)
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
