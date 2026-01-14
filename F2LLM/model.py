import torch
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


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

        if args and args.use_lora:
            self._apply_lora()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_seq_length = max_seq_length
    
    def _apply_lora(self):
        """Apply LoRA to the model if enabled."""
        # Process target modules
        if self.args.lora_target_modules == "all-linear":
            # For decoder-only models, common target modules are linear layers
            target_modules = [
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                "lm_head"
            ]
        else:
            target_modules = [module.strip() for module in self.args.lora_target_modules.split(",")]

        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,  # Feature extraction for embedding models
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=self.args.lora_dropout,
            bias="none",
            modules_to_save=[],  # We don't need to save any additional modules
        )

        self.lm = get_peft_model(self.lm, lora_config)

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

