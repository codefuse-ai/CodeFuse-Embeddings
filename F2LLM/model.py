import torch
from transformers import AutoModel, AutoTokenizer, GPT2LMHeadModel, AutoModelForCausalLM
import warnings


class F2LLM:
    def __init__(self,
                 model_path,
                 max_seq_length=512,
                 args=None
                 ):

        self.args = args
        self.dtype = torch.bfloat16
        self.device = None # set after accelerator.prepare
        
        # 根据配置选择注意力实现方式
        attn_implementation = getattr(args, 'attn_implementation', 'flash_attention_2') if args else 'flash_attention_2'
        use_flash_attention = getattr(args, 'use_flash_attention', True) if args else True
        
        # 尝试加载模型，支持多种decoder-only模型
        try:
            if use_flash_attention and attn_implementation:
                # 使用配置的注意力实现
                self.lm = AutoModelForCausalLM.from_pretrained(
                    model_path, 
                    trust_remote_code=True, 
                    torch_dtype=self.dtype, 
                    attn_implementation=attn_implementation
                )
            else:
                # 不使用特殊注意力实现
                self.lm = AutoModelForCausalLM.from_pretrained(
                    model_path, 
                    trust_remote_code=True, 
                    torch_dtype=self.dtype
                )
        except Exception as e:
            if use_flash_attention and attn_implementation:
                warnings.warn(f"Failed to load model with {attn_implementation}: {e}. Trying fallback options...")
            
            # 回退策略
            fallback_options = ['sdpa', None]  # 尝试sdpa，然后是不使用特殊注意力
            loaded = False
            
            for fallback_attn in fallback_options:
                try:
                    if fallback_attn:
                        self.lm = AutoModelForCausalLM.from_pretrained(
                            model_path, 
                            trust_remote_code=True, 
                            torch_dtype=self.dtype,
                            attn_implementation=fallback_attn
                        )
                    else:
                        self.lm = AutoModelForCausalLM.from_pretrained(
                            model_path, 
                            trust_remote_code=True, 
                            torch_dtype=self.dtype
                        )
                    warnings.warn(f"Successfully loaded model with {fallback_attn or 'default'} attention")
                    loaded = True
                    break
                except Exception as e2:
                    warnings.warn(f"Failed to load model with {fallback_attn or 'default'} attention: {e2}")
                    continue
            
            if not loaded:
                raise RuntimeError(f"Failed to load model {model_path} with any attention implementation")
        
        self.lm.config.use_cache = False
        
        # 加载分词器，添加trust_remote_code支持更多模型
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True,
            padding_side='right'  # 大多数decoder-only模型需要右侧填充
        )
        
        # 确保分词器有pad_token
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                # 添加新的pad_token
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                # 需要调整模型embedding大小
                self.lm.resize_token_embeddings(len(self.tokenizer))
        
        self.max_seq_length = max_seq_length

    def set_device(self):
        self.device = self.lm.device
    
    def forward(self, batch):
        bs = batch['bs']
        num_hard_neg = int((len(batch['input_ids']) - 2*bs) / bs)

        outputs = self.lm(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            return_dict=True,
            output_hidden_states=True
        )

        # 对于CausalLM模型，获取最后一层的隐藏状态
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            # hidden_states是一个元组，包含所有层的隐藏状态
            passage_features_all_tokens = outputs.hidden_states[-1]
        elif hasattr(outputs, 'last_hidden_state'):
            passage_features_all_tokens = outputs.last_hidden_state
        else:
            # 回退到使用transformer的输出
            passage_features_all_tokens = outputs[0]

        return {
            'query_passage_features': torch.stack([passage_features_all_tokens[i, [batch['seq_lens'][i]-1]] for i in range(bs)]),
            'passage_passage_features': torch.stack([passage_features_all_tokens[i, [batch['seq_lens'][i]-1]] for i in range(bs, 2*bs)]),
            'negative_passage_features': None if num_hard_neg == 0 else torch.stack([passage_features_all_tokens[i, [batch['seq_lens'][i]-1]] for i in range(2*bs, len(batch['seq_lens']))]).view(bs, num_hard_neg, -1)
        }

