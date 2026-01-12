import torch
from transformers import AutoModel, AutoTokenizer
import os
import json
from abc import ABC, abstractmethod


class BaseModelAdapter(ABC):
    """Base model adapter interface"""
    
    def __init__(self, model_path, max_seq_length=512, args=None):
        self.model_path = model_path
        self.max_seq_length = max_seq_length
        self.args = args
        self.dtype = torch.bfloat16
        self.device = None
        
    @abstractmethod
    def load_model(self):
        """Load model"""
        pass
    
    @abstractmethod
    def load_tokenizer(self):
        """Load tokenizer"""
        pass
    
    def get_model_config(self):
        """Get model configuration"""
        config_path = os.path.join(self.model_path, 'config.json')
        if os.path.exists(config_path):
            with open(config_path) as f:
                return json.load(f)
        return {}


class QwenAdapter(BaseModelAdapter):
    """Qwen series model adapter (Qwen, Qwen2, Qwen3)"""
    
    def load_model(self):
        return AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            attn_implementation='flash_attention_2'
        )
    
    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_path)


class LlamaAdapter(BaseModelAdapter):
    """Llama series model adapter (Llama-2, Llama-3, CodeLlama)"""
    
    def load_model(self):
        return AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            attn_implementation='flash_attention_2'
        )
    
    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer