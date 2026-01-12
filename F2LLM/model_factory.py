import os
import json
from typing import Dict, Type
from model_adapters import BaseModelAdapter, QwenAdapter, LlamaAdapter


class ModelFactory:
    """Model factory for creating adapters based on model type"""
    
    # Mapping of model types to adapters
    MODEL_ADAPTERS: Dict[str, Type[BaseModelAdapter]] = {
        'qwen': QwenAdapter,
        'qwen2': QwenAdapter,
        'qwen3': QwenAdapter,
        'llama': LlamaAdapter,
    }
    
    @classmethod
    def create_adapter(cls, model_path: str, max_seq_length: int = 512, args=None) -> BaseModelAdapter:
        """Create adapter based on model path and type"""
        model_type = cls.detect_model_type(model_path)
        adapter_class = cls.MODEL_ADAPTERS.get(model_type)
        
        if not adapter_class:
            # Use LlamaAdapter as fallback for unknown model types
            print(f"Warning: Unknown model type '{model_type}', using LlamaAdapter as fallback")
            adapter_class = LlamaAdapter
        
        return adapter_class(model_path, max_seq_length, args)
    
    @classmethod
    def detect_model_type(cls, model_path: str) -> str:
        """Detect model type"""
        # Method 1: Detect via config file
        config_path = os.path.join(model_path, 'config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path) as f:
                    config = json.load(f)
                model_type = config.get('model_type', '').lower()
                if model_type:
                    return model_type
            except Exception:
                pass
        
        # Method 2: Infer from path name
        path_lower = model_path.lower()
        model_type_mappings = {
            'qwen': ['qwen', 'qwen2', 'qwen3'],
            'llama': ['llama', 'llama-2', 'llama-3', 'meta-llama', 'codellama'],
        }
        
        for model_type, keywords in model_type_mappings.items():
            for keyword in keywords:
                if keyword in path_lower:
                    return model_type
        
        # Method 3: Detect via folder structure
        if os.path.exists(os.path.join(model_path, 'tokenizer_config.json')):
            try:
                tokenizer_config_path = os.path.join(model_path, 'tokenizer_config.json')
                with open(tokenizer_config_path) as f:
                    tokenizer_config = json.load(f)
                tokenizer_class = tokenizer_config.get('tokenizer_class', '').lower()
                
                if 'qwen' in tokenizer_class:
                    return 'qwen'
                elif 'llama' in tokenizer_class:
                    return 'llama'
            except Exception:
                pass
        
        return 'unknown'
    
    @classmethod
    def list_supported_models(cls) -> list:
        """Return list of supported model types"""
        return list(cls.MODEL_ADAPTERS.keys())
    
    @classmethod
    def get_model_info(cls, model_path: str) -> dict:
        """Get model information"""
        model_type = cls.detect_model_type(model_path)
        adapter_class = cls.MODEL_ADAPTERS.get(model_type)
        
        info = {
            'model_path': model_path,
            'detected_type': model_type,
            'adapter_class': adapter_class.__name__ if adapter_class else 'Unknown',
            'is_supported': model_type in cls.MODEL_ADAPTERS
        }
        
        # Try to get model configuration info
        config_path = os.path.join(model_path, 'config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path) as f:
                    config = json.load(f)
                info.update({
                    'model_name': config.get('_name_or_path', 'Unknown'),
                    'vocab_size': config.get('vocab_size', 0),
                    'hidden_size': config.get('hidden_size', 0),
                    'num_layers': config.get('num_hidden_layers', 0),
                    'num_attention_heads': config.get('num_attention_heads', 0),
                    'max_position_embeddings': config.get('max_position_embeddings', 0)
                })
            except Exception as e:
                info['config_error'] = str(e)
        
        return info