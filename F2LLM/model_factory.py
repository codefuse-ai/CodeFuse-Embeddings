"""
Model Factory for Dynamic Model Instantiation

This module provides a factory pattern for creating models with
proper configuration and handling of different model families.
"""

import torch
from typing import Optional, Dict, Any
from transformers import AutoModel, AutoTokenizer
import logging

from model_registry import (
    ModelConfig, 
    get_registry, 
    AttentionType,
)

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory for creating and configuring models"""
    
    def __init__(self):
        self.registry = get_registry()
        self._model_family_handlers = {
            'qwen3': self._configure_qwen_model,
            'qwen': self._configure_qwen_model,
            'llama2': self._configure_llama_model,
            'llama3': self._configure_llama_model,
            'mistral': self._configure_mistral_model,
            'phi': self._configure_phi_model,
            'code-llama': self._configure_code_llama_model,
            'gemma': self._configure_gemma_model,
        }
    
    def create_model(
        self,
        model_path: str,
        model_id: Optional[str] = None,
        use_flash_attention: bool = True,
        torch_dtype: torch.dtype = torch.bfloat16,
        **kwargs
    ) -> torch.nn.Module:
        """
        Create a model with appropriate configuration.
        
        Args:
            model_path: Path or HF model ID
            model_id: Optional model registry ID for configuration
            use_flash_attention: Whether to use Flash Attention 2
            torch_dtype: Data type for model
            **kwargs: Additional arguments passed to AutoModel.from_pretrained
        
        Returns:
            Configured model instance
        """
        
        # Get model configuration if provided
        model_config = None
        if model_id and self.registry.supports_model(model_id):
            model_config = self.registry.get(model_id)
            logger.info(f"Using configuration for model: {model_id}")
        else:
            logger.info(f"No explicit configuration found for {model_id}. Using defaults.")
        
        # Set up model loading arguments
        model_kwargs = {
            'trust_remote_code': True,
            'torch_dtype': torch_dtype,
            **kwargs
        }
        
        # Handle attention mechanism (only when CUDA is available)
        if use_flash_attention and (model_config is None or model_config.supports_flash_attention_2):
            if torch.cuda.is_available():
                model_kwargs['attn_implementation'] = 'flash_attention_2'
                logger.info("Enabling Flash Attention 2")
            else:
                logger.info("Flash Attention requested but no CUDA device found. Using standard attention.")
        
        # Load model
        logger.info(f"Loading model from: {model_path}")
        model = AutoModel.from_pretrained(model_path, **model_kwargs)
        
        # Apply model family-specific configurations
        if model_config:
            handler = self._model_family_handlers.get(model_config.family)
            if handler:
                logger.info(f"Applying {model_config.family} family configuration")
                model = handler(model, model_config)
        
        # Disable cache and other optimizations
        model.config.use_cache = False
        
        return model
    
    def create_tokenizer(
        self,
        model_path: str,
        model_id: Optional[str] = None,
        **kwargs
    ) -> AutoTokenizer:
        """
        Create a tokenizer with appropriate configuration.
        
        Args:
            model_path: Path or HF model ID
            model_id: Optional model registry ID for configuration
            **kwargs: Additional arguments passed to AutoTokenizer.from_pretrained
        
        Returns:
            Configured tokenizer instance
        """
        
        # Get model configuration
        tokenizer_kwargs = {
            'trust_remote_code': True,
        }
        
        if model_id and self.registry.supports_model(model_id):
            model_config = self.registry.get(model_id)
            
            # Apply model-specific tokenizer settings
            if model_config.tokenizer_type.value == 'qwen':
                tokenizer_kwargs.update({
                    'padding_side': 'right',
                    'truncation_side': 'right',
                })
        
        # Override with user-provided kwargs
        tokenizer_kwargs.update(kwargs)
        
        logger.info(f"Loading tokenizer from: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
        
        return tokenizer
    
    # ============ Model Family Handlers ============
    
    def _configure_qwen_model(
        self, 
        model: torch.nn.Module, 
        config: ModelConfig
    ) -> torch.nn.Module:
        """Configure Qwen family models"""
        logger.debug(f"Configuring Qwen model with hidden_size={config.hidden_size}")
        return model
    
    def _configure_llama_model(
        self, 
        model: torch.nn.Module, 
        config: ModelConfig
    ) -> torch.nn.Module:
        """Configure LLaMA family models"""
        logger.debug(f"Configuring LLaMA model with GQA: {config.num_key_value_heads} kv heads")
        return model
    
    def _configure_mistral_model(
        self, 
        model: torch.nn.Module, 
        config: ModelConfig
    ) -> torch.nn.Module:
        """Configure Mistral family models"""
        logger.debug(f"Configuring Mistral model with sliding window attention")
        return model
    
    def _configure_phi_model(
        self, 
        model: torch.nn.Module, 
        config: ModelConfig
    ) -> torch.nn.Module:
        """Configure Phi family models"""
        logger.debug(f"Configuring Phi model")
        return model
    
    def _configure_code_llama_model(
        self, 
        model: torch.nn.Module, 
        config: ModelConfig
    ) -> torch.nn.Module:
        """Configure Code-LLaMA models"""
        logger.debug(f"Configuring Code-LLaMA model with extended context: {config.recommended_max_seq_length}")
        return model
    
    def _configure_gemma_model(
        self, 
        model: torch.nn.Module, 
        config: ModelConfig
    ) -> torch.nn.Module:
        """Configure Gemma family models"""
        logger.debug(f"Configuring Gemma model")
        return model
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a model"""
        if not self.registry.supports_model(model_id):
            return None
        
        config = self.registry.get(model_id)
        return {
            'id': config.model_id,
            'family': config.family,
            'name': config.display_name,
            'description': config.description,
            'hidden_size': config.hidden_size,
            'num_heads': config.num_attention_heads,
            'kv_heads': config.num_key_value_heads,
            'num_layers': config.num_hidden_layers,
            'vocab_size': config.vocab_size,
            'attention_type': config.attention_type.value,
            'position_embedding': config.position_embedding.value,
            'max_seq_length': config.recommended_max_seq_length,
            'recommended_memory_gb': config.recommended_memory_gb,
            'supports_flash_attention_2': config.supports_flash_attention_2,
            'supports_gradient_checkpointing': config.supports_gradient_checkpointing,
            'quantization_support': config.quantization_support,
            'hf_model_id': config.hf_model_id,
        }
    
    def list_available_models(self) -> Dict[str, Dict[str, str]]:
        """Get list of all available models organized by family"""
        result = {}
        for family in self.registry.list_families():
            models = self.registry.get_by_family(family)
            result[family] = {
                m.model_id: m.display_name for m in models
            }
        return result


# Global factory instance
_default_factory: Optional[ModelFactory] = None


def get_factory() -> ModelFactory:
    """Get or create the global model factory"""
    global _default_factory
    if _default_factory is None:
        _default_factory = ModelFactory()
    return _default_factory
