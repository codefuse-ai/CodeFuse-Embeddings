"""
Model Registry System for CodeFuse-Embeddings

This module provides a centralized registry for supported base models,
enabling easy addition of new models and configuration management.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List
from enum import Enum


class AttentionType(Enum):
    """Supported attention mechanisms"""
    FLASH_ATTENTION_2 = "flash_attention_2"
    STANDARD = "standard"
    MULTI_QUERY = "multi_query"
    GROUPED_QUERY = "grouped_query"


class PositionEmbeddingType(Enum):
    """Supported position embedding types"""
    ROPE = "rope"
    ABSOLUTE = "absolute"
    ALIBI = "alibi"


class TokenizerType(Enum):
    """Supported tokenizer types"""
    BPE = "bpe"
    SENTENCEPIECE = "sentencepiece"
    QWEN = "qwen"
    CUSTOM = "custom"


@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    
    # Basic model information
    model_id: str
    family: str  # e.g., 'qwen3', 'llama2', 'mistral'
    display_name: str
    description: str = ""
    
    # Architecture details
    hidden_size: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: Optional[int] = None  # For GQA/MQA models
    intermediate_size: Optional[int] = None
    num_hidden_layers: int = 0
    vocab_size: int = 0
    
    # Attention configuration
    attention_type: AttentionType = AttentionType.FLASH_ATTENTION_2
    position_embedding: PositionEmbeddingType = PositionEmbeddingType.ROPE
    rope_theta: float = 1000000.0
    rope_scaling: Optional[Dict] = None
    
    # Tokenizer configuration
    tokenizer_type: TokenizerType = TokenizerType.BPE
    max_position_embeddings: int = 4096
    eos_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    unk_token_id: Optional[int] = None
    
    # Training recommendations
    recommended_max_seq_length: int = 2048
    recommended_batch_size: int = 32
    supports_flash_attention_2: bool = True
    supports_gradient_checkpointing: bool = True
    
    # Hardware requirements
    recommended_memory_gb: float = 16.0
    quantization_support: List[str] = field(default_factory=lambda: ["fp32", "fp16", "bf16"])
    
    # Additional metadata
    release_date: str = ""
    paper_url: str = ""
    hf_model_id: str = ""  # Hugging Face model ID
    notes: str = ""


class ModelRegistry:
    """Central registry for all supported models"""
    
    def __init__(self):
        self._registry: Dict[str, ModelConfig] = {}
        self._init_default_models()
    
    def _init_default_models(self):
        """Initialize registry with default supported models"""
        
        # ============ Qwen Series ============
        self.register(ModelConfig(
            model_id="qwen3-0.6b",
            family="qwen3",
            display_name="Qwen3 0.6B",
            description="Small efficient Qwen3 model",
            hidden_size=1152,
            num_attention_heads=16,
            intermediate_size=6144,
            num_hidden_layers=24,
            vocab_size=152064,
            attention_type=AttentionType.FLASH_ATTENTION_2,
            position_embedding=PositionEmbeddingType.ROPE,
            tokenizer_type=TokenizerType.QWEN,
            recommended_max_seq_length=1024,
            recommended_memory_gb=4.0,
            hf_model_id="Qwen/Qwen3-0.6B",
        ))
        
        self.register(ModelConfig(
            model_id="qwen3-1.7b",
            family="qwen3",
            display_name="Qwen3 1.7B",
            description="Small-medium Qwen3 model",
            hidden_size=2048,
            num_attention_heads=32,
            intermediate_size=8704,
            num_hidden_layers=24,
            vocab_size=152064,
            attention_type=AttentionType.FLASH_ATTENTION_2,
            position_embedding=PositionEmbeddingType.ROPE,
            tokenizer_type=TokenizerType.QWEN,
            recommended_max_seq_length=1024,
            recommended_memory_gb=8.0,
            hf_model_id="Qwen/Qwen3-1.7B",
        ))
        
        self.register(ModelConfig(
            model_id="qwen3-4b",
            family="qwen3",
            display_name="Qwen3 4B",
            description="Medium Qwen3 model",
            hidden_size=3072,
            num_attention_heads=32,
            intermediate_size=8704,
            num_hidden_layers=32,
            vocab_size=152064,
            attention_type=AttentionType.FLASH_ATTENTION_2,
            position_embedding=PositionEmbeddingType.ROPE,
            tokenizer_type=TokenizerType.QWEN,
            recommended_max_seq_length=2048,
            recommended_memory_gb=16.0,
            hf_model_id="Qwen/Qwen3-4B",
        ))
        
        # (Removed LLaMA series - requires gated access)
        
        # ============ Mistral Series ============
        self.register(ModelConfig(
            model_id="mistral-7b",
            family="mistral",
            display_name="Mistral 7B",
            description="Mistral AI's 7B model with GQA",
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=8,
            intermediate_size=14336,
            num_hidden_layers=32,
            vocab_size=32000,
            attention_type=AttentionType.GROUPED_QUERY,
            position_embedding=PositionEmbeddingType.ROPE,
            rope_theta=10000.0,
            tokenizer_type=TokenizerType.BPE,
            recommended_max_seq_length=8192,
            recommended_memory_gb=16.0,
            hf_model_id="mistralai/Mistral-7B-v0.1",
            paper_url="https://arxiv.org/abs/2310.06825",
        ))
        
        # ============ Phi Series ============
        self.register(ModelConfig(
            model_id="phi-2",
            family="phi",
            display_name="Phi-2",
            description="Microsoft's Phi-2 2.7B model",
            hidden_size=2560,
            num_attention_heads=32,
            num_key_value_heads=32,
            intermediate_size=6912,
            num_hidden_layers=32,
            vocab_size=50256,
            attention_type=AttentionType.STANDARD,
            position_embedding=PositionEmbeddingType.ABSOLUTE,
            tokenizer_type=TokenizerType.BPE,
            recommended_max_seq_length=4096,
            recommended_memory_gb=12.0,
            hf_model_id="microsoft/phi-2",
        ))
        
        self.register(ModelConfig(
            model_id="phi-3-mini",
            family="phi",
            display_name="Phi-3 Mini",
            description="Microsoft's Phi-3 Mini 3.8B model",
            hidden_size=3072,
            num_attention_heads=32,
            num_key_value_heads=8,
            intermediate_size=8192,
            num_hidden_layers=32,
            vocab_size=32064,
            attention_type=AttentionType.GROUPED_QUERY,
            position_embedding=PositionEmbeddingType.ROPE,
            rope_theta=10000.0,
            tokenizer_type=TokenizerType.BPE,
            recommended_max_seq_length=4096,
            recommended_memory_gb=12.0,
            hf_model_id="microsoft/Phi-3-mini-4k-instruct",
        ))
        
        # (Removed Code-LLaMA series - requires gated access)
        
        # (Removed Gemma series - requires gated access)
    
    def register(self, config: ModelConfig) -> None:
        """Register a new model configuration"""
        self._registry[config.model_id] = config
    
    def get(self, model_id: str) -> Optional[ModelConfig]:
        """Get a model configuration by ID"""
        return self._registry.get(model_id)
    
    def get_by_family(self, family: str) -> List[ModelConfig]:
        """Get all models from a specific family"""
        return [config for config in self._registry.values() if config.family == family]
    
    def list_all(self) -> Dict[str, ModelConfig]:
        """Get all registered models"""
        return dict(self._registry)
    
    def list_families(self) -> List[str]:
        """Get all model families"""
        return sorted(set(config.family for config in self._registry.values()))
    
    def supports_model(self, model_id: str) -> bool:
        """Check if a model is supported"""
        return model_id in self._registry
    
    def get_summary(self) -> str:
        """Get a formatted summary of all registered models"""
        summary = "CodeFuse-Embeddings Model Support Registry\n"
        summary += "=" * 60 + "\n\n"
        
        families = self.list_families()
        for family in families:
            summary += f"\n{family.upper()} Family:\n"
            summary += "-" * 40 + "\n"
            models = self.get_by_family(family)
            for model in models:
                summary += f"  • {model.model_id}: {model.description}\n"
                summary += f"    Size: {model.hidden_size}d, "
                summary += f"Heads: {model.num_attention_heads}, "
                summary += f"Memory: {model.recommended_memory_gb}GB\n"
        
        return summary


# Global registry instance
_default_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Get or create the global model registry"""
    global _default_registry
    if _default_registry is None:
        _default_registry = ModelRegistry()
    return _default_registry
