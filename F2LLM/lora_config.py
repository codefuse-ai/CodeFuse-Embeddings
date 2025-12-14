"""LoRA configuration and utilities for efficient model adaptation."""

from peft import LoraConfig, TaskType, get_peft_model
from dataclasses import dataclass
from typing import Optional


@dataclass
class LoRAConfig:
    """Configuration class for LoRA (Low-Rank Adaptation) parameters."""
    
    # LoRA settings
    r: int = 16  # LoRA rank
    lora_alpha: int = 32  # LoRA alpha (scaling factor)
    target_modules: list = None  # Target modules for LoRA adaptation
    lora_dropout: float = 0.05  # Dropout probability for LoRA layers
    bias: str = "none"  # Bias configuration ("none", "all", "lora_only")
    
    # Training strategy
    modules_to_save: Optional[list] = None  # Modules to save in addition to LoRA weights
    
    def __post_init__(self):
        """Set default target modules for common LLM architectures."""
        if self.target_modules is None:
            # Common target modules for Transformer models
            self.target_modules = [
                "q_proj",      # Query projection
                "v_proj",      # Value projection
                "k_proj",      # Key projection
                "o_proj",      # Output projection
                "gate_proj",   # Gate projection (for gating mechanisms)
                "up_proj",     # Up projection
                "down_proj",   # Down projection
            ]

    def get_peft_config(self):
        """Get PEFT LoRA configuration object."""
        return LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias=self.bias,
            task_type=TaskType.FEATURE_EXTRACTION,  # For embedding models
            modules_to_save=self.modules_to_save,
        )


def apply_lora_to_model(model, lora_config: LoRAConfig):
    """
    Apply LoRA to a model.
    
    Args:
        model: The base model to apply LoRA to
        lora_config: LoRA configuration object
        
    Returns:
        Model with LoRA applied
    """
    peft_config = lora_config.get_peft_config()
    model = get_peft_model(model, peft_config)
    
    # Print LoRA configuration and trainable parameters
    model.print_trainable_parameters()
    
    return model
